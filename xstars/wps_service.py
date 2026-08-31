"""Authenticated loopback broker for the XSTARS WPS add-in."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from .application.contracts import (
    SCHEMA_VERSION,
    Command,
    ContractError,
    ErrorCode,
    SelectionPayload,
)

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 3892
DEFAULT_CONFIG_PATH = Path.home() / ".xstars" / "wps_service.json"
DEFAULT_JOBS_ROOT = Path.home() / ".xstars" / "wps_jobs"
MAX_REQUEST_BYTES = 1_048_576
WORKER_TIMEOUT_SECONDS = 300.0
JOB_RETENTION_SECONDS = 86_400.0


class WorkerFailure(RuntimeError):
    """A worker could not produce a usable structured result."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class JobRunner(Protocol):
    def run(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


def _restrict_permissions(path: Path, mode: int) -> None:
    """Apply best-effort owner-only mode (also meaningful in POSIX tests)."""
    try:
        path.chmod(mode)
    except OSError as exc:
        raise RuntimeError(f"cannot restrict permissions for {path}: {exc}") from exc


def load_or_create_token(config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Load, or atomically create, the per-installation bearer token."""
    config_path = config_path.expanduser().resolve(strict=False)
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _restrict_permissions(config_path.parent, 0o700)
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid WPS service config: {exc}") from exc
        token = config.get("token") if isinstance(config, dict) else None
        if not isinstance(token, str) or len(token) < 32:
            raise RuntimeError("invalid WPS service token")
        _restrict_permissions(config_path, 0o600)
        return token

    token = secrets.token_urlsafe(32)
    encoded = json.dumps({"version": SCHEMA_VERSION, "token": token}, indent=2).encode(
        "utf-8"
    )
    try:
        descriptor = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return load_or_create_token(config_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        config_path.unlink(missing_ok=True)
        raise
    _restrict_permissions(config_path, 0o600)
    return token


def mask_token(token: str) -> str:
    if len(token) < 12:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


def origin_allowed(origin: str | None, addin_ports: frozenset[int]) -> bool:
    """Accept native requests plus the two observed WPS origins and dev ports."""
    if origin is None or origin in {"null", "file://"}:
        return True
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname == LOOPBACK_HOST
            and parsed.port in addin_ports
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def cleanup_stale_jobs(
    root: Path, *, older_than: float = JOB_RETENTION_SECONDS
) -> None:
    """Remove controlled job directories after the short artifact retention window."""
    if not root.exists():
        return
    cutoff = time.time() - older_than
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
        except OSError:
            continue


def _atomic_write_request(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SubprocessJobRunner:
    """Create a controlled job directory and execute exactly one worker process."""

    def __init__(
        self,
        jobs_root: Path = DEFAULT_JOBS_ROOT,
        *,
        timeout: float = WORKER_TIMEOUT_SECONDS,
        python_executable: str = sys.executable,
    ) -> None:
        self.jobs_root = jobs_root.expanduser().resolve(strict=False)
        self.timeout = timeout
        self.python_executable = python_executable
        self.jobs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _restrict_permissions(self.jobs_root, 0o700)
        cleanup_stale_jobs(self.jobs_root)

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        cleanup_stale_jobs(self.jobs_root)
        job_id = secrets.token_hex(16)
        job_directory = self.jobs_root / job_id
        job_directory.mkdir(mode=0o700)
        request_path = (job_directory / "request.json").resolve()
        result_path = (job_directory / "result.json").resolve()
        cancel_path = (job_directory / "cancel").resolve()
        worker_request = dict(request)
        worker_request["cancelPath"] = str(cancel_path)
        _atomic_write_request(request_path, worker_request)

        process = subprocess.Popen(
            [
                self.python_executable,
                "-m",
                "xstars.cli",
                "worker",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            cancel_path.touch()
            try:
                return_code = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
                return_code = process.returncode

        try:
            if timed_out:
                raise WorkerFailure(ErrorCode.TIMEOUT, "worker timed out")
            if not result_path.exists():
                raise WorkerFailure(
                    ErrorCode.INTERNAL_ERROR,
                    f"worker exited with code {return_code} without a result",
                )
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise WorkerFailure(
                    ErrorCode.INTERNAL_ERROR, "worker result is invalid"
                ) from exc
            if not isinstance(result, dict):
                raise WorkerFailure(
                    ErrorCode.INTERNAL_ERROR, "worker result must be an object"
                )
            result["jobId"] = job_id
            return result
        finally:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            cancel_path.unlink(missing_ok=True)
            if timed_out or not (job_directory / "chart.png").exists():
                shutil.rmtree(job_directory, ignore_errors=True)


class WPSHTTPServer(ThreadingHTTPServer):
    """Threading server with strict loopback bind and single-instance semantics."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        server_address: tuple[str, int],
        token: str,
        runner: JobRunner,
        *,
        addin_ports: frozenset[int] = frozenset({3889, 3890}),
        max_request_bytes: int = MAX_REQUEST_BYTES,
    ) -> None:
        if server_address[0] != LOOPBACK_HOST:
            raise ValueError("WPS service may only bind to 127.0.0.1")
        self.token = token
        self.runner = runner
        self.addin_ports = addin_ports
        self.max_request_bytes = max_request_bytes
        self.job_lock = threading.Lock()
        self.started_at = time.monotonic()
        super().__init__(server_address, WPSRequestHandler)

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


class WPSRequestHandler(BaseHTTPRequestHandler):
    @property
    def _wps_server(self) -> WPSHTTPServer:
        return cast(WPSHTTPServer, self.server)

    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _request_boundary_allowed(self) -> bool:
        host = self.headers.get("Host", "").partition(":")[0].lower()
        return host in {LOOPBACK_HOST, "localhost"} and origin_allowed(
            self._origin(), self._wps_server.addin_ports
        )

    def _cors_headers(self) -> None:
        origin = self._origin()
        if origin is not None and origin_allowed(origin, self._wps_server.addin_ports):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: ErrorCode | str, message: str) -> None:
        value = code.value if isinstance(code, ErrorCode) else code
        self._send_json(
            status,
            {
                "version": SCHEMA_VERSION,
                "ok": False,
                "error": {"code": value, "message": message},
            },
        )

    def _boundary_or_error(self) -> bool:
        if self._request_boundary_allowed():
            return True
        self._send_error(403, "ORIGIN_DENIED", "loopback host or origin is not allowed")
        return False

    def _authenticated(self) -> bool:
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = header[len(prefix) :] if header.startswith(prefix) else ""
        return bool(supplied) and hmac.compare_digest(supplied, self._wps_server.token)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._boundary_or_error():
            return
        if self.path != "/command":
            self._send_error(404, "NOT_FOUND", "endpoint not found")
            return
        requested_method = self.headers.get("Access-Control-Request-Method", "POST")
        if requested_method != "POST":
            self._send_error(405, "METHOD_NOT_ALLOWED", "only POST is allowed")
            return
        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._boundary_or_error():
            return
        if self.path != "/health":
            self._send_error(404, "NOT_FOUND", "endpoint not found")
            return
        self._send_json(
            200,
            {
                "version": SCHEMA_VERSION,
                "ok": True,
                "service": "xstars-wps-service",
                "pid": os.getpid(),
                "port": self._wps_server.server_address[1],
                "uptimeSeconds": round(
                    time.monotonic() - self._wps_server.started_at, 3
                ),
                "tokenMask": mask_token(self._wps_server.token),
                "busy": self._wps_server.job_lock.locked(),
            },
        )

    def _read_command(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_error(
                411, ErrorCode.INVALID_REQUEST, "Content-Length is required"
            )
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send_error(
                400, ErrorCode.INVALID_REQUEST, "Content-Length is invalid"
            )
            return None
        if length < 0 or length > self._wps_server.max_request_bytes:
            self._send_error(
                413, ErrorCode.PAYLOAD_TOO_LARGE, "request body is too large"
            )
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeError, json.JSONDecodeError):
            self._send_error(
                400, ErrorCode.INVALID_REQUEST, "request body is not valid JSON"
            )
            return None
        if not isinstance(payload, dict):
            self._send_error(
                400, ErrorCode.INVALID_REQUEST, "request body must be an object"
            )
            return None
        allowed_fields = {"version", "command", "selection", "config"}
        if set(payload) - allowed_fields:
            self._send_error(
                400, ErrorCode.INVALID_REQUEST, "request contains unsupported fields"
            )
            return None
        if payload.get("version") != SCHEMA_VERSION:
            self._send_error(
                400, ErrorCode.INVALID_REQUEST, "unsupported request version"
            )
            return None
        try:
            command = Command(payload.get("command"))
        except (TypeError, ValueError):
            self._send_error(
                400, ErrorCode.INVALID_COMMAND, "command is not whitelisted"
            )
            return None
        selection_data = payload.get("selection")
        if not isinstance(selection_data, Mapping):
            self._send_error(
                400, ErrorCode.INVALID_SELECTION, "selection must be an object"
            )
            return None
        try:
            selection = SelectionPayload.from_dict(selection_data)
        except ContractError as exc:
            self._send_error(400, exc.code, str(exc))
            return None
        config = payload.get("config", {})
        if not isinstance(config, dict):
            self._send_error(400, ErrorCode.INVALID_REQUEST, "config must be an object")
            return None
        return {
            "version": SCHEMA_VERSION,
            "command": command.value,
            "selection": selection.to_dict(),
            "config": config,
        }

    def do_POST(self) -> None:  # noqa: N802
        if not self._boundary_or_error():
            return
        if self.path != "/command":
            self._send_error(404, "NOT_FOUND", "endpoint not found")
            return
        if not self._authenticated():
            self._send_error(401, "UNAUTHORIZED", "a valid bearer token is required")
            return
        request = self._read_command()
        if request is None:
            return
        if not self._wps_server.job_lock.acquire(blocking=False):
            self._send_error(409, ErrorCode.BUSY, "another job is already running")
            return
        try:
            try:
                result = self._wps_server.runner.run(request)
            except WorkerFailure as exc:
                status = 504 if exc.code is ErrorCode.TIMEOUT else 500
                self._send_error(status, exc.code, str(exc))
                return
            except Exception as exc:
                self._send_error(
                    500,
                    ErrorCode.INTERNAL_ERROR,
                    f"worker failed: {type(exc).__name__}",
                )
                return
            self._send_json(200, result)
        finally:
            self._wps_server.job_lock.release()

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    port: int = DEFAULT_PORT,
    *,
    token: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    runner: JobRunner | None = None,
    addin_ports: frozenset[int] = frozenset({3889, 3890}),
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> WPSHTTPServer:
    """Create a configured service or raise a diagnostic bind error."""
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be from 0 to 65535")
    service_token = token or load_or_create_token(config_path)
    service_runner = runner or SubprocessJobRunner()
    try:
        return WPSHTTPServer(
            (LOOPBACK_HOST, port),
            service_token,
            service_runner,
            addin_ports=addin_ports,
            max_request_bytes=max_request_bytes,
        )
    except OSError as exc:
        raise RuntimeError(
            f"cannot bind XSTARS WPS service to {LOOPBACK_HOST}:{port}: {exc}"
        ) from exc


def serve(port: int = DEFAULT_PORT) -> int:
    """Run the broker until interrupted; service startup belongs to the installer."""
    try:
        server = create_server(port)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"XSTARS WPS SERVICE ERROR: {exc}", file=sys.stderr)
        return 2
    bound_host = server.server_address[0]
    bound_port = server.server_address[1]
    print(
        f"XSTARS WPS service listening on http://{bound_host}:{bound_port}", flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0
