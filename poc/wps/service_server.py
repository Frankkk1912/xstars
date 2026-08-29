"""Loopback Tkinter service for the WPS Gate 0 M0.3 lifecycle test.

Validates the server half of M0.3:

- ``OAAssist.ShellExecute`` launches this script headlessly (pythonw).
- The service binds 127.0.0.1 only and enforces the same strict CORS
  discipline as the M0.2 probe (Origin ``null`` / ``file://`` / dev server).
- A second instance on the same port fails with a diagnosable log entry.
- A Tkinter dialog runs on a dedicated thread while ``/health`` stays
  responsive, proving WPS must not freeze while the dialog is open.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
DEFAULT_PORT = 3892
MAX_BODY_BYTES = 65_536
DIALOG_TIMEOUT_SECONDS = 600
ALLOWED_ORIGINS = frozenset(
    {
        "null",  # Browsers serialize file:// origins as the literal value null.
        "file://",  # Retained for WPS builds that expose the displayed origin.
        "http://127.0.0.1:3889",  # Official wpsjs development server.
    }
)
LOG_TAIL_LINES = 40
EXIT_PORT_CONFLICT = 2

STARTED_AT = time.monotonic()


def artifact_root() -> Path:
    return Path(__file__).resolve().parent / ".gate0-artifacts"


def log_path() -> Path:
    return artifact_root() / "service.log"


def log_write(message: str) -> None:
    """Append a timestamped diagnostic line to the shared service log."""

    entry = (
        f"{datetime.now().isoformat(timespec='seconds')} os_pid={os.getpid()} {message}"
    )
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")


def read_log_tail(limit: int = LOG_TAIL_LINES) -> list[str]:
    path = log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


class TkDialogManager:
    """Runs Tkinter dialogs on a dedicated thread so HTTP stays responsive."""

    def __init__(self) -> None:
        self._requests: queue.Queue[tuple[str, queue.Queue[Any]]] = queue.Queue()
        self._init_error: str | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tk-dialog")
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self._init_error = "Tkinter dialog thread did not start within 10s"

    def _run(self) -> None:  # pragma: no cover - Tk main loop
        try:
            import tkinter as tk
            from tkinter import messagebox
        except Exception as error:  # pragma: no cover - headless environments
            self._init_error = f"Tkinter unavailable: {error}"
            self._ready.set()
            return
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception as error:  # pragma: no cover - no display
            self._init_error = f"Tkinter root failed: {error}"
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._requests.get()
            if item is None:
                break
            _message, result_queue = item
            try:
                confirmed = messagebox.askokcancel(
                    "XSTARS Gate 0 M0.3",
                    "Tkinter 模态对话框测试\n\n点击“确定”或“取消”任一按钮即可结束。\nWPS 此时应保持可操作（不被冻结）。",
                )
                result_queue.put({"confirmed": bool(confirmed), "error": None})
            except Exception as error:  # pragma: no cover - Tk runtime failure
                result_queue.put({"confirmed": None, "error": str(error)})

    def show_dialog(
        self, message: str, timeout: float = DIALOG_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        if self._init_error:
            return {"confirmed": None, "error": self._init_error}
        result_queue: queue.Queue[Any] = queue.Queue()
        self._requests.put((message, result_queue))
        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            return {"confirmed": None, "error": f"dialog timed out after {timeout}s"}


class ServiceValidationError(ValueError):
    """Raised when a service payload violates the bounded PoC contract."""


class Gate0ServiceServer(ThreadingHTTPServer):
    daemon_threads = True

    # Windows quirk: SO_REUSEADDR (allow_reuse_address) lets a second
    # instance silently re-bind an actively listening port. Use
    # SO_EXCLUSIVEADDRUSE so a double start fails with WSAEADDRINUSE and
    # produces a diagnosable log entry instead of a shadow server.
    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket

        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, server_address: tuple[str, int]):
        self.dialogs = TkDialogManager()
        super().__init__(server_address, ServiceRequestHandler)


class ServiceRequestHandler(BaseHTTPRequestHandler):
    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _origin_allowed(self) -> bool:
        origin = self._origin()
        return origin is None or origin in ALLOWED_ORIGINS

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "").partition(":")[0].lower()
        return host in {HOST, "localhost"}

    def _cors_headers(self) -> None:
        origin = self._origin()
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _origin_echo(self) -> str:
        origin = self._origin()
        return "（浏览器未携带 Origin 头）" if origin is None else origin

    def _request_allowed(self) -> bool:
        if self._host_allowed() and self._origin_allowed():
            return True
        self._send_error(403, "origin_denied", "loopback host or origin is not allowed")
        return False

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str) -> None:
        self._send_json(
            status, {"ok": False, "error": {"code": code, "message": message}}
        )

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        if self.path != "/dialog":
            self._send_error(404, "not_found", "endpoint not found")
            return

        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        server = self.server
        assert isinstance(server, Gate0ServiceServer)
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "xstars-wps-gate0-service",
                    "port": server.server_address[1],
                    "pid": os.getpid(),
                    "uptimeSeconds": round(time.monotonic() - STARTED_AT, 1),
                    "requestOrigin": self._origin_echo(),
                },
            )
            return
        if self.path == "/diagnostics":
            self._send_json(
                200,
                {
                    "ok": True,
                    "logTail": read_log_tail(),
                    "requestOrigin": self._origin_echo(),
                },
            )
            return
        self._send_error(404, "not_found", "endpoint not found")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        if self.path != "/dialog":
            self._send_error(404, "not_found", "endpoint not found")
            return
        if self.headers.get_content_type() != "application/json":
            self._send_error(
                415, "unsupported_media_type", "Content-Type must be application/json"
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(
                400, "invalid_content_length", "Content-Length must be an integer"
            )
            return
        if content_length < 0 or content_length > MAX_BODY_BYTES:
            self._send_error(
                413,
                "invalid_body_size",
                "request body size is outside the allowed range",
            )
            return

        server = self.server
        assert isinstance(server, Gate0ServiceServer)
        started = time.monotonic()
        result = server.dialogs.show_dialog("M0.3 Tkinter dialog")
        duration_ms = round((time.monotonic() - started) * 1000)
        if result.get("error"):
            log_write(f"dialog error: {result['error']}")
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": {"code": "dialog_failed", "message": result["error"]},
                },
            )
            return
        log_write(
            f"dialog resolved confirmed={result['confirmed']} durationMs={duration_ms}"
        )
        self._send_json(
            200,
            {
                "ok": True,
                "confirmed": result["confirmed"],
                "durationMs": duration_ms,
                "requestOrigin": self._origin_echo(),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def create_server(host: str = HOST, port: int = DEFAULT_PORT) -> Gate0ServiceServer:
    if host != HOST:
        raise ValueError("the Gate 0 service may bind only to 127.0.0.1")
    if not math.isfinite(port) or port < 0 or port > 65535:
        raise ValueError("port must be within 0-65535")
    return Gate0ServiceServer((host, port))


def run_conflict_diagnostic(port: int, error: OSError) -> None:
    log_write(
        f"PORT CONFLICT on 127.0.0.1:{port} while starting second instance: "
        f"{error.__class__.__name__}: {error}"
    )


def main() -> None:
    # Entry logging: proves whether the interpreter reached the script at all
    # (distinguishes "ShellExecute never launched" from "launched then died").
    log_write(f"process entry argv={sys.argv!r} exe={sys.executable!r}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST, choices=[HOST])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run an in-process health/port-conflict check and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        failures = self_test(args.port)
        if failures:
            print(f"SELF-TEST FAILED ({failures} failure(s))")
            sys.exit(1)
        print("SELF-TEST PASSED")
        return

    try:
        server = create_server(args.host, args.port)
    except OSError as error:
        run_conflict_diagnostic(args.port, error)
        print(f"Cannot bind {args.host}:{args.port}: {error}", file=sys.stderr)
        sys.exit(EXIT_PORT_CONFLICT)

    log_write(f"service started on {args.host}:{server.server_address[1]}")
    print(
        f"XSTARS WPS Gate 0 service listening on http://{HOST}:{server.server_address[1]}"
    )
    print(f"Allowed browser origins: {', '.join(sorted(ALLOWED_ORIGINS))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping XSTARS WPS Gate 0 service")
    finally:
        log_write(f"service stopped on port {server.server_address[1]}")
        server.server_close()


def self_test(port: int) -> int:
    """Verify health endpoint and port-conflict diagnostics in-process."""

    failures = 0
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        server = create_server(HOST, port)
    except OSError as error:
        print(f"FAIL: cannot bind {HOST}:{port}: {error}")
        return 1
    bound_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with opener.open(f"http://{HOST}:{bound_port}/health", timeout=5) as response:
            health = json.loads(response.read())
        if (
            health.get("ok")
            and health.get("requestOrigin") == "（浏览器未携带 Origin 头）"
        ):
            print(f"PASS: /health ok on port {bound_port}")
        else:
            print(f"FAIL: unexpected /health payload: {health}")
            failures += 1

        try:
            create_server(HOST, bound_port)
            print("FAIL: second bind unexpectedly succeeded")
            failures += 1
        except OSError as error:
            run_conflict_diagnostic(bound_port, error)
            tail = " ".join(read_log_tail(3))
            if "PORT CONFLICT" in tail:
                print("PASS: port conflict produced a diagnostic log entry")
            else:
                print(f"FAIL: conflict diagnostic missing from log tail: {tail}")
                failures += 1

        req = urllib.request.Request(
            f"http://{HOST}:{bound_port}/diagnostics",
            headers={"Origin": "null"},
        )
        with opener.open(req, timeout=5) as response:
            diag = json.loads(response.read())
        if diag.get("ok") and any(
            "PORT CONFLICT" in line for line in diag.get("logTail", [])
        ):
            print(
                "PASS: /diagnostics exposes the conflict entry (Origin null accepted)"
            )
        else:
            print(f"FAIL: /diagnostics payload unexpected: {diag}")
            failures += 1
    except urllib.error.URLError as error:
        print(f"FAIL: HTTP request failed: {error}")
        failures += 1
    finally:
        server.shutdown()
        server.server_close()
    return failures


if __name__ == "__main__":
    main()
