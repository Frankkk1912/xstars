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
EXPORT_FORMATS = frozenset({"png", "tiff", "jpg", "pdf"})
MIN_EXPORT_DPI = 72
MAX_EXPORT_DPI = 1200
BASE_CLIPBOARD_DPI = 96
MAX_EXPORT_PIXELS = 100_000_000

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


class ServiceEndpointError(ServiceValidationError):
    """A bounded endpoint failure with a stable client-facing error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def validate_matrix(values: Any) -> tuple[int, int, int]:
    """Validate a bounded rectangular 2-D matrix and return its statistics."""

    if not isinstance(values, list) or not values:
        raise ServiceEndpointError(
            400, "INVALID_VALUES", "values must be a non-empty 2-D array"
        )
    if not all(isinstance(row, list) and row for row in values):
        raise ServiceEndpointError(
            400, "INVALID_VALUES", "each values row must be a non-empty array"
        )
    columns = len(values[0])
    if any(len(row) != columns for row in values):
        raise ServiceEndpointError(
            400, "NON_RECTANGULAR_VALUES", "values must be rectangular"
        )
    rows = len(values)
    if rows > 200 or columns > 200:
        raise ServiceEndpointError(
            400, "SELECTION_TOO_LARGE", "each range is limited to 200 x 200 cells"
        )
    non_empty = 0
    for row in values:
        for value in row:
            if isinstance(value, float) and not math.isfinite(value):
                raise ServiceEndpointError(
                    400, "NON_FINITE_VALUE", "non-finite numeric cells are not supported"
                )
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ServiceEndpointError(
                    400,
                    "INVALID_CELL_VALUE",
                    f"unsupported cell value type: {type(value).__name__}",
                )
            if value is not None and value != "":
                non_empty += 1
    return rows, columns, non_empty


def validate_elisa_selection(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceEndpointError(400, "INVALID_REQUEST", "request must be an object")
    source = payload.get("source")
    if source not in {"inputbox", "two-stage", "address"}:
        raise ServiceEndpointError(
            400,
            "INVALID_SOURCE",
            "source must be inputbox, two-stage, or address",
        )
    ranges = payload.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ServiceEndpointError(
            400, "MISSING_RANGES", "ranges must be a non-empty array"
        )
    if source == "two-stage" and len(ranges) != 2:
        raise ServiceEndpointError(
            400, "INVALID_RANGE_COUNT", "two-stage requests require exactly 2 ranges"
        )

    response_ranges = []
    for index, item in enumerate(ranges):
        if not isinstance(item, dict):
            raise ServiceEndpointError(
                400, "INVALID_RANGE", f"ranges[{index}] must be an object"
            )
        address = item.get("address")
        if (
            not isinstance(address, str)
            or not address.strip()
            or len(address) > 256
        ):
            raise ServiceEndpointError(
                400,
                "INVALID_ADDRESS",
                f"ranges[{index}].address must contain 1-256 characters",
            )
        rows, columns, non_empty = validate_matrix(item.get("values"))
        response_ranges.append(
            {
                "address": address,
                "values": item["values"],
                "rows": rows,
                "columns": columns,
                "nonEmptyCells": non_empty,
            }
        )
    return {"ok": True, "source": source, "ranges": response_ranges}


def validate_shape_export(payload: Any) -> tuple[str, int]:
    if not isinstance(payload, dict):
        raise ServiceEndpointError(400, "INVALID_REQUEST", "request must be an object")
    image_format = payload.get("format")
    if not isinstance(image_format, str) or image_format.lower() not in EXPORT_FORMATS:
        raise ServiceEndpointError(
            400, "INVALID_FORMAT", "format must be png, tiff, jpg, or pdf"
        )
    dpi = payload.get("dpi")
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise ServiceEndpointError(
            400, "INVALID_DPI", "dpi must be an integer from 72 through 1200"
        )
    if dpi < MIN_EXPORT_DPI or dpi > MAX_EXPORT_DPI:
        raise ServiceEndpointError(
            400, "INVALID_DPI", "dpi must be an integer from 72 through 1200"
        )
    return image_format.lower(), dpi


def clipboard_format_diagnostics() -> str:
    """Best-effort Win32 format inventory when Pillow cannot read the clipboard."""

    try:
        import win32clipboard  # type: ignore[import-not-found]

        formats = []
        win32clipboard.OpenClipboard()
        try:
            current = 0
            while True:
                current = win32clipboard.EnumClipboardFormats(current)
                if not current:
                    break
                try:
                    name = win32clipboard.GetClipboardFormatName(current)
                except Exception:
                    name = str(current)
                formats.append(name)
        finally:
            win32clipboard.CloseClipboard()
        return ", ".join(formats) if formats else "none"
    except Exception as error:
        return f"unavailable ({error})"


def export_clipboard_image(image_format: str, dpi: int) -> dict[str, Any]:
    """Read the current clipboard image and persist a bounded PoC export."""

    try:
        from PIL import Image, ImageGrab
    except Exception as error:
        raise ServiceEndpointError(
            503, "PIL_UNAVAILABLE", f"Pillow clipboard support is unavailable: {error}"
        ) from error

    try:
        image: Any = ImageGrab.grabclipboard()
    except Exception as error:
        formats = clipboard_format_diagnostics()
        raise ServiceEndpointError(
            500,
            "CLIPBOARD_READ_FAILED",
            f"clipboard read failed: {error}; formats: {formats}",
        ) from error
    if image is None:
        formats = clipboard_format_diagnostics()
        raise ServiceEndpointError(
            422,
            "CLIPBOARD_EMPTY",
            f"clipboard does not contain a Pillow-readable image; formats: {formats}",
        )
    if not all(hasattr(image, member) for member in ("size", "resize", "convert", "save")):
        raise ServiceEndpointError(
            422, "CLIPBOARD_NOT_IMAGE", "clipboard content is not an image"
        )

    width, height = image.size
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ServiceEndpointError(
            422, "CLIPBOARD_NOT_IMAGE", "clipboard image dimensions are invalid"
        )
    scale = dpi / BASE_CLIPBOARD_DPI
    target_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    if target_size[0] * target_size[1] > MAX_EXPORT_PIXELS:
        raise ServiceEndpointError(
            413,
            "EXPORT_TOO_LARGE",
            f"target image exceeds {MAX_EXPORT_PIXELS} pixels",
        )
    resampling = getattr(Image, "Resampling", Image)
    lanczos = getattr(resampling, "LANCZOS", 1)
    exported = image.resize(target_size, lanczos)

    if image_format in {"jpg", "pdf"}:
        if getattr(exported, "mode", "RGB") in {"RGBA", "LA"}:
            rgba = exported.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            exported = background
        else:
            exported = exported.convert("RGB")

    output_dir = artifact_root() / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = output_dir / f"shape-{timestamp}-{dpi}dpi.{image_format}"
    save_kwargs: dict[str, Any] = {"dpi": (dpi, dpi)}
    if image_format == "tiff":
        save_kwargs["compression"] = "tiff_lzw"
    elif image_format == "jpg":
        save_kwargs["quality"] = 95
    elif image_format == "pdf":
        save_kwargs = {"resolution": dpi}
    exported.save(output_path, **save_kwargs)
    actual_dpi = [float(dpi), float(dpi)]
    if image_format != "pdf":
        try:
            with Image.open(output_path) as persisted:
                stored_dpi = persisted.info.get("dpi")
            if stored_dpi and len(stored_dpi) >= 2:
                actual_dpi = [round(float(stored_dpi[0]), 3), round(float(stored_dpi[1]), 3)]
        except Exception as error:
            log_write(f"M0.4 DPI metadata readback failed for {output_path}: {error}")
    return {
        "ok": True,
        "outputPath": str(output_path.resolve()),
        "format": image_format,
        "dpi": dpi,
        "actualDpi": actual_dpi,
        "sourceWidth": width,
        "sourceHeight": height,
        "width": target_size[0],
        "height": target_size[1],
    }


def probe_wps_com() -> dict[str, Any]:
    """Probe the running WPS ET COM server without making it a dependency."""

    try:
        import win32com.client  # type: ignore[import-not-found]

        application = win32com.client.GetActiveObject("Ket.Application")
        version = getattr(application, "Version", None)
        return {
            "ok": True,
            "progId": "Ket.Application",
            "version": None if version is None else str(version),
        }
    except Exception as error:
        return {
            "ok": False,
            "code": "COM_UNAVAILABLE",
            "detail": f"{error.__class__.__name__}: {error}",
        }


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
        if self.path not in {
            "/dialog",
            "/probe/elisa-selection",
            "/probe/shape-export",
        }:
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
        if self.path == "/probe/com-probe":
            self._send_json(200, probe_wps_com())
            return
        self._send_error(404, "not_found", "endpoint not found")

    def _read_json_body(self) -> dict[str, Any] | None:
        if self.headers.get_content_type() != "application/json":
            self._send_error(
                415, "unsupported_media_type", "Content-Type must be application/json"
            )
            return None
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(
                400, "invalid_content_length", "Content-Length must be an integer"
            )
            return None
        if content_length < 0 or content_length > MAX_BODY_BYTES:
            self._send_error(
                413,
                "invalid_body_size",
                "request body size is outside the allowed range",
            )
            return None
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._send_error(400, "invalid_json", f"request is not valid JSON: {error}")
            return None
        if not isinstance(payload, dict):
            self._send_error(400, "invalid_request", "request must be a JSON object")
            return None
        return payload

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        if self.path not in {
            "/dialog",
            "/probe/elisa-selection",
            "/probe/shape-export",
        }:
            self._send_error(404, "not_found", "endpoint not found")
            return
        payload = self._read_json_body()
        if payload is None:
            return

        if self.path == "/probe/elisa-selection":
            try:
                response = validate_elisa_selection(payload)
            except ServiceEndpointError as error:
                self._send_error(error.status, error.code, error.message)
                return
            log_write(
                f"M0.4 ELISA selection source={response['source']} "
                f"ranges={len(response['ranges'])}"
            )
            response["requestOrigin"] = self._origin_echo()
            self._send_json(200, response)
            return

        if self.path == "/probe/shape-export":
            try:
                image_format, dpi = validate_shape_export(payload)
                response = export_clipboard_image(image_format, dpi)
            except ServiceEndpointError as error:
                log_write(f"M0.4 shape export error code={error.code}: {error.message}")
                self._send_error(error.status, error.code, error.message)
                return
            except Exception as error:  # Defensive PoC boundary: never kill service.
                log_write(f"M0.4 shape export unexpected error: {error}")
                self._send_error(
                    500, "EXPORT_FAILED", f"unexpected export failure: {error}"
                )
                return
            response["requestOrigin"] = self._origin_echo()
            log_write(
                f"M0.4 shape export ok format={image_format} dpi={dpi} "
                f"path={response['outputPath']}"
            )
            self._send_json(200, response)
            return

        server = self.server
        assert isinstance(server, Gate0ServiceServer)
        started = time.monotonic()
        result = server.dialogs.show_dialog(str(payload.get("message", "M0.3 Tkinter dialog")))
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

    def self_test_post(
        port_number: int, path: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        if path not in {"/probe/elisa-selection", "/probe/shape-export"}:
            raise ValueError("self-test POST path is not allowed")
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port_number}{path}",
            data=encoded,
            headers={"Content-Type": "application/json", "Origin": "null"},
        )
        try:
            with opener.open(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

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

        status, elisa = self_test_post(
            bound_port,
            "/probe/elisa-selection",
            {
                "source": "inputbox",
                "ranges": [{"address": "$A$1:$B$2", "values": [[1, 2], [3, None]]}],
            },
        )
        if (
            status == 200
            and elisa.get("ok")
            and elisa.get("ranges", [{}])[0].get("nonEmptyCells") == 3
        ):
            print("PASS: /probe/elisa-selection validates and summarizes a matrix")
        else:
            print(f"FAIL: ELISA selection self-test unexpected: {status} {elisa}")
            failures += 1

        status, shape_error = self_test_post(
            bound_port,
            "/probe/shape-export",
            {"format": "bmp", "dpi": 300},
        )
        if (
            status == 400
            and shape_error.get("error", {}).get("code") == "INVALID_FORMAT"
        ):
            print("PASS: /probe/shape-export rejects an unsupported format")
        else:
            print(f"FAIL: shape export self-test unexpected: {status} {shape_error}")
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
