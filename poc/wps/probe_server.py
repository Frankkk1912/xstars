"""Loopback-only HTTP probe for the WPS Gate 0 vertical-slice test."""

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

HOST = "127.0.0.1"
DEFAULT_PORT = 3891
MAX_BODY_BYTES = 1_048_576
MAX_ROWS = 200
MAX_COLUMNS = 200
ALLOWED_ORIGINS = frozenset(
    {
        "null",  # Browsers serialize file:// origins as the literal value null.
        "file://",  # Retained for WPS builds that expose the displayed origin.
        "http://127.0.0.1:3889",  # Official wpsjs development server.
    }
)
PNG_WIDTH = 320
PNG_HEIGHT = 180


class ProbeValidationError(ValueError):
    """Raised when a probe payload violates the bounded PoC contract."""


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_probe_png(
    artifact_dir: Path,
    width: int = PNG_WIDTH,
    height: int = PNG_HEIGHT,
) -> Path:
    """Write a deterministic RGB PNG inside the configured artifact directory."""

    artifact_root = artifact_dir.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = (artifact_root / "xstars-gate0-probe.png").resolve()
    if path.parent != artifact_root:
        raise ValueError("probe image path escaped the artifact directory")
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # PNG filter: None.
        for x in range(width):
            if y < 36:
                rgb = (21, 128, 61)
            elif x < width // 2:
                rgb = (219 - y // 2, 238 - y // 3, 225)
            else:
                rgb = (219, 234 - y // 4, 254 - y // 3)
            raw.extend(max(0, min(255, channel)) for channel in rgb)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path.resolve()


def _normalize_cell(value: Any) -> str | int | float | bool | None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProbeValidationError("non-finite numeric cell values are not supported")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProbeValidationError(f"unsupported cell value type: {type(value).__name__}")


def validate_selection(payload: Any) -> tuple[dict[str, Any], list[list[Any]]]:
    if not isinstance(payload, dict):
        raise ProbeValidationError("request body must be a JSON object")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ProbeValidationError("selection must be a JSON object")

    address = selection.get("address")
    if not isinstance(address, str) or not address or len(address) > 256:
        raise ProbeValidationError("selection.address must be a non-empty string")

    values = selection.get("values")
    if not isinstance(values, list) or not values or len(values) > MAX_ROWS:
        raise ProbeValidationError(f"selection.values must contain 1-{MAX_ROWS} rows")
    if not all(isinstance(row, list) and row for row in values):
        raise ProbeValidationError("selection.values must be a non-empty 2D matrix")

    column_count = len(values[0])
    if column_count > MAX_COLUMNS or any(len(row) != column_count for row in values):
        raise ProbeValidationError(
            f"selection.values must be rectangular with at most {MAX_COLUMNS} columns"
        )

    normalized = [[_normalize_cell(value) for value in row] for row in values]
    expected_rows = selection.get("rows")
    expected_columns = selection.get("columns")
    if expected_rows != len(normalized) or expected_columns != column_count:
        raise ProbeValidationError("selection dimensions do not match the value matrix")

    return selection, normalized


def build_probe_response(payload: Any, artifact_dir: Path) -> dict[str, Any]:
    selection, matrix = validate_selection(payload)
    image_path = write_probe_png(artifact_dir)
    return {
        "ok": True,
        "selectionAddress": selection["address"],
        "matrix": matrix,
        "imagePath": str(image_path),
        "imageWidth": PNG_WIDTH,
        "imageHeight": PNG_HEIGHT,
    }


class ProbeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], artifact_dir: Path):
        self.artifact_dir = artifact_dir
        super().__init__(server_address, ProbeRequestHandler)


class ProbeRequestHandler(BaseHTTPRequestHandler):
    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _origin_allowed(self) -> bool:
        origin = self._origin()
        return origin is None or origin in ALLOWED_ORIGINS

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "").partition(":")[0].lower()
        return host in {"127.0.0.1", "localhost"}

    def _cors_headers(self) -> None:
        origin = self._origin()
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

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
        self._send_json(status, {"ok": False, "error": {"code": code, "message": message}})

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        if self.path != "/probe":
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
        if self.path != "/health":
            self._send_error(404, "not_found", "endpoint not found")
            return
        server = cast(ProbeHTTPServer, self.server)
        self._send_json(200, {"ok": True, "service": "xstars-wps-gate0", "port": server.server_port})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if not self._request_allowed():
            return
        if self.path != "/probe":
            self._send_error(404, "not_found", "endpoint not found")
            return
        if self.headers.get_content_type() != "application/json":
            self._send_error(415, "unsupported_media_type", "Content-Type must be application/json")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(400, "invalid_content_length", "Content-Length must be an integer")
            return
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self._send_error(413, "invalid_body_size", "request body size is outside the allowed range")
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            server = cast(ProbeHTTPServer, self.server)
            response = build_probe_response(payload, server.artifact_dir)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._send_error(400, "invalid_json", str(error))
            return
        except ProbeValidationError as error:
            self._send_error(400, "invalid_selection", str(error))
            return
        except OSError as error:
            self._send_error(500, "artifact_write_failed", str(error))
            return

        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def create_server(
    host: str = HOST,
    port: int = DEFAULT_PORT,
    artifact_dir: Path | None = None,
) -> ProbeHTTPServer:
    if host != HOST:
        raise ValueError("the Gate 0 probe may bind only to 127.0.0.1")
    artifacts = artifact_dir or Path(tempfile.gettempdir()) / "xstars-wps-gate0"
    return ProbeHTTPServer((host, port), artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST, choices=[HOST])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()

    server = create_server(args.host, args.port, args.artifact_dir)
    print(f"XSTARS WPS Gate 0 probe listening on http://{HOST}:{server.server_port}")
    print(f"Allowed browser origins: {', '.join(sorted(ALLOWED_ORIGINS))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping XSTARS WPS Gate 0 probe")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
