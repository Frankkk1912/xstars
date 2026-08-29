import http.client
import json
import struct
import threading
from pathlib import Path

import pytest

from poc.wps.probe_server import (
    HOST,
    ProbeValidationError,
    build_probe_response,
    create_server,
    validate_selection,
)


def sample_payload():
    return {
        "selection": {
            "address": "$A$1:$B$2",
            "rows": 2,
            "columns": 2,
            "values": [[1, 2], ["A", None]],
        }
    }


def request(server, method, path, *, body=None, headers=None):
    connection = http.client.HTTPConnection(HOST, server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


@pytest.fixture
def probe_server(tmp_path):
    server = create_server(port=0, artifact_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_validate_selection_rejects_ragged_matrix():
    payload = sample_payload()
    payload["selection"]["values"] = [[1, 2], [3]]

    with pytest.raises(ProbeValidationError, match="rectangular"):
        validate_selection(payload)


def test_build_probe_response_writes_valid_png(tmp_path):
    response = build_probe_response(sample_payload(), tmp_path)
    image = Path(response["imagePath"]).read_bytes()

    assert response["matrix"] == [[1, 2], ["A", None]]
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", image[16:24]) == (
        response["imageWidth"],
        response["imageHeight"],
    )


def test_health_is_loopback_only(probe_server):
    status, _, body = request(probe_server, "GET", "/health")

    assert probe_server.server_address[0] == HOST
    assert status == 200
    assert json.loads(body)["service"] == "xstars-wps-gate0"


def test_file_origin_preflight_is_explicitly_allowed(probe_server):
    status, headers, body = request(
        probe_server,
        "OPTIONS",
        "/probe",
        headers={
            "Host": f"127.0.0.1:{probe_server.server_port}",
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert status == 204
    assert body == b""
    assert headers["Access-Control-Allow-Origin"] == "null"
    assert headers["Access-Control-Allow-Methods"] == "POST, OPTIONS"
    assert headers["Access-Control-Allow-Headers"] == "Content-Type"


def test_probe_echoes_matrix_and_returns_local_png(probe_server):
    encoded = json.dumps(sample_payload()).encode()
    status, headers, body = request(
        probe_server,
        "POST",
        "/probe",
        body=encoded,
        headers={
            "Host": f"127.0.0.1:{probe_server.server_port}",
            "Origin": "null",
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded)),
        },
    )
    payload = json.loads(body)

    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "null"
    assert payload["matrix"] == [[1, 2], ["A", None]]
    assert Path(payload["imagePath"]).is_file()


def test_unknown_browser_origin_is_rejected(probe_server):
    encoded = json.dumps(sample_payload()).encode()
    status, headers, body = request(
        probe_server,
        "POST",
        "/probe",
        body=encoded,
        headers={
            "Host": f"127.0.0.1:{probe_server.server_port}",
            "Origin": "https://example.invalid",
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded)),
        },
    )

    assert status == 403
    assert "Access-Control-Allow-Origin" not in headers
    assert json.loads(body)["error"]["code"] == "origin_denied"
