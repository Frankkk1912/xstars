import http.client
import json
import struct
import sys
import threading
import types
from pathlib import Path

import pytest  # type: ignore[import-not-found]

import poc.wps.service_server as service_module
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


@pytest.fixture
def gate0_service(tmp_path, monkeypatch):
    monkeypatch.setattr(service_module, "artifact_root", lambda: tmp_path)
    server = service_module.create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def service_post(server, path, payload):
    encoded = json.dumps(payload).encode()
    return request(
        server,
        "POST",
        path,
        body=encoded,
        headers={
            "Host": f"127.0.0.1:{server.server_port}",
            "Origin": "null",
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded)),
        },
    )


def test_elisa_selection_echoes_matrix_statistics(gate0_service):
    status, _, body = service_post(
        gate0_service,
        "/probe/elisa-selection",
        {
            "source": "two-stage",
            "ranges": [
                {"address": "$A$1:$B$2", "values": [[1, None], ["", 4]]},
                {"address": "$D$1:$D$2", "values": [[5], [6]]},
            ],
        },
    )
    payload = json.loads(body)

    assert status == 200
    assert payload["source"] == "two-stage"
    assert payload["ranges"][0] == {
        "address": "$A$1:$B$2",
        "values": [[1, None], ["", 4]],
        "rows": 2,
        "columns": 2,
        "nonEmptyCells": 2,
    }
    assert payload["ranges"][1]["nonEmptyCells"] == 2


@pytest.mark.parametrize(
    ("request_payload", "error_code"),
    [
        (
            {
                "source": "inputbox",
                "ranges": [{"address": "$A$1:$B$2", "values": [[1, 2], [3]]}],
            },
            "NON_RECTANGULAR_VALUES",
        ),
        (
            {
                "source": "inputbox",
                "ranges": [
                    {
                        "address": "$A$1:$A$201",
                        "values": [[index] for index in range(201)],
                    }
                ],
            },
            "SELECTION_TOO_LARGE",
        ),
        ({"source": "inputbox"}, "MISSING_RANGES"),
    ],
)
def test_elisa_selection_rejects_invalid_ranges(
    gate0_service, request_payload, error_code
):
    status, _, body = service_post(
        gate0_service, "/probe/elisa-selection", request_payload
    )

    assert status == 400
    assert json.loads(body)["error"]["code"] == error_code


@pytest.mark.parametrize(
    ("request_payload", "error_code"),
    [
        ({"format": "bmp", "dpi": 300}, "INVALID_FORMAT"),
        ({"format": "png", "dpi": 71}, "INVALID_DPI"),
        ({"format": "png", "dpi": 1201}, "INVALID_DPI"),
        ({"format": "png", "dpi": 300.5}, "INVALID_DPI"),
    ],
)
def test_shape_export_rejects_invalid_parameters(
    gate0_service, request_payload, error_code
):
    status, _, body = service_post(
        gate0_service, "/probe/shape-export", request_payload
    )

    assert status == 400
    assert json.loads(body)["error"]["code"] == error_code


def test_shape_export_reports_empty_clipboard_without_real_pillow(
    gate0_service, monkeypatch
):
    fake_pil = types.ModuleType("PIL")
    fake_pil.__dict__.update(
        Image=types.SimpleNamespace(),
        ImageGrab=types.SimpleNamespace(grabclipboard=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    status, _, body = service_post(
        gate0_service, "/probe/shape-export", {"format": "png", "dpi": 300}
    )

    assert status == 422
    assert json.loads(body)["error"]["code"] == "CLIPBOARD_EMPTY"


def install_fake_win32com(monkeypatch, get_active_object):
    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")
    client.__dict__["GetActiveObject"] = get_active_object
    win32com.__dict__["client"] = client
    monkeypatch.setitem(sys.modules, "win32com", win32com)
    monkeypatch.setitem(sys.modules, "win32com.client", client)


def test_com_probe_reports_running_wps(gate0_service, monkeypatch):
    install_fake_win32com(
        monkeypatch,
        lambda prog_id: types.SimpleNamespace(Version="12.1.0")
        if prog_id == "Ket.Application"
        else None,
    )

    status, _, body = request(gate0_service, "GET", "/probe/com-probe")
    payload = json.loads(body)

    assert status == 200
    assert payload == {
        "ok": True,
        "progId": "Ket.Application",
        "version": "12.1.0",
    }


def test_com_probe_returns_diagnostic_failure(gate0_service, monkeypatch):
    def unavailable(_prog_id):
        raise OSError("class string is invalid")

    install_fake_win32com(monkeypatch, unavailable)

    status, _, body = request(gate0_service, "GET", "/probe/com-probe")
    payload = json.loads(body)

    assert status == 200
    assert payload["ok"] is False
    assert payload["code"] == "COM_UNAVAILABLE"
    assert "class string is invalid" in payload["detail"]


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
