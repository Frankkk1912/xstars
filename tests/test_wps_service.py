"""HTTP, security, lifecycle, and subprocess tests for the WPS broker."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xstars.application.contracts import SCHEMA_VERSION, TransformOptions
from xstars.config import ExperimentPreset, JournalPreset, PrismConfig

pytest = import_module("pytest")
_service = import_module("xstars.wps_service")
LOOPBACK_HOST = _service.LOOPBACK_HOST
SubprocessJobRunner = _service.SubprocessJobRunner
create_server = _service.create_server
load_or_create_token = _service.load_or_create_token
origin_allowed = _service.origin_allowed
persist_service_port = _service.persist_service_port

TOKEN = "t" * 48


def _payload(command: str = "run_quick") -> dict:
    return {
        "version": SCHEMA_VERSION,
        "command": command,
        "selection": {
            "version": SCHEMA_VERSION,
            "values": [
                ["Control", "Treatment"],
                [1.0, 2.0],
                [1.1, 2.1],
                [0.9, 1.9],
            ],
            "address": "$A$1:$B$4",
            "sheet": "Data",
        },
        "config": {"output_stats": True, "output_data": False},
    }


class FakeRunner:
    def __init__(self, response=None):
        self.response = response or {
            "version": SCHEMA_VERSION,
            "ok": True,
            "status": "ok",
            "writebackPlan": {"version": SCHEMA_VERSION, "tables": [], "images": []},
        }
        self.calls = []

    def run(self, request):
        self.calls.append(request)
        return self.response


class BlockingRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, request):
        self.calls.append(request)
        self.entered.set()
        assert self.release.wait(timeout=5)
        return self.response


@contextmanager
def _running_server(runner=None, **kwargs):
    service = create_server(0, token=TOKEN, runner=runner or FakeRunner(), **kwargs)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield service
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)


def _request(service, path, *, method="GET", payload=None, headers=None):
    url = f"http://{LOOPBACK_HOST}:{service.server_address[1]}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers or {},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            raw = response.read()
            return (
                response.status,
                dict(response.headers),
                json.loads(raw) if raw else {},
            )
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, dict(error.headers), json.loads(raw) if raw else {}


def _auth(origin="null"):
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Origin": origin,
    }


def test_server_is_bound_only_to_ipv4_loopback():
    service = create_server(0, token=TOKEN, runner=FakeRunner())
    try:
        assert service.server_address[0] == "127.0.0.1"
        assert service.server_address[0] != ".".join(("0", "0", "0", "0"))
    finally:
        service.server_close()
    with pytest.raises(ValueError, match="only bind"):
        all_interfaces = ".".join(("0", "0", "0", "0"))
        _service.WPSHTTPServer((all_interfaces, 0), TOKEN, FakeRunner())


def test_health_reports_pid_uptime_port_mask_and_not_full_token():
    with _running_server() as service:
        status, _headers, result = _request(service, "/health")
    assert status == 200
    assert result["ok"] is True
    assert result["pid"] == os.getpid()
    assert result["port"] > 0
    assert result["uptimeSeconds"] >= 0
    assert result["tokenMask"] != TOKEN
    assert TOKEN not in json.dumps(result)


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token"])
def test_command_rejects_missing_or_wrong_token(authorization):
    headers = {"Content-Type": "application/json", "Origin": "null"}
    if authorization is not None:
        headers["Authorization"] = authorization
    with _running_server() as service:
        status, _response_headers, result = _request(
            service, "/command", method="POST", payload=_payload(), headers=headers
        )
    assert status == 401
    assert result["error"]["code"] == "UNAUTHORIZED"


def test_origin_preflight_accepts_observed_origins_and_echoes_without_wildcard():
    with _running_server(addin_ports=frozenset({3890})) as service:
        for origin in ("null", "file://", "http://127.0.0.1:3890"):
            status, headers, result = _request(
                service,
                "/command",
                method="OPTIONS",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization,content-type",
                },
            )
            assert status == 204
            assert result == {}
            assert headers["Access-Control-Allow-Origin"] == origin
            assert headers["Access-Control-Allow-Origin"] != chr(42)


def test_origin_rejects_web_pages_and_unlisted_loopback_ports():
    assert not origin_allowed("https://evil.example", frozenset({3890}))
    assert not origin_allowed("http://127.0.0.1:9999", frozenset({3890}))
    with _running_server(addin_ports=frozenset({3890})) as service:
        status, _headers, result = _request(
            service,
            "/command",
            method="OPTIONS",
            headers={"Origin": "https://evil.example"},
        )
    assert status == 403
    assert result["error"]["code"] == "ORIGIN_DENIED"


def test_illegal_command_and_untrusted_extra_fields_are_rejected():
    with _running_server() as service:
        status, _headers, result = _request(
            service,
            "/command",
            method="POST",
            payload=_payload("__import__('os').system('whoami')"),
            headers=_auth(),
        )
        assert status == 400
        assert result["error"]["code"] == "INVALID_COMMAND"

        payload = _payload()
        payload["cancelPath"] = r"C:\outside\cancel"
        status, _headers, result = _request(
            service, "/command", method="POST", payload=payload, headers=_auth()
        )
        assert status == 400
        assert result["error"]["code"] == "INVALID_REQUEST"


def test_oversized_request_is_rejected_before_json_parsing():
    with _running_server(max_request_bytes=128) as service:
        status, _headers, result = _request(
            service,
            "/command",
            method="POST",
            payload={"padding": "x" * 512},
            headers=_auth(),
        )
    assert status == 413
    assert result["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_preset_elisa_export_and_setting_commands_are_shape_validated():
    runner = FakeRunner()
    with _running_server(runner=runner) as service:
        preset = _payload("run_wb")
        assert (
            _request(
                service, "/command", method="POST", payload=preset, headers=_auth()
            )[0]
            == 200
        )

        elisa = _payload("run_elisa")
        elisa["sampleSelection"] = {
            "version": SCHEMA_VERSION,
            "values": [["Control", "Treatment"], [0.2, 0.4], [0.3, 0.5]],
            "address": "E1:F3",
            "sheet": "Data",
        }
        assert (
            _request(
                service, "/command", method="POST", payload=elisa, headers=_auth()
            )[0]
            == 200
        )
        assert runner.calls[-1]["sampleSelection"]["address"] == "E1:F3"

        export = {
            "version": SCHEMA_VERSION,
            "command": "run_export",
            "config": {},
            "export": {
                "pictureId": "XSTARS_20260831_abcdef123456",
                "format": "png",
                "dpi": 300,
                "clipboard": False,
            },
        }
        assert (
            _request(
                service, "/command", method="POST", payload=export, headers=_auth()
            )[0]
            == 200
        )
        assert "selection" not in runner.calls[-1]

        setting = {
            "version": SCHEMA_VERSION,
            "command": "run_set_theme_nature",
            "config": {},
        }
        assert (
            _request(
                service, "/command", method="POST", payload=setting, headers=_auth()
            )[0]
            == 200
        )
        assert runner.calls[-1]["command"] == "run_set_theme_nature"


def test_standard_curve_stages_and_options_are_shape_validated():
    runner = FakeRunner()
    with _running_server(runner=runner) as service:
        configure = _payload("run_standard_curve")
        configure["stage"] = "configure"
        status = _request(
            service, "/command", method="POST", payload=configure, headers=_auth()
        )[0]
        assert status == 200
        assert runner.calls[-1]["stage"] == "configure"

        execute = _payload("run_standard_curve")
        execute.update(
            {
                "stage": "execute",
                "curveOptions": {
                    "fitMethod": "linear",
                    "backCalculate": False,
                },
            }
        )
        status = _request(
            service, "/command", method="POST", payload=execute, headers=_auth()
        )[0]
        assert status == 200
        assert runner.calls[-1]["curveOptions"]["fitMethod"] == "linear"

        invalid = _payload("run_standard_curve")
        invalid.update(
            {
                "stage": "execute",
                "curveOptions": {
                    "fitMethod": "__import__",
                    "backCalculate": False,
                },
            }
        )
        status, _headers, result = _request(
            service, "/command", method="POST", payload=invalid, headers=_auth()
        )
        assert status == 400
        assert result["error"]["code"] == "INVALID_REQUEST"


def test_elisa_requires_second_selection_and_export_rejects_extra_fields():
    with _running_server() as service:
        status, _headers, result = _request(
            service,
            "/command",
            method="POST",
            payload=_payload("run_elisa"),
            headers=_auth(),
        )
        assert status == 400
        assert result["error"]["code"] == "INVALID_SELECTION"

        export = {
            "version": SCHEMA_VERSION,
            "command": "run_export",
            "config": {},
            "export": {"format": "png", "dpi": 300, "targetPath": "C:/outside"},
        }
        status, _headers, result = _request(
            service, "/command", method="POST", payload=export, headers=_auth()
        )
        assert status == 400
        assert result["error"]["code"] == "INVALID_REQUEST"


def test_valid_command_is_normalized_before_runner_dispatch():
    runner = FakeRunner()
    with _running_server(runner=runner) as service:
        status, _headers, result = _request(
            service,
            "/command",
            method="POST",
            payload=_payload(),
            headers=_auth("file://"),
        )
    assert status == 200
    assert result["ok"] is True
    assert runner.calls[0]["command"] == "run_quick"
    assert runner.calls[0]["selection"]["address"] == "$A$1:$B$4"
    assert "cancelPath" not in runner.calls[0]


def test_cancel_endpoint_requires_authentication():
    with _running_server() as service:
        status, _headers, result = _request(
            service,
            "/cancel",
            method="POST",
            headers={"Origin": "null"},
        )
    assert status == 401
    assert result["error"]["code"] == "UNAUTHORIZED"


def test_cancel_endpoint_stops_active_job_and_releases_single_task_lock():
    class CancelableRunner(BlockingRunner):
        def cancel_current(self, *, grace=3.0):
            self.release.set()
            self.response = {
                "version": SCHEMA_VERSION,
                "ok": False,
                "status": "cancelled",
                "error": {"code": "CANCELLED", "message": "request cancelled"},
            }
            return True

    runner = CancelableRunner()
    first_response = []
    with _running_server(runner=runner) as service:
        first = threading.Thread(
            target=lambda: first_response.append(
                _request(
                    service,
                    "/command",
                    method="POST",
                    payload=_payload(),
                    headers=_auth(),
                )
            )
        )
        first.start()
        assert runner.entered.wait(timeout=5)
        status, _headers, cancelled = _request(
            service, "/cancel", method="POST", headers=_auth()
        )
        first.join(timeout=5)
        health = _request(service, "/health")[2]
        second = _request(
            service,
            "/command",
            method="POST",
            payload=_payload(),
            headers=_auth(),
        )
    assert status == 200
    assert cancelled == {
        "version": SCHEMA_VERSION,
        "ok": True,
        "status": "cancelled",
        "cancelled": True,
        "busy": False,
    }
    assert not first.is_alive()
    assert first_response[0][2]["status"] == "cancelled"
    assert health["busy"] is False
    assert second[0] == 200


def test_late_cancel_after_completed_generation_does_not_cancel_next_job(tmp_path):
    class CompletedProcess:
        returncode = 0

        def __init__(self, command):
            result_index = command.index("--result") + 1
            self.result_path = Path(command[result_index])
            self.cancel_path = self.result_path.parent / "cancel"
            self.saw_cancel = False

        def wait(self, timeout):
            self.saw_cancel = self.cancel_path.exists()
            self.result_path.write_text(
                json.dumps(
                    {
                        "version": SCHEMA_VERSION,
                        "ok": True,
                        "status": "ok",
                        "writebackPlan": {
                            "version": SCHEMA_VERSION,
                            "tables": [],
                            "images": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            return self.returncode

    class CompletionWindowRunner(SubprocessJobRunner):
        def __init__(self, jobs_root):
            super().__init__(jobs_root, timeout=1)
            self.completed = threading.Event()
            self.release_response = threading.Event()
            self.run_count = 0

        def run(self, request):
            result = super().run(request)
            self.run_count += 1
            if self.run_count == 1:
                self.completed.set()
                assert self.release_response.wait(timeout=5)
            return result

    processes = []

    def start_process(command, **_kwargs):
        process = CompletedProcess(command)
        processes.append(process)
        return process

    runner = CompletionWindowRunner(tmp_path / "jobs")
    first_response = []
    with (
        mock.patch.object(_service.subprocess, "Popen", side_effect=start_process),
        _running_server(runner=runner) as service,
    ):
        first = threading.Thread(
            target=lambda: first_response.append(
                _request(
                    service,
                    "/command",
                    method="POST",
                    payload=_payload(),
                    headers=_auth(),
                )
            )
        )
        first.start()
        try:
            assert runner.completed.wait(timeout=5)
            assert runner._active_state == "completed"
            assert service.job_lock.locked() is True

            status, _headers, cancellation = _request(
                service, "/cancel", method="POST", headers=_auth()
            )
            assert status == 200
            assert cancellation == {
                "version": SCHEMA_VERSION,
                "ok": True,
                "status": "idle",
                "cancelled": False,
                "busy": True,
                "reason": "no-active-job",
            }
            assert runner._pending_cancel_generation is None
        finally:
            runner.release_response.set()
            first.join(timeout=5)

        assert not first.is_alive()
        assert first_response[0][0] == 200
        second = _request(
            service,
            "/command",
            method="POST",
            payload=_payload(),
            headers=_auth(),
        )

    assert second[0] == 200
    assert runner._job_generation == 2
    assert len(processes) == 2
    assert processes[1].saw_cancel is False
    assert runner._pending_cancel_generation is None


def test_single_task_lock_returns_busy_for_concurrent_request():
    runner = BlockingRunner()
    first_response = []
    with _running_server(runner=runner) as service:
        first = threading.Thread(
            target=lambda: first_response.append(
                _request(
                    service,
                    "/command",
                    method="POST",
                    payload=_payload(),
                    headers=_auth(),
                )
            )
        )
        first.start()
        assert runner.entered.wait(timeout=5)
        status, _headers, result = _request(
            service, "/command", method="POST", payload=_payload(), headers=_auth()
        )
        assert status == 409
        assert result["error"]["code"] == "BUSY"
        runner.release.set()
        first.join(timeout=5)
    assert not first.is_alive()
    assert first_response[0][0] == 200


def test_active_port_conflict_has_diagnostic_error():
    first = create_server(0, token=TOKEN, runner=FakeRunner())
    port = first.server_address[1]
    try:
        with pytest.raises(RuntimeError, match=rf"127\.0\.0\.1:{port}"):
            create_server(port, token=TOKEN, runner=FakeRunner())
    finally:
        first.server_close()


def test_instance_token_is_random_persistent_and_stored_without_plaintext_logs(
    tmp_path,
):
    config_path = tmp_path / ".xstars" / "wps_service.json"
    first = load_or_create_token(config_path)
    second = load_or_create_token(config_path)
    assert first == second
    assert len(first) >= 32
    assert json.loads(config_path.read_text(encoding="utf-8"))["token"] == first
    if os.name != "nt":
        assert config_path.stat().st_mode & 0o077 == 0


def test_bound_port_is_atomically_persisted_without_rotating_token(tmp_path):
    config_path = tmp_path / ".xstars" / "wps_service.json"
    token = load_or_create_token(config_path)
    original = json.loads(config_path.read_text(encoding="utf-8"))
    original["installerField"] = "preserved"
    config_path.write_text(json.dumps(original), encoding="utf-8")

    persist_service_port(40123, config_path)

    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored == {
        "version": SCHEMA_VERSION,
        "token": token,
        "installerField": "preserved",
        "port": 40123,
    }
    assert not list(config_path.parent.glob("*.tmp"))
    if os.name != "nt":
        assert config_path.stat().st_mode & 0o077 == 0


def test_worker_routes_preset_setting_and_export_commands_without_host_io(tmp_path):
    worker = import_module("xstars.application.worker")
    preset_request = _payload("run_wb")
    preset_request["cancelPath"] = str((tmp_path / "cancel").resolve())
    observed = []
    fake_export = SimpleNamespace(
        new_picture_id=lambda: "XSTARS_20260831_abcdef123456",
        persist_render_payload=lambda *args, **kwargs: tmp_path / "payload.json",
    )

    def choose(_selection, config):
        observed.append(config.experiment_preset)
        return config

    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(worker, "import_module", return_value=fake_export),
    ):
        preset_result = worker.execute_request(
            preset_request,
            tmp_path,
            dialog_config=choose,
        )
    assert preset_result["ok"] is True
    assert observed == [ExperimentPreset.WB]
    assert preset_result["writebackPlan"]["images"][0]["pictureId"].startswith(
        "XSTARS_"
    )

    setting_request = {
        "version": SCHEMA_VERSION,
        "command": "run_set_theme_nature",
        "config": {},
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    settings_path = tmp_path / "settings.json"
    config_module = import_module("xstars.config")
    with mock.patch.object(config_module, "DEFAULT_SETTINGS_PATH", settings_path):
        setting_result = worker.execute_request(setting_request, tmp_path)
        reloaded = PrismConfig.load()
    assert setting_result["ok"] is True
    assert reloaded.journal_preset is JournalPreset.NATURE
    assert settings_path.is_file()

    exported = {
        "path": str((tmp_path / "chart.png").resolve()),
        "format": "png",
        "dpi": 300,
        "source": "render_payload",
    }
    fake_export = SimpleNamespace(
        render_payload_export=mock.Mock(return_value=exported)
    )
    export_request = {
        "version": SCHEMA_VERSION,
        "command": "run_export",
        "config": {},
        "export": {
            "pictureId": "XSTARS_20260831_abcdef123456",
            "format": "png",
            "dpi": 300,
        },
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(worker, "import_module", return_value=fake_export),
    ):
        export_result = worker.execute_request(export_request, tmp_path)
    assert export_result["export"] == exported
    fake_export.render_payload_export.assert_called_once_with(
        "XSTARS_20260831_abcdef123456", "png", 300
    )


def test_worker_transform_uses_specialized_dialog_and_include_stats(tmp_path):
    worker = import_module("xstars.application.worker")
    request = _payload("run_transform_only")
    request["cancelPath"] = str((tmp_path / "cancel").resolve())
    chosen = []

    def choose_transform(_selection, config):
        chosen.append(True)
        return config, TransformOptions(include_stats=True)

    with mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()):
        result = worker.execute_request(
            request,
            tmp_path,
            transform_dialog=choose_transform,
        )
    assert chosen == [True]
    assert len(result["writebackPlan"]["tables"]) == 3
    assert "with statistics" in result["writebackPlan"]["statusMessage"]
    assert "Ribbon" in result["writebackPlan"]["statusMessage"]


def test_transform_dialog_preconfigures_labeled_reference_mode():
    worker = import_module("xstars.application.worker")
    ui = import_module("xstars.ui_dialog")
    selection = worker.SelectionPayload.from_dict(
        {
            "version": SCHEMA_VERSION,
            "values": [
                ["Protein", "Control", "Treatment"],
                ["Target", 1.0, 2.0],
                ["GAPDH", 3.0, 3.1],
            ],
            "address": "A1:C3",
            "sheet": "WB",
        }
    )
    observed = {}
    dialog = mock.Mock()

    def make_dialog(_groups, _sizes, *, base_config):
        observed["config"] = base_config
        dialog.show.return_value = base_config
        return dialog

    with mock.patch.object(ui, "TransformOnlyDialog", side_effect=make_dialog):
        config, _options = worker._transform_dialog(selection, PrismConfig())
    assert config.preset_has_reference is True
    assert config.preset_control_group == "Control"


def test_wps_settings_dialog_disables_file_export_without_changing_default(tmp_path):
    worker = import_module("xstars.application.worker")
    ui = import_module("xstars.ui_dialog")
    selection = worker.SelectionPayload.from_dict(_payload()["selection"])
    chosen = PrismConfig(export_path="C:/must-not-survive.png")
    dialog = mock.Mock()
    dialog.show.return_value = chosen
    with mock.patch.object(ui, "SettingsDialog", return_value=dialog) as constructor:
        result = worker._dialog_config(selection, PrismConfig())
    assert result.export_path == ""
    assert constructor.call_args.kwargs["hide_file_export"] is True
    assert ui.SettingsDialog.__init__.__defaults__[-1] is False

    hidden = ui.SettingsDialog([], {}, base_config=PrismConfig(), hide_file_export=True)
    hidden._export_var = SimpleNamespace(get=lambda: True)
    hidden._export_fmt_combo = mock.Mock()
    hidden._export_dpi_combo = mock.Mock()
    hidden._export_entry = mock.Mock()
    hidden._browse_btn = mock.Mock()
    hidden._toggle_export()
    for widget in (
        hidden._export_fmt_combo,
        hidden._export_dpi_combo,
        hidden._export_entry,
        hidden._browse_btn,
    ):
        widget.configure.assert_called_once_with(state="disabled")


def test_worker_standard_curve_stages_dialog_before_optional_sample(tmp_path):
    worker = import_module("xstars.application.worker")
    curve = import_module("xstars.tools.standard_curve")
    standard = {
        "version": SCHEMA_VERSION,
        "values": [[1, 10, 100], [0.1, 1.0, 10.0], [0.11, 1.1, 10.1]],
        "address": "A1:C3",
        "sheet": "Curve",
    }
    fit = curve.fit_standard_curve(
        __import__("numpy").array([1, 10, 100], dtype=float),
        __import__("numpy").array([0.1, 1.0, 10.0], dtype=float),
        "linear",
    )
    configure = {
        "version": SCHEMA_VERSION,
        "command": "run_standard_curve",
        "selection": standard,
        "stage": "configure",
        "config": {},
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    with mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()):
        configured = worker.execute_request(
            configure,
            tmp_path,
            standard_dialog=lambda _selection, _config: (fit, True),
        )
    assert configured["writebackPlan"]["images"] == []
    assert configured["continuation"] == {
        "fitMethod": "linear",
        "backCalculate": True,
    }

    execute = dict(configure)
    execute.update(
        {
            "stage": "execute",
            "curveOptions": configured["continuation"],
            "sampleSelection": {
                "version": SCHEMA_VERSION,
                "values": [["Control", "Treatment"], [0.2, 0.4], [0.3, 0.5]],
                "address": "E1:F3",
                "sheet": "Curve",
            },
        }
    )
    fake_export = SimpleNamespace(
        new_picture_id=lambda: "XSTARS_20260831_abcdef123456",
        persist_render_payload=mock.Mock(),
    )
    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(worker, "import_module", return_value=fake_export),
    ):
        executed = worker.execute_request(
            execute,
            tmp_path,
            standard_dialog=mock.Mock(side_effect=AssertionError("dialog reopened")),
        )
    assert executed["writebackPlan"]["tables"][2]["values"] == [
        ["Back-Calculated Concentrations"]
    ]


def test_worker_standard_curve_dialog_choice_controls_back_calculation(tmp_path):
    worker = import_module("xstars.application.worker")
    curve = import_module("xstars.tools.standard_curve")
    request = {
        "version": SCHEMA_VERSION,
        "command": "run_standard_curve",
        "selection": {
            "version": SCHEMA_VERSION,
            "values": [[1, 10, 100], [0.1, 1.0, 10.0], [0.11, 1.1, 10.1]],
            "address": "A1:C3",
            "sheet": "Curve",
        },
        "sampleSelection": {
            "version": SCHEMA_VERSION,
            "values": [["Control", "Treatment"], [0.2, 0.4], [0.3, 0.5]],
            "address": "E1:F3",
            "sheet": "Curve",
        },
        "config": {},
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    fit = curve.fit_standard_curve(
        __import__("numpy").array([1, 10, 100], dtype=float),
        __import__("numpy").array([0.1, 1.0, 10.0], dtype=float),
        "linear",
    )
    fake_export = SimpleNamespace(
        new_picture_id=lambda: "XSTARS_20260831_abcdef123456",
        persist_render_payload=mock.Mock(),
    )
    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(worker, "import_module", return_value=fake_export),
    ):
        result = worker.execute_request(
            request,
            tmp_path,
            standard_dialog=lambda _selection, _config: (fit, True),
        )
    assert result["writebackPlan"]["tables"][2]["values"] == [
        ["Back-Calculated Concentrations"]
    ]
    assert Path(result["writebackPlan"]["images"][0]["artifact"]["path"]).is_file()


def test_worker_elisa_dialog_choice_adds_standard_curve_artifact(tmp_path):
    import_module("matplotlib.pyplot").close("all")
    worker = import_module("xstars.application.worker")
    curve = import_module("xstars.tools.standard_curve")
    request = {
        "version": SCHEMA_VERSION,
        "command": "run_elisa",
        "selection": {
            "version": SCHEMA_VERSION,
            "values": [[1, 10, 100], [0.1, 1.0, 10.0], [0.11, 1.1, 10.1]],
            "address": "A1:C3",
            "sheet": "ELISA",
        },
        "sampleSelection": {
            "version": SCHEMA_VERSION,
            "values": [
                ["Control", "Treatment"],
                [0.2, 0.4],
                [0.21, 0.42],
                [0.19, 0.38],
            ],
            "address": "E1:F4",
            "sheet": "ELISA",
        },
        "config": {},
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    fit = curve.fit_standard_curve(
        __import__("numpy").array([1, 10, 100], dtype=float),
        __import__("numpy").array([0.1, 1.0, 10.0], dtype=float),
        "linear",
    )
    picture_ids = iter(
        [
            "XSTARS_20260831_abcdef123456",
            "XSTARS_20260831_abcdef789012",
        ]
    )
    fake_export = SimpleNamespace(
        new_picture_id=lambda: next(picture_ids),
        persist_render_payload=mock.Mock(),
    )
    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(worker, "import_module", return_value=fake_export),
    ):
        result = worker.execute_request(
            request,
            tmp_path,
            elisa_dialog=lambda _selection, config: (config, fit, True),
        )
    images = result["writebackPlan"]["images"]
    assert len(images) == 2
    assert len({image["pictureId"] for image in images}) == 2
    assert all(Path(image["artifact"]["path"]).is_file() for image in images)
    persisted_frames = [
        call.args[1] for call in fake_export.persist_render_payload.call_args_list
    ]
    assert list(persisted_frames[1].columns) == ["1", "10", "100"]


def test_worker_labeled_wb_persists_one_artifact_and_render_payload_per_target(
    tmp_path,
):
    import_module("matplotlib.pyplot").close("all")
    worker = import_module("xstars.application.worker")
    export_module = import_module("xstars.application.export")
    request = {
        "version": SCHEMA_VERSION,
        "command": "run_wb",
        "selection": {
            "version": SCHEMA_VERSION,
            "values": [
                ["Protein", "Control", "Treatment_A", "Treatment_B"],
                ["Target-A", 12000, 28000, 6500],
                ["Target-A", 15000, 31000, 7200],
                ["Target-A", 11500, 26500, 5800],
                ["Target-B", 8000, 12000, 4000],
                ["Target-B", 9200, 13500, 3800],
                ["Target-B", 7800, 11000, 4200],
                ["GAPDH", 45000, 44000, 43000],
                ["GAPDH", 47000, 46000, 45000],
                ["GAPDH", 43000, 43500, 42500],
            ],
            "address": "A1:D10",
            "sheet": "WB",
        },
        "config": {
            "preset_reference_protein": "GAPDH",
            "output_stats": False,
            "output_data": True,
        },
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    artifacts_root = tmp_path / "render-payloads"
    real_persist = export_module.persist_render_payload

    def persist_in_test_root(picture_id, frame, config, figure, **kwargs):
        return real_persist(
            picture_id,
            frame,
            config,
            figure,
            artifacts_root=artifacts_root,
            **kwargs,
        )

    observed = []

    def choose(_selection, config):
        observed.append(
            (
                config.experiment_preset,
                config.preset_has_reference,
                config.preset_control_group,
            )
        )
        return config

    with (
        mock.patch.object(worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(
            export_module,
            "persist_render_payload",
            side_effect=persist_in_test_root,
        ),
        mock.patch.object(worker, "import_module", return_value=export_module),
    ):
        output = worker.execute_request(
            request,
            tmp_path,
            dialog_config=choose,
        )

    assert observed == [(ExperimentPreset.WB, True, "Control")]
    images = output["writebackPlan"]["images"]
    assert len(images) == 2
    assert len({image["pictureId"] for image in images}) == 2
    assert [Path(image["artifact"]["path"]).name for image in images] == [
        "chart_1.png",
        "chart_2.png",
    ]
    assert all(Path(image["artifact"]["path"]).is_file() for image in images)
    payloads = [
        json.loads(
            (artifacts_root / f"{image['pictureId']}.json").read_text(encoding="utf-8")
        )
        for image in images
    ]
    assert [payload["config"]["title"] for payload in payloads] == [
        "Target-A",
        "Target-B",
    ]
    assert all(
        payload["data"]["columns"] == ["Control", "Treatment_A", "Treatment_B"]
        for payload in payloads
    )


def _run_fake_completed_job(tmp_path, result_factory):
    jobs_root = tmp_path / "jobs"

    class CompletedProcess:
        returncode = 0

        def wait(self, timeout):
            job_directory = next(jobs_root.iterdir())
            result = result_factory(job_directory)
            (job_directory / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            return self.returncode

    runner = SubprocessJobRunner(jobs_root, timeout=1)
    with mock.patch.object(
        _service.subprocess, "Popen", return_value=CompletedProcess()
    ):
        result = runner.run(_payload())
    return jobs_root, result


@pytest.mark.parametrize("artifact_name", ["chart.png", "chart_1.png"])
def test_job_directory_is_preserved_for_referenced_internal_image(
    tmp_path, artifact_name
):
    def result_with_image(job_directory):
        artifact = job_directory / artifact_name
        artifact.write_bytes(b"image")
        return {"writebackPlan": {"images": [{"artifact": {"path": str(artifact)}}]}}

    jobs_root, result = _run_fake_completed_job(tmp_path, result_with_image)
    job_directory = jobs_root / result["jobId"]
    artifact = job_directory / artifact_name

    assert _service._should_preserve_job_directory(result, job_directory) is True
    assert job_directory.is_dir()
    assert artifact.is_file()
    assert not (job_directory / "request.json").exists()
    assert not (job_directory / "result.json").exists()


def test_job_directory_without_images_is_removed(tmp_path):
    jobs_root, result = _run_fake_completed_job(
        tmp_path, lambda _job_directory: {"writebackPlan": {"images": []}}
    )
    job_directory = jobs_root / result["jobId"]

    assert _service._should_preserve_job_directory(result, job_directory) is False
    assert not job_directory.exists()


def test_external_image_artifact_does_not_preserve_job_directory(tmp_path):
    external_artifact = tmp_path / "external.png"
    external_artifact.write_bytes(b"external image")

    def result_with_external_image(_job_directory):
        return {
            "writebackPlan": {
                "images": [{"artifact": {"path": str(external_artifact)}}]
            }
        }

    jobs_root, result = _run_fake_completed_job(tmp_path, result_with_external_image)
    job_directory = jobs_root / result["jobId"]

    assert _service._should_preserve_job_directory(result, job_directory) is False
    assert not job_directory.exists()
    assert external_artifact.is_file()


def test_subprocess_runner_cancel_touches_cancel_and_kills_after_grace(tmp_path):
    runner = SubprocessJobRunner(tmp_path / "jobs", timeout=1)

    class ActiveProcess:
        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = ActiveProcess()
    cancel_path = tmp_path / "jobs" / "active" / "cancel"
    cancel_path.parent.mkdir()
    runner._job_generation = 1
    runner._active_generation = 1
    runner._active_state = "active"
    runner._active_process = process
    runner._active_cancel_path = cancel_path

    assert runner.cancel_current(grace=0) is True
    assert cancel_path.is_file()
    assert process.killed is True


def test_subprocess_runner_active_generation_allows_cooperative_cancel(tmp_path):
    runner = SubprocessJobRunner(tmp_path / "jobs", timeout=1)
    cancel_path = tmp_path / "jobs" / "active" / "cancel"
    cancel_path.parent.mkdir()

    class CooperativeProcess:
        def __init__(self):
            self.killed = False

        def poll(self):
            return 0 if cancel_path.exists() else None

        def kill(self):
            self.killed = True

    process = CooperativeProcess()
    runner._job_generation = 1
    runner._active_generation = 1
    runner._active_state = "active"
    runner._active_process = process
    runner._active_cancel_path = cancel_path

    assert runner.cancel_current(grace=1) is True
    assert cancel_path.is_file()
    assert process.killed is False
    assert runner._pending_cancel_generation is None


def test_worker_timeout_signals_cancel_kills_process_and_cleans_job(tmp_path):
    jobs_root = tmp_path / "jobs"

    class HangingProcess:
        returncode = -9

        def __init__(self):
            self.wait_calls = 0
            self.killed = False
            self.saw_cancel = False

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 2:
                self.saw_cancel = any(jobs_root.glob("*/cancel"))
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

        def kill(self):
            self.killed = True

    process = HangingProcess()
    runner = SubprocessJobRunner(jobs_root, timeout=0.01)
    with (
        mock.patch.object(_service.subprocess, "Popen", return_value=process),
        pytest.raises(_service.WorkerFailure, match="timed out") as caught,
    ):
        runner.run(_payload())

    assert caught.value.code.value == "TIMEOUT"
    assert process.saw_cancel is True
    assert process.killed is True
    assert list(jobs_root.iterdir()) == []


def test_real_subprocess_quick_job_returns_result_and_cleans_ipc_files(tmp_path):
    runner = SubprocessJobRunner(tmp_path / "jobs", timeout=30)
    result = runner.run(_payload())
    job_directory = tmp_path / "jobs" / result["jobId"]

    assert result["ok"] is True
    assert result["command"] == "run_quick"
    artifact = Path(result["writebackPlan"]["images"][0]["artifact"]["path"])
    assert artifact.is_file()
    assert artifact.parent == job_directory
    assert not (job_directory / "request.json").exists()
    assert not (job_directory / "result.json").exists()
    assert not (job_directory / "cancel").exists()
