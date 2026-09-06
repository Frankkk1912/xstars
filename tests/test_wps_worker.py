"""Tests for the isolated WPS worker and CLI mode dispatch."""

from __future__ import annotations

import json
import sys
import threading
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")

from xstars.application.contracts import SCHEMA_VERSION
from xstars.config import PrismConfig

_worker = import_module("xstars.application.worker")
WorkerCancelled = _worker.WorkerCancelled
atomic_write_json = _worker.atomic_write_json
execute_request = _worker.execute_request
run_worker = _worker.run_worker


def _request(directory: Path, command: str = "run_quick") -> dict:
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
        "cancelPath": str((directory / "cancel").resolve()),
    }


def test_quick_worker_e2e_returns_plan_and_real_artifact(tmp_path):
    with mock.patch(
        "xstars.application.worker.PrismConfig.load", return_value=PrismConfig()
    ):
        output = execute_request(_request(tmp_path), tmp_path)

    assert output["ok"] is True
    assert output["command"] == "run_quick"
    plan = output["writebackPlan"]
    assert plan["tables"][0]["startCell"] == "A7"
    assert plan["images"][0]["anchorCell"] == "A10"
    artifact = plan["images"][0]["artifact"]
    assert Path(artifact["path"]).is_file()
    assert artifact["format"] == "png"
    assert "sourceKey" not in plan["images"][0]


def test_run_dialog_is_invoked_on_worker_main_thread(tmp_path):
    observed = []

    def choose_config(_selection, config):
        observed.append(threading.current_thread())
        return config

    with mock.patch(
        "xstars.application.worker.PrismConfig.load", return_value=PrismConfig()
    ):
        output = execute_request(
            _request(tmp_path, "run"), tmp_path, dialog_config=choose_config
        )

    assert output["ok"] is True
    assert observed == [threading.main_thread()]


def test_execute_request_rejects_non_main_thread_without_opening_gui(tmp_path):
    errors = []

    def invoke():
        try:
            execute_request(_request(tmp_path, "run"), tmp_path)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])


def test_pre_cancel_writes_cancelled_result_and_leaves_no_artifact(tmp_path):
    request_path = (tmp_path / "request.json").resolve()
    result_path = (tmp_path / "result.json").resolve()
    request = _request(tmp_path)
    Path(request["cancelPath"]).touch()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert run_worker(request_path, result_path) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert result["error"]["code"] == "CANCELLED"
    assert not (tmp_path / "chart.png").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_ui_cancel_is_structured_and_removes_temporary_artifact(tmp_path):
    request_path = (tmp_path / "request.json").resolve()
    result_path = (tmp_path / "result.json").resolve()
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")
    (tmp_path / "chart.png").write_bytes(b"partial")

    def cancelled(_request_data, _directory):
        raise WorkerCancelled("dialog cancelled")

    assert run_worker(request_path, result_path, executor=cancelled) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["error"]["code"] == "CANCELLED"
    assert not (tmp_path / "chart.png").exists()


def test_worker_crash_publishes_internal_error_and_nonzero_exit(tmp_path):
    request_path = (tmp_path / "request.json").resolve()
    result_path = (tmp_path / "result.json").resolve()
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")

    def crashes(_request_data, _directory):
        raise RuntimeError("boom")

    assert run_worker(request_path, result_path, executor=crashes) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert result["error"]["code"] == "INTERNAL_ERROR"
    assert "boom" in result["error"]["message"]


def test_result_write_uses_atomic_replace_and_cleans_temp_file(tmp_path):
    target = tmp_path / "result.json"
    real_replace = __import__("os").replace
    with mock.patch(
        "xstars.application.worker.os.replace", wraps=real_replace
    ) as replace:
        atomic_write_json(target, {"ok": True})
    replace.assert_called_once()
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_cli_command_and_workbook_syntax_is_unchanged(monkeypatch):
    from xstars import cli
    from xstars import main as application_main

    book = SimpleNamespace(set_mock_caller=mock.Mock())
    fake_xlwings = SimpleNamespace(Book=mock.Mock(return_value=book))
    monkeypatch.setitem(sys.modules, "xlwings", fake_xlwings)
    callback = mock.Mock()
    monkeypatch.setattr(application_main, "run_quick", callback)
    monkeypatch.setattr(
        sys,
        "argv",
        ["xstars.exe", "run_quick", r"C:\work\book.xlsx"],
    )

    cli.main()

    fake_xlwings.Book.assert_called_once_with(r"C:\work\book.xlsx")
    book.set_mock_caller.assert_called_once_with()
    callback.assert_called_once_with()


def test_serve_cli_mode_forwards_configurable_port(monkeypatch):
    from xstars import cli

    service_module = SimpleNamespace(DEFAULT_PORT=3892, serve=mock.Mock(return_value=6))
    real_import = cli.import_module

    def import_for_test(name):
        if name == "xstars.wps_service":
            return service_module
        return real_import(name)

    monkeypatch.setattr(cli, "import_module", import_for_test)
    monkeypatch.setattr(sys, "argv", ["xstars.exe", "serve", "--port", "40123"])

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 6
    else:
        raise AssertionError("serve mode must exit with its service status")
    service_module.serve.assert_called_once_with(40123)


def test_worker_cli_mode_forwards_only_request_and_result(monkeypatch, tmp_path):
    from xstars import cli

    worker_module = SimpleNamespace(run_worker=mock.Mock(return_value=7))
    real_import = cli.import_module

    def import_for_test(name):
        if name == "xstars.application.worker":
            return worker_module
        return real_import(name)

    monkeypatch.setattr(cli, "import_module", import_for_test)
    request = str(tmp_path / "request.json")
    result = str(tmp_path / "result.json")
    monkeypatch.setattr(
        sys,
        "argv",
        ["xstars.exe", "worker", "--request", request, "--result", result],
    )

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 7
    else:
        raise AssertionError("worker mode must exit with its worker status")
    worker_module.run_worker.assert_called_once_with(request, result)
