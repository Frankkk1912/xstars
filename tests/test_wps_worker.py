"""Tests for the isolated WPS worker and CLI mode dispatch."""

from __future__ import annotations

import gc
import json
import sys
import threading
import time
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import matplotlib
import pytest

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


def test_dialog_without_selection_uses_returned_payload_and_preserves_plan(tmp_path):
    request = _request(tmp_path, "run")
    payload = _worker.SelectionPayload.from_dict(request["selection"])
    with mock.patch.object(PrismConfig, "load", return_value=PrismConfig()):
        selected = execute_request(request, tmp_path, dialog_config=lambda selection, config: config)
        request.pop("selection")

        def choose(selection, config):
            assert selection is None
            return config, payload

        supplemented = execute_request(request, tmp_path, dialog_config=choose)
    assert supplemented["writebackPlan"]["tables"] == selected["writebackPlan"]["tables"]
    for key in ("anchorCell", "width", "height"):
        assert supplemented["writebackPlan"]["images"][0].get(key) == selected["writebackPlan"]["images"][0].get(key)
    assert supplemented["writebackPlan"]["images"][0]["artifact"]["path"] == selected["writebackPlan"]["images"][0]["artifact"]["path"]


def test_dialog_provider_populates_payload_and_range_label_without_tk(tmp_path):
    ui = import_module("xstars.ui_dialog")
    payload = _worker.SelectionPayload.from_dict(_request(tmp_path)["selection"])
    observed = []

    class FakeDialog:
        def __init__(self, groups, sizes, **kwargs):
            assert groups == []
            assert sizes == {}
            self.provider = kwargs["request_selection"]
            self._root = mock.Mock()

        def show(self):
            summary = self.provider()
            observed.append(summary)
            self.selected_payload = summary.payload
            return PrismConfig()

    with (
        mock.patch.object(ui, "SettingsDialog", FakeDialog),
        mock.patch.object(_worker, "_wait_dialog_interaction", return_value=payload),
    ):
        config, selected = _worker._dialog_config(None, PrismConfig(), job_directory=tmp_path,
                                                 cancel_path=tmp_path / "cancel")
    assert selected is payload
    assert observed[0].label == "Data!$A$1:$B$4"
    assert observed[0].group_sizes == {"Control": 3, "Treatment": 3}
    assert config.export_path == ""


def test_worker_interaction_file_success_cancel_timeout_and_stale_id(tmp_path):
    payload = _request(tmp_path)["selection"]
    cancel = tmp_path / "cancel"
    interaction_id = _worker.request_interaction(tmp_path, "Choose data")
    atomic_write_json(tmp_path / "interaction_response.json", {"id": "old", "selection": payload})
    assert _worker.wait_interaction(tmp_path, interaction_id, cancel, timeout=.01) is None
    interaction_id = _worker.request_interaction(tmp_path, "Choose data")
    atomic_write_json(tmp_path / "interaction_response.json", {"id": interaction_id, "selection": payload})
    assert _worker.wait_interaction(tmp_path, interaction_id, cancel).address == payload["address"]
    interaction_id = _worker.request_interaction(tmp_path, "Choose data")
    atomic_write_json(tmp_path / "interaction_response.json", {"id": interaction_id, "selection": None})
    assert _worker.wait_interaction(tmp_path, interaction_id, cancel) is None
    interaction_id = _worker.request_interaction(tmp_path, "Choose data")
    cancel.touch()
    import pytest
    with pytest.raises(WorkerCancelled):
        _worker.wait_interaction(tmp_path, interaction_id, cancel)
    assert not (tmp_path / "interaction_request.json").exists()


def test_preselection_without_interaction_hides_provider(tmp_path):
    ui = import_module("xstars.ui_dialog")
    selection = _worker.SelectionPayload.from_dict(_request(tmp_path)["selection"])
    dialog = mock.Mock()
    dialog.show.return_value = PrismConfig()
    with mock.patch.object(ui, "SettingsDialog", return_value=dialog) as constructor:
        _worker._dialog_config(selection, PrismConfig(), interaction_enabled=False)
    assert constructor.call_args.kwargs["request_selection"] is None
    assert constructor.call_args.kwargs["selection_label"] == "Data!$A$1:$B$4"


def test_worker_no_selection_cancel_keeps_job_cancel_semantics(tmp_path):
    request = _request(tmp_path, "run")
    request.pop("selection")
    with pytest.raises(WorkerCancelled):
        execute_request(request, tmp_path, dialog_config=lambda selection, config: config)


@pytest.fixture
def interaction_root():
    """Real Tk event loop: no doubles for after, variables, bind or wait_variable."""
    import tkinter as tk

    class MainThreadRoot(tk.Tk):
        def after(self, ms, func: Any = None, *args):
            assert threading.current_thread() is threading.main_thread()
            return super().after(ms, func, *args)

        def wait_variable(self, name="PY_VAR"):
            assert threading.current_thread() is threading.main_thread()
            return super().wait_variable(name)

    try:
        root = MainThreadRoot()
    except tk.TclError as exc:
        pytest.skip(f"real Tk display unavailable: {exc}")
    root.withdraw()
    # Bound a regression-induced deadlock without mocking the wait machinery.
    root.after(5000, root.destroy)
    yield root
    with suppress(tk.TclError):
        root.destroy()
    # Tk variables in callback cycles must be finalized on the main thread,
    # not by an unrelated broker thread's later cyclic-GC collection.
    gc.collect()


def publish_interaction_response(directory, selection, cancel=False):
    deadline = time.monotonic() + 3
    request_path = directory / "interaction_request.json"
    while not request_path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert request_path.exists()
    interaction_id = json.loads(request_path.read_text())["id"]
    if cancel:
        (directory / "cancel").touch()
    else:
        atomic_write_json(directory / "interaction_response.json",
                          {"id": interaction_id, "selection": selection})


@pytest.mark.parametrize("outcome", ["success", "cancel", "destroy", "invalid"])
def test_wait_dialog_interaction_real_tk_and_file_thread(tmp_path, interaction_root, outcome):
    root = interaction_root
    selection = _request(tmp_path)["selection"]
    if outcome == "invalid":
        selection = {**selection, "version": "bad"}
    if outcome == "destroy":
        root.after(100, root.destroy)
        responder = None
    else:
        responder = threading.Thread(target=publish_interaction_response,
                                     args=(tmp_path, selection, outcome == "cancel"))
        responder.start()
    try:
        if outcome in ("cancel", "destroy"):
            with pytest.raises(WorkerCancelled):
                _worker._wait_dialog_interaction(root, tmp_path, tmp_path / "cancel")
        elif outcome == "invalid":
            with pytest.raises(_worker.ContractError):
                _worker._wait_dialog_interaction(root, tmp_path, tmp_path / "cancel")
        else:
            selected = _worker._wait_dialog_interaction(root, tmp_path, tmp_path / "cancel")
            assert selected.to_dict() == selection
    finally:
        if responder is not None:
            responder.join(timeout=3)
            assert not responder.is_alive()
    deadline = time.monotonic() + 2
    while (tmp_path / "interaction_request.json").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert not list(tmp_path.glob("interaction_*.json"))
    if outcome != "destroy":
        # The temporary Destroy binding must have been removed even on cancel.
        assert root.bind("<Destroy>") == ""


def test_wait_interaction_accepts_response_at_deadline(tmp_path):
    selection = _request(tmp_path)["selection"]
    interaction_id = _worker.request_interaction(tmp_path, "Choose")
    atomic_write_json(tmp_path / "interaction_response.json",
                      {"id": interaction_id, "selection": selection})
    assert _worker.wait_interaction(tmp_path, interaction_id, tmp_path / "cancel", timeout=0).to_dict() == selection


def test_dialog_config_existing_selection_reselects_through_real_file_wait(tmp_path, interaction_root):
    ui = import_module("xstars.ui_dialog")
    original = _worker.SelectionPayload.from_dict(_request(tmp_path)["selection"])
    replacement = {**original.to_dict(), "address": "$D$8:$E$11"}

    class FileSelectingDialog:
        def __init__(self, groups, sizes, **kwargs):
            assert groups == ["Control", "Treatment"]
            self.provider = kwargs["request_selection"]
            self._root = interaction_root
            self._closed = False

        def show(self):
            summary = self.provider()
            assert summary.label == "Data!$D$8:$E$11"
            self.selected_payload = summary.payload
            return PrismConfig()

    responder = threading.Thread(target=publish_interaction_response, args=(tmp_path, replacement))
    responder.start()
    try:
        # Only the outer settings layout is substituted, not the concurrency path.
        with mock.patch.object(ui, "SettingsDialog", FileSelectingDialog):
            _, selected = _worker._dialog_config(original, PrismConfig(), job_directory=tmp_path,
                                                cancel_path=tmp_path / "cancel", interaction_enabled=True)
        assert selected.to_dict() == replacement
    finally:
        responder.join(timeout=3)
        assert not responder.is_alive()
