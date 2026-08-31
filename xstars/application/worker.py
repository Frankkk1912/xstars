"""Single-job WPS analysis worker.

The broker writes one validated request into a controlled job directory and
launches this module in a child process.  Any Tk dialog is created on the
worker's main thread; the HTTP server never imports or creates GUI objects.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import fields
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..config import PrismConfig
from ..data_handler import DataHandler
from .analysis import analyze_selection
from .contracts import (
    SCHEMA_VERSION,
    Artifact,
    ArtifactFormat,
    Command,
    ContractError,
    ErrorCode,
    SelectionPayload,
    cell_to_a1,
    parse_cell,
)

MAX_REQUEST_BYTES = 1_048_576


class WorkerCancelled(Exception):
    """Raised when either the caller or a GUI dialog cancels a job."""


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace *path* with JSON without exposing a partial result."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _controlled_paths(request_path: Path, result_path: Path) -> tuple[Path, Path]:
    if not request_path.is_absolute() or not result_path.is_absolute():
        raise ValueError(
            "request and result paths must be absolute and share a job directory"
        )
    request = request_path.resolve(strict=False)
    result = result_path.resolve(strict=False)
    if request.parent != result.parent:
        raise ValueError(
            "request and result paths must be absolute and share a job directory"
        )
    if request == result:
        raise ValueError("request and result paths must differ")
    return request, result


def _read_request(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"request file is unavailable: {exc}") from exc
    if size > MAX_REQUEST_BYTES:
        raise ContractError(ErrorCode.PAYLOAD_TOO_LARGE, "worker request is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"worker request is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("worker request must be an object")
    return payload


def _cancel_path(request: Mapping[str, Any], job_directory: Path) -> Path:
    raw = request.get("cancelPath")
    if not isinstance(raw, str):
        raise ValueError("cancelPath is required")
    candidate = Path(raw).resolve(strict=False)
    if not candidate.is_absolute() or candidate.parent != job_directory:
        raise ValueError("cancelPath must be inside the controlled job directory")
    return candidate


def _check_cancelled(cancel_path: Path) -> None:
    if cancel_path.exists():
        raise WorkerCancelled("request cancelled")


def _apply_config_overrides(config: PrismConfig, raw: Any) -> PrismConfig:
    if raw is None:
        return config
    if not isinstance(raw, Mapping):
        raise ValueError("config must be an object")
    allowed = {item.name for item in fields(PrismConfig)} - {
        "export_path",
        "ic50_fit_info",
        "elisa_fit_result",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unsupported config fields: {', '.join(sorted(unknown))}")
    for name, value in raw.items():
        current = getattr(config, name)
        if isinstance(current, Enum):
            try:
                value = type(current)(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value for config field {name}") from exc
        setattr(config, name, value)
    return config


def _output_start_cell(selection: SelectionPayload) -> str:
    first = selection.address.split(":", 1)[0]
    start_row, start_column = parse_cell(first)
    return cell_to_a1(start_row + len(selection.values) + 2, start_column)


def _dialog_config(
    selection: SelectionPayload, base_config: PrismConfig
) -> PrismConfig:
    """Show the existing settings dialog on the current (main) thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Tkinter dialogs must run on the worker main thread")
    frame = DataHandler.from_selection_payload(selection)
    from ..ui_dialog import SettingsDialog  # GUI import remains worker-only.

    dialog = SettingsDialog(
        DataHandler.group_names(frame),
        DataHandler.group_sizes(frame),
        base_config=base_config,
    )
    chosen = dialog.show()
    if chosen is None:
        raise WorkerCancelled("settings dialog cancelled")
    return chosen


def execute_request(
    request: Mapping[str, Any],
    job_directory: Path,
    *,
    dialog_config: Callable[
        [SelectionPayload, PrismConfig], PrismConfig
    ] = _dialog_config,
) -> dict[str, Any]:
    """Execute one validated Run/Quick request and return JSON-safe output."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("the WPS worker must execute on its main thread")
    if request.get("version") != SCHEMA_VERSION:
        raise ValueError("unsupported worker request version")
    try:
        command = Command(request.get("command"))
    except (TypeError, ValueError) as exc:
        raise ContractError(
            ErrorCode.INVALID_COMMAND, "command is not whitelisted"
        ) from exc
    if command not in {Command.RUN, Command.QUICK}:
        raise ContractError(
            ErrorCode.INVALID_COMMAND,
            f"command is not implemented by the M3 worker: {command.value}",
        )
    selection_data = request.get("selection")
    if not isinstance(selection_data, Mapping):
        raise ContractError(ErrorCode.INVALID_SELECTION, "selection must be an object")
    selection = SelectionPayload.from_dict(selection_data)
    cancel_path = _cancel_path(request, job_directory)
    _check_cancelled(cancel_path)

    config = _apply_config_overrides(PrismConfig.load(), request.get("config"))
    if command is Command.RUN:
        config = dialog_config(selection, config)
    _check_cancelled(cancel_path)

    artifact_path = (job_directory / "chart.png").resolve()
    config.export_path = str(artifact_path)
    config.export_format = "png"
    result = analyze_selection(
        selection,
        config,
        output_start_cell=_output_start_cell(selection),
        image_name="XSTARS_Plot_1",
    )
    _check_cancelled(cancel_path)

    for image in result.writeback_plan.images:
        image.artifact = Artifact(
            path=str(artifact_path),
            format=ArtifactFormat.PNG,
            dpi=config.export_dpi,
        )
        image.source_key = None
    from matplotlib import pyplot as plt

    plt.close(result.figure)
    return {
        "version": SCHEMA_VERSION,
        "ok": True,
        "status": "ok",
        "command": command.value,
        "writebackPlan": result.writeback_plan.to_dict(),
    }


def _error_result(code: ErrorCode, message: str) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "ok": False,
        "status": "cancelled" if code is ErrorCode.CANCELLED else "error",
        "error": {"version": SCHEMA_VERSION, "code": code.value, "message": message},
    }


def run_worker(
    request_path: str | Path,
    result_path: str | Path,
    *,
    executor: Callable[[Mapping[str, Any], Path], dict[str, Any]] = execute_request,
) -> int:
    """Run one job, always attempting to atomically publish a result file."""
    try:
        request_file, result_file = _controlled_paths(
            Path(request_path), Path(result_path)
        )
    except Exception:
        return 2

    artifact_path = request_file.parent / "chart.png"
    try:
        request = _read_request(request_file)
        output = executor(request, request_file.parent)
        exit_code = 0
    except WorkerCancelled as exc:
        artifact_path.unlink(missing_ok=True)
        output = _error_result(ErrorCode.CANCELLED, str(exc))
        exit_code = 0
    except ContractError as exc:
        artifact_path.unlink(missing_ok=True)
        output = _error_result(exc.code, str(exc))
        exit_code = 1
    except Exception as exc:
        artifact_path.unlink(missing_ok=True)
        output = _error_result(ErrorCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        exit_code = 1

    try:
        atomic_write_json(result_file, output)
    except OSError:
        return 2
    return exit_code
