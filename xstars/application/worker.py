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
from contextlib import suppress
from dataclasses import fields
from enum import Enum
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..config import (
    BaseTheme,
    ExperimentPreset,
    JournalPalette,
    JournalPreset,
    PalettePreset,
    PrismConfig,
)
from ..data_handler import DataHandler
from ..styles import get_palette
from .analysis import (
    AnalysisResult,
    StandardCurveResult,
    analyze_selection,
    elisa_selections,
    guess_blank,
    guess_control,
    standard_curve_selection,
    transform_selection,
)
from .contracts import (
    SCHEMA_VERSION,
    Artifact,
    ArtifactFormat,
    Command,
    ContractError,
    ErrorCode,
    SelectionPayload,
    WritebackPlan,
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


_SELECTION_COMMANDS = {
    Command.RUN,
    Command.QUICK,
    Command.WB,
    Command.QPCR,
    Command.CCK8,
    Command.TRANSFORM_ONLY,
    Command.STANDARD_CURVE,
    Command.ELISA,
}
_DIALOG_COMMANDS = {
    Command.RUN,
    Command.WB,
    Command.QPCR,
    Command.CCK8,
    Command.TRANSFORM_ONLY,
}
_PRESET_COMMANDS = {
    Command.WB: ExperimentPreset.WB,
    Command.QPCR: ExperimentPreset.QPCR,
    Command.CCK8: ExperimentPreset.CCK8,
}
_SETTING_VALUES: dict[Command, tuple[str, Enum]] = {
    Command.BASE_THEME_CLASSIC: ("base_theme", BaseTheme.CLASSIC),
    Command.BASE_THEME_BW: ("base_theme", BaseTheme.BW),
    Command.BASE_THEME_MINIMAL: ("base_theme", BaseTheme.MINIMAL),
    Command.BASE_THEME_DARK: ("base_theme", BaseTheme.DARK),
    Command.THEME_NONE: ("journal_preset", JournalPreset.NONE),
    Command.THEME_NATURE: ("journal_preset", JournalPreset.NATURE),
    Command.THEME_SCIENCE: ("journal_preset", JournalPreset.SCIENCE),
    Command.THEME_CELL: ("journal_preset", JournalPreset.CELL),
    Command.THEME_LANCET: ("journal_preset", JournalPreset.LANCET),
    Command.THEME_NEJM: ("journal_preset", JournalPreset.NEJM),
    Command.THEME_JAMA: ("journal_preset", JournalPreset.JAMA),
    Command.THEME_BMJ: ("journal_preset", JournalPreset.BMJ),
    Command.JOURNAL_PALETTE_DEFAULT: ("journal_palette", JournalPalette.DEFAULT),
    Command.JOURNAL_PALETTE_NATURE: ("journal_palette", JournalPalette.NATURE),
    Command.JOURNAL_PALETTE_SCIENCE: ("journal_palette", JournalPalette.SCIENCE),
    Command.JOURNAL_PALETTE_CELL: ("journal_palette", JournalPalette.CELL),
    Command.JOURNAL_PALETTE_LANCET: ("journal_palette", JournalPalette.LANCET),
    Command.JOURNAL_PALETTE_NEJM: ("journal_palette", JournalPalette.NEJM),
    Command.JOURNAL_PALETTE_JAMA: ("journal_palette", JournalPalette.JAMA),
    Command.JOURNAL_PALETTE_BMJ: ("journal_palette", JournalPalette.BMJ),
    Command.PALETTE_DEFAULT: ("palette_preset", PalettePreset.DEFAULT),
    Command.PALETTE_COLORBLIND: ("palette_preset", PalettePreset.COLORBLIND),
    Command.PALETTE_VIBRANT: ("palette_preset", PalettePreset.VIBRANT),
    Command.PALETTE_PASTEL: ("palette_preset", PalettePreset.PASTEL),
    Command.PALETTE_DEEP: ("palette_preset", PalettePreset.DEEP),
    Command.PALETTE_MUTED: ("palette_preset", PalettePreset.MUTED),
}


def _success(command: Command, plan: WritebackPlan, **extra: Any) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "ok": True,
        "status": "ok",
        "command": command.value,
        "writebackPlan": plan.to_dict(),
        **extra,
    }


def _execute_setting(command: Command, config: PrismConfig) -> dict[str, Any]:
    if command is Command.RESET_SETTINGS:
        from ..config import DEFAULT_SETTINGS_PATH

        DEFAULT_SETTINGS_PATH.unlink(missing_ok=True)
        return _success(command, WritebackPlan(status_message="XSTARS: Settings reset to defaults"))
    if command is Command.ABOUT:
        return _success(command, WritebackPlan(status_message="XSTARS v1.0.0 — offline WPS support"))
    field_name, value = _SETTING_VALUES[command]
    setattr(config, field_name, value)
    if field_name in {"palette_preset", "journal_palette"}:
        config.palette = get_palette(config.palette_preset, config.journal_palette)
    config.save()
    return _success(command, WritebackPlan(status_message=f"XSTARS: {command.value} saved"))


def _configure_preset(command: Command, selection: SelectionPayload, config: PrismConfig) -> None:
    preset = _PRESET_COMMANDS.get(command)
    if preset is None:
        return
    config.experiment_preset = preset
    frame = DataHandler.from_selection_payload(selection)
    groups = DataHandler.group_names(frame)
    if not config.preset_control_group and groups:
        config.preset_control_group = guess_control(groups)
    if preset is ExperimentPreset.CCK8 and not config.preset_blank_group:
        config.preset_blank_group = guess_blank(groups)


def _finalize_analysis(
    command: Command,
    result: AnalysisResult | StandardCurveResult,
    config: PrismConfig,
    artifact_path: Path,
    *,
    renderer: str = "plot_engine",
) -> dict[str, Any]:
    if not artifact_path.is_file():
        result.figure.savefig(artifact_path, format="png", dpi=config.export_dpi)
    export_module = import_module("xstars.application.export")
    picture_id = export_module.new_picture_id()
    frame = result.transformed_data if isinstance(result, AnalysisResult) else result.standard_data
    # Analysis/writeback remains usable; later export reports a stable missing-payload error.
    with suppress(Exception):
        export_module.persist_render_payload(
            picture_id,
            frame,
            config,
            result.figure,
            renderer=renderer,
        )
    for image in result.writeback_plan.images:
        image.picture_id = picture_id
        image.artifact = Artifact(
            path=str(artifact_path),
            format=ArtifactFormat.PNG,
            dpi=config.export_dpi,
        )
        image.source_key = None
    from matplotlib import pyplot as plt

    plt.close(result.figure)
    return _success(command, result.writeback_plan)


def execute_request(
    request: Mapping[str, Any],
    job_directory: Path,
    *,
    dialog_config: Callable[[SelectionPayload, PrismConfig], PrismConfig] = _dialog_config,
) -> dict[str, Any]:
    """Execute one validated WPS command and return JSON-safe output."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("the WPS worker must execute on its main thread")
    if request.get("version") != SCHEMA_VERSION:
        raise ValueError("unsupported worker request version")
    try:
        command = Command(request.get("command"))
    except (TypeError, ValueError) as exc:
        raise ContractError(ErrorCode.INVALID_COMMAND, "command is not whitelisted") from exc
    cancel_path = _cancel_path(request, job_directory)
    _check_cancelled(cancel_path)
    config = _apply_config_overrides(PrismConfig.load(), request.get("config"))

    if command in _SETTING_VALUES or command in {Command.RESET_SETTINGS, Command.ABOUT}:
        return _execute_setting(command, config)
    if command is Command.EXPORT:
        export_request = request.get("export")
        if not isinstance(export_request, Mapping):
            raise ContractError(ErrorCode.INVALID_REQUEST, "export must be an object")
        export_module = import_module("xstars.application.export")
        if export_request.get("clipboard") is True:
            exported = export_module.export_clipboard_image(
                export_request.get("format"),
                export_request.get("dpi"),
            )
        else:
            picture_id = export_request.get("pictureId")
            if not isinstance(picture_id, str):
                raise ContractError(ErrorCode.PAYLOAD_MISSING, "selected Shape has no XSTARS pictureId")
            exported = export_module.render_payload_export(
                picture_id,
                export_request.get("format"),
                export_request.get("dpi"),
            )
        return _success(
            command,
            WritebackPlan(status_message=f"XSTARS: Exported to {exported['path']}"),
            export=exported,
        )

    if command not in _SELECTION_COMMANDS:
        raise ContractError(ErrorCode.INVALID_COMMAND, f"command is not implemented: {command.value}")
    selection_data = request.get("selection")
    if not isinstance(selection_data, Mapping):
        raise ContractError(ErrorCode.INVALID_SELECTION, "selection must be an object")
    selection = SelectionPayload.from_dict(selection_data)
    _configure_preset(command, selection, config)
    if command in _DIALOG_COMMANDS:
        config = dialog_config(selection, config)
    _check_cancelled(cancel_path)

    artifact_path = (job_directory / "chart.png").resolve()
    config.export_path = str(artifact_path)
    config.export_format = "png"
    output_start = _output_start_cell(selection)
    if command is Command.TRANSFORM_ONLY:
        transformed = transform_selection(selection, config, output_start_cell=output_start)
        return _success(command, transformed.writeback_plan)
    if command is Command.STANDARD_CURVE:
        config.export_path = ""
        result = standard_curve_selection(selection, config, output_start_cell=output_start)
        return _finalize_analysis(command, result, config, artifact_path, renderer="standard_curve")
    if command is Command.ELISA:
        sample_data = request.get("sampleSelection")
        if not isinstance(sample_data, Mapping):
            raise ContractError(ErrorCode.INVALID_SELECTION, "sampleSelection must be an object")
        sample = SelectionPayload.from_dict(sample_data)
        result = elisa_selections(
            selection,
            sample,
            config,
            output_start_cell=output_start,
        )
    else:
        result = analyze_selection(
            selection,
            config,
            output_start_cell=output_start,
            include_processed_data=command in _PRESET_COMMANDS,
        )
    _check_cancelled(cancel_path)
    return _finalize_analysis(command, result, config, artifact_path)


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
    except (KeyError, TypeError, ValueError) as exc:
        artifact_path.unlink(missing_ok=True)
        output = _error_result(ErrorCode.ANALYSIS_FAILED, str(exc))
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
