"""Entry points for Excel-Prism, called via xlwings RunPython."""

# Excel/xlwings and tkinter expose host-specific dynamic APIs that lack complete stubs.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import pandas as pd
import xlwings as xw

from . import artifacts
from .config import ExperimentPreset, PrismConfig
from .data_handler import DataHandler
from .plot_engine import PlotEngine, export_figure
from .presets import get_preset
from .presets.cck8 import CCK8FitInfo, CCK8Options, CCK8Preset
from .presets.elisa import ELISAOptions
from .presets.qpcr import QPCROptions
from .presets.wb import WBOptions
from .stats_engine import StatsEngine
from .ui_dialog import SettingsDialog, TransformOnlyDialog

if TYPE_CHECKING:
    from .tools.standard_curve import CurveFitResult

try:
    importlib.import_module("ttkbootstrap")
    HAS_TTKB = True
except ImportError:
    HAS_TTKB = False

_ARTIFACT_LOGGER = logging.getLogger(f"{__name__}.artifacts")
_LAST_ARTIFACT_DIAGNOSTIC: str | None = None
_LAST_ARTIFACT_DISCOVERY_ERRORS: list[artifacts.ArtifactError] = []
_UNSAVED_WORKBOOK_ARTIFACT_MESSAGE = (
    "This workbook has not been saved. Save the workbook, then regenerate "
    "the chart to enable macOS export."
)


def get_last_artifact_diagnostic() -> str | None:
    """Return the last non-blocking artifact registration failure, if any."""
    return _LAST_ARTIFACT_DIAGNOSTIC


def _workbook_artifact_identifier(book: xw.Book) -> str:
    """Return a Save-As-sensitive path identity or fail closed if unsaved."""
    fullname = getattr(book, "fullname", None)
    if isinstance(fullname, str) and fullname.strip():
        return f"path:{os.path.abspath(os.path.expanduser(fullname.strip()))}"
    raise artifacts.ArtifactIdentityError(_UNSAVED_WORKBOOK_ARTIFACT_MESSAGE)


def _register_artifact_best_effort(
    book: xw.Book,
    sheet,
    picture,
    requested_name: str,
    dataframe: pd.DataFrame,
    config: PrismConfig,
    stats_result=None,
    *,
    renderer_kind: artifacts.RendererKind = artifacts.RendererKind.PLOT_ENGINE,
    renderer_params: dict | None = None,
) -> bool:
    """Register one inserted picture without ever blocking the generation flow."""
    global _LAST_ARTIFACT_DIAGNOSTIC
    try:
        sheet_name = getattr(sheet, "name", None)
        if not isinstance(sheet_name, str) or not sheet_name.strip():
            raise ValueError(
                "Worksheet identity is unavailable for artifact registration"
            )
        actual_name = getattr(picture, "name", None)
        if not isinstance(actual_name, str) or not actual_name.strip():
            actual_name = requested_name
        identity = artifacts.ArtifactIdentity(
            workbook=_workbook_artifact_identifier(book),
            sheet=sheet_name,
            picture=actual_name,
        )
        payload = artifacts.build_payload(
            identity,
            dataframe,
            config,
            stats_result,
            renderer_kind=renderer_kind,
            renderer_params=renderer_params,
        )
        artifacts.save_artifact(payload)
    except Exception as exc:
        _LAST_ARTIFACT_DIAGNOSTIC = f"{type(exc).__name__}: {exc}"
        _ARTIFACT_LOGGER.warning(
            "Chart inserted, but rebuild artifact registration failed: %s",
            _LAST_ARTIFACT_DIAGNOSTIC,
        )
        return False
    _LAST_ARTIFACT_DIAGNOSTIC = None
    return True


def _write_transformed_data(
    sheet, row: int, col: int, df: pd.DataFrame, title: str = "Processed Data"
) -> int:
    """Write transformed DataFrame to Excel. Returns next available row."""
    dest = sheet.range((row, col))
    dest.value = [[title]]
    row += 1
    dest = sheet.range((row, col))
    dest.value = [df.columns.tolist()] + df.values.tolist()
    row += len(df) + 2
    return row


def _next_plot_name(
    sheet, base: str = "XSTARS_Plot", *, book: xw.Book | None = None
) -> str:
    """Return a name unused by pictures or a valid persisted artifact."""
    existing = {p.name for p in sheet.pictures}
    workbook_identifier: str | None = None
    raw_sheet_name = getattr(sheet, "name", None)
    sheet_name = (
        raw_sheet_name.strip()
        if isinstance(raw_sheet_name, str) and raw_sheet_name.strip()
        else None
    )
    if sys.platform == "darwin" and book is not None and sheet_name is not None:
        # Unsaved workbooks do not register artifacts, so only Excel names can
        # reserve a candidate. Non-Darwin name allocation remains unchanged.
        with suppress(artifacts.ArtifactIdentityError):
            workbook_identifier = _workbook_artifact_identifier(book)
    i = 1
    while True:
        name = f"{base}_{i}"
        if name not in existing:
            if workbook_identifier is None or sheet_name is None:
                return name
            identity = artifacts.ArtifactIdentity(
                workbook=workbook_identifier, sheet=sheet_name, picture=name
            )
            if not artifacts.has_artifact(identity):
                return name
        i += 1


def _guess_control(groups: list[str]) -> str:
    """Pick the most likely control group from column names."""
    for g in groups:
        if g.lower() in ("control", "ctrl", "con", "ctl", "nc", "vehicle"):
            return g
    # Fall back to second column if first looks like blank, else first
    if len(groups) >= 2 and groups[0].lower() in ("blank", "blk", "bg", "background"):
        return groups[1]
    return groups[0]


def _guess_blank(groups: list[str]) -> str:
    """Pick the most likely blank group, or empty string if none found."""
    for g in groups:
        if g.lower() in ("blank", "blk", "bg", "background"):
            return g
    return ""


def _build_preset_options(config: PrismConfig):
    """Map flat PrismConfig fields to the appropriate preset Options dataclass."""
    preset_type = config.experiment_preset
    if preset_type == ExperimentPreset.WB:
        return WBOptions(
            control_group=config.preset_control_group,
            has_reference=config.preset_has_reference,
            reference_protein=config.preset_reference_protein,
        )
    elif preset_type == ExperimentPreset.QPCR:
        return QPCROptions(
            control_group=config.preset_control_group,
            input_format=config.preset_input_format,
            reference_gene=config.preset_reference_gene,
        )
    elif preset_type == ExperimentPreset.CCK8:
        concentrations = []
        if config.preset_concentrations:
            with suppress(ValueError):
                concentrations = [
                    float(x.strip())
                    for x in config.preset_concentrations.split(",")
                    if x.strip()
                ]
        return CCK8Options(
            control_group=config.preset_control_group,
            blank_group=config.preset_blank_group,
            fit_ic50=config.preset_fit_ic50,
            concentrations=concentrations,
            fit_method=config.preset_fit_method.value,
        )
    elif preset_type == ExperimentPreset.ELISA:
        return ELISAOptions(
            control_group=config.preset_control_group,
            fit_result=cast("CurveFitResult | None", config.elisa_fit_result),
        )
    return None


def _apply_preset(df_wide, config: PrismConfig):
    """Apply experiment preset transform if configured. Returns transformed df."""
    preset = get_preset(config.experiment_preset)
    if preset is None:
        return df_wide
    options = _build_preset_options(config)
    if options is None:
        raise ValueError(f"Unsupported experiment preset: {config.experiment_preset}")
    df_wide = preset.transform(df_wide, options)
    if config.y_label == "Value":
        config.y_label = preset.default_y_label
    return df_wide


def _friendly_message(message: str) -> str:
    """Translate raw Python/library error messages into user-friendly hints."""
    # Mapping: (substring in raw message) → friendly message
    _PATTERNS: list[tuple[str, str | None]] = [
        (
            "arg must be a list, tuple, 1-d array, or Series",
            "Please select a valid data range.\n\n"
            "Requirements:\n"
            "• First row should be group names (header)\n"
            "• Rows below should contain numeric data\n"
            "• Select at least 2 columns (2 groups)",
        ),
        (
            "DataFrame constructor not properly called",
            "Please select a valid data range.\n\n"
            "The current selection could not be read as data.\n"
            "Make sure:\n"
            "• First row contains group names (header)\n"
            "• Rows below contain numeric data",
        ),
        (
            "No objects to concatenate",
            "No valid data detected in the selection.\n"
            "Please check that the selected range contains numeric values.",
        ),
        (
            "Need at least",
            None,  # keep original — already user-friendly from our own validate()
        ),
        (
            "Control group mean is zero",
            "Control group mean is zero — cannot normalize.\n"
            "Please check your control group data.",
        ),
    ]
    for pattern, friendly in _PATTERNS:
        if pattern in message:
            return friendly if friendly is not None else message
    return message


def _show_error(book: xw.Book, message: str, *, is_unexpected: bool = False) -> None:
    """Display a user-friendly error message via a Tkinter popup and Excel status bar."""
    friendly = _friendly_message(message)
    if is_unexpected:
        # For unexpected errors, show a generic hint + original for debugging
        friendly = (
            "Something went wrong. Please check your data selection.\n\n"
            "Requirements:\n"
            "• First row should be group names (header)\n"
            "• Rows below should contain numeric data\n"
            "• Select at least 2 columns (2 groups)\n\n"
            f"Details: {message[:300]}"
        )
    book.app.status_bar = f"Excel-Prism Error: {message}"
    root = None
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        # Bring the message box to the front when the window manager supports it.
        try:
            root.attributes("-topmost", True)
        except Exception as exc:
            _ARTIFACT_LOGGER.debug("Tk topmost hint unavailable: %s", exc)
        messagebox.showerror("Excel-Prism", friendly, parent=root)
    except Exception:
        # Last resort: try VBA MsgBox
        try:
            book.app.macro("MsgBox")(
                f"Excel-Prism Error\n\n{friendly}",
                16,  # vbCritical
                "Excel-Prism",
            )
        except Exception as exc:
            _ARTIFACT_LOGGER.debug("VBA MsgBox fallback failed: %s", exc)
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception as exc:
                _ARTIFACT_LOGGER.debug("Tk error window cleanup failed: %s", exc)


_USER_ERRORS = (ValueError, TypeError, KeyError)


def run() -> None:
    """Full flow: read selection → settings dialog → stats → plot → insert."""
    book = xw.Book.caller()
    try:
        _run_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_quick() -> None:
    """Quick flow: read selection → default settings → stats → plot → insert."""
    book = xw.Book.caller()
    try:
        _run_quick_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_wb() -> None:
    """Western Blot preset: read selection → WB normalization → dialog → stats → plot."""
    book = xw.Book.caller()
    try:
        _run_preset_impl(book, ExperimentPreset.WB)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_qpcr() -> None:
    """qPCR preset: read selection → ΔΔCt calculation → dialog → stats → plot."""
    book = xw.Book.caller()
    try:
        _run_preset_impl(book, ExperimentPreset.QPCR)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_cck8() -> None:
    """CCK-8 preset: read selection → viability % + IC50 → dialog → stats → plot."""
    book = xw.Book.caller()
    try:
        _run_preset_impl(book, ExperimentPreset.CCK8)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_elisa() -> None:
    """ELISA preset: standard curve → back-calculate → stats → plot."""
    book = xw.Book.caller()
    try:
        _run_elisa_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def _run_elisa_impl(book: xw.Book) -> None:
    """ELISA flow: read std curve selection → fit → dialog → select samples → back-calc → stats → plot."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .presets.elisa_dialog import ELISADialog
    from .tools.standard_curve import (
        back_calculate,
        wide_to_conc_od,
    )

    sheet = book.selection.sheet
    sel = book.selection
    handler = DataHandler()

    # 1. Read current selection as standard curve data (columns = concentrations)
    raw = sel.options(pd.DataFrame, header=1, index=False).value
    raw.columns = [str(c).strip() for c in raw.columns]
    handler._selection_ref = sel

    if raw.shape[1] < 2:
        raise ValueError(
            "Please select at least 2 columns of standard data.\n"
            "Column headers should be concentration values (e.g. 0, 15.6, 31.2, ...)."
        )

    df_std = raw.copy()
    for col in df_std.columns:
        df_std[col] = pd.to_numeric(df_std[col], errors="coerce")
    df_std = df_std.dropna(how="all").reset_index(drop=True)

    conc, od = wide_to_conc_od(df_std)
    if len(conc) < 2:
        raise ValueError(
            "Need at least 2 valid data points for standard curve fitting."
        )

    # 2. Show ELISA dialog (fit + chart/theme/output settings in one window)
    dialog = ELISADialog(conc, od)
    dlg_result = dialog.show()
    if dlg_result is None or dlg_result.fit_result is None:
        return

    fit = dlg_result.fit_result
    config = dlg_result.config or PrismConfig.load()
    config.experiment_preset = ExperimentPreset.ELISA
    config.elisa_fit_result = fit

    # 3. Select sample data via InputBox
    sample_df = _select_sample_data(book, sheet)
    if sample_df is None:
        return

    # 4. Back-calculate: sample OD → concentration
    conc_df = sample_df.copy()
    for col in conc_df.columns:
        od_vals = np.asarray(pd.to_numeric(conc_df[col], errors="coerce"), dtype=float)
        conc_df[col] = back_calculate(fit, od_vals)

    # 5. Stats + plot
    handler.validate(conc_df)

    engine = StatsEngine(config)
    stats_result = engine.analyze(conc_df)

    plotter = PlotEngine(config)
    fig = plotter.plot(conc_df, stats_result)

    if config.export_path:
        export_figure(fig, config.export_path, config.export_dpi)

    # 6. Write output tables below selection
    out_row = sel.row + sel.rows.count + 2
    out_col = sel.column

    # Standard curve parameters
    sheet.range((out_row, out_col)).value = "Standard Curve Results"
    out_row += 1
    param_data = [
        ["Method", fit.method],
        ["Equation", fit.equation_str],
        ["R²", fit.r_squared if fit.r_squared is not None else "N/A"],
    ]
    for k, v in fit.params.items():
        param_data.append([k, v])
    sheet.range((out_row, out_col)).value = param_data
    out_row += len(param_data) + 1

    # Stats table
    if config.output_stats:
        stats_df = stats_result.to_dataframe()
        dest = sheet.range((out_row, out_col))
        dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
        out_row += len(stats_df) + 2

    # Concentration data
    if config.output_data:
        out_row = _write_transformed_data(
            sheet, out_row, out_col, conc_df, "Back-Calculated Concentrations"
        )

    # 7. Insert analysis chart below tables
    insert_left = sheet.range((out_row, out_col)).left
    insert_top = sheet.range((out_row, out_col)).top
    pic_name = _next_plot_name(sheet, book=book)
    pic = sheet.pictures.add(
        fig,
        name=pic_name,
        left=insert_left,
        top=insert_top,
    )
    _register_artifact_best_effort(
        book, sheet, pic, pic_name, conc_df, config, stats_result
    )
    current_top = insert_top + pic.height + 15

    # 8. Optional standard curve chart below analysis chart
    if dlg_result.show_fit_curve:
        renderer_params = artifacts.standard_curve_renderer_params(conc, od, fit)
        fig_std = artifacts.build_standard_curve_figure(
            conc, od, renderer_params["fit"], config
        )
        std_name = _next_plot_name(sheet, "XSTARS_ELISA_StdCurve", book=book)
        std_pic = sheet.pictures.add(
            fig_std,
            name=std_name,
            left=insert_left,
            top=current_top,
        )
        _register_artifact_best_effort(
            book,
            sheet,
            std_pic,
            std_name,
            df_std,
            config,
            renderer_kind=artifacts.RendererKind.STANDARD_CURVE,
            renderer_params=renderer_params,
        )
        plt.close(fig_std)

    r2_str = f", R²={fit.r_squared:.4f}" if fit.r_squared else ""
    book.app.status_bar = (
        f"XSTARS: ELISA ({fit.method}{r2_str}) — {stats_result.decision_path}"
    )


def _has_label_column(df: pd.DataFrame) -> bool:
    """Check if the first column of a raw DataFrame contains non-numeric labels."""
    import pandas as pd

    first_col = df.iloc[:, 0]
    numeric = pd.to_numeric(first_col, errors="coerce")
    # If more than half the values failed numeric conversion, it's a label column
    return numeric.isna().sum() > len(first_col) // 2


def _run_preset_impl(book: xw.Book, preset_type: ExperimentPreset) -> None:
    """Like _run_impl but pre-selects the experiment preset."""
    sheet = book.selection.sheet
    handler = DataHandler()

    # Auto-detect label column for all presets (strip non-numeric first column)
    wb_labels, df_wide = _read_selection_auto(handler, book)

    handler.validate(df_wide)

    groups = handler.group_names(df_wide)
    sizes = handler.group_sizes(df_wide)

    # Pre-configure the preset
    base_config = PrismConfig.load()
    base_config.experiment_preset = preset_type
    if not base_config.preset_control_group and groups:
        base_config.preset_control_group = _guess_control(groups)
    if preset_type == ExperimentPreset.CCK8 and not base_config.preset_blank_group:
        base_config.preset_blank_group = _guess_blank(groups)
    # If labels detected, default to has_reference=True for WB and qPCR
    if wb_labels is not None and preset_type in (
        ExperimentPreset.WB,
        ExperimentPreset.QPCR,
    ):
        base_config.preset_has_reference = True

    dialog = SettingsDialog(groups, sizes, base_config=base_config)
    config = dialog.show()
    if config is None:
        return

    # WB labeled reference mode: multi-target branch
    if (
        preset_type == ExperimentPreset.WB
        and config.preset_has_reference
        and wb_labels is not None
    ):
        _run_wb_labeled(book, sheet, handler, wb_labels, df_wide, config)
        return

    # qPCR labeled reference mode: multi-gene branch
    if (
        preset_type == ExperimentPreset.QPCR
        and config.preset_has_reference
        and wb_labels is not None
    ):
        _run_qpcr_labeled(book, sheet, handler, wb_labels, df_wide, config)
        return

    # Apply preset transform
    preset = get_preset(config.experiment_preset)
    df_wide = _apply_preset(df_wide, config)
    handler.validate(df_wide)

    # Populate IC50 fit info for dose-response chart
    if (
        isinstance(preset, CCK8Preset)
        and preset.last_result
        and preset.last_result.ic50 is not None
        and preset.last_result.fit_params is not None
    ):
        opts = _build_preset_options(config)
        if not isinstance(opts, CCK8Options):
            raise ValueError("CCK-8 options are unavailable")
        dose_cols = [c for c in df_wide.columns if c != opts.control_group]
        if opts.concentrations and len(opts.concentrations) == len(dose_cols):
            config.ic50_fit_info = CCK8FitInfo(
                concentrations=opts.concentrations,
                fit_params=preset.last_result.fit_params,
                dose_col_names=dose_cols,
            )

    # Stats
    engine = StatsEngine(config)
    stats_result = engine.analyze(df_wide)

    # Plot
    plotter = PlotEngine(config)
    fig = plotter.plot(df_wide, stats_result)

    if config.export_path:
        export_figure(fig, config.export_path, config.export_dpi)

    # Stats summary table below data selection
    sel = book.selection
    start_col = sel.column
    next_row = sel.row + sel.rows.count + 2

    if config.output_stats:
        stats_df = stats_result.to_dataframe()
        dest = sheet.range((next_row, start_col))
        dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
        next_row += len(stats_df) + 2

        # IC50 results for CCK-8
        if (
            isinstance(preset, CCK8Preset)
            and preset.last_result
            and preset.last_result.ic50 is not None
        ):
            ic50_dest = sheet.range((next_row, start_col))
            res = preset.last_result
            ic50_data = [
                ["IC50", res.ic50],
                ["R²", res.r_squared],
            ]
            if res.ic50_95ci:
                ic50_data.append(
                    ["IC50 95% CI", f"{res.ic50_95ci[0]:.4g} – {res.ic50_95ci[1]:.4g}"]
                )
            ic50_dest.value = ic50_data
            next_row += len(ic50_data) + 2

    # Write processed data
    if config.output_data:
        next_row = _write_transformed_data(
            sheet, next_row, start_col, df_wide, "Processed Data"
        )

    # Insert chart below stats/data tables
    pic_name = _next_plot_name(sheet, book=book)
    pic = sheet.pictures.add(
        fig,
        name=pic_name,
        left=sheet.range((next_row, start_col)).left,
        top=sheet.range((next_row, start_col)).top,
    )
    _register_artifact_best_effort(
        book, sheet, pic, pic_name, df_wide, config, stats_result
    )

    book.app.status_bar = f"XSTARS: {stats_result.decision_path}"


def _run_wb_labeled(
    book: xw.Book,
    sheet,
    handler: DataHandler,
    labels: pd.Series,
    df_numeric: pd.DataFrame,
    config: PrismConfig,
) -> None:
    """WB labeled reference mode: produce one figure per target protein."""
    from .presets.wb import WBPreset

    preset = WBPreset()
    options = _build_preset_options(config)
    target_dfs = preset.transform_labeled(labels, df_numeric, options)

    if config.y_label == "Value":
        config.y_label = preset.default_y_label

    sel = book.selection
    insert_cell = handler.get_insertion_cell(sheet)
    base_left = sheet.range(insert_cell).left
    current_top = sheet.range(insert_cell).top

    # Track where to place stats tables (below data selection)
    stats_start_row = sel.row + sel.rows.count + 2
    stats_col = sel.column

    for protein_name, fold_df in target_dfs:
        handler.validate(fold_df)

        engine = StatsEngine(config)
        stats_result = engine.analyze(fold_df)

        # Set title to protein name
        plot_config = PrismConfig(
            **{
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()
            }
        )
        plot_config.title = protein_name

        plotter = PlotEngine(plot_config)
        fig = plotter.plot(fold_df, stats_result)

        pic_name = f"XSTARS_Plot_{protein_name}"
        pic = sheet.pictures.add(
            fig,
            name=pic_name,
            update=True,
            left=base_left,
            top=current_top,
        )
        _register_artifact_best_effort(
            book, sheet, pic, pic_name, fold_df, plot_config, stats_result
        )

        if config.export_path:
            from pathlib import Path

            p = Path(config.export_path)
            export_path = str(p.with_stem(f"{p.stem}_{protein_name}"))
            export_figure(fig, export_path, config.export_dpi)

        # Move top down for the next figure using actual picture height
        current_top += pic.height + 15

        # Stats table
        if config.output_stats:
            stats_df = stats_result.to_dataframe()
            dest = sheet.range((stats_start_row, stats_col))
            dest.value = [[protein_name]]
            stats_start_row += 1
            dest = sheet.range((stats_start_row, stats_col))
            dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
            stats_start_row += len(stats_df) + 2

        # Write processed data
        if config.output_data:
            stats_start_row = _write_transformed_data(
                sheet,
                stats_start_row,
                stats_col,
                fold_df,
                f"Processed Data — {protein_name}",
            )

    book.app.status_bar = (
        f"XSTARS: WB labeled mode — {len(target_dfs)} target(s) analyzed"
    )


def _run_qpcr_labeled(
    book: xw.Book,
    sheet,
    handler: DataHandler,
    labels: pd.Series,
    df_numeric: pd.DataFrame,
    config: PrismConfig,
) -> None:
    """qPCR labeled reference mode: produce one figure per target gene."""
    from .presets.qpcr import QPCRPreset

    preset = QPCRPreset()
    options = _build_preset_options(config)
    target_dfs = preset.transform_labeled(labels, df_numeric, options)

    if config.y_label == "Value":
        config.y_label = preset.default_y_label

    sel = book.selection
    insert_cell = handler.get_insertion_cell(sheet)
    base_left = sheet.range(insert_cell).left
    current_top = sheet.range(insert_cell).top

    # Track where to place stats tables (below data selection)
    stats_start_row = sel.row + sel.rows.count + 2
    stats_col = sel.column

    for gene_name, fold_df in target_dfs:
        handler.validate(fold_df)

        engine = StatsEngine(config)
        stats_result = engine.analyze(fold_df)

        # Set title to gene name
        plot_config = PrismConfig(
            **{
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()
            }
        )
        plot_config.title = gene_name

        plotter = PlotEngine(plot_config)
        fig = plotter.plot(fold_df, stats_result)

        pic_name = f"XSTARS_Plot_{gene_name}"
        pic = sheet.pictures.add(
            fig,
            name=pic_name,
            update=True,
            left=base_left,
            top=current_top,
        )
        _register_artifact_best_effort(
            book, sheet, pic, pic_name, fold_df, plot_config, stats_result
        )

        if config.export_path:
            from pathlib import Path

            p = Path(config.export_path)
            export_path = str(p.with_stem(f"{p.stem}_{gene_name}"))
            export_figure(fig, export_path, config.export_dpi)

        # Move top down for the next figure
        current_top += pic.height + 15

        # Stats table
        if config.output_stats:
            stats_df = stats_result.to_dataframe()
            dest = sheet.range((stats_start_row, stats_col))
            dest.value = [[gene_name]]
            stats_start_row += 1
            dest = sheet.range((stats_start_row, stats_col))
            dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
            stats_start_row += len(stats_df) + 2

        # Write processed data
        if config.output_data:
            stats_start_row = _write_transformed_data(
                sheet,
                stats_start_row,
                stats_col,
                fold_df,
                f"Processed Data — {gene_name}",
            )

    book.app.status_bar = (
        f"XSTARS: qPCR labeled mode — {len(target_dfs)} gene(s) analyzed"
    )


def _read_selection_auto(handler: DataHandler, book: xw.Book):
    """Read Excel selection, auto-detecting a label column in the first column.

    Returns (wb_labels, df_wide) where wb_labels is a Series of string labels
    if a label column was detected, or None otherwise.
    """
    import pandas as pd

    sel = book.selection
    raw = sel.options(pd.DataFrame, header=1, index=False).value
    raw.columns = [str(c).strip() for c in raw.columns]
    handler._selection_ref = sel

    if _has_label_column(raw):
        label_col = raw.columns[0]
        wb_labels = raw[label_col].astype(str).str.strip()
        df_wide = raw.drop(columns=[label_col])
        for col in df_wide.columns:
            df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")
        valid = ~df_wide.isna().all(axis=1)
        wb_labels = wb_labels.loc[valid].reset_index(drop=True)
        df_wide = df_wide.loc[valid].reset_index(drop=True)
        return wb_labels, df_wide

    return None, handler.clean(raw)


def _run_impl(book: xw.Book) -> None:
    sheet = book.selection.sheet

    # 1. Read data (auto-detect label column)
    handler = DataHandler()
    wb_labels, df_wide = _read_selection_auto(handler, book)
    handler.validate(df_wide)

    groups = handler.group_names(df_wide)
    sizes = handler.group_sizes(df_wide)

    # 2. Settings dialog
    base_config = PrismConfig.load()
    if wb_labels is not None:
        base_config.experiment_preset = ExperimentPreset.WB
        base_config.preset_has_reference = True
        if not base_config.preset_control_group and groups:
            base_config.preset_control_group = _guess_control(groups)
    dialog = SettingsDialog(groups, sizes, base_config=base_config)
    config = dialog.show()
    if config is None:
        return  # user cancelled

    # WB labeled reference mode
    if (
        config.experiment_preset == ExperimentPreset.WB
        and config.preset_has_reference
        and wb_labels is not None
    ):
        _run_wb_labeled(book, sheet, handler, wb_labels, df_wide, config)
        return

    # 2b. Apply experiment preset transform
    df_wide = _apply_preset(df_wide, config)
    handler.validate(df_wide)

    # 3. Statistics
    engine = StatsEngine(config)
    stats_result = engine.analyze(df_wide)

    # 4. Plot
    plotter = PlotEngine(config)
    fig = plotter.plot(df_wide, stats_result)

    # 5. Export figure to file if requested
    if config.export_path:
        export_figure(fig, config.export_path, config.export_dpi)

    # 6. Write stats summary table below the data selection
    sel = book.selection
    start_col = sel.column
    next_row = sel.row + sel.rows.count + 2

    if config.output_stats:
        stats_df = stats_result.to_dataframe()
        dest = sheet.range((next_row, start_col))
        dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
        next_row += len(stats_df) + 2

        # Write IC50 results if CCK-8 preset was used
        preset = get_preset(config.experiment_preset)
        if (
            isinstance(preset, CCK8Preset)
            and preset.last_result
            and preset.last_result.ic50 is not None
        ):
            ic50_dest = sheet.range((next_row, start_col))
            res = preset.last_result
            ic50_data = [
                ["IC50", res.ic50],
                ["R²", res.r_squared],
            ]
            if res.ic50_95ci:
                ic50_data.append(
                    ["IC50 95% CI", f"{res.ic50_95ci[0]:.4g} – {res.ic50_95ci[1]:.4g}"]
                )
            ic50_dest.value = ic50_data
            next_row += len(ic50_data) + 2

    # 7. Insert chart below stats table
    pic_name = _next_plot_name(sheet, book=book)
    pic = sheet.pictures.add(
        fig,
        name=pic_name,
        left=sheet.range((next_row, start_col)).left,
        top=sheet.range((next_row, start_col)).top,
    )
    _register_artifact_best_effort(
        book, sheet, pic, pic_name, df_wide, config, stats_result
    )

    # Show decision path in status bar
    book.app.status_bar = f"XSTARS: {stats_result.decision_path}"


def _run_quick_impl(book: xw.Book) -> None:
    sheet = book.selection.sheet

    config = PrismConfig.load()

    handler = DataHandler(config)
    _labels, df_wide = _read_selection_auto(handler, book)
    handler.validate(df_wide)

    # Apply experiment preset transform
    df_wide = _apply_preset(df_wide, config)
    handler.validate(df_wide)

    engine = StatsEngine(config)
    stats_result = engine.analyze(df_wide)

    plotter = PlotEngine(config)
    fig = plotter.plot(df_wide, stats_result)

    sel = book.selection
    start_col = sel.column
    next_row = sel.row + sel.rows.count + 2

    # Write stats summary table below the data selection
    if config.output_stats:
        stats_df = stats_result.to_dataframe()
        dest = sheet.range((next_row, start_col))
        dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
        next_row += len(stats_df) + 2

    # Insert chart below stats table
    pic_name = _next_plot_name(sheet, book=book)
    pic = sheet.pictures.add(
        fig,
        name=pic_name,
        left=sheet.range((next_row, start_col)).left,
        top=sheet.range((next_row, start_col)).top,
    )
    _register_artifact_best_effort(
        book, sheet, pic, pic_name, df_wide, config, stats_result
    )

    book.app.status_bar = f"XSTARS: {stats_result.decision_path}"


def run_export() -> None:
    """Export every valid artifact-backed XSTARS picture on the active sheet."""
    book = xw.Book.caller()
    try:
        _run_export_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def _artifact_identity_for_picture(
    book: xw.Book, sheet, picture
) -> artifacts.ArtifactIdentity:
    """Build the exact persisted identity for one xlwings picture."""
    sheet_name = getattr(sheet, "name", None)
    picture_name = getattr(picture, "name", None)
    if not isinstance(sheet_name, str) or not sheet_name.strip():
        raise ValueError("Worksheet identity is unavailable for artifact export")
    if not isinstance(picture_name, str) or not picture_name.strip():
        raise ValueError("Picture identity is unavailable for artifact export")
    return artifacts.ArtifactIdentity(
        workbook=_workbook_artifact_identifier(book),
        sheet=sheet_name,
        picture=picture_name,
    )


def _get_selected_shapes(book: xw.Book) -> list:
    """Return host export candidates.

    On macOS this means every valid artifact-backed XSTARS picture on the
    active sheet; Excel picture selection is intentionally not consulted.
    """
    global _LAST_ARTIFACT_DISCOVERY_ERRORS
    _LAST_ARTIFACT_DISCOVERY_ERRORS = []
    if sys.platform == "darwin":
        # Excel for Mac exposes appscript rather than COM.  Do not touch
        # book.app.api/ShapeRange: only validated artifact-backed pictures are
        # eligible for the reconstruction export path.
        sheet = book.selection.sheet
        pictures = []
        for picture in sheet.pictures:
            picture_name = getattr(picture, "name", "")
            if not isinstance(picture_name, str) or not picture_name.startswith(
                "XSTARS_"
            ):
                continue
            try:
                identity = _artifact_identity_for_picture(book, sheet, picture)
                artifacts.load_artifact(identity)
            except artifacts.ArtifactError as exc:
                _LAST_ARTIFACT_DISCOVERY_ERRORS.append(exc)
                _ARTIFACT_LOGGER.info(
                    "Skipping macOS picture without a valid rebuild artifact (%s): %s",
                    picture_name,
                    exc,
                )
                continue
            except ValueError as exc:
                identity_error = artifacts.ArtifactIdentityError(str(exc))
                _LAST_ARTIFACT_DISCOVERY_ERRORS.append(identity_error)
                _ARTIFACT_LOGGER.info(
                    "Skipping macOS picture with an invalid identity (%s): %s",
                    picture_name,
                    exc,
                )
                continue
            pictures.append(picture)
        return pictures

    # Preserve the existing Windows ShapeRange-first behavior and fallback.
    try:
        sel = book.app.api.Selection
        _shape_type = sel.ShapeRange.Item(1).Type  # succeeds only for shape selection
        return [sel.ShapeRange.Item(i) for i in range(1, sel.ShapeRange.Count + 1)]
    except Exception as exc:
        _ARTIFACT_LOGGER.debug("ShapeRange selection unavailable: %s", exc)
    # Fallback: find XSTARS charts on the active sheet
    sheet = book.selection.sheet
    pics = [p for p in sheet.pictures if p.name.startswith("XSTARS_Plot")]
    if pics:
        return [p.api for p in pics]
    return []


def _export_shape_highres(shape, save_path: str, dpi: int) -> None:
    """Export a single COM Shape object at the requested DPI.

    Strategy: temporarily scale the shape up so that CopyPicture(xlBitmap)
    captures at higher resolution, then restore original dimensions.
    """
    import time

    from PIL import ImageGrab

    scale = dpi / 96.0
    orig_w = shape.Width
    orig_h = shape.Height
    orig_left = shape.Left
    orig_top = shape.Top
    lock_aspect = shape.LockAspectRatio
    shape.LockAspectRatio = 0  # msoFalse — allow independent scaling

    try:
        shape.Width = orig_w * scale
        shape.Height = orig_h * scale

        # CopyPicture: Appearance=1 (xlScreen), Format=2 (xlBitmap)
        shape.CopyPicture(1, 2)
        time.sleep(0.3)  # let clipboard settle

        img = ImageGrab.grabclipboard()
        if img is None:
            raise RuntimeError("Could not capture image from clipboard.")

        ext = save_path.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            img.save(save_path, "PDF", resolution=dpi)
        elif ext in ("tif", "tiff"):
            img.save(save_path, "TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
        elif ext == "jpg" or ext == "jpeg":
            img.convert("RGB").save(save_path, "JPEG", dpi=(dpi, dpi), quality=95)
        else:
            img.save(save_path, "PNG", dpi=(dpi, dpi))
    finally:
        # Restore original size
        shape.LockAspectRatio = 0
        shape.Width = orig_w
        shape.Height = orig_h
        shape.Left = orig_left
        shape.Top = orig_top
        shape.LockAspectRatio = lock_aspect


def _show_export_dialog() -> tuple[str, int] | None:
    """Show export dialog with format, DPI, and path options. Returns (path, dpi) or None."""
    import tkinter as tk
    from tkinter import filedialog

    if HAS_TTKB:
        import ttkbootstrap as ttkb
    else:
        ttkb = None

    _FORMAT_MAP = {
        "PNG image": ("png", ".png"),
        "TIFF image": ("tiff", ".tif"),
        "JPEG image": ("jpg", ".jpg"),
        "SVG image": ("svg", ".svg"),
        "PDF document": ("pdf", ".pdf"),
    }
    _DPI_OPTIONS = ["150", "300", "600", "1200"]

    result = {}

    root = (
        ttkb.Window(title="Export Image", themename="cosmo", size=(460, 240))
        if ttkb
        else tk.Tk()
    )
    if not ttkb:
        root.title("Export Image")
        root.geometry("460x240")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frame = ttkb.Frame(root, padding=15) if ttkb else tk.Frame(root, padx=15, pady=15)
    frame.pack(fill="both", expand=True)

    Label = ttkb.Label if ttkb else tk.Label
    Entry = ttkb.Entry if ttkb else tk.Entry
    Combo = ttkb.Combobox if ttkb else tk.ttk.Combobox

    # Format
    Label(frame, text="Format:").grid(row=0, column=0, sticky="w", pady=(0, 8))
    fmt_var = tk.StringVar(value="PNG image")
    fmt_combo = Combo(
        frame,
        textvariable=fmt_var,
        values=list(_FORMAT_MAP.keys()),
        state="readonly",
        width=20,
    )
    fmt_combo.grid(row=0, column=1, columnspan=2, sticky="w", pady=(0, 8), padx=(8, 0))

    # DPI
    Label(frame, text="DPI:").grid(row=1, column=0, sticky="w", pady=(0, 8))
    dpi_var = tk.StringVar(value="300")
    dpi_combo = Combo(
        frame, textvariable=dpi_var, values=_DPI_OPTIONS, state="readonly", width=20
    )
    dpi_combo.grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, 8), padx=(8, 0))

    # Output path
    Label(frame, text="Save to:").grid(row=2, column=0, sticky="w", pady=(0, 8))
    path_var = tk.StringVar(value="")
    path_entry = Entry(frame, textvariable=path_var, width=30)
    path_entry.grid(row=2, column=1, sticky="ew", pady=(0, 8), padx=(8, 0))

    def browse():
        _, ext = _FORMAT_MAP.get(fmt_var.get(), ("png", ".png"))
        fmt_label = fmt_var.get()
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(fmt_label, f"*{ext}"), ("All files", "*.*")],
            title="Save As",
            parent=root,
        )
        if path:
            path_var.set(path)

    if ttkb:
        ttkb.Button(
            frame, text="Browse...", command=browse, bootstyle="secondary"
        ).grid(row=2, column=2, sticky="w", pady=(0, 8), padx=(4, 0))
    else:
        tk.Button(frame, text="Browse...", command=browse).grid(
            row=2, column=2, sticky="w", pady=(0, 8), padx=(4, 0)
        )

    # Auto-update extension when format changes
    def on_fmt_change(_event=None):
        cur = path_var.get()
        if cur:
            _, ext = _FORMAT_MAP.get(fmt_var.get(), ("png", ".png"))
            from pathlib import Path

            path_var.set(str(Path(cur).with_suffix(ext)))

    fmt_combo.bind("<<ComboboxSelected>>", on_fmt_change)

    frame.columnconfigure(1, weight=1)

    def on_export():
        path = path_var.get().strip()
        if not path:
            browse()
            path = path_var.get().strip()
        if path:
            result["path"] = path
            result["dpi"] = int(dpi_var.get())
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = ttkb.Frame(frame) if ttkb else tk.Frame(frame)
    btn_frame.grid(row=3, column=0, columnspan=3, sticky="e", pady=(12, 0))

    if ttkb:
        ttkb.Button(
            btn_frame, text="Export", bootstyle="primary", command=on_export, width=10
        ).pack(side="right", padx=(8, 0))
        ttkb.Button(
            btn_frame, text="Cancel", bootstyle="secondary", command=on_cancel, width=10
        ).pack(side="right")
    else:
        tk.Button(btn_frame, text="Export", command=on_export, width=10).pack(
            side="right", padx=(8, 0)
        )
        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(
            side="right"
        )

    # Center on screen
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()

    if "path" in result:
        return result["path"], result["dpi"]
    return None


def _export_artifact_picture(
    book: xw.Book, sheet, picture, save_path: str, dpi: int
) -> None:
    """Atomically rebuild and export one artifact-backed macOS picture."""
    import tempfile
    from pathlib import Path

    import matplotlib.pyplot as plt

    target = Path(save_path)
    temp_path: Path | None = None
    fig = None
    try:
        identity = _artifact_identity_for_picture(book, sheet, picture)
        payload = artifacts.load_artifact(identity)
        try:
            fig = artifacts.rebuild_figure(payload)
        except artifacts.ArtifactError:
            raise
        except Exception as exc:
            raise artifacts.CorruptArtifactError(
                "The chart rebuild information could not be rendered."
            ) from exc

        # Matplotlib infers the format from the suffix.  Export beside the
        # destination and replace only after success so failures cannot expose
        # a partial output file or overwrite an existing valid export.
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}.",
            suffix=target.suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        export_figure(fig, str(temp_path), dpi)
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        if fig is not None:
            plt.close(fig)


def _run_export_impl(book: xw.Book) -> None:
    """Export all valid active-sheet artifacts on macOS, or selected Windows shapes."""
    global _LAST_ARTIFACT_DISCOVERY_ERRORS
    _LAST_ARTIFACT_DISCOVERY_ERRORS = []
    shapes = _get_selected_shapes(book)
    macos_sheet = book.selection.sheet if sys.platform == "darwin" else None
    if not shapes:
        if sys.platform == "darwin":
            if _LAST_ARTIFACT_DISCOVERY_ERRORS:
                message = _LAST_ARTIFACT_DISCOVERY_ERRORS[0].user_message
            else:
                message = (
                    "No exportable XSTARS chart was found.\n\n"
                    "macOS export only supports charts generated by this XSTARS "
                    "version with valid rebuild information. Please regenerate the "
                    "chart and try again."
                )
        else:
            message = (
                "No image selected.\n\n"
                "Please click on a picture in the spreadsheet first, then run Export."
            )
        _show_error(book, message)
        return

    dlg_result = _show_export_dialog()
    if dlg_result is None:
        return
    path, dpi = dlg_result

    from pathlib import Path

    for i, shape in enumerate(shapes):
        if len(shapes) == 1:
            save_path = path
        else:
            p = Path(path)
            save_path = str(p.with_stem(f"{p.stem}_{i + 1}"))
        if sys.platform == "darwin":
            try:
                _export_artifact_picture(book, macos_sheet, shape, save_path, dpi)
            except artifacts.ArtifactError as exc:
                _ARTIFACT_LOGGER.warning("macOS artifact export failed: %s", exc)
                _show_error(book, exc.user_message)
                return
        else:
            _export_shape_highres(shape, save_path, dpi)

    book.app.status_bar = f"XSTARS: Exported to {path} ({dpi} DPI)"


def run_reset_settings() -> None:
    """Delete saved settings and restore defaults."""
    book = xw.Book.caller()
    try:
        from .config import DEFAULT_SETTINGS_PATH

        if DEFAULT_SETTINGS_PATH.exists():
            DEFAULT_SETTINGS_PATH.unlink()
        try:
            book.app.macro("MsgBox")(
                "Settings have been reset to defaults.",
                64,  # vbInformation
                "Excel-Prism",
            )
        except Exception:
            book.app.status_bar = "XSTARS: Settings reset to defaults"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_set_theme(theme_name: str) -> None:
    """Set the journal preset and persist it."""
    book = xw.Book.caller()
    try:
        from .config import JournalPreset

        preset = JournalPreset(theme_name)
        config = PrismConfig.load()
        config.journal_preset = preset
        config.save()
        label = {
            "none": "Default (Prism)",
            "nature": "Nature",
            "science": "Science",
            "cell": "Cell",
            "lancet": "Lancet",
            "nejm": "NEJM",
            "jama": "JAMA",
            "bmj": "BMJ",
        }.get(theme_name, theme_name)
        book.app.status_bar = f"XSTARS: Theme set to {label}"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_set_base_theme(theme_name: str) -> None:
    """Set the base visual theme and persist it."""
    book = xw.Book.caller()
    try:
        from .config import BaseTheme

        theme = BaseTheme(theme_name)
        config = PrismConfig.load()
        config.base_theme = theme
        config.save()
        book.app.status_bar = f"XSTARS: Base theme set to {theme_name.title()}"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_set_palette(palette_name: str) -> None:
    """Set the color style (post-processing) and persist it."""
    book = xw.Book.caller()
    try:
        from .config import PalettePreset
        from .styles import get_palette

        preset = PalettePreset(palette_name)
        config = PrismConfig.load()
        config.palette_preset = preset
        config.palette = get_palette(preset, config.journal_palette)
        config.save()
        label = "Original" if palette_name == "default" else palette_name.title()
        book.app.status_bar = f"XSTARS: Color style set to {label}"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_set_journal_palette(palette_name: str) -> None:
    """Set the journal color palette and persist it."""
    book = xw.Book.caller()
    try:
        from .config import JournalPalette
        from .styles import get_palette

        jp = JournalPalette(palette_name)
        config = PrismConfig.load()
        config.journal_palette = jp
        config.palette = get_palette(config.palette_preset, jp)
        config.save()
        label = "Default" if palette_name == "default" else palette_name.upper()
        book.app.status_bar = f"XSTARS: Journal palette set to {label}"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


# ── Standalone (frozen exe) wrappers ──────────────────────────────────────
# These are called by RunFrozenPython which can only pass simple string args.
# Each wraps a parameterized setter with a fixed argument.


def run_set_base_theme_classic() -> None:
    run_set_base_theme("classic")


def run_set_base_theme_bw() -> None:
    run_set_base_theme("bw")


def run_set_base_theme_minimal() -> None:
    run_set_base_theme("minimal")


def run_set_base_theme_dark() -> None:
    run_set_base_theme("dark")


def run_set_theme_none() -> None:
    run_set_theme("none")


def run_set_theme_nature() -> None:
    run_set_theme("nature")


def run_set_theme_science() -> None:
    run_set_theme("science")


def run_set_theme_cell() -> None:
    run_set_theme("cell")


def run_set_theme_lancet() -> None:
    run_set_theme("lancet")


def run_set_theme_nejm() -> None:
    run_set_theme("nejm")


def run_set_theme_jama() -> None:
    run_set_theme("jama")


def run_set_theme_bmj() -> None:
    run_set_theme("bmj")


def run_set_journal_palette_default() -> None:
    run_set_journal_palette("default")


def run_set_journal_palette_nature() -> None:
    run_set_journal_palette("nature")


def run_set_journal_palette_science() -> None:
    run_set_journal_palette("science")


def run_set_journal_palette_cell() -> None:
    run_set_journal_palette("cell")


def run_set_journal_palette_lancet() -> None:
    run_set_journal_palette("lancet")


def run_set_journal_palette_nejm() -> None:
    run_set_journal_palette("nejm")


def run_set_journal_palette_jama() -> None:
    run_set_journal_palette("jama")


def run_set_journal_palette_bmj() -> None:
    run_set_journal_palette("bmj")


def run_set_palette_default() -> None:
    run_set_palette("default")


def run_set_palette_colorblind() -> None:
    run_set_palette("colorblind")


def run_set_palette_vibrant() -> None:
    run_set_palette("vibrant")


def run_set_palette_pastel() -> None:
    run_set_palette("pastel")


def run_set_palette_deep() -> None:
    run_set_palette("deep")


def run_set_palette_muted() -> None:
    run_set_palette("muted")


def run_about() -> None:
    """Show version info dialog."""
    book = xw.Book.caller()
    try:
        version = "1.0.0"
        url = "https://github.com/Frankkk1912/excel-prism"
        msg = (
            f"XSTARS v{version}\n\n"
            "Quick statistical analysis and publication-quality\n"
            "visualization inside Excel.\n\n"
            "Author: Frank-SYSU\n"
            "Powered by scipy, matplotlib, seaborn & xlwings.\n\n"
            "License: MIT\n\n"
            f"Documentation & source:\n{url}"
        )
        try:
            book.app.macro("MsgBox")(msg, 64, "About Excel-Prism")
        except Exception:
            book.app.status_bar = f"Excel-Prism v{version}"
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def run_standard_curve() -> None:
    """Standard curve fitting and back-calculation tool."""
    book = xw.Book.caller()
    try:
        _run_standard_curve_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def _run_standard_curve_impl(book: xw.Book) -> None:
    """Read wide-format standard data, fit curve, optionally back-calculate samples.

    Input: wide DataFrame where column headers are concentration values and
    rows are OD replicates (same format as CCK-8 / other presets).
    """
    import numpy as np

    from .tools.standard_curve import back_calculate, wide_to_conc_od
    from .tools.standard_curve_dialog import StandardCurveDialog

    sheet = book.selection.sheet
    sel = book.selection
    handler = DataHandler()

    # Read wide-format selection (columns = concentration labels, rows = OD)
    raw = sel.options(pd.DataFrame, header=1, index=False).value
    raw.columns = [str(c).strip() for c in raw.columns]
    handler._selection_ref = sel

    if raw.shape[1] < 2:
        raise ValueError(
            "Please select at least 2 columns.\n"
            "Column headers should be concentration values (e.g. 0, 0.1, 1, 10, 100)."
        )

    # Clean numeric data
    df_wide = raw.copy()
    for col in df_wide.columns:
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")
    df_wide = df_wide.dropna(how="all").reset_index(drop=True)

    # Extract flattened conc/od arrays for fitting
    conc, od = wide_to_conc_od(df_wide)

    if len(conc) < 2:
        raise ValueError("Need at least 2 valid data points for curve fitting.")

    # Group info for dialog
    group_names = list(df_wide.columns)
    group_sizes = {col: int(df_wide[col].notna().sum()) for col in df_wide.columns}

    # Show dialog
    dialog = StandardCurveDialog(conc, od, group_names, group_sizes)
    config = dialog.show()
    if config is None or config.fit_result is None:
        return

    fit = config.fit_result

    # Output to Excel
    out_row = sel.row + sel.rows.count + 2
    out_col = sel.column

    # 1. Parameter table
    sheet.range((out_row, out_col)).value = [["Standard Curve Results"]]
    out_row += 1

    param_data = [
        ["Method", fit.method],
        ["Equation", fit.equation_str],
        ["R²", fit.r_squared if fit.r_squared is not None else "N/A"],
    ]
    for k, v in fit.params.items():
        param_data.append([k, v])
    sheet.range((out_row, out_col)).value = param_data
    out_row += len(param_data) + 1

    # 2. Back-calculate from a second selection if requested
    if config.back_calculate:
        sample_df = _select_sample_data(book, sheet)
        if sample_df is not None:
            result_df = sample_df.copy()
            for col in result_df.columns:
                od_vals = np.asarray(
                    pd.to_numeric(result_df[col], errors="coerce"), dtype=float
                )
                result_df[col] = back_calculate(fit, od_vals)

            out_row = _write_transformed_data(
                sheet,
                out_row,
                out_col,
                result_df,
                "Back-Calculated Concentrations",
            )

    # 3. Insert standard curve chart
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = PrismConfig.load()
    renderer_params = artifacts.standard_curve_renderer_params(conc, od, fit)
    fig = artifacts.build_standard_curve_figure(conc, od, renderer_params["fit"], cfg)
    pic_name = _next_plot_name(sheet, "XSTARS_StdCurve", book=book)
    pic = sheet.pictures.add(
        fig,
        name=pic_name,
        left=sheet.range((out_row, out_col)).left,
        top=sheet.range((out_row, out_col)).top,
    )
    _register_artifact_best_effort(
        book,
        sheet,
        pic,
        pic_name,
        df_wide,
        cfg,
        renderer_kind=artifacts.RendererKind.STANDARD_CURVE,
        renderer_params=renderer_params,
    )
    plt.close(fig)

    r2_str = f", R²={fit.r_squared:.4f}" if fit.r_squared else ""
    book.app.status_bar = f"XSTARS: Standard curve fitted ({fit.method}{r2_str})"


_A1_RANGE_RE = re.compile(
    r"^\$?[A-Za-z]{1,3}\$?[1-9]\d*(?::\$?[A-Za-z]{1,3}\$?[1-9]\d*)?$"
)


def _select_sample_data_macos(book: xw.Book, sheet) -> pd.DataFrame | None:
    """Prompt for an active-sheet A1 range without using Excel COM APIs."""
    root = None
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception as exc:
            _ARTIFACT_LOGGER.debug("Tk topmost hint unavailable: %s", exc)

        while True:
            address = simpledialog.askstring(
                "Standard Curve — Select Sample Data",
                "Enter the sample OD data range on the active sheet "
                "(with headers), for example A1:C6:",
                parent=root,
            )
            if address is None:
                return None
            address = address.strip()
            if not _A1_RANGE_RE.fullmatch(address):
                messagebox.showerror(
                    "Invalid Range",
                    "Enter an A1-style range on the active sheet, for example "
                    "A1:C6. Cross-sheet references and named ranges are not supported.",
                    parent=root,
                )
                continue

            try:
                rng = sheet.range(address)
                raw = rng.options(pd.DataFrame, header=1, index=False).value
                raw.columns = [str(c).strip() for c in raw.columns]
                for col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
                return raw.dropna(how="all").reset_index(drop=True)
            except Exception as exc:
                _ARTIFACT_LOGGER.info("Invalid macOS sample range %r: %s", address, exc)
                messagebox.showerror(
                    "Invalid Range",
                    "That range could not be read from the active sheet. "
                    "Check the A1 address and try again.",
                    parent=root,
                )
    except Exception as exc:
        _ARTIFACT_LOGGER.warning("macOS sample-range dialog failed: %s", exc)
        _show_error(
            book,
            "Could not open the sample range dialog. Please check your Python "
            "tkinter installation and try again.",
        )
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception as exc:
                _ARTIFACT_LOGGER.debug("Tk sample dialog cleanup failed: %s", exc)


def _select_sample_data(book: xw.Book, sheet) -> pd.DataFrame | None:
    """Prompt user for sample OD data and return a cleaned wide DataFrame."""
    if sys.platform == "darwin":
        return _select_sample_data_macos(book, sheet)

    try:
        # Type=8 returns a Range object; user can select with mouse
        result = book.app.api.InputBox(
            "Select the sample OD data range (with headers):",
            "Standard Curve — Select Sample Data",
            Type=8,
        )
    except Exception:
        # User clicked Cancel (InputBox raises on cancel with Type=8)
        return None

    if result is None:
        return None

    try:
        rng = sheet.range(result.Address)
        raw = rng.options(pd.DataFrame, header=1, index=False).value
        raw.columns = [str(c).strip() for c in raw.columns]
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        return raw.dropna(how="all").reset_index(drop=True)
    except Exception:
        return None


def run_transform_only() -> None:
    """Transform-only: apply preset data processing and write results back."""
    book = xw.Book.caller()
    try:
        _run_transform_only_impl(book)
    except _USER_ERRORS as exc:
        _show_error(book, str(exc))
    except Exception:
        _show_error(book, traceback.format_exc(), is_unexpected=True)


def _run_transform_only_impl(book: xw.Book) -> None:
    """Apply preset transform and write processed data only (no stats/plot)."""
    sheet = book.selection.sheet
    handler = DataHandler()
    wb_labels, df_wide = _read_selection_auto(handler, book)
    handler.validate(df_wide)

    groups = handler.group_names(df_wide)
    sizes = handler.group_sizes(df_wide)

    # Pre-configure dialog
    base_config = PrismConfig.load()
    if not base_config.preset_control_group and groups:
        base_config.preset_control_group = _guess_control(groups)
    if wb_labels is not None:
        base_config.preset_has_reference = True

    dialog = TransformOnlyDialog(groups, sizes, base_config=base_config)
    config = dialog.show()
    if config is None:
        return

    include_stats = getattr(config, "_include_stats", False)

    sel = book.selection
    start_row = sel.row + sel.rows.count + 2
    start_col = sel.column

    preset_type = config.experiment_preset

    # Labeled modes
    if (
        preset_type in (ExperimentPreset.WB, ExperimentPreset.QPCR)
        and config.preset_has_reference
        and wb_labels is not None
    ):
        if preset_type == ExperimentPreset.WB:
            from .presets.wb import WBPreset

            preset_cls = WBPreset
        else:
            from .presets.qpcr import QPCRPreset

            preset_cls = QPCRPreset

        preset = preset_cls()
        options = _build_preset_options(config)
        target_dfs = preset.transform_labeled(wb_labels, df_wide, options)

        current_row = start_row
        for target_name, fold_df in target_dfs:
            if include_stats:
                engine = StatsEngine(config)
                stats_result = engine.analyze(fold_df)
                stats_df = stats_result.to_dataframe()
                dest = sheet.range((current_row, start_col))
                dest.value = [[f"Statistics — {target_name}"]]
                current_row += 1
                dest = sheet.range((current_row, start_col))
                dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
                current_row += len(stats_df) + 2
            current_row = _write_transformed_data(
                sheet,
                current_row,
                start_col,
                fold_df,
                f"Processed Data — {target_name}",
            )
        count = len(target_dfs)
        book.app.status_bar = f"XSTARS: Transform only — {count} target(s) processed"
        return

    # Single-target mode
    df_wide = _apply_preset(df_wide, config)
    handler.validate(df_wide)

    current_row = start_row
    if include_stats:
        engine = StatsEngine(config)
        stats_result = engine.analyze(df_wide)
        stats_df = stats_result.to_dataframe()
        dest = sheet.range((current_row, start_col))
        dest.value = [stats_df.columns.tolist()] + stats_df.values.tolist()
        current_row += len(stats_df) + 2

    _write_transformed_data(sheet, current_row, start_col, df_wide, "Processed Data")
    book.app.status_bar = "XSTARS: Transform only — data written"
