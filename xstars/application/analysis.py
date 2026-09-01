"""Host-independent XSTARS analysis use cases.

This module owns data transformation, statistics, plotting, artifact creation,
and writeback-plan construction.  Excel/WPS adapters remain responsible only
for reading a selection, presenting dialogs, and executing a WritebackPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..config import ExperimentPreset, PrismConfig
from ..data_handler import DataHandler
from ..plot_engine import PlotEngine, export_figure
from ..presets import BasePreset, get_preset
from ..presets.cck8 import CCK8FitInfo, CCK8Options, CCK8Preset
from ..presets.elisa import ELISAOptions
from ..presets.qpcr import QPCROptions, QPCRPreset
from ..presets.wb import WBOptions
from ..stats_engine import StatsEngine, StatsResult
from ..styles import get_prism_context
from ..tools.standard_curve import (
    CurveFitResult,
    back_calculate,
    fit_standard_curve,
    wide_to_conc_od,
)
from .contracts import (
    ContractError,
    ErrorCode,
    ImageWriteback,
    SelectionPayload,
    TableWriteback,
    WritebackPlan,
    cell_to_a1,
    parse_cell,
)


@dataclass
class AnalysisResult:
    """Pure analysis outputs plus a host-neutral writeback description."""

    transformed_data: pd.DataFrame
    stats_result: StatsResult
    figure: Any
    preset: BasePreset | None = None
    writeback_plan: WritebackPlan = field(default_factory=WritebackPlan)
    extra_figures: dict[str, Any] = field(default_factory=dict)
    extra_render_data: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def figure_sources(self) -> dict[str, Any]:
        """In-process figure registry used by host plan executors."""
        return {"primary_figure": self.figure, **self.extra_figures}

    @property
    def render_data_sources(self) -> dict[str, pd.DataFrame]:
        return {"primary_figure": self.transformed_data, **self.extra_render_data}


@dataclass
class TargetAnalysisResult:
    """One target from a labeled WB/qPCR selection."""

    name: str
    transformed_data: pd.DataFrame
    stats_result: StatsResult
    figure: Any
    render_config: PrismConfig


@dataclass
class LabeledAnalysisResult:
    """Per-target outputs for a labeled WB/qPCR selection."""

    target_results: list[TargetAnalysisResult]
    preset: BasePreset
    writeback_plan: WritebackPlan = field(default_factory=WritebackPlan)

    @property
    def figure_sources(self) -> dict[str, Any]:
        """In-process figure registry keyed by each writeback image."""
        return {
            f"target_figure_{index}": target.figure
            for index, target in enumerate(self.target_results, start=1)
        }


@dataclass
class TransformResult:
    """Transform-only output and its host-neutral writeback description."""

    transformed_data: pd.DataFrame
    writeback_plan: WritebackPlan
    target_data: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


@dataclass
class StandardCurveResult:
    """Standard-curve fit, figure, and host-neutral writeback description."""

    standard_data: pd.DataFrame
    fit_result: CurveFitResult
    figure: Any
    writeback_plan: WritebackPlan

    @property
    def figure_sources(self) -> dict[str, Any]:
        return {"primary_figure": self.figure}


def guess_control(groups: list[str]) -> str:
    """Pick the most likely control group from column names."""
    for group in groups:
        if group.lower() in ("control", "ctrl", "con", "ctl", "nc", "vehicle"):
            return group
    if len(groups) >= 2 and groups[0].lower() in ("blank", "blk", "bg", "background"):
        return groups[1]
    return groups[0]


def guess_blank(groups: list[str]) -> str:
    """Pick the most likely blank group, or an empty string."""
    for group in groups:
        if group.lower() in ("blank", "blk", "bg", "background"):
            return group
    return ""


def build_preset_options(
    config: PrismConfig,
) -> WBOptions | QPCROptions | CCK8Options | ELISAOptions | None:
    """Map flat PrismConfig fields to the selected preset options DTO."""
    preset_type = config.experiment_preset
    if preset_type == ExperimentPreset.WB:
        return WBOptions(
            control_group=config.preset_control_group,
            has_reference=config.preset_has_reference,
            reference_protein=config.preset_reference_protein,
        )
    if preset_type == ExperimentPreset.QPCR:
        return QPCROptions(
            control_group=config.preset_control_group,
            input_format=config.preset_input_format,
            reference_gene=config.preset_reference_gene,
        )
    if preset_type == ExperimentPreset.CCK8:
        concentrations: list[float] = []
        if config.preset_concentrations:
            try:
                concentrations = [
                    float(value.strip())
                    for value in config.preset_concentrations.split(",")
                    if value.strip()
                ]
            except ValueError:
                concentrations = []
        return CCK8Options(
            control_group=config.preset_control_group,
            blank_group=config.preset_blank_group,
            fit_ic50=config.preset_fit_ic50,
            concentrations=concentrations,
            fit_method=config.preset_fit_method.value,
        )
    if preset_type == ExperimentPreset.ELISA:
        return ELISAOptions(
            control_group=config.preset_control_group,
            fit_result=cast(CurveFitResult | None, config.elisa_fit_result),
        )
    return None


def apply_preset(
    df_wide: pd.DataFrame, config: PrismConfig
) -> tuple[pd.DataFrame, BasePreset | None]:
    """Apply the configured experiment transform without host I/O."""
    preset = get_preset(config.experiment_preset)
    if preset is None:
        return df_wide, None
    options = build_preset_options(config)
    if options is None:
        raise ValueError(
            f"Missing options for preset {config.experiment_preset.value}."
        )
    transformed = preset.transform(df_wide, options)
    if config.y_label == "Value":
        config.y_label = preset.default_y_label
    return transformed, preset


def _populate_cck8_fit_info(
    transformed: pd.DataFrame,
    config: PrismConfig,
    preset: BasePreset | None,
) -> None:
    if not isinstance(preset, CCK8Preset):
        return
    cck8_preset = cast(Any, preset)
    result = cck8_preset.last_result
    if result is None or result.ic50 is None or result.fit_params is None:
        return
    options = build_preset_options(config)
    if not isinstance(options, CCK8Options):
        return
    dose_columns = [
        column for column in transformed.columns if column != options.control_group
    ]
    if options.concentrations and len(options.concentrations) == len(dose_columns):
        config.ic50_fit_info = CCK8FitInfo(
            concentrations=options.concentrations,
            fit_params=result.fit_params,
            dose_col_names=dose_columns,
        )


def transform_dataframe(
    df_wide: pd.DataFrame, config: PrismConfig
) -> tuple[pd.DataFrame, BasePreset | None]:
    """Clean, transform, and validate a wide DataFrame without host access."""
    cleaned = DataHandler.clean(df_wide.copy())
    DataHandler.validate(cleaned)
    transformed, preset = apply_preset(cleaned, config)
    DataHandler.validate(transformed)
    return transformed, preset


def _stats_input_frame(
    transformed: pd.DataFrame, preset: BasePreset | None
) -> pd.DataFrame:
    """Return the space in which hypothesis tests should run.

    For qPCR, fold-change values (2^-ΔΔCt) live on a nonlinear ratio scale,
    so the decision tree is evaluated on the linear log2 fold-change space
    (identically −ΔΔCt) instead — matching the Prism/ΔΔCt convention of
    testing on ΔCt and reporting 2^-ΔΔCt.  ``log2(2^-ΔΔCt) = -ΔΔCt`` holds
    exactly, so the log view can be recovered from the fold-change frame
    without extra preset state.
    """
    if isinstance(preset, QPCRPreset):
        values = np.log2(transformed.to_numpy(dtype=float))
        return pd.DataFrame(
            values, index=transformed.index, columns=transformed.columns
        )
    return transformed


def analyze_dataframe(df_wide: pd.DataFrame, config: PrismConfig) -> AnalysisResult:
    """Run the shared transform → stats → plot pipeline without host I/O."""
    transformed, preset = transform_dataframe(df_wide, config)
    _populate_cck8_fit_info(transformed, config, preset)

    stats_result = StatsEngine(config).analyze(_stats_input_frame(transformed, preset))
    figure = PlotEngine(config).plot(transformed, stats_result)
    if config.export_path:
        export_figure(figure, config.export_path, config.export_dpi)

    return AnalysisResult(
        transformed_data=transformed,
        stats_result=stats_result,
        figure=figure,
        preset=preset,
    )


def _json_cell(value: Any) -> Any:
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def dataframe_values(frame: pd.DataFrame) -> list[list[Any]]:
    """Convert a DataFrame to JSON-safe header + row values."""
    return [frame.columns.tolist()] + [
        [_json_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]


def build_analysis_writeback_plan(
    result: AnalysisResult,
    config: PrismConfig,
    *,
    start_row: int,
    start_column: int,
    image_name: str,
    include_processed_data: bool,
    processed_data_title: str = "Processed Data",
) -> WritebackPlan:
    """Describe existing Excel table spacing, picture anchor, and status text."""
    tables: list[TableWriteback] = []
    current_row = start_row

    if config.output_stats:
        stats_frame = result.stats_result.to_dataframe()
        tables.append(
            TableWriteback(
                start_cell=cell_to_a1(current_row, start_column),
                values=dataframe_values(stats_frame),
            )
        )
        current_row += len(stats_frame) + 2

        if isinstance(result.preset, CCK8Preset):
            cck8 = cast(Any, result.preset).last_result
            if cck8 is not None and cck8.ic50 is not None:
                ic50_values: list[list[Any]] = [
                    ["IC50", _json_cell(cck8.ic50)],
                    ["R²", _json_cell(cck8.r_squared)],
                ]
                if cck8.ic50_95ci:
                    ic50_values.append(
                        [
                            "IC50 95% CI",
                            f"{cck8.ic50_95ci[0]:.4g} – {cck8.ic50_95ci[1]:.4g}",
                        ]
                    )
                tables.append(
                    TableWriteback(
                        start_cell=cell_to_a1(current_row, start_column),
                        values=ic50_values,
                    )
                )
                current_row += len(ic50_values) + 2

    if include_processed_data and config.output_data:
        tables.append(
            TableWriteback(
                start_cell=cell_to_a1(current_row, start_column),
                values=[[processed_data_title]],
            )
        )
        current_row += 1
        tables.append(
            TableWriteback(
                start_cell=cell_to_a1(current_row, start_column),
                values=dataframe_values(result.transformed_data),
            )
        )
        current_row += len(result.transformed_data) + 2

    plan = WritebackPlan(
        tables=tables,
        images=[
            ImageWriteback(
                anchor_cell=cell_to_a1(current_row, start_column),
                name=image_name,
                source_key="primary_figure",
            )
        ],
        status_message=f"XSTARS: {result.stats_result.decision_path}",
    )
    result.writeback_plan = plan
    return plan


def split_selection_labels(
    payload: SelectionPayload,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """Split a mostly non-numeric first column from numeric selection data.

    This mirrors Excel's ``_read_selection_auto`` rule: the first data column
    is a label column only when more than half of its values fail numeric
    conversion. Rows whose numeric cells are all empty are removed from both
    outputs so labels remain aligned with their measurements.
    """
    headers = [str(value).strip() for value in payload.values[0]]
    raw = pd.DataFrame(payload.values[1:], columns=headers)
    if raw.empty:
        return None, DataHandler.clean(raw)

    first_column = pd.Series(raw.iloc[:, 0])
    numeric_first = pd.Series(pd.to_numeric(first_column, errors="coerce"))
    is_labeled = numeric_first.isna().sum() > len(first_column) // 2
    if not is_labeled:
        return None, DataHandler.clean(raw)

    labels = first_column.astype(str).str.strip()
    numeric = raw.iloc[:, 1:].copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    valid = ~numeric.isna().all(axis=1)
    return (
        labels.loc[valid].reset_index(drop=True),
        numeric.loc[valid].reset_index(drop=True),
    )


def _analyze_labeled(
    labels: pd.Series,
    frame: pd.DataFrame,
    config: PrismConfig,
    *,
    start_row: int,
    start_column: int,
) -> LabeledAnalysisResult:
    """Run Excel-equivalent labeled WB/qPCR analysis and plan each target."""
    DataHandler.validate(frame)
    preset = get_preset(config.experiment_preset)
    options = build_preset_options(config)
    if preset is None or options is None or not hasattr(preset, "transform_labeled"):
        raise ValueError("The selected preset does not support labeled data.")

    target_frames = cast(Any, preset).transform_labeled(labels, frame, options)
    if config.y_label == "Value":
        config.y_label = preset.default_y_label

    targets: list[TargetAnalysisResult] = []
    tables: list[TableWriteback] = []
    current_row = start_row
    for index, (target_name, transformed) in enumerate(target_frames, start=1):
        DataHandler.validate(transformed)
        stats_result = StatsEngine(config).analyze(
            _stats_input_frame(transformed, preset)
        )
        render_config = replace(config, title=str(target_name), export_path="")
        figure = PlotEngine(render_config).plot(transformed, stats_result)

        if config.export_path:
            path = Path(config.export_path)
            export_figure(
                figure,
                str(path.with_stem(f"{path.stem}_{index}")),
                config.export_dpi,
            )

        target = TargetAnalysisResult(
            name=str(target_name),
            transformed_data=transformed,
            stats_result=stats_result,
            figure=figure,
            render_config=render_config,
        )
        targets.append(target)

        if config.output_stats:
            stats_frame = stats_result.to_dataframe()
            tables.append(
                TableWriteback(
                    start_cell=cell_to_a1(current_row, start_column),
                    values=[[target.name]],
                )
            )
            current_row += 1
            tables.append(
                TableWriteback(
                    start_cell=cell_to_a1(current_row, start_column),
                    values=dataframe_values(stats_frame),
                )
            )
            current_row += len(stats_frame) + 2

        if config.output_data:
            tables.append(
                TableWriteback(
                    start_cell=cell_to_a1(current_row, start_column),
                    values=[[f"Processed Data — {target.name}"]],
                )
            )
            current_row += 1
            tables.append(
                TableWriteback(
                    start_cell=cell_to_a1(current_row, start_column),
                    values=dataframe_values(transformed),
                )
            )
            current_row += len(transformed) + 2

    images: list[ImageWriteback] = []
    image_row = current_row
    for index, target in enumerate(targets, start=1):
        width_inches, height_inches = target.figure.get_size_inches()
        width_points = width_inches * 72
        height_points = height_inches * 72
        images.append(
            ImageWriteback(
                anchor_cell=cell_to_a1(image_row, start_column),
                name=f"XSTARS_Plot_{target.name}",
                source_key=f"target_figure_{index}",
                width=width_points,
                height=height_points,
            )
        )
        # WPS anchors pictures to cells; approximate Excel's image-height +
        # 15-point vertical stacking using the default 15-point row height.
        image_row += ceil(height_points / 15) + 1

    preset_label, kind = (
        ("WB", "target")
        if config.experiment_preset is ExperimentPreset.WB
        else ("qPCR", "gene")
    )
    plan = WritebackPlan(
        tables=tables,
        images=images,
        status_message=(
            f"XSTARS: {preset_label} labeled mode — {len(targets)} {kind}(s) analyzed"
        ),
    )
    return LabeledAnalysisResult(targets, preset, plan)


def analyze_selection(
    payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    image_name: str = "XSTARS_Plot_1",
    include_processed_data: bool = False,
) -> AnalysisResult | LabeledAnalysisResult:
    """Analyze a serialized selection and return a host-neutral result."""
    labels, frame = split_selection_labels(payload)
    row, column = parse_cell(output_start_cell)
    if (
        labels is not None
        and config.preset_has_reference
        and config.experiment_preset in (ExperimentPreset.WB, ExperimentPreset.QPCR)
    ):
        return _analyze_labeled(
            labels,
            frame,
            config,
            start_row=row,
            start_column=column,
        )

    result = analyze_dataframe(frame, config)
    build_analysis_writeback_plan(
        result,
        config,
        start_row=row,
        start_column=column,
        image_name=image_name,
        include_processed_data=include_processed_data,
    )
    return result


def _curve_parameter_values(fit: CurveFitResult) -> list[list[Any]]:
    values: list[list[Any]] = [
        ["Method", fit.method],
        ["Equation", fit.equation_str],
        ["R²", _json_cell(fit.r_squared) if fit.r_squared is not None else "N/A"],
    ]
    values.extend([[str(key), _json_cell(value)] for key, value in fit.params.items()])
    return values


def plot_standard_curve(
    standard_data: pd.DataFrame,
    config: PrismConfig,
    *,
    fit_result: CurveFitResult | None = None,
) -> tuple[CurveFitResult, Any]:
    """Fit (or reuse a selected fit) and plot numeric wide standard data."""
    numeric = standard_data.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(how="all").reset_index(drop=True)
    conc, od = wide_to_conc_od(numeric)
    fit = fit_result or fit_standard_curve(conc, od, config.preset_elisa_fit_method)

    from matplotlib import pyplot as plt

    with get_prism_context(config.journal_preset, config.base_theme):
        figure, axis = plt.subplots(
            figsize=(config.fig_width, config.fig_height),
            dpi=config.dpi,
        )
        axis.scatter(
            conc, od, color=config.palette[0], s=30, zorder=5, label="Standards"
        )
        positive = conc[conc > 0]
        if fit.method == "linear" or len(positive) < 2:
            x_fit = np.linspace(conc.min(), conc.max() * 1.1, 200)
        else:
            x_fit = np.geomspace(positive.min() * 0.5, positive.max() * 1.5, 200)
        axis.plot(
            x_fit, fit.predict(x_fit), "-", color=config.palette[1], label=fit.method
        )
        if len(positive) >= 2 and positive.max() / positive.min() > 10:
            axis.set_xscale("log")
        axis.set_xlabel("Concentration")
        axis.set_ylabel("OD")
        title = "Standard Curve"
        if fit.r_squared is not None:
            title += f" (R² = {fit.r_squared:.4f})"
        axis.set_title(title)
        axis.legend(fontsize=8)
        figure.tight_layout()
    return fit, figure


def standard_curve_selection(
    payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    image_name: str = "XSTARS_StdCurve_1",
    fit_result: CurveFitResult | None = None,
    sample_payload: SelectionPayload | None = None,
) -> StandardCurveResult:
    """Fit a standard curve and optionally back-calculate a sample selection."""
    frame = DataHandler.from_selection_payload(payload)
    if frame.shape[1] < 2:
        raise ValueError("Select at least two concentration columns.")
    fit, figure = plot_standard_curve(frame, config, fit_result=fit_result)
    row, column = parse_cell(output_start_cell)
    parameter_values = _curve_parameter_values(fit)
    tables = [
        TableWriteback(
            start_cell=cell_to_a1(row, column), values=[["Standard Curve Results"]]
        ),
        TableWriteback(start_cell=cell_to_a1(row + 1, column), values=parameter_values),
    ]
    current_row = row + len(parameter_values) + 2
    if sample_payload is not None:
        samples = DataHandler.from_selection_payload(sample_payload)
        calculated = samples.copy()
        for sample_column in calculated.columns:
            od_values = pd.Series(
                pd.to_numeric(calculated[sample_column], errors="coerce"),
                dtype=float,
            )
            calculated[sample_column] = back_calculate(
                fit, od_values.to_numpy(dtype=float)
            )
        tables.extend(
            [
                TableWriteback(
                    start_cell=cell_to_a1(current_row, column),
                    values=[["Back-Calculated Concentrations"]],
                ),
                TableWriteback(
                    start_cell=cell_to_a1(current_row + 1, column),
                    values=dataframe_values(calculated),
                ),
            ]
        )
        current_row += len(calculated) + 3
    plan = WritebackPlan(
        tables=tables,
        images=[
            ImageWriteback(
                anchor_cell=cell_to_a1(current_row, column),
                name=image_name,
                source_key="primary_figure",
            )
        ],
        status_message=(
            f"XSTARS: Standard curve fitted ({fit.method})"
            + ("; samples back-calculated" if sample_payload is not None else "")
        ),
    )
    return StandardCurveResult(frame, fit, figure, plan)


def _elisa_standard_context(payload: SelectionPayload) -> str:
    """Return a bounded selection-shape and header preview for user errors."""
    received_rows = len(payload.values)
    received_columns = len(payload.values[0])
    headers = []
    for value in payload.values[0][:5]:
        compact = " ".join(str(value).split())
        headers.append(compact[:32] + ("…" if len(compact) > 32 else ""))
    preview = ", ".join(repr(header) for header in headers)
    if received_columns > 5:
        preview += f", …（共 {received_columns} 列）"
    return f"收到 {received_rows} 行 × {received_columns} 列；列头预览：[{preview}]"


def prepare_elisa_standard(
    standard_payload: SelectionPayload,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Validate an ELISA standard selection before opening the fit dialog."""
    standard_frame = DataHandler.from_selection_payload(standard_payload)
    standard_context = _elisa_standard_context(standard_payload)
    if standard_frame.shape[0] == 0:
        raise ContractError(
            ErrorCode.ANALYSIS_FAILED,
            f"标准品区域只包含 {standard_frame.shape[1]} 列表头、没有 OD 数据行——"
            f"请从浓度表头行开始框选并包含下方数据行（{standard_context}；清洗后有效数据 0 行）。",
        )
    if standard_frame.shape[1] < 2:
        raise ContractError(
            ErrorCode.ANALYSIS_FAILED,
            f"标准品区域的浓度数值列不足 2 列（实际 {standard_frame.shape[1]} 列；"
            f"{standard_context}）。请至少选择 2 个浓度列。",
        )
    try:
        conc, od = wide_to_conc_od(standard_frame)
    except ValueError as exc:
        if "cannot be parsed as a concentration" not in str(exc):
            raise
        raise ContractError(
            ErrorCode.ANALYSIS_FAILED,
            f"标准品区域的列头无法解析为浓度数值（{standard_context}）。"
            "请从浓度数值表头行开始选择，不要包含标题或说明文字。",
        ) from exc
    if len(conc) < 2:
        raise ContractError(
            ErrorCode.ANALYSIS_FAILED,
            f"标准曲线拟合至少需要 2 个有效数据点，实际收到 {len(conc)} 个（"
            f"标准品数据清洗后 {standard_frame.shape[0]} 行 × {standard_frame.shape[1]} 列；"
            f"{standard_context}）。",
        )
    return standard_frame, conc, od


def elisa_selections(
    standard_payload: SelectionPayload,
    sample_payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    image_name: str = "XSTARS_ELISA_1",
    fit_result: CurveFitResult | None = None,
    show_fit_curve: bool = False,
) -> AnalysisResult:
    """Fit standards then back-calculate and analyze a second sample selection."""
    standard_frame, conc, od = prepare_elisa_standard(standard_payload)
    fit = fit_result or fit_standard_curve(conc, od, config.preset_elisa_fit_method)
    config.experiment_preset = ExperimentPreset.ELISA
    config.elisa_fit_result = fit
    sample_frame = DataHandler.from_selection_payload(sample_payload)
    result = analyze_dataframe(sample_frame, config)
    row, column = parse_cell(output_start_cell)
    parameter_values = _curve_parameter_values(fit)
    build_analysis_writeback_plan(
        result,
        config,
        start_row=row + len(parameter_values) + 2,
        start_column=column,
        image_name=image_name,
        include_processed_data=True,
        processed_data_title="Back-Calculated Concentrations",
    )
    result.writeback_plan.tables[0:0] = [
        TableWriteback(
            start_cell=cell_to_a1(row, column), values=[["Standard Curve Results"]]
        ),
        TableWriteback(start_cell=cell_to_a1(row + 1, column), values=parameter_values),
    ]
    r2 = f", R²={fit.r_squared:.4f}" if fit.r_squared is not None else ""
    if show_fit_curve:
        _curve_fit, curve_figure = plot_standard_curve(
            standard_frame, config, fit_result=fit
        )
        result.extra_figures["standard_curve_figure"] = curve_figure
        result.extra_render_data["standard_curve_figure"] = standard_frame
        analysis_image = result.writeback_plan.images[0]
        analysis_row, _analysis_column = parse_cell(analysis_image.anchor_cell)
        height_points = float(result.figure.get_size_inches()[1]) * 72
        curve_anchor_row = analysis_row + ceil(height_points / 15) + 1
        result.writeback_plan.images.append(
            ImageWriteback(
                anchor_cell=cell_to_a1(curve_anchor_row, column),
                name="XSTARS_ELISA_StdCurve_1",
                source_key="standard_curve_figure",
            )
        )
    result.writeback_plan.status_message = (
        f"XSTARS: ELISA ({fit.method}{r2}) — {result.stats_result.decision_path}"
    )
    return result


def transform_selection(
    payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    title: str = "Processed Data",
    include_stats: bool = False,
) -> TransformResult:
    """Apply a preset only, including Excel-equivalent labeled/stat outputs."""
    labels, frame = split_selection_labels(payload)
    row, column = parse_cell(output_start_cell)
    if (
        labels is not None
        and config.preset_has_reference
        and config.experiment_preset in (ExperimentPreset.WB, ExperimentPreset.QPCR)
    ):
        DataHandler.validate(frame)
        preset = get_preset(config.experiment_preset)
        options = build_preset_options(config)
        if (
            preset is None
            or options is None
            or not hasattr(preset, "transform_labeled")
        ):
            raise ValueError("The selected preset does not support labeled data.")
        target_data = list(cast(Any, preset).transform_labeled(labels, frame, options))
        if not target_data:
            raise ValueError("Labeled transform produced no target data.")
        tables: list[TableWriteback] = []
        current_row = row
        for target_name, transformed in target_data:
            if include_stats:
                stats = (
                    StatsEngine(config)
                    .analyze(_stats_input_frame(transformed, preset))
                    .to_dataframe()
                )
                tables.extend(
                    [
                        TableWriteback(
                            start_cell=cell_to_a1(current_row, column),
                            values=[[f"Statistics — {target_name}"]],
                        ),
                        TableWriteback(
                            start_cell=cell_to_a1(current_row + 1, column),
                            values=dataframe_values(stats),
                        ),
                    ]
                )
                current_row += len(stats) + 3
            tables.extend(
                [
                    TableWriteback(
                        start_cell=cell_to_a1(current_row, column),
                        values=[[f"Processed Data — {target_name}"]],
                    ),
                    TableWriteback(
                        start_cell=cell_to_a1(current_row + 1, column),
                        values=dataframe_values(transformed),
                    ),
                ]
            )
            current_row += len(transformed) + 3
        normalized_targets = [(str(name), data) for name, data in target_data]
        return TransformResult(
            transformed_data=normalized_targets[0][1],
            target_data=normalized_targets,
            writeback_plan=WritebackPlan(
                tables=tables,
                status_message=(
                    f"XSTARS: Transform only — {len(normalized_targets)} target(s) processed"
                ),
            ),
        )

    transformed, preset = transform_dataframe(frame, config)
    tables = []
    current_row = row
    if include_stats:
        stats = (
            StatsEngine(config)
            .analyze(_stats_input_frame(transformed, preset))
            .to_dataframe()
        )
        tables.append(
            TableWriteback(
                start_cell=cell_to_a1(current_row, column),
                values=dataframe_values(stats),
            )
        )
        current_row += len(stats) + 2
    tables.extend(
        [
            TableWriteback(
                start_cell=cell_to_a1(current_row, column), values=[[title]]
            ),
            TableWriteback(
                start_cell=cell_to_a1(current_row + 1, column),
                values=dataframe_values(transformed),
            ),
        ]
    )
    suffix = " with statistics" if include_stats else ""
    return TransformResult(
        transformed_data=transformed,
        writeback_plan=WritebackPlan(
            tables=tables,
            status_message=f"XSTARS: Transform only — data written{suffix}",
        ),
    )
