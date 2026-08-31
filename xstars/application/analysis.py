"""Host-independent XSTARS analysis use cases.

This module owns data transformation, statistics, plotting, artifact creation,
and writeback-plan construction.  Excel/WPS adapters remain responsible only
for reading a selection, presenting dialogs, and executing a WritebackPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from ..config import ExperimentPreset, PrismConfig
from ..data_handler import DataHandler
from ..plot_engine import PlotEngine, export_figure
from ..presets import BasePreset, get_preset
from ..presets.cck8 import CCK8FitInfo, CCK8Options, CCK8Preset
from ..presets.elisa import ELISAOptions
from ..presets.qpcr import QPCROptions
from ..presets.wb import WBOptions
from ..stats_engine import StatsEngine, StatsResult
from ..tools.standard_curve import CurveFitResult
from .contracts import (
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

    @property
    def figure_sources(self) -> dict[str, Any]:
        """In-process figure registry used by the Excel plan executor."""
        return {"primary_figure": self.figure}


@dataclass
class TransformResult:
    """Transform-only output and its host-neutral writeback description."""

    transformed_data: pd.DataFrame
    writeback_plan: WritebackPlan


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


def apply_preset(df_wide: pd.DataFrame, config: PrismConfig) -> tuple[pd.DataFrame, BasePreset | None]:
    """Apply the configured experiment transform without host I/O."""
    preset = get_preset(config.experiment_preset)
    if preset is None:
        return df_wide, None
    options = build_preset_options(config)
    if options is None:
        raise ValueError(f"Missing options for preset {config.experiment_preset.value}.")
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
    dose_columns = [column for column in transformed.columns if column != options.control_group]
    if options.concentrations and len(options.concentrations) == len(dose_columns):
        config.ic50_fit_info = CCK8FitInfo(
            concentrations=options.concentrations,
            fit_params=result.fit_params,
            dose_col_names=dose_columns,
        )


def transform_dataframe(df_wide: pd.DataFrame, config: PrismConfig) -> tuple[pd.DataFrame, BasePreset | None]:
    """Clean, transform, and validate a wide DataFrame without host access."""
    cleaned = DataHandler.clean(df_wide.copy())
    DataHandler.validate(cleaned)
    transformed, preset = apply_preset(cleaned, config)
    DataHandler.validate(transformed)
    return transformed, preset


def analyze_dataframe(df_wide: pd.DataFrame, config: PrismConfig) -> AnalysisResult:
    """Run the shared transform → stats → plot pipeline without host I/O."""
    transformed, preset = transform_dataframe(df_wide, config)
    _populate_cck8_fit_info(transformed, config, preset)

    stats_result = StatsEngine(config).analyze(transformed)
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
                        ["IC50 95% CI", f"{cck8.ic50_95ci[0]:.4g} – {cck8.ic50_95ci[1]:.4g}"]
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


def analyze_selection(
    payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    image_name: str = "XSTARS_Plot_1",
    include_processed_data: bool = False,
) -> AnalysisResult:
    """Analyze a serialized selection and return a host-neutral result."""
    frame = DataHandler.from_selection_payload(payload)
    result = analyze_dataframe(frame, config)
    row, column = parse_cell(output_start_cell)
    build_analysis_writeback_plan(
        result,
        config,
        start_row=row,
        start_column=column,
        image_name=image_name,
        include_processed_data=include_processed_data,
    )
    return result


def transform_selection(
    payload: SelectionPayload,
    config: PrismConfig,
    *,
    output_start_cell: str,
    title: str = "Processed Data",
) -> TransformResult:
    """Apply only the selected preset and describe the table writeback."""
    frame = DataHandler.from_selection_payload(payload)
    transformed, _preset = transform_dataframe(frame, config)
    row, column = parse_cell(output_start_cell)
    plan = WritebackPlan(
        tables=[
            TableWriteback(start_cell=cell_to_a1(row, column), values=[[title]]),
            TableWriteback(
                start_cell=cell_to_a1(row + 1, column),
                values=dataframe_values(transformed),
            ),
        ],
        status_message="XSTARS: Transform only — data written",
    )
    return TransformResult(transformed_data=transformed, writeback_plan=plan)
