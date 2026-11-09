"""Canvas sizing (Fixed / Panel) and compact-bracket regressions (headless)."""

from __future__ import annotations

import re
import warnings
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from xstars.annotations import AnnotationLayoutError, AnnotationLayoutWarning
from xstars.application.analysis import analyze_dataframe, build_analysis_writeback_plan
from xstars.application.export import _config_from_snapshot, _config_snapshot
from xstars.application.worker import _apply_config_overrides
from xstars.config import (
    AnnotationFormat,
    BarWidth,
    CanvasMode,
    ChartType,
    JournalPreset,
    PrismConfig,
    YAxisScale,
)
from xstars.plot_engine import (
    PANEL_BAR_WIDTH_IN,
    PlotEngine,
    export_figure,
    resolve_figsize,
)
from xstars.stats_engine import PairResult, StatsEngine, StatsResult


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def groups(count: int) -> pd.DataFrame:
    return pd.DataFrame({f"G{i + 1}": [i + 0.9, i + 1.0, i + 1.1] for i in range(count)})


def selected_stats(pairs: list[tuple[str, str, float]]) -> StatsResult:
    return StatsResult(
        decision_path="precomputed", normality_test="Shapiro-Wilk",
        pairs=[PairResult(a, b, "precomputed", 0.0, p, "ns" if p > .05 else "*")
               for a, b, p in pairs],
    )


def plot(count: int, stats: bool = True, **config: Any):
    cfg = PrismConfig(**config)
    data = groups(count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AnnotationLayoutWarning)
        return PlotEngine(cfg).plot(data, StatsEngine(cfg).analyze(data) if stats else None)


def plot_area_in(fig: Any) -> tuple[float, float]:
    ax = fig.axes[0]
    return ax.bbox.width / fig.dpi, ax.bbox.height / fig.dpi


def bracket_spans(ax: Any) -> list[tuple[int, int, float]]:
    spans = []
    for line in ax.lines:
        if line.get_gid() == "significance_bracket":
            xs, ys = line.get_xdata(), line.get_ydata()
            spans.append((round(min(xs)), round(max(xs)), max(ys)))
    return spans


# ---------------------------------------------------------------- sizing


def test_fixed_keeps_legacy_custom_and_journal_sizes():
    assert resolve_figsize(PrismConfig()) == (4.0, 3.5)
    assert resolve_figsize(PrismConfig(journal_preset=JournalPreset.NATURE)) == (3.5, 3.0)
    assert resolve_figsize(PrismConfig(
        journal_preset=JournalPreset.NATURE, fig_width=6, fig_height=5,
    )) == (6, 5)
    for count in (3, 6):
        assert plot(count).get_size_inches() == pytest.approx((4.0, 3.5))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_canvas_size_is_rejected(value):
    with pytest.raises(ValueError, match="finite positive"):
        resolve_figsize(PrismConfig(fig_width=value))


@pytest.mark.parametrize("bar_width", list(BarWidth))
@pytest.mark.parametrize("chart", list(ChartType))
def test_panel_gives_every_group_the_same_width(bar_width, chart):
    for count in (2, 4, 6):
        fig = plot(count, canvas_mode=CanvasMode.PANEL, bar_width=bar_width, chart_type=chart)
        width, _ = plot_area_in(fig)
        assert width == pytest.approx(count * PANEL_BAR_WIDTH_IN[bar_width], abs=0.01)


@pytest.mark.parametrize("journal", list(JournalPreset))
def test_panel_heights_match_across_group_counts_with_and_without_brackets(journal):
    figures = [
        plot(count, stats=stats, canvas_mode=CanvasMode.PANEL, journal_preset=journal, title="T")
        for count in (2, 3, 5, 6)
        for stats in (False, True)
    ]
    expected = resolve_figsize(PrismConfig(journal_preset=journal))[1]
    for fig in figures:
        assert fig.get_size_inches()[1] == pytest.approx(expected)
        assert plot_area_in(fig)[1] == pytest.approx(plot_area_in(figures[0])[1], abs=0.01)


def test_panel_without_groups_keeps_the_configured_width():
    from xstars.application.analysis import plot_standard_curve

    data = pd.DataFrame({"1": [.1, .11, .09], "10": [1.0, 1.1, .9], "100": [10, 10.1, 9.9]})
    config = PrismConfig(canvas_mode=CanvasMode.PANEL, preset_elisa_fit_method="linear")
    _, fig = plot_standard_curve(data, config)
    assert fig.get_size_inches() == pytest.approx(resolve_figsize(config))


# -------------------------------------------------------------- brackets


@pytest.mark.parametrize("mode", list(CanvasMode))
@pytest.mark.parametrize("chart", list(ChartType))
@pytest.mark.parametrize("fmt", list(AnnotationFormat))
def test_brackets_sit_inside_an_axis_that_ends_on_a_tick(mode, chart, fmt):
    data = groups(4)
    config = PrismConfig(
        canvas_mode=mode, bar_width=BarWidth.WIDE, chart_type=chart,
        annotation_format=fmt, title="My experiment",
    )
    stats = StatsEngine(config).analyze(data)
    before = stats.to_dataframe().copy(deep=True)
    unannotated = PlotEngine(config).plot(data)
    fig = PlotEngine(config).plot(data, stats)
    ax = fig.axes[0]
    bottom, top = ax.get_ylim()
    assert bottom == pytest.approx(unannotated.axes[0].get_ylim()[0])
    assert any(np.isclose(top, tick) for tick in ax.yaxis.get_majorticklocs())
    labels = [t for t in ax.texts if t.get_gid() == "significance_label"]
    assert len(labels) == len(stats.pairs)
    pd.testing.assert_frame_equal(stats.to_dataframe(), before)
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    data_top_px = ax.transData.transform((0, unannotated.axes[0].get_ylim()[1]))[1]
    title_box = ax.title.get_window_extent(renderer)
    boxes = [t.get_window_extent(renderer) for t in labels]
    for box in boxes:
        assert box.y0 > data_top_px
        assert box.y1 <= ax.bbox.y1 + 0.5
        assert box.y1 < title_box.y0
        assert fig.bbox.contains(box.x0, box.y0)
        assert fig.bbox.contains(box.x1, box.y1)
    for i, first in enumerate(boxes):
        assert not any(first.overlaps(second) for second in boxes[i + 1:])


def test_four_groups_stack_compactly():
    fig = plot(4)
    # Six comparisons: the three neighbour pairs share the lowest row.
    assert cast(Any, fig.axes[0])._xstars_annotation_layout["lanes"] == 4


def test_longer_brackets_sit_above_the_brackets_they_span():
    spans = bracket_spans(plot(5).axes[0])
    assert len(spans) == 10
    for left, right, y in spans:
        for inner_left, inner_right, inner_y in spans:
            if (left, right) != (inner_left, inner_right) and left <= inner_left and inner_right <= right:
                assert y > inner_y


def test_disconnected_comparisons_share_a_lane():
    stats = selected_stats([("G1", "G2", .001), ("G3", "G4", .02)])
    fig = PlotEngine().plot(groups(4), stats)
    assert cast(Any, fig.axes[0])._xstars_annotation_layout["lanes"] == 1
    assert len(fig.axes[0].texts) == 2


def test_comparisons_sharing_a_group_share_a_lane_without_touching():
    stats = selected_stats([("G1", "G2", .001), ("G2", "G3", .02)])
    fig = PlotEngine().plot(groups(3), stats)
    assert cast(Any, fig.axes[0])._xstars_annotation_layout["lanes"] == 1
    first, second = sorted(
        (min(line.get_xdata()), max(line.get_xdata()))
        for line in fig.axes[0].lines if line.get_gid() == "significance_bracket"
    )
    assert first[1] < second[0]


def test_scientific_label_wider_than_plot_fails_clearly():
    stats = selected_stats([("G1", "G2", .01), ("G3", "G4", .02)])
    config = PrismConfig(fig_width=2.0, fig_height=4, annotation_format=AnnotationFormat.SCIENTIFIC)
    with pytest.raises(AnnotationLayoutError, match="too wide"):
        PlotEngine(config).plot(groups(4), stats)


def test_hide_ns_only_changes_display_not_comparisons():
    stats = selected_stats([("G1", "G2", .001), ("G1", "G3", .7)])
    before = stats.to_dataframe().copy(deep=True)
    fig = PlotEngine(PrismConfig(show_ns=False)).plot(groups(3), stats)
    assert len(fig.axes[0].texts) == 1
    pd.testing.assert_frame_equal(stats.to_dataframe(), before)
    all_ns = selected_stats([("G1", "G2", .7)])
    fig = PlotEngine(PrismConfig(show_ns=False)).plot(groups(2), all_ns)
    assert not fig.axes[0].texts
    assert not hasattr(fig.axes[0], "_xstars_annotation_layout")


def test_dense_layout_warns_and_reaches_host_status():
    config = PrismConfig()
    with pytest.warns(AnnotationLayoutWarning, match="Dense annotations"):
        result = analyze_dataframe(groups(6), config)
    plan = build_analysis_writeback_plan(
        result, config, start_row=10, start_column=1,
        image_name="XSTARS_Plot", include_processed_data=False,
    )
    assert "Dense annotations" in plan.status_message
    assert "hide ns" in plan.status_message
    assert len(result.figure.axes[0].texts) == 15


@pytest.mark.parametrize("mode", list(CanvasMode))
def test_many_groups_always_render_every_comparison_inside_the_canvas(mode):
    data = groups(8)
    stats = StatsEngine().analyze(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AnnotationLayoutWarning)
        fig = PlotEngine(PrismConfig(canvas_mode=mode)).plot(data, stats)
    labels = [t for t in fig.axes[0].texts if t.get_gid() == "significance_label"]
    assert len(labels) == len(stats.pairs) == 28
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    for label in labels:
        box = label.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)


def test_missing_group_is_not_silently_omitted():
    with pytest.raises(ValueError, match="missing from the plot"):
        PlotEngine().plot(groups(2), selected_stats([("G1", "Missing", .001)]))


def test_negative_data_keeps_its_lower_limit():
    data = groups(3) - 2.0
    stats = StatsEngine().analyze(data)
    plain = PlotEngine().plot(data)
    fig = PlotEngine().plot(data, stats)
    assert fig.axes[0].get_ylim()[0] == pytest.approx(plain.axes[0].get_ylim()[0])
    assert fig.axes[0].get_ylim()[1] > plain.axes[0].get_ylim()[1]


def test_star_thresholds_match_statistical_engine():
    from xstars.annotations import _format_p_stars
    from xstars.stats_engine import _stars

    for p in [0, .0001, .001, .01, .05, .050001, .9]:
        assert _format_p_stars(p) == _stars(p)


def test_no_warnings_or_band_when_all_comparisons_hidden():
    with warnings.catch_warnings(record=True) as recorded:
        fig = PlotEngine(PrismConfig(show_ns=False)).plot(
            groups(2), selected_stats([("G1", "G2", .9)]),
        )
    assert not recorded
    assert not fig.axes[0].texts


# ------------------------------------------------------ export and storage


@pytest.mark.parametrize("mode", list(CanvasMode))
def test_png_pdf_export_keeps_full_canvas_even_under_tight_rc(tmp_path, mode):
    config = PrismConfig(canvas_mode=mode, title="A title")
    fig = PlotEngine(config).plot(groups(3), StatsEngine(config).analyze(groups(3)))
    with plt.rc_context({"savefig.bbox": "tight"}):
        export_figure(fig, str(tmp_path / "plot.png"), dpi=100)
        export_figure(fig, str(tmp_path / "plot.pdf"), dpi=100)
    with Image.open(tmp_path / "plot.png") as image:
        assert image.size == pytest.approx(tuple(fig.get_size_inches() * 100), abs=1)
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]",
                      (tmp_path / "plot.pdf").read_bytes())
    assert match is not None
    assert (float(match[1]), float(match[2])) == pytest.approx(
        tuple(fig.get_size_inches() * 72), abs=.001,
    )


def test_canvas_mode_survives_render_snapshot_and_worker_override():
    config = PrismConfig(canvas_mode=CanvasMode.PANEL, bar_width=BarWidth.NARROW)
    snapshot = _config_snapshot(config)
    assert (snapshot["canvas_mode"], snapshot["bar_width"]) == ("panel", "narrow")
    restored = _config_from_snapshot(snapshot)
    assert (restored.canvas_mode, restored.bar_width) == (CanvasMode.PANEL, BarWidth.NARROW)
    override = _apply_config_overrides(PrismConfig(), {"canvas_mode": "panel", "bar_width": "wide"})
    assert (override.canvas_mode, override.bar_width) == (CanvasMode.PANEL, BarWidth.WIDE)


@pytest.mark.parametrize("mode", list(CanvasMode))
def test_payload_reexport_preserves_pixel_dimensions(tmp_path, mode):
    from xstars.application.export import persist_render_payload, render_payload_export

    config = PrismConfig(canvas_mode=mode, title="Persisted figure")
    data = groups(3)
    fig = PlotEngine(config).plot(data, StatsEngine(config).analyze(data))
    picture_id = "XSTARS_20261011_abcdef123456"
    persist_render_payload(picture_id, data, config, fig, artifacts_root=tmp_path / "artifacts")
    with plt.rc_context({"savefig.bbox": "tight"}):
        exported = render_payload_export(
            picture_id, "png", 96,
            artifacts_root=tmp_path / "artifacts", exports_root=tmp_path / "exports",
        )
    with Image.open(exported["path"]) as image:
        assert image.size == pytest.approx(tuple(fig.get_size_inches() * 96), abs=1)


@pytest.mark.parametrize("mode", list(CanvasMode))
def test_excel_xlwings_adapter_does_not_tight_crop_the_canvas(mode):
    from pathlib import Path

    from xlwings.utils import process_image

    from xstars.plot_engine import canvas_export_options

    config = PrismConfig(canvas_mode=mode, title="Excel figure")
    fig = PlotEngine(config).plot(groups(2), StatsEngine(config).analyze(groups(2)))
    expected = tuple(int(value * 200) for value in fig.get_size_inches())
    with plt.rc_context({"savefig.bbox": "tight"}):
        path, is_temporary = process_image(fig, "png", canvas_export_options(fig))
    try:
        assert is_temporary
        with Image.open(path) as image:
            assert image.size == pytest.approx(expected, abs=1)
    finally:
        Path(path).unlink(missing_ok=True)


def test_excel_artifact_restores_canvas_mode_and_accepts_legacy_missing_keys():
    from xstars.artifacts import _config_from_dict, _config_to_dict

    snapshot = _config_to_dict(PrismConfig(canvas_mode=CanvasMode.PANEL, bar_width=BarWidth.WIDE))
    restored = _config_from_dict(snapshot)
    assert (restored.canvas_mode, restored.bar_width) == (CanvasMode.PANEL, BarWidth.WIDE)
    del snapshot["canvas_mode"], snapshot["bar_width"]
    legacy = _config_from_dict(snapshot)
    assert (legacy.canvas_mode, legacy.bar_width) == (CanvasMode.FIXED, BarWidth.MEDIUM)


@pytest.mark.parametrize("mode", list(CanvasMode))
def test_excel_standard_curve_rebuild_sizes(mode):
    from xstars.artifacts import build_standard_curve_figure

    config = PrismConfig(canvas_mode=mode)
    figure = build_standard_curve_figure(
        [1, 10, 100], [.1, 1.0, 10.0],
        {"method": "linear", "params": {"slope": .1, "intercept": 0}, "r_squared": 1.0},
        config,
    )
    expected = (4.5, 3.5) if mode is CanvasMode.FIXED else resolve_figsize(config)
    assert figure.get_size_inches() == pytest.approx(expected)


# ------------------------------------------------------------- fit data


@pytest.mark.parametrize("mode", list(CanvasMode))
@pytest.mark.parametrize("chart", list(ChartType))
def test_fit_data_scales_axis_to_data_and_keeps_plot_area(mode, chart):
    config = {
        "canvas_mode": mode, "chart_type": chart,
        "y_axis_scale": YAxisScale.FIT_DATA, "title": "T",
    }
    plain = plot(3, stats=False, **config)
    heights = []
    for count in (3, 4, 6):
        fig = plot(count, **config)
        ax = fig.axes[0]
        top = ax.get_ylim()[1]
        assert any(np.isclose(top, tick) for tick in ax.yaxis.get_majorticklocs())
        heights.append(plot_area_in(fig)[1])
        labels = [t for t in ax.texts if t.get_gid() == "significance_label"]
        assert len(labels) == count * (count - 1) // 2
        fig.canvas.draw()
        renderer = cast(Any, fig.canvas).get_renderer()
        title_box = ax.title.get_window_extent(renderer)
        boxes = [t.get_window_extent(renderer) for t in labels]
        for box in boxes:
            assert box.y1 < title_box.y0
            assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        for i, first in enumerate(boxes):
            assert not any(first.overlaps(second) for second in boxes[i + 1:])
    assert heights == pytest.approx([plot_area_in(plain)[1]] * 3, abs=0.01)


def test_fit_data_puts_the_tallest_value_near_80_percent():
    data = pd.DataFrame({"A": [1.0, 1.1, 0.9], "B": [7.8, 8.0, 7.9]})
    fig = PlotEngine(PrismConfig(y_axis_scale=YAxisScale.FIT_DATA)).plot(data)
    ax = fig.axes[0]
    assert ax.get_ylim()[1] == pytest.approx(10.0)


def test_fit_data_grows_canvas_only_when_brackets_overflow():
    few = plot(3, stats=False, y_axis_scale=YAxisScale.FIT_DATA)
    one = PlotEngine(PrismConfig(y_axis_scale=YAxisScale.FIT_DATA)).plot(
        groups(3), selected_stats([("G1", "G2", .001)]),
    )
    assert one.get_size_inches() == pytest.approx(few.get_size_inches())
    many = plot(6, y_axis_scale=YAxisScale.FIT_DATA)
    assert many.get_size_inches()[1] > few.get_size_inches()[1]


def test_fit_data_never_warns_about_dense_brackets():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AnnotationLayoutWarning)
        plot(6, y_axis_scale=YAxisScale.FIT_DATA)


# ------------------------------------------------------ hidden brackets


def test_hidden_comparisons_are_not_drawn_but_statistics_keep_them():
    data = groups(4)
    config = PrismConfig(hidden_comparisons=[["G3", "G1"], ["G2", "G4"]])
    stats = StatsEngine(config).analyze(data)
    assert len(stats.pairs) == 6
    fig = PlotEngine(config).plot(data, stats)
    spans = {(left, right) for left, right, _ in bracket_spans(fig.axes[0])}
    assert len(spans) == 4
    assert (0, 2) not in spans and (1, 3) not in spans


def test_hiding_every_comparison_draws_no_brackets():
    data = groups(3)
    config = PrismConfig(hidden_comparisons=[["G1", "G2"], ["G1", "G3"], ["G2", "G3"]])
    fig = PlotEngine(config).plot(data, StatsEngine(config).analyze(data))
    assert not fig.axes[0].texts


def test_hidden_comparisons_and_y_axis_survive_snapshots_and_artifacts():
    from xstars.artifacts import _config_from_dict, _config_to_dict

    config = PrismConfig(y_axis_scale=YAxisScale.FIT_DATA, hidden_comparisons=[["A", "B"]])
    restored = _config_from_snapshot(_config_snapshot(config))
    assert restored.y_axis_scale is YAxisScale.FIT_DATA
    assert restored.hidden_comparisons == [["A", "B"]]
    snapshot = _config_to_dict(config)
    artifact = _config_from_dict(snapshot)
    assert (artifact.y_axis_scale, artifact.hidden_comparisons) == (YAxisScale.FIT_DATA, [["A", "B"]])
    del snapshot["y_axis_scale"], snapshot["hidden_comparisons"]
    legacy = _config_from_dict(snapshot)
    assert (legacy.y_axis_scale, legacy.hidden_comparisons) == (YAxisScale.FIT_BRACKETS, [])
