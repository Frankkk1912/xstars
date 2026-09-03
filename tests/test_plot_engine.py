"""Tests for plot_engine module."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from xstars.config import (
    AnnotationFormat,
    ChartType,
    ErrorBarType,
    ExperimentPreset,
    PrismConfig,
)
from xstars.annotations import _format_p_scientific
from xstars.plot_engine import PlotEngine
from xstars.presets.qpcr import QPCRPreset, stats_input_frame
from xstars.stats_engine import StatsEngine


class TestBarScatter:
    def test_creates_figure(self, two_group_normal):
        engine = PlotEngine()
        fig = engine.plot(two_group_normal)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_with_stats(self, two_group_normal):
        stats = StatsEngine().analyze(two_group_normal)
        engine = PlotEngine()
        fig = engine.plot(two_group_normal, stats)
        assert fig is not None
        plt.close(fig)

    def test_three_groups(self, three_group_normal):
        stats = StatsEngine().analyze(three_group_normal)
        engine = PlotEngine()
        fig = engine.plot(three_group_normal, stats)
        assert fig is not None
        plt.close(fig)


class TestQPCRBars:
    def test_geo_stats_derive_from_log_space(self):
        engine = PlotEngine(
            PrismConfig(experiment_preset=ExperimentPreset.QPCR)
        )
        geo, lower, upper = engine._qpcr_geo_stats(pd.Series([1.0, 2.0, 4.0]))

        assert geo == pytest.approx(2.0)
        sem_log = np.std([0.0, 1.0, 2.0], ddof=1) / np.sqrt(3)
        assert lower == pytest.approx(geo - 2.0 ** (1.0 - sem_log))
        assert upper == pytest.approx(2.0 ** (1.0 + sem_log) - geo)
        assert upper > lower

    def test_qpcr_bar_uses_geometric_means_and_group_ticks(self):
        groups = ["Control", "siRNA", "OE"]
        df_wide = pd.DataFrame(
            {
                "Control": [1.0, 0.9, 1.1],
                "siRNA": [0.4, 0.5, 0.45],
                "OE": [2.8, 3.1, 2.9],
            }
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            show_points=False,
        )
        engine = PlotEngine(config)
        stats = StatsEngine(config).analyze(
            stats_input_frame(df_wide, QPCRPreset())
        )
        fig = engine.plot(df_wide, stats)
        ax = fig.axes[0]

        assert [tick.get_text() for tick in ax.get_xticklabels()] == groups
        assert len(ax.patches) == len(groups)
        from matplotlib.container import ErrorbarContainer
        from matplotlib.patches import Rectangle

        expected_heights = [
            float(2.0 ** np.log2(df_wide[g].to_numpy(dtype=float)).mean())
            for g in groups
        ]
        rectangles = [
            patch for patch in ax.patches if isinstance(patch, Rectangle)
        ]
        assert [patch.get_height() for patch in rectangles] == pytest.approx(
            expected_heights, abs=1e-9
        )
        assert sum(
            isinstance(container, ErrorbarContainer) for container in ax.containers
        ) == 1
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_non_qpcr_uses_standard_bar_path(self, monkeypatch, two_group_normal):
        engine = PlotEngine(
            PrismConfig(experiment_preset=ExperimentPreset.WB)
        )

        def fail_if_called(*_args, **_kwargs):
            pytest.fail("non-qPCR data must not use _qpcr_bars")

        monkeypatch.setattr(engine, "_qpcr_bars", fail_if_called)
        fig = engine.plot(two_group_normal)
        assert fig is not None
        plt.close(fig)


class TestViolin:
    def test_creates_figure(self, two_group_normal):
        config = PrismConfig(chart_type=ChartType.VIOLIN)
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal)
        assert fig is not None
        plt.close(fig)


class TestLine:
    def test_creates_figure(self, three_group_normal):
        config = PrismConfig(chart_type=ChartType.LINE)
        engine = PlotEngine(config)
        fig = engine.plot(three_group_normal)
        assert fig is not None
        plt.close(fig)


class TestErrorBars:
    @pytest.mark.parametrize("eb", list(ErrorBarType))
    def test_each_error_type(self, two_group_normal, eb):
        config = PrismConfig(error_bar=eb)
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal)
        assert fig is not None
        plt.close(fig)


class TestOptions:
    def test_no_points(self, two_group_normal):
        config = PrismConfig(show_points=False)
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal)
        assert fig is not None
        plt.close(fig)

    def test_custom_labels(self, two_group_normal):
        config = PrismConfig(y_label="Expression Level", title="My Experiment")
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Expression Level"
        assert ax.get_title() == "My Experiment"
        plt.close(fig)


class TestAnnotationFormat:
    def test_scientific_annotation(self, two_group_normal):
        config = PrismConfig(annotation_format=AnnotationFormat.SCIENTIFIC)
        stats = StatsEngine(config).analyze(two_group_normal)
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal, stats)
        # Should not raise; bracket text should contain "p"
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("p" in t for t in texts), f"Expected scientific p-value text, got {texts}"
        plt.close(fig)

    def test_stars_annotation(self, two_group_normal):
        config = PrismConfig(annotation_format=AnnotationFormat.STARS)
        stats = StatsEngine(config).analyze(two_group_normal)
        engine = PlotEngine(config)
        fig = engine.plot(two_group_normal, stats)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Stars format should have * characters
        assert any("*" in t for t in texts), f"Expected star text, got {texts}"
        plt.close(fig)


class TestFormatPScientific:
    def test_very_small(self):
        assert _format_p_scientific(0.00001) == "p<0.0001"

    def test_normal_value(self):
        result = _format_p_scientific(0.0123)
        assert result.startswith("p=")
        assert "e" in result


class TestConfigDefaults:
    def test_annotation_format_default(self):
        cfg = PrismConfig()
        assert cfg.annotation_format == AnnotationFormat.STARS

    def test_control_group_default(self):
        cfg = PrismConfig()
        assert cfg.control_group is None
