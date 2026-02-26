"""Tests for plot_engine module."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import pytest

from xstars.config import AnnotationFormat, ChartType, ErrorBarType, PrismConfig
from xstars.annotations import _format_p_scientific
from xstars.plot_engine import PlotEngine
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
