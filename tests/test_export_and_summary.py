"""Tests for export_figure, StatsResult.to_dataframe, and config.export_path."""

import matplotlib
matplotlib.use("Agg")

import os
import tempfile

import pandas as pd
import pytest

from xstars.config import PrismConfig
from xstars.plot_engine import PlotEngine, export_figure
from xstars.stats_engine import StatsEngine


class TestExportPath:
    def test_default_empty(self):
        assert PrismConfig().export_path == ""


class TestStatsResultToDataframe:
    def test_columns_and_rows(self, two_group_normal):
        result = StatsEngine().analyze(two_group_normal)
        df = result.to_dataframe()
        assert list(df.columns) == [
            "Group A", "Group B", "Test", "Statistic", "p-value", "Significance",
        ]
        assert len(df) == len(result.pairs)

    def test_values_match_pairs(self, three_group_normal):
        result = StatsEngine().analyze(three_group_normal)
        df = result.to_dataframe()
        for i, pair in enumerate(result.pairs):
            assert df.iloc[i]["Group A"] == pair.group_a
            assert df.iloc[i]["Group B"] == pair.group_b
            assert df.iloc[i]["p-value"] == pair.p_value
            assert df.iloc[i]["Significance"] == pair.stars

    def test_empty_pairs(self):
        from xstars.stats_engine import StatsResult
        sr = StatsResult(decision_path="none", normality_test="Shapiro-Wilk")
        df = sr.to_dataframe()
        assert len(df) == 0
        assert list(df.columns) == [
            "Group A", "Group B", "Test", "Statistic", "p-value", "Significance",
        ]


class TestExportFigure:
    @pytest.fixture
    def sample_fig(self, two_group_normal):
        engine = PlotEngine()
        return engine.plot(two_group_normal)

    @pytest.mark.parametrize("ext", [".png", ".svg", ".pdf"])
    def test_export_creates_file(self, sample_fig, ext):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"test_chart{ext}")
            export_figure(sample_fig, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
