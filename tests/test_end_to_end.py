"""End-to-end integration tests for XSTARS.

Two test suites
---------------
TestPipelinePython
    Pure-Python pipeline (no xlwings): DataHandler.clean → StatsEngine →
    PlotEngine.  Verifies all module hand-offs produce the correct result
    types and counts.

TestRunQuickMocked
    Patches ``xlwings.Book.caller`` so that ``main.run_quick()`` executes
    the full orchestration loop without a live Excel instance.  Asserts
    that ``sheet.pictures.add()`` is called with the expected kwargs.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from unittest.mock import MagicMock, patch

from xstars.config import PrismConfig
from xstars.data_handler import DataHandler
from xstars.plot_engine import PlotEngine
from xstars.stats_engine import StatsEngine


# ---------------------------------------------------------------------------
# Helper: minimal xlwings mock
# ---------------------------------------------------------------------------

def _xw_mocks(df):
    """Return (mock_book, mock_sheet) that simulate a live Excel Range/Sheet.

    Wiring mirrors what main.run_quick() and DataHandler.read_selection()
    expect from the xlwings API:

    * book.selection            → Range-like mock (mock_sel)
    * book.selection.sheet      → Sheet-like mock (mock_sheet)
    * sel.options(...).value    → df.copy()
    * sel.row / .column / .columns.count → used by get_insertion_cell()
    * sheet.range(...).left/top → floats for picture placement
    * sheet.pictures.add(...)   → captured for assertions
    """
    mock_sel = MagicMock()
    mock_sel.options.return_value.value = df.copy()
    mock_sel.row = 1
    mock_sel.column = 1
    mock_sel.columns.count = len(df.columns)

    mock_sheet = MagicMock()
    mock_sheet.range.return_value.left = 0.0
    mock_sheet.range.return_value.top = 0.0
    mock_sheet.range.return_value.address = "$E$1"

    mock_book = MagicMock()
    mock_book.selection = mock_sel
    mock_sel.sheet = mock_sheet     # book.selection.sheet → mock_sheet

    return mock_book, mock_sheet


# ---------------------------------------------------------------------------
# 1. Pure-Python pipeline (no xlwings)
# ---------------------------------------------------------------------------

class TestPipelinePython:
    """Full data → stats → plot chain, no Excel involved."""

    def test_two_group_normal(self, two_group_normal):
        config = PrismConfig()
        result = StatsEngine(config).analyze(two_group_normal)
        fig = PlotEngine(config).plot(two_group_normal, result)

        assert len(result.pairs) == 1
        assert result.pairs[0].stars in {"ns", "*", "**", "***", "****"}
        plt.close(fig)

    def test_three_group_normal(self, three_group_normal):
        config = PrismConfig()
        result = StatsEngine(config).analyze(three_group_normal)
        fig = PlotEngine(config).plot(three_group_normal, result)

        assert len(result.pairs) == 3       # C(3, 2)
        assert result.omnibus_test == "One-way ANOVA"
        plt.close(fig)

    def test_three_group_nonnormal(self, three_group_nonnormal):
        config = PrismConfig()
        result = StatsEngine(config).analyze(three_group_nonnormal)
        fig = PlotEngine(config).plot(three_group_nonnormal, result)

        assert len(result.pairs) == 3
        if not result.all_normal or not result.equal_variance:
            assert result.omnibus_test == "Kruskal-Wallis"
        plt.close(fig)

    def test_paired(self, paired_data):
        config = PrismConfig(paired=True)
        result = StatsEngine(config).analyze(paired_data)
        fig = PlotEngine(config).plot(paired_data, result)

        assert "paired" in result.decision_path.lower()
        plt.close(fig)

    def test_data_handler_clean_passthrough(self, two_group_normal):
        """DataHandler.clean() preserves a well-formed wide DataFrame."""
        cleaned = DataHandler.clean(two_group_normal.copy())
        assert list(cleaned.columns) == ["Control", "Treatment"]
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) == len(two_group_normal)


# ---------------------------------------------------------------------------
# 2. run_quick() with mocked xlwings
# ---------------------------------------------------------------------------

class TestRunQuickMocked:
    """main.run_quick() must orchestrate all modules and insert the figure."""

    def test_pictures_add_called_once(self, two_group_normal):
        mock_book, mock_sheet = _xw_mocks(two_group_normal)
        with patch("xlwings.Book.caller", return_value=mock_book):
            from xstars.main import run_quick
            run_quick()
        mock_sheet.pictures.add.assert_called_once()

    def test_picture_name_and_update_flag(self, two_group_normal):
        mock_book, mock_sheet = _xw_mocks(two_group_normal)
        with patch("xlwings.Book.caller", return_value=mock_book):
            from xstars.main import run_quick
            run_quick()
        _, kwargs = mock_sheet.pictures.add.call_args
        assert kwargs["name"] == "XSTARS_Plot"
        assert kwargs["update"] is True

    def test_three_groups_completes(self, three_group_normal):
        mock_book, mock_sheet = _xw_mocks(three_group_normal)
        with patch("xlwings.Book.caller", return_value=mock_book):
            from xstars.main import run_quick
            run_quick()
        mock_sheet.pictures.add.assert_called_once()

    def test_status_bar_contains_decision_path(self, two_group_normal):
        mock_book, mock_sheet = _xw_mocks(two_group_normal)
        with patch("xlwings.Book.caller", return_value=mock_book):
            from xstars.main import run_quick
            run_quick()
        assert "XSTARS:" in str(mock_book.app.status_bar)
