"""Characterization tests for the existing Excel/xlwings entry points.

These assertions deliberately describe host API calls made by ``xstars.main``.
They are the regression boundary for extracting host-independent application
logic; no live Excel instance is used.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from xstars import main
from xstars.config import PrismConfig

CORE_ENTRY_POINTS = {
    "run": "_run_impl",
    "run_quick": "_run_quick_impl",
    "run_wb": "_run_preset_impl",
    "run_qpcr": "_run_preset_impl",
    "run_cck8": "_run_preset_impl",
    "run_elisa": "_run_elisa_impl",
    "run_transform_only": "_run_transform_only_impl",
    "run_standard_curve": "_run_standard_curve_impl",
    "run_export": "_run_export_impl",
}


def _book_with_selection(df: pd.DataFrame, *, row: int = 5, col: int = 3):
    selection = MagicMock()
    selection.options.return_value.value = df.copy()
    selection.row = row
    selection.column = col
    selection.rows.count = len(df) + 1  # header plus data rows
    selection.columns.count = len(df.columns)

    ranges: dict[object, MagicMock] = {}

    def range_for(key):
        rng = ranges.setdefault(key, MagicMock())
        rng.left = 120.0
        rng.top = 240.0
        return rng

    sheet = MagicMock()
    sheet.range.side_effect = range_for
    sheet.pictures.__iter__.return_value = iter(())
    selection.sheet = sheet

    book = MagicMock()
    book.selection = selection
    book.app.status_bar = "before"
    return book, sheet, ranges


def test_public_entry_points_keep_zero_arg_signature_and_route_to_impl():
    for entry_name, implementation_name in CORE_ENTRY_POINTS.items():
        entry = getattr(main, entry_name)
        assert list(inspect.signature(entry).parameters) == []

        book = MagicMock()
        with (
            patch("xlwings.Book.caller", return_value=book) as caller,
            patch.object(main, implementation_name) as implementation,
        ):
            entry()

        caller.assert_called_once_with()
        implementation.assert_called_once()
        assert implementation.call_args.args[0] is book


def test_public_entry_points_route_user_errors_to_show_error():
    for entry_name, implementation_name in CORE_ENTRY_POINTS.items():
        book = MagicMock()
        with (
            patch("xlwings.Book.caller", return_value=book),
            patch.object(
                main, implementation_name, side_effect=ValueError("bad selection")
            ),
            patch.object(main, "_show_error") as show_error,
        ):
            getattr(main, entry_name)()

        show_error.assert_called_once_with(book, "bad selection")


def test_unexpected_entry_error_uses_traceback_route():
    book = MagicMock()
    with (
        patch("xlwings.Book.caller", return_value=book),
        patch.object(main, "_run_quick_impl", side_effect=RuntimeError("boom")),
        patch.object(main, "_show_error") as show_error,
    ):
        main.run_quick()

    args, kwargs = show_error.call_args
    assert args[0] is book
    assert "RuntimeError: boom" in args[1]
    assert kwargs == {"is_unexpected": True}


def test_quick_run_writes_stats_below_selection_and_inserts_named_picture():
    df = pd.DataFrame(
        {"Control": [1.0, 1.1, 0.9], "Treatment": [2.0, 2.1, 1.9]}
    )
    book, sheet, ranges = _book_with_selection(df)
    config = PrismConfig(output_stats=True)

    with patch.object(PrismConfig, "load", return_value=config):
        main._run_quick_impl(book)

    # Header + 3 data rows => four selected rows; output starts at row 11.
    stats_values = ranges[(11, 3)].value
    assert stats_values[0] == [
        "Group A",
        "Group B",
        "Test",
        "Statistic",
        "p-value",
        "Significance",
    ]
    assert stats_values[1][0:2] == ["Control", "Treatment"]

    # One stats row + header, then one blank row => plot anchor row 14.
    sheet.pictures.add.assert_called_once()
    figure, = sheet.pictures.add.call_args.args
    kwargs = sheet.pictures.add.call_args.kwargs
    assert figure is not None
    assert kwargs == {
        "name": "XSTARS_Plot_1",
        "left": 120.0,
        "top": 240.0,
    }
    assert (14, 3) in ranges
    assert str(book.app.status_bar).startswith("XSTARS: ")


def test_full_run_cancel_does_not_write_or_insert():
    df = pd.DataFrame(
        {"Control": [1.0, 1.1, 0.9], "Treatment": [2.0, 2.1, 1.9]}
    )
    book, sheet, _ = _book_with_selection(df)
    with (
        patch.object(PrismConfig, "load", return_value=PrismConfig()),
        patch.object(main.SettingsDialog, "show", return_value=None),
    ):
        main._run_impl(book)

    sheet.pictures.add.assert_not_called()
    assert book.app.status_bar == "before"


def test_second_selection_inputbox_cancel_returns_none():
    for inputbox_result in (None, RuntimeError("cancelled")):
        book = MagicMock()
        sheet = MagicMock()
        if isinstance(inputbox_result, Exception):
            book.app.api.InputBox.side_effect = inputbox_result
        else:
            book.app.api.InputBox.return_value = inputbox_result

        assert main._select_sample_data(book, sheet) is None
        book.app.api.InputBox.assert_called_once_with(
            "Select the sample OD data range (with headers):",
            "Standard Curve — Select Sample Data",
            Type=8,
        )
        sheet.range.assert_not_called()


def test_export_multiple_shapes_uses_numbered_paths_and_status(tmp_path: Path):
    book = MagicMock()
    shapes = [MagicMock(), MagicMock()]
    save_path = str(tmp_path / "chart.png")
    with (
        patch.object(main, "_get_selected_shapes", return_value=shapes),
        patch.object(main, "_show_export_dialog", return_value=(save_path, 300)),
        patch.object(main, "_export_shape_highres") as export,
    ):
        main._run_export_impl(book)

    assert export.call_count == 2
    assert export.call_args_list[0].args == (shapes[0], str(tmp_path / "chart_1.png"), 300)
    assert export.call_args_list[1].args == (shapes[1], str(tmp_path / "chart_2.png"), 300)
    assert book.app.status_bar == f"XSTARS: Exported to {save_path} (300 DPI)"


def test_reset_settings_deletes_file_and_uses_excel_message_box(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    book = MagicMock()
    with (
        patch("xlwings.Book.caller", return_value=book),
        patch("xstars.config.DEFAULT_SETTINGS_PATH", settings),
    ):
        main.run_reset_settings()

    assert not settings.exists()
    book.app.macro.assert_called_once_with("MsgBox")
    book.app.macro.return_value.assert_called_once_with(
        "Settings have been reset to defaults.", 64, "Excel-Prism"
    )
