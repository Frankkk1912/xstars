"""Characterization tests for the existing Excel/xlwings entry points.

These assertions deliberately describe host API calls made by ``xstars.main``.
They are the regression boundary for extracting host-independent application
logic; no live Excel instance is used.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

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


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS dispatches _select_sample_data to the native Tk range dialog; "
    "the COM InputBox cancel contract under test is unreachable headlessly",
)
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


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS _run_export_impl routes to the artifact rebuild export path; "
    "the COM shape-export numbered-paths contract under test is unreachable on darwin",
)
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


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_run_without_selection_selects_data_and_anchors_same_range(platform):
    df = pd.DataFrame({"Control": [1., 1.1, .9], "Drug": [2., 2.1, 1.9]})
    book, sheet, ranges = _book_with_selection(df, row=8, col=4)
    selected = book.selection
    book.selection = None

    def show(dialog):
        dialog._root = MagicMock()
        summary = dialog.request_selection()
        dialog.selected_payload = summary.payload
        assert summary.group_sizes["Control"] == 3
        # A subsequent host click must not change the writeback anchor.
        book.selection = MagicMock(row=100, column=20)
        return PrismConfig(output_stats=True)

    with (
        patch.object(main.sys, "platform", platform),
        patch.object(main, "_prompt_data_range", return_value=selected),
        patch.object(main.SettingsDialog, "show", show),
    ):
        main._run_impl(book)
    assert (14, 4) in ranges
    sheet.pictures.add.assert_called_once()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_run_without_selection_cancel_is_clean(platform):
    book, sheet, _ = _book_with_selection(pd.DataFrame({"A": [1, 2], "B": [2, 3]}))
    book.selection = None

    def cancel(dialog):
        dialog._root = MagicMock()
        assert dialog.request_selection() is None
        return None

    with (
        patch.object(main.sys, "platform", platform),
        patch.object(main, "_prompt_data_range", return_value=None),
        patch.object(main.SettingsDialog, "show", cancel),
        patch.object(main, "_show_error") as error,
    ):
        main._run_impl(book)
    sheet.pictures.add.assert_not_called()
    error.assert_not_called()


def test_run_selected_range_is_reported_to_dialog():
    book, _, _ = _book_with_selection(pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}))
    book.selection.sheet.name = "Data"
    book.selection.address = "$C$5:$D$8"

    def cancel(dialog):
        assert dialog.selection_label == "Data!$C$5:$D$8"
        assert dialog.group_sizes == {"A": 3, "B": 3}
        return None

    with patch.object(main.SettingsDialog, "show", cancel):
        main._run_impl(book)


def test_windows_prompt_rejects_multiarea_crosssheet_and_invalid_address_then_retries():
    book = MagicMock()
    sheet = MagicMock(name="sheet")
    sheet.name = "Data"
    from types import SimpleNamespace as NS
    book.app.api.InputBox.side_effect = [
        NS(Areas=NS(Count=2), Worksheet=NS(Name="Data"), Address="$A$1"),
        NS(Areas=NS(Count=1), Worksheet=NS(Name="Other"), Address="$A$1"),
        NS(Areas=NS(Count=1), Worksheet=NS(Name="Data"), Address="named_range"),
        NS(Areas=NS(Count=1), Worksheet=NS(Name="Data"), Address="$A$1:$B$4"),
    ]
    with patch.object(main, "_show_error") as error:
        assert main._prompt_range_windows(book, sheet, "Select", "Data") is sheet.range.return_value
    assert error.call_count == 3
    assert error.call_args.kwargs["parent_root"] is None
    assert "Invalid Range" in error.call_args.args[1]
    sheet.range.assert_called_once_with("$A$1:$B$4")


def test_read_valid_selection_rejects_shapes_and_disjoint_ranges():
    from types import SimpleNamespace as NS
    assert main._read_valid_selection(NS(selection=None)) is None
    assert main._read_valid_selection(NS(selection=NS())) is None
    book, _, _ = _book_with_selection(pd.DataFrame({"A": [1, 2]}))
    book.selection.api.Areas.Count = 2
    assert main._read_valid_selection(book) is None
    book.selection.api.Areas.Count = 1
    assert main._read_valid_selection(book) is book.selection


def test_read_valid_selection_uses_cheap_first_cell_probe():
    book, _, _ = _book_with_selection(pd.DataFrame({"A": [1, 2]}))
    book.selection[0, 0].value = "A"
    # The expensive full matrix property must not be read on the common path.
    from unittest.mock import PropertyMock
    with patch.object(type(book.selection), "value", new_callable=PropertyMock, create=True,
                      side_effect=AssertionError("full matrix read")):
        assert main._read_valid_selection(book) is book.selection


def test_read_valid_selection_logs_and_propagates_host_failure(caplog):
    from unittest.mock import PropertyMock
    book = MagicMock()
    with (
        patch.object(type(book), "selection", new_callable=PropertyMock, create=True,
                     side_effect=RuntimeError("AppleScript failed")),
        pytest.raises(RuntimeError, match="AppleScript failed"),
    ):
        main._read_valid_selection(book)
    assert "Failed to read worksheet selection" in caplog.text


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_analysis_provider_validation_error_is_parented_and_retries(platform):
    book, _, _ = _book_with_selection(pd.DataFrame({"Only": [1, 2, 3]}))
    selected = book.selection
    book.selection = None
    root = MagicMock()

    def show(dialog):
        dialog._root = root
        assert dialog.request_selection() is None
        return None

    with (
        patch.object(main.sys, "platform", platform),
        patch.object(main, "_prompt_data_range", side_effect=[selected, None]),
        patch.object(main.SettingsDialog, "show", show),
        patch.object(main, "_show_error") as error,
    ):
        main._analysis_settings(book)
    assert error.call_args.kwargs["parent_root"] is root


def test_analysis_provider_reuses_macos_prompt_frame_without_second_host_read():
    frame = pd.DataFrame({"Control": [1, 2, 3], "Drug": [2, 3, 4]})
    book, _, _ = _book_with_selection(frame)
    selected = book.selection
    book.selection = None

    def show(dialog):
        dialog._root = MagicMock()
        summary = dialog.request_selection()
        assert summary.group_sizes == {"Control": 3, "Drug": 3}
        return None

    with (
        patch.object(main.sys, "platform", "darwin"),
        patch.object(main, "_prompt_range_macos", return_value=(selected, frame)),
        patch.object(main.SettingsDialog, "show", show),
    ):
        main._analysis_settings(book)
    selected.options.assert_not_called()


@pytest.mark.parametrize("entry", [main._run_quick_impl, main._run_elisa_impl])
def test_quick_and_elisa_reuse_already_read_macos_prompt_frame(entry):
    from matplotlib import pyplot as plt
    frame = pd.DataFrame({"0": [.1, .2, .3], "10": [.5, .6, .7]})
    book, sheet, _ = _book_with_selection(frame)
    selected = book.selection
    book.selection = None
    with (
        patch.object(main.sys, "platform", "darwin"),
        patch.object(main, "_prompt_range_macos", return_value=(selected, frame)),
        patch("xstars.presets.elisa_dialog.ELISADialog.show", return_value=None),
    ):
        entry(book)
    selected.options.assert_not_called()
    if sheet.pictures.add.called:
        plt.close(sheet.pictures.add.call_args.args[0])


@pytest.mark.parametrize("values,rows", [([["A", "B"]], 1), ([[None, ""], [None, None]], 2)])
def test_prompt_range_windows_rejects_empty_data_and_parents_error(values, rows):
    from types import SimpleNamespace as NS
    book, sheet, _ = _book_with_selection(pd.DataFrame({"A": [1, 2]}))
    sheet.name = "Data"
    sheet.range.side_effect = None
    rng = sheet.range.return_value
    rng.rows.count = rows
    rng.value = values
    book.app.api.InputBox.side_effect = [NS(Address="$A$1:$B$2"), False]
    root = MagicMock()
    with patch.object(main, "_show_error") as error:
        assert main._prompt_range_windows(book, sheet, "Select", "Data", root) is None
    error.assert_called_once()
    assert error.call_args.kwargs["parent_root"] is root


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_quick_missing_selection_uses_minimal_prompt(platform):
    book, sheet, _ = _book_with_selection(pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}))
    sel = book.selection
    book.selection = None
    with (
        patch.object(main.sys, "platform", platform),
        patch.object(main, "_prompt_data_range", return_value=sel),
        patch.object(main.SettingsDialog, "show") as dialog,
    ):
        main._run_quick_impl(book)
    dialog.assert_not_called()
    sheet.pictures.add.assert_called_once()


def test_elisa_missing_standard_selection_cancel_exits_before_dialog():
    book, sheet, _ = _book_with_selection(pd.DataFrame({"0": [1, 2], "10": [2, 3]}))
    book.selection = None
    with patch.object(main, "_prompt_data_range", return_value=None) as prompt:
        main._run_elisa_impl(book)
    assert "standard" in prompt.call_args.args[3]
    sheet.pictures.add.assert_not_called()


@pytest.mark.parametrize("values,row_count", [(None, 1), ([[None, ""], [None, None]], 2), ([["A", "B"]], 1)])
def test_empty_or_header_only_selection_is_missing(values, row_count):
    book, _, _ = _book_with_selection(pd.DataFrame({"A": [1, 2, 3]}))
    book.selection.rows.count = row_count
    book.selection.value = values
    assert main._read_valid_selection(book) is None


def test_header_plus_empty_data_rows_keeps_existing_validation():
    book, _, _ = _book_with_selection(pd.DataFrame({"A": [None, None, None]}))
    book.selection.value = [["A"], [None], [None], [None]]
    assert main._read_valid_selection(book) is book.selection


@pytest.mark.parametrize("mode", ["inputbox", "a1"])
def test_windows_modal_selection_has_both_inputbox_and_a1_paths(mode):
    book = MagicMock()
    sheet = MagicMock()
    root = MagicMock()
    selected = MagicMock()
    with (
        patch.object(main.sys, "platform", "win32"),
        patch.object(main, "_WINDOWS_SELECTION_MODE", mode),
        patch.object(main, "_prompt_range_windows", return_value=selected) as inputbox,
        patch.object(main, "_prompt_range_macos", return_value=(selected, pd.DataFrame())) as a1,
    ):
        assert main._prompt_data_range(book, sheet, "Select", "Data", root) is selected
    if mode == "inputbox":
        inputbox.assert_called_once()
        a1.assert_not_called()
        root.grab_release.assert_called_once()
        root.grab_set.assert_called_once()
    else:
        a1.assert_called_once_with(book, sheet, "Select", "Data", root)
        inputbox.assert_not_called()


@pytest.mark.parametrize("preset,branch", [(main.ExperimentPreset.WB, "_run_wb_labeled"), (main.ExperimentPreset.QPCR, "_run_qpcr_labeled")])
def test_preset_supplemented_labels_use_final_data_and_reference_selection(preset, branch):
    df = pd.DataFrame({"Label": ["Target"] * 3 + ["GAPDH"] * 3,
                       "Control": [1, 2, 3, 2, 3, 4], "Drug": [2, 3, 4, 3, 4, 5]})
    book, _, _ = _book_with_selection(df)
    selected = book.selection
    book.selection = None

    def show(dialog):
        dialog._root = MagicMock()
        summary = dialog.request_selection()
        assert summary.has_label_column
        dialog.selected_payload = summary.payload
        return PrismConfig(experiment_preset=preset, preset_has_reference=True)

    with (
        patch.object(main, "_prompt_data_range", return_value=selected),
        patch.object(main.SettingsDialog, "show", show),
        patch.object(main, branch) as labeled,
    ):
        main._run_preset_impl(book, preset)
    labeled.assert_called_once()
    assert labeled.call_args.args[2]._selection_ref is selected
    assert labeled.call_args.args[3].tolist() == df["Label"].tolist()
    assert labeled.call_args.args[4].columns.tolist() == ["Control", "Drug"]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_elisa_supplements_standards_before_existing_fit_dialog(platform):
    from types import SimpleNamespace
    book, _, _ = _book_with_selection(pd.DataFrame({"0": [.1, .12, .11], "10": [.8, .9, .85]}))
    selected = book.selection
    book.selection = None
    with (
        patch.object(main.sys, "platform", platform),
        patch.object(main, "_prompt_data_range", return_value=selected),
        patch("xstars.presets.elisa_dialog.ELISADialog") as dialog,
    ):
        dialog.return_value.show.return_value = SimpleNamespace(fit_result=None)
        main._run_elisa_impl(book)
    concentrations, od = dialog.call_args.args
    assert len(concentrations) == len(od) == 6
    assert set(concentrations) == {0, 10}
