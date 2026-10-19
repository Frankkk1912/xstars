"""Selection state and real ttk refresh regressions for SettingsDialog."""
import gc
import tkinter as tk
from contextlib import suppress
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from xstars.config import PrismConfig
from xstars.ui_dialog import SelectionSummary, SettingsDialog, selection_banner


def dialog_with_widgets(provider=None) -> Any:
    dialog: Any = SettingsDialog([], {}, request_selection=provider)
    for name in ("summary_label", "selection_button", "notebook", "ok_button",
                 "wb_control_combo", "qpcr_control_combo", "cck8_control_combo",
                 "wb_control_var", "qpcr_control_var", "cck8_control_var",
                 "cck8_blank_combo", "cck8_blank_var", "control_combo", "wb_ref_var",
                 "qpcr_ref_var", "paired_check", "paired_var", "exp_var", "ns_check"):
        setattr(dialog, "_" + name, MagicMock())
    dialog._data_tabs = ["preset", "figure", "theme", "export"]
    dialog._notebook.select.return_value = "preset"
    dialog._exp_var.get.return_value = "None"
    dialog._exp_label_map = {"wb": "Western Blot"}
    dialog._on_preset_change = MagicMock()
    dialog._on_wb_ref_toggle = MagicMock()
    dialog._on_qpcr_ref_toggle = MagicMock()
    return dialog


def test_selected_range_banner_has_group_sizes_and_address():
    assert selection_banner(["Control"], {"Control": 3}, "Data!$A$1:$A$4") == (
        "Control: n=3 · range Data!$A$1:$A$4"
    )


def test_empty_selection_banner_is_english():
    assert selection_banner([], {}, None) == "No data range detected. Please select data."


def test_empty_selection_then_apply_unlocks_and_refreshes_reselection():
    summary = SelectionSummary(["Control", "Treatment"], {"Control": 3}, "Data!A1:B4", True, object())
    dialog = dialog_with_widgets(lambda: summary)
    dialog._set_data_enabled()
    dialog._ok_button.configure.assert_called_with(state="disabled")
    dialog._on_select_data()
    assert dialog.selected_payload is summary.payload
    dialog._ok_button.configure.assert_called_with(state="normal")
    dialog._wb_control_combo.configure.assert_called_with(values=summary.group_names)
    dialog._wb_control_var.set.assert_called_with("Control")
    dialog._wb_ref_var.set.assert_called_with(True)
    dialog._apply_data(SelectionSummary(["New"], {"New": 4}, "Data!C1:C5", False, "new"))
    assert dialog.selected_payload == "new"
    dialog._paired_check.pack_forget.assert_called_once()
    dialog._cck8_blank_combo.configure.assert_called_with(values=["(none)", "New"])


def test_cancel_selection_keeps_empty_and_ok_disabled():
    dialog = dialog_with_widgets(lambda: None)
    dialog._set_data_enabled()
    dialog._on_select_data()
    assert dialog.selected_payload is None
    assert not dialog.group_names
    dialog._ok_button.configure.assert_called_with(state="disabled")
    assert "cancelled" in dialog._summary_label.configure.call_args.kwargs["text"]


def test_invalid_selection_displays_friendly_error_and_can_retry():
    def invalid():
        raise ValueError("Enter an A1-style range")
    dialog = dialog_with_widgets(invalid)
    dialog._root = MagicMock()
    with patch("xstars.ui_dialog.messagebox.showerror") as error:
        dialog._on_select_data()
    assert "A1-style" in error.call_args.args[1]
    assert error.call_args.kwargs["parent"] is dialog._root
    dialog._selection_button.configure.assert_called_with(state="normal")


def test_apply_data_preserves_valid_user_choices_and_reference_flags():
    dialog = dialog_with_widgets()
    dialog.group_names = ["Control", "Drug"]
    for name in ("wb", "qpcr", "cck8"):
        getattr(dialog, f"_{name}_control_var").get.return_value = "Drug"
    dialog._cck8_blank_var.get.return_value = "(none)"
    dialog._control_combo.get.return_value = "Drug"
    dialog._apply_data(SelectionSummary(["Control", "Drug"], {}, "Data!A1:B4", False, None))
    for name in ("wb", "qpcr", "cck8"):
        getattr(dialog, f"_{name}_control_var").set.assert_not_called()
    dialog._cck8_blank_var.set.assert_not_called()
    dialog._control_combo.set.assert_not_called()
    dialog._wb_ref_var.set.assert_not_called()
    dialog._qpcr_ref_var.set.assert_not_called()
    dialog._paired_check.pack.assert_called_once_with(anchor="w", padx=8, before=dialog._ns_check)


def test_unexpected_provider_error_uses_same_parented_messagebox():
    def broken():
        raise RuntimeError("host read failed")
    dialog = dialog_with_widgets(broken)
    dialog._root = MagicMock()
    with patch("xstars.ui_dialog.messagebox.showerror") as error:
        dialog._on_select_data()
    error.assert_called_once_with("Invalid Range", "host read failed", parent=dialog._root)
    dialog._selection_button.configure.assert_called_with(state="normal")


@pytest.mark.parametrize("visible_tab", ["preset", "figure", "theme", "export"])
def test_apply_data_reselects_visible_tab_and_flushes_idle_redraw(visible_tab):
    dialog = dialog_with_widgets()
    dialog._notebook.select.return_value = visible_tab
    dialog._apply_data(SelectionSummary(["Control"], {"Control": 3}, "A1:A4", False, None))
    assert dialog._notebook.select.call_args_list == [call(), call(visible_tab)]
    dialog._notebook.update_idletasks.assert_called_once_with()
    for tab in dialog._data_tabs:
        dialog._notebook.tab.assert_any_call(tab, state="normal")


@pytest.fixture
def selection_dialog(request):
    from xstars import ui_dialog

    try:
        root = ui_dialog.ttkb.Window(themename="cosmo") if ui_dialog.HAS_TTKB else tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"real Tk display unavailable: {exc}")
    root.withdraw()
    summary = SelectionSummary(["Control", "Drug"], {"Control": 3, "Drug": 3},
                               "Data!A1:B4", False, object())
    dialog = SettingsDialog([], {}, base_config=PrismConfig(),
                            hide_file_export=getattr(request, "param", False),
                            request_selection=lambda: summary)
    dialog._root = root
    dialog._build_ui()
    yield dialog
    with suppress(tk.TclError):
        root.destroy()
    del dialog
    # Finalize Tk callback cycles on the main thread before broker tests run.
    gc.collect()


@pytest.mark.parametrize("preset,expected_state", [("qpcr", "readonly"), ("wb", "disabled")])
def test_select_data_restores_error_space_child_only_for_qpcr(selection_dialog, preset, expected_state):
    dialog = selection_dialog
    dialog._exp_var.set(dialog._exp_label_map[preset])
    dialog._on_preset_change()
    assert str(dialog._error_space_combo.cget("state")) == "disabled"
    figure_tab = dialog._data_tabs[1]
    dialog._notebook.tab(figure_tab, state="normal")
    dialog._notebook.select(figure_tab)
    dialog._set_data_enabled()
    assert all(dialog._notebook.tab(tab, "state") == "disabled" for tab in dialog._data_tabs)
    assert dialog._notebook.select() == ""
    with patch.object(dialog._notebook, "select", wraps=dialog._notebook.select) as select, \
         patch.object(dialog._notebook, "update_idletasks",
                      wraps=dialog._notebook.update_idletasks) as redraw:
        dialog._on_select_data()
    select.assert_any_call(str(figure_tab))
    assert redraw.call_count == 2  # apply_data and selection callback finalization
    assert dialog._notebook.select() == str(figure_tab)
    assert str(dialog._error_space_combo.cget("state")) == expected_state
    assert all(dialog._notebook.tab(tab, "state") == "normal" for tab in dialog._data_tabs)
    if preset == "qpcr":
        assert not dialog._error_space_combo.instate(["disabled"])
    # Preset changes after unlocking must still apply their own restrictions.
    dialog._exp_var.set(dialog._exp_label_map["wb" if preset == "qpcr" else "qpcr"])
    dialog._on_preset_change()
    assert str(dialog._error_space_combo.cget("state")) == (
        "disabled" if preset == "qpcr" else "readonly"
    )


@pytest.mark.parametrize("selection_dialog", [True], indirect=True)
def test_select_data_preserves_wps_file_export_restrictions(selection_dialog):
    dialog = selection_dialog
    assert dialog.hide_file_export
    assert dialog._export_check.instate(["disabled"])
    dialog._on_select_data()
    for control in (dialog._export_check, dialog._export_fmt_combo,
                    dialog._export_dpi_combo, dialog._export_entry, dialog._browse_btn):
        assert control.instate(["disabled"])
    assert dialog._notebook.tab(dialog._data_tabs[3], "state") == "normal"
    assert dialog._notebook.select() == str(dialog._data_tabs[0])
