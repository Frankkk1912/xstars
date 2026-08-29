"""Platform-branch regression tests for macOS support and Windows behavior."""

# Test doubles intentionally model dynamic xlwings/tkinter host APIs.
# pyright: reportAttributeAccessIssue=false

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from xstars import main
from xstars.artifacts import (
    CorruptArtifactError,
    MissingArtifactError,
    UnsupportedRendererError,
    UnsupportedSchemaError,
)
from xstars.plot_engine import export_figure


class StrictPicture:
    """Picture double that fails if the Darwin path touches COM shape members."""

    _COM_MEMBERS = {
        "CopyPicture",
        "Height",
        "Left",
        "LockAspectRatio",
        "Top",
        "Width",
        "api",
    }

    def __init__(self, name: str):
        self.name = name
        self.com_accesses: list[str] = []

    def __getattr__(self, name):
        if name in self._COM_MEMBERS:
            self.com_accesses.append(name)
            raise AssertionError(f"Darwin path accessed COM member {name}")
        raise AttributeError(name)


class DarwinApp:
    """App double with no ``api`` member, proving ShapeRange is not touched."""

    def __init__(self):
        self.status_bar = None

    def __getattr__(self, name):
        if name == "api":
            raise AssertionError("Darwin path accessed book.app.api")
        raise AttributeError(name)


def _darwin_book(tmp_path, pictures=()):
    sheet = SimpleNamespace(name="Analysis", pictures=list(pictures))
    book = SimpleNamespace(
        path=str(tmp_path),
        fullname=str(tmp_path / "experiment.xlsx"),
        selection=SimpleNamespace(sheet=sheet),
        app=DarwinApp(),
    )
    return book, sheet


def _install_tk_modules(monkeypatch, root, *, answers=()):
    """Install inert tkinter doubles without importing or opening a GUI."""
    tk = ModuleType("tkinter")
    tk.Tk = MagicMock(return_value=root)
    tk.messagebox = MagicMock()
    tk.simpledialog = MagicMock()
    tk.simpledialog.askstring.side_effect = list(answers)
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk.messagebox)
    monkeypatch.setitem(sys.modules, "tkinter.simpledialog", tk.simpledialog)
    return tk


def test_darwin_picture_discovery_filters_to_valid_artifact_backed_xstars(tmp_path):
    valid = StrictPicture("XSTARS_Plot_1")
    missing = StrictPicture("XSTARS_Plot_2")
    corrupt = StrictPicture("XSTARS_StdCurve_1")
    unrelated = StrictPicture("UserChart_1")
    book, _ = _darwin_book(tmp_path, [valid, missing, corrupt, unrelated])

    def load(identity):
        if identity.picture == valid.name:
            return object()
        if identity.picture == missing.name:
            raise MissingArtifactError("missing")
        raise CorruptArtifactError("corrupt")

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main.artifacts.load_artifact", side_effect=load) as loader,
    ):
        result = main._get_selected_shapes(book)

    assert result == [valid]
    assert [item.args[0].picture for item in loader.call_args_list] == [
        valid.name,
        missing.name,
        corrupt.name,
    ]
    assert all(not picture.com_accesses for picture in (valid, missing, corrupt))


def test_darwin_picture_discovery_returns_empty_for_legacy_or_unsupported(tmp_path):
    legacy = StrictPicture("XSTARS_Plot_Legacy")
    book, _ = _darwin_book(tmp_path, [legacy])

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch(
            "xstars.main.artifacts.load_artifact",
            side_effect=UnsupportedSchemaError("old schema"),
        ),
    ):
        assert main._get_selected_shapes(book) == []

    assert legacy.com_accesses == []


@pytest.mark.parametrize(
    ("suffix", "dpi"),
    [(".png", 150), (".tif", 300), (".jpg", 600), (".svg", 300), (".pdf", 1200)],
)
def test_darwin_rebuild_export_supports_formats_without_com_or_clipboard(
    tmp_path, suffix, dpi
):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    target = tmp_path / f"chart{suffix}"
    fig, axis = plt.subplots()
    axis.plot([0, 1], [1, 0])

    with (
        patch("xstars.main.artifacts.load_artifact", return_value=object()),
        patch("xstars.main.artifacts.rebuild_figure", return_value=fig),
        patch("xstars.main.export_figure", wraps=export_figure) as exporter,
        patch("PIL.ImageGrab.grabclipboard") as clipboard,
    ):
        main._export_artifact_picture(book, sheet, picture, str(target), dpi)

    assert target.exists()
    assert target.stat().st_size > 0
    exported_path = Path(exporter.call_args.args[1])
    assert exported_path.parent == tmp_path
    assert exported_path.suffix == suffix
    assert exporter.call_args.args[2] == dpi
    assert not list(tmp_path.glob(f".{target.stem}.*{suffix}"))
    assert picture.com_accesses == []
    clipboard.assert_not_called()
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    "error",
    [
        MissingArtifactError("missing payload"),
        CorruptArtifactError("corrupt payload"),
        UnsupportedSchemaError("unsupported schema"),
        UnsupportedRendererError("unsupported renderer"),
    ],
)
def test_darwin_payload_load_errors_fail_closed_without_partial_output(tmp_path, error):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    target = tmp_path / "failed.png"

    with (
        patch("xstars.main.artifacts.load_artifact", side_effect=error),
        pytest.raises(type(error), match=str(error)),
    ):
        main._export_artifact_picture(book, sheet, picture, str(target), 300)

    assert not target.exists()
    assert not list(tmp_path.glob(".failed.*.png"))
    assert picture.com_accesses == []


def test_darwin_render_error_is_reported_as_corrupt_and_creates_no_output(tmp_path):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    target = tmp_path / "failed.svg"

    with (
        patch("xstars.main.artifacts.load_artifact", return_value=object()),
        patch("xstars.main.artifacts.rebuild_figure", side_effect=ValueError("bad")),
        pytest.raises(CorruptArtifactError, match="could not be rendered"),
    ):
        main._export_artifact_picture(book, sheet, picture, str(target), 300)

    assert not target.exists()
    assert not list(tmp_path.glob(".failed.*.svg"))


def test_darwin_failed_save_preserves_existing_output_and_removes_temp(tmp_path):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    target = tmp_path / "existing.png"
    target.write_bytes(b"known-good")
    fig = plt.figure()

    with (
        patch("xstars.main.artifacts.load_artifact", return_value=object()),
        patch("xstars.main.artifacts.rebuild_figure", return_value=fig),
        patch("xstars.main.export_figure", side_effect=RuntimeError("save failed")),
        pytest.raises(RuntimeError, match="save failed"),
    ):
        main._export_artifact_picture(book, sheet, picture, str(target), 300)

    assert target.read_bytes() == b"known-good"
    assert not list(tmp_path.glob(".existing.*.png"))
    assert not plt.fignum_exists(fig.number)


def test_darwin_multi_export_uses_numbered_names_dpi_and_no_windows_helper(tmp_path):
    pictures = [StrictPicture("XSTARS_Plot_1"), StrictPicture("XSTARS_Plot_2")]
    book, sheet = _darwin_book(tmp_path, pictures)
    target = tmp_path / "plots.pdf"

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._get_selected_shapes", return_value=pictures),
        patch("xstars.main._show_export_dialog", return_value=(str(target), 600)),
        patch("xstars.main._export_artifact_picture") as artifact_export,
        patch("xstars.main._export_shape_highres") as windows_export,
    ):
        main._run_export_impl(book)

    assert artifact_export.call_args_list == [
        call(book, sheet, pictures[0], str(tmp_path / "plots_1.pdf"), 600),
        call(book, sheet, pictures[1], str(tmp_path / "plots_2.pdf"), 600),
    ]
    windows_export.assert_not_called()
    assert book.app.status_bar == f"XSTARS: Exported to {target} (600 DPI)"


@pytest.mark.parametrize(
    "error",
    [
        MissingArtifactError("missing payload"),
        CorruptArtifactError("corrupt payload"),
        UnsupportedSchemaError("unsupported schema"),
    ],
)
def test_darwin_export_reports_payload_error_and_stops_without_output(tmp_path, error):
    picture = StrictPicture("XSTARS_Plot_1")
    book, _ = _darwin_book(tmp_path, [picture])
    target = tmp_path / "never-created.png"

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._get_selected_shapes", return_value=[picture]),
        patch("xstars.main._show_export_dialog", return_value=(str(target), 300)),
        patch("xstars.main._export_artifact_picture", side_effect=error),
        patch("xstars.main._show_error") as show_error,
    ):
        main._run_export_impl(book)

    show_error.assert_called_once_with(book, error.user_message)
    assert not target.exists()
    assert book.app.status_bar is None


def test_darwin_no_exportable_chart_explains_regeneration(tmp_path):
    book, _ = _darwin_book(tmp_path)

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._get_selected_shapes", return_value=[]),
        patch("xstars.main._show_export_dialog") as dialog,
        patch("xstars.main._show_error") as show_error,
    ):
        main._run_export_impl(book)

    dialog.assert_not_called()
    message = show_error.call_args.args[1]
    assert "regenerate" in message.lower()
    assert "macOS export only supports" in message


def test_darwin_a1_input_returns_clean_dataframe_and_tolerates_topmost_failure(
    monkeypatch,
):
    root = MagicMock()
    root.attributes.side_effect = RuntimeError("unsupported")
    tk = _install_tk_modules(monkeypatch, root, answers=[" $A$1:$B$4 "])
    raw = pd.DataFrame({" Control ": ["1", "bad", None], "Drug": ["2", "3", None]})
    rng = MagicMock()
    rng.options.return_value.value = raw
    sheet = MagicMock()
    sheet.range.return_value = rng
    book = SimpleNamespace(app=DarwinApp())

    with patch("xstars.main.sys.platform", "darwin"):
        result = main._select_sample_data(book, sheet)

    assert result is not None
    assert list(result.columns) == ["Control", "Drug"]
    assert result.iloc[0].tolist() == [1.0, 2.0]
    assert pd.isna(result.iloc[1, 0])
    assert result.iloc[1, 1] == 3.0
    assert len(result) == 2
    sheet.range.assert_called_once_with("$A$1:$B$4")
    rng.options.assert_called_once_with(pd.DataFrame, header=1, index=False)
    tk.messagebox.showerror.assert_not_called()
    root.destroy.assert_called_once()


def test_darwin_a1_input_cancel_returns_none_without_reading_sheet(monkeypatch):
    root = MagicMock()
    _install_tk_modules(monkeypatch, root, answers=[None])
    sheet = MagicMock()
    book = SimpleNamespace(app=DarwinApp())

    with patch("xstars.main.sys.platform", "darwin"):
        result = main._select_sample_data(book, sheet)

    assert result is None
    sheet.range.assert_not_called()
    root.destroy.assert_called_once()


def test_darwin_invalid_a1_input_shows_error_and_retries(monkeypatch):
    root = MagicMock()
    tk = _install_tk_modules(
        monkeypatch, root, answers=["Sheet2!A1:B3", "named_range", "A1:B3"]
    )
    rng = MagicMock()
    rng.options.return_value.value = pd.DataFrame({"A": [1], "B": [2]})
    sheet = MagicMock()
    sheet.range.return_value = rng
    book = SimpleNamespace(app=DarwinApp())

    with patch("xstars.main.sys.platform", "darwin"):
        result = main._select_sample_data(book, sheet)

    assert result is not None
    assert result.equals(pd.DataFrame({"A": [1], "B": [2]}))
    assert tk.messagebox.showerror.call_count == 2
    sheet.range.assert_called_once_with("A1:B3")
    root.destroy.assert_called_once()


def test_darwin_unreadable_a1_input_shows_error_and_retries(monkeypatch):
    root = MagicMock()
    tk = _install_tk_modules(monkeypatch, root, answers=["A1:B3", "C1:D3"])
    good_range = MagicMock()
    good_range.options.return_value.value = pd.DataFrame({"C": [3], "D": [4]})
    sheet = MagicMock()
    sheet.range.side_effect = [RuntimeError("invalid address"), good_range]
    book = SimpleNamespace(app=DarwinApp())

    with patch("xstars.main.sys.platform", "darwin"):
        result = main._select_sample_data(book, sheet)

    assert result is not None
    assert result.equals(pd.DataFrame({"C": [3], "D": [4]}))
    tk.messagebox.showerror.assert_called_once()
    assert sheet.range.call_args_list == [call("A1:B3"), call("C1:D3")]
    root.destroy.assert_called_once()


def test_darwin_sample_dialog_failure_uses_common_error_path(monkeypatch):
    tk = ModuleType("tkinter")
    tk.Tk = MagicMock(side_effect=RuntimeError("no Tk"))
    tk.messagebox = MagicMock()
    tk.simpledialog = MagicMock()
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    sheet = MagicMock()
    book = SimpleNamespace(app=DarwinApp())

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._show_error") as show_error,
    ):
        assert main._select_sample_data(book, sheet) is None

    show_error.assert_called_once()
    assert "tkinter installation" in show_error.call_args.args[1]
    sheet.range.assert_not_called()


def test_show_error_still_uses_messagebox_when_topmost_fails(monkeypatch):
    root = MagicMock()
    root.attributes.side_effect = RuntimeError("unsupported")
    tk = _install_tk_modules(monkeypatch, root)
    app = MagicMock()
    book = SimpleNamespace(app=app)

    main._show_error(book, "visible error")

    tk.messagebox.showerror.assert_called_once_with(
        "Excel-Prism", "visible error", parent=root
    )
    root.destroy.assert_called_once()
    app.macro.assert_not_called()
    assert app.status_bar == "Excel-Prism Error: visible error"


def test_show_error_uses_vba_fallback_when_tkinter_fails(monkeypatch):
    tk = ModuleType("tkinter")
    tk.Tk = MagicMock(side_effect=RuntimeError("no display"))
    tk.messagebox = MagicMock()
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    app = MagicMock()
    msgbox = MagicMock()
    app.macro.return_value = msgbox
    book = SimpleNamespace(app=app)

    main._show_error(book, "fallback error")

    app.macro.assert_called_once_with("MsgBox")
    msgbox.assert_called_once_with(
        "Excel-Prism Error\n\nfallback error", 16, "Excel-Prism"
    )
    assert app.status_bar == "Excel-Prism Error: fallback error"


def test_windows_shape_range_first_contract_is_preserved():
    first = SimpleNamespace(Type=13)
    second = SimpleNamespace(Type=13)
    shape_range = MagicMock()
    shape_range.Count = 2
    shape_range.Item.side_effect = lambda index: {1: first, 2: second}[index]
    selection = SimpleNamespace(ShapeRange=shape_range)
    book = MagicMock()
    book.app.api.Selection = selection

    with patch("xstars.main.sys.platform", "win32"):
        result = main._get_selected_shapes(book)

    assert result == [first, second]
    assert shape_range.Item.call_args_list == [call(1), call(1), call(2)]
    assert book.selection.sheet.pictures.mock_calls == []


def test_windows_picture_fallback_keeps_xstars_plot_filter():
    shape_range = MagicMock()
    shape_range.Item.side_effect = RuntimeError("not a selected shape")
    book = MagicMock()
    book.app.api.Selection.ShapeRange = shape_range
    plot = SimpleNamespace(name="XSTARS_Plot_1", api=object())
    standard_curve = SimpleNamespace(name="XSTARS_StdCurve_1", api=object())
    unrelated = SimpleNamespace(name="UserChart_1", api=object())
    book.selection.sheet.pictures = [plot, standard_curve, unrelated]

    with patch("xstars.main.sys.platform", "win32"):
        result = main._get_selected_shapes(book)

    assert result == [plot.api]
    shape_range.Item.assert_called_once_with(1)


def test_windows_export_calls_original_com_helper_and_not_artifact_path(tmp_path):
    shape = object()
    book = MagicMock()
    target = tmp_path / "windows.png"

    with (
        patch("xstars.main.sys.platform", "win32"),
        patch("xstars.main._get_selected_shapes", return_value=[shape]),
        patch("xstars.main._show_export_dialog", return_value=(str(target), 300)),
        patch("xstars.main._export_shape_highres") as windows_export,
        patch("xstars.main._export_artifact_picture") as artifact_export,
    ):
        main._run_export_impl(book)

    windows_export.assert_called_once_with(shape, str(target), 300)
    artifact_export.assert_not_called()
    assert book.app.status_bar == f"XSTARS: Exported to {target} (300 DPI)"


def test_windows_sample_inputbox_type8_contract_is_preserved():
    book = MagicMock()
    selected_range = SimpleNamespace(Address="$A$1:$B$3")
    book.app.api.InputBox.return_value = selected_range
    raw = pd.DataFrame({" Control ": ["1", "bad"], "Drug": ["2", "3"]})
    rng = MagicMock()
    rng.options.return_value.value = raw
    sheet = MagicMock()
    sheet.range.return_value = rng

    with patch("xstars.main.sys.platform", "win32"):
        result = main._select_sample_data(book, sheet)

    assert result is not None
    book.app.api.InputBox.assert_called_once_with(
        "Select the sample OD data range (with headers):",
        "Standard Curve — Select Sample Data",
        Type=8,
    )
    sheet.range.assert_called_once_with("$A$1:$B$3")
    rng.options.assert_called_once_with(pd.DataFrame, header=1, index=False)
    assert list(result.columns) == ["Control", "Drug"]
    assert result.iloc[0].tolist() == [1.0, 2.0]
    assert pd.isna(result.iloc[1, 0])


def test_windows_sample_inputbox_cancel_returns_none():
    book = MagicMock()
    book.app.api.InputBox.side_effect = RuntimeError("cancel")
    sheet = MagicMock()

    with patch("xstars.main.sys.platform", "win32"):
        result = main._select_sample_data(book, sheet)

    assert result is None
    sheet.range.assert_not_called()


def test_run_export_entrypoint_still_uses_book_caller_and_impl():
    book = MagicMock()

    with (
        patch("xstars.main.xw.Book.caller", return_value=book) as caller,
        patch("xstars.main._run_export_impl") as implementation,
        patch("xstars.main._show_error") as show_error,
    ):
        main.run_export()

    caller.assert_called_once_with()
    implementation.assert_called_once_with(book)
    show_error.assert_not_called()
