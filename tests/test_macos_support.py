"""Platform-branch regression tests for macOS support and Windows behavior."""

# Test doubles intentionally model dynamic xlwings/tkinter host APIs.
# pyright: reportAttributeAccessIssue=false

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from xstars import artifacts, main
from xstars.artifacts import (
    ArtifactIdentity,
    CorruptArtifactError,
    MissingArtifactError,
    UnsupportedRendererError,
    UnsupportedSchemaError,
    build_payload,
    load_artifact,
    save_artifact,
)
from xstars.config import ExperimentPreset, PrismConfig
from xstars.plot_engine import export_figure
from xstars.stats_engine import StatsResult
from xstars.tools.standard_curve import fit_standard_curve


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


def test_unsaved_workbook_registration_is_skipped_with_diagnostic(tmp_path):
    book = SimpleNamespace(fullname="", name="Book1")
    sheet = SimpleNamespace(name="Analysis")
    picture = SimpleNamespace(name="XSTARS_Plot_1")
    frame = pd.DataFrame({"Control": [1.0], "Drug": [2.0]})

    with patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path):
        assert not main._register_artifact_best_effort(
            book, sheet, picture, picture.name, frame, PrismConfig()
        )

    diagnostic = main.get_last_artifact_diagnostic()
    assert diagnostic is not None
    assert "Save the workbook" in diagnostic
    assert not list(tmp_path.glob("*.json"))


def test_darwin_unsaved_workbook_export_reports_save_and_regenerate(tmp_path):
    picture = StrictPicture("XSTARS_Plot_1")
    book, _ = _darwin_book(tmp_path, [picture])
    book.fullname = ""
    book.name = "Book1"

    with (
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._show_export_dialog") as dialog,
        patch("xstars.main._show_error") as show_error,
    ):
        main._run_export_impl(book)

    dialog.assert_not_called()
    message = show_error.call_args.args[1]
    assert "Save the workbook" in message
    assert "regenerate" in message.lower()


@pytest.mark.parametrize(
    ("failure", "message_fragment"),
    [
        ("missing", "No rebuild information"),
        ("corrupt", "cannot be read"),
        ("unsupported", "not supported"),
    ],
)
def test_darwin_real_discovery_reports_specific_artifact_error(
    tmp_path, failure, message_fragment
):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    identity = ArtifactIdentity(
        workbook=f"path:{book.fullname}", sheet=sheet.name, picture=picture.name
    )
    payload_path = tmp_path / f"{identity.key}.json"
    if failure == "corrupt":
        payload_path.write_text("{not-json", encoding="utf-8")
    elif failure == "unsupported":
        save_artifact(
            build_payload(
                identity,
                pd.DataFrame({"Control": [1.0], "Drug": [2.0]}),
                PrismConfig(),
            ),
            tmp_path,
        )
        document = json.loads(payload_path.read_text(encoding="utf-8"))
        document["schema_version"] = artifacts.SCHEMA_VERSION + 1
        payload_path.write_text(json.dumps(document), encoding="utf-8")

    with (
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch("xstars.main.sys.platform", "darwin"),
        patch("xstars.main._show_export_dialog") as dialog,
        patch("xstars.main._show_error") as show_error,
    ):
        main._run_export_impl(book)

    dialog.assert_not_called()
    assert message_fragment in show_error.call_args.args[1]


def test_reused_picture_name_failed_registration_cannot_export_stale_payload(
    tmp_path,
):
    picture = StrictPicture("XSTARS_Plot_1")
    book, sheet = _darwin_book(tmp_path, [picture])
    # The old picture was deleted, so generation reuses its Excel name.
    assert main._next_plot_name(SimpleNamespace(pictures=[])) == picture.name
    identity = main._artifact_identity_for_picture(book, sheet, picture)
    old_frame = pd.DataFrame({"Control": [1.0], "Drug": [2.0]})
    save_artifact(
        build_payload(identity, old_frame, PrismConfig(title="old")), tmp_path
    )
    with (
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch("xstars.main.sys.platform", "darwin"),
    ):
        assert (
            main._next_plot_name(
                SimpleNamespace(name=sheet.name, pictures=[]), book=book
            )
            == "XSTARS_Plot_2"
        )
    with (
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch("xstars.main.sys.platform", "win32"),
    ):
        assert (
            main._next_plot_name(
                SimpleNamespace(name=sheet.name, pictures=[]), book=book
            )
            == "XSTARS_Plot_1"
        )
    real_write = artifacts._atomic_write_json

    def fail_replacement(path, document):
        if Path(path).name == f"{identity.key}.json":
            raise PermissionError("replacement denied")
        return real_write(path, document)

    with (
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch("xstars.artifacts._atomic_write_json", side_effect=fail_replacement),
    ):
        assert not main._register_artifact_best_effort(
            book,
            sheet,
            picture,
            picture.name,
            pd.DataFrame({"Control": [9.0], "Drug": [10.0]}),
            PrismConfig(title="new"),
        )

    with pytest.raises(MissingArtifactError):
        load_artifact(identity, tmp_path)


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


@pytest.mark.parametrize(
    ("suffix", "expected_format", "expected_kwargs"),
    [
        ("png", "PNG", {"dpi": (300, 300)}),
        ("tif", "TIFF", {"dpi": (300, 300), "compression": "tiff_lzw"}),
        ("pdf", "PDF", {"resolution": 300}),
    ],
)
def test_windows_highres_helper_preserves_com_contract_and_restores_shape(
    tmp_path, suffix, expected_format, expected_kwargs
):
    shape = MagicMock()
    shape.Width = 96.0
    shape.Height = 48.0
    shape.Left = 12.0
    shape.Top = 24.0
    shape.LockAspectRatio = 1
    image = MagicMock()
    target = tmp_path / f"chart.{suffix}"

    with (
        patch("time.sleep"),
        patch("PIL.ImageGrab.grabclipboard", return_value=image) as clipboard,
    ):
        main._export_shape_highres(shape, str(target), 300)

    shape.CopyPicture.assert_called_once_with(1, 2)
    clipboard.assert_called_once_with()
    image.save.assert_called_once_with(str(target), expected_format, **expected_kwargs)
    assert shape.Width == 96.0
    assert shape.Height == 48.0
    assert shape.Left == 12.0
    assert shape.Top == 24.0
    assert shape.LockAspectRatio == 1


def test_windows_highres_jpeg_converts_rgb_and_restores_on_save_failure(tmp_path):
    shape = MagicMock()
    shape.Width, shape.Height = 96.0, 48.0
    shape.Left, shape.Top, shape.LockAspectRatio = 12.0, 24.0, 1
    image = MagicMock()
    rgb = image.convert.return_value
    rgb.save.side_effect = OSError("disk full")
    target = tmp_path / "chart.jpg"

    with (
        patch("time.sleep"),
        patch("PIL.ImageGrab.grabclipboard", return_value=image),
        pytest.raises(OSError, match="disk full"),
    ):
        main._export_shape_highres(shape, str(target), 600)

    shape.CopyPicture.assert_called_once_with(1, 2)
    image.convert.assert_called_once_with("RGB")
    rgb.save.assert_called_once_with(str(target), "JPEG", dpi=(600, 600), quality=95)
    assert (shape.Width, shape.Height, shape.Left, shape.Top) == (
        96.0,
        48.0,
        12.0,
        24.0,
    )
    assert shape.LockAspectRatio == 1


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


def _generation_host(tmp_path, frame, events):
    pictures = MagicMock()
    pictures.__iter__.return_value = iter([])

    def add_picture(_figure, **kwargs):
        # Model Excel assigning a final name that differs from the request.
        actual_name = f"{kwargs['name']}_Final"
        events.append(("add", actual_name))
        return SimpleNamespace(name=actual_name, height=100)

    pictures.add.side_effect = add_picture
    cell = MagicMock(left=10.0, top=20.0)
    sheet = MagicMock(name="Analysis", pictures=pictures)
    sheet.name = "Analysis"
    sheet.pictures = pictures
    sheet.range.return_value = cell
    selection = MagicMock(sheet=sheet, row=1, column=1)
    selection.sheet = sheet
    selection.row = 1
    selection.column = 1
    selection.rows.count = len(frame)
    selection.options.return_value.value = frame.copy()
    book = MagicMock(fullname=str(tmp_path / "experiment.xlsx"), path=str(tmp_path))
    book.fullname = str(tmp_path / "experiment.xlsx")
    book.path = str(tmp_path)
    book.selection = selection
    return book, sheet, pictures


@pytest.mark.parametrize(
    "entry",
    ["run", "quick", "preset", "wb_labeled", "qpcr_labeled", "elisa", "standard"],
)
def test_generation_entries_register_after_add_and_rebuild(tmp_path, entry):
    frame = pd.DataFrame({"Control": [1.0, 1.2], "Drug": [2.0, 2.2]})
    std_frame = pd.DataFrame({"0": [0.1, 0.11], "1": [0.4, 0.41]})
    events = []
    book, sheet, pictures = _generation_host(
        tmp_path, std_frame if entry in {"elisa", "standard"} else frame, events
    )
    config = PrismConfig(output_stats=False, output_data=False)
    stats = StatsResult(decision_path="test", normality_test="none")
    fit = fit_standard_curve(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.1, 0.11, 0.4, 0.41]),
        method="linear",
    )
    source_figures = []

    def new_figure(*_args, **_kwargs):
        fig = plt.figure()
        source_figures.append(fig)
        return fig

    original_register = main._register_artifact_best_effort

    def register_after_add(*args, **kwargs):
        picture = args[2]
        events.append(("register", picture.name))
        return original_register(*args, **kwargs)

    plotter = MagicMock()
    plotter.plot.side_effect = new_figure
    analyzer = MagicMock()
    analyzer.analyze.return_value = stats
    common = [
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch(
            "xstars.main._register_artifact_best_effort", side_effect=register_after_add
        ),
        patch("xstars.main.StatsEngine", return_value=analyzer),
        patch("xstars.main.PlotEngine", return_value=plotter),
        patch("xstars.main.DataHandler.validate"),
    ]
    for item in common:
        item.start()
    try:
        if entry == "run":
            with (
                patch("xstars.main._read_selection_auto", return_value=(None, frame)),
                patch("xstars.main.SettingsDialog") as dialog,
                patch("xstars.main._apply_preset", return_value=frame),
            ):
                dialog.return_value.show.return_value = config
                main._run_impl(book)
        elif entry == "quick":
            with (
                patch("xstars.main.PrismConfig.load", return_value=config),
                patch("xstars.main._read_selection_auto", return_value=(None, frame)),
                patch("xstars.main._apply_preset", return_value=frame),
            ):
                main._run_quick_impl(book)
        elif entry == "preset":
            config.experiment_preset = ExperimentPreset.WB
            with (
                patch("xstars.main._read_selection_auto", return_value=(None, frame)),
                patch("xstars.main.PrismConfig.load", return_value=PrismConfig()),
                patch("xstars.main.SettingsDialog") as dialog,
                patch("xstars.main._apply_preset", return_value=frame),
                patch("xstars.main.get_preset", return_value=object()),
            ):
                dialog.return_value.show.return_value = config
                main._run_preset_impl(book, ExperimentPreset.WB)
        elif entry in {"wb_labeled", "qpcr_labeled"}:
            handler = MagicMock()
            handler.get_insertion_cell.return_value = (5, 1)
            labels = pd.Series(["target", "reference"])
            transform_path = (
                "xstars.presets.wb.WBPreset.transform_labeled"
                if entry == "wb_labeled"
                else "xstars.presets.qpcr.QPCRPreset.transform_labeled"
            )
            runner = (
                main._run_wb_labeled
                if entry == "wb_labeled"
                else main._run_qpcr_labeled
            )
            with patch(transform_path, return_value=[("Target", frame)]):
                runner(book, sheet, handler, labels, frame, config)
        elif entry == "elisa":
            config.experiment_preset = ExperimentPreset.ELISA
            result = SimpleNamespace(fit_result=fit, config=config, show_fit_curve=True)
            with (
                patch("xstars.presets.elisa_dialog.ELISADialog") as dialog,
                patch("xstars.main._select_sample_data", return_value=frame),
                patch(
                    "xstars.main.artifacts.build_standard_curve_figure",
                    side_effect=lambda *_args, **_kwargs: new_figure(),
                ),
            ):
                dialog.return_value.show.return_value = result
                main._run_elisa_impl(book)
        else:
            result = SimpleNamespace(fit_result=fit, back_calculate=False)
            with (
                patch(
                    "xstars.tools.standard_curve_dialog.StandardCurveDialog"
                ) as dialog,
                patch("xstars.main.PrismConfig.load", return_value=config),
                patch(
                    "xstars.main.artifacts.build_standard_curve_figure",
                    side_effect=lambda *_args, **_kwargs: new_figure(),
                ),
            ):
                dialog.return_value.show.return_value = result
                main._run_standard_curve_impl(book)
    finally:
        for item in reversed(common):
            item.stop()

    assert events
    assert len(events) % 2 == 0
    assert all(
        events[index][0] == "add"
        and events[index + 1] == ("register", events[index][1])
        for index in range(0, len(events), 2)
    )
    payload_files = [
        path for path in tmp_path.glob("*.json") if path.name != "manifest.json"
    ]
    assert len(payload_files) == len(events) // 2
    for _, picture_name in events[::2]:
        identity = ArtifactIdentity(
            workbook=f"path:{book.fullname}", sheet="Analysis", picture=picture_name
        )
        payload = load_artifact(identity, tmp_path)
        rebuilt = artifacts.rebuild_figure(payload)
        plt.close(rebuilt)
    for figure in source_figures:
        plt.close(figure)
    assert pictures.add.call_count == len(events) // 2
    assert all(
        add_call.kwargs["name"] != actual_name
        for add_call, (_, actual_name) in zip(
            pictures.add.call_args_list, events[::2], strict=True
        )
    )


def test_full_preset_registration_failure_preserves_outputs_picture_and_status(
    tmp_path,
):
    frame = pd.DataFrame({"Control": [1.0, 1.2], "Drug": [2.0, 2.2]})
    events = []
    book, sheet, pictures = _generation_host(tmp_path, frame, events)
    config = PrismConfig(
        experiment_preset=ExperimentPreset.WB,
        output_stats=True,
        output_data=True,
    )
    stats = StatsResult(decision_path="test", normality_test="none")
    plotter = MagicMock()
    plotter.plot.return_value = plt.figure()
    analyzer = MagicMock()
    analyzer.analyze.return_value = stats
    real_write = artifacts._atomic_write_json

    def fail_payload_write(path, document):
        if Path(path).name != "manifest.json":
            raise PermissionError("read-only artifact directory")
        return real_write(path, document)

    with (
        patch.object(artifacts, "DEFAULT_ARTIFACT_ROOT", tmp_path),
        patch("xstars.main._read_selection_auto", return_value=(None, frame)),
        patch("xstars.main.PrismConfig.load", return_value=PrismConfig()),
        patch("xstars.main.SettingsDialog") as dialog,
        patch("xstars.main._apply_preset", return_value=frame),
        patch("xstars.main.get_preset", return_value=object()),
        patch("xstars.main.DataHandler.validate"),
        patch("xstars.main.StatsEngine", return_value=analyzer),
        patch("xstars.main.PlotEngine", return_value=plotter),
        patch(
            "xstars.main._write_transformed_data", wraps=main._write_transformed_data
        ) as write_data,
        patch("xstars.artifacts._atomic_write_json", side_effect=fail_payload_write),
    ):
        dialog.return_value.show.return_value = config
        main._run_preset_impl(book, ExperimentPreset.WB)

    pictures.add.assert_called_once()
    write_data.assert_called_once()
    assert write_data.call_args.args[4] == "Processed Data"
    assert write_data.call_args.args[3].equals(frame)
    assert sheet.range.call_count >= 4
    assert book.app.status_bar == "XSTARS: test"
    diagnostic = main.get_last_artifact_diagnostic()
    assert diagnostic is not None
    assert "ArtifactWriteError" in diagnostic
    assert "read-only artifact directory" in diagnostic
    plt.close(plotter.plot.return_value)


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
