"""Tests for persistent WPS render payloads and controlled export."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from unittest import mock

import matplotlib
from PIL import Image

matplotlib.use("Agg")

from xstars.application.analysis import analyze_dataframe
from xstars.application.contracts import ContractError, ErrorCode
from xstars.config import PrismConfig
from xstars.data_handler import DataHandler

pytest = import_module("pytest")
_export = import_module("xstars.application.export")
PAYLOAD_SCHEMA_VERSION = _export.PAYLOAD_SCHEMA_VERSION
export_clipboard_image = _export.export_clipboard_image
load_render_payload = _export.load_render_payload
new_picture_id = _export.new_picture_id
persist_render_payload = _export.persist_render_payload
render_payload_export = _export.render_payload_export
validate_export_request = _export.validate_export_request


def _analysis():
    frame = DataHandler.from_values(
        [["Control", "Treatment"], [1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]
    )
    config = PrismConfig(fig_width=2.0, fig_height=1.5, output_stats=True)
    return frame, config, analyze_dataframe(frame, config)


def test_picture_id_and_payload_json_round_trip_are_bounded_and_pickle_free(tmp_path):
    picture_id = new_picture_id(datetime(2026, 8, 31, tzinfo=timezone.utc))
    frame, config, result = _analysis()
    path = persist_render_payload(
        picture_id,
        result.transformed_data,
        config,
        result.figure,
        artifacts_root=tmp_path / "artifacts",
    )

    assert picture_id.startswith("XSTARS_20260831_")
    assert path.suffix == ".json"
    assert b"pickle" not in path.read_bytes().lower()
    payload = load_render_payload(picture_id, artifacts_root=tmp_path / "artifacts")
    assert payload["schemaVersion"] == PAYLOAD_SCHEMA_VERSION
    assert payload["data"]["columns"] == ["Control", "Treatment"]
    assert payload["figure"] == {"widthInches": 2.0, "heightInches": 1.5}
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("image_format", ["png", "tiff", "jpg", "pdf"])
@pytest.mark.parametrize("dpi", [96, 300])
def test_payload_rerender_writes_four_formats_at_requested_dpi(
    tmp_path, image_format, dpi
):
    picture_id = "XSTARS_20260831_abcdef123456"
    _frame, config, result = _analysis()
    persist_render_payload(
        picture_id,
        result.transformed_data,
        config,
        result.figure,
        artifacts_root=tmp_path / "artifacts",
    )

    exported = render_payload_export(
        picture_id,
        image_format,
        dpi,
        artifacts_root=tmp_path / "artifacts",
        exports_root=tmp_path / "exports",
    )
    output = tmp_path / "exports" / f"{picture_id}_{dpi}dpi.{image_format}"
    assert exported == {
        "path": str(output.resolve()),
        "format": image_format,
        "dpi": dpi,
        "source": "render_payload",
    }
    assert output.stat().st_size > 100
    if image_format == "pdf":
        assert output.read_bytes().startswith(b"%PDF")
    else:
        with Image.open(output) as persisted:
            assert persisted.size == (2 * dpi, round(1.5 * dpi))
            stored = persisted.info.get("dpi")
            assert stored is not None
            assert stored[0] == pytest.approx(dpi, abs=1.0)
            assert stored[1] == pytest.approx(dpi, abs=1.0)


def test_missing_corrupt_and_version_mismatch_are_structured(tmp_path):
    root = tmp_path / "artifacts"
    picture_id = "XSTARS_20260831_abcdef123456"
    with pytest.raises(ContractError) as missing:
        load_render_payload(picture_id, artifacts_root=root)
    assert missing.value.code is ErrorCode.PAYLOAD_MISSING
    with pytest.raises(ContractError) as traversal:
        load_render_payload("../../settings", artifacts_root=root)
    assert traversal.value.code is ErrorCode.INVALID_REQUEST

    root.mkdir()
    path = root / f"{picture_id}.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ContractError) as corrupt:
        load_render_payload(picture_id, artifacts_root=root)
    assert corrupt.value.code is ErrorCode.PAYLOAD_CORRUPT

    path.write_text(
        json.dumps({"schemaVersion": "999", "pictureId": picture_id}),
        encoding="utf-8",
    )
    with pytest.raises(ContractError) as version:
        load_render_payload(picture_id, artifacts_root=root)
    assert version.value.code is ErrorCode.PAYLOAD_VERSION


@pytest.mark.parametrize(
    ("image_format", "dpi", "code"),
    [
        ("svg", 300, ErrorCode.EXPORT_FORMAT),
        ("png", 71, ErrorCode.EXPORT_DPI),
        ("png", 1201, ErrorCode.EXPORT_DPI),
        ("png", True, ErrorCode.EXPORT_DPI),
    ],
)
def test_export_validation_boundaries(image_format, dpi, code):
    with pytest.raises(ContractError) as caught:
        validate_export_request(image_format, dpi)
    assert caught.value.code is code
    assert validate_export_request("TIFF", 72) == ("tiff", 72)
    assert validate_export_request("png", 1200) == ("png", 1200)


def test_clipboard_bonus_path_is_mockable_and_sets_dpi(tmp_path):
    source = Image.new("RGBA", (96, 48), (255, 0, 0, 128))
    exported = export_clipboard_image(
        "jpg",
        300,
        exports_root=tmp_path,
        image_grabber=lambda: source,
    )
    output = __import__("pathlib").Path(exported["path"])
    with Image.open(output) as persisted:
        assert persisted.mode == "RGB"
        assert persisted.size == (300, 150)
        assert persisted.info["dpi"][0] == pytest.approx(300, abs=1.0)
    assert exported["source"] == "clipboard"


def test_payload_config_rejects_unknown_fields_instead_of_executing_them(tmp_path):
    picture_id = "XSTARS_20260831_abcdef123456"
    _frame, config, result = _analysis()
    path = persist_render_payload(
        picture_id,
        result.transformed_data,
        config,
        result.figure,
        artifacts_root=tmp_path / "artifacts",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["callable"] = "os.system('whoami')"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        render_payload_export(
            picture_id,
            "png",
            300,
            artifacts_root=tmp_path / "artifacts",
            exports_root=tmp_path / "exports",
        )
    assert caught.value.code is ErrorCode.PAYLOAD_CORRUPT

    payload["config"].pop("callable")
    payload["config"]["fig_width"] = "not-a-number"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError) as malformed:
        render_payload_export(
            picture_id,
            "png",
            300,
            artifacts_root=tmp_path / "artifacts",
            exports_root=tmp_path / "exports",
        )
    assert malformed.value.code is ErrorCode.PAYLOAD_CORRUPT


# ---------------------------------------------------------------------------
# New tests: user-chosen output path, cancellation, extension normalisation,
# and Tk-unavailable fallback.
# ---------------------------------------------------------------------------

_worker = import_module("xstars.application.worker")
_ask_export_path = _worker._ask_export_path


def _setup_artifacts(tmp_path: Path) -> str:
    """Persist a minimal render payload and return the picture_id."""
    picture_id = "XSTARS_20260831_abcdef123456"
    _frame, config, result = _analysis()
    persist_render_payload(
        picture_id,
        result.transformed_data,
        config,
        result.figure,
        artifacts_root=tmp_path / "artifacts",
    )
    return picture_id


def test_render_payload_export_writes_to_user_chosen_path(tmp_path):
    """output_path overrides the automatic naming scheme."""
    picture_id = _setup_artifacts(tmp_path)
    chosen = tmp_path / "my_export" / "result.png"
    exported = render_payload_export(
        picture_id,
        "png",
        300,
        artifacts_root=tmp_path / "artifacts",
        output_path=chosen,
    )
    assert Path(exported["path"]).resolve() == chosen.resolve()
    assert chosen.is_file()
    assert chosen.stat().st_size > 100
    assert exported["source"] == "render_payload"


def test_render_payload_export_appends_extension_when_mismatched(tmp_path):
    """If the user omits or mismatches the extension, the correct one is appended."""
    picture_id = _setup_artifacts(tmp_path)
    # Path without any extension
    stem_only = tmp_path / "noext"
    exported = render_payload_export(
        picture_id,
        "png",
        300,
        artifacts_root=tmp_path / "artifacts",
        output_path=stem_only,
    )
    result_path = Path(exported["path"])
    assert result_path.suffix == ".png"
    assert result_path.is_file()

    # Path with wrong extension (e.g. user chose .txt)
    wrong_ext = tmp_path / "chart.txt"
    exported2 = render_payload_export(
        picture_id,
        "tiff",
        150,
        artifacts_root=tmp_path / "artifacts",
        output_path=wrong_ext,
    )
    result_path2 = Path(exported2["path"])
    assert result_path2.suffix == ".tiff"
    assert result_path2.is_file()


def test_clipboard_export_writes_to_user_chosen_path(tmp_path):
    """output_path overrides automatic naming for clipboard export."""
    source = Image.new("RGB", (96, 48), (0, 128, 255))
    chosen = tmp_path / "clip_export.png"
    exported = export_clipboard_image(
        "png",
        150,
        image_grabber=lambda: source,
        output_path=chosen,
    )
    assert Path(exported["path"]).resolve() == chosen.resolve()
    assert chosen.is_file()
    assert exported["source"] == "clipboard"


def test_ask_export_path_returns_none_on_user_cancel(tmp_path):
    """A dialog that returns '' (user hit Cancel) maps to None."""
    result = _ask_export_path(
        "png",
        "my_stem",
        exports_root=tmp_path,
        _dialog_fn=lambda **kw: "",   # simulate Cancel
    )
    assert result is None


def test_ask_export_path_returns_path_on_confirm(tmp_path):
    """A dialog that returns a non-empty string becomes a Path."""
    chosen_str = str(tmp_path / "my_chart.png")
    result = _ask_export_path(
        "png",
        "my_stem",
        exports_root=tmp_path,
        _dialog_fn=lambda **kw: chosen_str,
    )
    assert result == Path(chosen_str)


def test_worker_export_cancelled_when_dialog_returns_none(tmp_path):
    """User cancel from save-as dialog → {cancelled: True} result, no file written."""
    from xstars.application.contracts import SCHEMA_VERSION

    _setup_artifacts(tmp_path / "artifacts" / "dummy")
    # We don't need real artifacts for cancel path; worker cancels before export.
    export_request = {
        "version": SCHEMA_VERSION,
        "command": "run_export",
        "config": {},
        "export": {
            "pictureId": "XSTARS_20260831_abcdef123456",
            "format": "png",
            "dpi": 300,
        },
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }
    with mock.patch(
        "xstars.application.export.validate_export_request", return_value=("png", 300)
    ):
        result = _worker.execute_request(
            export_request,
            tmp_path,
            ask_export_path=lambda **kw: None,   # simulate cancel
        )
    assert result["ok"] is True
    assert result["export"] == {"cancelled": True}
    assert "Exported" not in result["writebackPlan"]["statusMessage"]


def test_worker_export_fallback_to_auto_name_when_tk_unavailable(tmp_path):
    """When _ask_export_path raises (Tk unavailable), fallback to auto-naming."""
    from xstars.application.contracts import SCHEMA_VERSION

    picture_id = _setup_artifacts(tmp_path)
    auto_result = {
        "path": str(tmp_path / "auto.png"),
        "format": "png",
        "dpi": 300,
        "source": "render_payload",
    }
    fake_export_mod = mock.MagicMock()
    fake_export_mod.validate_export_request.return_value = ("png", 300)
    fake_export_mod.render_payload_export.return_value = auto_result

    export_request = {
        "version": SCHEMA_VERSION,
        "command": "run_export",
        "config": {},
        "export": {
            "pictureId": picture_id,
            "format": "png",
            "dpi": 300,
        },
        "cancelPath": str((tmp_path / "cancel").resolve()),
    }

    def _raising_ask(**kw):
        raise RuntimeError("Tk unavailable in headless CI")

    with (
        mock.patch.object(_worker.PrismConfig, "load", return_value=PrismConfig()),
        mock.patch.object(_worker, "import_module", return_value=fake_export_mod),
        mock.patch(
            "xstars.application.export.validate_export_request", return_value=("png", 300)
        ),
    ):
        result = _worker.execute_request(
            export_request,
            tmp_path,
            ask_export_path=_raising_ask,
        )

    assert result["ok"] is True
    assert result["export"]["path"] == auto_result["path"]
    assert "fallback" in result["export"]
    fake_export_mod.render_payload_export.assert_called_once()
    # output_path=None means auto-naming was used
    call_kwargs = fake_export_mod.render_payload_export.call_args.kwargs
    assert call_kwargs.get("output_path") is None
