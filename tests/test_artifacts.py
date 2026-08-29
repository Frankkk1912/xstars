"""Core M1 tests for safe rebuild-artifact persistence and rendering."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from xstars import artifacts
from xstars.artifacts import (
    ArtifactIdentity,
    ArtifactWriteError,
    CorruptArtifactError,
    MissingArtifactError,
    RendererKind,
    UnsupportedRendererError,
    UnsupportedSchemaError,
    build_payload,
    load_artifact,
    rebuild_figure,
    save_artifact,
    standard_curve_renderer_params,
)
from xstars.config import ChartType, ErrorBarType, ExperimentPreset, PrismConfig
from xstars.stats_engine import PairResult, StatsResult
from xstars.tools.standard_curve import fit_standard_curve


@pytest.fixture
def identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        workbook="path:/tmp/example.xlsx",
        sheet="Analysis",
        picture="XSTARS_Plot_1",
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Control": [1.0, np.nan, 3.0],
            "Treatment": [2.0, 4.0, 6.0],
        },
        index=pd.Index([10, 11, 12], name="replicate"),
    )


@pytest.fixture
def stats_result() -> StatsResult:
    return StatsResult(
        decision_path="Welch's t-test",
        normality_test="Shapiro-Wilk",
        normality_pvalues={"Control": 0.9, "Treatment": 0.8},
        all_normal=True,
        variance_test="Levene",
        variance_p=0.02,
        equal_variance=False,
        pairs=[
            PairResult(
                group_a="Control",
                group_b="Treatment",
                test_name="Welch's t-test",
                statistic=2.5,
                p_value=0.03,
                stars="*",
            )
        ],
    )


def _saved_document(root: Path, identity: ArtifactIdentity) -> tuple[Path, dict]:
    path = root / f"{identity.key}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rewrite_document(root: Path, identity: ArtifactIdentity, document: dict) -> None:
    path = root / f"{identity.key}.json"
    path.write_text(json.dumps(document), encoding="utf-8")


def test_round_trip_preserves_dataframe_config_stats_and_checksum(
    tmp_path, identity, frame, stats_result
):
    config = PrismConfig(
        chart_type=ChartType.LINE,
        error_bar=ErrorBarType.SD,
        experiment_preset=ExperimentPreset.WB,
        control_group="Control",
        export_path=str(tmp_path / "not-persisted-by-settings.json"),
        title="Round trip",
    )
    payload = build_payload(identity, frame, config, stats_result)

    artifact_path = save_artifact(payload, tmp_path)
    loaded = load_artifact(identity, tmp_path)

    assert artifact_path.parent == tmp_path
    assert artifact_path.name == f"{identity.key}.json"
    assert list(loaded.dataframe.columns) == ["Control", "Treatment"]
    assert loaded.dataframe.index.tolist() == [10, 11, 12]
    assert loaded.dataframe.index.name == "replicate"
    assert np.isnan(loaded.dataframe.loc[11, "Control"])
    assert loaded.config.chart_type is ChartType.LINE
    assert loaded.config.error_bar is ErrorBarType.SD
    assert loaded.config.experiment_preset is ExperimentPreset.WB
    assert loaded.config.control_group == "Control"
    assert loaded.config.export_path == str(tmp_path / "not-persisted-by-settings.json")
    assert loaded.stats_result is not None
    assert loaded.stats_result.decision_path == stats_result.decision_path
    assert loaded.stats_result.pairs == stats_result.pairs
    assert len(loaded.checksum) == 64

    _, document = _saved_document(tmp_path, identity)
    assert document["schema_version"] == artifacts.SCHEMA_VERSION
    assert document["artifact_key"] == identity.key
    assert document["checksum"] == loaded.checksum
    assert "pickle" not in artifact_path.read_text(encoding="utf-8").lower()


def test_missing_artifact_has_regeneration_message(tmp_path, identity):
    with pytest.raises(MissingArtifactError) as exc_info:
        load_artifact(identity, tmp_path)
    assert "regenerate" in exc_info.value.user_message.lower()


@pytest.mark.parametrize("field", ["config", "dataframe", "renderer", "identity"])
def test_missing_required_field_is_rejected(
    tmp_path, identity, frame, stats_result, field
):
    save_artifact(build_payload(identity, frame, PrismConfig(), stats_result), tmp_path)
    _, document = _saved_document(tmp_path, identity)
    document.pop(field)
    _rewrite_document(tmp_path, identity, document)

    with pytest.raises(CorruptArtifactError):
        load_artifact(identity, tmp_path)


def test_unsupported_schema_is_distinct(tmp_path, identity, frame):
    save_artifact(build_payload(identity, frame, PrismConfig()), tmp_path)
    _, document = _saved_document(tmp_path, identity)
    document["schema_version"] = artifacts.SCHEMA_VERSION + 1
    _rewrite_document(tmp_path, identity, document)

    with pytest.raises(UnsupportedSchemaError) as exc_info:
        load_artifact(identity, tmp_path)
    assert "regenerate" in exc_info.value.user_message.lower()


def test_unknown_renderer_is_distinct(tmp_path, identity, frame):
    save_artifact(build_payload(identity, frame, PrismConfig()), tmp_path)
    _, document = _saved_document(tmp_path, identity)
    document["renderer"]["kind"] = "unknown_renderer"
    old_checksum = document["checksum"]
    document_without_checksum = dict(document)
    document_without_checksum.pop("checksum")
    document["checksum"] = artifacts._document_checksum(document_without_checksum)
    _rewrite_document(tmp_path, identity, document)

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["artifacts"][identity.key]
    entry["renderer_kind"] = "unknown_renderer"
    entry["checksum"] = document["checksum"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert document["checksum"] != old_checksum

    with pytest.raises(UnsupportedRendererError):
        load_artifact(identity, tmp_path)


def test_corrupt_json_and_checksum_are_rejected(tmp_path, identity, frame):
    save_artifact(build_payload(identity, frame, PrismConfig()), tmp_path)
    artifact_path, _ = _saved_document(tmp_path, identity)
    artifact_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CorruptArtifactError):
        load_artifact(identity, tmp_path)

    save_artifact(build_payload(identity, frame, PrismConfig()), tmp_path)
    _, document = _saved_document(tmp_path, identity)
    document["config"]["title"] = "tampered"
    _rewrite_document(tmp_path, identity, document)
    with pytest.raises(CorruptArtifactError, match="checksum"):
        load_artifact(identity, tmp_path)


def test_atomic_replace_is_used_and_leaves_no_temp_files(tmp_path, identity, frame):
    payload = build_payload(identity, frame, PrismConfig())
    with patch("xstars.artifacts.os.replace", wraps=os.replace) as replace:
        save_artifact(payload, tmp_path)

    assert replace.call_count == 2
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_replace_is_wrapped_and_temp_file_removed(
    tmp_path, identity, frame
):
    payload = build_payload(identity, frame, PrismConfig())
    with (
        patch("xstars.artifacts.os.replace", side_effect=PermissionError("denied")),
        pytest.raises(ArtifactWriteError, match="denied"),
    ):
        save_artifact(payload, tmp_path)

    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))
    assert not (tmp_path / f"{identity.key}.json").exists()


def test_registration_failure_is_best_effort_and_diagnostic(
    tmp_path, frame, stats_result
):
    from xstars.main import (
        _register_artifact_best_effort,
        get_last_artifact_diagnostic,
    )

    book = MagicMock()
    book.path = str(tmp_path)
    book.fullname = str(tmp_path / "example.xlsx")
    sheet = MagicMock()
    sheet.name = "Analysis"
    picture = MagicMock()
    picture.name = "XSTARS_Plot_1"

    with patch(
        "xstars.main.artifacts.save_artifact",
        side_effect=PermissionError("read-only artifact directory"),
    ):
        result = _register_artifact_best_effort(
            book,
            sheet,
            picture,
            "XSTARS_Plot_1",
            frame,
            PrismConfig(),
            stats_result,
        )

    assert result is False
    diagnostic = get_last_artifact_diagnostic()
    assert diagnostic is not None
    assert "PermissionError" in diagnostic
    assert "read-only artifact directory" in diagnostic


def test_plot_engine_rebuild_exports_nonempty_figure(
    tmp_path, identity, frame, stats_result
):
    save_artifact(build_payload(identity, frame, PrismConfig(), stats_result), tmp_path)
    loaded = load_artifact(identity, tmp_path)
    fig = rebuild_figure(loaded)
    output = tmp_path / "plot.png"
    fig.savefig(output, dpi=100)
    plt.close(fig)

    assert output.stat().st_size > 0


def test_standard_curve_round_trip_rebuild_exports_nonempty_figure(tmp_path, frame):
    concentrations = np.array([0.0, 1.0, 2.0, 3.0])
    od = np.array([0.1, 0.6, 1.1, 1.6])
    fit = fit_standard_curve(concentrations, od, method="linear")
    identity = ArtifactIdentity(
        workbook="path:/tmp/example.xlsx",
        sheet="Analysis",
        picture="XSTARS_StdCurve_1",
    )
    params = standard_curve_renderer_params(concentrations, od, fit)
    payload = build_payload(
        identity,
        frame,
        PrismConfig(),
        renderer_kind=RendererKind.STANDARD_CURVE,
        renderer_params=params,
    )
    save_artifact(payload, tmp_path)

    loaded = load_artifact(identity, tmp_path)
    assert loaded.renderer_kind is RendererKind.STANDARD_CURVE
    assert loaded.renderer_params is not None
    assert loaded.renderer_params["fit"]["method"] == "linear"
    fig = rebuild_figure(loaded)
    output = tmp_path / "standard-curve.svg"
    fig.savefig(output)
    plt.close(fig)

    assert output.stat().st_size > 0
