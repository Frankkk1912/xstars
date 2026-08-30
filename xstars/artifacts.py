"""Safe, versioned rebuild artifacts for XSTARS-generated figures.

Artifacts are derived local cache data.  They contain the processed data and the
exact rendering inputs needed to rebuild a Matplotlib figure in a later
RunPython invocation, without reading pixels back from Excel.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AnnotationFormat,
    BaseTheme,
    ChartType,
    DoseAxisScale,
    ErrorBarType,
    ExperimentPreset,
    FitMethod,
    JournalPalette,
    JournalPreset,
    PalettePreset,
    PrismConfig,
)
from .plot_engine import PlotEngine
from .presets.cck8 import CCK8FitInfo
from .stats_engine import PairResult, StatsResult

SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path.home() / ".xstars" / "artifacts"
_MANIFEST_NAME = "manifest.json"
_SAFE_KEY = re.compile(r"^[0-9a-f]{64}$")
_LOGGER = logging.getLogger(__name__)


class RendererKind(str, Enum):
    """Figure builders supported by the artifact schema."""

    PLOT_ENGINE = "plot_engine"
    STANDARD_CURVE = "standard_curve"


class ArtifactError(RuntimeError):
    """Base error for artifact persistence and validation failures."""

    recovery_message = "Please regenerate the chart and try again."

    @property
    def user_message(self) -> str:
        return f"{self} {self.recovery_message}".strip()


class MissingArtifactError(ArtifactError):
    """No registered artifact exists for the requested Excel picture."""


class CorruptArtifactError(ArtifactError):
    """An artifact or manifest is malformed or fails integrity checks."""


class UnsupportedSchemaError(ArtifactError):
    """The artifact was written by an unsupported schema version."""


class UnsupportedRendererError(ArtifactError):
    """The artifact names a renderer this XSTARS version cannot rebuild."""


class ArtifactIdentityError(ArtifactError):
    """The artifact belongs to a different workbook, sheet, or picture."""


class ArtifactWriteError(ArtifactError):
    """The artifact could not be serialized or persisted."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable inputs used to associate one payload with one Excel picture."""

    workbook: str
    sheet: str
    picture: str

    def __post_init__(self) -> None:
        for label, value in (
            ("workbook", self.workbook),
            ("sheet", self.sheet),
            ("picture", self.picture),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Artifact {label} identity must be a non-empty string"
                )

    @property
    def key(self) -> str:
        raw = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {
            "workbook": self.workbook,
            "sheet": self.sheet,
            "picture": self.picture,
        }


@dataclass
class ArtifactPayload:
    """Validated in-memory rebuild inputs."""

    identity: ArtifactIdentity
    renderer_kind: RendererKind
    dataframe: pd.DataFrame
    config: PrismConfig
    stats_result: StatsResult | None = None
    renderer_params: dict[str, Any] | None = None
    created_at: str = ""
    checksum: str = ""

    @property
    def artifact_key(self) -> str:
        return self.identity.key


def build_payload(
    identity: ArtifactIdentity,
    dataframe: pd.DataFrame,
    config: PrismConfig,
    stats_result: StatsResult | None = None,
    *,
    renderer_kind: RendererKind | str = RendererKind.PLOT_ENGINE,
    renderer_params: Mapping[str, Any] | None = None,
) -> ArtifactPayload:
    """Create a detached artifact payload from one successful chart insertion."""
    try:
        kind = RendererKind(renderer_kind)
    except ValueError as exc:
        raise UnsupportedRendererError(
            f"Unsupported artifact renderer: {renderer_kind}"
        ) from exc
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Artifact dataframe must be a pandas DataFrame")
    if not isinstance(config, PrismConfig):
        raise TypeError("Artifact config must be a PrismConfig")
    return ArtifactPayload(
        identity=identity,
        renderer_kind=kind,
        dataframe=dataframe.copy(deep=True),
        config=copy.deepcopy(config),
        stats_result=copy.deepcopy(stats_result),
        renderer_params=copy.deepcopy(dict(renderer_params or {})),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def invalidate_artifact(
    identity: ArtifactIdentity,
    root: Path | str | None = None,
) -> None:
    """Fail-closed invalidation before an identity is reused.

    The payload file is authoritative and is unlinked first.  The manifest is
    diagnostic metadata only, so a failed or racing manifest update cannot make
    an invalidated payload loadable again.
    """
    root_path = Path(root) if root is not None else DEFAULT_ARTIFACT_ROOT
    artifact_path = _artifact_path(root_path, identity.key)
    try:
        artifact_path.unlink(missing_ok=True)
    except OSError as exc:
        raise ArtifactWriteError(
            f"Could not invalidate the previous chart rebuild artifact: {exc}"
        ) from exc
    if (root_path / _MANIFEST_NAME).exists():
        _update_manifest_best_effort(root_path, identity.key, None)


def save_artifact(
    payload: ArtifactPayload,
    root: Path | str | None = None,
) -> Path:
    """Atomically replace a payload and best-effort diagnostic manifest entry.

    Any prior payload for the same identity is invalidated before serialization
    or writing.  Therefore a failed replacement cannot leave stale chart data
    loadable under a reused Excel picture name.
    """
    root_path = Path(root) if root is not None else DEFAULT_ARTIFACT_ROOT
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        # Some filesystems do not implement POSIX permissions.
        with suppress(OSError):
            root_path.chmod(0o700)

        invalidate_artifact(payload.identity, root_path)
        document = _payload_to_document(payload)
        checksum = _document_checksum(document)
        document["checksum"] = checksum
        payload.checksum = checksum

        artifact_path = _artifact_path(root_path, payload.artifact_key)
        _atomic_write_json(artifact_path, document)
        _update_manifest_best_effort(
            root_path,
            payload.artifact_key,
            {
                "file": artifact_path.name,
                "identity": payload.identity.to_dict(),
                "renderer_kind": payload.renderer_kind.value,
                "checksum": checksum,
            },
        )
        return artifact_path
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactWriteError(
            f"Could not save chart rebuild artifact: {exc}"
        ) from exc


def load_artifact(
    identity: ArtifactIdentity,
    root: Path | str | None = None,
) -> ArtifactPayload:
    """Load by deterministic key and validate the authoritative payload.

    ``manifest.json`` is intentionally not consulted: it is best-effort
    diagnostic metadata and may be missing or stale after concurrent writers.
    """
    root_path = Path(root) if root is not None else DEFAULT_ARTIFACT_ROOT
    artifact_path = _artifact_path(root_path, identity.key)
    if not artifact_path.exists():
        raise MissingArtifactError(
            "No rebuild information is registered for this chart."
        )
    document = _read_json(artifact_path, "chart rebuild artifact")
    return _document_to_payload(document, expected_identity=identity)


def has_artifact(identity: ArtifactIdentity, root: Path | str | None = None) -> bool:
    """Return True only when the matching artifact loads and validates fully."""
    try:
        load_artifact(identity, root)
    except ArtifactError:
        return False
    return True


def rebuild_figure(payload: ArtifactPayload):
    """Rebuild a Matplotlib Figure using the payload's declared renderer."""
    if payload.renderer_kind == RendererKind.PLOT_ENGINE:
        return PlotEngine(payload.config).plot(payload.dataframe, payload.stats_result)
    if payload.renderer_kind == RendererKind.STANDARD_CURVE:
        params = payload.renderer_params or {}
        required = {"concentrations", "od", "fit"}
        missing = sorted(required - params.keys())
        if missing:
            raise CorruptArtifactError(
                f"Standard-curve artifact is missing fields: {', '.join(missing)}."
            )
        return build_standard_curve_figure(
            params["concentrations"], params["od"], params["fit"], payload.config
        )
    raise UnsupportedRendererError(
        f"Unsupported artifact renderer: {payload.renderer_kind}"
    )


def curve_fit_snapshot(fit: Any) -> dict[str, Any]:
    """Return the executable-free fields needed to redraw a standard curve."""
    required = ("method", "params", "r_squared", "equation_str", "conc_range")
    if not all(hasattr(fit, field) for field in required):
        raise TypeError("Curve fit result is missing required fields")
    return {
        "method": str(fit.method),
        "params": _json_safe(dict(fit.params)),
        "r_squared": _json_safe(fit.r_squared),
        "equation_str": str(fit.equation_str),
        "conc_range": _json_safe(list(fit.conc_range)),
    }


def standard_curve_renderer_params(
    concentrations: Any,
    od: Any,
    fit: Any,
) -> dict[str, Any]:
    """Build JSON-safe dedicated parameters for the standard-curve renderer."""
    return {
        "concentrations": _json_safe(np.asarray(concentrations, dtype=float).tolist()),
        "od": _json_safe(np.asarray(od, dtype=float).tolist()),
        "fit": curve_fit_snapshot(fit),
    }


def build_standard_curve_figure(
    concentrations: Any,
    od: Any,
    fit_snapshot: Mapping[str, Any],
    config: PrismConfig,
):
    """Render the standard-curve figure used by generation and rebuild paths."""
    import matplotlib.pyplot as plt

    from .styles import get_prism_context

    conc = np.asarray(concentrations, dtype=float)
    od_values = np.asarray(od, dtype=float)
    fit = dict(fit_snapshot)
    method = fit.get("method")
    params = fit.get("params")
    if method not in {
        "four_pl",
        "three_pl",
        "linear",
        "log_linear_reg",
        "interpolation",
    }:
        raise UnsupportedRendererError(
            f"Unsupported standard-curve fit method: {method}"
        )
    if not isinstance(params, dict):
        raise CorruptArtifactError("Standard-curve fit parameters are invalid.")
    if len(conc) != len(od_values) or len(conc) < 2:
        raise CorruptArtifactError("Standard-curve data arrays are invalid.")

    with get_prism_context(config.journal_preset, config.base_theme):
        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=config.dpi)
        ax.scatter(
            conc,
            od_values,
            color=config.palette[0],
            s=30,
            zorder=5,
            label="Standards",
        )

        conc_pos = conc[conc > 0]
        cmin_pos = conc_pos.min() if len(conc_pos) > 0 else 1e-6
        cmax_pos = conc_pos.max() if len(conc_pos) > 0 else 1.0
        if method == "linear":
            x_fit = np.linspace(conc.min(), conc.max() * 1.1, 200)
        else:
            x_fit = np.geomspace(cmin_pos * 0.5, cmax_pos * 1.5, 200)
        y_fit = _predict_standard_curve(method, params, x_fit, conc, od_values)
        ax.plot(x_fit, y_fit, "-", color=config.palette[1], linewidth=1.5, label=method)

        use_log = len(conc_pos) >= 2 and cmax_pos / cmin_pos > 10
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("OD")
        r_squared = _json_restore(fit.get("r_squared"))
        if r_squared is not None:
            ax.set_title(f"Standard Curve (R² = {float(r_squared):.4f})")
        else:
            ax.set_title("Standard Curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def _predict_standard_curve(
    method: str,
    params: dict[str, Any],
    x_values: np.ndarray,
    concentrations: np.ndarray,
    od: np.ndarray,
) -> np.ndarray:
    restored = {key: _json_restore(value) for key, value in params.items()}
    try:
        if method == "linear":
            return restored["slope"] * x_values + restored["intercept"]
        if method == "log_linear_reg":
            return restored["slope"] * np.log10(x_values) + restored["intercept"]
        if method == "four_pl":
            from .tools.standard_curve import four_param_logistic

            return four_param_logistic(
                x_values,
                restored["bottom"],
                restored["top"],
                restored["ec50"],
                restored["hill"],
            )
        if method == "three_pl":
            return restored["bottom"] + (restored["top"] - restored["bottom"]) / (
                1.0 + (x_values / restored["ec50"]) ** restored["hill"]
            )
        if method == "interpolation":
            from .tools.standard_curve import fit_standard_curve

            refit = fit_standard_curve(concentrations, od, method="interpolation")
            return np.asarray(refit.predict(x_values), dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptArtifactError(
            "Standard-curve fit parameters are invalid."
        ) from exc
    raise UnsupportedRendererError(f"Unsupported standard-curve fit method: {method}")


def _payload_to_document(payload: ArtifactPayload) -> dict[str, Any]:
    if not isinstance(payload, ArtifactPayload):
        raise TypeError("save_artifact expects an ArtifactPayload")
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_key": payload.artifact_key,
        "identity": payload.identity.to_dict(),
        "renderer": {
            "kind": payload.renderer_kind.value,
            "params": _json_safe(payload.renderer_params or {}),
        },
        "dataframe": _dataframe_to_dict(payload.dataframe),
        "config": _config_to_dict(payload.config),
        "stats_result": _stats_to_dict(payload.stats_result),
        "created_at": payload.created_at or datetime.now(timezone.utc).isoformat(),
    }
    _validate_document_structure(document, check_checksum=False)
    return document


def _document_to_payload(
    document: Any,
    *,
    expected_identity: ArtifactIdentity,
) -> ArtifactPayload:
    _validate_document_structure(document, check_checksum=True)
    version = document["schema_version"]
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"Artifact schema {version!r} is not supported by this XSTARS version."
        )

    supplied_checksum = document["checksum"]
    without_checksum = dict(document)
    without_checksum.pop("checksum")
    if not _constant_time_equal(
        supplied_checksum, _document_checksum(without_checksum)
    ):
        raise CorruptArtifactError("The chart rebuild artifact checksum is invalid.")

    try:
        identity = ArtifactIdentity(**document["identity"])
    except (TypeError, ValueError) as exc:
        raise CorruptArtifactError("The artifact identity is invalid.") from exc
    if (
        identity != expected_identity
        or document["artifact_key"] != expected_identity.key
    ):
        raise ArtifactIdentityError(
            "The rebuild artifact belongs to a different chart."
        )

    renderer = document["renderer"]
    try:
        renderer_kind = RendererKind(renderer["kind"])
    except ValueError as exc:
        raise UnsupportedRendererError(
            f"Unsupported artifact renderer: {renderer.get('kind')}"
        ) from exc

    payload = ArtifactPayload(
        identity=identity,
        renderer_kind=renderer_kind,
        dataframe=_dataframe_from_dict(document["dataframe"]),
        config=_config_from_dict(document["config"]),
        stats_result=_stats_from_dict(document["stats_result"]),
        renderer_params=_json_restore(renderer["params"]),
        created_at=document["created_at"],
        checksum=supplied_checksum,
    )
    if renderer_kind == RendererKind.STANDARD_CURVE:
        required = {"concentrations", "od", "fit"}
        if not isinstance(payload.renderer_params, dict) or not required.issubset(
            payload.renderer_params
        ):
            raise CorruptArtifactError(
                "Standard-curve renderer parameters are incomplete."
            )
    return payload


def _validate_document_structure(document: Any, *, check_checksum: bool) -> None:
    if not isinstance(document, dict):
        raise CorruptArtifactError("The chart rebuild artifact must be a JSON object.")
    required = {
        "schema_version",
        "artifact_key",
        "identity",
        "renderer",
        "dataframe",
        "config",
        "stats_result",
        "created_at",
    }
    if check_checksum:
        required.add("checksum")
    missing = sorted(required - document.keys())
    if missing:
        raise CorruptArtifactError(
            f"The chart rebuild artifact is missing fields: {', '.join(missing)}."
        )
    if type(document["schema_version"]) is not int:
        raise CorruptArtifactError("The artifact schema version is invalid.")
    if document["schema_version"] != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"Artifact schema {document['schema_version']!r} is not supported by this XSTARS version."
        )
    key = document["artifact_key"]
    if not isinstance(key, str) or not _SAFE_KEY.fullmatch(key):
        raise CorruptArtifactError("The artifact key is invalid.")
    if not isinstance(document["identity"], dict):
        raise CorruptArtifactError("The artifact identity is invalid.")
    if not isinstance(document["renderer"], dict):
        raise CorruptArtifactError("The artifact renderer is invalid.")
    if {"kind", "params"} - document["renderer"].keys():
        raise CorruptArtifactError("The artifact renderer is incomplete.")
    if not isinstance(document["created_at"], str) or not document["created_at"]:
        raise CorruptArtifactError("The artifact creation timestamp is invalid.")
    if check_checksum:
        checksum = document["checksum"]
        if not isinstance(checksum, str) or not _SAFE_KEY.fullmatch(checksum):
            raise CorruptArtifactError("The artifact checksum is invalid.")


def _config_to_dict(config: PrismConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "elisa_fit_result" and value is not None:
            value = (
                dict(value) if isinstance(value, Mapping) else curve_fit_snapshot(value)
            )
        snapshot[field.name] = _json_safe(value)
    return snapshot


def _config_from_dict(snapshot: Any) -> PrismConfig:
    if not isinstance(snapshot, dict):
        raise CorruptArtifactError("The artifact plot configuration is invalid.")
    expected_fields = {field.name for field in fields(PrismConfig)}
    missing = sorted(expected_fields - snapshot.keys())
    if missing:
        raise CorruptArtifactError(
            f"The artifact plot configuration is missing fields: {', '.join(missing)}."
        )

    enum_fields = {
        "chart_type": ChartType,
        "error_bar": ErrorBarType,
        "annotation_format": AnnotationFormat,
        "journal_preset": JournalPreset,
        "base_theme": BaseTheme,
        "palette_preset": PalettePreset,
        "journal_palette": JournalPalette,
        "experiment_preset": ExperimentPreset,
        "preset_dose_axis_scale": DoseAxisScale,
        "preset_fit_method": FitMethod,
    }
    kwargs: dict[str, Any] = {}
    for name in expected_fields:
        value = _json_restore(snapshot[name])
        if name in enum_fields:
            try:
                value = enum_fields[name](value)
            except ValueError as exc:
                raise CorruptArtifactError(
                    f"The artifact plot configuration has invalid {name}."
                ) from exc
        elif name == "ic50_fit_info" and value is not None:
            if not isinstance(value, dict):
                raise CorruptArtifactError(
                    "The artifact IC50 fit configuration is invalid."
                )
            try:
                value = CCK8FitInfo(**value)
            except TypeError as exc:
                raise CorruptArtifactError(
                    "The artifact IC50 fit configuration is incomplete."
                ) from exc
        kwargs[name] = value
    try:
        return PrismConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CorruptArtifactError(
            "The artifact plot configuration is invalid."
        ) from exc


def _stats_to_dict(stats_result: StatsResult | None) -> dict[str, Any] | None:
    if stats_result is None:
        return None
    if not isinstance(stats_result, StatsResult):
        raise TypeError("Artifact stats_result must be a StatsResult or None")
    return {
        "decision_path": stats_result.decision_path,
        "normality_test": stats_result.normality_test,
        "normality_pvalues": _json_safe(stats_result.normality_pvalues),
        "all_normal": stats_result.all_normal,
        "variance_test": stats_result.variance_test,
        "variance_p": _json_safe(stats_result.variance_p),
        "equal_variance": stats_result.equal_variance,
        "omnibus_test": stats_result.omnibus_test,
        "omnibus_statistic": _json_safe(stats_result.omnibus_statistic),
        "omnibus_p": _json_safe(stats_result.omnibus_p),
        "pairs": [
            {
                "group_a": pair.group_a,
                "group_b": pair.group_b,
                "test_name": pair.test_name,
                "statistic": _json_safe(pair.statistic),
                "p_value": _json_safe(pair.p_value),
                "stars": pair.stars,
            }
            for pair in stats_result.pairs
        ],
    }


def _stats_from_dict(snapshot: Any) -> StatsResult | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise CorruptArtifactError("The artifact statistics payload is invalid.")
    required = {
        "decision_path",
        "normality_test",
        "normality_pvalues",
        "all_normal",
        "variance_test",
        "variance_p",
        "equal_variance",
        "omnibus_test",
        "omnibus_statistic",
        "omnibus_p",
        "pairs",
    }
    missing = sorted(required - snapshot.keys())
    if missing:
        raise CorruptArtifactError(
            f"The artifact statistics payload is missing fields: {', '.join(missing)}."
        )
    if not isinstance(snapshot["pairs"], list):
        raise CorruptArtifactError("The artifact statistics pairs are invalid.")
    try:
        pairs = [
            PairResult(
                group_a=pair["group_a"],
                group_b=pair["group_b"],
                test_name=pair["test_name"],
                statistic=_json_restore(pair["statistic"]),
                p_value=_json_restore(pair["p_value"]),
                stars=pair["stars"],
            )
            for pair in snapshot["pairs"]
        ]
        return StatsResult(
            decision_path=snapshot["decision_path"],
            normality_test=snapshot["normality_test"],
            normality_pvalues=_json_restore(snapshot["normality_pvalues"]),
            all_normal=snapshot["all_normal"],
            variance_test=snapshot["variance_test"],
            variance_p=_json_restore(snapshot["variance_p"]),
            equal_variance=snapshot["equal_variance"],
            omnibus_test=snapshot["omnibus_test"],
            omnibus_statistic=_json_restore(snapshot["omnibus_statistic"]),
            omnibus_p=_json_restore(snapshot["omnibus_p"]),
            pairs=pairs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptArtifactError(
            "The artifact statistics payload is invalid."
        ) from exc


def _dataframe_to_dict(dataframe: pd.DataFrame) -> dict[str, Any]:
    rows: list[list[Any]] = []
    missing: list[list[bool]] = []
    for row in dataframe.to_numpy(dtype=object).tolist():
        row_values: list[Any] = []
        row_missing: list[bool] = []
        for value in row:
            is_missing = bool(pd.isna(value))
            row_missing.append(is_missing)
            row_values.append(None if is_missing else _json_safe(value))
        rows.append(row_values)
        missing.append(row_missing)
    return {
        "columns": [_json_safe(column) for column in dataframe.columns.tolist()],
        "index": [_json_safe(value) for value in dataframe.index.tolist()],
        "index_name": _json_safe(dataframe.index.name),
        "data": rows,
        "missing": missing,
    }


def _dataframe_from_dict(snapshot: Any) -> pd.DataFrame:
    if not isinstance(snapshot, dict):
        raise CorruptArtifactError("The artifact DataFrame payload is invalid.")
    required = {"columns", "index", "index_name", "data", "missing"}
    missing_fields = sorted(required - snapshot.keys())
    if missing_fields:
        raise CorruptArtifactError(
            f"The artifact DataFrame is missing fields: {', '.join(missing_fields)}."
        )
    columns = _json_restore(snapshot["columns"])
    index = _json_restore(snapshot["index"])
    rows = _json_restore(snapshot["data"])
    missing = snapshot["missing"]
    if not all(isinstance(value, list) for value in (columns, index, rows, missing)):
        raise CorruptArtifactError("The artifact DataFrame payload is invalid.")
    if len(rows) != len(index) or len(missing) != len(rows):
        raise CorruptArtifactError("The artifact DataFrame dimensions are invalid.")
    width = len(columns)
    if any(len(row) != width for row in rows) or any(
        not isinstance(mask, list) or len(mask) != width for mask in missing
    ):
        raise CorruptArtifactError("The artifact DataFrame row dimensions are invalid.")
    for row_index, mask in enumerate(missing):
        for column_index, is_missing in enumerate(mask):
            if not isinstance(is_missing, bool):
                raise CorruptArtifactError(
                    "The artifact DataFrame missing mask is invalid."
                )
            if is_missing:
                rows[row_index][column_index] = np.nan
    try:
        frame = pd.DataFrame(rows, columns=columns, index=index)
        frame.index.name = _json_restore(snapshot["index_name"])
        return frame
    except (TypeError, ValueError) as exc:
        raise CorruptArtifactError(
            "The artifact DataFrame payload is invalid."
        ) from exc


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"__xstars_float__": "nan"}
        if math.isinf(number):
            return {"__xstars_float__": "inf" if number > 0 else "-inf"}
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Artifact JSON object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-safe")


def _json_restore(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__xstars_float__"}:
            marker = value["__xstars_float__"]
            if marker == "nan":
                return float("nan")
            if marker == "inf":
                return float("inf")
            if marker == "-inf":
                return float("-inf")
            raise CorruptArtifactError("The artifact contains an invalid float marker.")
        return {key: _json_restore(item) for key, item in value.items()}
    return value


def _document_checksum(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left, right)


def _artifact_path(root: Path, key: str) -> Path:
    if not isinstance(key, str) or not _SAFE_KEY.fullmatch(key):
        raise CorruptArtifactError("The artifact key is not path-safe.")
    return root / f"{key}.json"


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MissingArtifactError(f"The {label} is missing.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptArtifactError(f"The {label} cannot be read: {exc}") from exc


def _read_manifest_for_update(root: Path) -> dict[str, Any]:
    path = root / _MANIFEST_NAME
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "artifacts": {}}
    manifest = _read_json(path, "artifact manifest")
    _validate_manifest(manifest)
    return manifest


def _update_manifest_best_effort(
    root: Path, key: str, entry: dict[str, Any] | None
) -> None:
    """Update diagnostic metadata without affecting payload availability."""
    try:
        manifest = _read_manifest_for_update(root)
        if entry is None:
            if key not in manifest["artifacts"]:
                return
            manifest["artifacts"].pop(key)
        else:
            manifest["artifacts"][key] = entry
        _atomic_write_json(root / _MANIFEST_NAME, manifest)
    except Exception as exc:
        _LOGGER.warning("Artifact manifest update failed (payload unaffected): %s", exc)


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise CorruptArtifactError("The artifact manifest must be a JSON object.")
    version = manifest.get("schema_version")
    if type(version) is not int:
        raise CorruptArtifactError("The artifact manifest schema is invalid.")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"Artifact manifest schema {version} is not supported."
        )
    if not isinstance(manifest.get("artifacts"), dict):
        raise CorruptArtifactError("The artifact manifest entries are invalid.")


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with suppress(OSError):
            temp_path.chmod(0o600)
        os.replace(temp_path, path)
        temp_path = None
        with suppress(OSError):
            path.chmod(0o600)
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
