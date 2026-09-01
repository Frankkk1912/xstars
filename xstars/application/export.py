"""Persistent render payloads and controlled high-resolution export for WPS."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from ..config import PrismConfig
from ..plot_engine import PlotEngine
from ..presets.cck8 import CCK8FitInfo
from ..stats_engine import StatsEngine
from .contracts import ContractError, ErrorCode

PAYLOAD_SCHEMA_VERSION = "1.0"
DEFAULT_ARTIFACTS_ROOT = Path.home() / ".xstars" / "artifacts"
DEFAULT_EXPORTS_ROOT = Path.home() / ".xstars" / "exports"
EXPORT_FORMATS = frozenset({"png", "tiff", "jpg", "pdf"})
MIN_EXPORT_DPI = 72
MAX_EXPORT_DPI = 1200
BASE_CLIPBOARD_DPI = 96
MAX_EXPORT_PIXELS = 100_000_000
MAX_PAYLOAD_BYTES = 5_000_000


def new_picture_id(now: datetime | None = None) -> str:
    """Return a non-secret identifier safe for both Shape.Name and a file stem."""
    instant = now or datetime.now(timezone.utc)
    return f"XSTARS_{instant:%Y%m%d}_{secrets.token_hex(6)}"


def _json_cell(value: Any) -> Any:
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _config_snapshot(config: PrismConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    transient = {"export_path", "elisa_fit_result"}
    for item in fields(config):
        if item.name in transient:
            continue
        value = getattr(config, item.name)
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, CCK8FitInfo):
            value = asdict(value)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        snapshot[item.name] = value
    return snapshot


def _config_from_snapshot(raw: Mapping[str, Any]) -> PrismConfig:
    config = PrismConfig()
    allowed = {item.name for item in fields(config)} - {"export_path", "elisa_fit_result"}
    unknown = set(raw) - allowed
    if unknown:
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload has unknown config fields")
    for name, value in raw.items():
        current = getattr(config, name)
        if isinstance(current, Enum):
            try:
                value = type(current)(value)
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    ErrorCode.PAYLOAD_CORRUPT,
                    f"render payload has invalid config field: {name}",
                ) from exc
        elif name == "ic50_fit_info" and isinstance(value, Mapping):
            try:
                value = CCK8FitInfo(**value)
            except TypeError as exc:
                raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "invalid CCK-8 render state") from exc
        setattr(config, name, value)
    config.export_path = ""
    return config


def _atomic_write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        path.parent.chmod(0o700)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        with suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, path)
        temporary = None
        with suppress(OSError):
            path.chmod(0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_render_payload(
    picture_id: str,
    frame: pd.DataFrame,
    config: PrismConfig,
    figure: Any,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    renderer: str = "plot_engine",
) -> Path:
    """Persist a JSON-only payload sufficient to re-run PlotEngine later."""
    from .contracts import ImageWriteback

    # Reuse the boundary validator rather than maintaining a second id grammar.
    ImageWriteback(anchor_cell="A1", name=picture_id, source_key="primary_figure", picture_id=picture_id)
    width, height = (float(value) for value in figure.get_size_inches())
    payload = {
        "schemaVersion": PAYLOAD_SCHEMA_VERSION,
        "pictureId": picture_id,
        "renderer": renderer,
        "data": {
            "columns": [str(column) for column in frame.columns],
            "rows": [[_json_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)],
        },
        "config": _config_snapshot(config),
        "figure": {"widthInches": width, "heightInches": height},
    }
    target = artifacts_root.expanduser().resolve(strict=False) / f"{picture_id}.json"
    _atomic_write_payload(target, payload)
    return target


def load_render_payload(
    picture_id: str,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> dict[str, Any]:
    from .contracts import ImageWriteback

    ImageWriteback(
        anchor_cell="A1",
        name=picture_id,
        source_key="primary_figure",
        picture_id=picture_id,
    )
    path = artifacts_root.expanduser().resolve(strict=False) / f"{picture_id}.json"
    if path.is_symlink():
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload may not be a symbolic link")
    if not path.is_file():
        raise ContractError(
            ErrorCode.PAYLOAD_MISSING,
            "High-resolution source is unavailable; regenerate this XSTARS chart.",
        )
    try:
        if path.stat().st_size > MAX_PAYLOAD_BYTES:
            raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(
            ErrorCode.PAYLOAD_CORRUPT,
            "High-resolution source is damaged; regenerate this XSTARS chart.",
        ) from exc
    if not isinstance(payload, dict):
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload must be an object")
    if payload.get("schemaVersion") != PAYLOAD_SCHEMA_VERSION:
        raise ContractError(
            ErrorCode.PAYLOAD_VERSION,
            "High-resolution source version is unsupported; regenerate this XSTARS chart.",
        )
    if payload.get("pictureId") != picture_id or payload.get("renderer") not in {
        "plot_engine",
        "standard_curve",
    }:
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload identity is invalid")
    data = payload.get("data")
    config = payload.get("config")
    figure = payload.get("figure")
    if not isinstance(data, dict) or not isinstance(config, dict) or not isinstance(figure, dict):
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload fields are invalid")
    columns, rows = data.get("columns"), data.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) for column in columns)
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
    ):
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload data is invalid")
    for key in ("widthInches", "heightInches"):
        value = figure.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 100:
            raise ContractError(ErrorCode.PAYLOAD_CORRUPT, "render payload figure size is invalid")
    return payload


def validate_export_request(image_format: Any, dpi: Any) -> tuple[str, int]:
    if not isinstance(image_format, str) or image_format.lower() not in EXPORT_FORMATS:
        raise ContractError(ErrorCode.EXPORT_FORMAT, "format must be png, tiff, jpg, or pdf")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not MIN_EXPORT_DPI <= dpi <= MAX_EXPORT_DPI:
        raise ContractError(ErrorCode.EXPORT_DPI, "DPI must be an integer from 72 to 1200")
    return image_format.lower(), dpi


def _output_path(picture_id: str, image_format: str, dpi: int, exports_root: Path) -> Path:
    root = exports_root.expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / f"{picture_id}_{dpi}dpi.{image_format}"


def _ensure_extension(path: Path, image_format: str) -> Path:
    """Append the correct extension if the path does not already end with it."""
    ext = ".jpg" if image_format == "jpg" else f".{image_format}"
    if path.suffix.lower() != ext:
        path = path.with_name(path.name + ext)
    return path


def render_payload_export(
    picture_id: str,
    image_format: Any,
    dpi: Any,
    *,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    exports_root: Path = DEFAULT_EXPORTS_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Load a trusted payload and re-render it at the requested DPI.

    When *output_path* is supplied (must be an absolute path from a local
    file-chooser dialog) it is used instead of the automatic naming scheme.
    The extension is normalised to match *image_format* if needed.
    """
    image_format, dpi = validate_export_request(image_format, dpi)
    payload = load_render_payload(picture_id, artifacts_root=artifacts_root)
    data = payload["data"]
    try:
        frame = pd.DataFrame(data["rows"], columns=data["columns"])
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame.dropna(how="all").empty:
            raise ValueError("render payload contains no numeric data")
        config = _config_from_snapshot(payload["config"])
        if payload["renderer"] == "standard_curve":
            from .analysis import plot_standard_curve

            _fit, figure = plot_standard_curve(frame, config)
        else:
            stats_result = StatsEngine(config).analyze(frame)
            figure = PlotEngine(config).plot(frame, stats_result)
    except ContractError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ContractError(ErrorCode.PAYLOAD_CORRUPT, f"render payload cannot be rendered: {exc}") from exc
    figure.set_size_inches(
        float(payload["figure"]["widthInches"]),
        float(payload["figure"]["heightInches"]),
        forward=True,
    )
    if output_path is not None:
        if not output_path.is_absolute():
            raise ContractError(ErrorCode.EXPORT_PATH, "output_path must be absolute")
        output = _ensure_extension(output_path, image_format)
        with suppress(OSError):
            output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output = _output_path(picture_id, image_format, dpi, exports_root)
    save_format = "jpeg" if image_format == "jpg" else image_format
    try:
        figure.savefig(output, format=save_format, dpi=dpi)
    except (OSError, ValueError) as exc:
        raise ContractError(ErrorCode.EXPORT_PATH, f"cannot write export: {exc}") from exc
    finally:
        from matplotlib import pyplot as plt

        plt.close(figure)
    with suppress(OSError):
        output.chmod(0o600)
    return {
        "path": str(output.resolve()),
        "format": image_format,
        "dpi": dpi,
        "source": "render_payload",
    }


def export_clipboard_image(
    image_format: Any,
    dpi: Any,
    *,
    exports_root: Path = DEFAULT_EXPORTS_ROOT,
    image_grabber: Callable[[], Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Bonus path for arbitrary selected pictures copied by the WPS add-in.

    When *output_path* is supplied (must be an absolute path from a local
    file-chooser dialog) it is used instead of the automatic naming scheme.
    The extension is normalised to match *image_format* if needed.
    """
    image_format, dpi = validate_export_request(image_format, dpi)
    if image_grabber is None:
        from PIL import ImageGrab

        image_grabber = ImageGrab.grabclipboard
    try:
        image = image_grabber()
    except Exception as exc:
        raise ContractError(ErrorCode.PAYLOAD_MISSING, f"clipboard image could not be read: {exc}") from exc
    if image is None or not all(hasattr(image, name) for name in ("size", "resize", "save", "convert")):
        raise ContractError(ErrorCode.PAYLOAD_MISSING, "clipboard does not contain an image")
    width, height = image.size
    target = (max(1, round(width * dpi / BASE_CLIPBOARD_DPI)), max(1, round(height * dpi / BASE_CLIPBOARD_DPI)))
    if target[0] * target[1] > MAX_EXPORT_PIXELS:
        raise ContractError(ErrorCode.EXPORT_DPI, "clipboard export exceeds the pixel limit")
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    exported = image.resize(target, getattr(resampling, "LANCZOS", 1))
    if image_format in {"jpg", "pdf"}:
        if getattr(exported, "mode", "RGB") in {"RGBA", "LA"}:
            rgba = exported.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            exported = background
        else:
            exported = exported.convert("RGB")
    if output_path is not None:
        if not output_path.is_absolute():
            raise ContractError(ErrorCode.EXPORT_PATH, "output_path must be absolute")
        output = _ensure_extension(output_path, image_format)
        with suppress(OSError):
            output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output = _output_path(new_picture_id(), image_format, dpi, exports_root)
    kwargs: dict[str, Any] = {"dpi": (dpi, dpi)}
    if image_format == "tiff":
        kwargs["compression"] = "tiff_lzw"
    elif image_format == "jpg":
        kwargs["quality"] = 95
    elif image_format == "pdf":
        kwargs = {"resolution": dpi}
    try:
        exported.save(output, **kwargs)
    except (OSError, ValueError) as exc:
        raise ContractError(ErrorCode.EXPORT_PATH, f"cannot write export: {exc}") from exc
    return {"path": str(output.resolve()), "format": image_format, "dpi": dpi, "source": "clipboard"}
