"""Host-independent request and writeback contracts for XSTARS.

Only plain JSON data crosses the WPS broker boundary.  These DTOs deliberately
exclude callables, pickle payloads, shell fragments, and unbounded paths.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_SELECTION_ROWS = 200
MAX_SELECTION_COLUMNS = 200
MAX_TABLE_CELLS = MAX_SELECTION_ROWS * MAX_SELECTION_COLUMNS
MAX_MESSAGE_LENGTH = 2_000

_CELL_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?([1-9][0-9]*)$")
_SOURCE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PICTURE_ID_RE = re.compile(r"^XSTARS_[0-9]{8}_[0-9a-z]{8,32}$")


class Command(str, Enum):
    """Complete whitelist of zero-argument Excel command entry points."""

    RUN = "run"
    QUICK = "run_quick"
    WB = "run_wb"
    QPCR = "run_qpcr"
    CCK8 = "run_cck8"
    ELISA = "run_elisa"
    TRANSFORM_ONLY = "run_transform_only"
    STANDARD_CURVE = "run_standard_curve"
    EXPORT = "run_export"
    RESET_SETTINGS = "run_reset_settings"
    ABOUT = "run_about"

    BASE_THEME_CLASSIC = "run_set_base_theme_classic"
    BASE_THEME_BW = "run_set_base_theme_bw"
    BASE_THEME_MINIMAL = "run_set_base_theme_minimal"
    BASE_THEME_DARK = "run_set_base_theme_dark"

    THEME_NONE = "run_set_theme_none"
    THEME_NATURE = "run_set_theme_nature"
    THEME_SCIENCE = "run_set_theme_science"
    THEME_CELL = "run_set_theme_cell"
    THEME_LANCET = "run_set_theme_lancet"
    THEME_NEJM = "run_set_theme_nejm"
    THEME_JAMA = "run_set_theme_jama"
    THEME_BMJ = "run_set_theme_bmj"

    JOURNAL_PALETTE_DEFAULT = "run_set_journal_palette_default"
    JOURNAL_PALETTE_NATURE = "run_set_journal_palette_nature"
    JOURNAL_PALETTE_SCIENCE = "run_set_journal_palette_science"
    JOURNAL_PALETTE_CELL = "run_set_journal_palette_cell"
    JOURNAL_PALETTE_LANCET = "run_set_journal_palette_lancet"
    JOURNAL_PALETTE_NEJM = "run_set_journal_palette_nejm"
    JOURNAL_PALETTE_JAMA = "run_set_journal_palette_jama"
    JOURNAL_PALETTE_BMJ = "run_set_journal_palette_bmj"

    PALETTE_DEFAULT = "run_set_palette_default"
    PALETTE_COLORBLIND = "run_set_palette_colorblind"
    PALETTE_VIBRANT = "run_set_palette_vibrant"
    PALETTE_PASTEL = "run_set_palette_pastel"
    PALETTE_DEEP = "run_set_palette_deep"
    PALETTE_MUTED = "run_set_palette_muted"


class ArtifactFormat(str, Enum):
    PNG = "png"
    TIFF = "tiff"
    JPG = "jpg"
    PDF = "pdf"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_COMMAND = "INVALID_COMMAND"
    INVALID_SELECTION = "INVALID_SELECTION"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INVALID_PATH = "INVALID_PATH"
    CANCELLED = "CANCELLED"
    BUSY = "BUSY"
    TIMEOUT = "TIMEOUT"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    PAYLOAD_MISSING = "PAYLOAD_MISSING"
    PAYLOAD_CORRUPT = "PAYLOAD_CORRUPT"
    PAYLOAD_VERSION = "PAYLOAD_VERSION"
    EXPORT_FORMAT = "EXPORT_FORMAT"
    EXPORT_DPI = "EXPORT_DPI"
    EXPORT_PATH = "EXPORT_PATH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ContractError(ValueError):
    """Raised when an untrusted DTO violates the contract."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require_version(version: str) -> None:
    if version != SCHEMA_VERSION:
        raise ContractError(
            ErrorCode.INVALID_REQUEST,
            f"Unsupported schema version: {version!r}",
        )


def _json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _validate_matrix(
    values: Sequence[Sequence[Any]],
    *,
    max_rows: int = MAX_SELECTION_ROWS,
    max_columns: int = MAX_SELECTION_COLUMNS,
    code: ErrorCode = ErrorCode.INVALID_SELECTION,
) -> list[list[Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(code, "values must be a two-dimensional array")
    rows = [list(row) for row in values]
    if not rows:
        raise ContractError(code, "values must not be empty")
    if len(rows) > max_rows:
        raise ContractError(ErrorCode.PAYLOAD_TOO_LARGE, "too many rows")
    width = len(rows[0])
    if width == 0:
        raise ContractError(code, "rows must not be empty")
    if width > max_columns or len(rows) * width > MAX_TABLE_CELLS:
        raise ContractError(ErrorCode.PAYLOAD_TOO_LARGE, "too many cells")
    for row in rows:
        if len(row) != width:
            raise ContractError(code, "values must be rectangular")
        if not all(_json_scalar(value) for value in row):
            raise ContractError(code, "cells must contain finite JSON scalar values")
    return rows


def _column_number(label: str) -> int:
    result = 0
    for char in label.upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def parse_cell(cell: str) -> tuple[int, int]:
    """Parse an A1 cell reference into one-based ``(row, column)``."""
    if not isinstance(cell, str):
        raise ContractError(ErrorCode.INVALID_REQUEST, "cell must be a string")
    match = _CELL_RE.fullmatch(cell.strip())
    if match is None:
        raise ContractError(ErrorCode.INVALID_REQUEST, f"Invalid cell reference: {cell!r}")
    return int(match.group(2)), _column_number(match.group(1))


def cell_to_a1(row: int, column: int) -> str:
    """Return an A1 reference for one-based coordinates."""
    if row < 1 or column < 1 or column > 16_384:
        raise ContractError(ErrorCode.INVALID_REQUEST, "cell coordinates are out of range")
    label = ""
    current = column
    while current:
        current, remainder = divmod(current - 1, 26)
        label = chr(ord("A") + remainder) + label
    return f"{label}{row}"


def _parse_address(address: str) -> tuple[int, int, int, int]:
    if not isinstance(address, str) or not address.strip():
        raise ContractError(ErrorCode.INVALID_SELECTION, "address is required")
    clean = address.strip()
    if any(token in clean for token in (",", ";", "!")):
        raise ContractError(ErrorCode.INVALID_SELECTION, "only one rectangular area is allowed")
    parts = clean.split(":")
    if len(parts) not in (1, 2):
        raise ContractError(ErrorCode.INVALID_SELECTION, "invalid selection address")
    start_row, start_col = parse_cell(parts[0])
    end_row, end_col = parse_cell(parts[-1])
    if end_row < start_row or end_col < start_col:
        raise ContractError(ErrorCode.INVALID_SELECTION, "selection address is reversed")
    return start_row, start_col, end_row, end_col


def ensure_path_within(path: str | Path, allowed_root: str | Path) -> Path:
    """Resolve *path* and require it to be under an explicit trusted root."""
    candidate = Path(path)
    root = Path(allowed_root)
    if not candidate.is_absolute() or not root.is_absolute():
        raise ContractError(ErrorCode.INVALID_PATH, "paths must be absolute")
    if ".." in candidate.parts:
        raise ContractError(ErrorCode.INVALID_PATH, "path traversal is not allowed")
    resolved = candidate.resolve(strict=False)
    trusted = root.resolve(strict=False)
    try:
        resolved.relative_to(trusted)
    except ValueError as exc:
        raise ContractError(ErrorCode.INVALID_PATH, "path is outside the controlled root") from exc
    return resolved


@dataclass
class SelectionPayload:
    values: list[list[Any]]
    address: str
    sheet: str
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version)
        self.values = _validate_matrix(self.values)
        if not isinstance(self.sheet, str) or not self.sheet.strip() or len(self.sheet) > 31:
            raise ContractError(ErrorCode.INVALID_SELECTION, "sheet name is invalid")
        start_row, start_col, end_row, end_col = _parse_address(self.address)
        if end_row - start_row + 1 != len(self.values):
            raise ContractError(ErrorCode.INVALID_SELECTION, "address row count does not match values")
        if end_col - start_col + 1 != len(self.values[0]):
            raise ContractError(ErrorCode.INVALID_SELECTION, "address column count does not match values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "values": self.values,
            "address": self.address,
            "sheet": self.sheet,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionPayload:
        if not isinstance(data, Mapping):
            raise ContractError(ErrorCode.INVALID_REQUEST, "selection must be an object")
        try:
            return cls(
                values=data["values"],
                address=data["address"],
                sheet=data["sheet"],
                version=data.get("version", ""),
            )
        except KeyError as exc:
            raise ContractError(ErrorCode.INVALID_REQUEST, f"Missing field: {exc.args[0]}") from exc


_STANDARD_CURVE_METHODS = {
    "auto",
    "four_pl",
    "three_pl",
    "linear",
    "log_linear_reg",
    "interpolation",
}


@dataclass
class StandardCurveOptions:
    """Serializable choices returned by the staged standard-curve dialog."""

    fit_method: str = "auto"
    back_calculate: bool = True

    def __post_init__(self) -> None:
        if self.fit_method not in _STANDARD_CURVE_METHODS:
            raise ContractError(ErrorCode.INVALID_REQUEST, "fitMethod is invalid")
        if not isinstance(self.back_calculate, bool):
            raise ContractError(ErrorCode.INVALID_REQUEST, "backCalculate must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitMethod": self.fit_method,
            "backCalculate": self.back_calculate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StandardCurveOptions:
        if not isinstance(data, Mapping) or set(data) - {"fitMethod", "backCalculate"}:
            raise ContractError(ErrorCode.INVALID_REQUEST, "standard curve options are invalid")
        return cls(
            fit_method=data.get("fitMethod", "auto"),
            back_calculate=data.get("backCalculate", True),
        )


@dataclass
class TransformOptions:
    """Transient transform-only choices shared by host adapters."""

    include_stats: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.include_stats, bool):
            raise ContractError(ErrorCode.INVALID_REQUEST, "includeStats must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"includeStats": self.include_stats}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransformOptions:
        if not isinstance(data, Mapping) or set(data) - {"includeStats"}:
            raise ContractError(ErrorCode.INVALID_REQUEST, "transform options are invalid")
        return cls(include_stats=data.get("includeStats", False))


@dataclass
class TableWriteback:
    start_cell: str
    values: list[list[Any]]

    def __post_init__(self) -> None:
        parse_cell(self.start_cell)
        self.values = _validate_matrix(self.values, code=ErrorCode.INVALID_REQUEST)

    def to_dict(self) -> dict[str, Any]:
        return {"startCell": self.start_cell, "values": self.values}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TableWriteback:
        try:
            return cls(start_cell=data["startCell"], values=data["values"])
        except KeyError as exc:
            raise ContractError(ErrorCode.INVALID_REQUEST, f"Missing field: {exc.args[0]}") from exc


@dataclass
class Artifact:
    path: str
    format: ArtifactFormat
    dpi: int
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version)
        if not isinstance(self.format, ArtifactFormat):
            try:
                self.format = ArtifactFormat(self.format)
            except (TypeError, ValueError) as exc:
                raise ContractError(ErrorCode.INVALID_REQUEST, "unsupported artifact format") from exc
        if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or not 72 <= self.dpi <= 1200:
            raise ContractError(ErrorCode.INVALID_REQUEST, "DPI must be an integer from 72 to 1200")
        candidate = Path(self.path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ContractError(ErrorCode.INVALID_PATH, "artifact path must be absolute without traversal")
        suffixes = {
            ArtifactFormat.PNG: {".png"},
            ArtifactFormat.TIFF: {".tif", ".tiff"},
            ArtifactFormat.JPG: {".jpg", ".jpeg"},
            ArtifactFormat.PDF: {".pdf"},
        }
        if candidate.suffix.lower() not in suffixes[self.format]:
            raise ContractError(ErrorCode.INVALID_PATH, "artifact extension does not match format")

    def validate_under(self, allowed_root: str | Path) -> Path:
        return ensure_path_within(self.path, allowed_root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "format": self.format.value,
            "dpi": self.dpi,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Artifact:
        try:
            return cls(
                path=data["path"],
                format=data["format"],
                dpi=data["dpi"],
                version=data.get("version", ""),
            )
        except KeyError as exc:
            raise ContractError(ErrorCode.INVALID_REQUEST, f"Missing field: {exc.args[0]}") from exc


@dataclass
class ImageWriteback:
    anchor_cell: str
    name: str
    artifact: Artifact | None = None
    source_key: str | None = None
    width: float | None = None
    height: float | None = None
    picture_id: str | None = None

    def __post_init__(self) -> None:
        parse_cell(self.anchor_cell)
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 255:
            raise ContractError(ErrorCode.INVALID_REQUEST, "image name is invalid")
        if self.artifact is None and self.source_key is None:
            raise ContractError(ErrorCode.INVALID_REQUEST, "image source is required")
        if self.source_key is not None and _SOURCE_KEY_RE.fullmatch(self.source_key) is None:
            raise ContractError(ErrorCode.INVALID_REQUEST, "image source key is invalid")
        if self.picture_id is not None and _PICTURE_ID_RE.fullmatch(self.picture_id) is None:
            raise ContractError(ErrorCode.INVALID_REQUEST, "image pictureId is invalid")
        for dimension in (self.width, self.height):
            if dimension is not None and (not isinstance(dimension, (int, float)) or dimension <= 0):
                raise ContractError(ErrorCode.INVALID_REQUEST, "image dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"anchorCell": self.anchor_cell, "name": self.name}
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        if self.source_key is not None:
            result["sourceKey"] = self.source_key
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height
        if self.picture_id is not None:
            result["pictureId"] = self.picture_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageWriteback:
        try:
            artifact_data = data.get("artifact")
            return cls(
                anchor_cell=data["anchorCell"],
                name=data["name"],
                artifact=Artifact.from_dict(artifact_data) if artifact_data is not None else None,
                source_key=data.get("sourceKey"),
                width=data.get("width"),
                height=data.get("height"),
                picture_id=data.get("pictureId"),
            )
        except KeyError as exc:
            raise ContractError(ErrorCode.INVALID_REQUEST, f"Missing field: {exc.args[0]}") from exc


@dataclass
class WritebackPlan:
    tables: list[TableWriteback] = field(default_factory=list)
    images: list[ImageWriteback] = field(default_factory=list)
    status_message: str = ""
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version)
        if not isinstance(self.status_message, str) or len(self.status_message) > MAX_MESSAGE_LENGTH:
            raise ContractError(ErrorCode.INVALID_REQUEST, "status message is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tables": [table.to_dict() for table in self.tables],
            "images": [image.to_dict() for image in self.images],
            "statusMessage": self.status_message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WritebackPlan:
        if not isinstance(data, Mapping):
            raise ContractError(ErrorCode.INVALID_REQUEST, "writeback plan must be an object")
        return cls(
            tables=[TableWriteback.from_dict(item) for item in data.get("tables", [])],
            images=[ImageWriteback.from_dict(item) for item in data.get("images", [])],
            status_message=data.get("statusMessage", ""),
            version=data.get("version", ""),
        )


@dataclass
class ErrorDTO:
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_version(self.version)
        if not isinstance(self.code, ErrorCode):
            try:
                self.code = ErrorCode(self.code)
            except (TypeError, ValueError) as exc:
                raise ContractError(ErrorCode.INVALID_REQUEST, "unknown error code") from exc
        if not isinstance(self.message, str) or not self.message or len(self.message) > MAX_MESSAGE_LENGTH:
            raise ContractError(ErrorCode.INVALID_REQUEST, "error message is invalid")
        if not isinstance(self.details, dict) or not all(
            isinstance(key, str) and _json_scalar(value)
            for key, value in self.details.items()
        ):
            raise ContractError(ErrorCode.INVALID_REQUEST, "error details must be JSON scalar values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ErrorDTO:
        try:
            return cls(
                code=data["code"],
                message=data["message"],
                details=dict(data.get("details", {})),
                version=data.get("version", ""),
            )
        except KeyError as exc:
            raise ContractError(ErrorCode.INVALID_REQUEST, f"Missing field: {exc.args[0]}") from exc
