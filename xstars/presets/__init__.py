"""Experiment preset framework for XSTARS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..config import ExperimentPreset


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

@dataclass
class PresetOptions:
    """Base options shared by all presets."""
    control_group: str = ""


class BasePreset(ABC):
    """Abstract base class for experiment presets."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this preset."""

    @property
    @abstractmethod
    def default_y_label(self) -> str:
        """Default Y-axis label when this preset is active."""

    @abstractmethod
    def validate_input(self, df: pd.DataFrame, options: PresetOptions) -> None:
        """Raise ``ValueError`` if *df* is not suitable for this preset."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, options: PresetOptions) -> pd.DataFrame:
        """Transform raw wide-format data into analysis-ready values."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BasePreset] = {}


def register_preset(key: "ExperimentPreset"):
    """Decorator that registers a preset instance under *key*."""
    def decorator(cls: type[BasePreset]):
        _REGISTRY[key.value if hasattr(key, "value") else key] = cls()
        return cls
    return decorator


def get_preset(key: "ExperimentPreset") -> BasePreset | None:
    """Return the preset registered for *key*, or ``None``."""
    val = key.value if hasattr(key, "value") else key
    if val == "none":
        return None
    return _REGISTRY.get(val)


# Import submodules so decorators execute
from . import wb, qpcr, cck8, elisa  # noqa: E402, F401
