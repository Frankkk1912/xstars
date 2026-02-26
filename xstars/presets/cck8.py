"""CCK-8 preset — computes viability % and optionally fits IC50 via 4PL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..config import ExperimentPreset
from ..tools.standard_curve import four_param_logistic as _four_param_logistic
from . import BasePreset, PresetOptions, register_preset


@dataclass
class CCK8Options(PresetOptions):
    """Options specific to CCK-8 viability assay."""
    blank_group: str = ""
    concentrations: list[float] = field(default_factory=list)
    fit_ic50: bool = True
    fit_method: str = "auto"  # "auto", "three_pl", "log_linear"


@dataclass
class CCK8FitInfo:
    """Lightweight data passed to PlotEngine for dose-response visualization."""
    concentrations: list[float]
    fit_params: dict  # {bottom, top, ic50, hill}
    dose_col_names: list[str]


@dataclass
class CCK8Result:
    """Stores IC50 fitting results from CCK-8 analysis."""
    viability_df: pd.DataFrame
    ic50: float | None = None
    ic50_95ci: tuple[float, float] | None = None
    r_squared: float | None = None
    fit_params: dict | None = None


def _three_param_logistic(x, bottom, ic50, hill):
    """3-parameter logistic (3PL) dose-response curve (top fixed at 100)."""
    return bottom + (100.0 - bottom) / (1.0 + (x / ic50) ** hill)


def _log_linear_ic50(means, concentrations):
    """Find IC50 by linear interpolation in log(concentration) space.

    Looks for where viability crosses 50% between adjacent concentration points.
    Returns IC50 value or None if no crossing found.
    """
    log_conc = np.log10(np.array(concentrations, dtype=float))

    for i in range(len(means) - 1):
        y1, y2 = means[i], means[i + 1]
        # Check if 50% crossing exists between these two points
        if (y1 - 50) * (y2 - 50) <= 0 and y1 != y2:
            # Linear interpolation in log space
            frac = (50 - y1) / (y2 - y1)
            log_ic50 = log_conc[i] + frac * (log_conc[i + 1] - log_conc[i])
            return 10 ** log_ic50

    return None


@register_preset(ExperimentPreset.CCK8)
class CCK8Preset(BasePreset):

    def __init__(self):
        self.last_result: CCK8Result | None = None

    @property
    def description(self) -> str:
        return "CCK-8 viability assay with optional IC50 fitting"

    @property
    def default_y_label(self) -> str:
        return "Viability (%)"

    def validate_input(self, df: pd.DataFrame, options: PresetOptions) -> None:
        opts = options if isinstance(options, CCK8Options) else CCK8Options(**vars(options))
        if opts.control_group and opts.control_group not in df.columns:
            raise ValueError(
                f"Control group '{opts.control_group}' not found in data columns: "
                f"{list(df.columns)}"
            )
        if opts.blank_group and opts.blank_group not in df.columns:
            raise ValueError(
                f"Blank group '{opts.blank_group}' not found in data columns: "
                f"{list(df.columns)}"
            )
        if opts.fit_ic50 and opts.concentrations:
            # concentrations should match the number of dose columns
            dose_cols = [
                c for c in df.columns
                if c != opts.blank_group and c != opts.control_group
            ]
            if len(opts.concentrations) != len(dose_cols):
                raise ValueError(
                    f"Number of concentrations ({len(opts.concentrations)}) "
                    f"does not match number of dose columns ({len(dose_cols)}): "
                    f"{dose_cols}"
                )

    def transform(self, df: pd.DataFrame, options: PresetOptions) -> pd.DataFrame:
        opts = options if isinstance(options, CCK8Options) else CCK8Options(**vars(options))
        self.validate_input(df, opts)

        control = opts.control_group or df.columns[0]

        # Blank subtraction
        if opts.blank_group:
            blank_mean = df[opts.blank_group].dropna().mean()
        else:
            blank_mean = 0.0

        control_mean = df[control].dropna().mean() - blank_mean
        if control_mean == 0:
            raise ValueError("Control - Blank OD is zero; cannot compute viability.")

        # Compute viability for all columns except blank
        # Control is kept (viability ≈ 100%) as the reference bar
        exclude = {opts.blank_group} if opts.blank_group else set()
        keep_cols = [c for c in df.columns if c not in exclude]
        dose_cols = [c for c in keep_cols if c != control]

        viability_data: dict[str, pd.Series] = {}
        for col in keep_cols:
            viability_data[col] = (df[col] - blank_mean) / control_mean * 100.0

        viability_df = pd.DataFrame(viability_data)

        # IC50 fitting
        self.last_result = CCK8Result(viability_df=viability_df)

        if opts.fit_ic50 and opts.concentrations and len(opts.concentrations) == len(dose_cols):
            self._fit_ic50(viability_df[dose_cols], opts.concentrations, opts.fit_method)

        return viability_df

    def _fit_ic50(
        self,
        viability_df: pd.DataFrame,
        concentrations: list[float],
        method: str = "auto",
    ) -> None:
        """Fit IC50 using the specified method (auto, three_pl, log_linear)."""
        means = viability_df.mean().to_numpy()
        conc = np.array(concentrations, dtype=float)

        if method == "log_linear":
            self._fit_log_linear(means, conc)
            return

        if method == "three_pl":
            self._fit_three_pl(means, conc)
            return

        # auto: try 3PL first, fallback to log-linear
        if self._fit_three_pl(means, conc):
            # Check R² threshold
            if self.last_result.r_squared is not None and self.last_result.r_squared >= 0.8:
                return
        # Fallback to log-linear
        self._fit_log_linear(means, conc)

    def _fit_three_pl(self, means: np.ndarray, conc: np.ndarray) -> bool:
        """Fit 3PL curve (top=100). Returns True if fit succeeded."""
        p0 = [0.0, np.median(conc), 1.0]
        bounds = (
            [-np.inf, 1e-20, -np.inf],
            [np.inf, np.inf, np.inf],
        )

        try:
            popt, pcov = curve_fit(
                _three_param_logistic, conc, means,
                p0=p0, bounds=bounds, maxfev=10000,
            )
        except (RuntimeError, ValueError):
            return False

        bottom, ic50, hill = popt

        predicted = _three_param_logistic(conc, *popt)
        ss_res = np.sum((means - predicted) ** 2)
        ss_tot = np.sum((means - np.mean(means)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else None

        ic50_ci = None
        if pcov is not None:
            se = np.sqrt(np.diag(pcov))
            if len(se) > 1 and np.isfinite(se[1]):
                ic50_ci = (ic50 - 1.96 * se[1], ic50 + 1.96 * se[1])

        assert self.last_result is not None
        self.last_result.ic50 = float(ic50)
        self.last_result.ic50_95ci = ic50_ci
        self.last_result.r_squared = float(r_squared) if r_squared is not None else None
        self.last_result.fit_params = {
            "method": "three_pl",
            "bottom": float(bottom),
            "top": 100.0,
            "ic50": float(ic50),
            "hill": float(hill),
        }
        return True

    def _fit_log_linear(self, means: np.ndarray, conc: np.ndarray) -> None:
        """Find IC50 by log-linear interpolation."""
        ic50 = _log_linear_ic50(means, conc)
        if ic50 is None:
            return

        assert self.last_result is not None
        self.last_result.ic50 = float(ic50)
        self.last_result.ic50_95ci = None
        self.last_result.r_squared = None
        self.last_result.fit_params = {
            "method": "log_linear",
            "ic50": float(ic50),
            "top": 100.0,
        }
