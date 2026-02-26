"""Standard curve fitting and back-calculation for ELISA, BCA, etc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def four_param_logistic(x, bottom, top, ec50, hill):
    """4-parameter logistic (4PL) curve: bottom + (top - bottom) / (1 + (x/ec50)^hill)."""
    return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)


def three_param_logistic(x, bottom, ec50, hill):
    """3-parameter logistic (3PL) curve with top fixed at max observed OD."""
    # top is baked in at fit time via closure; this is the raw formula with top=1.0
    # We use a wrapper at fit time to inject the actual top value.
    return bottom + (1.0 - bottom) / (1.0 + (x / ec50) ** hill)


def _three_pl_factory(top: float):
    """Return a 3PL function with a fixed top value."""
    def func(x, bottom, ec50, hill):
        return bottom + (top - bottom) / (1.0 + (x / ec50) ** hill)
    return func


@dataclass
class CurveFitResult:
    """Result of standard curve fitting."""
    method: str               # "four_pl" / "three_pl" / "linear" / "log_linear_reg" / "interpolation"
    params: dict              # fitted parameters
    r_squared: float | None
    equation_str: str         # human-readable equation
    predict: Callable         # concentration -> OD (forward)
    inverse: Callable         # OD -> concentration (reverse)
    conc_range: tuple[float, float]  # (min, max) concentration in standards


def _r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


def _fit_four_pl(conc, od) -> CurveFitResult | None:
    """Fit 4PL curve to standard data."""
    try:
        p0 = [np.min(od), np.max(od), np.median(conc), 1.0]
        bounds = (
            [-np.inf, -np.inf, 1e-20, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        )
        popt, _ = curve_fit(four_param_logistic, conc, od, p0=p0, bounds=bounds, maxfev=10000)
        bottom, top, ec50, hill = popt
        predicted = four_param_logistic(conc, *popt)
        r2 = _r_squared(od, predicted)

        def predict(c):
            return four_param_logistic(np.asarray(c, dtype=float), bottom, top, ec50, hill)

        def inverse(y):
            y = np.asarray(y, dtype=float)
            result = np.full_like(y, np.nan, dtype=float)
            # Analytical inverse: c = ec50 * ((top - bottom) / (y - bottom) - 1) ^ (1/hill)
            for i, yi in enumerate(y.flat):
                try:
                    ratio = (top - bottom) / (yi - bottom) - 1.0
                    if ratio <= 0:
                        continue
                    result.flat[i] = ec50 * ratio ** (1.0 / hill)
                except (ZeroDivisionError, ValueError):
                    continue
            return result

        return CurveFitResult(
            method="four_pl",
            params={"bottom": float(bottom), "top": float(top), "ec50": float(ec50), "hill": float(hill)},
            r_squared=float(r2) if r2 is not None else None,
            equation_str=f"y = {bottom:.4g} + ({top:.4g} - {bottom:.4g}) / (1 + (x/{ec50:.4g})^{hill:.4g})",
            predict=predict,
            inverse=inverse,
            conc_range=(float(conc.min()), float(conc.max())),
        )
    except (RuntimeError, ValueError):
        return None


def _fit_three_pl(conc, od) -> CurveFitResult | None:
    """Fit 3PL curve (top fixed at max OD)."""
    top_val = float(np.max(od))
    func = _three_pl_factory(top_val)
    try:
        p0 = [np.min(od), np.median(conc), 1.0]
        bounds = (
            [-np.inf, 1e-20, -np.inf],
            [np.inf, np.inf, np.inf],
        )
        popt, _ = curve_fit(func, conc, od, p0=p0, bounds=bounds, maxfev=10000)
        bottom, ec50, hill = popt
        predicted = func(conc, *popt)
        r2 = _r_squared(od, predicted)

        def predict(c):
            return func(np.asarray(c, dtype=float), bottom, ec50, hill)

        def inverse(y):
            y = np.asarray(y, dtype=float)
            result = np.full_like(y, np.nan, dtype=float)
            for i, yi in enumerate(y.flat):
                try:
                    ratio = (top_val - bottom) / (yi - bottom) - 1.0
                    if ratio <= 0:
                        continue
                    result.flat[i] = ec50 * ratio ** (1.0 / hill)
                except (ZeroDivisionError, ValueError):
                    continue
            return result

        return CurveFitResult(
            method="three_pl",
            params={"bottom": float(bottom), "top": top_val, "ec50": float(ec50), "hill": float(hill)},
            r_squared=float(r2) if r2 is not None else None,
            equation_str=f"y = {bottom:.4g} + ({top_val:.4g} - {bottom:.4g}) / (1 + (x/{ec50:.4g})^{hill:.4g})",
            predict=predict,
            inverse=inverse,
            conc_range=(float(conc.min()), float(conc.max())),
        )
    except (RuntimeError, ValueError):
        return None


def _fit_linear(conc, od) -> CurveFitResult | None:
    """Fit OD = slope * conc + intercept."""
    try:
        coeffs = np.polyfit(conc, od, 1)
        slope, intercept = coeffs
        predicted = np.polyval(coeffs, conc)
        r2 = _r_squared(od, predicted)

        def predict(c):
            return slope * np.asarray(c, dtype=float) + intercept

        def inverse(y):
            y = np.asarray(y, dtype=float)
            if slope == 0:
                return np.full_like(y, np.nan, dtype=float)
            return (y - intercept) / slope

        return CurveFitResult(
            method="linear",
            params={"slope": float(slope), "intercept": float(intercept)},
            r_squared=float(r2) if r2 is not None else None,
            equation_str=f"y = {slope:.4g} * x + {intercept:.4g}",
            predict=predict,
            inverse=inverse,
            conc_range=(float(conc.min()), float(conc.max())),
        )
    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_log_linear_reg(conc, od) -> CurveFitResult | None:
    """Fit OD = slope * log10(conc) + intercept."""
    try:
        log_conc = np.log10(conc)
        coeffs = np.polyfit(log_conc, od, 1)
        slope, intercept = coeffs
        predicted = np.polyval(coeffs, log_conc)
        r2 = _r_squared(od, predicted)

        def predict(c):
            return slope * np.log10(np.asarray(c, dtype=float)) + intercept

        def inverse(y):
            y = np.asarray(y, dtype=float)
            if slope == 0:
                return np.full_like(y, np.nan, dtype=float)
            return 10.0 ** ((y - intercept) / slope)

        return CurveFitResult(
            method="log_linear_reg",
            params={"slope": float(slope), "intercept": float(intercept)},
            r_squared=float(r2) if r2 is not None else None,
            equation_str=f"y = {slope:.4g} * log10(x) + {intercept:.4g}",
            predict=predict,
            inverse=inverse,
            conc_range=(float(conc.min()), float(conc.max())),
        )
    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_interpolation(conc, od) -> CurveFitResult:
    """Piecewise log-linear interpolation (no parametric model)."""
    sort_idx = np.argsort(conc)
    conc_sorted = conc[sort_idx]
    od_sorted = od[sort_idx]

    # Group by unique concentrations (average replicates)
    unique_conc = np.unique(conc_sorted)
    mean_od = np.array([od_sorted[conc_sorted == c].mean() for c in unique_conc])

    # Check monotonicity — if not monotonic, still do interpolation but warn
    log_conc = np.log10(unique_conc)

    # Forward: log(conc) -> OD
    fwd = interp1d(log_conc, mean_od, kind="linear", fill_value="extrapolate")
    # Inverse: OD -> log(conc) — requires monotonic; enforce by sorting
    if np.all(np.diff(mean_od) > 0) or np.all(np.diff(mean_od) < 0):
        inv_sort = np.argsort(mean_od)
        inv = interp1d(mean_od[inv_sort], log_conc[inv_sort], kind="linear",
                       bounds_error=False, fill_value=np.nan)
    else:
        # Non-monotonic: use the forward function with numerical inversion
        inv = interp1d(mean_od, log_conc, kind="linear",
                       bounds_error=False, fill_value=np.nan)

    def predict(c):
        return fwd(np.log10(np.asarray(c, dtype=float)))

    def inverse(y):
        log_c = inv(np.asarray(y, dtype=float))
        return 10.0 ** log_c

    return CurveFitResult(
        method="interpolation",
        params={"n_points": len(unique_conc)},
        r_squared=None,
        equation_str=f"Piecewise log-linear interpolation ({len(unique_conc)} points)",
        predict=predict,
        inverse=inverse,
        conc_range=(float(conc.min()), float(conc.max())),
    )


def wide_to_conc_od(df_wide: "pd.DataFrame") -> tuple["np.ndarray", "np.ndarray"]:
    """Extract (conc, od) arrays from a wide DataFrame.

    Columns are concentration labels (parsed as floats), rows are OD replicates.
    Each column becomes N data points (one per non-NaN row).

    Returns
    -------
    conc : 1-D array of concentrations (repeated per replicate)
    od : 1-D array of OD values
    """
    import pandas as pd

    conc_list = []
    od_list = []
    for col in df_wide.columns:
        try:
            c = float(col)
        except (ValueError, TypeError):
            raise ValueError(
                f"Column name '{col}' cannot be parsed as a concentration. "
                "All column headers must be numeric (e.g. 0.1, 1, 10, 100)."
            )
        vals = pd.to_numeric(df_wide[col], errors="coerce").dropna().to_numpy()
        conc_list.append(np.full(len(vals), c))
        od_list.append(vals)

    conc = np.concatenate(conc_list)
    od = np.concatenate(od_list)
    return conc, od


def fit_standard_curve(
    conc: np.ndarray,
    od: np.ndarray,
    method: str = "auto",
) -> CurveFitResult:
    """Fit a standard curve to concentration/OD data.

    Parameters
    ----------
    conc : array of known concentrations
    od : array of OD readings (same length as conc)
    method : "auto", "four_pl", "three_pl", "linear", "log_linear_reg", "interpolation"

    Returns
    -------
    CurveFitResult with predict/inverse callables
    """
    conc = np.asarray(conc, dtype=float)
    od = np.asarray(od, dtype=float)

    if len(conc) != len(od):
        raise ValueError(f"conc ({len(conc)}) and od ({len(od)}) must have same length")
    if len(conc) < 2:
        raise ValueError("Need at least 2 data points for curve fitting")
    if np.any(conc < 0):
        raise ValueError("Concentrations must be non-negative")

    # Separate zero-concentration points; log-based methods can't use them
    has_zeros = np.any(conc == 0)
    pos_mask = conc > 0
    conc_pos = conc[pos_mask]
    od_pos = od[pos_mask]

    if method == "linear":
        # Linear works fine with zeros
        result = _fit_linear(conc, od)
        if result is None:
            raise ValueError("Linear fitting failed")
        return result
    elif method in ("four_pl", "three_pl", "log_linear_reg", "interpolation", "auto"):
        if len(conc_pos) < 2:
            if method == "auto":
                # Fall back to linear if not enough positive points
                result = _fit_linear(conc, od)
                if result is None:
                    raise ValueError("Linear fitting failed (only method usable with zero concentrations)")
                return result
            raise ValueError(
                f"Method '{method}' requires at least 2 positive concentrations, "
                f"but only {len(conc_pos)} found. Use 'linear' or 'auto' instead."
            )

    # For non-linear methods, fit on positive-concentration data only,
    # but record the full range (including 0) for back-calculation clipping.
    full_conc_range = (float(conc.min()), float(conc.max()))

    if method == "four_pl":
        result = _fit_four_pl(conc_pos, od_pos)
        if result is None:
            raise ValueError("4PL fitting failed to converge")
        result.conc_range = full_conc_range
        return result
    elif method == "three_pl":
        result = _fit_three_pl(conc_pos, od_pos)
        if result is None:
            raise ValueError("3PL fitting failed to converge")
        result.conc_range = full_conc_range
        return result
    elif method == "log_linear_reg":
        result = _fit_log_linear_reg(conc_pos, od_pos)
        if result is None:
            raise ValueError("Log-linear regression failed")
        result.conc_range = full_conc_range
        return result
    elif method == "interpolation":
        result = _fit_interpolation(conc_pos, od_pos)
        result.conc_range = full_conc_range
        return result
    elif method == "auto":
        result = _fit_auto(conc_pos, od_pos)
        result.conc_range = full_conc_range
        # If we have zeros and a linear fit might be better overall, compare
        if has_zeros:
            linear = _fit_linear(conc, od)
            if linear is not None and linear.r_squared is not None:
                if result.r_squared is None or linear.r_squared > result.r_squared:
                    return linear
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def _fit_auto(conc, od) -> CurveFitResult:
    """Try 4PL → 3PL → log-linear regression, pick best R²."""
    candidates = []
    for fit_fn in (_fit_four_pl, _fit_three_pl, _fit_log_linear_reg):
        result = fit_fn(conc, od)
        if result is not None and result.r_squared is not None:
            candidates.append(result)

    if not candidates:
        # Fallback to interpolation
        return _fit_interpolation(conc, od)

    return max(candidates, key=lambda r: r.r_squared)


def back_calculate(
    fit_result: CurveFitResult,
    sample_od: np.ndarray,
    clip_to_range: bool = True,
) -> np.ndarray:
    """Back-calculate concentrations from sample OD values.

    Parameters
    ----------
    fit_result : result from fit_standard_curve
    sample_od : array of OD values to convert
    clip_to_range : if True, values outside the standard curve range become NaN

    Returns
    -------
    Array of calculated concentrations (NaN for out-of-range or invalid)
    """
    sample_od = np.asarray(sample_od, dtype=float)
    concentrations = fit_result.inverse(sample_od)
    concentrations = np.asarray(concentrations, dtype=float)

    if clip_to_range:
        cmin, cmax = fit_result.conc_range
        mask = (concentrations < cmin * 0.1) | (concentrations > cmax * 10)
        concentrations[mask] = np.nan

    return concentrations
