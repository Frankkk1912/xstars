"""Tests for ELISA preset module."""

import numpy as np
import pandas as pd
import pytest

from xstars.config import ExperimentPreset, PrismConfig
from xstars.presets.elisa import ELISAOptions, ELISAPreset
from xstars.tools.standard_curve import (
    CurveFitResult,
    back_calculate,
    fit_standard_curve,
    wide_to_conc_od,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_std_data():
    """Create standard curve data (concentration / OD pairs)."""
    df = pd.DataFrame({
        "0":    [0.052, 0.048, 0.055],
        "15.6": [0.108, 0.112, 0.105],
        "31.2": [0.198, 0.205, 0.192],
        "62.5": [0.385, 0.392, 0.378],
        "125":  [0.710, 0.725, 0.698],
        "250":  [1.245, 1.260, 1.232],
        "500":  [1.850, 1.870, 1.835],
        "1000": [2.180, 2.195, 2.168],
    })
    conc, od = wide_to_conc_od(df)
    return conc, od


def _make_sample_df():
    """Sample OD data grouped by treatment."""
    return pd.DataFrame({
        "Control":     [0.352, 0.365, 0.348],
        "Treatment_A": [0.890, 0.875, 0.905],
        "Treatment_B": [1.520, 1.545, 1.510],
    })


# ── Standard curve fitting + back-calculation ────────────────────────────

class TestELISAStdCurve:
    """Test standard curve fitting and back-calculation for ELISA."""

    def test_fit_auto(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")
        assert fit.r_squared is not None
        assert fit.r_squared > 0.95

    def test_back_calculate_within_range(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")
        # Back-calculate a known OD value — should get a reasonable concentration
        sample_od = np.array([0.385])  # ~62.5 pg/mL
        result = back_calculate(fit, sample_od)
        assert not np.isnan(result[0])
        assert 30 < result[0] < 120  # within reasonable range

    def test_back_calculate_multiple(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")
        sample_od = np.array([0.108, 0.710, 1.850])
        result = back_calculate(fit, sample_od)
        # All should be valid concentrations
        assert np.all(~np.isnan(result))
        # Should be roughly in order (monotonic)
        assert result[0] < result[1] < result[2]

    def test_back_calculate_out_of_range(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")
        # Very high OD that's far beyond the standard curve
        sample_od = np.array([50.0])
        result = back_calculate(fit, sample_od, clip_to_range=True)
        assert np.isnan(result[0])


# ── ELISAPreset transform ────────────────────────────────────────────────

class TestELISAPreset:
    """Test the ELISA preset transform."""

    def test_transform_basic(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")

        preset = ELISAPreset()
        sample_df = _make_sample_df()
        options = ELISAOptions(fit_result=fit)
        result = preset.transform(sample_df, options)

        # Result should have same shape
        assert result.shape == sample_df.shape
        assert list(result.columns) == list(sample_df.columns)

        # Concentrations should be positive (within standard curve range)
        for col in result.columns:
            valid = result[col].dropna()
            assert len(valid) > 0
            assert (valid > 0).all()

        # Treatment_B had higher OD → higher concentration
        assert result["Treatment_B"].mean() > result["Treatment_A"].mean()
        assert result["Treatment_A"].mean() > result["Control"].mean()

    def test_transform_stores_result(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")

        preset = ELISAPreset()
        sample_df = _make_sample_df()
        options = ELISAOptions(fit_result=fit)
        preset.transform(sample_df, options)

        assert preset.last_result is not None
        assert preset.last_result.fit_result is fit
        assert preset.last_result.concentration_df is not None

    def test_transform_no_fit_raises(self):
        preset = ELISAPreset()
        sample_df = _make_sample_df()
        options = ELISAOptions(fit_result=None)
        with pytest.raises(ValueError, match="No standard curve"):
            preset.transform(sample_df, options)

    def test_transform_empty_raises(self):
        conc, od = _make_std_data()
        fit = fit_standard_curve(conc, od, method="auto")

        preset = ELISAPreset()
        empty_df = pd.DataFrame()
        options = ELISAOptions(fit_result=fit)
        with pytest.raises(ValueError, match="empty"):
            preset.transform(empty_df, options)

    def test_default_y_label(self):
        preset = ELISAPreset()
        assert preset.default_y_label == "Concentration"


# ── Existing parameters mode ─────────────────────────────────────────────

class TestELISAExistingParams:
    """Test using manually provided parameters."""

    def test_linear_params(self):
        """Build a linear fit from params and use it for back-calculation."""
        slope = 0.002
        intercept = 0.05

        def predict(c):
            return slope * np.asarray(c, dtype=float) + intercept

        def inverse(y):
            return (np.asarray(y, dtype=float) - intercept) / slope

        fit = CurveFitResult(
            method="linear",
            params={"slope": slope, "intercept": intercept},
            r_squared=None,
            equation_str=f"y = {slope} * x + {intercept}",
            predict=predict,
            inverse=inverse,
            conc_range=(0.0, 1e6),
        )

        preset = ELISAPreset()
        sample_df = _make_sample_df()
        options = ELISAOptions(fit_result=fit)
        result = preset.transform(sample_df, options)

        # Check back-calculation: OD = 0.352 → conc = (0.352 - 0.05) / 0.002 = 151
        expected = (0.352 - intercept) / slope
        assert abs(result["Control"].iloc[0] - expected) < 0.1

    def test_four_pl_params(self):
        """Build a 4PL fit from params and use it."""
        from xstars.tools.standard_curve import four_param_logistic

        bottom, top, ec50, hill = 0.05, 2.2, 100.0, 1.2

        def predict(c):
            return four_param_logistic(np.asarray(c, dtype=float), bottom, top, ec50, hill)

        def inverse(y):
            y = np.asarray(y, dtype=float)
            result = np.full_like(y, np.nan, dtype=float)
            for i, yi in enumerate(y.flat):
                try:
                    ratio = (top - bottom) / (yi - bottom) - 1.0
                    if ratio <= 0:
                        continue
                    result.flat[i] = ec50 * ratio ** (1.0 / hill)
                except (ZeroDivisionError, ValueError):
                    continue
            return result

        fit = CurveFitResult(
            method="four_pl",
            params={"bottom": bottom, "top": top, "ec50": ec50, "hill": hill},
            r_squared=None,
            equation_str="4PL manual",
            predict=predict,
            inverse=inverse,
            conc_range=(0.0, 1e6),
        )

        preset = ELISAPreset()
        sample_df = _make_sample_df()
        options = ELISAOptions(fit_result=fit)
        result = preset.transform(sample_df, options)

        # All valid concentrations
        for col in result.columns:
            valid = result[col].dropna()
            assert len(valid) > 0


# ── Registration ─────────────────────────────────────────────────────────

class TestELISARegistration:
    """Test that ELISA preset is properly registered."""

    def test_get_preset(self):
        from xstars.presets import get_preset
        preset = get_preset(ExperimentPreset.ELISA)
        assert preset is not None
        assert isinstance(preset, ELISAPreset)

    def test_enum_exists(self):
        assert ExperimentPreset.ELISA.value == "elisa"
