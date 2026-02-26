"""Tests for standard curve fitting and back-calculation."""

import numpy as np
import pandas as pd
import pytest

from xstars.tools.standard_curve import (
    CurveFitResult,
    back_calculate,
    fit_standard_curve,
    four_param_logistic,
    wide_to_conc_od,
)


class TestWideToConc:
    def test_basic(self):
        df = pd.DataFrame({"0.1": [0.05, 0.06], "1": [0.5, 0.52], "10": [1.2, 1.3]})
        conc, od = wide_to_conc_od(df)
        assert len(conc) == 6
        assert len(od) == 6
        np.testing.assert_allclose(conc[:2], [0.1, 0.1])
        np.testing.assert_allclose(conc[4:], [10, 10])

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"Control": [1, 2], "10": [3, 4]})
        with pytest.raises(ValueError, match="cannot be parsed"):
            wide_to_conc_od(df)

    def test_nan_handling(self):
        df = pd.DataFrame({"1": [0.1, np.nan, 0.12], "10": [1.0, 1.1, 1.2]})
        conc, od = wide_to_conc_od(df)
        assert len(conc) == 5  # 2 from col "1" + 3 from col "10"


class TestFitLinear:
    def test_perfect_linear(self):
        conc = np.array([1, 2, 3, 4, 5], dtype=float)
        od = 0.5 * conc + 0.1
        result = fit_standard_curve(conc, od, method="linear")
        assert result.method == "linear"
        assert result.r_squared is not None
        assert result.r_squared > 0.999

    def test_linear_back_calculate(self):
        conc = np.array([1, 2, 3, 4, 5], dtype=float)
        od = 0.5 * conc + 0.1
        result = fit_standard_curve(conc, od, method="linear")
        back = back_calculate(result, np.array([0.6, 1.1, 1.6]))
        np.testing.assert_allclose(back, [1.0, 2.0, 3.0], atol=0.01)


class TestFit4PL:
    @pytest.fixture
    def sigmoidal_data(self):
        conc = np.array([0.1, 0.5, 1, 5, 10, 50, 100], dtype=float)
        od = four_param_logistic(conc, bottom=0.05, top=2.0, ec50=5.0, hill=1.2)
        rng = np.random.default_rng(42)
        od += rng.normal(0, 0.02, size=len(od))
        return conc, od

    def test_fit_4pl(self, sigmoidal_data):
        conc, od = sigmoidal_data
        result = fit_standard_curve(conc, od, method="four_pl")
        assert result.method == "four_pl"
        assert result.r_squared is not None
        assert result.r_squared > 0.95
        assert "ec50" in result.params
        assert abs(result.params["ec50"] - 5.0) < 2.0

    def test_back_calculate_4pl(self, sigmoidal_data):
        conc, od = sigmoidal_data
        result = fit_standard_curve(conc, od, method="four_pl")
        test_conc = np.array([1.0, 5.0, 10.0])
        test_od = result.predict(test_conc)
        recovered = back_calculate(result, test_od)
        np.testing.assert_allclose(recovered, test_conc, rtol=0.15)


class TestFit3PL:
    def test_fit_3pl(self):
        conc = np.array([0.1, 0.5, 1, 5, 10, 50, 100], dtype=float)
        od = four_param_logistic(conc, bottom=0.05, top=2.0, ec50=5.0, hill=1.2)
        result = fit_standard_curve(conc, od, method="three_pl")
        assert result.method == "three_pl"
        assert result.r_squared is not None
        assert result.r_squared > 0.9
        # top is fixed at max(od), which is close to but not exactly 2.0
        assert result.params["top"] == pytest.approx(max(od), abs=0.01)


class TestFitAuto:
    def test_auto_selects_best(self):
        conc = np.array([0.1, 0.5, 1, 5, 10, 50, 100], dtype=float)
        od = four_param_logistic(conc, bottom=0.05, top=2.0, ec50=5.0, hill=1.2)
        result = fit_standard_curve(conc, od, method="auto")
        assert result.r_squared is not None
        assert result.r_squared > 0.95

    def test_auto_linear_data(self):
        conc = np.array([1, 2, 3, 4, 5], dtype=float)
        od = 0.3 * conc + 0.05
        result = fit_standard_curve(conc, od, method="auto")
        assert result.r_squared is not None
        assert result.r_squared > 0.99


class TestFitLogLinear:
    def test_log_linear_regression(self):
        conc = np.array([0.1, 1, 10, 100], dtype=float)
        od = 0.5 * np.log10(conc) + 1.0
        result = fit_standard_curve(conc, od, method="log_linear_reg")
        assert result.method == "log_linear_reg"
        assert result.r_squared is not None
        assert result.r_squared > 0.999


class TestFitInterpolation:
    def test_interpolation(self):
        conc = np.array([1, 2, 5, 10, 20], dtype=float)
        od = np.array([0.1, 0.3, 0.7, 1.2, 1.8])
        result = fit_standard_curve(conc, od, method="interpolation")
        assert result.method == "interpolation"
        assert result.r_squared is None
        pred = result.predict(np.array([2.0, 10.0]))
        np.testing.assert_allclose(pred, [0.3, 1.2], atol=0.05)


class TestBackCalculateOutOfRange:
    def test_out_of_range_returns_nan(self):
        conc = np.array([1, 2, 3, 4, 5], dtype=float)
        od = 0.5 * conc + 0.1
        result = fit_standard_curve(conc, od, method="linear")
        # OD value way beyond range
        back = back_calculate(result, np.array([100.0]), clip_to_range=True)
        assert np.isnan(back[0])


class TestBackCalculateWideFormat:
    """Test the full wide-format workflow: wide → fit → back-calculate wide."""

    def test_roundtrip(self):
        # Standards: wide format
        std_df = pd.DataFrame({
            "1": [0.1, 0.12, 0.11],
            "5": [0.5, 0.52, 0.48],
            "10": [1.0, 1.02, 0.98],
            "50": [4.8, 5.0, 5.2],
        })
        conc, od = wide_to_conc_od(std_df)
        fit = fit_standard_curve(conc, od, method="auto")

        # Sample data: wide format (different groups)
        sample_df = pd.DataFrame({
            "Group_A": [0.5, 0.48, 0.52],
            "Group_B": [1.0, 1.05, 0.95],
        })
        # Back-calculate each column
        result_df = sample_df.copy()
        for col in result_df.columns:
            od_vals = result_df[col].to_numpy()
            result_df[col] = back_calculate(fit, od_vals)

        # Group_A OD ~0.5 should give conc ~5
        assert result_df["Group_A"].mean() == pytest.approx(5.0, rel=0.3)
        # Group_B OD ~1.0 should give conc ~10
        assert result_df["Group_B"].mean() == pytest.approx(10.0, rel=0.3)


class TestZeroConcentration:
    """Standard curves often include 0 concentration (blank/control)."""

    def test_linear_with_zero(self):
        conc = np.array([0, 1, 2, 5, 10], dtype=float)
        od = 0.5 * conc + 0.05
        result = fit_standard_curve(conc, od, method="linear")
        assert result.r_squared > 0.999
        assert result.method == "linear"

    def test_auto_with_zero(self):
        conc = np.array([0, 1, 2, 5, 10], dtype=float)
        od = 0.5 * conc + 0.05
        result = fit_standard_curve(conc, od, method="auto")
        assert result.r_squared is not None
        assert result.r_squared > 0.95

    def test_4pl_with_zero_excludes_zero_points(self):
        conc = np.array([0, 0.1, 0.5, 1, 5, 10, 50, 100], dtype=float)
        od_pos = four_param_logistic(conc[1:], bottom=0.05, top=2.0, ec50=5.0, hill=1.2)
        od = np.concatenate([[0.04], od_pos])  # zero-conc has low OD
        result = fit_standard_curve(conc, od, method="four_pl")
        assert result.method == "four_pl"
        assert result.r_squared > 0.9
        # conc_range should include 0
        assert result.conc_range[0] == 0.0

    def test_auto_with_only_one_positive_falls_back_to_linear(self):
        conc = np.array([0, 0, 5], dtype=float)
        od = np.array([0.1, 0.12, 2.5])
        result = fit_standard_curve(conc, od, method="auto")
        assert result.method == "linear"

    def test_back_calculate_includes_zero_range(self):
        conc = np.array([0, 1, 5, 10], dtype=float)
        od = 0.3 * conc + 0.05
        result = fit_standard_curve(conc, od, method="linear")
        # Back-calculate OD that maps to ~0 should not be NaN
        back = back_calculate(result, np.array([0.05]))
        assert not np.isnan(back[0])
        assert back[0] == pytest.approx(0.0, abs=0.1)

    def test_wide_df_with_zero_column(self):
        df = pd.DataFrame({
            "0": [0.05, 0.06],
            "1": [0.5, 0.52],
            "10": [1.2, 1.3],
        })
        conc, od = wide_to_conc_od(df)
        result = fit_standard_curve(conc, od, method="auto")
        assert result.r_squared is not None
        assert result.r_squared > 0.9

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_standard_curve(np.array([-1, 1, 2]), np.array([0.1, 0.2, 0.3]))


class TestValidation:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            fit_standard_curve(np.array([1, 2]), np.array([1, 2, 3]))

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_standard_curve(np.array([1.0]), np.array([0.5]))

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            fit_standard_curve(np.array([1, 2]), np.array([0.1, 0.2]), method="bogus")


class TestCCK8StillWorks:
    """Verify CCK-8 preset still works after moving _four_param_logistic."""

    def test_cck8_import(self):
        from xstars.presets.cck8 import _four_param_logistic, _three_param_logistic
        result = _four_param_logistic(5.0, 0, 100, 5.0, 1.0)
        assert abs(result - 50.0) < 1e-10

    def test_cck8_preset_transform(self):
        from xstars.presets.cck8 import CCK8Options, CCK8Preset

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Control": rng.normal(1.5, 0.1, size=6),
            "Dose_1": rng.normal(1.0, 0.1, size=6),
            "Dose_2": rng.normal(0.5, 0.1, size=6),
        })
        preset = CCK8Preset()
        opts = CCK8Options(control_group="Control", fit_ic50=False)
        result = preset.transform(df, opts)
        assert abs(result["Control"].mean() - 100.0) < 5.0
