"""Tests for experiment presets (WB, qPCR, CCK-8)."""

import numpy as np
import pandas as pd
import pytest

from xstars.config import DoseAxisScale, ExperimentPreset, PrismConfig
from xstars.presets import BasePreset, get_preset, register_preset, _REGISTRY
from xstars.presets.wb import WBOptions, WBPreset
from xstars.presets.qpcr import QPCROptions, QPCRPreset
from xstars.presets.cck8 import CCK8FitInfo, CCK8Options, CCK8Preset, CCK8Result


# ── Registry tests ──────────────────────────────────────────────────────

class TestRegistry:
    def test_get_preset_none(self):
        assert get_preset(ExperimentPreset.NONE) is None

    def test_get_preset_wb(self):
        p = get_preset(ExperimentPreset.WB)
        assert isinstance(p, WBPreset)

    def test_get_preset_qpcr(self):
        p = get_preset(ExperimentPreset.QPCR)
        assert isinstance(p, QPCRPreset)

    def test_get_preset_cck8(self):
        p = get_preset(ExperimentPreset.CCK8)
        assert isinstance(p, CCK8Preset)

    def test_all_registered_presets_are_base(self):
        for key, preset in _REGISTRY.items():
            assert isinstance(preset, BasePreset)


# ── WB Preset ───────────────────────────────────────────────────────────

class TestWBPreset:
    @pytest.fixture
    def wb_data(self):
        """Simple WB intensity data: 3 groups, 5 replicates each."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "Control": rng.uniform(1000, 2000, size=5),
            "Treatment_A": rng.uniform(2000, 4000, size=5),
            "Treatment_B": rng.uniform(500, 1500, size=5),
        })

    @pytest.fixture
    def wb_ref_data(self):
        """WB data: top 3 rows = target protein, bottom 3 rows = reference (GAPDH)."""
        return pd.DataFrame({
            "Control":   [1000, 1200,  900, 500, 600, 450],
            "Treatment": [2000, 2400, 1800, 500, 600, 450],
        })

    @pytest.fixture
    def wb_labeled_single(self):
        """Labeled WB data: 1 target + 1 reference, 3 replicates."""
        labels = pd.Series(["Target-A", "Target-A", "Target-A",
                            "GAPDH", "GAPDH", "GAPDH"])
        df = pd.DataFrame({
            "Control":   [1000, 1200,  900, 500, 600, 450],
            "Treatment": [2000, 2400, 1800, 500, 600, 450],
        })
        return labels, df

    @pytest.fixture
    def wb_labeled_multi(self):
        """Labeled WB data: 2 targets + 1 reference, 3 replicates each."""
        labels = pd.Series([
            "Target-A", "Target-A", "Target-A",
            "Target-B", "Target-B", "Target-B",
            "GAPDH", "GAPDH", "GAPDH",
        ])
        df = pd.DataFrame({
            "Control":   [1000, 1200,  900,
                           800,  900,  750,
                           500,  600,  450],
            "Treatment": [2000, 2400, 1800,
                           400,  350,  380,
                           500,  600,  450],
        })
        return labels, df

    def test_basic_normalization(self, wb_data):
        preset = WBPreset()
        opts = WBOptions(control_group="Control")
        result = preset.transform(wb_data, opts)

        # Control mean should be ~1.0
        assert abs(result["Control"].mean() - 1.0) < 1e-10

        # All columns should be normalized
        assert result.shape == wb_data.shape

    def test_reference_mode(self, wb_ref_data):
        """Legacy top/bottom split still works via transform()."""
        preset = WBPreset()
        opts = WBOptions(control_group="Control", has_reference=True)
        result = preset.transform(wb_ref_data, opts)

        # Should have 3 rows (6 rows split: top 3 target / bottom 3 reference)
        assert result.shape[0] == 3

        # Control mean should be ~1.0
        assert abs(result["Control"].mean() - 1.0) < 1e-10

        # Treatment has higher target/ref ratio → fold > 1
        assert result["Treatment"].mean() > 1.0

    def test_transform_labeled_single_target(self, wb_labeled_single):
        labels, df = wb_labeled_single
        preset = WBPreset()
        opts = WBOptions(control_group="Control")
        results = preset.transform_labeled(labels, df, opts)

        assert len(results) == 1
        name, fold_df = results[0]
        assert name == "Target-A"
        assert fold_df.shape[0] == 3

        # Control mean should be ≈ 1.0
        assert abs(fold_df["Control"].mean() - 1.0) < 1e-10

        # Treatment has higher target/ref ratio → fold > 1
        assert fold_df["Treatment"].mean() > 1.0

    def test_transform_labeled_multi_target(self, wb_labeled_multi):
        labels, df = wb_labeled_multi
        preset = WBPreset()
        opts = WBOptions(control_group="Control")
        results = preset.transform_labeled(labels, df, opts)

        assert len(results) == 2
        names = [r[0] for r in results]
        assert names == ["Target-A", "Target-B"]

        # Both targets: control mean ≈ 1.0
        for _, fold_df in results:
            assert abs(fold_df["Control"].mean() - 1.0) < 1e-10

        # Target-A treatment > 1 (upregulated)
        assert results[0][1]["Treatment"].mean() > 1.0
        # Target-B treatment < 1 (downregulated)
        assert results[1][1]["Treatment"].mean() < 1.0

    def test_transform_labeled_custom_reference(self, wb_labeled_multi):
        labels, df = wb_labeled_multi
        preset = WBPreset()
        # Use Target-B as reference instead of default (GAPDH)
        opts = WBOptions(control_group="Control", reference_protein="Target-B")
        results = preset.transform_labeled(labels, df, opts)

        # Should have 2 targets: Target-A and GAPDH (Target-B is now reference)
        assert len(results) == 2
        names = [r[0] for r in results]
        assert "Target-A" in names
        assert "GAPDH" in names
        assert "Target-B" not in names

        for _, fold_df in results:
            assert abs(fold_df["Control"].mean() - 1.0) < 1e-10

    def test_default_control_is_first_column(self, wb_data):
        preset = WBPreset()
        opts = WBOptions()
        result = preset.transform(wb_data, opts)
        assert abs(result["Control"].mean() - 1.0) < 1e-10

    def test_validation_missing_control(self, wb_data):
        preset = WBPreset()
        opts = WBOptions(control_group="NonExistent")
        with pytest.raises(ValueError, match="not found"):
            preset.transform(wb_data, opts)

    def test_validation_odd_rows_reference_mode(self):
        df = pd.DataFrame({
            "Control": [1.0, 2.0, 3.0],
            "Treatment": [4.0, 5.0, 6.0],
        })
        preset = WBPreset()
        opts = WBOptions(has_reference=True)
        with pytest.raises(ValueError, match="even number"):
            preset.transform(df, opts)


# ── qPCR Preset ─────────────────────────────────────────────────────────

class TestQPCRPreset:
    @pytest.fixture
    def delta_ct_data(self):
        """Pre-computed ΔCt values."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "Control": rng.normal(5.0, 0.5, size=6),
            "KD": rng.normal(8.0, 0.5, size=6),
            "OE": rng.normal(2.0, 0.5, size=6),
        })

    @pytest.fixture
    def raw_ct_data(self):
        """Raw Ct: top 3 rows = target, bottom 3 = reference."""
        return pd.DataFrame({
            "Control": [25.0, 26.0, 24.5, 20.0, 20.5, 19.5],
            "Treatment": [28.0, 29.0, 27.5, 20.0, 20.5, 19.5],
        })

    def test_delta_ct_mode(self, delta_ct_data):
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control", input_format="delta_ct")
        result = preset.transform(delta_ct_data, opts)

        # Control should be ~1.0 (2^0 = 1)
        assert abs(result["Control"].mean() - 1.0) < 0.5

        # KD has higher ΔCt → lower expression → fold < 1
        assert result["KD"].mean() < 1.0

        # OE has lower ΔCt → higher expression → fold > 1
        assert result["OE"].mean() > 1.0

    def test_raw_ct_mode(self, raw_ct_data):
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control", input_format="raw_ct")
        result = preset.transform(raw_ct_data, opts)

        # Should have 3 rows (from 6 split in half)
        assert result.shape[0] == 3

        # Control should be ~1.0
        assert abs(result["Control"].mean() - 1.0) < 0.5

    def test_validation_missing_control(self, delta_ct_data):
        preset = QPCRPreset()
        opts = QPCROptions(control_group="NonExistent")
        with pytest.raises(ValueError, match="not found"):
            preset.transform(delta_ct_data, opts)

    def test_validation_odd_rows_raw_ct(self):
        df = pd.DataFrame({
            "Control": [1.0, 2.0, 3.0],
            "Treatment": [4.0, 5.0, 6.0],
        })
        preset = QPCRPreset()
        opts = QPCROptions(input_format="raw_ct")
        with pytest.raises(ValueError, match="even number"):
            preset.transform(df, opts)

    def test_validation_unknown_format(self, delta_ct_data):
        preset = QPCRPreset()
        opts = QPCROptions(input_format="unknown")
        with pytest.raises(ValueError, match="Unknown input_format"):
            preset.transform(delta_ct_data, opts)

    # ── Labeled mode tests ──

    @pytest.fixture
    def qpcr_labeled_single(self):
        """Labeled qPCR data: 1 target gene + 1 reference, 3 replicates."""
        labels = pd.Series(["Gene-A", "Gene-A", "Gene-A",
                            "GAPDH", "GAPDH", "GAPDH"])
        df = pd.DataFrame({
            "Control":   [25.0, 25.5, 25.2, 18.0, 18.2, 17.8],
            "Treatment": [22.0, 22.3, 21.9, 18.0, 18.2, 17.8],
        })
        return labels, df

    @pytest.fixture
    def qpcr_labeled_multi(self):
        """Labeled qPCR data: 2 target genes + 1 reference, 3 replicates each."""
        labels = pd.Series([
            "Gene-A", "Gene-A", "Gene-A",
            "Gene-B", "Gene-B", "Gene-B",
            "GAPDH", "GAPDH", "GAPDH",
        ])
        df = pd.DataFrame({
            "Control":   [25.0, 25.5, 25.2,
                          28.0, 28.2, 27.8,
                          18.0, 18.2, 17.8],
            "Treatment": [22.0, 22.3, 21.9,
                          30.0, 30.5, 30.2,
                          18.0, 18.2, 17.8],
        })
        return labels, df

    def test_transform_labeled_single_gene(self, qpcr_labeled_single):
        labels, df = qpcr_labeled_single
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control")
        results = preset.transform_labeled(labels, df, opts)

        assert len(results) == 1
        name, fold_df = results[0]
        assert name == "Gene-A"
        assert fold_df.shape[0] == 3

        # Control should be ~1.0
        assert abs(fold_df["Control"].mean() - 1.0) < 0.5

        # Treatment has lower Ct → higher expression → fold > 1
        assert fold_df["Treatment"].mean() > 1.0

    def test_transform_labeled_multi_gene(self, qpcr_labeled_multi):
        labels, df = qpcr_labeled_multi
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control")
        results = preset.transform_labeled(labels, df, opts)

        assert len(results) == 2
        names = [r[0] for r in results]
        assert names == ["Gene-A", "Gene-B"]

        # Both genes: control mean ≈ 1.0
        for _, fold_df in results:
            assert abs(fold_df["Control"].mean() - 1.0) < 0.5

        # Gene-A: treatment Ct lower than control → upregulated (fold > 1)
        assert results[0][1]["Treatment"].mean() > 1.0
        # Gene-B: treatment Ct higher than control → downregulated (fold < 1)
        assert results[1][1]["Treatment"].mean() < 1.0

    def test_transform_labeled_custom_reference(self, qpcr_labeled_multi):
        labels, df = qpcr_labeled_multi
        preset = QPCRPreset()
        # Use Gene-B as reference instead of default (GAPDH)
        opts = QPCROptions(control_group="Control", reference_gene="Gene-B")
        results = preset.transform_labeled(labels, df, opts)

        assert len(results) == 2
        names = [r[0] for r in results]
        assert "Gene-A" in names
        assert "GAPDH" in names
        assert "Gene-B" not in names

        for _, fold_df in results:
            assert abs(fold_df["Control"].mean() - 1.0) < 0.5

    def test_transform_labeled_missing_reference(self):
        labels = pd.Series(["Gene-A", "Gene-A", "GAPDH", "GAPDH"])
        df = pd.DataFrame({"Control": [25, 26, 18, 19], "Treatment": [22, 23, 18, 19]})
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control", reference_gene="NonExistent")
        with pytest.raises(ValueError, match="not found in label column"):
            preset.transform_labeled(labels, df, opts)

    def test_transform_labeled_mismatched_replicates(self):
        labels = pd.Series(["Gene-A", "Gene-A", "Gene-A", "GAPDH", "GAPDH"])
        df = pd.DataFrame({"Control": [25, 26, 27, 18, 19], "Treatment": [22, 23, 24, 18, 19]})
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control")
        with pytest.raises(ValueError, match="replicates"):
            preset.transform_labeled(labels, df, opts)


# ── CCK-8 Preset ────────────────────────────────────────────────────────

class TestCCK8Preset:
    @pytest.fixture
    def cck8_data(self):
        """OD readings: Blank, Control, and 4 dose columns."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "Blank": rng.normal(0.1, 0.02, size=6),
            "Control": rng.normal(1.5, 0.1, size=6),
            "Dose_1": rng.normal(1.4, 0.1, size=6),
            "Dose_2": rng.normal(1.0, 0.1, size=6),
            "Dose_3": rng.normal(0.6, 0.1, size=6),
            "Dose_4": rng.normal(0.2, 0.1, size=6),
        })

    def test_viability_calculation(self, cck8_data):
        preset = CCK8Preset()
        opts = CCK8Options(
            control_group="Control",
            blank_group="Blank",
            fit_ic50=False,
        )
        result = preset.transform(cck8_data, opts)

        # Blank excluded, Control kept as 100% reference
        assert "Blank" not in result.columns
        assert "Control" in result.columns

        # Control viability should be ~100%
        assert abs(result["Control"].mean() - 100.0) < 5.0

        # Dose_1 viability should be close to 100%
        assert result["Dose_1"].mean() > 80

        # Dose_4 viability should be much lower
        assert result["Dose_4"].mean() < result["Dose_1"].mean()

    def test_viability_without_blank(self, cck8_data):
        preset = CCK8Preset()
        opts = CCK8Options(
            control_group="Control",
            blank_group="",
            fit_ic50=False,
        )
        result = preset.transform(cck8_data, opts)
        # Without blank_group set, all columns kept (including Blank as a regular column)
        assert "Control" in result.columns
        assert abs(result["Control"].mean() - 100.0) < 5.0
        assert len(result.columns) == 6  # all original columns

    def test_ic50_fitting(self, cck8_data):
        preset = CCK8Preset()
        concentrations = [0.1, 1.0, 10.0, 100.0]
        opts = CCK8Options(
            control_group="Control",
            blank_group="Blank",
            concentrations=concentrations,
            fit_ic50=True,
        )
        result = preset.transform(cck8_data, opts)

        assert preset.last_result is not None
        assert preset.last_result.ic50 is not None
        assert preset.last_result.ic50 > 0
        assert preset.last_result.fit_params is not None
        assert "method" in preset.last_result.fit_params

    def test_ic50_three_pl(self, cck8_data):
        preset = CCK8Preset()
        concentrations = [0.1, 1.0, 10.0, 100.0]
        opts = CCK8Options(
            control_group="Control",
            blank_group="Blank",
            concentrations=concentrations,
            fit_ic50=True,
            fit_method="three_pl",
        )
        preset.transform(cck8_data, opts)

        assert preset.last_result is not None
        assert preset.last_result.ic50 is not None
        assert preset.last_result.ic50 > 0
        assert preset.last_result.fit_params is not None
        assert preset.last_result.fit_params["method"] == "three_pl"
        assert preset.last_result.fit_params["top"] == 100.0

    def test_ic50_log_linear(self, cck8_data):
        preset = CCK8Preset()
        concentrations = [0.1, 1.0, 10.0, 100.0]
        opts = CCK8Options(
            control_group="Control",
            blank_group="Blank",
            concentrations=concentrations,
            fit_ic50=True,
            fit_method="log_linear",
        )
        preset.transform(cck8_data, opts)

        assert preset.last_result is not None
        assert preset.last_result.ic50 is not None
        assert preset.last_result.ic50 > 0
        assert preset.last_result.fit_params is not None
        assert preset.last_result.fit_params["method"] == "log_linear"

    def test_validation_missing_control(self, cck8_data):
        preset = CCK8Preset()
        opts = CCK8Options(control_group="NonExistent")
        with pytest.raises(ValueError, match="not found"):
            preset.transform(cck8_data, opts)

    def test_validation_missing_blank(self, cck8_data):
        preset = CCK8Preset()
        opts = CCK8Options(blank_group="NonExistent")
        with pytest.raises(ValueError, match="not found"):
            preset.transform(cck8_data, opts)

    def test_last_result_dataclass(self, cck8_data):
        preset = CCK8Preset()
        opts = CCK8Options(
            control_group="Control",
            blank_group="Blank",
            fit_ic50=False,
        )
        preset.transform(cck8_data, opts)
        assert isinstance(preset.last_result, CCK8Result)
        assert isinstance(preset.last_result.viability_df, pd.DataFrame)


# ── Integration: Preset → Stats → Plot ──────────────────────────────────

class TestPresetIntegration:
    """End-to-end: preset transform output feeds into StatsEngine."""

    def test_wb_to_stats(self):
        from xstars.stats_engine import StatsEngine

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Control": rng.uniform(1000, 2000, size=8),
            "Treatment": rng.uniform(2000, 4000, size=8),
        })
        preset = WBPreset()
        opts = WBOptions(control_group="Control")
        transformed = preset.transform(df, opts)

        engine = StatsEngine()
        result = engine.analyze(transformed)
        assert len(result.pairs) >= 1
        assert result.pairs[0].p_value < 1.0

    def test_qpcr_to_stats(self):
        from xstars.stats_engine import StatsEngine

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Control": rng.normal(5.0, 0.3, size=8),
            "KD": rng.normal(8.0, 0.3, size=8),
            "OE": rng.normal(2.0, 0.3, size=8),
        })
        preset = QPCRPreset()
        opts = QPCROptions(control_group="Control")
        transformed = preset.transform(df, opts)

        engine = StatsEngine()
        result = engine.analyze(transformed)
        assert len(result.pairs) >= 1

    def test_wb_to_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        from xstars.plot_engine import PlotEngine
        from xstars.stats_engine import StatsEngine

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Control": rng.uniform(1000, 2000, size=8),
            "Treatment": rng.uniform(2000, 4000, size=8),
        })
        preset = WBPreset()
        opts = WBOptions(control_group="Control")
        transformed = preset.transform(df, opts)

        config = PrismConfig(y_label="Fold Change")
        engine = StatsEngine(config)
        stats = engine.analyze(transformed)

        plotter = PlotEngine(config)
        fig = plotter.plot(transformed, stats)
        assert fig is not None

    def test_cck8_dose_response_chart(self):
        """IC50 fit info triggers dose-response chart with 4PL curve and IC50 markers."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from xstars.plot_engine import PlotEngine

        rng = np.random.default_rng(42)
        concentrations = [0.1, 1.0, 10.0, 100.0]
        df = pd.DataFrame({
            "Control": rng.normal(100, 5, size=6),
            "Dose_1": rng.normal(95, 5, size=6),
            "Dose_2": rng.normal(70, 5, size=6),
            "Dose_3": rng.normal(40, 5, size=6),
            "Dose_4": rng.normal(10, 5, size=6),
        })

        # Run CCK-8 preset to get fit params
        preset = CCK8Preset()
        opts = CCK8Options(
            control_group="Control",
            concentrations=concentrations,
            fit_ic50=True,
        )
        viability = preset.transform(df, opts)
        assert preset.last_result is not None
        assert preset.last_result.fit_params is not None
        assert "method" in preset.last_result.fit_params

        dose_cols = [c for c in viability.columns if c != "Control"]
        fit_info = CCK8FitInfo(
            concentrations=concentrations,
            fit_params=preset.last_result.fit_params,
            dose_col_names=dose_cols,
        )

        config = PrismConfig(y_label="Viability (%)", ic50_fit_info=fit_info)
        plotter = PlotEngine(config)
        fig = plotter.plot(viability)

        assert fig is not None
        ax = fig.axes[0]

        # Should have a log-scale x-axis
        assert ax.get_xscale() == "log"

        # Should have lines (4PL fit curve)
        lines = ax.get_lines()
        assert len(lines) >= 1, "Expected at least one line (4PL fit)"

        # Should have text annotation with IC50
        texts = [t.get_text() for t in ax.texts]
        assert any("IC50" in t for t in texts), f"Expected IC50 annotation, got {texts}"

        plt.close(fig)

    def test_cck8_dose_response_linear_scale(self):
        """Narrow concentration range with LINEAR override uses linear x-axis."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from xstars.plot_engine import PlotEngine

        # Narrow range: 1, 2, 5, 8 — ratio = 8, below 10x threshold
        concentrations = [1.0, 2.0, 5.0, 8.0]
        fit_info = CCK8FitInfo(
            concentrations=concentrations,
            fit_params={"bottom": 5, "top": 100, "ic50": 4.0, "hill": 1.5},
            dose_col_names=["D1", "D2", "D3", "D4"],
        )
        df = pd.DataFrame({
            "Control": [100, 98, 102, 99, 101, 97],
            "D1": [92, 95, 90, 93, 91, 94],
            "D2": [70, 72, 68, 71, 69, 73],
            "D3": [35, 38, 32, 36, 34, 37],
            "D4": [12, 15, 10, 13, 11, 14],
        })

        # Force linear
        config = PrismConfig(
            y_label="Viability (%)",
            ic50_fit_info=fit_info,
            preset_dose_axis_scale=DoseAxisScale.LINEAR,
        )
        plotter = PlotEngine(config)
        fig = plotter.plot(df)
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"
        plt.close(fig)

    def test_cck8_dose_response_auto_selects_linear(self):
        """AUTO mode picks linear when concentration range <= 100x."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from xstars.plot_engine import PlotEngine

        concentrations = [1.0, 2.0, 5.0, 8.0]  # max/min = 8 <= 100
        fit_info = CCK8FitInfo(
            concentrations=concentrations,
            fit_params={"bottom": 5, "top": 100, "ic50": 4.0, "hill": 1.5},
            dose_col_names=["D1", "D2", "D3", "D4"],
        )
        df = pd.DataFrame({
            "Control": [100, 98, 102, 99, 101, 97],
            "D1": [92, 95, 90, 93, 91, 94],
            "D2": [70, 72, 68, 71, 69, 73],
            "D3": [35, 38, 32, 36, 34, 37],
            "D4": [12, 15, 10, 13, 11, 14],
        })

        config = PrismConfig(
            y_label="Viability (%)",
            ic50_fit_info=fit_info,
            preset_dose_axis_scale=DoseAxisScale.AUTO,
        )
        plotter = PlotEngine(config)
        fig = plotter.plot(df)
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"
        plt.close(fig)

    def test_cck8_dose_response_auto_selects_log(self):
        """AUTO mode picks log when concentration range > 100x."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from xstars.plot_engine import PlotEngine

        concentrations = [0.1, 1.0, 10.0, 100.0]  # max/min = 1000
        fit_info = CCK8FitInfo(
            concentrations=concentrations,
            fit_params={"bottom": 5, "top": 100, "ic50": 5.0, "hill": 1.0},
            dose_col_names=["D1", "D2", "D3", "D4"],
        )
        df = pd.DataFrame({
            "Control": [100, 98, 102, 99, 101, 97],
            "D1": [92, 95, 90, 93, 91, 94],
            "D2": [70, 72, 68, 71, 69, 73],
            "D3": [35, 38, 32, 36, 34, 37],
            "D4": [12, 15, 10, 13, 11, 14],
        })

        config = PrismConfig(
            y_label="Viability (%)",
            ic50_fit_info=fit_info,
            preset_dose_axis_scale=DoseAxisScale.AUTO,
        )
        plotter = PlotEngine(config)
        fig = plotter.plot(df)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        plt.close(fig)

    def test_cck8_fit_info_propagation(self):
        """CCK8FitInfo dataclass stores correct values."""
        fit_info = CCK8FitInfo(
            concentrations=[0.1, 1.0, 10.0],
            fit_params={"bottom": 0, "top": 100, "ic50": 5.0, "hill": 1.0},
            dose_col_names=["D1", "D2", "D3"],
        )
        assert fit_info.concentrations == [0.1, 1.0, 10.0]
        assert fit_info.fit_params["ic50"] == 5.0
        assert len(fit_info.dose_col_names) == 3
