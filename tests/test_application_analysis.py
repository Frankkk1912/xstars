"""Tests for host-independent analysis and WritebackPlan construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from xstars.application import ContractError, ErrorCode, SelectionPayload, WritebackPlan
from xstars.config import ExperimentPreset, PrismConfig
from xstars.data_handler import DataHandler

_analysis = import_module("xstars.application.analysis")
analyze_selection = _analysis.analyze_selection
elisa_selections = _analysis.elisa_selections
split_selection_labels = _analysis.split_selection_labels
standard_curve_selection = _analysis.standard_curve_selection
transform_selection = _analysis.transform_selection


def _selection() -> SelectionPayload:
    return SelectionPayload(
        values=[
            ["Control", "Treatment"],
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
        ],
        address="$A$1:$B$4",
        sheet="Data",
    )


def _wb_labeled_selection(*, target_b_replicates: int = 3) -> SelectionPayload:
    rows = [
        ["Target-A", 12000, 28000, 6500],
        ["Target-A", 15000, 31000, 7200],
        ["Target-A", 11500, 26500, 5800],
        ["Target-B", 8000, 12000, 4000],
        ["Target-B", 9200, 13500, 3800],
        ["Target-B", 7800, 11000, 4200],
        ["GAPDH", 45000, 44000, 43000],
        ["GAPDH", 47000, 46000, 45000],
        ["GAPDH", 43000, 43500, 42500],
    ]
    if target_b_replicates == 2:
        rows.pop(5)
    return SelectionPayload(
        values=[["Protein", "Control", "Treatment_A", "Treatment_B"], *rows],
        address=f"A1:D{len(rows) + 1}",
        sheet="WB",
    )


def _qpcr_labeled_selection() -> SelectionPayload:
    return SelectionPayload(
        values=[
            ["Gene", "Control", "Treatment_A", "Treatment_B"],
            ["Gene-A", 25.0, 27.0, 24.0],
            ["Gene-A", 25.2, 27.1, 24.2],
            ["Gene-A", 24.8, 26.9, 23.8],
            ["Gene-B", 23.0, 24.5, 22.0],
            ["Gene-B", 23.2, 24.7, 22.1],
            ["Gene-B", 22.8, 24.3, 21.9],
            ["GAPDH", 20.0, 20.1, 20.0],
            ["GAPDH", 20.1, 20.0, 20.2],
            ["GAPDH", 19.9, 20.2, 19.8],
        ],
        address="A1:D10",
        sheet="qPCR",
    )


class ApplicationAnalysisTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_data_handler_constructs_clean_frame_from_two_dimensional_values(self):
        frame = DataHandler.from_values(
            [[" Control ", "Treatment"], [1, "2.5"], ["bad", None]]
        )
        self.assertEqual(list(frame.columns), ["Control", "Treatment"])
        self.assertEqual(frame.iloc[0].tolist(), [1.0, 2.5])
        self.assertEqual(len(frame), 1)

    def test_label_detection_uses_excel_threshold_and_preserves_numeric_alignment(self):
        labels, numeric = split_selection_labels(
            _wb_labeled_selection(target_b_replicates=2)
        )

        self.assertIsNotNone(labels)
        self.assertEqual(labels.tolist().count("Target-A"), 3)
        self.assertEqual(labels.tolist().count("Target-B"), 2)
        self.assertEqual(labels.tolist().count("GAPDH"), 3)
        self.assertEqual(
            list(numeric.columns), ["Control", "Treatment_A", "Treatment_B"]
        )
        self.assertEqual(len(numeric), len(labels))

        no_labels, ordinary = split_selection_labels(_selection())
        self.assertIsNone(no_labels)
        self.assertEqual(list(ordinary.columns), ["Control", "Treatment"])

    def test_wb_labeled_selection_builds_per_target_tables_and_images(self):
        config = PrismConfig(
            experiment_preset=ExperimentPreset.WB,
            preset_has_reference=True,
            preset_reference_protein="GAPDH",
            preset_control_group="Control",
            output_stats=True,
            output_data=True,
        )

        result = analyze_selection(
            _wb_labeled_selection(),
            config,
            output_start_cell="A13",
            include_processed_data=True,
        )

        self.assertEqual(
            [target.name for target in result.target_results],
            ["Target-A", "Target-B"],
        )
        self.assertEqual(
            [image.name for image in result.writeback_plan.images],
            ["XSTARS_Plot_Target-A", "XSTARS_Plot_Target-B"],
        )
        self.assertEqual(
            [image.source_key for image in result.writeback_plan.images],
            ["target_figure_1", "target_figure_2"],
        )
        self.assertNotEqual(
            result.writeback_plan.images[0].anchor_cell,
            result.writeback_plan.images[1].anchor_cell,
        )
        self.assertEqual(
            [
                result.writeback_plan.tables[index].values[0][0]
                for index in (0, 2, 4, 6)
            ],
            [
                "Target-A",
                "Processed Data — Target-A",
                "Target-B",
                "Processed Data — Target-B",
            ],
        )
        self.assertEqual(
            [target.render_config.title for target in result.target_results],
            ["Target-A", "Target-B"],
        )
        self.assertIn("2 target(s) analyzed", result.writeback_plan.status_message)

    def test_qpcr_labeled_selection_uses_reference_gene_per_target(self):
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_has_reference=True,
            preset_reference_gene="GAPDH",
            preset_control_group="Control",
            preset_input_format="raw_ct",
            output_stats=False,
            output_data=True,
        )

        result = analyze_selection(
            _qpcr_labeled_selection(),
            config,
            output_start_cell="F13",
            include_processed_data=True,
        )

        self.assertEqual(
            [target.name for target in result.target_results],
            ["Gene-A", "Gene-B"],
        )
        self.assertEqual(len(result.writeback_plan.images), 2)
        self.assertEqual(
            [table.values[0][0] for table in result.writeback_plan.tables[::2]],
            ["Processed Data — Gene-A", "Processed Data — Gene-B"],
        )
        for target in result.target_results:
            self.assertEqual(
                list(target.transformed_data.columns),
                ["Control", "Treatment_A", "Treatment_B"],
            )
            self.assertEqual(len(target.transformed_data), 3)
        self.assertIn("2 gene(s) analyzed", result.writeback_plan.status_message)

    def test_selection_analysis_returns_stats_figure_and_writeback_plan(self):
        result = analyze_selection(
            _selection(),
            PrismConfig(output_stats=True, output_data=False),
            output_start_cell="D10",
            image_name="XSTARS_Plot_7",
        )

        self.assertEqual(
            list(result.transformed_data.columns), ["Control", "Treatment"]
        )
        self.assertEqual(len(result.stats_result.pairs), 1)
        self.assertIsNotNone(result.figure)
        self.assertEqual(result.writeback_plan.tables[0].start_cell, "D10")
        self.assertEqual(result.writeback_plan.images[0].anchor_cell, "D13")
        self.assertEqual(result.writeback_plan.images[0].name, "XSTARS_Plot_7")
        self.assertEqual(result.writeback_plan.images[0].source_key, "primary_figure")
        self.assertTrue(result.writeback_plan.status_message.startswith("XSTARS: "))

        encoded = json.dumps(result.writeback_plan.to_dict())
        self.assertEqual(
            WritebackPlan.from_dict(json.loads(encoded)), result.writeback_plan
        )

    def test_analysis_can_generate_requested_figure_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chart.png"
            config = PrismConfig(export_path=str(output), export_dpi=300)
            result = analyze_selection(
                _selection(),
                config,
                output_start_cell="A8",
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertIsNotNone(result.figure)

    def test_wb_qpcr_and_cck8_presets_use_shared_analysis_pipeline(self):
        cases = [
            (
                ExperimentPreset.WB,
                _selection(),
                {"preset_control_group": "Control"},
                "Fold Change",
            ),
            (
                ExperimentPreset.QPCR,
                SelectionPayload(
                    values=[
                        ["Control", "Treatment"],
                        [1.0, 0.0],
                        [1.2, 0.2],
                        [0.8, -0.1],
                    ],
                    address="A1:B4",
                    sheet="qPCR",
                ),
                {"preset_control_group": "Control", "preset_input_format": "delta_ct"},
                "Relative Expression",
            ),
            (
                ExperimentPreset.CCK8,
                SelectionPayload(
                    values=[
                        ["Blank", "Control", "Dose1", "Dose2"],
                        [0.1, 1.0, 0.8, 0.4],
                        [0.1, 1.1, 0.75, 0.35],
                        [0.1, 0.9, 0.7, 0.3],
                    ],
                    address="A1:D4",
                    sheet="CCK8",
                ),
                {
                    "preset_control_group": "Control",
                    "preset_blank_group": "Blank",
                    "preset_fit_ic50": False,
                },
                "Viability",
            ),
        ]
        for preset, selection, overrides, expected_label in cases:
            with self.subTest(preset=preset.value):
                config = PrismConfig(experiment_preset=preset, **overrides)
                result = analyze_selection(
                    selection,
                    config,
                    output_start_cell="F8",
                    include_processed_data=True,
                )
                self.assertGreaterEqual(len(result.writeback_plan.tables), 2)
                self.assertIn(expected_label, config.y_label)
                self.assertIsNotNone(result.preset)

    def test_elisa_rejects_standard_headers_without_od_rows_with_context(self):
        standard = SelectionPayload(
            values=[[0, 15.6, 31.2]],
            address="B4:D4",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42]],
            address="L3:M5",
            sheet="ELISA",
        )

        with self.assertRaises(ContractError) as caught:
            elisa_selections(
                standard,
                samples,
                PrismConfig(preset_elisa_fit_method="linear"),
                output_start_cell="A8",
            )

        self.assertEqual(caught.exception.code, ErrorCode.ANALYSIS_FAILED)
        message = str(caught.exception)
        self.assertIn("没有 OD 数据行", message)
        self.assertIn("收到 1 行 × 3 列", message)
        self.assertIn("'0', '15.6', '31.2'", message)

    def test_elisa_rejects_text_standard_headers_with_selection_context(self):
        standard = SelectionPayload(
            values=[
                ["ELISA standard curve", "Instructions"],
                [0.1, 1.0],
                [0.2, 2.0],
            ],
            address="B1:C3",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42]],
            address="L3:M5",
            sheet="ELISA",
        )

        with self.assertRaises(ContractError) as caught:
            elisa_selections(
                standard,
                samples,
                PrismConfig(preset_elisa_fit_method="linear"),
                output_start_cell="A8",
            )

        self.assertEqual(caught.exception.code, ErrorCode.ANALYSIS_FAILED)
        message = str(caught.exception)
        self.assertIn("列头无法解析为浓度数值", message)
        self.assertIn("收到 3 行 × 2 列", message)
        self.assertIn("不要包含标题或说明文字", message)

    def test_elisa_rejects_fewer_than_two_standard_columns(self):
        standard = SelectionPayload(
            values=[[10], [0.1], [0.2]],
            address="B4:B6",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42]],
            address="L3:M5",
            sheet="ELISA",
        )

        with self.assertRaises(ContractError) as caught:
            elisa_selections(
                standard,
                samples,
                PrismConfig(preset_elisa_fit_method="linear"),
                output_start_cell="A8",
            )

        self.assertEqual(caught.exception.code, ErrorCode.ANALYSIS_FAILED)
        message = str(caught.exception)
        self.assertIn("浓度数值列不足 2 列", message)
        self.assertIn("'10'", message)

    def test_elisa_reports_effective_point_count_before_curve_fit(self):
        standard = SelectionPayload(
            values=[[1, 10], [0.5, None]],
            address="B4:C5",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42]],
            address="L3:M5",
            sheet="ELISA",
        )

        with self.assertRaises(ContractError) as caught:
            elisa_selections(
                standard,
                samples,
                PrismConfig(preset_elisa_fit_method="linear"),
                output_start_cell="A8",
            )

        self.assertEqual(caught.exception.code, ErrorCode.ANALYSIS_FAILED)
        message = str(caught.exception)
        self.assertIn("实际收到 1 个", message)
        self.assertIn("清洗后 1 行 × 2 列", message)
        self.assertIn("收到 2 行 × 2 列", message)

    def test_standard_curve_and_two_selection_elisa_are_host_independent(self):
        standard = SelectionPayload(
            values=[
                [1, 10, 100],
                [0.1, 1.0, 10.0],
                [0.11, 1.1, 10.1],
                [0.09, 0.9, 9.9],
            ],
            address="A1:C4",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[
                ["Control", "Treatment"],
                [0.2, 0.4],
                [0.21, 0.42],
                [0.19, 0.38],
            ],
            address="E1:F4",
            sheet="ELISA",
        )
        config = PrismConfig(preset_elisa_fit_method="linear")

        curve = standard_curve_selection(standard, config, output_start_cell="A8")
        self.assertEqual(curve.fit_result.method, "linear")
        self.assertEqual(
            curve.writeback_plan.tables[0].values, [["Standard Curve Results"]]
        )
        self.assertEqual(curve.writeback_plan.images[0].source_key, "primary_figure")

        elisa = elisa_selections(standard, samples, config, output_start_cell="A8")
        self.assertEqual(config.experiment_preset, ExperimentPreset.ELISA)
        self.assertEqual(list(elisa.transformed_data.columns), ["Control", "Treatment"])
        self.assertEqual(
            elisa.writeback_plan.tables[0].values, [["Standard Curve Results"]]
        )
        self.assertIn("ELISA (linear", elisa.writeback_plan.status_message)

    def test_standard_curve_selected_fit_can_back_calculate_samples(self):
        standard = SelectionPayload(
            values=[[1, 10, 100], [0.1, 1.0, 10.0], [0.11, 1.1, 10.1]],
            address="A1:C3",
            sheet="Curve",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.3, 0.5]],
            address="E1:F3",
            sheet="Curve",
        )
        selected = standard_curve_selection(
            standard,
            PrismConfig(preset_elisa_fit_method="linear"),
            output_start_cell="A6",
        )
        result = standard_curve_selection(
            standard,
            PrismConfig(preset_elisa_fit_method="auto"),
            output_start_cell="A6",
            fit_result=selected.fit_result,
            sample_payload=samples,
        )
        self.assertEqual(result.fit_result.method, "linear")
        self.assertEqual(
            result.writeback_plan.tables[2].values, [["Back-Calculated Concentrations"]]
        )
        self.assertEqual(
            result.writeback_plan.tables[3].values[0], ["Control", "Treatment"]
        )
        self.assertIn("back-calculated", result.writeback_plan.status_message)

    def test_elisa_can_reuse_dialog_fit_and_append_standard_curve_image(self):
        standard = SelectionPayload(
            values=[[1, 10, 100], [0.1, 1.0, 10.0], [0.11, 1.1, 10.1]],
            address="A1:C3",
            sheet="ELISA",
        )
        samples = SelectionPayload(
            values=[["Control", "Treatment"], [0.2, 0.4], [0.21, 0.42], [0.19, 0.38]],
            address="E1:F4",
            sheet="ELISA",
        )
        config = PrismConfig(preset_elisa_fit_method="linear")
        selected = standard_curve_selection(standard, config, output_start_cell="A6")
        result = elisa_selections(
            standard,
            samples,
            config,
            output_start_cell="A6",
            fit_result=selected.fit_result,
            show_fit_curve=True,
        )
        self.assertEqual(len(result.writeback_plan.images), 2)
        self.assertEqual(
            result.writeback_plan.images[1].source_key, "standard_curve_figure"
        )
        self.assertIn("standard_curve_figure", result.figure_sources)
        self.assertEqual(
            list(result.render_data_sources["standard_curve_figure"].columns),
            ["1", "10", "100"],
        )

    def test_transform_only_can_include_statistics(self):
        result = transform_selection(
            _selection(),
            PrismConfig(),
            output_start_cell="E5",
            include_stats=True,
        )
        self.assertEqual(result.writeback_plan.tables[0].start_cell, "E5")
        self.assertEqual(result.writeback_plan.tables[1].values, [["Processed Data"]])
        self.assertIn("with statistics", result.writeback_plan.status_message)

    def test_transform_only_labeled_wb_builds_per_target_tables_without_images(self):
        result = transform_selection(
            _wb_labeled_selection(),
            PrismConfig(
                experiment_preset=ExperimentPreset.WB,
                preset_has_reference=True,
                preset_reference_protein="GAPDH",
                preset_control_group="Control",
            ),
            output_start_cell="A13",
            include_stats=True,
        )
        self.assertEqual(
            [name for name, _frame in result.target_data], ["Target-A", "Target-B"]
        )
        self.assertEqual(
            [table.values[0][0] for table in result.writeback_plan.tables[::2]],
            [
                "Statistics — Target-A",
                "Processed Data — Target-A",
                "Statistics — Target-B",
                "Processed Data — Target-B",
            ],
        )
        self.assertEqual(result.writeback_plan.images, [])

    def test_transform_only_returns_data_table_plan_without_host_calls(self):
        result = transform_selection(
            _selection(),
            PrismConfig(output_data=True),
            output_start_cell="E5",
        )
        self.assertEqual(result.writeback_plan.tables[0].start_cell, "E5")
        self.assertEqual(result.writeback_plan.tables[0].values, [["Processed Data"]])
        self.assertEqual(result.writeback_plan.tables[1].start_cell, "E6")
        self.assertEqual(
            result.writeback_plan.tables[1].values[0], ["Control", "Treatment"]
        )
        self.assertEqual(result.writeback_plan.images, [])
        self.assertEqual(
            result.writeback_plan.status_message,
            "XSTARS: Transform only — data written",
        )


if __name__ == "__main__":
    unittest.main()


class QPCRStatsSpaceTests(unittest.TestCase):
    """qPCR hypothesis tests run on the linear log2 fold-change (−ΔΔCt) space,
    while processed data and the y-axis remain in 2^-ΔΔCt fold-change units."""

    def tearDown(self):
        plt.close("all")

    def test_engine_pvalues_match_manual_log2_space_analysis(self):
        scipy_stats = import_module("scipy.stats")
        np_module = import_module("numpy")

        payload = SelectionPayload(
            values=[
                ["Control", "siRNA", "OE"],
                [0.0, 3.3, -2.8],
                [0.4, 3.1, -2.9],
                [-0.2, 3.4, -2.7],
            ],
            address="A1:C4",
            sheet="qPCR",
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_input_format="delta_ct",
            preset_control_group="Control",
        )

        result = analyze_selection(payload, config, output_start_cell="A10")

        # Fold-change space is what gets written back and plotted…
        self.assertTrue((result.transformed_data > 0).all().all())
        # …but the decision tree runs on the linear −ΔΔCt space.
        self.assertIn("ANOVA", result.stats_result.decision_path)

        log2fc = np_module.log2(result.transformed_data.to_numpy(dtype=float))
        tukey = scipy_stats.tukey_hsd(log2fc[:, 0], log2fc[:, 1], log2fc[:, 2])
        engine_ps = [pair.p_value for pair in result.stats_result.pairs]
        manual_ps = [
            float(tukey.pvalue[0][1]),
            float(tukey.pvalue[0][2]),
            float(tukey.pvalue[1][2]),
        ]
        self.assertEqual(len(engine_ps), len(manual_ps))
        for engine_p, manual_p in zip(engine_ps, manual_ps, strict=True):
            self.assertAlmostEqual(engine_p, manual_p, delta=1e-12)

    def test_labeled_mode_pvalues_match_manual_log2_space_analysis(self):
        scipy_stats = import_module("scipy.stats")
        np_module = import_module("numpy")

        # Every group has variable ΔCt so the parametric path is taken;
        # a constant-ΔCt group would legitimately fall back to Kruskal-Wallis.
        payload = SelectionPayload(
            values=[
                ["Gene", "Control", "Treatment_A", "Treatment_B"],
                ["Gene-A", 25.0, 26.9, 24.0],
                ["Gene-A", 25.5, 27.5, 24.4],
                ["Gene-A", 24.7, 27.1, 23.7],
                ["GAPDH", 20.0, 20.1, 20.0],
                ["GAPDH", 20.2, 20.0, 20.2],
                ["GAPDH", 19.9, 20.4, 19.9],
            ],
            address="A1:D7",
            sheet="qPCR",
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_has_reference=True,
            preset_reference_gene="GAPDH",
            preset_control_group="Control",
        )
        result = analyze_selection(
            payload,
            config,
            output_start_cell="A13",
        )

        self.assertEqual(len(result.target_results), 1)
        for target in result.target_results:
            self.assertIn("ANOVA", target.stats_result.decision_path)
            log2fc = np_module.log2(target.transformed_data.to_numpy(dtype=float))
            columns = list(target.transformed_data.columns)
            tukey = scipy_stats.tukey_hsd(*[log2fc[:, i] for i in range(len(columns))])
            for pair, (i, j) in zip(
                target.stats_result.pairs, [(0, 1), (0, 2), (1, 2)], strict=True
            ):
                self.assertAlmostEqual(
                    pair.p_value, float(tukey.pvalue[i][j]), delta=1e-12
                )

    def test_qpcr_geo_stats_derive_from_log_space(self):
        np_module = import_module("numpy")
        pd_module = import_module("pandas")
        plot_engine = import_module("xstars.plot_engine")

        config = PrismConfig(experiment_preset=ExperimentPreset.QPCR)
        engine = plot_engine.PlotEngine(config)

        # Fold-change replicates whose log2 values are 0.0, 1.0, 2.0
        series = pd_module.Series([1.0, 2.0, 4.0])
        geo, lower, upper = engine._qpcr_geo_stats(series)

        self.assertAlmostEqual(geo, 2.0)  # 2^(mean([0,1,2])) = 2
        # SEM in log space = 0.5774 → asymmetric endpoints 2^(1±0.5774)
        sem_log = np_module.std([0.0, 1.0, 2.0], ddof=1) / np_module.sqrt(3)
        self.assertAlmostEqual(geo - 2.0 ** (1.0 - sem_log), lower)
        self.assertAlmostEqual(2.0 ** (1.0 + sem_log) - geo, upper)
        self.assertGreater(upper, lower)  # right-skewed on linear axis


class QPCRLogSpaceExtendedTests(unittest.TestCase):
    """Extended coverage: transform_selection stats path, nonparametric fallback,
    and _qpcr_bars tick-label regression (Fix 1)."""

    def tearDown(self):
        plt.close("all")

    # ------------------------------------------------------------------
    # Fix 3a — transform_selection(include_stats=True) runs in log space
    # ------------------------------------------------------------------

    def test_transform_selection_plain_stats_run_in_log_space(self):
        scipy_stats = import_module("scipy.stats")
        np_module = import_module("numpy")

        payload = SelectionPayload(
            values=[
                ["Control", "siRNA", "OE"],
                [0.0, 3.3, -2.8],
                [0.4, 3.1, -2.9],
                [-0.2, 3.4, -2.7],
            ],
            address="A1:C4",
            sheet="qPCR",
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_input_format="delta_ct",
            preset_control_group="Control",
        )
        result = transform_selection(
            payload, config, output_start_cell="A10", include_stats=True
        )

        # Stats table must be present
        self.assertTrue(len(result.writeback_plan.tables) >= 1)

        # Recover transformed data from the result and cross-check p-values
        analysis_result = analyze_selection(payload, config, output_start_cell="A10")
        log2fc = np_module.log2(
            analysis_result.transformed_data.to_numpy(dtype=float)
        )
        tukey = scipy_stats.tukey_hsd(log2fc[:, 0], log2fc[:, 1], log2fc[:, 2])

        engine_ps = [pair.p_value for pair in analysis_result.stats_result.pairs]
        manual_ps = [
            float(tukey.pvalue[0][1]),
            float(tukey.pvalue[0][2]),
            float(tukey.pvalue[1][2]),
        ]
        for engine_p, manual_p in zip(engine_ps, manual_ps, strict=True):
            self.assertAlmostEqual(engine_p, manual_p, delta=1e-12)

    def test_transform_selection_labeled_stats_run_in_log_space(self):
        scipy_stats = import_module("scipy.stats")
        np_module = import_module("numpy")

        payload = SelectionPayload(
            values=[
                ["Gene", "Control", "Treatment_A", "Treatment_B"],
                ["Gene-A", 25.0, 26.9, 24.0],
                ["Gene-A", 25.5, 27.5, 24.4],
                ["Gene-A", 24.7, 27.1, 23.7],
                ["GAPDH", 20.0, 20.1, 20.0],
                ["GAPDH", 20.2, 20.0, 20.2],
                ["GAPDH", 19.9, 20.4, 19.9],
            ],
            address="A1:D7",
            sheet="qPCR",
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_has_reference=True,
            preset_reference_gene="GAPDH",
            preset_control_group="Control",
        )
        result = transform_selection(
            payload, config, output_start_cell="A13", include_stats=True
        )
        # Stats table written back for the labeled path
        self.assertTrue(len(result.writeback_plan.tables) >= 1)

        # Cross-check via analyze_selection which also uses the log-space path
        analysis_result = analyze_selection(
            payload, config, output_start_cell="A13"
        )
        self.assertEqual(len(analysis_result.target_results), 1)
        for target in analysis_result.target_results:
            log2fc = np_module.log2(target.transformed_data.to_numpy(dtype=float))
            columns = list(target.transformed_data.columns)
            tukey = scipy_stats.tukey_hsd(
                *[log2fc[:, i] for i in range(len(columns))]
            )
            for pair, (i, j) in zip(
                target.stats_result.pairs, [(0, 1), (0, 2), (1, 2)], strict=True
            ):
                self.assertAlmostEqual(
                    pair.p_value, float(tukey.pvalue[i][j]), delta=1e-12
                )

    # ------------------------------------------------------------------
    # Fix 3b — constant-ΔCt group legitimately falls back to Kruskal-Wallis
    # ------------------------------------------------------------------

    def test_constant_dct_group_falls_back_to_nonparametric(self):
        """A group with zero variance in log space (constant ΔCt) has zero
        variance in log2 fold-change too, triggering the nonparametric
        fallback.  This is a property of the input data — the log-space
        migration must not break it.  Three groups are used so the
        decision tree reaches Kruskal-Wallis rather than Mann-Whitney U.
        """
        payload = SelectionPayload(
            values=[
                # Control has constant ΔCt → zero variance → Levene p ≈ 0
                ["Control", "siRNA", "OE"],
                [5.0, 7.5, 3.0],
                [5.0, 8.0, 2.5],
                [5.0, 7.8, 2.8],
                [5.0, 7.9, 2.7],
            ],
            address="A1:C5",
            sheet="qPCR",
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            preset_input_format="delta_ct",
            preset_control_group="Control",
        )
        result = analyze_selection(payload, config, output_start_cell="A10")
        self.assertIn("Kruskal", result.stats_result.decision_path)

    # ------------------------------------------------------------------
    # Fix 3c — _qpcr_bars sets x tick labels even when show_points=False
    # ------------------------------------------------------------------

    def test_qpcr_bar_tick_labels_without_show_points(self):
        pd_module = import_module("pandas")
        plot_engine_mod = import_module("xstars.plot_engine")

        groups = ["Control", "siRNA", "OE"]
        df_wide = pd_module.DataFrame(
            {
                "Control": [1.0, 0.9, 1.1],
                "siRNA": [0.4, 0.5, 0.45],
                "OE": [2.8, 3.1, 2.9],
            }
        )
        config = PrismConfig(
            experiment_preset=ExperimentPreset.QPCR,
            show_points=False,
        )
        engine = plot_engine_mod.PlotEngine(config)
        fig = engine.plot(df_wide)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(tick_labels, groups,
            msg=f"Expected group name ticks {groups}, got {tick_labels}")
        plt.close(fig)
