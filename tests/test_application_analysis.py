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

from xstars.application import SelectionPayload, WritebackPlan
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
            [result.writeback_plan.tables[index].values[0][0] for index in (0, 2, 4, 6)],
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
            self.assertEqual(list(target.transformed_data.columns), [
                "Control", "Treatment_A", "Treatment_B"
            ])
            self.assertEqual(len(target.transformed_data), 3)
        self.assertIn("2 gene(s) analyzed", result.writeback_plan.status_message)

    def test_selection_analysis_returns_stats_figure_and_writeback_plan(self):
        result = analyze_selection(
            _selection(),
            PrismConfig(output_stats=True, output_data=False),
            output_start_cell="D10",
            image_name="XSTARS_Plot_7",
        )

        self.assertEqual(list(result.transformed_data.columns), ["Control", "Treatment"])
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
                    values=[["Control", "Treatment"], [1.0, 0.0], [1.2, 0.2], [0.8, -0.1]],
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
                result = analyze_selection(selection, config, output_start_cell="F8", include_processed_data=True)
                self.assertGreaterEqual(len(result.writeback_plan.tables), 2)
                self.assertIn(expected_label, config.y_label)
                self.assertIsNotNone(result.preset)

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
        self.assertEqual(curve.writeback_plan.tables[0].values, [["Standard Curve Results"]])
        self.assertEqual(curve.writeback_plan.images[0].source_key, "primary_figure")

        elisa = elisa_selections(standard, samples, config, output_start_cell="A8")
        self.assertEqual(config.experiment_preset, ExperimentPreset.ELISA)
        self.assertEqual(list(elisa.transformed_data.columns), ["Control", "Treatment"])
        self.assertEqual(elisa.writeback_plan.tables[0].values, [["Standard Curve Results"]])
        self.assertIn("ELISA (linear", elisa.writeback_plan.status_message)

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
