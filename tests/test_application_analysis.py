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
from xstars.config import PrismConfig
from xstars.data_handler import DataHandler

_analysis = import_module("xstars.application.analysis")
analyze_selection = _analysis.analyze_selection
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
