"""Contract tests for the host-independent WPS/Excel application boundary."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from xstars import main
from xstars.application import (
    MAX_SELECTION_COLUMNS,
    MAX_SELECTION_ROWS,
    SCHEMA_VERSION,
    Artifact,
    ArtifactFormat,
    Command,
    ContractError,
    ErrorCode,
    ErrorDTO,
    ImageWriteback,
    SelectionPayload,
    TableWriteback,
    WritebackPlan,
    cell_to_a1,
    ensure_path_within,
    parse_cell,
)


class ContractTests(unittest.TestCase):
    def test_command_whitelist_matches_zero_argument_public_entries(self):
        public_entries = {
            name
            for name, function in inspect.getmembers(main, inspect.isfunction)
            if name == "run" or (
                name.startswith("run_")
                and all(
                    parameter.default is not inspect.Parameter.empty
                    or parameter.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                    for parameter in inspect.signature(function).parameters.values()
                )
            )
        }
        self.assertEqual({command.value for command in Command}, public_entries)
        with self.assertRaises(ValueError):
            Command("__import__('os').system('whoami')")

    def test_selection_json_round_trip_preserves_unicode_and_null(self):
        payload = SelectionPayload(
            values=[["对照", "处理"], [1.0, None], [2, 3.5]],
            address="$A$1:$B$3",
            sheet="数据",
        )
        decoded = json.loads(json.dumps(payload.to_dict(), ensure_ascii=False))
        self.assertEqual(SelectionPayload.from_dict(decoded), payload)
        self.assertEqual(decoded["version"], SCHEMA_VERSION)

    def test_selection_rejects_ragged_nonfinite_and_oversized_values(self):
        invalid_values = (
            [["A", "B"], [1]],
            [["A"], [float("nan")]],
            [["A"], [object()]],
            [[0] for _ in range(MAX_SELECTION_ROWS + 1)],
            [list(range(MAX_SELECTION_COLUMNS + 1))],
        )
        for values in invalid_values:
            with self.subTest(values_type=type(values)), self.assertRaises(ContractError):
                SelectionPayload(values=values, address="A1:A1", sheet="Data")

    def test_selection_rejects_address_mismatch_and_multiple_areas(self):
        with self.assertRaises(ContractError):
            SelectionPayload(values=[[1, 2]], address="A1:A1", sheet="Data")
        with self.assertRaises(ContractError):
            SelectionPayload(values=[[1, 2]], address="A1:B1,D1:E1", sheet="Data")

    def test_writeback_plan_json_round_trip(self):
        root = Path(tempfile.gettempdir()) / "xstars-contracts"
        artifact = Artifact(
            path=str(root / "chart.png"),
            format=ArtifactFormat.PNG,
            dpi=600,
        )
        plan = WritebackPlan(
            tables=[TableWriteback(start_cell="C10", values=[["p-value"], [0.01]])],
            images=[
                ImageWriteback(
                    anchor_cell="C14",
                    name="XSTARS_Plot_1",
                    artifact=artifact,
                    width=320,
                    height=240,
                )
            ],
            status_message="XSTARS: complete",
        )
        decoded = json.loads(json.dumps(plan.to_dict()))
        self.assertEqual(WritebackPlan.from_dict(decoded), plan)
        self.assertEqual(artifact.validate_under(root), (root / "chart.png").resolve())

    def test_in_process_figure_source_is_explicit_not_executable(self):
        image = ImageWriteback(
            anchor_cell="A1",
            name="XSTARS_Plot_1",
            source_key="primary_figure",
        )
        self.assertEqual(image.to_dict()["sourceKey"], "primary_figure")
        with self.assertRaises(ContractError):
            ImageWriteback(anchor_cell="A1", name="bad", source_key="os.system('x')")

    def test_artifact_path_format_dpi_and_controlled_root_boundaries(self):
        root = Path(tempfile.gettempdir()) / "xstars-artifacts"
        outside = root.parent / "outside.png"
        valid = Artifact(str(root / "chart.tiff"), ArtifactFormat.TIFF, 300)
        self.assertEqual(valid.validate_under(root), (root / "chart.tiff").resolve())

        invalid = (
            ("relative.png", ArtifactFormat.PNG, 300),
            (str(root / "chart.exe"), ArtifactFormat.PNG, 300),
            (str(root / "chart.png"), ArtifactFormat.PNG, 71),
            (str(root / "chart.png"), ArtifactFormat.PNG, 1201),
        )
        for path, format_, dpi in invalid:
            with self.subTest(path=path, dpi=dpi), self.assertRaises(ContractError):
                Artifact(path, format_, dpi)
        with self.assertRaises(ContractError):
            ensure_path_within(outside, root)
        with self.assertRaises(ContractError):
            ensure_path_within(root / ".." / "outside.png", root)

    def test_error_dto_has_stable_code_and_json_round_trip(self):
        error = ErrorDTO(
            code=ErrorCode.INVALID_SELECTION,
            message="Selection must be rectangular",
            details={"rows": 2, "retryable": False},
        )
        decoded = json.loads(json.dumps(error.to_dict()))
        self.assertEqual(ErrorDTO.from_dict(decoded), error)
        self.assertEqual(decoded["code"], "INVALID_SELECTION")

    def test_cell_conversion_is_stable(self):
        self.assertEqual(parse_cell("$XFD$1048576"), (1_048_576, 16_384))
        self.assertEqual(cell_to_a1(1_048_576, 16_384), "XFD1048576")
        with self.assertRaises(ContractError):
            cell_to_a1(1, 16_385)


if __name__ == "__main__":
    unittest.main()
