"""Tests for PrismConfig save/load persistence."""

import json
from pathlib import Path

from xstars.config import ChartType, ErrorBarType, AnnotationFormat, PrismConfig


class TestConfigPersistence:
    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "settings.json"
        cfg = PrismConfig(
            chart_type=ChartType.VIOLIN,
            error_bar=ErrorBarType.SD,
            show_points=False,
            paired=True,
            alpha=0.01,
            annotation_format=AnnotationFormat.SCIENTIFIC,
            y_label="Fold Change",
            show_ns=False,
        )
        cfg.save(p)
        loaded = PrismConfig.load(p)
        assert loaded.chart_type == ChartType.VIOLIN
        assert loaded.error_bar == ErrorBarType.SD
        assert loaded.show_points is False
        assert loaded.paired is True
        assert loaded.alpha == 0.01
        assert loaded.annotation_format == AnnotationFormat.SCIENTIFIC
        assert loaded.y_label == "Fold Change"
        assert loaded.show_ns is False

    def test_transient_fields_not_persisted(self, tmp_path: Path):
        p = tmp_path / "settings.json"
        cfg = PrismConfig(export_path="/some/path.png", control_group="CtrlA")
        cfg.save(p)
        data = json.loads(p.read_text())
        assert "export_path" not in data
        assert "control_group" not in data

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        loaded = PrismConfig.load(tmp_path / "nope.json")
        assert loaded == PrismConfig()

    def test_load_corrupt_file_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all")
        loaded = PrismConfig.load(p)
        assert loaded == PrismConfig()

    def test_load_partial_file(self, tmp_path: Path):
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"chart_type": "violin"}))
        loaded = PrismConfig.load(p)
        assert loaded.chart_type == ChartType.VIOLIN
        # Other fields should be defaults
        assert loaded.error_bar == ErrorBarType.SEM
