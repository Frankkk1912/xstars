"""Default configuration for XSTARS analysis and plotting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

DEFAULT_SETTINGS_PATH = Path.home() / ".xstars" / "settings.json"


class BaseTheme(Enum):
    CLASSIC = "classic"    # white bg, no grid, left/bottom spines
    BW = "bw"              # white bg, all spines, light grid
    MINIMAL = "minimal"    # no spines, light grid
    DARK = "dark"          # dark bg, light text, subtle grid


class PalettePreset(Enum):
    DEFAULT = "default"        # original journal colors, no adjustment
    PASTEL = "pastel"          # lighten + desaturate
    DEEP = "deep"              # darken
    VIBRANT = "vibrant"        # boost saturation
    MUTED = "muted"            # desaturate
    COLORBLIND = "colorblind"  # Wong colorblind-safe (overrides base colors)


class JournalPalette(Enum):
    DEFAULT = "default"     # D3 category10
    NATURE = "nature"       # Nature Publishing Group
    SCIENCE = "science"     # Science (AAAS)
    CELL = "cell"           # Cell Press
    LANCET = "lancet"       # Lancet
    NEJM = "nejm"           # NEJM
    JAMA = "jama"           # JAMA
    BMJ = "bmj"             # BMJ


class AnnotationFormat(Enum):
    STARS = "stars"            # "***"
    SCIENTIFIC = "scientific"  # "p=1.2e-4"


class JournalPreset(Enum):
    NONE = "none"
    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"
    LANCET = "lancet"
    NEJM = "nejm"
    JAMA = "jama"
    BMJ = "bmj"


class ExperimentPreset(Enum):
    NONE = "none"
    WB = "wb"
    QPCR = "qpcr"
    CCK8 = "cck8"
    ELISA = "elisa"


class ChartType(Enum):
    BAR_SCATTER = "bar_scatter"
    VIOLIN = "violin"
    LINE = "line"


class ErrorBarType(Enum):
    SEM = "sem"
    SD = "sd"
    CI95 = "ci95"


class DoseAxisScale(Enum):
    AUTO = "auto"      # log if range > 100×, linear otherwise
    LOG = "log"
    LINEAR = "linear"


class FitMethod(Enum):
    AUTO = "auto"            # try 3PL first, fallback to log-linear
    THREE_PL = "three_pl"    # 3-parameter logistic (top fixed at 100)
    LOG_LINEAR = "log_linear"  # log-linear interpolation


@dataclass
class PrismConfig:
    """All user-configurable settings for a single analysis run."""

    # Chart
    chart_type: ChartType = ChartType.BAR_SCATTER
    error_bar: ErrorBarType = ErrorBarType.SEM
    show_points: bool = True
    point_size: float = 5.0
    point_alpha: float = 0.7

    # Statistics
    paired: bool = False
    alpha: float = 0.05
    annotation_format: AnnotationFormat = AnnotationFormat.STARS
    control_group: str | None = None  # None = all pairwise; group name = compare all vs control

    # Labels
    y_label: str = "Value"
    x_label: str = ""
    title: str = ""

    # Theme
    journal_preset: JournalPreset = JournalPreset.NONE
    base_theme: BaseTheme = BaseTheme.CLASSIC
    palette_preset: PalettePreset = PalettePreset.DEFAULT
    journal_palette: JournalPalette = JournalPalette.DEFAULT

    # Figure
    fig_width: float = 4.0
    fig_height: float = 3.5
    dpi: int = 300

    # Colors — default Prism-like palette
    palette: list[str] = field(
        default_factory=lambda: [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]
    )

    # Significance display
    show_ns: bool = True  # show "ns" for non-significant pairs

    # Output options
    output_stats: bool = True   # write stats table to Excel
    output_data: bool = True    # write processed data table to Excel

    # Export
    export_path: str = ""  # empty = no file export; set to path to save chart
    export_format: str = "png"  # png, tiff, jpg, svg, pdf
    export_dpi: int = 300  # 150, 300, 600, 1200

    # Experiment preset
    experiment_preset: ExperimentPreset = ExperimentPreset.NONE
    preset_control_group: str = ""
    preset_has_reference: bool = False       # WB: labeled reference protein mode
    preset_reference_protein: str = ""       # WB: which protein is the reference
    preset_input_format: str = "delta_ct"    # qPCR: "delta_ct" or "raw_ct"
    preset_reference_gene: str = ""          # qPCR: which gene is the reference
    preset_blank_group: str = ""             # CCK-8: blank column name
    preset_fit_ic50: bool = True             # CCK-8: fit 4PL curve
    preset_concentrations: str = ""          # CCK-8: comma-separated concentrations
    preset_dose_axis_scale: DoseAxisScale = DoseAxisScale.AUTO  # CCK-8: x-axis scale
    preset_fit_method: FitMethod = FitMethod.AUTO  # CCK-8: IC50 fitting method

    # ELISA
    preset_elisa_fit_method: str = "auto"          # standard_curve fit method
    preset_elisa_use_existing_params: bool = False  # skip std curve, use user params
    preset_elisa_existing_params: str = ""          # JSON string of param dict

    # CCK-8 IC50 dose-response (transient, not persisted)
    ic50_fit_info: object | None = None

    # ELISA fit result (transient, not persisted)
    elisa_fit_result: object | None = None

    # xlwings insertion
    insert_offset_cols: int = 2  # columns to the right of selection

    # ---- persistence ----

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        d = asdict(self)
        # Convert enums to their string values
        d["chart_type"] = self.chart_type.value
        d["error_bar"] = self.error_bar.value
        d["annotation_format"] = self.annotation_format.value
        d["journal_preset"] = self.journal_preset.value
        d["base_theme"] = self.base_theme.value
        d["palette_preset"] = self.palette_preset.value
        d["journal_palette"] = self.journal_palette.value
        d["experiment_preset"] = self.experiment_preset.value
        # Drop transient fields that shouldn't persist
        for key in (
            "export_path", "control_group",
            "preset_control_group", "preset_has_reference",
            "preset_reference_protein", "preset_reference_gene",
            "preset_input_format", "preset_blank_group", "preset_fit_ic50",
            "preset_concentrations", "preset_dose_axis_scale", "preset_fit_method",
            "ic50_fit_info",
            "preset_elisa_fit_method", "preset_elisa_use_existing_params",
            "preset_elisa_existing_params", "elisa_fit_result",
        ):
            d.pop(key, None)
        return d

    def save(self, path: Path | None = None) -> None:
        """Save current settings to a JSON file."""
        path = path or DEFAULT_SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> PrismConfig:
        """Load settings from JSON, falling back to defaults for missing keys."""
        path = path or DEFAULT_SETTINGS_PATH
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        # Map enum strings back
        enum_map = {
            "chart_type": ChartType,
            "error_bar": ErrorBarType,
            "annotation_format": AnnotationFormat,
            "journal_preset": JournalPreset,
            "base_theme": BaseTheme,
            "palette_preset": PalettePreset,
            "journal_palette": JournalPalette,
            "experiment_preset": ExperimentPreset,
        }
        kwargs: dict = {}
        defaults = cls()
        for f in cls.__dataclass_fields__:
            if f in (
                "export_path", "control_group",
                "preset_control_group", "preset_has_reference",
                "preset_reference_protein", "preset_reference_gene",
                "preset_input_format", "preset_blank_group", "preset_fit_ic50",
                "ic50_fit_info",
                "preset_elisa_fit_method", "preset_elisa_use_existing_params",
                "preset_elisa_existing_params", "elisa_fit_result",
            ):
                continue
            if f in raw:
                val = raw[f]
                if f in enum_map:
                    try:
                        val = enum_map[f](val)
                    except ValueError:
                        val = getattr(defaults, f)
                kwargs[f] = val
        return cls(**kwargs)
