"""Publication-quality matplotlib style presets — three-layer architecture.

Layers (merged in order, later overrides earlier):
  1. BASE_RCPARAMS — universal defaults (font family, tick direction, etc.)
  2. BASE_THEMES  — visual style (background, spines, grid)
  3. JOURNAL_PRESETS — typographic specs (font size, line width, fig size)

Palette is orthogonal and returned separately via ``get_palette()``.
"""

from __future__ import annotations

import colorsys

import matplotlib as mpl
import matplotlib.colors as mcolors

from .config import BaseTheme, JournalPalette, JournalPreset, PalettePreset

# ── Layer 1: universal defaults (theme-independent) ──────────────────────

BASE_RCPARAMS: dict = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    # Axes
    "axes.linewidth": 1.0,

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,

    # Legend
    "legend.frameon": False,

    # Lines / markers
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# ── Layer 2: Base Themes — visual style ──────────────────────────────────

BASE_THEMES: dict[BaseTheme, dict] = {
    BaseTheme.CLASSIC: {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
    },
    BaseTheme.BW: {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
    },
    BaseTheme.MINIMAL: {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
    },
    BaseTheme.DARK: {
        "axes.facecolor": "#2d2d2d",
        "figure.facecolor": "#1e1e1e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#444444",
        "grid.linewidth": 0.5,
    },
}

# ── Layer 3: Journal Presets — typographic parameters ────────────────────

JOURNAL_PRESETS: dict[JournalPreset, dict] = {
    JournalPreset.NATURE: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        },
        "fig_width": 3.5,   # 89mm single column
        "fig_height": 3.0,
    },
    JournalPreset.SCIENCE: {
        "rcparams": {
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
        },
        "fig_width": 3.54,  # 9cm single column
        "fig_height": 3.0,
    },
    JournalPreset.CELL: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "lines.linewidth": 1.25,
            "lines.markersize": 4,
        },
        "fig_width": 3.35,  # 85mm single column
        "fig_height": 3.0,
    },
    JournalPreset.LANCET: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
        },
        "fig_width": 3.54,  # 90mm single column
        "fig_height": 3.0,
    },
    JournalPreset.NEJM: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        },
        "fig_width": 3.5,   # 88mm single column
        "fig_height": 3.0,
    },
    JournalPreset.JAMA: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
        },
        "fig_width": 3.5,   # 89mm single column
        "fig_height": 3.0,
    },
    JournalPreset.BMJ: {
        "rcparams": {
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
        },
        "fig_width": 3.54,  # 90mm single column
        "fig_height": 3.0,
    },
}

# ── Color adjustment utilities ───────────────────────────────────────────

# Wong colorblind-safe palette (used when PalettePreset.COLORBLIND)
_COLORBLIND_PALETTE: list[str] = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#999999",
]


def _hex_to_hls(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to HLS (hue, lightness, saturation)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, l: float, s: float) -> str:
    """Convert HLS back to hex color string."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r, g, b))


def _adjust_colors(colors: list[str], preset: PalettePreset) -> list[str]:
    """Apply color style adjustment to a list of base hex colors."""
    if preset == PalettePreset.DEFAULT:
        return colors
    if preset == PalettePreset.COLORBLIND:
        return list(_COLORBLIND_PALETTE)

    result = []
    for c in colors:
        h, l, s = _hex_to_hls(c)
        if preset == PalettePreset.PASTEL:
            l = min(1.0, l * 1.3 + 0.15)
            s = s * 0.6
        elif preset == PalettePreset.DEEP:
            l = max(0.0, l * 0.7)
        elif preset == PalettePreset.VIBRANT:
            s = min(1.0, s * 1.4)
            l = min(0.95, max(0.15, l * 0.95))
        elif preset == PalettePreset.MUTED:
            s = s * 0.55
        result.append(_hls_to_hex(h, l, s))
    return result


# ── Journal palettes (ggsci-inspired) ────────────────────────────────────

JOURNAL_PALETTES: dict[JournalPalette, list[str]] = {
    JournalPalette.DEFAULT: [
        "#1F77B4", "#D62728", "#2CA02C", "#FF7F0E",
        "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
        "#BCBD22", "#17BECF",
    ],
    JournalPalette.NATURE: [
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
        "#7E6148", "#B09C85",
    ],
    JournalPalette.SCIENCE: [
        "#3B4992", "#EE0000", "#008B45", "#631879",
        "#008280", "#BB0021", "#5F559B", "#A20056",
        "#808180", "#1B1919",
    ],
    JournalPalette.CELL: [
        "#635547", "#C72228", "#F9DEC9", "#2E6E9E",
        "#FACB12", "#6C9146", "#139992", "#EF4E22",
        "#8DB5CE", "#B51D8D",
    ],
    JournalPalette.LANCET: [
        "#00468B", "#ED0000", "#42B540", "#0099B4",
        "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6",
        "#1B1919",
    ],
    JournalPalette.NEJM: [
        "#BC3C29", "#0072B5", "#E18727", "#20854E",
        "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97",
    ],
    JournalPalette.JAMA: [
        "#374E55", "#DF8F44", "#00A1D5", "#B24745",
        "#79AF97", "#4A6990", "#80796B",
    ],
    JournalPalette.BMJ: [
        "#2166AC", "#B2182B", "#1B7837", "#762A83",
        "#E08214", "#80CDC1", "#F4A582", "#999999",
    ],
}

# ── Public API ───────────────────────────────────────────────────────────


def get_palette(
    preset: PalettePreset = PalettePreset.DEFAULT,
    journal_palette: JournalPalette = JournalPalette.DEFAULT,
) -> list[str]:
    """Return the colour list: journal base colors adjusted by color style."""
    base = list(JOURNAL_PALETTES.get(journal_palette, JOURNAL_PALETTES[JournalPalette.DEFAULT]))
    return _adjust_colors(base, preset)


def get_theme_rcparams(
    base_theme: BaseTheme = BaseTheme.CLASSIC,
    journal_preset: JournalPreset = JournalPreset.NONE,
) -> dict:
    """Merge: BASE_RCPARAMS → base_theme overrides → journal overrides."""
    rc = dict(BASE_RCPARAMS)
    rc.update(BASE_THEMES.get(base_theme, BASE_THEMES[BaseTheme.CLASSIC]))
    if journal_preset in JOURNAL_PRESETS:
        rc.update(JOURNAL_PRESETS[journal_preset]["rcparams"])
    return rc


def get_journal_figsize(
    journal_preset: JournalPreset = JournalPreset.NONE,
) -> tuple[float, float] | None:
    """Return (width, height) recommended by the journal, or None."""
    if journal_preset in JOURNAL_PRESETS:
        jp = JOURNAL_PRESETS[journal_preset]
        return (jp["fig_width"], jp["fig_height"])
    return None


def get_prism_context(
    preset: JournalPreset = JournalPreset.NONE,
    base_theme: BaseTheme = BaseTheme.CLASSIC,
) -> mpl.rc_context:
    """Return a context manager that temporarily applies the merged style."""
    return mpl.rc_context(rc=get_theme_rcparams(base_theme, preset))


# Legacy helper
def apply_prism_style() -> None:
    """Apply default Prism-like rcParams globally."""
    mpl.rcParams.update(get_theme_rcparams())
