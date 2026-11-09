"""Draw pre-computed comparisons as compact brackets above the data.

Brackets are planned in physical units (points) so their spacing does not
depend on the data range, then drawn inside the y-axis. The caller makes room
for them (see ``plot_engine.fit_canvas``) by extending the y-axis within the
canvas height.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from matplotlib import rcParams

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .config import AnnotationFormat, PrismConfig
from .stats_engine import StatsResult

# Vertical geometry, in points.
_BASE_PT = 5.0  # data top -> lowest bracket line (tips drop 4pt of this)
_TIP_PT = 4.0  # length of the downward bracket tips
_TEXT_PT = 1.0  # bracket line -> bottom of its label
_CLEAR_PT = 1.0  # label top -> tips of the bracket above
_TOP_PAD_PT = 2.0  # top label -> end of the y-axis
# Horizontal geometry, in points. Pulling each end in lets brackets that
# share a group (G1-G2, G2-G3) sit in one row without touching.
_INSET_PT = 2.5
_GAP_PT = 4.0


class AnnotationLayoutWarning(UserWarning):
    """So many comparisons that the brackets take more room than the data."""


class AnnotationLayoutError(ValueError):
    """A comparison label cannot fit inside the plot width."""


@dataclass
class _Bracket:
    left: int
    right: int
    text: str
    extent_left: float
    extent_right: float
    lane: int = 0


@dataclass
class BracketPlan:
    brackets: list[_Bracket]
    lanes: int
    font_size: float
    stride_pt: float
    text_height_pt: float

    @property
    def band_pt(self) -> float:
        """Height needed above the data top for every lane, in points."""
        return (
            _BASE_PT
            + (self.lanes - 1) * self.stride_pt
            + _TEXT_PT
            + self.text_height_pt
            + _TOP_PAD_PT
        )


def _format_p_scientific(p: float) -> str:
    """Format a p-value for scientific notation display."""
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.2e}"


def _format_p_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    for threshold, label in ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p <= threshold:
            return label
    return "ns"


def _is_stars(text: str) -> bool:
    return bool(text) and all(c == "*" for c in text)


def _pack_lanes(brackets: list[_Bracket], gap: float) -> int:
    """Assign rows: short brackets low, long ones above what they span.

    Brackets are placed shortest first. Each goes in the lowest row that is
    above every bracket nested inside it and not used by a bracket it would
    overlap (including label width).
    """
    placed: list[_Bracket] = []
    for bracket in sorted(brackets, key=lambda b: (b.right - b.left, b.left)):
        floor = 1 + max(
            (p.lane for p in placed if bracket.left <= p.left and p.right <= bracket.right),
            default=-1,
        )
        busy = {
            p.lane
            for p in placed
            if not (
                p.extent_right + gap < bracket.extent_left
                or bracket.extent_right + gap < p.extent_left
            )
        }
        lane = floor
        while lane in busy:
            lane += 1
        bracket.lane = lane
        placed.append(bracket)
    return 1 + max((b.lane for b in brackets), default=-1)


def layout_notice(figure: Any) -> str:
    """Host-visible notice, also retained for non-interactive worker results."""
    return str(getattr(figure, "_xstars_layout_notice", ""))


def plan_brackets(
    ax: Axes,
    long_df: pd.DataFrame,
    stats_result: StatsResult,
    config: PrismConfig | None = None,
) -> BracketPlan | None:
    """Measure labels and pack the selected comparisons into rows.

    Returns None when no comparison is shown. Uses the current horizontal
    layout, so call it after the figure's margins are settled.
    """
    cfg = config or PrismConfig()
    hidden = {frozenset(pair) for pair in cfg.hidden_comparisons}
    selected = [
        (pair, p)
        for pair, p in zip(stats_result.pair_tuples, stats_result.p_values, strict=True)
        if (cfg.show_ns or p < cfg.alpha) and frozenset(pair) not in hidden
    ]
    if not selected:
        return None

    figure = ax.get_figure()
    assert figure is not None
    canvas = cast(Any, figure.canvas)
    canvas.draw()
    renderer = canvas.get_renderer()
    px_per_pt = figure.dpi / 72.0
    # Group order is the plotting order, not alphabetical order.
    positions = {
        group: i for i, group in enumerate(pd.unique(cast(pd.Series, long_df["Group"])))
    }
    scientific = cfg.annotation_format is AnnotationFormat.SCIENTIFIC
    font_size = float(rcParams["font.size"]) * (0.9 if scientific else 1.1)
    inset_px = _INSET_PT * px_per_pt

    brackets: list[_Bracket] = []
    text_height_px = 0.0
    for (group_a, group_b), p in selected:
        if group_a not in positions or group_b not in positions:
            raise ValueError("A comparison refers to a group missing from the plot.")
        left, right = sorted((positions[group_a], positions[group_b]))
        text = _format_p_scientific(p) if scientific else _format_p_stars(p)
        probe = ax.text(
            0, 0, text, fontsize=font_size,
            fontweight="bold" if _is_stars(text) else "normal",
        )
        bounds = probe.get_window_extent(renderer)
        probe.remove()
        text_height_px = max(text_height_px, bounds.height)
        left_px = ax.transData.transform((left, 0))[0] + inset_px
        right_px = ax.transData.transform((right, 0))[0] - inset_px
        center_px = (left_px + right_px) / 2
        bracket = _Bracket(
            left, right, text,
            min(left_px, center_px - bounds.width / 2),
            max(right_px, center_px + bounds.width / 2),
        )
        if bracket.extent_left < ax.bbox.x0 or bracket.extent_right > ax.bbox.x1:
            raise AnnotationLayoutError(
                "Comparison labels are too wide for this canvas. Increase figure width, "
                "choose a wider Panel bar width, or use stars instead of scientific "
                "p-values. No comparisons were removed."
            )
        brackets.append(bracket)

    lanes = _pack_lanes(brackets, _GAP_PT * px_per_pt)
    text_height_pt = text_height_px / px_per_pt
    stride_pt = _TIP_PT + _TEXT_PT + text_height_pt + _CLEAR_PT
    return BracketPlan(brackets, lanes, font_size, stride_pt, text_height_pt)


def draw_brackets(ax: Axes, plan: BracketPlan, data_top: float) -> None:
    """Draw *plan* in data coordinates, starting just above *data_top*.

    The figure size and y-limits must be final: points are converted to data
    units once, which stays exact at any export DPI.
    """
    figure = ax.get_figure()
    assert figure is not None
    bottom, top = ax.get_ylim()
    data_per_pt = (top - bottom) / (ax.bbox.height * 72.0 / figure.dpi)
    x_per_pt = 1.0 / (
        (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
        * 72.0 / figure.dpi
    )
    inset = _INSET_PT * x_per_pt
    color = rcParams["text.color"]
    for bracket in plan.brackets:
        y = data_top + (_BASE_PT + bracket.lane * plan.stride_pt) * data_per_pt
        tip = _TIP_PT * data_per_pt
        x0, x1 = bracket.left + inset, bracket.right - inset
        line, = ax.plot(
            [x0, x0, x1, x1], [y - tip, y, y, y - tip],
            color=color, linewidth=rcParams["axes.linewidth"],
            clip_on=False, scalex=False, scaley=False,
        )
        line.set_gid("significance_bracket")
        label = ax.text(
            (bracket.left + bracket.right) / 2, y + _TEXT_PT * data_per_pt,
            bracket.text, ha="center", va="bottom", fontsize=plan.font_size,
            color=color, clip_on=False,
            fontweight="bold" if _is_stars(bracket.text) else "normal",
        )
        label.set_gid("significance_label")


def record_layout(ax: Axes, plan: BracketPlan, data_top: float, warn: bool = True) -> None:
    """Store layout facts and warn when brackets outweigh the data."""
    bottom, top = ax.get_ylim()
    data_fraction = (data_top - bottom) / (top - bottom)
    cast(Any, ax)._xstars_annotation_layout = {
        "lanes": plan.lanes,
        "comparisons": len(plan.brackets),
        "data_height_fraction": float(data_fraction),
    }
    if warn and data_fraction < 0.5:
        notice = (
            f"Dense annotations ({len(plan.brackets)} comparisons, {plan.lanes} rows). "
            "Try a taller figure, hide ns, or compare against a control for more data space."
        )
        figure = ax.get_figure()
        cast(Any, figure)._xstars_layout_notice = notice
        warnings.warn(notice, AnnotationLayoutWarning, stacklevel=3)
