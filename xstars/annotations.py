"""Add significance brackets to seaborn axes using statannotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes

from .config import AnnotationFormat, PrismConfig
from .stats_engine import StatsResult


def _format_p_scientific(p: float) -> str:
    """Format a p-value for scientific notation display."""
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.2e}"


def annotate(
    ax: "matplotlib.axes.Axes",
    long_df: pd.DataFrame,
    stats_result: StatsResult,
    config: PrismConfig | None = None,
) -> "matplotlib.axes.Axes":
    """Draw significance brackets on *ax* based on pre-computed *stats_result*.

    Uses ``statannotations.Annotator`` with ``set_pvalues_and_annotate``
    so that no re-computation happens.
    """
    import matplotlib as mpl
    from statannotations.Annotator import Annotator

    cfg = config or PrismConfig()

    pairs = stats_result.pair_tuples
    p_values = stats_result.p_values

    if not pairs:
        return ax

    # Filter out ns pairs if user doesn't want them
    if not cfg.show_ns:
        filtered = [
            (pair, p) for pair, p in zip(pairs, p_values) if p < cfg.alpha
        ]
        if not filtered:
            return ax
        pairs, p_values = zip(*filtered)
        pairs, p_values = list(pairs), list(p_values)

    # Scale annotation parameters to match current typography
    base_font = mpl.rcParams.get("font.size", 10)
    line_w = mpl.rcParams.get("axes.linewidth", 1.0)
    star_fontsize = base_font * 1.4   # stars slightly larger than base text
    sci_fontsize = base_font * 0.9    # p-values slightly smaller
    bracket_lw = line_w               # bracket matches axis line width

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=long_df,
        x="Group",
        y="Value",
        verbose=False,
    )

    if cfg.annotation_format == AnnotationFormat.SCIENTIFIC:
        annotator.configure(
            test=None,
            text_format="simple",
            loc="inside",
            line_height=0.02,
            line_width=bracket_lw,
            text_offset=0.5,
            fontsize=sci_fontsize,
        )
        custom_texts = [_format_p_scientific(p) for p in p_values]
        annotator.set_custom_annotations(custom_texts)
        annotator.annotate()
    else:
        annotator.configure(
            test=None,
            text_format="star",
            loc="inside",
            line_height=0.02,
            line_width=bracket_lw,
            text_offset=0.5,
            fontsize=star_fontsize,
        )
        annotator.set_pvalues_and_annotate(p_values)

    # Bold the significance stars (not "ns")
    for txt in ax.texts:
        content = txt.get_text().strip()
        if content and content != "ns" and all(c == "*" for c in content):
            txt.set_fontsize(star_fontsize)
            txt.set_fontweight("bold")

    return ax
