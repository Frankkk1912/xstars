"""Generate publication-quality figures (bar+scatter, violin, line)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, MultipleLocator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from . import annotations as ann
from .config import (
    BarWidth,
    CanvasMode,
    ChartType,
    DoseAxisScale,
    ErrorBarType,
    ExperimentPreset,
    PrismConfig,
    QPCRErrorSpace,
    YAxisScale,
)
from .stats_engine import StatsResult
from .styles import get_journal_figsize, get_prism_context


def export_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    """Save the full canvas. Format is inferred from extension by matplotlib."""
    fig.savefig(path, **canvas_export_options(fig, dpi))


def canvas_export_options(fig: Figure, dpi: int = 200) -> dict[str, Any]:
    """Explicit full-canvas bounds, including for xlwings' figure adapter."""
    # A Bbox also defeats a caller's global savefig.bbox='tight'. None would
    # fall back to that rcParam, and xlwings defaults to tight cropping.
    return {"bbox_inches": fig.bbox_inches, "dpi": dpi}


# Fit-data mode: space kept between the top bracket label and the title.
_TOP_PAD_ABOVE_AXIS_PT = 4.0

# Plot-area width per group in Panel mode, in inches.
PANEL_BAR_WIDTH_IN = {
    BarWidth.NARROW: 0.5,
    BarWidth.MEDIUM: 0.7,
    BarWidth.WIDE: 0.9,
}


def resolve_figsize(config: PrismConfig) -> tuple[float, float]:
    """Starting canvas size: configured, or the journal preset's.

    Fixed mode keeps it. Panel mode keeps the height, and ``fit_canvas``
    sets the width from the number of groups.
    """
    journal_size = get_journal_figsize(config.journal_preset)
    defaults = PrismConfig()
    size = (
        journal_size
        if journal_size
        and config.fig_width == defaults.fig_width
        and config.fig_height == defaults.fig_height
        else (config.fig_width, config.fig_height)
    )
    if not all(np.isfinite(value) and value > 0 for value in size):
        raise ValueError("Figure width and height must be finite positive numbers.")
    return size


def _nice_tops(bottom: float, top: float) -> list[tuple[float, float]]:
    """(candidate top, step) pairs near *top* that give 3-7 tick intervals.

    Steps are 1, 2, 2.5 or 5 times a power of ten; each step offers the tick
    just below and just above *top*.
    """
    span = top - bottom
    magnitude = int(np.floor(np.log10(span))) if span > 0 else 0
    found = []
    for exponent in range(magnitude - 2, magnitude + 2):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0**exponent
            for candidate in (np.floor(top / step + 1e-9), np.ceil(top / step - 1e-9)):
                candidate = float(candidate * step)
                if 3 <= (candidate - bottom) / step <= 7:
                    found.append((candidate, step))
    return found


def _apply_top(ax: Axes, bottom: float, choice: tuple[float, float] | None, top: float) -> float:
    if choice is None:
        ax.set_ylim(bottom, top)
        return top
    ax.set_ylim(bottom, choice[0])
    ax.yaxis.set_major_locator(MultipleLocator(choice[1]))
    return choice[0]


def _round_up_to_tick(ax: Axes, bottom: float, top: float) -> float:
    """Raise the y-limit to the lowest round value at or above *top*.

    Ticks the axis at the chosen step so the axis line ends on a label.
    Ties go to the larger step (fewer ticks).
    """
    above = [c for c in _nice_tops(bottom, top) if c[0] >= top - 1e-12]
    choice = min(above, key=lambda c: (c[0], -c[1]), default=None)
    return _apply_top(ax, bottom, choice, top)


def _scale_to_data(ax: Axes, fill: float = 0.8) -> float:
    """Set a round y-limit so the data fills about *fill* of the axis.

    The tallest value (bars, error bars, points) keeps at least 5% headroom.
    """
    bottom = ax.get_ylim()[0]
    data_max = float(ax.dataLim.y1)
    target = bottom + (data_max - bottom) / fill
    minimum = bottom + (data_max - bottom) * 1.05
    usable = [c for c in _nice_tops(bottom, target) if c[0] >= minimum]
    choice = min(usable, key=lambda c: (abs(c[0] - target), -c[1]), default=None)
    return _apply_top(ax, bottom, choice, max(target, minimum))


def _grow_top(fig: Figure, ax: Axes, extra_pt: float) -> None:
    """Add *extra_pt* above the plot area without changing its size."""
    width, height = fig.get_size_inches()
    position = ax.get_position()
    new_height = height + extra_pt / 72.0
    fig.set_size_inches(width, new_height)
    ax.set_position((
        position.x0, position.y0 * height / new_height,
        position.width, position.height * height / new_height,
    ))
    if ax.get_title():
        ax.set_title(ax.get_title(), y=1.0 + extra_pt / 72.0 / (position.height * height))


def _room_above_pt(ax: Axes, data_top: float) -> float:
    """Physical space between *data_top* and the top of the y-axis."""
    bottom, top = ax.get_ylim()
    height_pt = ax.bbox.height * 72.0 / ax.figure.dpi
    return (top - data_top) / (top - bottom) * height_pt


def _fit_panel_width(fig: Figure, ax: Axes, plot_width_in: float) -> None:
    """Set the canvas width so the plot area is *plot_width_in* wide."""
    for _ in range(4):
        width, height = fig.get_size_inches()
        new_width = plot_width_in + width * (1.0 - ax.get_position().width)
        if abs(new_width - width) < 1e-3:
            return
        fig.set_size_inches(new_width, height)
        fig.tight_layout()


def fit_canvas(
    fig: Figure,
    ax: Axes,
    config: PrismConfig,
    band_pt: float = 0.0,
    data_limits: tuple[float, float] | None = None,
    n_groups: int | None = None,
) -> tuple[float, float]:
    """Final layout: size the canvas and make room for brackets.

    Fixed keeps the canvas size. Panel keeps the height and sets the width
    so every group gets the same plot-area width (``bar_width``); charts
    without groups keep the configured width. In both, *band_pt* of
    brackets fit inside the height by extending the y-axis to a round
    tick. The height only grows when brackets would leave the data under
    20% of the plot.

    Returns the (bottom, data top) limits the brackets are placed above.
    """
    panel_width_in = (
        n_groups * PANEL_BAR_WIDTH_IN[config.bar_width]
        if config.canvas_mode is CanvasMode.PANEL and n_groups
        else None
    )

    def settle() -> None:
        fig.tight_layout()
        if panel_width_in is not None:
            _fit_panel_width(fig, ax, panel_width_in)

    settle()
    bottom, data_top = data_limits or ax.get_ylim()
    if band_pt <= 0:
        return bottom, data_top
    span = data_top - bottom
    for _ in range(4):
        height_pt = ax.bbox.height * 72.0 / fig.dpi
        share = band_pt / height_pt
        if share > 0.8:
            width, height = fig.get_size_inches()
            fig.set_size_inches(width, height + (band_pt / 0.8 - height_pt) / 72.0)
            settle()
            continue
        _round_up_to_tick(ax, bottom, bottom + span / (1.0 - share))
        settle()
        if _room_above_pt(ax, data_top) >= band_pt - 0.01:
            break
    return bottom, data_top


class PlotEngine:
    """Create a figure from wide-format data + stats results."""

    def __init__(self, config: PrismConfig | None = None) -> None:
        self.config = config or PrismConfig()

    def plot(
        self,
        df_wide: pd.DataFrame,
        stats_result: StatsResult | None = None,
    ) -> Figure:
        """Dispatch to the correct chart builder and return the Figure."""
        cfg = self.config

        # IC50 dose-response mode: bypass normal chart builders
        if cfg.ic50_fit_info is not None:
            return self._dose_response_figure(df_wide)

        chart_builders = {
            ChartType.BAR_SCATTER: self._bar_scatter,
            ChartType.VIOLIN: self._violin,
            ChartType.LINE: self._line,
        }
        builder = chart_builders[cfg.chart_type]

        with get_prism_context(cfg.journal_preset, cfg.base_theme):
            fw, fh = self._resolve_figsize()
            fig, ax = plt.subplots(figsize=(fw, fh))
            long_df = df_wide.melt(var_name="Group", value_name="Value").dropna(
                subset=["Value"]
            )
            builder(ax, long_df, df_wide)

            # Labels
            ax.set_xlabel(cfg.x_label)
            ax.set_ylabel(cfg.y_label)
            if cfg.title:
                ax.set_title(cfg.title)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            try:
                self._layout_with_brackets(fig, ax, long_df, stats_result)
            except ValueError:
                plt.close(fig)
                raise

        return fig

    def _layout_with_brackets(
        self,
        fig: Figure,
        ax: Axes,
        long_df: pd.DataFrame,
        stats_result: StatsResult | None,
    ) -> None:
        """Fit the canvas, then draw brackets in the space it reserved."""
        cfg = self.config
        n_groups = int(long_df["Group"].nunique())
        # Width first: bracket rows depend on where the groups sit.
        fit_canvas(fig, ax, cfg, n_groups=n_groups)
        plan = (
            ann.plan_brackets(ax, long_df, stats_result, cfg)
            if stats_result is not None
            else None
        )
        if cfg.y_axis_scale is YAxisScale.FIT_DATA:
            self._brackets_above_data(fig, ax, long_df, stats_result, n_groups)
            return
        if plan is None:
            return
        limits = fit_canvas(fig, ax, cfg, plan.band_pt, n_groups=n_groups)
        # Tick labels can change width after the y-axis grows, shifting the
        # groups slightly; re-measure and refit if that changes the rows.
        final = ann.plan_brackets(ax, long_df, cast(StatsResult, stats_result), cfg)
        assert final is not None
        if final.lanes != plan.lanes:
            limits = fit_canvas(fig, ax, cfg, final.band_pt, limits, n_groups)
        ann.draw_brackets(ax, final, limits[1])
        ann.record_layout(ax, final, limits[1])

    def _brackets_above_data(
        self,
        fig: Figure,
        ax: Axes,
        long_df: pd.DataFrame,
        stats_result: StatsResult | None,
        n_groups: int,
    ) -> None:
        """Fit-data y-axis: brackets never shrink the bars.

        The y-axis is scaled to the data alone. Brackets start just above
        the tallest value; any that do not fit under the axis top continue
        above it, and the canvas grows by that amount.
        """
        cfg = self.config
        _scale_to_data(ax)
        fit_canvas(fig, ax, cfg, n_groups=n_groups)
        if stats_result is None:
            return
        plan = ann.plan_brackets(ax, long_df, stats_result, cfg)
        if plan is None:
            return
        data_top = float(ax.dataLim.y1)
        overflow_pt = plan.band_pt - _TOP_PAD_ABOVE_AXIS_PT - _room_above_pt(ax, data_top)
        if overflow_pt > 0:
            _grow_top(fig, ax, overflow_pt + _TOP_PAD_ABOVE_AXIS_PT)
        ann.draw_brackets(ax, plan, data_top)
        ann.record_layout(ax, plan, data_top, warn=False)

    # ------------------------------------------------------------------
    # Chart types
    # ------------------------------------------------------------------

    def _bar_scatter(
        self,
        ax: Axes,
        long_df: pd.DataFrame,
        df_wide: pd.DataFrame,
    ) -> None:
        cfg = self.config
        groups = list(df_wide.columns)

        if self._qpcr_log_display():
            self._qpcr_bars(ax, df_wide, groups)
        else:
            ci_param = self._seaborn_ci()

            sns.barplot(
                data=long_df,
                x="Group",
                y="Value",
                hue="Group",
                order=groups,
                hue_order=groups,
                palette=cfg.palette[: len(groups)],
                errorbar=ci_param,
                capsize=0.15,
                edgecolor="black",
                linewidth=0.8,
                ax=ax,
                alpha=0.7,
                legend=False,
            )

        if cfg.show_points:
            sns.stripplot(
                data=long_df,
                x="Group",
                y="Value",
                order=groups,
                color="black",
                size=cfg.point_size,
                alpha=cfg.point_alpha,
                jitter=True,
                ax=ax,
            )

    def _qpcr_bars(
        self,
        ax: Axes,
        df_wide: pd.DataFrame,
        groups: list[str],
    ) -> None:
        """qPCR bars: geometric mean with asymmetric error bars.

        Statistics run on the log2 fold-change (−ΔΔCt) space, so the
        displayed central tendency and error bars are derived there and
        rotated back to the fold-change axis: bar height = 2^(mean log2FC),
        error-bar endpoints = 2^(mean log2FC ± err) — visually asymmetric
        on the linear fold-change axis, consistent with the tested space.
        """
        cfg = self.config
        x_positions = np.arange(len(groups))
        geo_means: list[float] = []
        lower_errs: list[float] = []
        upper_errs: list[float] = []
        for group in groups:
            geo, lower, upper = self._qpcr_geo_stats(cast(Any, df_wide[group]))
            geo_means.append(geo)
            lower_errs.append(lower)
            upper_errs.append(upper)

        ax.bar(
            x_positions,
            geo_means,
            color=cfg.palette[: len(groups)],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.errorbar(
            x_positions,
            geo_means,
            yerr=[lower_errs, upper_errs],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            capthick=1.0,
            zorder=6,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups)

    def _violin(
        self,
        ax: Axes,
        long_df: pd.DataFrame,
        df_wide: pd.DataFrame,
    ) -> None:
        cfg = self.config
        groups = list(df_wide.columns)

        sns.violinplot(
            data=long_df,
            x="Group",
            y="Value",
            hue="Group",
            order=groups,
            hue_order=groups,
            palette=cfg.palette[: len(groups)],
            inner="quartile",
            linewidth=0.8,
            ax=ax,
            legend=False,
        )

        if cfg.show_points:
            sns.stripplot(
                data=long_df,
                x="Group",
                y="Value",
                order=groups,
                color="black",
                size=cfg.point_size,
                alpha=cfg.point_alpha,
                jitter=True,
                ax=ax,
            )

    def _line(
        self,
        ax: Axes,
        long_df: pd.DataFrame,
        df_wide: pd.DataFrame,
    ) -> None:
        cfg = self.config
        groups = list(df_wide.columns)
        x_positions = np.arange(len(groups))

        means = [df_wide[g].mean() for g in groups]
        errors = [self._error_value(cast(pd.Series, df_wide[g]).dropna()) for g in groups]

        if self._qpcr_log_display():
            geo_means, lower_err, upper_err = [], [], []
            for group in groups:
                geo, lower, upper = self._qpcr_geo_stats(cast(pd.Series, df_wide[group]))
                geo_means.append(geo)
                lower_err.append(lower)
                upper_err.append(upper)
            means, errors = geo_means, [lower_err, upper_err]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            fmt="-o",
            color=cfg.palette[0],
            capsize=4,
            capthick=1.2,
            linewidth=1.5,
            markersize=6,
        )

        if cfg.show_points:
            for i, g in enumerate(groups):
                vals = df_wide[g].dropna().to_numpy()
                jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(vals))
                ax.scatter(
                    np.full_like(vals, i, dtype=float) + jitter,
                    vals,
                    color="black",
                    s=cfg.point_size**2,
                    alpha=cfg.point_alpha,
                    zorder=5,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups)

    # ------------------------------------------------------------------
    # Dose-response (IC50)
    # ------------------------------------------------------------------

    def _dose_response_figure(self, df_wide: pd.DataFrame) -> Figure:
        """Generate a dose-response curve with fit and IC50 markers."""
        cfg = self.config
        fit_info = cast(Any, cfg.ic50_fit_info)
        conc = np.array(fit_info.concentrations, dtype=float)
        dose_cols = fit_info.dose_col_names
        params = fit_info.fit_params
        method = params.get("method", "four_pl")

        with get_prism_context(cfg.journal_preset, cfg.base_theme):
            fw, fh = self._resolve_figsize()
            fig, ax = plt.subplots(figsize=(fw, fh))

            # Scatter individual data points at each concentration
            for i, col in enumerate(dose_cols):
                vals = df_wide[col].dropna().to_numpy()
                x_pts = np.full_like(vals, conc[i])
                ax.scatter(
                    x_pts,
                    vals,
                    color=cfg.palette[0],
                    s=cfg.point_size**2,
                    alpha=cfg.point_alpha,
                    zorder=5,
                )

            # Mean ± error bars
            means = np.array([df_wide[col].dropna().mean() for col in dose_cols])
            errors = np.array(
                [self._error_value(cast(pd.Series, df_wide[col]).dropna()) for col in dose_cols]
            )
            ax.errorbar(
                conc,
                means,
                yerr=errors,
                fmt="o",
                color=cfg.palette[0],
                capsize=4,
                capthick=1.2,
                markersize=6,
                zorder=6,
                label="Mean",
            )

            # Decide x-axis scale: auto-detect or user override
            scale = cfg.preset_dose_axis_scale
            if scale == DoseAxisScale.AUTO:
                use_log = (conc.max() / conc.min()) > 100
            else:
                use_log = scale == DoseAxisScale.LOG

            # Draw fit curve based on method
            if method == "three_pl":
                from .presets.cck8 import _three_param_logistic

                if use_log:
                    x_fit = np.geomspace(conc.min() * 0.5, conc.max() * 2, 200)
                else:
                    margin = (conc.max() - conc.min()) * 0.1
                    x_fit = np.linspace(conc.min() - margin, conc.max() + margin, 200)
                    x_fit = x_fit[x_fit > 0]
                y_fit = _three_param_logistic(
                    x_fit,
                    params["bottom"],
                    params["ic50"],
                    params["hill"],
                )
                ax.plot(
                    x_fit,
                    y_fit,
                    "-",
                    color=cfg.palette[0],
                    linewidth=1.5,
                    label="3PL fit",
                    zorder=4,
                )
            elif method == "log_linear":
                # Connect mean points with lines instead of a smooth curve
                sort_idx = np.argsort(conc)
                ax.plot(
                    conc[sort_idx],
                    means[sort_idx],
                    "-",
                    color=cfg.palette[0],
                    linewidth=1.5,
                    label="Interpolation",
                    zorder=4,
                )
            else:
                # Legacy 4PL
                from .tools.standard_curve import (
                    four_param_logistic as _four_param_logistic,
                )

                if use_log:
                    x_fit = np.geomspace(conc.min() * 0.5, conc.max() * 2, 200)
                else:
                    margin = (conc.max() - conc.min()) * 0.1
                    x_fit = np.linspace(conc.min() - margin, conc.max() + margin, 200)
                    x_fit = x_fit[x_fit > 0]
                y_fit = _four_param_logistic(
                    x_fit,
                    params["bottom"],
                    params["top"],
                    params["ic50"],
                    params["hill"],
                )
                ax.plot(
                    x_fit,
                    y_fit,
                    "-",
                    color=cfg.palette[0],
                    linewidth=1.5,
                    label="4PL fit",
                    zorder=4,
                )

            # Horizontal dashed line at 50%
            ax.axhline(y=50, linestyle="--", color="gray", linewidth=0.8, zorder=3)

            # Vertical dashed line at IC50
            ic50_val = params["ic50"]
            ax.axvline(
                x=ic50_val, linestyle="--", color="gray", linewidth=0.8, zorder=3
            )

            # IC50 text annotation
            ax.annotate(
                f"IC50 = {ic50_val:.4g}",
                xy=(ic50_val, 50),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "gray"},
                zorder=7,
            )

            if use_log:
                ax.set_xscale("log")
            ax.set_xlabel(cfg.x_label or "Concentration")
            ax.set_ylabel(cfg.y_label)
            if cfg.title:
                ax.set_title(cfg.title)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            fit_canvas(fig, ax, cfg)

        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_qpcr(self) -> bool:
        """Whether the current preset is the qPCR ΔΔCt transform."""
        return self.config.experiment_preset is ExperimentPreset.QPCR

    def _qpcr_log_display(self) -> bool:
        """Whether qPCR charts use log-space geometric-mean display.

        Linear space falls back to the standard arithmetic-mean chart paths
        with symmetric error bars; hypothesis tests always run in log2 space
        regardless of this display choice.
        """
        return (
            self._is_qpcr()
            and self.config.preset_qpcr_error_space is QPCRErrorSpace.LOG
        )

    def _qpcr_geo_stats(self, series: pd.Series) -> tuple[float, float, float]:
        """Geometric mean plus asymmetric error-bar offsets, in fold-change units.

        The error magnitude is computed on log2(fold change) (= −ΔΔCt) with
        the configured :class:`ErrorBarType`, then rotated back through the
        exponential so the bar/error display stays consistent with the space
        the statistics were computed on.
        """
        logs = np.log2(series.dropna().to_numpy(dtype=float))
        if logs.size == 0:
            return 0.0, 0.0, 0.0
        mean_log = float(np.mean(logs))
        err_log = self._log_error_value(logs)
        geo_mean = 2.0**mean_log
        lower = geo_mean - float(2.0 ** (mean_log - err_log))
        upper = float(2.0 ** (mean_log + err_log)) - geo_mean
        return geo_mean, lower, upper

    def _log_error_value(self, logs: np.ndarray) -> float:
        """Error-bar magnitude in log2 space for the configured error type."""
        n = logs.size
        if n == 0:
            return 0.0
        if self.config.error_bar == ErrorBarType.SEM:
            return float(np.std(logs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if self.config.error_bar == ErrorBarType.SD:
            return float(np.std(logs, ddof=1)) if n > 1 else 0.0
        # 95% CI
        from scipy import stats as sp_stats

        if n < 3:
            return 0.0
        se = np.std(logs, ddof=1) / np.sqrt(n)
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
        return float(se * t_crit)

    def _resolve_figsize(self) -> tuple[float, float]:
        """Resolve canvas shape, retaining legacy Auto sizing."""
        return resolve_figsize(self.config)

    def _seaborn_ci(self) -> tuple[str, int] | str:
        """Map ErrorBarType to seaborn ``errorbar`` parameter."""
        mapping = {
            ErrorBarType.SEM: ("se", 1),
            ErrorBarType.SD: "sd",
            ErrorBarType.CI95: ("ci", 95),
        }
        return mapping[self.config.error_bar]

    def _error_value(self, series: pd.Series) -> float:
        """Compute single error bar value for line chart."""
        vals = series.to_numpy()
        n = len(vals)
        if n == 0:
            return 0.0
        if self.config.error_bar == ErrorBarType.SEM:
            return float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if self.config.error_bar == ErrorBarType.SD:
            return float(np.std(vals, ddof=1)) if n > 1 else 0.0
        # 95% CI
        from scipy import stats as sp_stats

        if n < 3:
            return 0.0
        se = np.std(vals, ddof=1) / np.sqrt(n)
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
        return float(se * t_crit)
