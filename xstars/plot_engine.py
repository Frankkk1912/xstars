"""Generate publication-quality figures (bar+scatter, violin, line)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from . import annotations as ann
from .config import ChartType, DoseAxisScale, ErrorBarType, PrismConfig
from .stats_engine import StatsResult
from .styles import get_journal_figsize, get_prism_context


def export_figure(fig: "Figure", path: str, dpi: int = 300) -> None:
    """Save figure to file. Format is inferred from extension by matplotlib."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


class PlotEngine:
    """Create a figure from wide-format data + stats results."""

    def __init__(self, config: PrismConfig | None = None) -> None:
        self.config = config or PrismConfig()

    def plot(
        self,
        df_wide: pd.DataFrame,
        stats_result: StatsResult | None = None,
    ) -> "Figure":
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

            # Significance annotations
            if stats_result is not None:
                ann.annotate(ax, long_df, stats_result, cfg)

            # Labels
            ax.set_xlabel(cfg.x_label)
            ax.set_ylabel(cfg.y_label)
            if cfg.title:
                ax.set_title(cfg.title)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Chart types
    # ------------------------------------------------------------------

    def _bar_scatter(
        self,
        ax: plt.Axes,
        long_df: pd.DataFrame,
        df_wide: pd.DataFrame,
    ) -> None:
        cfg = self.config
        groups = list(df_wide.columns)
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

    def _violin(
        self,
        ax: plt.Axes,
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
        ax: plt.Axes,
        long_df: pd.DataFrame,
        df_wide: pd.DataFrame,
    ) -> None:
        cfg = self.config
        groups = list(df_wide.columns)
        x_positions = np.arange(len(groups))

        means = [df_wide[g].mean() for g in groups]
        errors = [self._error_value(df_wide[g].dropna()) for g in groups]

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
                jitter = np.random.default_rng(42).uniform(
                    -0.05, 0.05, size=len(vals)
                )
                ax.scatter(
                    np.full_like(vals, i, dtype=float) + jitter,
                    vals,
                    color="black",
                    s=cfg.point_size ** 2,
                    alpha=cfg.point_alpha,
                    zorder=5,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups)

    # ------------------------------------------------------------------
    # Dose-response (IC50)
    # ------------------------------------------------------------------

    def _dose_response_figure(self, df_wide: pd.DataFrame) -> "Figure":
        """Generate a dose-response curve with fit and IC50 markers."""
        cfg = self.config
        fit_info = cfg.ic50_fit_info
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
                    x_pts, vals,
                    color=cfg.palette[0],
                    s=cfg.point_size ** 2,
                    alpha=cfg.point_alpha,
                    zorder=5,
                )

            # Mean ± error bars
            means = np.array([df_wide[col].dropna().mean() for col in dose_cols])
            errors = np.array([
                self._error_value(df_wide[col].dropna()) for col in dose_cols
            ])
            ax.errorbar(
                conc, means, yerr=errors,
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
                use_log = (scale == DoseAxisScale.LOG)

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
                    x_fit, params["bottom"], params["ic50"], params["hill"],
                )
                ax.plot(x_fit, y_fit, "-", color=cfg.palette[0], linewidth=1.5,
                        label="3PL fit", zorder=4)
            elif method == "log_linear":
                # Connect mean points with lines instead of a smooth curve
                sort_idx = np.argsort(conc)
                ax.plot(conc[sort_idx], means[sort_idx], "-", color=cfg.palette[0],
                        linewidth=1.5, label="Interpolation", zorder=4)
            else:
                # Legacy 4PL
                from .tools.standard_curve import four_param_logistic as _four_param_logistic
                if use_log:
                    x_fit = np.geomspace(conc.min() * 0.5, conc.max() * 2, 200)
                else:
                    margin = (conc.max() - conc.min()) * 0.1
                    x_fit = np.linspace(conc.min() - margin, conc.max() + margin, 200)
                    x_fit = x_fit[x_fit > 0]
                y_fit = _four_param_logistic(
                    x_fit, params["bottom"], params["top"],
                    params["ic50"], params["hill"],
                )
                ax.plot(x_fit, y_fit, "-", color=cfg.palette[0], linewidth=1.5,
                        label="4PL fit", zorder=4)

            # Horizontal dashed line at 50%
            ax.axhline(y=50, linestyle="--", color="gray", linewidth=0.8, zorder=3)

            # Vertical dashed line at IC50
            ic50_val = params["ic50"]
            ax.axvline(x=ic50_val, linestyle="--", color="gray", linewidth=0.8, zorder=3)

            # IC50 text annotation
            ax.annotate(
                f"IC50 = {ic50_val:.4g}",
                xy=(ic50_val, 50),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"),
                zorder=7,
            )

            if use_log:
                ax.set_xscale("log")
            ax.set_xlabel(cfg.x_label or "Concentration")
            ax.set_ylabel(cfg.y_label)
            if cfg.title:
                ax.set_title(cfg.title)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_figsize(self) -> tuple[float, float]:
        """Use journal-recommended size if available and user hasn't customized."""
        cfg = self.config
        journal_size = get_journal_figsize(cfg.journal_preset)
        defaults = PrismConfig()
        if journal_size and cfg.fig_width == defaults.fig_width and cfg.fig_height == defaults.fig_height:
            return journal_size
        return (cfg.fig_width, cfg.fig_height)

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
