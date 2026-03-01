"""Statistical decision tree — automatically choose and run the right test."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .config import PrismConfig


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class PairResult:
    """Result for one pairwise comparison."""
    group_a: str
    group_b: str
    test_name: str
    statistic: float
    p_value: float
    stars: str


@dataclass
class StatsResult:
    """Container for all statistical results of one analysis."""

    # Decision path description (human-readable)
    decision_path: str

    # Normality
    normality_test: str  # "Shapiro-Wilk"
    normality_pvalues: dict[str, float] = field(default_factory=dict)
    all_normal: bool = False

    # Variance homogeneity
    variance_test: str | None = None
    variance_p: float | None = None
    equal_variance: bool | None = None

    # Omnibus test (ANOVA / Kruskal-Wallis) — None for 2-group
    omnibus_test: str | None = None
    omnibus_statistic: float | None = None
    omnibus_p: float | None = None

    # Pairwise results
    pairs: list[PairResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame of pairwise results."""
        rows = []
        for p in self.pairs:
            rows.append({
                "Group A": p.group_a,
                "Group B": p.group_b,
                "Test": p.test_name,
                "Statistic": p.statistic,
                "p-value": p.p_value,
                "Significance": p.stars,
            })
        columns = ["Group A", "Group B", "Test", "Statistic", "p-value", "Significance"]
        return pd.DataFrame(rows, columns=columns)

    @property
    def pair_tuples(self) -> list[tuple[str, str]]:
        """Return list of (group_a, group_b) for annotation."""
        return [(p.group_a, p.group_b) for p in self.pairs]

    @property
    def p_values(self) -> list[float]:
        return [p.p_value for p in self.pairs]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _stars(p: float, alpha: float = 0.05) -> str:
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= alpha:
        return "*"
    return "ns"


# Minimum n per group to trust a normality test result.
# Below this threshold Shapiro-Wilk is unreliable on tiny samples
# (spurious rejections), so we assume normality and prefer parametric
# tests — matching Prism's "N too small" behaviour for D'Agostino,
# Anderson-Darling, and KS tests.
_MIN_N_FOR_NORMALITY_TEST = 5


def _shapiro(values: np.ndarray) -> float:
    """Return Shapiro-Wilk p-value, or 1.0 when n is too small to trust.

    When n < _MIN_N_FOR_NORMALITY_TEST the test has negligible power and
    commonly produces false rejections, so we return 1.0 (assume normal)
    rather than forcing a non-parametric path on tiny samples.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 3 or np.ptp(clean) == 0:
        return 0.0  # degenerate / constant → treat as non-normal
    if len(clean) < _MIN_N_FOR_NORMALITY_TEST:
        return 1.0  # N too small — assume normal
    _, p = stats.shapiro(clean)
    return float(p)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class StatsEngine:
    """Run the statistical decision tree on wide-format data."""

    def __init__(self, config: PrismConfig | None = None) -> None:
        self.config = config or PrismConfig()

    def analyze(self, df_wide: pd.DataFrame) -> StatsResult:
        """Main entry point.  *df_wide* has columns = groups, rows = replicates."""
        groups = list(df_wide.columns)
        n_groups = len(groups)
        alpha = self.config.alpha

        if n_groups < 2:
            raise ValueError("Need at least 2 groups for comparison.")

        # --- Normality check per group ---
        norm_ps: dict[str, float] = {}
        n_too_small: list[str] = []
        for g in groups:
            vals = df_wide[g].dropna().to_numpy()
            if len(vals) < _MIN_N_FOR_NORMALITY_TEST:
                n_too_small.append(g)
            norm_ps[g] = _shapiro(vals)

        all_normal = all(p >= alpha for p in norm_ps.values())
        normality_test_label = (
            f"Shapiro-Wilk (N too small, assumed normal: {', '.join(n_too_small)})"
            if n_too_small
            else "Shapiro-Wilk"
        )

        # --- Variance homogeneity (Levene) ---
        group_arrays = [df_wide[g].dropna().to_numpy() for g in groups]
        lev_stat, lev_p = stats.levene(*group_arrays)
        equal_var = lev_p >= alpha

        if n_groups == 2:
            return self._two_groups(
                df_wide, groups, norm_ps, all_normal, equal_var, lev_p, alpha,
                normality_test_label,
            )
        else:
            return self._multi_groups(
                df_wide, groups, norm_ps, all_normal, equal_var, lev_p, alpha,
                normality_test_label,
            )

    # ------------------------------------------------------------------
    # Two-group comparison
    # ------------------------------------------------------------------

    def _two_groups(
        self,
        df_wide: pd.DataFrame,
        groups: list[str],
        norm_ps: dict[str, float],
        all_normal: bool,
        equal_var: bool,
        lev_p: float,
        alpha: float,
        normality_test_label: str = "Shapiro-Wilk",
    ) -> StatsResult:
        a_vals = df_wide[groups[0]].dropna().to_numpy()
        b_vals = df_wide[groups[1]].dropna().to_numpy()
        paired = self.config.paired

        if paired:
            # Paired tests require equal n
            n = min(len(a_vals), len(b_vals))
            a_vals, b_vals = a_vals[:n], b_vals[:n]

        if paired:
            if all_normal:
                stat, p = stats.ttest_rel(a_vals, b_vals)
                test_name = "Paired t-test"
                path = "2 groups → paired → normal → Paired t-test"
            else:
                stat, p = stats.wilcoxon(a_vals, b_vals)
                test_name = "Wilcoxon signed-rank"
                path = "2 groups → paired → non-normal → Wilcoxon signed-rank"
        elif all_normal:
            if equal_var:
                stat, p = stats.ttest_ind(a_vals, b_vals, equal_var=True)
                test_name = "Independent t-test"
                path = "2 groups → normal → equal variance → Independent t-test"
            else:
                stat, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
                test_name = "Welch's t-test"
                path = "2 groups → normal → unequal variance → Welch's t-test"
        else:
            stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            test_name = "Mann-Whitney U"
            path = "2 groups → non-normal → Mann-Whitney U"

        pair = PairResult(
            group_a=groups[0],
            group_b=groups[1],
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p),
            stars=_stars(float(p), alpha),
        )

        return StatsResult(
            decision_path=path,
            normality_test=normality_test_label,
            normality_pvalues=norm_ps,
            all_normal=all_normal,
            variance_test="Levene",
            variance_p=float(lev_p),
            equal_variance=equal_var,
            omnibus_test=None,
            omnibus_statistic=None,
            omnibus_p=None,
            pairs=[pair],
        )

    # ------------------------------------------------------------------
    # Multi-group comparison
    # ------------------------------------------------------------------

    def _multi_groups(
        self,
        df_wide: pd.DataFrame,
        groups: list[str],
        norm_ps: dict[str, float],
        all_normal: bool,
        equal_var: bool,
        lev_p: float,
        alpha: float,
        normality_test_label: str = "Shapiro-Wilk",
    ) -> StatsResult:
        group_arrays = [df_wide[g].dropna().to_numpy() for g in groups]
        long = df_wide.melt(var_name="Group", value_name="Value").dropna(
            subset=["Value"]
        )

        if all_normal and equal_var:
            # One-way ANOVA + Tukey HSD
            f_stat, f_p = stats.f_oneway(*group_arrays)
            omnibus_name = "One-way ANOVA"
            path = "≥3 groups → normal + equal var → ANOVA → Tukey HSD"

            # Tukey HSD via scipy
            tukey = stats.tukey_hsd(*group_arrays)
            pairs: list[PairResult] = []
            for i, j in combinations(range(len(groups)), 2):
                p_val = float(tukey.pvalue[i][j])
                pairs.append(PairResult(
                    group_a=groups[i],
                    group_b=groups[j],
                    test_name="Tukey HSD",
                    statistic=float(tukey.statistic[i][j]),
                    p_value=p_val,
                    stars=_stars(p_val, alpha),
                ))
        else:
            # Kruskal-Wallis + Dunn's test
            import scikit_posthocs as sp

            h_stat, h_p = stats.kruskal(*group_arrays)
            omnibus_name = "Kruskal-Wallis"
            f_stat, f_p = h_stat, h_p
            path = "≥3 groups → non-normal or unequal var → Kruskal-Wallis → Dunn's test"

            # Dunn's test with Bonferroni correction
            dunn_matrix = sp.posthoc_dunn(
                long, val_col="Value", group_col="Group", p_adjust="bonferroni"
            )
            pairs = []
            for gi, gj in combinations(groups, 2):
                p_val = float(dunn_matrix.loc[gi, gj])
                pairs.append(PairResult(
                    group_a=gi,
                    group_b=gj,
                    test_name="Dunn's test (Bonferroni)",
                    statistic=np.nan,
                    p_value=p_val,
                    stars=_stars(p_val, alpha),
                ))

        # Filter to control-group pairs only if requested
        control = self.config.control_group
        if control is not None:
            pairs = [
                p for p in pairs
                if p.group_a == control or p.group_b == control
            ]

        return StatsResult(
            decision_path=path,
            normality_test=normality_test_label,
            normality_pvalues=norm_ps,
            all_normal=all_normal,
            variance_test="Levene",
            variance_p=float(lev_p),
            equal_variance=equal_var,
            omnibus_test=omnibus_name,
            omnibus_statistic=float(f_stat),
            omnibus_p=float(f_p),
            pairs=pairs,
        )
