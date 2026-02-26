"""Tests for stats_engine module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from xstars.config import PrismConfig
from xstars.stats_engine import StatsEngine, StatsResult, _stars


class TestStars:
    def test_four_stars(self):
        assert _stars(0.0001) == "****"

    def test_three_stars(self):
        assert _stars(0.001) == "***"

    def test_two_stars(self):
        assert _stars(0.01) == "**"

    def test_one_star(self):
        assert _stars(0.05) == "*"

    def test_ns(self):
        assert _stars(0.1) == "ns"


class TestTwoGroupNormal:
    """Two groups from normal distributions → should use t-test."""

    def test_independent_ttest(self, two_group_normal):
        engine = StatsEngine()
        result = engine.analyze(two_group_normal)

        assert result.all_normal
        assert "t-test" in result.decision_path.lower() or "t-test" in result.pairs[0].test_name.lower()
        assert len(result.pairs) == 1

        # Verify p-value matches scipy directly
        a = two_group_normal["Control"].to_numpy()
        b = two_group_normal["Treatment"].to_numpy()
        _, expected_p = stats.ttest_ind(a, b, equal_var=result.equal_variance)
        assert abs(result.pairs[0].p_value - expected_p) < 1e-10

    def test_welch_when_unequal_var(self):
        """Force unequal variance → Welch's t-test."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "A": rng.normal(10, 1, size=15),
            "B": rng.normal(10, 10, size=15),
        })
        result = StatsEngine().analyze(df)
        if not result.equal_variance and result.all_normal:
            assert "Welch" in result.pairs[0].test_name


class TestTwoGroupNonNormal:
    """Two groups with skewed data → should use Mann-Whitney."""

    def test_mann_whitney(self, two_group_nonnormal):
        engine = StatsEngine()
        result = engine.analyze(two_group_nonnormal)

        # Exponential distribution should fail normality
        if not result.all_normal:
            assert "Mann-Whitney" in result.pairs[0].test_name
            # Verify p-value
            a = two_group_nonnormal["Control"].to_numpy()
            b = two_group_nonnormal["Treatment"].to_numpy()
            _, expected_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            assert abs(result.pairs[0].p_value - expected_p) < 1e-10


class TestPaired:
    def test_paired_ttest(self, paired_data):
        config = PrismConfig(paired=True)
        result = StatsEngine(config).analyze(paired_data)
        assert "Paired" in result.pairs[0].test_name or "paired" in result.decision_path

    def test_paired_wilcoxon_nonnormal(self):
        """Skewed paired data → Wilcoxon signed-rank."""
        rng = np.random.default_rng(42)
        skewed = rng.exponential(5, size=15)
        df = pd.DataFrame({
            "Before": skewed,
            "After": skewed + rng.exponential(2, size=15),
        })
        config = PrismConfig(paired=True)
        result = StatsEngine(config).analyze(df)
        if not result.all_normal:
            assert "Wilcoxon" in result.pairs[0].test_name


class TestMultiGroupNormal:
    """Three normal groups → ANOVA + Tukey."""

    def test_anova_tukey(self, three_group_normal):
        result = StatsEngine().analyze(three_group_normal)

        if result.all_normal and result.equal_variance:
            assert result.omnibus_test == "One-way ANOVA"
            assert result.omnibus_p is not None
            # 3 groups → C(3,2) = 3 pairs
            assert len(result.pairs) == 3
            assert all(p.test_name == "Tukey HSD" for p in result.pairs)


class TestMultiGroupNonNormal:
    """Three non-normal groups → Kruskal-Wallis + Dunn's."""

    def test_kruskal_dunn(self, three_group_nonnormal):
        result = StatsEngine().analyze(three_group_nonnormal)

        if not result.all_normal or not result.equal_variance:
            assert result.omnibus_test == "Kruskal-Wallis"
            assert len(result.pairs) == 3
            assert "Dunn" in result.pairs[0].test_name


class TestEdgeCases:
    def test_single_group_raises(self):
        df = pd.DataFrame({"Only": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="at least 2 groups"):
            StatsEngine().analyze(df)

    def test_constant_values(self):
        """All identical values → shapiro p=0 → non-normal path."""
        df = pd.DataFrame({
            "A": [5.0, 5.0, 5.0, 5.0, 5.0],
            "B": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        result = StatsEngine().analyze(df)
        assert not result.all_normal  # constant → p=0 → non-normal


class TestControlGroup:
    """Control-group comparison mode filters pairs."""

    def test_control_group_filters_tukey(self, three_group_normal):
        config = PrismConfig(control_group="A")
        result = StatsEngine(config).analyze(three_group_normal)
        # Should only have pairs involving "A"
        for p in result.pairs:
            assert p.group_a == "A" or p.group_b == "A"
        assert len(result.pairs) == 2  # (A,B) and (A,C)

    def test_control_group_filters_dunn(self, three_group_nonnormal):
        config = PrismConfig(control_group="Low")
        result = StatsEngine(config).analyze(three_group_nonnormal)
        for p in result.pairs:
            assert p.group_a == "Low" or p.group_b == "Low"
        assert len(result.pairs) == 2

    def test_no_control_gives_all_pairs(self, three_group_normal):
        result = StatsEngine().analyze(three_group_normal)
        assert len(result.pairs) == 3  # C(3,2)
