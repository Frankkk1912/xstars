"""Shared test fixtures for XSTARS."""

import numpy as np
import pandas as pd
import pytest

from xstars.config import PrismConfig


@pytest.fixture
def two_group_normal() -> pd.DataFrame:
    """Two groups drawn from normal distributions with different means."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Control": rng.normal(10, 2, size=15),
        "Treatment": rng.normal(14, 2, size=15),
    })


@pytest.fixture
def two_group_nonnormal() -> pd.DataFrame:
    """Two groups with skewed (exponential) distributions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Control": rng.exponential(2, size=20),
        "Treatment": rng.exponential(5, size=20),
    })


@pytest.fixture
def three_group_normal() -> pd.DataFrame:
    """Three groups from normal distributions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "A": rng.normal(10, 2, size=12),
        "B": rng.normal(14, 2, size=12),
        "C": rng.normal(18, 2, size=12),
    })


@pytest.fixture
def three_group_nonnormal() -> pd.DataFrame:
    """Three groups with non-normal data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Low": rng.exponential(1, size=15),
        "Mid": rng.exponential(3, size=15),
        "High": rng.exponential(6, size=15),
    })


@pytest.fixture
def wide_with_nans() -> pd.DataFrame:
    """Wide-format data with NaN values and an all-NaN row."""
    return pd.DataFrame({
        "Control": [1.0, 2.0, np.nan, 4.0, np.nan],
        "Treatment": [5.0, 6.0, 7.0, np.nan, np.nan],
    })


@pytest.fixture
def paired_data() -> pd.DataFrame:
    """Paired measurements (before/after) on the same subjects."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(50, 5, size=10)
    return pd.DataFrame({
        "Before": baseline,
        "After": baseline + rng.normal(5, 2, size=10),
    })


@pytest.fixture
def default_config() -> PrismConfig:
    return PrismConfig()
