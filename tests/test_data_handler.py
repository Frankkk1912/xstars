"""Tests for data_handler module."""

import numpy as np
import pandas as pd
import pytest

from xstars.data_handler import DataHandler


class TestClean:
    def test_coerces_non_numeric(self):
        df = pd.DataFrame({"A": [1, "two", 3], "B": [4, 5, "six"]})
        result = DataHandler.clean(df)
        assert result["A"].isna().sum() == 1
        assert result["B"].isna().sum() == 1

    def test_drops_all_nan_rows(self):
        df = pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, np.nan, 6.0],
        })
        result = DataHandler.clean(df)
        assert len(result) == 2

    def test_strips_column_names(self):
        df = pd.DataFrame({" A ": [1], " B": [2]})
        result = DataHandler.clean(df)
        assert list(result.columns) == ["A", "B"]


class TestWideToLong:
    def test_basic_conversion(self, two_group_normal):
        long = DataHandler.wide_to_long(two_group_normal)
        assert set(long.columns) == {"Group", "Value"}
        assert set(long["Group"].unique()) == {"Control", "Treatment"}
        assert len(long) == 30  # 15 + 15

    def test_drops_nans(self, wide_with_nans):
        long = DataHandler.wide_to_long(wide_with_nans)
        assert long["Value"].isna().sum() == 0
        # Control has 3 valid, Treatment has 3 valid
        assert len(long) == 6


class TestGroupInfo:
    def test_group_names(self, three_group_normal):
        assert DataHandler.group_names(three_group_normal) == ["A", "B", "C"]

    def test_group_sizes(self, wide_with_nans):
        sizes = DataHandler.group_sizes(wide_with_nans)
        assert sizes["Control"] == 3
        assert sizes["Treatment"] == 3


class TestValidation:
    def test_too_few_groups(self):
        df = pd.DataFrame({"Only": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 groups"):
            DataHandler.validate(df)

    def test_too_few_values(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only 1 valid values"):
            DataHandler.validate(df)

    def test_valid_data_passes(self, two_group_normal):
        DataHandler.validate(two_group_normal)  # should not raise
