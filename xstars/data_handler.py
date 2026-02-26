"""Read Excel selection and convert between wide / long table formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xlwings as xw

from .config import PrismConfig


class DataHandler:
    """Read the active Excel selection and prepare data for analysis."""

    def __init__(self, config: PrismConfig | None = None) -> None:
        self.config = config or PrismConfig()

    # ------------------------------------------------------------------
    # Excel integration
    # ------------------------------------------------------------------

    def read_selection(self) -> pd.DataFrame:
        """Read the current Excel selection as a wide-format DataFrame.

        Columns = group names (from the first row of the selection).
        Rows = replicate values.
        """
        import xlwings as xw

        book = xw.Book.caller()
        sel: xw.Range = book.selection
        self._selection_ref = sel  # cache for insertion later

        raw = sel.options(pd.DataFrame, header=1, index=False).value
        return self.clean(raw)

    def read_selection_with_labels(self) -> tuple[pd.Series, pd.DataFrame]:
        """Read the current Excel selection where the first column contains
        string labels (e.g. protein names) and remaining columns are numeric.

        Returns
        -------
        (labels, numeric_df) where labels is a Series of strings and
        numeric_df is a wide-format DataFrame with group columns.
        """
        import xlwings as xw

        book = xw.Book.caller()
        sel: xw.Range = book.selection
        self._selection_ref = sel

        raw = sel.options(pd.DataFrame, header=1, index=False).value
        raw.columns = [str(c).strip() for c in raw.columns]

        # First column is the label column
        label_col = raw.columns[0]
        labels = raw[label_col].astype(str).str.strip()
        numeric_df = raw.drop(columns=[label_col])

        # Coerce numeric columns
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

        # Drop all-NaN rows
        valid = ~numeric_df.isna().all(axis=1)
        labels = labels.loc[valid].reset_index(drop=True)
        numeric_df = numeric_df.loc[valid].reset_index(drop=True)

        return labels, numeric_df

    def read_from_range(self, sheet: "xw.Sheet", address: str) -> pd.DataFrame:
        """Read a specific range (e.g. ``"A1:D10"``) as wide DataFrame."""
        rng = sheet.range(address)
        raw = rng.options(pd.DataFrame, header=1, index=False).value
        self._selection_ref = rng
        return self.clean(raw)

    def get_insertion_cell(self, sheet: "xw.Sheet") -> str:
        """Return the cell address where the chart should be placed.

        Default: *insert_offset_cols* columns to the right of the selection.
        """
        sel = self._selection_ref
        top_row = sel.row
        right_col = sel.column + sel.columns.count + self.config.insert_offset_cols
        return sheet.range(top_row, right_col).address

    # ------------------------------------------------------------------
    # Data cleaning & transformation
    # ------------------------------------------------------------------

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Drop all-NaN rows and coerce non-numeric values to NaN."""
        # Ensure column names are strings
        df.columns = [str(c).strip() for c in df.columns]
        # Coerce each column to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Drop rows where every value is NaN
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    @staticmethod
    def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide DataFrame (cols=groups) to long format with
        columns ``['Group', 'Value']``, dropping NaN values.
        """
        long = df.melt(var_name="Group", value_name="Value")
        long = long.dropna(subset=["Value"]).reset_index(drop=True)
        return long

    @staticmethod
    def group_names(df: pd.DataFrame) -> list[str]:
        """Return the list of group (column) names from a wide DataFrame."""
        return list(df.columns)

    @staticmethod
    def group_sizes(df: pd.DataFrame) -> dict[str, int]:
        """Return {group_name: n_valid_values} for a wide DataFrame."""
        return {col: int(df[col].dropna().shape[0]) for col in df.columns}

    @staticmethod
    def validate(df: pd.DataFrame, min_groups: int = 2, min_n: int = 3) -> None:
        """Raise ``ValueError`` if data does not meet minimum requirements."""
        n_groups = len(df.columns)
        if n_groups < min_groups:
            raise ValueError(
                f"Need at least {min_groups} groups, got {n_groups}."
            )
        for col in df.columns:
            n = df[col].dropna().shape[0]
            if n < min_n:
                raise ValueError(
                    f"Group '{col}' has only {n} valid values (minimum {min_n})."
                )
