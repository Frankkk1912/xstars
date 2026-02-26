"""ELISA preset — standard curve back-calculation to concentration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..config import ExperimentPreset
from ..tools.standard_curve import CurveFitResult, back_calculate
from . import BasePreset, PresetOptions, register_preset


@dataclass
class ELISAOptions(PresetOptions):
    """Options specific to ELISA preset."""
    fit_result: CurveFitResult | None = None


@dataclass
class ELISAResult:
    """Stores ELISA processing results."""
    concentration_df: pd.DataFrame
    fit_result: CurveFitResult | None = None


@register_preset(ExperimentPreset.ELISA)
class ELISAPreset(BasePreset):

    def __init__(self):
        self.last_result: ELISAResult | None = None

    @property
    def description(self) -> str:
        return "ELISA — standard curve back-calculation to concentration"

    @property
    def default_y_label(self) -> str:
        return "Concentration"

    def validate_input(self, df: pd.DataFrame, options: PresetOptions) -> None:
        if df.empty:
            raise ValueError("Sample data is empty.")
        # All values should be numeric
        non_numeric = df.apply(lambda s: pd.to_numeric(s, errors="coerce").isna().all())
        bad_cols = [c for c, v in non_numeric.items() if v]
        if bad_cols:
            raise ValueError(
                f"Columns contain no numeric values: {bad_cols}"
            )

    def transform(self, df: pd.DataFrame, options: PresetOptions) -> pd.DataFrame:
        """Back-calculate sample OD values to concentrations using fit_result.

        The fit_result must be provided in options (pre-fitted by main.py).
        """
        opts = options if isinstance(options, ELISAOptions) else ELISAOptions(**vars(options))
        self.validate_input(df, opts)

        if opts.fit_result is None:
            raise ValueError("No standard curve fit result provided for ELISA back-calculation.")

        fit = opts.fit_result
        result_df = df.copy()
        for col in result_df.columns:
            od_vals = pd.to_numeric(result_df[col], errors="coerce").to_numpy()
            result_df[col] = back_calculate(fit, od_vals)

        self.last_result = ELISAResult(
            concentration_df=result_df,
            fit_result=fit,
        )
        return result_df
