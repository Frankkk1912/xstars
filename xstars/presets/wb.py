"""Western Blot preset — normalizes band intensities to fold change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ExperimentPreset
from . import BasePreset, PresetOptions, register_preset


@dataclass
class WBOptions(PresetOptions):
    """Options specific to Western Blot analysis."""
    has_reference: bool = False
    reference_protein: str = ""  # for labeled mode: which protein is the reference


@register_preset(ExperimentPreset.WB)
class WBPreset(BasePreset):

    @property
    def description(self) -> str:
        return "Western Blot fold-change normalization"

    @property
    def default_y_label(self) -> str:
        return "Fold Change"

    def validate_input(self, df: pd.DataFrame, options: PresetOptions) -> None:
        opts = options if isinstance(options, WBOptions) else WBOptions(**vars(options))
        if opts.control_group and opts.control_group not in df.columns:
            raise ValueError(
                f"Control group '{opts.control_group}' not found in data columns: "
                f"{list(df.columns)}"
            )
        if opts.has_reference:
            for col in df.columns:
                n_valid = df[col].dropna().shape[0]
                if n_valid % 2 != 0:
                    raise ValueError(
                        f"Reference mode requires even number of valid values "
                        f"per group (top half=target, bottom half=reference), "
                        f"but '{col}' has {n_valid}."
                    )

    def transform(self, df: pd.DataFrame, options: PresetOptions) -> pd.DataFrame:
        opts = options if isinstance(options, WBOptions) else WBOptions(**vars(options))
        self.validate_input(df, opts)

        result = df.copy()

        # Step 1: If reference protein present, compute target/reference ratio
        # Layout: top half rows = target protein, bottom half = reference (e.g. GAPDH)
        if opts.has_reference:
            new_data: dict[str, list[float]] = {}
            for col in result.columns:
                vals = result[col].dropna().to_numpy()
                half = len(vals) // 2
                targets = vals[:half]
                refs = vals[half:]
                ratios = targets / refs
                new_data[col] = list(ratios)
            max_len = max(len(v) for v in new_data.values())
            for col in new_data:
                while len(new_data[col]) < max_len:
                    new_data[col].append(np.nan)
            result = pd.DataFrame(new_data)

        # Step 2: Normalize to control group mean → fold change
        control = opts.control_group or result.columns[0]
        control_mean = result[control].dropna().mean()
        if control_mean == 0:
            raise ValueError("Control group mean is zero; cannot normalize.")
        result = result / control_mean

        return result

    def transform_labeled(
        self,
        labels: pd.Series,
        df_numeric: pd.DataFrame,
        options: PresetOptions,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Transform labeled WB data (label column + numeric columns) into
        per-target fold-change DataFrames.

        Parameters
        ----------
        labels : pd.Series
            Protein name for each row (e.g. "Target-A", "GAPDH").
        df_numeric : pd.DataFrame
            Wide-format numeric data (columns = treatment groups).
        options : PresetOptions
            Must include ``control_group`` and optionally ``reference_protein``.

        Returns
        -------
        list of (protein_name, fold_change_df) tuples, one per target protein.
        """
        opts = options if isinstance(options, WBOptions) else WBOptions(**vars(options))

        proteins = list(dict.fromkeys(labels))  # unique, preserving order
        ref_name = opts.reference_protein or proteins[-1]

        if ref_name not in proteins:
            raise ValueError(
                f"Reference protein '{ref_name}' not found in label column. "
                f"Available proteins: {proteins}"
            )

        targets = [p for p in proteins if p != ref_name]
        if not targets:
            raise ValueError("No target proteins found (all rows are reference).")

        ref_mask = labels.values == ref_name
        ref_rows = df_numeric.loc[ref_mask].reset_index(drop=True)

        control = opts.control_group or df_numeric.columns[0]
        if control not in df_numeric.columns:
            raise ValueError(
                f"Control group '{control}' not found in data columns: "
                f"{list(df_numeric.columns)}"
            )

        results: list[tuple[str, pd.DataFrame]] = []
        for target in targets:
            t_mask = labels.values == target
            t_rows = df_numeric.loc[t_mask].reset_index(drop=True)

            n_target = len(t_rows)
            n_ref = len(ref_rows)
            if n_target != n_ref:
                raise ValueError(
                    f"Target '{target}' has {n_target} replicates but reference "
                    f"'{ref_name}' has {n_ref}. They must match."
                )

            # Row-wise ratio: target / reference
            ratio_df = t_rows / ref_rows.values

            # Normalize to control mean → fold change
            ctrl_mean = ratio_df[control].dropna().mean()
            if ctrl_mean == 0:
                raise ValueError(
                    f"Control group mean is zero for target '{target}'; "
                    f"cannot normalize."
                )
            fold_df = ratio_df / ctrl_mean
            results.append((target, fold_df))

        return results
