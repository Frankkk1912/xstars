"""qPCR ΔΔCt preset — converts Ct values to relative fold change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ExperimentPreset
from . import BasePreset, PresetOptions, register_preset


@dataclass
class QPCROptions(PresetOptions):
    """Options specific to qPCR analysis."""
    input_format: str = "delta_ct"  # "delta_ct" or "raw_ct"
    reference_gene: str = ""  # for labeled mode: which gene is the reference


@register_preset(ExperimentPreset.QPCR)
class QPCRPreset(BasePreset):

    @property
    def description(self) -> str:
        return "qPCR ΔΔCt relative quantification"

    @property
    def default_y_label(self) -> str:
        return "Relative Expression (2^-ΔΔCt)"

    def validate_input(self, df: pd.DataFrame, options: PresetOptions) -> None:
        opts = options if isinstance(options, QPCROptions) else QPCROptions(**vars(options))
        if opts.control_group and opts.control_group not in df.columns:
            raise ValueError(
                f"Control group '{opts.control_group}' not found in data columns: "
                f"{list(df.columns)}"
            )
        if opts.input_format == "raw_ct":
            for col in df.columns:
                n_valid = df[col].dropna().shape[0]
                if n_valid % 2 != 0:
                    raise ValueError(
                        f"Raw Ct mode requires even number of valid values per group "
                        f"(top half=target, bottom half=reference), "
                        f"but '{col}' has {n_valid}."
                    )
        elif opts.input_format != "delta_ct":
            raise ValueError(
                f"Unknown input_format '{opts.input_format}'. "
                f"Use 'delta_ct' or 'raw_ct'."
            )

    def transform(self, df: pd.DataFrame, options: PresetOptions) -> pd.DataFrame:
        opts = options if isinstance(options, QPCROptions) else QPCROptions(**vars(options))
        self.validate_input(df, opts)

        # Step 1: Compute ΔCt values
        if opts.input_format == "raw_ct":
            delta_ct_data: dict[str, list[float]] = {}
            for col in df.columns:
                vals = df[col].dropna().to_numpy()
                half = len(vals) // 2
                target_ct = vals[:half]
                ref_ct = vals[half:]
                delta_ct_data[col] = list(target_ct - ref_ct)
            max_len = max(len(v) for v in delta_ct_data.values())
            for col in delta_ct_data:
                while len(delta_ct_data[col]) < max_len:
                    delta_ct_data[col].append(np.nan)
            delta_ct_df = pd.DataFrame(delta_ct_data)
        else:
            # Input is already ΔCt
            delta_ct_df = df.copy()

        # Step 2: ΔΔCt = ΔCt - mean(ΔCt_control)
        control = opts.control_group or delta_ct_df.columns[0]
        control_mean = delta_ct_df[control].dropna().mean()
        delta_delta_ct = delta_ct_df - control_mean

        # Step 3: Fold change = 2^(-ΔΔCt)
        fold_change = np.power(2.0, -delta_delta_ct)

        return fold_change

    def transform_labeled(
        self,
        labels: pd.Series,
        df_numeric: pd.DataFrame,
        options: PresetOptions,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Transform labeled qPCR data (multiple target genes + one reference gene)
        into per-gene fold-change DataFrames via the ΔΔCt method.

        Parameters
        ----------
        labels : pd.Series
            Gene name for each row (e.g. "Gene-A", "Gene-B", "GAPDH").
        df_numeric : pd.DataFrame
            Wide-format Ct values (columns = treatment groups).
        options : PresetOptions
            Must include ``control_group`` and optionally ``reference_gene``.

        Returns
        -------
        list of (gene_name, fold_change_df) tuples, one per target gene.
        """
        opts = options if isinstance(options, QPCROptions) else QPCROptions(**vars(options))

        genes = list(dict.fromkeys(labels))  # unique, preserving order
        ref_name = opts.reference_gene or genes[-1]

        if ref_name not in genes:
            raise ValueError(
                f"Reference gene '{ref_name}' not found in label column. "
                f"Available genes: {genes}"
            )

        targets = [g for g in genes if g != ref_name]
        if not targets:
            raise ValueError("No target genes found (all rows are reference).")

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

            # Step 1: ΔCt = target Ct - reference Ct (row-wise)
            delta_ct = t_rows - ref_rows.values

            # Step 2: ΔΔCt = ΔCt - mean(ΔCt_control)
            control_mean = delta_ct[control].dropna().mean()
            delta_delta_ct = delta_ct - control_mean

            # Step 3: Fold change = 2^(-ΔΔCt)
            fold_df = np.power(2.0, -delta_delta_ct)
            results.append((target, fold_df))

        return results
