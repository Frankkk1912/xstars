"""Tkinter dialog for standard curve fitting settings."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

import numpy as np

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    HAS_TTKB = True
except ImportError:
    HAS_TTKB = False

from .standard_curve import CurveFitResult, fit_standard_curve


@dataclass
class StandardCurveConfig:
    """Configuration returned by the standard curve dialog."""
    fit_method: str = "auto"
    fit_result: CurveFitResult | None = None
    back_calculate: bool = True


class StandardCurveDialog:
    """Dialog for standard curve fitting: pick method, preview fit, choose back-calc.

    Input is a wide DataFrame (columns = concentration labels, rows = OD replicates).
    The dialog extracts concentrations from column names and flattens OD values.
    """

    def __init__(
        self,
        conc: np.ndarray,
        od: np.ndarray,
        group_names: list[str],
        group_sizes: dict[str, int],
    ) -> None:
        self.conc = np.asarray(conc, dtype=float)
        self.od = np.asarray(od, dtype=float)
        self.group_names = group_names
        self.group_sizes = group_sizes
        self.result: StandardCurveConfig | None = None
        self._fit_result: CurveFitResult | None = None

    def show(self) -> StandardCurveConfig | None:
        if HAS_TTKB:
            self._root = ttkb.Window(
                title="Standard Curve Tool",
                themename="cosmo",
                resizable=(False, True),
                minsize=(520, 420),
            )
        else:
            self._root = tk.Tk()
            self._root.title("Standard Curve Tool")
            self._root.resizable(False, True)
            self._root.minsize(480, 400)
        self._build_ui()
        self._root.grab_set()
        self._root.mainloop()
        return self.result

    def _build_ui(self) -> None:
        root = self._root
        pad = {"padx": 10, "pady": 5}

        if HAS_TTKB:
            ttk = ttkb
        else:
            from tkinter import ttk as ttk

        # --- Data summary ---
        summary_frame = ttk.Labelframe(root, text="Standard Data", padding=8)
        summary_frame.pack(fill="x", **pad)

        summary_text = "  |  ".join(
            f"{g}: n={self.group_sizes.get(g, 0)}" for g in self.group_names
        )
        n_unique = len(np.unique(self.conc))
        n_total = len(self.conc)
        conc_range = f"{self.conc.min():.4g} – {self.conc.max():.4g}"
        od_range = f"{self.od.min():.4g} – {self.od.max():.4g}"
        detail_text = (
            f"{summary_text}\n"
            f"{n_unique} concentration levels, {n_total} data points\n"
            f"Concentration range: {conc_range}\n"
            f"OD range: {od_range}"
        )
        ttk.Label(summary_frame, text=detail_text, wraplength=460).pack(anchor="w")

        # --- Fit method ---
        method_frame = ttk.Labelframe(root, text="Fitting Method", padding=6)
        method_frame.pack(fill="x", **pad)

        self._method_var = tk.StringVar(value="Auto")
        method_labels = {
            "Auto": "auto",
            "4PL": "four_pl",
            "3PL (top=max)": "three_pl",
            "Linear": "linear",
            "Log-linear": "log_linear_reg",
            "Interpolation": "interpolation",
        }
        self._method_map = method_labels
        combo = ttk.Combobox(
            method_frame,
            textvariable=self._method_var,
            values=list(method_labels.keys()),
            state="readonly",
            width=20,
        )
        combo.pack(side="left", padx=8, pady=4)

        if HAS_TTKB:
            fit_btn = ttkb.Button(
                method_frame, text="Fit", command=self._on_fit,
                bootstyle="primary", width=8,
            )
        else:
            fit_btn = ttk.Button(method_frame, text="Fit", command=self._on_fit)
        fit_btn.pack(side="left", padx=8)

        # --- Fit results ---
        result_frame = ttk.Labelframe(root, text="Fit Results", padding=6)
        result_frame.pack(fill="both", **pad)

        self._result_text = tk.Text(result_frame, height=6, width=55, state="disabled",
                                    font=("Consolas", 9) if HAS_TTKB else None)
        self._result_text.pack(fill="both", expand=True, padx=4, pady=4)


        # --- Back-calculate option ---
        opts_frame = ttk.Labelframe(root, text="Options", padding=6)
        opts_frame.pack(fill="x", **pad)

        self._back_calc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts_frame,
            text="Back-calculate sample concentrations (select data next)",
            variable=self._back_calc_var,
        ).pack(anchor="w", padx=8)

        # --- OK / Cancel ---
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))
        if HAS_TTKB:
            ttkb.Button(
                btn_frame, text="OK", command=self._on_ok,
                bootstyle="success", width=10,
            ).pack(side="right", padx=4)
            ttkb.Button(
                btn_frame, text="Cancel", command=self._on_cancel,
                bootstyle="secondary", width=10,
            ).pack(side="right", padx=4)
        else:
            ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(
                side="right", padx=4
            )
            ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
                side="right", padx=4
            )

        # Auto-fit on open
        self._on_fit()

    def _on_fit(self) -> None:
        method_label = self._method_var.get()
        method = self._method_map.get(method_label, "auto")

        try:
            self._fit_result = fit_standard_curve(self.conc, self.od, method=method)
        except ValueError as e:
            self._show_result_text(f"Fitting failed: {e}")
            self._fit_result = None
            return

        fr = self._fit_result
        lines = [
            f"Method: {fr.method}",
            f"Equation: {fr.equation_str}",
        ]
        if fr.r_squared is not None:
            lines.append(f"R² = {fr.r_squared:.6f}")
        for k, v in fr.params.items():
            if k != "n_points":
                lines.append(f"  {k} = {v:.6g}")
            else:
                lines.append(f"  {k} = {v}")
        self._show_result_text("\n".join(lines))

    def _show_result_text(self, text: str) -> None:
        self._result_text.configure(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", text)
        self._result_text.configure(state="disabled")

    def _on_ok(self) -> None:
        if self._fit_result is None:
            return
        self.result = StandardCurveConfig(
            fit_method=self._fit_result.method,
            fit_result=self._fit_result,
            back_calculate=self._back_calc_var.get(),
        )
        self._root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self._root.destroy()
