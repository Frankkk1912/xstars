"""Tkinter dialog for ELISA preset — std curve fit + full analysis settings."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog

import numpy as np

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    try:
        from ttkbootstrap.scrolled import ScrolledFrame
    except ImportError:
        from ttkbootstrap.widgets.scrolled import ScrolledFrame
    HAS_TTKB = True
except ImportError:
    HAS_TTKB = False

from ..config import (
    AnnotationFormat, BaseTheme, ChartType, ErrorBarType,
    JournalPalette, JournalPreset, PalettePreset, PrismConfig,
)
from ..tools.standard_curve import (
    CurveFitResult,
    fit_standard_curve,
    four_param_logistic,
)


@dataclass
class ELISADialogResult:
    """Configuration returned by the ELISA dialog."""
    fit_result: CurveFitResult | None = None
    use_existing_params: bool = False
    config: PrismConfig | None = None
    show_fit_curve: bool = True


class ELISADialog:
    """Dialog for ELISA: std curve fitting + analysis settings in one window.

    Tabs: General | Theme | Standard Curve | Export
    """

    def __init__(
        self,
        conc: np.ndarray,
        od: np.ndarray,
        base_config: PrismConfig | None = None,
    ) -> None:
        self.conc = np.asarray(conc, dtype=float)
        self.od = np.asarray(od, dtype=float)
        self.config = base_config if base_config is not None else PrismConfig.load()
        self.result: ELISADialogResult | None = None
        self._fit_result: CurveFitResult | None = None

    def show(self) -> ELISADialogResult | None:
        if HAS_TTKB:
            self._root = ttkb.Window(
                title="ELISA — Analysis Settings",
                themename="cosmo",
                resizable=(False, True),
                minsize=(640, 780),
            )
        else:
            self._root = tk.Tk()
            self._root.title("ELISA — Analysis Settings")
            self._root.resizable(False, True)
            self._root.minsize(560, 700)
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

        # --- Data summary banner ---
        summary_frame = ttk.Labelframe(root, text="Standard Data", padding=8)
        summary_frame.pack(fill="x", **pad)

        n_unique = len(np.unique(self.conc))
        n_total = len(self.conc)
        conc_range = f"{self.conc.min():.4g} – {self.conc.max():.4g}"
        od_range = f"{self.od.min():.4g} – {self.od.max():.4g}"
        detail_text = (
            f"{n_unique} concentration levels, {n_total} data points  |  "
            f"Conc: {conc_range}  |  OD: {od_range}"
        )
        ttk.Label(summary_frame, text=detail_text, wraplength=580).pack(anchor="w")

        # --- Notebook (tabs) ---
        if HAS_TTKB:
            notebook = ttkb.Notebook(root, bootstyle="primary")
        else:
            notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, **pad)

        # ===================== Tab 1: General =====================
        general_tab, general_frame = self._make_scrollable(notebook)
        notebook.add(general_tab, text="  General  ")

        # -- Chart type --
        chart_frame = ttk.Labelframe(general_frame, text="Chart Type", padding=6)
        chart_frame.pack(fill="x", **pad)
        self._chart_var = tk.StringVar(value=self.config.chart_type.value)
        chart_labels = {
            ChartType.BAR_SCATTER: "Bar + Scatter",
            ChartType.VIOLIN: "Violin",
            ChartType.LINE: "Line",
        }
        row = ttk.Frame(chart_frame)
        row.pack(fill="x")
        for ct in ChartType:
            ttk.Radiobutton(
                row, text=chart_labels[ct],
                variable=self._chart_var, value=ct.value,
            ).pack(side="left", padx=8)

        # -- Error bars --
        eb_frame = ttk.Labelframe(general_frame, text="Error Bars", padding=6)
        eb_frame.pack(fill="x", **pad)
        self._eb_var = tk.StringVar(value=self.config.error_bar.value)
        eb_row = ttk.Frame(eb_frame)
        eb_row.pack(fill="x")
        for eb in ErrorBarType:
            label = {"sem": "SEM", "sd": "SD", "ci95": "95% CI"}[eb.value]
            ttk.Radiobutton(
                eb_row, text=label, variable=self._eb_var, value=eb.value,
            ).pack(side="left", padx=8)

        # -- Options --
        opts_frame = ttk.Labelframe(general_frame, text="Options", padding=6)
        opts_frame.pack(fill="x", **pad)

        self._points_var = tk.BooleanVar(value=self.config.show_points)
        ttk.Checkbutton(
            opts_frame, text="Show data points", variable=self._points_var
        ).pack(anchor="w", padx=8)

        self._ns_var = tk.BooleanVar(value=self.config.show_ns)
        ttk.Checkbutton(
            opts_frame, text='Show "ns" labels', variable=self._ns_var
        ).pack(anchor="w", padx=8)

        # -- Annotation Format --
        annot_frame = ttk.Labelframe(general_frame, text="Annotation Format", padding=6)
        annot_frame.pack(fill="x", **pad)
        self._annot_var = tk.StringVar(value=self.config.annotation_format.value)
        annot_row = ttk.Frame(annot_frame)
        annot_row.pack(fill="x")
        ttk.Radiobutton(
            annot_row, text="Stars (***)", variable=self._annot_var,
            value=AnnotationFormat.STARS.value,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            annot_row, text="Scientific (p=1.2e-4)", variable=self._annot_var,
            value=AnnotationFormat.SCIENTIFIC.value,
        ).pack(side="left", padx=8)

        # -- Title --
        title_frame = ttk.Labelframe(general_frame, text="Chart Title (optional)", padding=6)
        title_frame.pack(fill="x", **pad)
        self._title_var = tk.StringVar(value=self.config.title)
        ttk.Entry(title_frame, textvariable=self._title_var, width=30).pack(
            fill="x", padx=8, pady=4)

        # -- Y-axis label --
        label_frame = ttk.Labelframe(general_frame, text="Y-axis Label", padding=6)
        label_frame.pack(fill="x", **pad)
        self._ylabel_var = tk.StringVar(value="Concentration")
        ttk.Entry(label_frame, textvariable=self._ylabel_var, width=30).pack(
            fill="x", padx=8, pady=4)

        # ===================== Tab 2: Theme =====================
        theme_tab, theme_content = self._make_scrollable(notebook)
        notebook.add(theme_tab, text="  Theme  ")

        # -- Base Theme --
        bt_frame = ttk.Labelframe(theme_content, text="Base Theme", padding=6)
        bt_frame.pack(fill="x", **pad)
        self._theme_var = tk.StringVar(value=self.config.base_theme.value)
        theme_labels = {
            BaseTheme.CLASSIC: "Classic", BaseTheme.BW: "BW",
            BaseTheme.MINIMAL: "Minimal", BaseTheme.DARK: "Dark",
        }
        bt_row = ttk.Frame(bt_frame)
        bt_row.pack(fill="x")
        for bt in BaseTheme:
            ttk.Radiobutton(
                bt_row, text=theme_labels[bt],
                variable=self._theme_var, value=bt.value,
            ).pack(side="left", padx=8)

        # -- Journal Typography --
        jp_frame = ttk.Labelframe(theme_content, text="Journal Typography", padding=6)
        jp_frame.pack(fill="x", **pad)
        self._preset_var = tk.StringVar(value=self.config.journal_preset.value)
        preset_labels = {
            JournalPreset.NONE: "Default", JournalPreset.NATURE: "Nature",
            JournalPreset.SCIENCE: "Science", JournalPreset.CELL: "Cell",
            JournalPreset.LANCET: "Lancet", JournalPreset.NEJM: "NEJM",
            JournalPreset.JAMA: "JAMA", JournalPreset.BMJ: "BMJ",
        }
        jp_row1 = ttk.Frame(jp_frame)
        jp_row1.pack(fill="x")
        jp_row2 = ttk.Frame(jp_frame)
        jp_row2.pack(fill="x", pady=(2, 0))
        for i, jp in enumerate(JournalPreset):
            r = jp_row1 if i < 4 else jp_row2
            ttk.Radiobutton(
                r, text=preset_labels[jp],
                variable=self._preset_var, value=jp.value,
            ).pack(side="left", padx=8)

        # -- Journal Palette --
        jpal_frame = ttk.Labelframe(theme_content, text="Journal Palette", padding=6)
        jpal_frame.pack(fill="x", **pad)
        self._jpal_var = tk.StringVar(value=self.config.journal_palette.value)
        jpal_labels = {
            JournalPalette.DEFAULT: "Default", JournalPalette.NATURE: "Nature",
            JournalPalette.SCIENCE: "Science", JournalPalette.CELL: "Cell",
            JournalPalette.LANCET: "Lancet", JournalPalette.NEJM: "NEJM",
            JournalPalette.JAMA: "JAMA", JournalPalette.BMJ: "BMJ",
        }
        jpal_row1 = ttk.Frame(jpal_frame)
        jpal_row1.pack(fill="x")
        jpal_row2 = ttk.Frame(jpal_frame)
        jpal_row2.pack(fill="x", pady=(2, 0))
        for i, jp in enumerate(JournalPalette):
            r = jpal_row1 if i < 4 else jpal_row2
            ttk.Radiobutton(
                r, text=jpal_labels[jp],
                variable=self._jpal_var, value=jp.value,
            ).pack(side="left", padx=8)

        # -- Color Style --
        pal_frame = ttk.Labelframe(theme_content, text="Color Style", padding=6)
        pal_frame.pack(fill="x", **pad)
        self._palette_var = tk.StringVar(value=self.config.palette_preset.value)
        pal_labels = {
            PalettePreset.DEFAULT: "Original", PalettePreset.PASTEL: "Pastel",
            PalettePreset.DEEP: "Deep", PalettePreset.VIBRANT: "Vibrant",
            PalettePreset.MUTED: "Muted", PalettePreset.COLORBLIND: "Colorblind",
        }
        pal_row1 = ttk.Frame(pal_frame)
        pal_row1.pack(fill="x")
        pal_row2 = ttk.Frame(pal_frame)
        pal_row2.pack(fill="x", pady=(2, 0))
        for i, pp in enumerate(PalettePreset):
            r = pal_row1 if i < 3 else pal_row2
            ttk.Radiobutton(
                r, text=pal_labels[pp],
                variable=self._palette_var, value=pp.value,
            ).pack(side="left", padx=8)

        # ===================== Tab 3: Standard Curve =====================
        curve_tab, curve_frame = self._make_scrollable(notebook)
        notebook.add(curve_tab, text="  Standard Curve  ")

        # -- Mode toggle --
        self._use_existing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            curve_frame, text="Use existing parameters (skip fitting)",
            variable=self._use_existing_var,
            command=self._on_mode_toggle,
        ).pack(anchor="w", padx=8, pady=(8, 4))

        # -- Fit from data frame --
        self._fit_frame = ttk.Labelframe(curve_frame, text="Curve Fitting", padding=6)
        self._fit_frame.pack(fill="both", expand=True, **pad)

        method_row = ttk.Frame(self._fit_frame)
        method_row.pack(fill="x", padx=4, pady=4)
        ttk.Label(method_row, text="Method:").pack(side="left")
        self._method_var = tk.StringVar(value="Auto")
        self._method_map = {
            "Auto": "auto",
            "4PL": "four_pl",
            "3PL": "three_pl",
            "Linear": "linear",
            "Log-linear": "log_linear_reg",
            "Interpolation": "interpolation",
        }
        ttk.Combobox(
            method_row, textvariable=self._method_var,
            values=list(self._method_map.keys()),
            state="readonly", width=16,
        ).pack(side="left", padx=(8, 4))

        if HAS_TTKB:
            ttkb.Button(
                method_row, text="Fit", command=self._on_fit,
                bootstyle="primary", width=6,
            ).pack(side="left", padx=4)
        else:
            ttk.Button(method_row, text="Fit", command=self._on_fit).pack(
                side="left", padx=4)

        self._result_text = tk.Text(
            self._fit_frame, height=6, width=55, state="disabled",
            font=("Consolas", 9) if HAS_TTKB else None,
        )
        self._result_text.pack(fill="both", expand=True, padx=4, pady=4)

        # -- Existing params frame --
        self._existing_frame = ttk.Labelframe(
            curve_frame, text="Existing Parameters", padding=6)

        ep_method_row = ttk.Frame(self._existing_frame)
        ep_method_row.pack(fill="x", padx=4, pady=4)
        ttk.Label(ep_method_row, text="Method:").pack(side="left")
        self._ep_method_var = tk.StringVar(value="4PL")
        self._ep_method_map = {
            "4PL": "four_pl",
            "3PL": "three_pl",
            "Linear": "linear",
        }
        ttk.Combobox(
            ep_method_row, textvariable=self._ep_method_var,
            values=list(self._ep_method_map.keys()),
            state="readonly", width=16,
        ).pack(side="left", padx=(8, 0))
        self._ep_method_var.trace_add("write", lambda *_: self._on_ep_method_change())

        self._ep_params_frame = ttk.Frame(self._existing_frame)
        self._ep_params_frame.pack(fill="x", padx=4, pady=2)

        self._ep_param_vars: dict[str, tk.StringVar] = {}
        for name in ("bottom", "top", "ec50", "hill", "slope", "intercept"):
            row = ttk.Frame(self._ep_params_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{name}:", width=10).pack(side="left")
            var = tk.StringVar(value="")
            ttk.Entry(row, textvariable=var, width=14).pack(side="left", padx=4)
            self._ep_param_vars[name] = var
            setattr(self, f"_ep_row_{name}", row)

        self._on_ep_method_change()

        # -- Output Options (inside Standard Curve tab) --
        output_frame = ttk.Labelframe(curve_frame, text="Output Options", padding=6)
        output_frame.pack(fill="x", **pad)

        self._show_curve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_frame, text="Insert standard curve chart",
            variable=self._show_curve_var,
        ).pack(anchor="w", padx=8)

        self._output_stats_var = tk.BooleanVar(value=self.config.output_stats)
        ttk.Checkbutton(
            output_frame, text="Write stats table to Excel",
            variable=self._output_stats_var,
        ).pack(anchor="w", padx=8)

        self._output_data_var = tk.BooleanVar(value=self.config.output_data)
        ttk.Checkbutton(
            output_frame, text="Write processed data to Excel",
            variable=self._output_data_var,
        ).pack(anchor="w", padx=8)

        # ===================== Tab 4: Export =====================
        export_tab, export_content = self._make_scrollable(notebook)
        notebook.add(export_tab, text="  Export  ")

        export_frame = ttk.Labelframe(export_content, text="Export Settings", padding=6)
        export_frame.pack(fill="x", **pad)

        self._export_var = tk.BooleanVar(value=bool(self.config.export_path))
        self._export_check = ttk.Checkbutton(
            export_frame, text="Export to file",
            variable=self._export_var, command=self._toggle_export,
        )
        self._export_check.pack(anchor="w", padx=8)

        # Format
        fmt_row = ttk.Frame(export_frame)
        fmt_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(fmt_row, text="Format:").pack(side="left")
        self._export_fmt_labels = {
            "PNG image": "png",
            "TIFF image": "tiff",
            "JPEG image": "jpg",
            "SVG image": "svg",
            "PDF document": "pdf",
        }
        _fmt_rev = {v: k for k, v in self._export_fmt_labels.items()}
        _cur_fmt = _fmt_rev.get(self.config.export_format, "PNG image")
        self._export_fmt_var = tk.StringVar(value=_cur_fmt)
        self._export_fmt_combo = ttk.Combobox(
            fmt_row, textvariable=self._export_fmt_var,
            values=list(self._export_fmt_labels.keys()),
            state="readonly", width=16,
        )
        self._export_fmt_combo.pack(side="left", padx=(8, 0))
        self._export_fmt_combo.bind("<<ComboboxSelected>>", lambda _: self._update_export_ext())

        # DPI
        dpi_row = ttk.Frame(export_frame)
        dpi_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(dpi_row, text="DPI:").pack(side="left")
        self._export_dpi_var = tk.StringVar(value=str(self.config.export_dpi))
        self._export_dpi_combo = ttk.Combobox(
            dpi_row, textvariable=self._export_dpi_var,
            values=["150", "300", "600", "1200"],
            state="readonly", width=16,
        )
        self._export_dpi_combo.pack(side="left", padx=(8, 0))

        # Path
        path_row = ttk.Frame(export_frame)
        path_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(path_row, text="Path:").pack(side="left")
        self._export_path_var = tk.StringVar(value=self.config.export_path)
        self._export_entry = ttk.Entry(
            path_row, textvariable=self._export_path_var, width=24,
        )
        self._export_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self._browse_btn = ttk.Button(
            path_row, text="Browse...", command=self._browse_export,
        )
        self._browse_btn.pack(side="left", padx=(4, 0))
        self._toggle_export()

        # --- OK / Cancel buttons ---
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
                side="right", padx=4)
            ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
                side="right", padx=4)

        # Auto-fit on open
        self._on_fit()

    def _make_scrollable(self, parent):
        if HAS_TTKB:
            wrapper = ttkb.Frame(parent)
            sf = ScrolledFrame(wrapper, autohide=True, padding=5)
            sf.pack(fill="both", expand=True)
            return wrapper, sf
        else:
            from tkinter import ttk
            f = ttk.Frame(parent, padding=5)
            return f, f

    # ------------------------------------------------------------------
    # Standard Curve callbacks
    # ------------------------------------------------------------------

    def _on_mode_toggle(self) -> None:
        if self._use_existing_var.get():
            self._fit_frame.pack_forget()
            self._existing_frame.pack(fill="both", expand=True, padx=10, pady=5)
        else:
            self._existing_frame.pack_forget()
            self._fit_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def _on_ep_method_change(self) -> None:
        method = self._ep_method_map.get(self._ep_method_var.get(), "four_pl")
        show = {"slope", "intercept"} if method == "linear" else {
            "bottom", "top", "ec50", "hill"}
        for name in ("bottom", "top", "ec50", "hill", "slope", "intercept"):
            row = getattr(self, f"_ep_row_{name}")
            if name in show:
                row.pack(fill="x", pady=1)
            else:
                row.pack_forget()

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

    # ------------------------------------------------------------------
    # Export callbacks
    # ------------------------------------------------------------------

    def _toggle_export(self) -> None:
        state = "normal" if self._export_var.get() else "disabled"
        self._export_fmt_combo.configure(state=state if state == "disabled" else "readonly")
        self._export_dpi_combo.configure(state=state if state == "disabled" else "readonly")
        self._export_entry.configure(state=state)
        self._browse_btn.configure(state=state)

    def _update_export_ext(self) -> None:
        ext = self._export_fmt_labels.get(self._export_fmt_var.get(), "png")
        cur = self._export_path_var.get()
        if cur:
            from pathlib import Path as _P
            self._export_path_var.set(str(_P(cur).with_suffix(f".{ext}")))

    def _browse_export(self) -> None:
        ext = self._export_fmt_labels.get(self._export_fmt_var.get(), "png")
        fmt_label = self._export_fmt_var.get()
        path = filedialog.asksaveasfilename(
            defaultextension=f".{ext}",
            filetypes=[(fmt_label, f"*.{ext}"), ("All files", "*.*")],
        )
        if path:
            self._export_path_var.set(path)

    # ------------------------------------------------------------------
    # Build fit from existing params
    # ------------------------------------------------------------------

    def _build_fit_from_params(self) -> CurveFitResult | None:
        """Build a CurveFitResult from user-entered parameters."""
        method = self._ep_method_map.get(self._ep_method_var.get(), "four_pl")
        try:
            if method == "linear":
                slope = float(self._ep_param_vars["slope"].get())
                intercept = float(self._ep_param_vars["intercept"].get())

                def predict(c):
                    return slope * np.asarray(c, dtype=float) + intercept

                def inverse(y):
                    y = np.asarray(y, dtype=float)
                    if slope == 0:
                        return np.full_like(y, np.nan, dtype=float)
                    return (y - intercept) / slope

                return CurveFitResult(
                    method="linear",
                    params={"slope": slope, "intercept": intercept},
                    r_squared=None,
                    equation_str=f"y = {slope:.4g} * x + {intercept:.4g}",
                    predict=predict,
                    inverse=inverse,
                    conc_range=(0.0, 1e6),
                )
            else:
                bottom = float(self._ep_param_vars["bottom"].get())
                top = float(self._ep_param_vars["top"].get())
                ec50 = float(self._ep_param_vars["ec50"].get())
                hill = float(self._ep_param_vars["hill"].get())

                def predict(c):
                    return four_param_logistic(
                        np.asarray(c, dtype=float), bottom, top, ec50, hill)

                def inverse(y):
                    y = np.asarray(y, dtype=float)
                    result = np.full_like(y, np.nan, dtype=float)
                    for i, yi in enumerate(y.flat):
                        try:
                            ratio = (top - bottom) / (yi - bottom) - 1.0
                            if ratio <= 0:
                                continue
                            result.flat[i] = ec50 * ratio ** (1.0 / hill)
                        except (ZeroDivisionError, ValueError):
                            continue
                    return result

                return CurveFitResult(
                    method=method,
                    params={"bottom": bottom, "top": top,
                            "ec50": ec50, "hill": hill},
                    r_squared=None,
                    equation_str=(
                        f"y = {bottom:.4g} + ({top:.4g} - {bottom:.4g}) "
                        f"/ (1 + (x/{ec50:.4g})^{hill:.4g})"
                    ),
                    predict=predict,
                    inverse=inverse,
                    conc_range=(0.0, 1e6),
                )
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # OK / Cancel
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        # Resolve fit result
        if self._use_existing_var.get():
            fit = self._build_fit_from_params()
            if fit is None:
                return
            use_existing = True
        else:
            if self._fit_result is None:
                return
            fit = self._fit_result
            use_existing = False

        # Build PrismConfig from dialog state
        from ..styles import get_palette

        palette_preset = PalettePreset(self._palette_var.get())
        journal_palette = JournalPalette(self._jpal_var.get())

        config = PrismConfig(
            chart_type=ChartType(self._chart_var.get()),
            error_bar=ErrorBarType(self._eb_var.get()),
            show_points=self._points_var.get(),
            show_ns=self._ns_var.get(),
            annotation_format=AnnotationFormat(self._annot_var.get()),
            journal_preset=JournalPreset(self._preset_var.get()),
            base_theme=BaseTheme(self._theme_var.get()),
            palette_preset=palette_preset,
            journal_palette=journal_palette,
            palette=get_palette(palette_preset, journal_palette),
            title=self._title_var.get(),
            y_label=self._ylabel_var.get(),
            output_stats=self._output_stats_var.get(),
            output_data=self._output_data_var.get(),
            export_path=(
                self._export_path_var.get() if self._export_var.get() else ""
            ),
            export_format=self._export_fmt_labels.get(self._export_fmt_var.get(), "png"),
            export_dpi=int(self._export_dpi_var.get()),
        )
        config.save()

        self.result = ELISADialogResult(
            fit_result=fit,
            use_existing_params=use_existing,
            config=config,
            show_fit_curve=self._show_curve_var.get(),
        )
        self._root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self._root.destroy()
