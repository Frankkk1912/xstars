"""Tkinter settings dialog for XSTARS (ttkbootstrap modernized)."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    try:
        from ttkbootstrap.widgets.scrolled import ScrolledFrame
    except ImportError:
        from ttkbootstrap.scrolled import ScrolledFrame
    HAS_TTKB = True
except ImportError:
    HAS_TTKB = False

from .config import (
    AnnotationFormat, BaseTheme, ChartType, DoseAxisScale, ErrorBarType,
    ExperimentPreset, FitMethod, JournalPalette, JournalPreset, PalettePreset,
    PrismConfig,
)


class SettingsDialog:
    """Modal dialog that lets the user tweak analysis settings.

    Returns an updated ``PrismConfig`` when the user clicks OK,
    or ``None`` if cancelled.
    """

    def __init__(
        self,
        group_names: list[str],
        group_sizes: dict[str, int],
        base_config: PrismConfig | None = None,
    ) -> None:
        self.group_names = group_names
        self.group_sizes = group_sizes
        # Load persisted settings as defaults, then overlay any explicit base_config
        saved = PrismConfig.load()
        if base_config is not None:
            self.config = base_config
        else:
            self.config = saved
        self.result: PrismConfig | None = None

    def show(self) -> PrismConfig | None:
        """Display the dialog and block until closed. Returns config or None."""
        if HAS_TTKB:
            self._root = ttkb.Window(
                title="XSTARS Settings",
                themename="cosmo",
                resizable=(False, True),
                minsize=(640, 780),
            )
        else:
            self._root = tk.Tk()
            self._root.title("XSTARS Settings")
            self._root.resizable(False, True)
            self._root.minsize(560, 700)
        self._build_ui()
        self._root.grab_set()
        self._root.mainloop()
        return self.result

    # ------------------------------------------------------------------
    # UI building
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self._root
        pad = {"padx": 10, "pady": 5}

        if HAS_TTKB:
            ttk = ttkb
        else:
            from tkinter import ttk as ttk

        # --- Data summary banner ---
        summary_frame = ttk.Labelframe(root, text="Detected Data", padding=8)
        summary_frame.pack(fill="x", **pad)
        summary_text = "  |  ".join(
            f"{g}: n={self.group_sizes.get(g, 0)}" for g in self.group_names
        )
        ttk.Label(summary_frame, text=summary_text, wraplength=380).pack(anchor="w")

        # --- Notebook (tabs) ---
        if HAS_TTKB:
            notebook = ttkb.Notebook(root, bootstyle="primary")
        else:
            notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, **pad)

        # ===================== General tab =====================
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
            rb = ttk.Radiobutton(
                row, text=chart_labels[ct],
                variable=self._chart_var, value=ct.value,
            )
            rb.pack(side="left", padx=8)

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

        self._paired_var = tk.BooleanVar(value=self.config.paired)
        if len(self.group_names) == 2:
            ttk.Checkbutton(
                opts_frame, text="Paired comparison", variable=self._paired_var
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

        # -- Comparison Mode --
        comp_frame = ttk.Labelframe(general_frame, text="Comparison Mode", padding=6)
        comp_frame.pack(fill="x", **pad)
        control_val = self.config.control_group if self.config.control_group else "all_pairwise"
        self._control_var = tk.StringVar(value=control_val)
        choices = ["All pairwise"] + list(self.group_names)
        self._control_combo = ttk.Combobox(
            comp_frame, textvariable=self._control_var,
            values=choices, state="readonly", width=28,
        )
        self._control_combo.pack(fill="x", padx=8, pady=4)
        if self.config.control_group and self.config.control_group in self.group_names:
            self._control_combo.set(self.config.control_group)
        else:
            self._control_combo.set("All pairwise")

        # -- Title --
        title_frame = ttk.Labelframe(general_frame, text="Chart Title (optional)", padding=6)
        title_frame.pack(fill="x", **pad)
        self._title_var = tk.StringVar(value=self.config.title)
        ttk.Entry(title_frame, textvariable=self._title_var, width=30).pack(
            fill="x", padx=8, pady=4
        )

        # -- Y-axis label --
        label_frame = ttk.Labelframe(general_frame, text="Y-axis Label", padding=6)
        label_frame.pack(fill="x", **pad)
        self._ylabel_var = tk.StringVar(value=self.config.y_label)
        ttk.Entry(label_frame, textvariable=self._ylabel_var, width=30).pack(
            fill="x", padx=8, pady=4
        )

        # ===================== Theme tab =====================
        theme_tab, theme_content = self._make_scrollable(notebook)
        notebook.add(theme_tab, text="  Theme  ")

        # -- Base Theme --
        theme_frame = ttk.Labelframe(theme_content, text="Base Theme", padding=6)
        theme_frame.pack(fill="x", **pad)
        self._theme_var = tk.StringVar(value=self.config.base_theme.value)
        theme_labels = {
            BaseTheme.CLASSIC: "Classic",
            BaseTheme.BW: "BW",
            BaseTheme.MINIMAL: "Minimal",
            BaseTheme.DARK: "Dark",
        }
        theme_row = ttk.Frame(theme_frame)
        theme_row.pack(fill="x")
        for bt in BaseTheme:
            ttk.Radiobutton(
                theme_row, text=theme_labels[bt],
                variable=self._theme_var, value=bt.value,
            ).pack(side="left", padx=8)

        # -- Journal Typography --
        journal_frame = ttk.Labelframe(theme_content, text="Journal Typography", padding=6)
        journal_frame.pack(fill="x", **pad)
        self._preset_var = tk.StringVar(value=self.config.journal_preset.value)
        preset_labels = {
            JournalPreset.NONE: "Default",
            JournalPreset.NATURE: "Nature",
            JournalPreset.SCIENCE: "Science",
            JournalPreset.CELL: "Cell",
            JournalPreset.LANCET: "Lancet",
            JournalPreset.NEJM: "NEJM",
            JournalPreset.JAMA: "JAMA",
            JournalPreset.BMJ: "BMJ",
        }
        jp_row1 = ttk.Frame(journal_frame)
        jp_row1.pack(fill="x")
        jp_row2 = ttk.Frame(journal_frame)
        jp_row2.pack(fill="x", pady=(2, 0))
        for i, jp in enumerate(JournalPreset):
            row = jp_row1 if i < 4 else jp_row2
            ttk.Radiobutton(
                row, text=preset_labels[jp],
                variable=self._preset_var, value=jp.value,
            ).pack(side="left", padx=8)

        # -- Journal Palette (ggsci-style) --
        jpal_frame = ttk.Labelframe(theme_content, text="Journal Palette", padding=6)
        jpal_frame.pack(fill="x", **pad)
        self._jpal_var = tk.StringVar(value=self.config.journal_palette.value)
        jpal_labels = {
            JournalPalette.DEFAULT: "Default",
            JournalPalette.NATURE: "Nature",
            JournalPalette.SCIENCE: "Science",
            JournalPalette.CELL: "Cell",
            JournalPalette.LANCET: "Lancet",
            JournalPalette.NEJM: "NEJM",
            JournalPalette.JAMA: "JAMA",
            JournalPalette.BMJ: "BMJ",
        }
        jpal_row1 = ttk.Frame(jpal_frame)
        jpal_row1.pack(fill="x")
        jpal_row2 = ttk.Frame(jpal_frame)
        jpal_row2.pack(fill="x", pady=(2, 0))
        for i, jp in enumerate(JournalPalette):
            row = jpal_row1 if i < 4 else jpal_row2
            ttk.Radiobutton(
                row, text=jpal_labels[jp],
                variable=self._jpal_var, value=jp.value,
            ).pack(side="left", padx=8)

        # -- Color Palette --
        pal_frame = ttk.Labelframe(theme_content, text="Color Style", padding=6)
        pal_frame.pack(fill="x", **pad)
        self._palette_var = tk.StringVar(value=self.config.palette_preset.value)
        pal_labels = {
            PalettePreset.DEFAULT: "Original",
            PalettePreset.PASTEL: "Pastel",
            PalettePreset.DEEP: "Deep",
            PalettePreset.VIBRANT: "Vibrant",
            PalettePreset.MUTED: "Muted",
            PalettePreset.COLORBLIND: "Colorblind",
        }
        pal_row1 = ttk.Frame(pal_frame)
        pal_row1.pack(fill="x")
        pal_row2 = ttk.Frame(pal_frame)
        pal_row2.pack(fill="x", pady=(2, 0))
        for i, pp in enumerate(PalettePreset):
            row = pal_row1 if i < 3 else pal_row2
            ttk.Radiobutton(
                row, text=pal_labels[pp],
                variable=self._palette_var, value=pp.value,
            ).pack(side="left", padx=8)

        # -- Reset Settings --
        reset_frame = ttk.Labelframe(theme_content, text="Reset", padding=6)
        reset_frame.pack(fill="x", **pad)
        if HAS_TTKB:
            ttkb.Button(
                reset_frame, text="Reset All Settings to Defaults",
                command=self._on_reset, bootstyle="warning-outline",
            ).pack(anchor="w", padx=8, pady=4)
        else:
            ttk.Button(
                reset_frame, text="Reset All Settings to Defaults",
                command=self._on_reset,
            ).pack(anchor="w", padx=8, pady=4)

        # ===================== Preset tab =====================
        preset_tab, preset_content = self._make_scrollable(notebook)
        notebook.add(preset_tab, text="  Preset  ")

        # --- Experiment Preset ---
        exp_frame = ttk.Labelframe(preset_content, text="Experiment Preset", padding=6)
        exp_frame.pack(fill="x", **pad)

        self._exp_var = tk.StringVar(value=self.config.experiment_preset.value)
        exp_labels = {
            ExperimentPreset.NONE: "None",
            ExperimentPreset.WB: "Western Blot",
            ExperimentPreset.QPCR: "qPCR (\u0394\u0394Ct)",
            ExperimentPreset.CCK8: "CCK-8 Viability",
            ExperimentPreset.ELISA: "ELISA",
        }
        exp_combo = ttk.Combobox(
            exp_frame,
            textvariable=self._exp_var,
            values=[f"{exp_labels[ep]}" for ep in ExperimentPreset],
            state="readonly", width=28,
        )
        exp_combo.pack(fill="x", padx=8, pady=4)
        self._exp_value_map = {exp_labels[ep]: ep.value for ep in ExperimentPreset}
        self._exp_label_map = {ep.value: exp_labels[ep] for ep in ExperimentPreset}
        self._exp_var.set(self._exp_label_map.get(self.config.experiment_preset.value, "None"))

        # -- WB sub-frame --
        self._wb_frame = ttk.Frame(exp_frame)
        ttk.Label(self._wb_frame, text="Control group:").pack(anchor="w", padx=8)
        self._wb_control_var = tk.StringVar(value=self.config.preset_control_group or self.group_names[0])
        ttk.Combobox(
            self._wb_frame, textvariable=self._wb_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        self._wb_ref_var = tk.BooleanVar(value=self.config.preset_has_reference)
        ttk.Checkbutton(
            self._wb_frame, text="Has reference protein (e.g. GAPDH)",
            variable=self._wb_ref_var,
            command=self._on_wb_ref_toggle,
        ).pack(anchor="w", padx=8)
        self._wb_ref_hint = ttk.Label(
            self._wb_frame,
            text="First column = protein name; last protein = reference",
            foreground="gray",
        )
        self._on_wb_ref_toggle()

        # -- qPCR sub-frame --
        self._qpcr_frame = ttk.Frame(exp_frame)
        ttk.Label(self._qpcr_frame, text="Control group:").pack(anchor="w", padx=8)
        self._qpcr_control_var = tk.StringVar(value=self.config.preset_control_group or self.group_names[0])
        ttk.Combobox(
            self._qpcr_frame, textvariable=self._qpcr_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        self._qpcr_ref_var = tk.BooleanVar(value=self.config.preset_has_reference)
        ttk.Checkbutton(
            self._qpcr_frame, text="Has reference gene (e.g. GAPDH)",
            variable=self._qpcr_ref_var,
            command=self._on_qpcr_ref_toggle,
        ).pack(anchor="w", padx=8)
        self._qpcr_ref_hint = ttk.Label(
            self._qpcr_frame,
            text="First column = gene name; last gene = reference",
            foreground="gray",
        )
        self._qpcr_format_frame = ttk.Frame(self._qpcr_frame)
        self._qpcr_format_var = tk.StringVar(value=self.config.preset_input_format)
        ttk.Radiobutton(
            self._qpcr_format_frame, text="\u0394Ct (pre-computed)",
            variable=self._qpcr_format_var, value="delta_ct",
        ).pack(anchor="w", padx=8)
        ttk.Radiobutton(
            self._qpcr_format_frame, text="Raw Ct (target + reference rows)",
            variable=self._qpcr_format_var, value="raw_ct",
        ).pack(anchor="w", padx=8)
        self._on_qpcr_ref_toggle()

        # -- CCK-8 sub-frame --
        self._cck8_frame = ttk.Frame(exp_frame)
        ttk.Label(self._cck8_frame, text="Control group:").pack(anchor="w", padx=8)
        cck8_ctrl_default = self.config.preset_control_group or self._guess_control()
        self._cck8_control_var = tk.StringVar(value=cck8_ctrl_default)
        ttk.Combobox(
            self._cck8_frame, textvariable=self._cck8_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        ttk.Label(self._cck8_frame, text="Blank group:").pack(anchor="w", padx=8)
        blank_choices = ["(none)"] + list(self.group_names)
        cck8_blank_default = self.config.preset_blank_group or self._guess_blank()
        self._cck8_blank_var = tk.StringVar(
            value=cck8_blank_default or "(none)"
        )
        ttk.Combobox(
            self._cck8_frame, textvariable=self._cck8_blank_var,
            values=blank_choices, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        self._cck8_ic50_var = tk.BooleanVar(value=self.config.preset_fit_ic50)
        ttk.Checkbutton(
            self._cck8_frame, text="Fit IC50 (4PL curve)",
            variable=self._cck8_ic50_var,
            command=self._on_cck8_ic50_toggle,
        ).pack(anchor="w", padx=8)
        # Concentration input
        self._cck8_conc_frame = ttk.Frame(self._cck8_frame)
        ttk.Label(
            self._cck8_conc_frame,
            text="Concentrations (comma-separated):",
        ).pack(anchor="w", padx=8)
        self._cck8_conc_var = tk.StringVar(
            value=self.config.preset_concentrations
        )
        ttk.Entry(
            self._cck8_conc_frame,
            textvariable=self._cck8_conc_var,
            width=30,
        ).pack(fill="x", padx=8, pady=2)
        ttk.Label(
            self._cck8_conc_frame,
            text="e.g. 0.1, 1, 10, 100 (one per dose column)",
            foreground="gray",
        ).pack(anchor="w", padx=8)
        # X-axis scale selector
        axis_row = ttk.Frame(self._cck8_conc_frame)
        axis_row.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(axis_row, text="X-axis scale:").pack(side="left")
        self._cck8_axis_scale_var = tk.StringVar(
            value=self.config.preset_dose_axis_scale.value
        )
        self._cck8_axis_labels = {
            "Auto": "auto", "Log": "log", "Linear": "linear",
        }
        self._cck8_axis_label_rev = {v: k for k, v in self._cck8_axis_labels.items()}
        self._cck8_axis_scale_var.set(
            self._cck8_axis_label_rev.get(
                self.config.preset_dose_axis_scale.value, "Auto"
            )
        )
        ttk.Combobox(
            axis_row,
            textvariable=self._cck8_axis_scale_var,
            values=["Auto", "Log", "Linear"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            self._cck8_conc_frame,
            text="Auto: log if range > 100x, linear otherwise",
            foreground="gray",
        ).pack(anchor="w", padx=8)
        # Fit method selector
        fit_row = ttk.Frame(self._cck8_conc_frame)
        fit_row.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(fit_row, text="Fit method:").pack(side="left")
        self._cck8_fit_method_var = tk.StringVar()
        self._cck8_fit_labels = {
            "Auto": "auto", "3PL (top=100)": "three_pl", "Log-linear": "log_linear",
        }
        self._cck8_fit_label_rev = {v: k for k, v in self._cck8_fit_labels.items()}
        self._cck8_fit_method_var.set(
            self._cck8_fit_label_rev.get(
                self.config.preset_fit_method.value, "Auto"
            )
        )
        ttk.Combobox(
            fit_row,
            textvariable=self._cck8_fit_method_var,
            values=["Auto", "3PL (top=100)", "Log-linear"],
            state="readonly",
            width=14,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            self._cck8_conc_frame,
            text="Auto: 3PL if R²≥0.8, else log-linear interpolation",
            foreground="gray",
        ).pack(anchor="w", padx=8)
        self._on_cck8_ic50_toggle()

        # -- ELISA sub-frame --
        self._elisa_frame = ttk.Frame(exp_frame)
        ttk.Label(
            self._elisa_frame,
            text="ELISA uses a separate workflow:\nSelect standard data → run ELISA button",
            foreground="gray",
        ).pack(anchor="w", padx=8)

        # Wire up preset switching
        self._exp_var.trace_add("write", lambda *_: self._on_preset_change())
        self._on_preset_change()

        # --- Output Options ---
        output_frame = ttk.Labelframe(preset_content, text="Output Options", padding=6)
        output_frame.pack(fill="x", **pad)

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

        # ===================== Export tab =====================
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
                side="right", padx=4
            )
            ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
                side="right", padx=4
            )

    def _make_scrollable(self, parent):
        """Create a scrollable frame inside parent.

        Returns (tab_widget, content_frame) where tab_widget is added to the
        Notebook and content_frame is where child widgets should be packed.
        """
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
    # Helpers (unchanged logic)
    # ------------------------------------------------------------------

    def _guess_control(self) -> str:
        for g in self.group_names:
            if g.lower() in ("control", "ctrl", "con", "ctl", "nc", "vehicle"):
                return g
        if len(self.group_names) >= 2 and self.group_names[0].lower() in (
            "blank", "blk", "bg", "background",
        ):
            return self.group_names[1]
        return self.group_names[0]

    def _guess_blank(self) -> str:
        for g in self.group_names:
            if g.lower() in ("blank", "blk", "bg", "background"):
                return g
        return ""

    def _on_preset_change(self) -> None:
        label = self._exp_var.get()
        val = self._exp_value_map.get(label, "none")
        for frame in (self._wb_frame, self._qpcr_frame, self._cck8_frame, self._elisa_frame):
            frame.pack_forget()
        if val == "wb":
            self._wb_frame.pack(fill="x", pady=4)
        elif val == "qpcr":
            self._qpcr_frame.pack(fill="x", pady=4)
        elif val == "cck8":
            self._cck8_frame.pack(fill="x", pady=4)
        elif val == "elisa":
            self._elisa_frame.pack(fill="x", pady=4)

    def _on_wb_ref_toggle(self) -> None:
        if self._wb_ref_var.get():
            self._wb_ref_hint.pack(anchor="w", padx=16)
        else:
            self._wb_ref_hint.pack_forget()

    def _on_qpcr_ref_toggle(self) -> None:
        if self._qpcr_ref_var.get():
            self._qpcr_ref_hint.pack(anchor="w", padx=16)
            self._qpcr_format_frame.pack_forget()
        else:
            self._qpcr_ref_hint.pack_forget()
            self._qpcr_format_frame.pack(fill="x", pady=2)

    def _on_cck8_ic50_toggle(self) -> None:
        if self._cck8_ic50_var.get():
            self._cck8_conc_frame.pack(fill="x", pady=2)
        else:
            self._cck8_conc_frame.pack_forget()

    def _toggle_export(self) -> None:
        state = "normal" if self._export_var.get() else "disabled"
        self._export_fmt_combo.configure(state=state if state == "disabled" else "readonly")
        self._export_dpi_combo.configure(state=state if state == "disabled" else "readonly")
        self._export_entry.configure(state=state)
        self._browse_btn.configure(state=state)

    def _update_export_ext(self) -> None:
        """Update file extension in path when format changes."""
        ext = self._export_fmt_labels.get(self._export_fmt_var.get(), "png")
        cur = self._export_path_var.get()
        if cur:
            from pathlib import Path
            p = Path(cur)
            self._export_path_var.set(str(p.with_suffix(f".{ext}")))

    def _browse_export(self) -> None:
        ext = self._export_fmt_labels.get(self._export_fmt_var.get(), "png")
        fmt_label = self._export_fmt_var.get()
        path = filedialog.asksaveasfilename(
            defaultextension=f".{ext}",
            filetypes=[(fmt_label, f"*.{ext}"), ("All files", "*.*")],
        )
        if path:
            self._export_path_var.set(path)

    def _on_reset(self) -> None:
        """Reset all settings to defaults and update the UI."""
        from .config import DEFAULT_SETTINGS_PATH
        if DEFAULT_SETTINGS_PATH.exists():
            DEFAULT_SETTINGS_PATH.unlink()
        defaults = PrismConfig()
        self._chart_var.set(defaults.chart_type.value)
        self._eb_var.set(defaults.error_bar.value)
        self._points_var.set(defaults.show_points)
        self._paired_var.set(defaults.paired)
        self._ns_var.set(defaults.show_ns)
        self._annot_var.set(defaults.annotation_format.value)
        self._preset_var.set(defaults.journal_preset.value)
        self._theme_var.set(defaults.base_theme.value)
        self._palette_var.set(defaults.palette_preset.value)
        self._jpal_var.set(defaults.journal_palette.value)
        self._title_var.set(defaults.title)
        self._ylabel_var.set(defaults.y_label)
        self._control_combo.set("All pairwise")
        self._output_stats_var.set(defaults.output_stats)
        self._output_data_var.set(defaults.output_data)

    def _on_ok(self) -> None:
        control_sel = self._control_combo.get()
        control_group = None if control_sel == "All pairwise" else control_sel

        # Resolve experiment preset
        exp_label = self._exp_var.get()
        exp_val = self._exp_value_map.get(exp_label, "none")
        exp_preset = ExperimentPreset(exp_val)

        # Collect preset-specific fields
        preset_control = ""
        preset_has_ref = False
        preset_input_fmt = "delta_ct"
        preset_blank = ""
        preset_fit_ic50 = True
        if exp_val == "wb":
            preset_control = self._wb_control_var.get()
            preset_has_ref = self._wb_ref_var.get()
        elif exp_val == "qpcr":
            preset_control = self._qpcr_control_var.get()
            preset_has_ref = self._qpcr_ref_var.get()
            if not preset_has_ref:
                preset_input_fmt = self._qpcr_format_var.get()
        elif exp_val == "cck8":
            preset_control = self._cck8_control_var.get()
            blank_sel = self._cck8_blank_var.get()
            preset_blank = "" if blank_sel == "(none)" else blank_sel
            preset_fit_ic50 = self._cck8_ic50_var.get()

        preset_concentrations = ""
        preset_dose_axis_scale = DoseAxisScale.AUTO
        preset_fit_method = FitMethod.AUTO
        if exp_val == "cck8" and preset_fit_ic50:
            preset_concentrations = self._cck8_conc_var.get().strip()
            axis_label = self._cck8_axis_scale_var.get()
            axis_val = self._cck8_axis_labels.get(axis_label, "auto")
            preset_dose_axis_scale = DoseAxisScale(axis_val)
            fit_label = self._cck8_fit_method_var.get()
            fit_val = self._cck8_fit_labels.get(fit_label, "auto")
            preset_fit_method = FitMethod(fit_val)

        palette_preset = PalettePreset(self._palette_var.get())
        journal_palette = JournalPalette(self._jpal_var.get())
        from .styles import get_palette
        self.result = PrismConfig(
            chart_type=ChartType(self._chart_var.get()),
            error_bar=ErrorBarType(self._eb_var.get()),
            show_points=self._points_var.get(),
            paired=self._paired_var.get(),
            show_ns=self._ns_var.get(),
            annotation_format=AnnotationFormat(self._annot_var.get()),
            journal_preset=JournalPreset(self._preset_var.get()),
            base_theme=BaseTheme(self._theme_var.get()),
            palette_preset=palette_preset,
            journal_palette=journal_palette,
            palette=get_palette(palette_preset, journal_palette),
            control_group=control_group,
            title=self._title_var.get(),
            y_label=self._ylabel_var.get(),
            output_stats=self._output_stats_var.get(),
            output_data=self._output_data_var.get(),
            export_path=self._export_path_var.get() if self._export_var.get() else "",
            export_format=self._export_fmt_labels.get(self._export_fmt_var.get(), "png"),
            export_dpi=int(self._export_dpi_var.get()),
            experiment_preset=exp_preset,
            preset_control_group=preset_control,
            preset_has_reference=preset_has_ref,
            preset_input_format=preset_input_fmt,
            preset_blank_group=preset_blank,
            preset_fit_ic50=preset_fit_ic50,
            preset_concentrations=preset_concentrations,
            preset_dose_axis_scale=preset_dose_axis_scale,
            preset_fit_method=preset_fit_method,
        )
        # Persist settings for next session
        self.result.save()
        self._root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self._root.destroy()


class TransformOnlyDialog:
    """Simplified dialog for transform-only mode.

    Shows only experiment preset selection and an option to include statistics.
    Returns an updated ``PrismConfig`` or ``None`` on cancel.
    """

    def __init__(
        self,
        group_names: list[str],
        group_sizes: dict[str, int],
        base_config: PrismConfig | None = None,
    ) -> None:
        self.group_names = group_names
        self.group_sizes = group_sizes
        saved = PrismConfig.load()
        self.config = base_config if base_config is not None else saved
        self.result: PrismConfig | None = None

    def show(self) -> PrismConfig | None:
        if HAS_TTKB:
            self._root = ttkb.Window(
                title="XSTARS — Transform Only",
                themename="cosmo",
                resizable=(False, False),
                minsize=(460, 400),
            )
        else:
            self._root = tk.Tk()
            self._root.title("XSTARS — Transform Only")
            self._root.resizable(False, False)
            self._root.minsize(420, 380)
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
        summary_frame = ttk.Labelframe(root, text="Detected Data", padding=8)
        summary_frame.pack(fill="x", **pad)
        summary_text = "  |  ".join(
            f"{g}: n={self.group_sizes.get(g, 0)}" for g in self.group_names
        )
        ttk.Label(summary_frame, text=summary_text, wraplength=380).pack(anchor="w")

        # --- Experiment Preset ---
        exp_frame = ttk.Labelframe(root, text="Experiment Type", padding=6)
        exp_frame.pack(fill="x", **pad)

        self._exp_var = tk.StringVar(value=self.config.experiment_preset.value)
        exp_labels = {
            ExperimentPreset.NONE: "None",
            ExperimentPreset.WB: "Western Blot",
            ExperimentPreset.QPCR: "qPCR (ΔΔCt)",
            ExperimentPreset.CCK8: "CCK-8 Viability",
            ExperimentPreset.ELISA: "ELISA",
        }
        exp_combo = ttk.Combobox(
            exp_frame,
            textvariable=self._exp_var,
            values=[exp_labels[ep] for ep in ExperimentPreset],
            state="readonly", width=28,
        )
        exp_combo.pack(fill="x", padx=8, pady=4)
        self._exp_value_map = {exp_labels[ep]: ep.value for ep in ExperimentPreset}
        self._exp_label_map = {ep.value: exp_labels[ep] for ep in ExperimentPreset}
        self._exp_var.set(self._exp_label_map.get(self.config.experiment_preset.value, "None"))

        # -- WB sub-frame --
        self._wb_frame = ttk.Frame(exp_frame)
        ttk.Label(self._wb_frame, text="Control group:").pack(anchor="w", padx=8)
        self._wb_control_var = tk.StringVar(
            value=self.config.preset_control_group or self.group_names[0]
        )
        ttk.Combobox(
            self._wb_frame, textvariable=self._wb_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        self._wb_ref_var = tk.BooleanVar(value=self.config.preset_has_reference)
        ttk.Checkbutton(
            self._wb_frame, text="Has reference protein (e.g. GAPDH)",
            variable=self._wb_ref_var,
        ).pack(anchor="w", padx=8)

        # -- qPCR sub-frame --
        self._qpcr_frame = ttk.Frame(exp_frame)
        ttk.Label(self._qpcr_frame, text="Control group:").pack(anchor="w", padx=8)
        self._qpcr_control_var = tk.StringVar(
            value=self.config.preset_control_group or self.group_names[0]
        )
        ttk.Combobox(
            self._qpcr_frame, textvariable=self._qpcr_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        self._qpcr_ref_var = tk.BooleanVar(value=self.config.preset_has_reference)
        ttk.Checkbutton(
            self._qpcr_frame, text="Has reference gene (e.g. GAPDH)",
            variable=self._qpcr_ref_var,
            command=self._on_qpcr_ref_toggle,
        ).pack(anchor="w", padx=8)
        self._qpcr_format_frame = ttk.Frame(self._qpcr_frame)
        self._qpcr_format_var = tk.StringVar(value=self.config.preset_input_format)
        ttk.Radiobutton(
            self._qpcr_format_frame, text="ΔCt (pre-computed)",
            variable=self._qpcr_format_var, value="delta_ct",
        ).pack(anchor="w", padx=8)
        ttk.Radiobutton(
            self._qpcr_format_frame, text="Raw Ct (target + reference rows)",
            variable=self._qpcr_format_var, value="raw_ct",
        ).pack(anchor="w", padx=8)
        self._on_qpcr_ref_toggle()

        # -- CCK-8 sub-frame --
        self._cck8_frame = ttk.Frame(exp_frame)
        ttk.Label(self._cck8_frame, text="Control group:").pack(anchor="w", padx=8)
        self._cck8_control_var = tk.StringVar(
            value=self.config.preset_control_group or self.group_names[0]
        )
        ttk.Combobox(
            self._cck8_frame, textvariable=self._cck8_control_var,
            values=self.group_names, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)
        ttk.Label(self._cck8_frame, text="Blank group:").pack(anchor="w", padx=8)
        blank_choices = ["(none)"] + list(self.group_names)
        self._cck8_blank_var = tk.StringVar(
            value=self.config.preset_blank_group or "(none)"
        )
        ttk.Combobox(
            self._cck8_frame, textvariable=self._cck8_blank_var,
            values=blank_choices, state="readonly", width=24,
        ).pack(fill="x", padx=8, pady=2)

        # -- ELISA sub-frame --
        self._elisa_frame = ttk.Frame(exp_frame)
        ttk.Label(
            self._elisa_frame,
            text="ELISA uses a separate workflow:\nSelect standard data → run ELISA button",
            foreground="gray",
        ).pack(anchor="w", padx=8)

        # Wire up preset switching
        self._exp_var.trace_add("write", lambda *_: self._on_preset_change())
        self._on_preset_change()

        # --- Options ---
        opts_frame = ttk.Labelframe(root, text="Options", padding=6)
        opts_frame.pack(fill="x", **pad)

        self._stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame, text="Include statistics summary",
            variable=self._stats_var,
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

    def _on_preset_change(self) -> None:
        label = self._exp_var.get()
        val = self._exp_value_map.get(label, "none")
        for frame in (self._wb_frame, self._qpcr_frame, self._cck8_frame, self._elisa_frame):
            frame.pack_forget()
        if val == "wb":
            self._wb_frame.pack(fill="x", pady=4)
        elif val == "qpcr":
            self._qpcr_frame.pack(fill="x", pady=4)
        elif val == "cck8":
            self._cck8_frame.pack(fill="x", pady=4)
        elif val == "elisa":
            self._elisa_frame.pack(fill="x", pady=4)

    def _on_qpcr_ref_toggle(self) -> None:
        if self._qpcr_ref_var.get():
            self._qpcr_format_frame.pack_forget()
        else:
            self._qpcr_format_frame.pack(fill="x", pady=2)

    def _on_ok(self) -> None:
        exp_label = self._exp_var.get()
        exp_val = self._exp_value_map.get(exp_label, "none")
        exp_preset = ExperimentPreset(exp_val)

        preset_control = ""
        preset_has_ref = False
        preset_input_fmt = "delta_ct"
        preset_blank = ""
        if exp_val == "wb":
            preset_control = self._wb_control_var.get()
            preset_has_ref = self._wb_ref_var.get()
        elif exp_val == "qpcr":
            preset_control = self._qpcr_control_var.get()
            preset_has_ref = self._qpcr_ref_var.get()
            if not preset_has_ref:
                preset_input_fmt = self._qpcr_format_var.get()
        elif exp_val == "cck8":
            preset_control = self._cck8_control_var.get()
            blank_sel = self._cck8_blank_var.get()
            preset_blank = "" if blank_sel == "(none)" else blank_sel

        # Build a config that inherits saved theme/chart settings
        saved = PrismConfig.load()
        self.result = PrismConfig(
            # Inherit display settings from saved config
            chart_type=saved.chart_type,
            error_bar=saved.error_bar,
            show_points=saved.show_points,
            paired=saved.paired,
            show_ns=saved.show_ns,
            annotation_format=saved.annotation_format,
            journal_preset=saved.journal_preset,
            base_theme=saved.base_theme,
            palette_preset=saved.palette_preset,
            journal_palette=saved.journal_palette,
            palette=saved.palette,
            y_label=saved.y_label,
            title=saved.title,
            # Preset fields from this dialog
            experiment_preset=exp_preset,
            preset_control_group=preset_control,
            preset_has_reference=preset_has_ref,
            preset_input_format=preset_input_fmt,
            preset_blank_group=preset_blank,
        )
        # Stash stats preference as a transient attribute
        self.result._include_stats = self._stats_var.get()  # type: ignore[attr-defined]
        self._root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self._root.destroy()
