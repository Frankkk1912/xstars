# Canvas ratio and compact comparison annotations

## Approved scope

- Auto (existing/journal sizing), Square (1:1), Landscape (4:3), Portrait (3:4) in Figure settings.
- Explicit ratios override journal sizing, persist in settings and render payloads, and survive PNG/PDF/Excel/WPS export without tight cropping.
- Primary problem: stacked brackets inflate the data Y range and flatten bars. Keep data limits unchanged; reserve a physical annotation band above the data axes instead.
- Pack non-overlapping comparison intervals into shared lanes using rendered label widths and point-based spacing. Preserve all selected comparison labels and statistical results.
- Keep at least 55% of the original usable data-axis height. Give a dense-layout warning above 30% annotation usage; if labels cannot fit within the 45% budget, give an actionable error instead of exporting clipped, overlapping, or misleading annotations. Never silently hide comparisons or stretch the chosen canvas.
- No letter displays, significance matrices, statistical-method changes, installer builds or release/version bump in this implementation.

## Verification

- Configuration default/round-trip/invalid legacy value; Figure UI initialization and collection.
- All explicit ratios, journal precedence, raster/vector export dimensions and metadata re-export.
- Unchanged data Y limits and statistics; disconnected brackets share a lane; label/line bounds remain inside the canvas, including scientific labels and titles.
- Dense annotations warn; impossible layouts fail clearly without leaking figures.
- Existing full test suite and changed-file LSP diagnostics. Save reproducible plot examples for visual inspection.

## Revision 2026-11-07 (review)

Review of the first implementation found: the y-axis line ended below the brackets (looked truncated); 5+ groups with all pairwise comparisons failed with an error at default size; long brackets could sit below shorter ones they spanned. Agreed changes:

- Explicit ratios set the shape of the plot area (data part of the axes), not the whole image. Width stays at the configured width; image height grows for title, labels and brackets.
- Auto keeps the fixed image size (legacy/journal). Brackets are drawn inside the y-axis, which extends to the lowest round tick above them. The image grows only if brackets would leave the data under 20% of the axes.
- No height-budget error: every selected comparison is always drawn. A dense notice appears when brackets take more than half the axes. The too-wide label error remains.
- Rows: shortest comparisons first; a bracket sits above every bracket nested inside it; brackets that share a group may share a row (ends inset 2.5pt).
- `statannotations` is no longer used and was removed from dependencies, PyInstaller specs and the macOS import check.

## Revision 2026-11-07 (2): Fixed / Panel replaces canvas ratios

Plots with different heights are hard to assemble into multi-panel figures, and a journal column width applies to the whole figure, not each plot. The ratio setting (Auto/Square/Landscape/Portrait) is replaced before release by:

- `canvas_mode`: **Fixed** (default; configured or journal size, unchanged from v1.2.2) or **Panel** (height fixed; plot-area width = groups × `bar_width`).
- `bar_width`: Narrow 0.35 in, Medium 0.5 in, Wide 0.7 in per group (Panel only). Charts without groups (standard curve, dose-response) keep the configured width.
- Brackets fit inside the height in both modes by extending the y-axis to a round tick. Height grows only if brackets would leave the data under 20% of the plot (e.g. 8 groups, all 28 pairs).
- Artifacts and settings without the new keys fall back to Fixed / Medium.

## Revision 2026-11-07 (3): wider panels, fit-data y-axis, bracket picker

- Panel widths per group raised to Narrow 0.5 / Medium 0.7 / Wide 0.9 in (Medium 4 groups ≈ square plot area at the default height).
- New `y_axis_scale`: **Fit brackets** (default, as before) or **Fit data**. Fit data picks a round axis top so the tallest value (bars, error bars, points) fills ~80% (≥5% headroom). Brackets start just above the data and may run above the axis line; the canvas grows by the overflow so the plot area never shrinks. No dense warning in this mode.
- New `hidden_comparisons` ([a, b] pairs): brackets not drawn, statistics unchanged. Set via "Choose brackets…" in the main dialog (follows All pairwise / vs control). Kept in render payloads and Excel artifacts; never written to global settings.
