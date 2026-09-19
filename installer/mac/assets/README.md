# Prebuilt Office Artifacts (macOS installer)

This directory holds the **binary Office artifacts** distributed by the macOS
`.pkg` installer. They cannot be generated on macOS (Excel for Mac exposes no
`VBProject` automation), so they are committed here and rebuilt only when the
VBA sources change.

| Artifact | Carries | Consumed by |
| --- | --- | --- |
| `XSTARS.xlam` | XSTARS ribbon tab (2006-format customUI) + `RibbonCallbacks` + the xlwings custom-addin bridge + `Dictionary` | Installed to the Excel startup folder by `postinstall` |
| `XSTARS_mac.xlsm` | Data-workbook template embedding `RibbonCallbacks` + the standard `xlwings.bas` + `Dictionary` | Shipped as the user's starting workbook |

Both artifacts need a self-contained three-module VBA project because
unqualified calls such as `RunPython` do not cross project boundaries
(`docs/macos-developer-setup.md:88-99`). The add-in must use xlwings'
`xlwings_custom_addin.bas` variant: its Mac bridge passes the active workbook
to Python so `Book.caller()` resolves the user's workbook rather than the
add-in itself.

## Hard invariants (CI-asserted once `tests/test_macos_installer.py` lands)

1. **customUI must be 2006-format** (`customUI/customUI.xml`, namespace
   `http://schemas.microsoft.com/office/2006/01/customui`, no
   `insertAfterMso`). The `customUI14` (2009 namespace) variant does **not**
   render on the verified Excel for Mac build (`docs/macos-developer-setup.md:29-37`).
2. **Embedded VBA must equal the repo sources after line-ending
   normalization** (`\r\n` → `\n`, Plan D11). Excel/VBE stores module source
   as CRLF inside `vbaProject.bin` regardless of input, so a raw byte compare
   will always fail by design.
3. **Both embedded xlwings bridges must match the xlwings bundled in the
   runtime** (currently 0.37.0). `XSTARS_mac.xlsm` embeds the wheel's standard
   `xlwings.bas`; `XSTARS.xlam` embeds the wheel's
   `xlwings_custom_addin.bas`. Installer assembly compares the add-in module
   byte-for-source after line-ending normalization and fails closed on the
   standard/wrong variant.
4. **`XSTARS_mac.xlsm`'s `xlwings.conf!Interpreter` must stay empty.** The
   rebuilt add-in has no workbook-level config sheet and therefore uses the
   installer-written user config. A workbook-level interpreter value would
   override `INTERPRETER_MAC` and point users at a non-existent interpreter.
5. **No `absPath` metadata**: Excel 2010+ writes the save directory
   (`x15ac:absPath`) into `xl/workbook.xml` on every save. It leaks the build
   machine's user path and must be stripped after any re-save (see below).

## Rebuild procedures

### `XSTARS.xlam` (Mac host, manual VBE + OOXML injection)

Create a blank workbook and import the unchanged
`ribbon/ribbon_callbacks.bas` plus the pinned wheel's
`xlwings_custom_addin.bas`. Mac VBE may reject `Dictionary.cls`; if so, create
a class module named exactly `Dictionary` and paste the class source after
removing **all** export-only `Attribute ...` lines, including procedure-level
attributes interspersed through the file. Compile the VBA project and save it
as an Excel Add-In.

Use the rebuilt add-in as the OOXML package base, inject the proven 2006-format
`customUI/customUI.xml` and package-root extensibility relationship, and scrub
`x15ac:absPath`. Do not transplant only `vbaProject.bin` into an older add-in:
the workbook sheet/document-module topology may differ.

### `XSTARS_mac.xlsm` (Mac host, manual VBE)

Per `docs/macos-developer-setup.md:88-99`: import the unchanged
`ribbon/ribbon_callbacks.bas` and the `xlwings.bas` from the xlwings wheel;
**Mac VBE rejects `.cls` imports**, so create the `Dictionary` class with
Insert → Class Module and paste the source (extracted from the installed
`xlwings.xlam`). Keep `xlwings.conf!Interpreter` empty.

## Metadata scrub (after any re-save)

Remove every `<mc:AlternateContent>` block containing `x15ac:absPath` from
`xl/workbook.xml` (pure metadata; `vbaProject.bin` is untouched and Excel
regenerates the entry on the next save). The installer assembly step re-scans
for `absPath`/`._*`/`.DS_Store` before packaging as a safety net.

## Rebuild trigger

Any change to `ribbon/*.bas` or `ribbon/customUI14.xml` invalidates both
artifacts (RK-04) — rebuild, re-verify, and re-commit in the same PR.
