# Prebuilt Office Artifacts (macOS installer)

This directory holds the **binary Office artifacts** distributed by the macOS
`.pkg` installer. They cannot be generated on macOS (Excel for Mac exposes no
`VBProject` automation), so they are committed here and rebuilt only when the
VBA sources change.

| Artifact | Carries | Consumed by |
| --- | --- | --- |
| `XSTARS.xlam` | XSTARS ribbon tab (2006-format customUI) + `RibbonCallbacks` module | Installed to the Excel startup folder by `postinstall` |
| `XSTARS_mac.xlsm` | Data-workbook template embedding `RibbonCallbacks` + `xlwings.bas` + `Dictionary` | Shipped as the user's starting workbook |

Why the workbook needs the three modules: ribbon callbacks raised by the
add-in resolve in the **active workbook** (verified, `ribbon/README.md:67`),
and VBA unqualified calls such as `RunPython` do not cross project boundaries
(`docs/macos-developer-setup.md:88-99`).

## Hard invariants (CI-asserted once `tests/test_macos_installer.py` lands)

1. **customUI must be 2006-format** (`customUI/customUI.xml`, namespace
   `http://schemas.microsoft.com/office/2006/01/customui`, no
   `insertAfterMso`). The `customUI14` (2009 namespace) variant does **not**
   render on the verified Excel for Mac build (`docs/macos-developer-setup.md:29-37`).
2. **Embedded VBA must equal the repo sources after line-ending
   normalization** (`\r\n` → `\n`, Plan D11). Excel/VBE stores module source
   as CRLF inside `vbaProject.bin` regardless of input, so a raw byte compare
   will always fail by design.
3. **`xlwings.bas` version must match the xlwings bundled in the runtime**
   (CI-asserted: staging collects the version-encoded AppleScript filename from the xlwings wheel and the
   test suite compares it with the `xlwings.bas` version embedded in the shipped `XSTARS_mac.xlsm`;
   currently 0.37.0).
4. **`xlwings.conf!Interpreter` must stay empty** in both artifacts. A
   workbook-level entry would override the installer-written
   `INTERPRETER_MAC` and point every user at a non-existent interpreter.
5. **No `absPath` metadata**: Excel 2010+ writes the save directory
   (`x15ac:absPath`) into `xl/workbook.xml` on every save. It leaks the build
   machine's user path and must be stripped after any re-save (see below).

## Rebuild procedures

### `XSTARS.xlam` (Windows host)

Either path works; both must end with invariant 5 applied:

- **Automated (used for the current artifact, 2026-09-08)**: real Excel 16 COM
  combined with OOXML injection. On hosts with WPS Office installed, WPS hijacks the Excel
  CLSID under HKCU (`{00024500-...}` → `et.exe`); launch
  `EXCEL.EXE /automation` for Office16 and bind through the ROT instead.
- **Manual**: Office RibbonX Editor — convert `ribbon/customUI14.xml` to the
  2006 format (swap the namespace, drop every `insertAfterMso`), insert as the
  custom UI part, import the unchanged `ribbon/ribbon_callbacks.bas`, save as
  Excel Add-In (`ribbon/README.md:44-51`).

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
