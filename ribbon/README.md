# Ribbon Installation

The XSTARS Ribbon has two hosts:

- **Windows:** a macro-enabled workbook (`.xlsm`) that embeds the Ribbon XML from `customUI14.xml` together with the existing RunPython callbacks from `ribbon_callbacks.bas`.
- **macOS (verified):** an `.xlam` add-in in the Excel startup folder whose custom UI part uses the **2006-format custom UI** (see the macOS section below), while the data workbook imports the unchanged `ribbon_callbacks.bas` plus the xlwings support modules. Do not create or modify a separate Mac `.bas` file.

This repository does **not** ship a prebuilt `.xlsm` or `.xlam`. See the platform sections below for the verified preparation steps.

## Windows

1. **Install Office RibbonX Editor**
   Download it from <https://github.com/fernandreu/office-ribbonx-editor>.

2. **Add the Ribbon XML**
   - Open the macro-enabled workbook in Office RibbonX Editor.
   - Right-click → **Insert Office 2010+ Custom UI Part**.
   - Paste the contents of `customUI14.xml`.
   - Save and close the editor.

3. **Add the existing VBA callbacks**
   - Open the workbook in Excel.
   - Press `Alt+F11` to open the Visual Basic Editor.
   - Choose **File → Import File…** and import `ribbon_callbacks.bas`.
   - Save the workbook as `.xlsm`.

4. **Reopen Excel**
   The **XSTARS** tab should appear in the Ribbon.

## macOS developer mode

Complete the Python and xlwings prerequisites first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
xlwings addin install
xlwings runpython install
```

See the full [macOS developer-mode setup guide](../docs/macos-developer-setup.md) for the supported versions, interpreter configuration, macro policy, macOS Automation permissions, artifact privacy, and troubleshooting.

**Verified on Excel for Mac 16.112.3 / macOS 15.7.4 (Apple Silicon):** this Excel build does **not** render a document-level custom UI embedded in an `.xlsm`, and does **not** render a `customUI14`-format (2009 namespace) ribbon delivered via an add-in. The XSTARS tab must be delivered as an `.xlam` add-in whose custom UI part uses the **2006-format custom UI**:

- part path `customUI/customUI.xml` (converted from this directory's `customUI14.xml`: replace the namespace with `http://schemas.microsoft.com/office/2006/01/customui` and remove every `insertAfterMso` attribute, which the 2006 schema does not support);
- relationship type `http://schemas.microsoft.com/office/2006/relationships/ui/extensibility` targeting `customUI/customUI.xml`.

To install the ribbon and callbacks in Excel for Mac:

1. Build the XSTARS `.xlam` on Windows or another host with the Office RibbonX Editor: start from a macro-enabled workbook, insert the converted 2006-format custom UI part, import the unchanged [`ribbon_callbacks.bas`](ribbon_callbacks.bas), save as **Excel Add-In (`.xlam`)**, and copy it to the Mac. (The repository does not ship a prebuilt `.xlam`; on the verification machine the `.xlam` was assembled by direct OOXML packaging of the converted custom UI part plus a copy of the callback module.)
2. Copy the `.xlam` into the Excel for Mac startup folder:

   ```text
   ~/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel/
   ```

   and confirm it is checked under **Tools → Excel Add-ins**.
3. Open the macro-enabled XSTARS data workbook and import the **unchanged** [`ribbon_callbacks.bas`](ribbon_callbacks.bas) (File → Import File… in the Visual Basic Editor, reached via **Tools → Macro → Visual Basic Editor** or `Fn+Option+F11`). Do not create or modify a separate Mac `.bas` file. Confirm the imported module is named `RibbonCallbacks`.
4. VBA unqualified calls such as `RunPython` do not cross project boundaries, so also import into the same workbook project:
   - `xlwings.bas` from the installed xlwings wheel (`<venv>/lib/python3.x/site-packages/xlwings/xlwings.bas`), matching the installed xlwings version; and
   - the `Dictionary` class module (VBA-Dictionary), extractable from the installed `xlwings.xlam`, which the Mac branch of the xlwings module requires.
5. If the workbook's `xlwings.conf` sheet carries an `Interpreter` entry, point it at the same `.venv/bin/python` interpreter (a workbook sheet overrides the user config).
6. Save as **Excel Macro-Enabled Workbook (`.xlsm`)**, quit Excel fully, and reopen.
7. Confirm that both the **xlwings** and **XSTARS** Ribbon tabs appear (the XSTARS tab comes from the add-in and is available in all workbooks), and configure xlwings to use the same `.venv/bin/python` interpreter in which XSTARS was installed.

Callback resolution note: with a ribbon hosted by the add-in, Excel resolves `RibbonCallbacks.*` in the **active workbook**, so the workbook's imported modules are what execute. Keeping an unchanged copy of `ribbon_callbacks.bas` inside the `.xlam` (as in the verified configuration) is harmless.

No standalone macOS `.app` is provided, and no separate Mac callback module is required. This workflow supports Microsoft Excel for Mac only; WPS for Mac is not supported.

## Troubleshooting

- **xlwings tab missing:** activate `.venv`, rerun `xlwings addin install`, and restart Excel.
- **XSTARS tab missing:** confirm the 2006-format `.xlam` add-in is in the startup folder and checked under **Tools → Excel Add-ins**, and that macros are allowed for the trusted workbook. A workbook-embedded custom UI and a `customUI14`-format add-in ribbon do not render on the tested Excel for Mac build.
- **`RunPython` unavailable:** activate `.venv`, rerun `xlwings runpython install`, and verify that xlwings uses `<repository-path>/.venv/bin/python`.
- **`Sub or Function not defined` / user-defined type errors on Mac:** the active workbook project must embed `xlwings.bas` and the `Dictionary` class module in addition to `RibbonCallbacks`; VBA unqualified calls do not cross project boundaries.
- **Module import error:** confirm that the imported module is the unchanged `ribbon_callbacks.bas` from this repository, then verify the Python environment with `python -c "import xstars; print(xstars.__file__)"`.
- **macOS permission failure:** review **System Settings → Privacy & Security → Automation**, restart Excel, and retry. Detailed steps are in the [macOS setup guide](../docs/macos-developer-setup.md).
