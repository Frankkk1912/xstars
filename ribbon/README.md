# Ribbon Installation

The XSTARS Ribbon requires two workbook components:

1. the Ribbon XML from `customUI14.xml`; and
2. the existing RunPython callbacks from `ribbon_callbacks.bas`.

Use a macro-enabled workbook (`.xlsm`). The macOS developer mode reuses the same `ribbon_callbacks.bas`; do not create or modify a separate Mac `.bas` file.

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

To install the callbacks in Excel for Mac:

1. Open the macro-enabled XSTARS workbook in Excel for Mac. Its Ribbon XML must already contain the contents of `customUI14.xml`; use Office RibbonX Editor on a supported host if the XML still needs to be embedded.
2. Open **Tools → Macro → Visual Basic Editor**. Depending on the keyboard and Excel version, `Fn+Option+F11` may also work; `Alt+F11` is the Windows shortcut.
3. In the workbook project, choose **File → Import File…** (or use the project context menu).
4. Select the existing [`ribbon_callbacks.bas`](ribbon_callbacks.bas) from this directory. Confirm that the imported module is named `RibbonCallbacks`.
5. Save as **Excel Macro-Enabled Workbook (`.xlsm`)**, close Excel, and reopen the workbook.
6. Confirm that both the **xlwings** and **XSTARS** Ribbon tabs appear. Configure xlwings to use the same `.venv/bin/python` interpreter in which XSTARS was installed.

No standalone macOS `.app` is provided, and no separate Mac callback module is required. This workflow supports Microsoft Excel for Mac only; WPS for Mac is not supported.

## Troubleshooting

- **xlwings tab missing:** activate `.venv`, rerun `xlwings addin install`, and restart Excel.
- **XSTARS tab missing:** confirm the workbook contains `customUI14.xml`, was saved as `.xlsm`, and macros are allowed for the trusted workbook.
- **`RunPython` unavailable:** activate `.venv`, rerun `xlwings runpython install`, and verify that xlwings uses `<repository-path>/.venv/bin/python`.
- **Module import error:** confirm that the imported module is the unchanged `ribbon_callbacks.bas` from this repository, then verify the Python environment with `python -c "import xstars; print(xstars.__file__)"`.
- **macOS permission failure:** review **System Settings → Privacy & Security → Automation**, restart Excel, and retry. Detailed steps are in the [macOS setup guide](../docs/macos-developer-setup.md).
