# macOS Developer-Mode Setup

XSTARS supports Microsoft Excel for Mac through **developer mode**: Excel uses the existing xlwings `RunPython` callbacks, and a local Python environment runs XSTARS. This is not a standalone macOS application.

This guide does not provide a `.app`, DMG, signing, or notarization workflow. WPS for Mac is not supported.

## Support matrix

| Component | Supported target | Verification status |
| --- | --- | --- |
| macOS | 10.14 or later | Based on the upstream xlwings support statement; not every release has been tested on real hardware in this project |
| Excel | Microsoft Excel for Mac 2016 or later | Based on the upstream xlwings support statement; real-Excel results must be recorded separately |
| Processor | Intel and Apple Silicon | Both are in the declared support target; untested combinations must not be described as verified |
| Python | 3.10 or later | Required by `pyproject.toml` |
| Office host | Microsoft Excel for Mac only | WPS for Mac is explicitly unsupported |

Automated tests do not start Excel, execute VBA, display tkinter windows, or approve macOS Automation prompts. Treat combinations without a recorded real-Excel run as **supported according to upstream xlwings documentation**, not as verified by this project.

## 1. Prerequisites

Install:

1. Microsoft Excel for Mac 2016 or later.
2. Python 3.10 or later. Confirm with:

   ```bash
   python3 --version
   ```

3. Git, to clone the repository.
4. A macro-enabled Excel workbook (`.xlsm`) that contains the XSTARS Ribbon XML. See the [Ribbon installation guide](../ribbon/README.md).

Keep the repository in a stable location. The editable Python installation refers back to this checkout.

## 2. Create the Python environment

Run these commands in Terminal:

```bash
git clone https://github.com/Frankkk1912/xstars.git
cd xstars
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra is defined by this repository and installs the test tools in addition to XSTARS. XSTARS itself depends on xlwings and the scientific Python packages declared in `pyproject.toml`.

Whenever you work from a new Terminal session, reactivate the environment from the repository root:

```bash
source .venv/bin/activate
```

## 3. Install the xlwings Excel bridge

With the same virtual environment active, run:

```bash
xlwings addin install
xlwings runpython install
```

The first command installs the xlwings add-in for Excel. The second installs the macOS script support used by xlwings `RunPython`.

Open Excel and confirm that the **xlwings** tab is available. In the xlwings settings, set the macOS Python interpreter to the virtual environment used above. From the repository root, its path is:

```text
<repository-path>/.venv/bin/python
```

Using another interpreter can produce `ModuleNotFoundError: No module named 'xstars'`, even when the editable install succeeded in `.venv`.

If either xlwings command is unavailable, confirm that `.venv` is active and run:

```bash
python -m pip show xlwings
command -v xlwings
```

## 4. Import the existing XSTARS callbacks

XSTARS reuses [`ribbon/ribbon_callbacks.bas`](../ribbon/ribbon_callbacks.bas) unchanged on Windows and macOS. Do not create a separate Mac callback module and do not edit the existing `.bas` file.

1. Open the macro-enabled XSTARS workbook in Excel for Mac.
2. Open the Visual Basic Editor through **Tools → Macro → Visual Basic Editor**. Depending on the keyboard and Excel version, `Fn+Option+F11` may also open it; the Windows `Alt+F11` shortcut does not generally apply.
3. In the workbook project, choose **File → Import File…** (or use the project context menu) and select `ribbon/ribbon_callbacks.bas` from this repository.
4. Confirm that the imported module is named `RibbonCallbacks`.
5. Save the workbook as **Excel Macro-Enabled Workbook (`.xlsm`)**, close Excel, and reopen the workbook.

The XSTARS Ribbon XML is installed separately from the VBA module. Follow the [Ribbon installation guide](../ribbon/README.md) if the **XSTARS** tab is absent.

## 5. Macro and Automation permissions

### Excel macro settings

Excel must be allowed to run macros in the trusted `.xlsm` workbook. The exact labels differ between Excel releases; check **Excel → Preferences/Settings → Security & Privacy** and use the notification or per-workbook trust flow provided by your Excel version.

Do not disable macro security globally for unrelated workbooks. If an organization manages Excel security policy, ask its administrator to trust this workbook and the xlwings add-in.

### macOS Automation (Apple Events)

On first use, macOS may ask whether Excel, Terminal, or Python can control Microsoft Excel or another required application. Approve only the Automation access shown for this XSTARS/xlwings workflow.

If the prompt was denied or the operation silently fails:

1. Open **System Settings → Privacy & Security → Automation**. On older macOS versions, open **System Preferences → Security & Privacy → Privacy → Automation**.
2. Locate Microsoft Excel, Terminal, or the Python launcher shown by macOS.
3. Enable the requested Microsoft Excel automation permission.
4. Quit and reopen Excel, then retry once.

If no entry exists, trigger one XSTARS or xlwings `RunPython` action so macOS can present the prompt, then check the Automation panel again.

## 6. First run

1. Confirm that xlwings is configured to use `<repository-path>/.venv/bin/python`.
2. Open the trusted `.xlsm` workbook and enable its macros.
3. Confirm that both the **xlwings** and **XSTARS** Ribbon tabs appear.
4. Select a small wide-format data range, including headers.
5. Click **XSTARS → Run** or **Quick Run**.
6. Confirm that a Matplotlib picture is inserted into the active worksheet and that Excel reports success in its status bar.

Standard-curve and ELISA workflows can request a second sample range. On macOS, enter an A1 address from the **active worksheet**, for example `A1:C6`. Cross-sheet addresses and named ranges are not supported by this input dialog; cancel returns to Excel without selecting sample data.

## 7. macOS Export behavior and limits

macOS Export does not capture pixels from Excel. For each successfully generated XSTARS picture, XSTARS saves local rebuild information and later rebuilds the Matplotlib figure before exporting it.

Supported in this MVP:

- charts generated by the current XSTARS version with a valid registered artifact;
- current XSTARS renderers, including standard-curve and ELISA fit-curve output;
- the formats and DPI choices offered by the XSTARS Export dialog.

Not supported:

- arbitrary Excel Shapes, user-created Excel charts, ranges, or clipboard screenshots;
- legacy XSTARS pictures that were never registered in `~/.xstars/artifacts/`;
- pictures whose artifact is missing, corrupt, incompatible, or belongs to a different workbook/sheet/picture identity;
- WPS for Mac.

For an unsupported or legacy picture, regenerate the chart with the current XSTARS version and export the newly inserted picture. XSTARS fails closed: it does not fall back to an Excel screenshot.

Moving or renaming a workbook, worksheet, or registered picture can change its artifact identity. The MVP does not automatically rebind artifacts after **Save As** or a move; regenerate the chart instead.

## 8. Artifact storage, privacy, and cleanup

XSTARS stores rebuild data under:

```text
~/.xstars/artifacts/
```

The directory contains a manifest and versioned JSON payloads (the current implementation uses schema version 1). A payload can include processed experimental data, workbook/sheet/picture identity, plotting configuration, statistical results, and renderer parameters. SHA-256 checksums detect corruption; they do **not** encrypt the data. XSTARS attempts owner-only directory/file permissions where the filesystem supports them and does not upload these artifacts, but normal device backup or synchronization software may copy them.

The MVP does not automatically expire or delete artifacts. Treat the directory as experimental data and apply your organization's retention and device-access policies.

To remove all rebuild artifacts, first quit Excel/XSTARS and run:

```bash
rm -rf "$HOME/.xstars/artifacts"
```

Do **not** delete `~/.xstars/settings.json` unless you also intend to reset XSTARS settings. Removing artifacts does not alter workbook source data, but macOS Export will require you to regenerate each chart before exporting it again. XSTARS recreates the artifact directory when it next saves a chart successfully.

## 9. Troubleshooting

### `No module named 'xstars'`

- Confirm the xlwings macOS interpreter points to `<repository-path>/.venv/bin/python`.
- Reactivate `.venv` and rerun `python -m pip install -e ".[dev]"`.
- Verify with `python -c "import xstars; print(xstars.__file__)"`.

### The xlwings or XSTARS Ribbon tab is missing

- For the xlwings tab, rerun `xlwings addin install` in the active `.venv`, then restart Excel.
- For the XSTARS tab, confirm the workbook contains `customUI14.xml` and that the existing `ribbon_callbacks.bas` module was imported; see the [Ribbon installation guide](../ribbon/README.md).
- Confirm the workbook was saved as `.xlsm` and macros are enabled for that workbook.

### `RunPython` is unavailable or does nothing

- Rerun `xlwings runpython install` in `.venv`.
- Check Excel's macro policy and the macOS Automation panel.
- Confirm the workbook imports the existing `RibbonCallbacks` module.
- Quit and reopen Excel after changing add-in, macro, or Automation settings.

### Chart creation succeeds but later Export asks you to regenerate

Artifact registration is best-effort: a disk or permission failure must not block statistics, chart insertion, or the success status. Check the local directory:

```bash
ls -ld "$HOME/.xstars" "$HOME/.xstars/artifacts"
```

If appropriate for your account and policy, create and restrict it:

```bash
mkdir -p "$HOME/.xstars/artifacts"
chmod 700 "$HOME/.xstars/artifacts"
```

Then regenerate the chart. Missing, corrupt, unsupported-schema, unsupported-renderer, and identity-mismatch artifacts are rejected rather than exported.

### A legacy or moved-workbook chart cannot be exported

This is expected for charts without a current matching artifact. Reopen the source data, regenerate the chart in its current workbook and worksheet, then export the new picture.

### The sample-range dialog rejects an address

Enter a range on the active worksheet in A1 notation, such as `A1:C6`. Do not include a worksheet name and do not use a named range. Correct the address in the dialog or cancel it.

### A tkinter dialog appears behind Excel or looks different

Window focus and appearance vary by macOS and Tk version. Bring the Python dialog to the foreground manually. Failure to apply the topmost hint is non-fatal; if the entire tkinter path is unavailable, XSTARS retains its existing Excel/status fallback where possible.

## Related documentation

- [Excel for Mac manual acceptance checklist](macos-manual-acceptance.md)
- [Project README](../README.md)
- [中文 README](../README.zh-CN.md)
- [Ribbon installation](../ribbon/README.md)
- [xlwings installation documentation](https://docs.xlwings.org/en/stable/installation.html)
- [xlwings command-line documentation](https://docs.xlwings.org/en/stable/command_line.html)
