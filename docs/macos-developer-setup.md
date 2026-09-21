# macOS Developer-Mode Setup

Apple Silicon end users should install the standalone, per-user `.pkg` described in the [macOS Installer Guide](macos-installer.md); it includes Python and does not require the source setup below.

This document is only for source development and debugging. In **developer mode**, Excel uses the existing xlwings `RunPython` callbacks and a local Python environment runs XSTARS. Keep this workflow as a fallback when inspecting or changing the source.

The developer workflow does not produce a `.app`, DMG, signed package, or notarized package. WPS for Mac is not supported.

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
4. A macro-enabled Excel workbook (`.xlsm`) holding your data, plus an **XSTARS ribbon add-in (`.xlam`)** that provides the XSTARS tab. See the [Ribbon installation guide](../ribbon/README.md).

> **macOS Ribbon reality check (verified on Excel for Mac 16.112.3 / macOS 15.7.4, Apple Silicon):** Excel for Mac in this tested build does **not** render a document-level custom UI embedded in an `.xlsm`, and it also does **not** render a `customUI14`-format (2009 namespace) ribbon delivered via an add-in. The only configuration observed to work is an `.xlam` add-in in the Excel startup folder whose custom UI part uses the **2006-format custom UI** (`customUI/customUI.xml`, namespace `http://schemas.microsoft.com/office/2006/01/customui`, relationship type `http://schemas.microsoft.com/office/2006/relationships/ui/extensibility`, and no `insertAfterMso` attribute, which the 2006 schema does not support). Package the ribbon accordingly on Windows or another host with the Office RibbonX Editor before copying the `.xlam` to the Mac; the VBA callbacks themselves are imported unchanged on the Mac as described below.

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

## 4. Import the existing XSTARS callbacks and xlwings support modules

XSTARS reuses [`ribbon/ribbon_callbacks.bas`](../ribbon/ribbon_callbacks.bas) unchanged on Windows and macOS. Do not create a separate Mac callback module and do not edit the existing `.bas` file.

VBA unqualified calls such as `RunPython` do **not** cross project boundaries, so the workbook must also embed the xlwings support modules that the xlwings add-in alone does not provide to other projects. On macOS the xlwings module additionally requires the `Dictionary` class (VBA-Dictionary), which ships inside the installed `xlwings.xlam` add-in.

1. Open the macro-enabled XSTARS workbook in Excel for Mac.
2. Open the Visual Basic Editor through **Tools → Macro → Visual Basic Editor**. Depending on the keyboard and Excel version, `Fn+Option+F11` may also open it; the Windows `Alt+F11` shortcut does not generally apply.
3. In the workbook project, choose **File → Import File…** (or use the project context menu) and select `ribbon/ribbon_callbacks.bas` from this repository.
4. Confirm that the imported module is named `RibbonCallbacks`.
5. Import `xlwings.bas` from the installed xlwings wheel (for example `<venv>/lib/python3.x/site-packages/xlwings/xlwings.bas`), so the imported module is named `xlwings` and its version matches the installed xlwings version.
6. Import the `Dictionary` class module (extract `Dictionary.cls` from the VBA project of the installed `xlwings.xlam`, or obtain VBA-Dictionary from its upstream source). Confirm the class module is named `Dictionary`.
7. If the workbook's `xlwings.conf` sheet contains an `Interpreter` entry, point it at the virtual-environment interpreter from §2 (a workbook sheet overrides the user config).
8. Save the workbook as **Excel Macro-Enabled Workbook (`.xlsm`)**, close Excel, and reopen the workbook.

The XSTARS Ribbon tab is provided by the `.xlam` add-in described in the [Ribbon installation guide](../ribbon/README.md), not by this workbook. Callbacks raised by the add-in's ribbon resolve in the active workbook, which is why `RibbonCallbacks`, `xlwings`, and `Dictionary` must live in the workbook project.

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
- pictures in an unsaved workbook (save the workbook, then regenerate them);
- pictures whose artifact is missing, corrupt, incompatible, or belongs to a different workbook/sheet/picture identity;
- WPS for Mac.

For an unsupported or legacy picture, regenerate the chart with the current XSTARS version and export the newly inserted picture. XSTARS fails closed: it does not fall back to an Excel screenshot.

Moving or renaming a workbook, worksheet, or registered picture can change its artifact identity. The MVP does not automatically rebind artifacts after **Save As** or a move; regenerate the chart instead.

## 8. Artifact storage, privacy, and cleanup

Chart generation on **both Windows and macOS** stores rebuild data under:

```text
~/.xstars/artifacts/
```

The directory contains versioned JSON payloads (the current implementation uses schema version 1) plus an optional `manifest.json`. Payloads are the authoritative data: loading validates a payload's checksum and embedded identity directly, so a missing, stale, or damaged manifest never makes a valid payload unloadable — the manifest is best-effort diagnostic metadata only. A payload can include processed experimental data, workbook/sheet/picture identity, plotting configuration, statistical results, and renderer parameters. SHA-256 checksums detect corruption; they do **not** encrypt the data. XSTARS attempts owner-only directory/file permissions where the filesystem supports them and does not upload these artifacts, but normal device backup or synchronization software may copy them.

The MVP does not automatically expire or delete artifacts. Treat the directory as experimental data and apply your organization's retention and device-access policies.

To remove all rebuild artifacts, first quit Excel/XSTARS and run:

```bash
rm -rf "$HOME/.xstars/artifacts"
```

Do **not** delete `~/.xstars/settings.json` unless you also intend to reset XSTARS settings. Removing artifacts does not alter workbook source data, but macOS Export will require you to regenerate each chart before exporting it again. XSTARS recreates the artifact directory when it next saves a chart successfully.

**Save As caution:** artifact identity is derived from the workbook path, sheet name, and picture name. If you save the workbook to a different path, generate charts there, and later save it back to a path where charts were generated before (A → B → A), the current pictures can pair with payloads that were produced by an earlier generation. XSTARS does not rebind or fingerprint workbooks. After such a Save As round trip, regenerate the charts you intend to export before using macOS Export.

## 9. Troubleshooting

### `No module named 'xstars'`

- Confirm the xlwings macOS interpreter points to `<repository-path>/.venv/bin/python`.
- Reactivate `.venv` and rerun `python -m pip install -e ".[dev]"`.
- Verify with `python -c "import xstars; print(xstars.__file__)"`.

### The xlwings or XSTARS Ribbon tab is missing

- For the xlwings tab, rerun `xlwings addin install` in the active `.venv`, then restart Excel.
- For the XSTARS tab, confirm the `.xlam` add-in that carries the XSTARS ribbon (2006-format custom UI; see the [Ribbon installation guide](../ribbon/README.md)) is present in the Excel startup folder and listed/checked under **Tools → Excel Add-ins**, then restart Excel. Note that on the tested Excel for Mac build a workbook-embedded custom UI and a `customUI14`-format add-in ribbon do not render; only the 2006-format add-in was observed to work.
- Confirm the workbook was saved as `.xlsm` and macros are enabled for that workbook (when macros are disabled, Excel for Mac hides the custom tab instead of merely failing its buttons).

### Excel reports a missing add-in at startup

If a previously installed XSTARS add-in was deleted while Excel still auto-loads it, Excel shows a "cannot find add-in" prompt on every start. Remove the stale entry: with Excel closed, delete the corresponding `OPENn` record from the Office registration database (`~/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB/*.reg`, table `HKEY_CURRENT_USER_values`, keys `OPEN`, `OPEN1`, … under the Excel options node) or re-check/uncheck the add-in in **Tools → Excel Add-ins**. Back up the database file before editing it.

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

### An unsaved, legacy, or moved-workbook chart cannot be exported

Unsaved workbooks have no stable cross-RunPython identity, so artifact registration is skipped without blocking chart generation. Save the workbook and regenerate the chart. For legacy, moved, or Save-As workbooks, reopen the source data and regenerate the chart in its current workbook and worksheet before exporting.

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
