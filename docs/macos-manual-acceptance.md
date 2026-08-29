# Excel for Mac Manual Acceptance Checklist

This checklist is the real-Excel acceptance gate for XSTARS macOS developer
mode. Automated CI does not start Excel, execute VBA, display tkinter dialogs,
or exercise macOS Automation permission prompts.

**Owner:** the user performing the test on a real Excel for Mac installation.
Do not mark the Draft PR ready until every blocker item passes or the user has
explicitly approved a documented exception. Record the completed checklist and
evidence in the Draft PR description or a PR comment.

Complete the [macOS developer-mode setup](macos-developer-setup.md) before
starting this checklist. That guide also explains supported versions, artifact
privacy, cleanup, and troubleshooting.

## Result legend

Use one result for every row:

- `PASS` — observed behavior matches the expected result.
- `FAIL` — expected behavior was not observed; attach evidence and reproduction
  steps.
- `NOT RUN` — not executed; explain why. A blocker marked `NOT RUN` does not
  satisfy this acceptance gate.
- `N/A` — use only for an explicitly non-blocking item, with a reason.

Items labeled **Blocker** must pass unless the user explicitly records and
accepts an exception. Non-blocking differences must be entered in the residual
risk table and explicitly accepted by the user.

## 1. Test environment record

| Field | Recorded value |
| --- | --- |
| Test date and time (include time zone) | |
| Tester | |
| Mac model | |
| Processor / chip (`Intel` or Apple Silicon model) | |
| macOS version and build | |
| Microsoft Excel for Mac version and build | |
| Python version (`python --version`) | |
| xlwings version (`python -m pip show xlwings`) | |
| XSTARS commit SHA | |
| Workbook name and location | |
| Virtual-environment Python path configured in xlwings | |
| Evidence folder or PR link | |

If both Intel and Apple Silicon devices are available, create a separate copy
of this record and checklist for each device. Untested combinations must be
reported as supported according to upstream xlwings documentation, not as
verified by this project.

## 2. Installation and permission checks

| ID | Gate | Check | Expected result | Result | Evidence / notes |
| --- | --- | --- | --- | --- | --- |
| I-01 | **Blocker** | Create/activate the Python 3.10+ environment and run `python -m pip install -e ".[dev]"`. | Installation completes and `python -c "import xstars"` succeeds. | | |
| I-02 | **Blocker** | Run `xlwings addin install`. | Command succeeds and the xlwings add-in is available after Excel restarts. | | |
| I-03 | **Blocker** | Run `xlwings runpython install`. | Command succeeds and xlwings `RunPython` support is installed. | | |
| I-04 | **Blocker** | Import the existing `ribbon/ribbon_callbacks.bas` in the Excel VBA editor; do not edit it. | Module `RibbonCallbacks` is present and the macro-enabled workbook saves/reopens. | | |
| I-05 | **Blocker** | Open the workbook with macros enabled and inspect the Ribbon. | Both xlwings and XSTARS Ribbon tabs are available. | | |
| I-06 | **Blocker** | Trigger the first XSTARS/xlwings action and handle any macOS Automation prompt. | The relevant Python/Terminal/Excel process can control Microsoft Excel after permission is granted. Record whether a prompt appeared. | | |
| I-07 | Non-blocking | Reopen Excel after bridge and permission setup. | No repeated unexpected permission prompt; otherwise document the behavior. | | |

## 3. Chart-generation workflows

Use a small, non-sensitive test workbook. Preserve the input ranges and
screenshots needed to reproduce each result.

| ID | Gate | Check | Expected result | Result | Evidence / notes |
| --- | --- | --- | --- | --- | --- |
| G-01 | **Blocker** | Select valid wide-format data and run **Run**. | Statistics/output complete, a Matplotlib picture is inserted, and the Excel status bar reports success. | | |
| G-02 | **Blocker** | Run **Quick Run** on valid data. | A picture is inserted and the status bar reports success. | | |
| G-03 | **Blocker** | Run at least one named preset and record its name. | The preset chart is inserted with its expected labels/style and a success status. | | |
| G-04 | **Blocker** | Run a WB or qPCR labeled workflow if available in the workbook. | The labeled chart is inserted and remains associated with a rebuild artifact. | | |
| G-05 | **Blocker** | Run the standard-curve workflow and enter a valid active-sheet A1 sample range. | Range data is accepted; fit/output and fit-curve picture are produced without a COM `InputBox` error. | | |
| G-06 | **Blocker** | Run the ELISA workflow and enter a valid active-sheet A1 sample range. | Range data is accepted; ELISA output and fit-curve picture are produced. | | |
| G-07 | **Blocker** | After generating ordinary and specialized charts, inspect `~/.xstars/artifacts/`. | A manifest and matching JSON artifact payloads exist for the newly inserted pictures. | | |
| G-08 | **Blocker** | Change a visible XSTARS setting, save it, then reopen the settings/workflow (restart Excel if required). | The setting remains persisted and normal chart generation still succeeds. | | |
| G-09 | Non-blocking | Observe picture appearance and fonts in Excel, including dark-mode use if applicable. | Chart remains readable; record cosmetic differences that do not affect data or export correctness. | | |

## 4. Sample-range and error-dialog behavior

Use only active-sheet A1 addresses such as `A1:C6`. Cross-sheet addresses and
named ranges are outside the supported MVP input contract.

| ID | Gate | Check | Expected result | Result | Evidence / notes |
| --- | --- | --- | --- | --- | --- |
| R-01 | **Blocker** | Enter a valid active-sheet A1 address for the second sample range. | The intended data is read and the workflow completes. | | |
| R-02 | **Blocker** | Cancel the range dialog. | The dialog closes, no traceback appears, and XSTARS returns control to Excel without using a sample range. | | |
| R-03 | **Blocker** | Enter an invalid address, then correct it with a valid address. | A clear error is shown and the dialog permits retry; no COM call or traceback is exposed. | | |
| R-04 | **Blocker** | Enter a cross-sheet or named-range value. | Input is rejected with an understandable message; it is not silently resolved. | | |
| R-05 | **Blocker** | Observe tkinter range/error-dialog focus with Excel in front. | The dialog is discoverable and usable. If the topmost hint is unavailable, the message still appears and the dialog can be brought forward. | | |
| R-06 | **Blocker** | Trigger a recoverable XSTARS error that uses the error dialog. | The message identifies the failure without a raw traceback; closing it reliably returns control to Excel. | | |

## 5. Artifact-backed export checks

Generate a new ordinary XSTARS chart and a specialized standard-curve or ELISA
fit chart in the current workbook before testing. Export must rebuild the
Matplotlib Figure; it must not use an Excel screenshot or arbitrary Shape.

### 5.1 Successful export matrix

Run every row. Use at least two DPI values; this template uses 300 and 600 DPI.
Record output size and attach or link the resulting files.

| ID | Gate | Source chart | Format | DPI | Expected result | Result | File / evidence |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| E-01 | **Blocker** | Ordinary XSTARS chart | PNG | 300 | Non-empty file opens and reflects the generated chart. | | |
| E-02 | **Blocker** | Ordinary XSTARS chart | PNG | 600 | Non-empty file opens; pixel dimensions exceed or match the 300-DPI export as expected. | | |
| E-03 | **Blocker** | Ordinary XSTARS chart | TIFF | 300 | Non-empty TIFF opens and reflects the generated chart. | | |
| E-04 | **Blocker** | Ordinary XSTARS chart | SVG | 300 | Non-empty SVG opens and contains vector figure content. | | |
| E-05 | **Blocker** | Ordinary XSTARS chart | PDF | 600 | Non-empty PDF opens and reflects the generated chart. | | |
| E-06 | **Blocker** | Standard-curve or ELISA fit chart | PNG | 300 | Rebuilt fit chart exports with curve, labels, and parameters intact. | | |
| E-07 | **Blocker** | Standard-curve or ELISA fit chart | PDF | 600 | Rebuilt specialized chart exports as a non-empty PDF. | | |
| E-08 | **Blocker** | Select multiple artifact-backed XSTARS pictures and export. | Every file is created using the expected `_1`, `_2`, ... naming without overwriting another output. | | |

### 5.2 Fail-closed and recovery checks

Before changing artifacts, copy test payloads to a temporary backup. Do not use
production or sensitive experimental data.

| ID | Gate | Check | Expected result | Result | Evidence / notes |
| --- | --- | --- | --- | --- | --- |
| F-01 | **Blocker** | Attempt to export a legacy XSTARS picture created before artifact registration. | Export refuses it and clearly instructs the user to regenerate the chart; no screenshot fallback occurs. | | |
| F-02 | **Blocker** | Temporarily remove the matching payload for a newly generated test picture, then export. | Export reports a missing artifact and instructs regeneration/checking the artifact directory; no output or partial file is left. | | |
| F-03 | **Blocker** | Corrupt a copied test payload, then export the matching picture. | Export reports corruption/incompatibility and does not screenshot Excel or create a partial output. | | |
| F-04 | **Blocker** | Attempt to export an arbitrary Excel Shape or user-created chart. | The Shape is not treated as an artifact-backed XSTARS picture and is not exported. | | |
| F-05 | **Blocker** | Restore the artifact state or regenerate the affected test charts, then retry. | Normal artifact-backed export succeeds again. | | |

## 6. Blocker failure report

Create one entry for every failed or not-run blocker. Attach screenshots and
logs that do not expose sensitive experimental data.

### Failure `<ID>` — `<short title>`

| Field | Record |
| --- | --- |
| Checklist item | |
| Result (`FAIL` or `NOT RUN`) | |
| First observed date/time | |
| Exact reproduction steps | |
| Expected behavior | |
| Actual behavior | |
| Screenshot/file link | |
| Relevant Terminal output or log | |
| macOS/Excel/Python/xlwings versions | |
| Reproducible after Excel restart? | |
| Reproducible after permission review? | |
| Workaround, if any | |
| Blocking impact | |

Duplicate this section as needed. Never replace a failed blocker with a prose
claim that it passed.

## 7. Non-blocking differences and residual risks

| ID | Difference / limitation | User impact | Evidence | Proposed disposition | Explicitly accepted by user (name/date) |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## 8. Acceptance summary and user sign-off

| Summary field | Record |
| --- | --- |
| Total blocker items | |
| Blockers passed | |
| Blockers failed | |
| Blockers not run | |
| Non-blocking differences | |
| Draft PR comment/description link containing this record | |
| Final result (`PASS` / `FAIL` / `INCOMPLETE`) | |
| User name | |
| Sign-off date | |

Acceptance result rules:

- `PASS`: every blocker passes, or each exception has a documented user-approved
  waiver; all non-blocking differences are recorded and explicitly accepted.
- `FAIL`: one or more blockers failed without an approved exception.
- `INCOMPLETE`: one or more blockers were not run, required environment fields
  are missing, or evidence is insufficient.

A `PASS` record satisfies the real-Excel manual gate but does not itself make
the Draft PR ready. The user must still review the final diff, CI results, and
review findings before manually changing PR status.
