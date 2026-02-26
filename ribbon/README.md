# Ribbon Installation

To add the XSTARS ribbon tab to `XSTARS.xlsm`:

## Steps

1. **Install Office Custom UI Editor**
   Download from https://github.com/fernandreu/office-ribbonx-editor

2. **Add the Ribbon XML**
   - Open `XSTARS.xlsm` in the Custom UI Editor
   - Right-click → Insert Office 2010+ Custom UI Part
   - Paste the contents of `customUI14.xml`
   - Save and close

3. **Add VBA Callbacks**
   - Open `XSTARS.xlsm` in Excel
   - Press Alt+F11 to open VBA Editor
   - Import `ribbon_callbacks.bas` (File → Import File)
   - Save the workbook

4. **Reopen** — the "XSTARS" tab should appear in the ribbon.
