#!/usr/bin/env python
"""Build the XSTARS Windows installer (Excel edition).

Steps:
  1. PyInstaller -> dist/xstars/xstars.exe   (via xstars.spec at repo root)
  2. Build XSTARS.xlam via Excel COM, then inject customUI14.xml
  3. Inno Setup 6 -> installer/excel/output/XSTARS_Setup_vX.Y.Z.exe

Prerequisites:
  - pip install pyinstaller xlwings (repo .venv)
  - Inno Setup 6 installed
  - Real Microsoft Excel for Windows (for .xlam creation via COM).
    NOTE: WPS Office hijacks the Excel.Application COM ProgID; make sure
    the CLSID resolves to real Excel or the VBProject step will fail.
  - "Trust access to the VBA project object model" enabled in Excel
    (File -> Options -> Trust Center -> Macro Settings)

Usage:
  .venv/Scripts/python.exe installer/excel/build_installer.py
      [--skip-pyinstaller] [--skip-xlam] [--skip-package]

macOS is intentionally not handled here: the macOS installer pipeline
lives under installer/mac with its own build/CI flow.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXCEL_DIR = ROOT / "installer" / "excel"
DIST_DIR = ROOT / "dist"
RIBBON_DIR = ROOT / "ribbon"
XLAM_PATH = EXCEL_DIR / "XSTARS.xlam"


# ── Helpers ─────────────────────────────────────────────────────────────

def _extract_version() -> str:
    """Read __version__ from xstars/__init__.py without importing the package."""
    init_text = (ROOT / "xstars" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\'](.+?)["\']', init_text)
    if not m:
        raise RuntimeError("Cannot find __version__ in xstars/__init__.py")
    return m.group(1)


# ── Step 1: PyInstaller ─────────────────────────────────────────────────

def build_pyinstaller() -> None:
    print("=" * 60)
    print("Step 1: Building frozen binary with PyInstaller")
    print("=" * 60)

    spec_file = ROOT / "xstars.spec"
    if not spec_file.exists():
        print(f"ERROR: {spec_file} not found", file=sys.stderr)
        sys.exit(1)

    # Clean previous build
    for d in [ROOT / "build", DIST_DIR]:
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError as exc:
                print(
                    f"ERROR: Cannot remove {d} ({exc}) — close any process using it.",
                    file=sys.stderr,
                )
                sys.exit(1)

    subprocess.check_call(
        [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean"],
        cwd=str(ROOT),
    )

    artifact = DIST_DIR / "xstars" / "xstars.exe"
    if not artifact.exists():
        print(f"ERROR: Expected {artifact} but it was not created.", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {artifact}")


# ── Step 2: Build .xlam ─────────────────────────────────────────────────

def build_xlam_win() -> None:
    """Create XSTARS.xlam on Windows via Excel COM.

    The generated .xlam is a build artifact (gitignored): it embeds the
    current ribbon_callbacks_installed.bas and is re-created on every
    installer build so it always matches the released ribbon code.
    """
    print()
    print("=" * 60)
    print("Step 2: Building XSTARS.xlam via Excel COM")
    print("=" * 60)

    import xlwings as xw

    if XLAM_PATH.exists():
        try:
            XLAM_PATH.unlink()
        except PermissionError:
            print(
                f"ERROR: Cannot overwrite {XLAM_PATH} — close it in Excel first.",
                file=sys.stderr,
            )
            sys.exit(1)

    app = xw.App(visible=False)
    app.display_alerts = False
    try:
        wb = app.books.add()
        proj = wb.api.VBProject

        # Import RibbonCallbacks module (uses Shell, no xlwings.bas dependency)
        callbacks_path = RIBBON_DIR / "ribbon_callbacks_installed.bas"
        for comp in list(proj.VBComponents):
            if comp.Name == "RibbonCallbacks":
                proj.VBComponents.Remove(comp)
                break
        proj.VBComponents.Import(str(callbacks_path))

        # Save as .xlam (FileFormat 55 = xlOpenXMLAddIn)
        wb.api.SaveAs(str(XLAM_PATH), 55)
        wb.close()
    except Exception as exc:
        print(f"ERROR creating xlam: {exc}", file=sys.stderr)
        print(
            "\nEnsure 'Trust access to the VBA project object model' is enabled:\n"
            "Excel -> File -> Options -> Trust Center -> Macro Settings\n"
            "Also make sure Excel.Application COM resolves to real Microsoft\n"
            "Excel, not WPS Office's et.exe (WPS hijacks the ProgID and has no\n"
            "VBProject support).",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        app.quit()

    _inject_ribbon_xml(XLAM_PATH)
    print(f"OK: {XLAM_PATH}")


def _inject_ribbon_xml(xlam_path: Path) -> None:
    """Inject customUI/customUI14.xml into the .xlam Open XML package."""
    ribbon_xml = (RIBBON_DIR / "customUI14.xml").read_text(encoding="utf-8")

    tmp = xlam_path.with_suffix(".tmp")

    with zipfile.ZipFile(xlam_path, "r") as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == "[Content_Types].xml":
                ct = data.decode("utf-8")
                if "customUI14" not in ct:
                    ct = ct.replace(
                        "</Types>",
                        '  <Override PartName="/customUI/customUI14.xml" '
                        'ContentType="application/xml" />\n</Types>',
                    )
                data = ct.encode("utf-8")

            zout.writestr(item, data)

        zout.writestr("customUI/customUI14.xml", ribbon_xml)

    _add_ribbon_relationship(tmp)

    xlam_path.unlink()
    tmp.rename(xlam_path)


def _add_ribbon_relationship(xlam_path: Path) -> None:
    """Add the customUI relationship to _rels/.rels."""
    rel_id = "rCustomUI"
    rel_type = "http://schemas.microsoft.com/office/2007/relationships/ui/extensibility"
    rel_target = "customUI/customUI14.xml"

    tmp = xlam_path.with_suffix(".tmp2")

    with zipfile.ZipFile(xlam_path, "r") as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == "_rels/.rels":
                rels = data.decode("utf-8")
                if "customUI" not in rels:
                    new_rel = (
                        f'  <Relationship Id="{rel_id}" '
                        f'Type="{rel_type}" '
                        f'Target="/{rel_target}" />\n'
                    )
                    rels = rels.replace("</Relationships>", new_rel + "</Relationships>")
                data = rels.encode("utf-8")

            zout.writestr(item, data)

    xlam_path.unlink()
    tmp.rename(xlam_path)


# ── Step 3: Inno Setup ──────────────────────────────────────────────────

def build_inno() -> None:
    print()
    print("=" * 60)
    print("Step 3: Building installer with Inno Setup")
    print("=" * 60)

    iss_file = EXCEL_DIR / "XSTARS.iss"

    iscc_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    iscc = None
    for p in iscc_paths:
        if Path(p).exists():
            iscc = p
            break

    if iscc is None:
        env_iscc = os.environ.get("ISCC")
        if env_iscc and Path(env_iscc).exists():
            iscc = env_iscc
        else:
            print(
                "ERROR: Inno Setup (ISCC.exe) not found.\n"
                "Download from https://jrsoftware.org/isinfo.php\n"
                "Or set ISCC environment variable to the ISCC.exe path.",
                file=sys.stderr,
            )
            sys.exit(1)

    output_dir = EXCEL_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    subprocess.check_call([iscc, str(iss_file)], cwd=str(ROOT))

    version = _extract_version()
    expected = output_dir / f"XSTARS_Setup_v{version}.exe"
    if expected.exists():
        print(f"\nOK: {expected}")
        print(f"    Size: {expected.stat().st_size / 1024 / 1024:.1f} MB")
        return

    print("WARNING: Expected", expected, "not found in", output_dir)


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the XSTARS Excel installer (Windows)")
    parser.add_argument("--skip-pyinstaller", action="store_true",
                        help="Skip PyInstaller step (use existing dist/)")
    parser.add_argument("--skip-xlam", action="store_true",
                        help="Skip .xlam step (use existing installer/excel/XSTARS.xlam)")
    parser.add_argument("--skip-package", action="store_true",
                        help="Skip Inno Setup step")
    args = parser.parse_args()

    if not sys.platform.startswith("win"):
        print("ERROR: This script only runs on Windows.", file=sys.stderr)
        sys.exit(1)

    os.chdir(ROOT)

    if not args.skip_pyinstaller:
        build_pyinstaller()
    else:
        print("Skipping PyInstaller (--skip-pyinstaller)")

    if not args.skip_xlam:
        build_xlam_win()
    else:
        print("Skipping xlam step (--skip-xlam)")

    if not args.skip_package:
        build_inno()
    else:
        print("Skipping package step (--skip-package)")

    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
