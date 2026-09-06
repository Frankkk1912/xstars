# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the standalone XSTARS WPS service, worker, and helper.

Packages into an onedir distribution containing:
1. xstars-wps.exe (console=False, windowed):
   - xstars-wps.exe serve --port 3892 (runs background loopback broker)
   - xstars-wps.exe worker --request <req.json> --result <res.json> (runs worker with Tkinter)
2. xstars-wps-helper.exe (console=True, interactive):
   - xstars-wps-helper.exe bootstrap --config ... --template ... --out ...
   - xstars-wps-helper.exe backup --jsaddons-dir ...
   - xstars-wps-helper.exe sync-config --config ...
   - xstars-wps-helper.exe install-page --dir ... --port 3890
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Resolve repository root
SPEC_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_DIR.parent.parent.resolve()

datas = []
datas += collect_data_files("ttkbootstrap")
try:
    datas += collect_data_files("statannotations")
except Exception:
    pass

hiddenimports = []
hiddenimports += collect_submodules("xstars")
hiddenimports += collect_submodules("ttkbootstrap")
hiddenimports += [
    "scipy.special",
    "scipy.spatial.transform._rotation_groups",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
    "PIL",
    "PIL.Image",
    "PIL.ImageGrab",
]

block_cipher = None

a_service = Analysis(
    [str(REPO_ROOT / "xstars" / "cli.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["xlwings"],  # WPS loopback broker does not need Excel xlwings COM bridge
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_service = PYZ(a_service.pure, a_service.zipped_data, cipher=block_cipher)

exe_service = EXE(
    pyz_service,
    a_service.scripts,
    [],
    exclude_binaries=True,
    name="xstars-wps",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed executable: no CMD popup for background broker or Tkinter worker
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

a_helper = Analysis(
    [str(SPEC_DIR / "wps_helper.py")],
    pathex=[str(SPEC_DIR), str(REPO_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["xlwings"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_helper = PYZ(a_helper.pure, a_helper.zipped_data, cipher=block_cipher)

exe_helper = EXE(
    pyz_helper,
    a_helper.scripts,
    [],
    exclude_binaries=True,
    name="xstars-wps-helper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Interactive console for install-page and CLI commands
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_service,
    a_service.binaries,
    a_service.zipfiles,
    a_service.datas,
    exe_helper,
    a_helper.binaries,
    a_helper.zipfiles,
    a_helper.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="xstars-wps",
)
