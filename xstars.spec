# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the XSTARS standalone Excel distribution.

Build command:
    .venv/Scripts/python.exe -m PyInstaller xstars.spec --clean

Output:
    Windows: dist/xstars/xstars.exe  (one-dir mode)
    macOS:   dist/xstars/xstars       (one-dir Unix binary)
             dist/XSTARS.app          (BUNDLE wrapping the same binary so
                                       Tkinter dialogs render correctly)
"""

import re
import sys
from pathlib import Path

block_cipher = None
IS_MAC = sys.platform == "darwin"

# Read version from xstars/__init__.py without importing the package
# (importing would pull in xlwings/scipy/etc. which slows the build).
_init_text = (Path(SPECPATH) / "xstars" / "__init__.py").read_text(encoding="utf-8")
APP_VERSION = re.search(r'__version__\s*=\s*["\'](.+?)["\']', _init_text).group(1)

# Locate site-packages for data collection
import matplotlib
import ttkbootstrap
import statannotations

mpl_data = Path(matplotlib.get_data_path())
ttkb_root = Path(ttkbootstrap.__file__).parent
sa_root = Path(statannotations.__file__).parent

# xlwings ships separate Windows / macOS backends. The Mac backend pulls in
# AppleScript via py-appscript, which PyInstaller can't always autodetect.
if IS_MAC:
    xlwings_backend = [
        "xlwings",
        "xlwings._xlmac",
        "appscript",
        "aem",
    ]
else:
    xlwings_backend = [
        "xlwings",
        "xlwings._xlwindows",
    ]

a = Analysis(
    ["xstars/cli.py"],
    pathex=[],
    binaries=[],
    datas=[
        # matplotlib needs fonts + style sheets
        (str(mpl_data), "matplotlib/mpl-data"),
        # ttkbootstrap themes
        (str(ttkb_root), "ttkbootstrap"),
        # statannotations package data
        (str(sa_root), "statannotations"),
    ],
    hiddenimports=[
        *xlwings_backend,
        # scipy submodules that PyInstaller misses
        "scipy.special._cdflib",
        "scipy.stats",
        "scipy.optimize",
        "scipy.interpolate",
        # scikit-posthocs
        "scikit_posthocs",
        # seaborn
        "seaborn",
        "seaborn.objects",
        # matplotlib backends
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends.backend_pdf",
        # pandas internals
        "pandas._libs.tslibs.timedeltas",
        "pandas._libs.tslibs.np_datetime",
        "pandas._libs.tslibs.nattype",
        # tkinter
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        # ttkbootstrap
        "ttkbootstrap",
        "ttkbootstrap.constants",
        "ttkbootstrap.scrolled",
        "ttkbootstrap.dialogs",
        # our own package
        "xstars",
        "xstars.main",
        "xstars.cli",
        "xstars.config",
        "xstars.data_handler",
        "xstars.stats_engine",
        "xstars.plot_engine",
        "xstars.annotations",
        "xstars.styles",
        "xstars.ui_dialog",
        "xstars.artifacts",
        "xstars.wps_service",
        "xstars.application",
        "xstars.application.analysis",
        "xstars.application.contracts",
        "xstars.application.export",
        "xstars.application.worker",
        "xstars.presets",
        "xstars.presets.wb",
        "xstars.presets.qpcr",
        "xstars.presets.cck8",
        "xstars.presets.elisa",
        "xstars.presets.elisa_dialog",
        "xstars.tools",
        "xstars.tools.standard_curve",
        "xstars.tools.standard_curve_dialog",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "pytest_cov",
        "_pytest",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="xstars",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is Windows-only in practice; PyInstaller skips it on Mac if not present
    upx=not IS_MAC,
    console=False,  # Win: no console window. Mac: still a Unix binary, no Terminal popup.
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,  # add .ico (Win) / .icns (Mac) path here if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=not IS_MAC,
    upx_exclude=[],
    name="xstars",
)

# On macOS, wrap the one-dir output in a .app bundle. This is required for
# Tkinter to behave correctly (focus, menu bar, GUI event loop) when the
# binary is launched headlessly from Excel via `do shell script`.
if IS_MAC:
    app = BUNDLE(
        coll,
        name="XSTARS.app",
        icon=None,  # add .icns (Win) / .icns (Mac) path here if desired
        bundle_identifier="com.frank-sysu.xstars",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": "XSTARS",
            "CFBundleDisplayName": "XSTARS",
            "CFBundleVersion": APP_VERSION,
            "CFBundleShortVersionString": APP_VERSION,
            # Background app — no Dock icon when launched headlessly from Excel
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            # Allow the binary to read user-selected workbooks (file path comes via argv)
            "NSDesktopFolderUsageDescription": "XSTARS reads Excel files you select.",
            "NSDocumentsFolderUsageDescription": "XSTARS reads Excel files you select.",
            "NSDownloadsFolderUsageDescription": "XSTARS reads Excel files you select.",
        },
    )
