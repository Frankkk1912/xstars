"""Core modules import without Excel, ttkbootstrap, or tkinter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORE_MODULES = (
    "xstars",
    "xstars.config",
    "xstars.stats_engine",
    "xstars.plot_engine",
    "xstars.styles",
    "xstars.annotations",
    "xstars.data_handler",
    "xstars.tools.standard_curve",
    "xstars.presets",
    "xstars.presets.wb",
    "xstars.presets.qpcr",
    "xstars.presets.cck8",
    "xstars.presets.elisa",
    "xstars.application.analysis",
    "xstars.application.export",
)


def test_pyproject_keeps_excel_ui_out_of_the_core_install():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core, _, extras = text.partition("[project.optional-dependencies]")
    assert 'version = "1.3.1"' in core
    assert 'license = "MIT"' in core
    assert 'name = "Frankkk1912"' in core
    assert "email" not in core
    assert 'readme = "README.pypi.md"' in core
    for package in (
        "pandas",
        "numpy",
        "scipy",
        "scikit-posthocs",
        "matplotlib",
        "seaborn",
        "Pillow",
    ):
        assert package in core
    for package in ("xlwings", "ttkbootstrap"):
        assert package not in core
        assert package in extras
    assert "xstars[excel]" in extras


def test_core_modules_import_without_excel_or_gui():
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys

        BLOCKED = {"xlwings", "ttkbootstrap", "tkinter"}

        class _BlockLoader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ImportError(module.__name__ + " is not installed")

        class _BlockFinder:
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] in BLOCKED:
                    return importlib.machinery.ModuleSpec(fullname, _BlockLoader())
                return None

        sys.meta_path.insert(0, _BlockFinder())
        modules = sys.argv[1:]
        for name in modules:
            __import__(name)
        leaked = sorted(name for name in BLOCKED if name in sys.modules)
        if leaked:
            raise SystemExit("core import loaded " + ", ".join(leaked))
        forbidden = (
            "xstars.main",
            "xstars.ui_dialog",
            "xstars.presets.elisa_dialog",
            "xstars.tools.standard_curve_dialog",
        )
        loaded = sorted(name for name in forbidden if name in sys.modules)
        if loaded:
            raise SystemExit("core import loaded " + ", ".join(loaded))
        """
    )
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    completed = subprocess.run(
        [sys.executable, "-c", script, *CORE_MODULES],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
