"""Static checks on the WPS Inno Setup script."""

from __future__ import annotations

import re
from pathlib import Path

ISS = Path(__file__).resolve().parent.parent / "installer" / "wps" / "XSTARS_WPS.iss"


def _section(text: str, start: str) -> str:
    match = re.search(rf"{re.escape(start)}.*?\nend;", text, re.S)
    assert match, f"{start} not found"
    return match.group(0)


def test_update_stops_the_running_service_before_copying_files():
    """The windowless service locks its files; Restart Manager cannot close it."""
    prepare = _section(ISS.read_text(encoding="utf-8"), "function PrepareToInstall")
    assert "taskkill.exe" in prepare
    assert "/IM xstars-wps.exe" in prepare
    assert "/IM xstars-wps-helper.exe" in prepare
    assert "ewWaitUntilTerminated" in prepare


def test_silent_update_restarts_the_service_after_config_sync():
    post = _section(ISS.read_text(encoding="utf-8"), "procedure CurStepChanged")
    assert "WizardSilent" in post
    assert post.index("sync-config") < post.index("'serve'")
