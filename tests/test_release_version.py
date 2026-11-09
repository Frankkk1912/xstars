"""Every place that carries the release version agrees with xstars.__version__."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import xstars

ROOT = Path(__file__).resolve().parent.parent
VERSION = xstars.__version__


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_distribution_sources_carry_the_package_version():
    assert re.search(r'^version = "(.+)"', _read("pyproject.toml"), re.MULTILINE)[1] == VERSION
    assert json.loads(_read("wps-addon/package.json"))["version"] == VERSION
    assert f'const ADDIN_VERSION = "{VERSION}";' in _read("wps-addon/main.js")
    excel_iss = _read("installer/excel/XSTARS.iss")
    assert f'#define MyAppVersion "{VERSION}"' in excel_iss
    assert f"XSTARS_Setup_v{VERSION}.exe" in excel_iss
    assert f'#define AppVersion "{VERSION}"' in _read("installer/wps/XSTARS_WPS.iss")
    assert f'$Version = "{VERSION}"' in _read("installer/wps/build.ps1")
    # Windows add-in modules are injected at build time, so the text is static.
    assert f'msg = "XSTARS v{VERSION}"' in _read("ribbon/ribbon_callbacks_installed.bas")


def test_mac_about_reads_the_version_from_python():
    source = _read("ribbon/ribbon_callbacks.bas")
    about = source[source.index("Sub ShowAbout"):]
    about = about[:about.index("End Sub")]
    assert 'RunPython "import xstars.main; xstars.main.run_about()"' in about
    assert not re.search(r"XSTARS v\d", about)


def test_about_text_and_alert():
    from xstars import main

    assert main.about_text().startswith(f"XSTARS v{VERSION}\n")
    book = MagicMock()
    with patch.object(main.xw.Book, "caller", return_value=book):
        main.run_about()
    book.app.alert.assert_called_once()
    args, kwargs = book.app.alert.call_args
    assert args[0] == main.about_text()
    assert kwargs["title"] == "About XSTARS"


def test_about_falls_back_to_status_bar():
    from xstars import main

    book = MagicMock()
    book.app.alert.side_effect = RuntimeError("no alert on this host")
    book.app.macro.side_effect = RuntimeError("no MsgBox either")
    with patch.object(main.xw.Book, "caller", return_value=book):
        main.run_about()
    assert book.app.status_bar == f"XSTARS v{VERSION}"
