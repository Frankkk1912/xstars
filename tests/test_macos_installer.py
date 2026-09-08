"""Fast, network-free contracts for the macOS installer assembly pipeline."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import lxml.etree as ET
import pytest
import xlwings
from oletools.olevba import VBA_Parser

REPO_ROOT = Path(__file__).resolve().parents[1]
MAC_INSTALLER_DIR = REPO_ROOT / "installer" / "mac"
ASSETS_DIR = MAC_INSTALLER_DIR / "assets"
BUILD_SCRIPT = MAC_INSTALLER_DIR / "build_pkg.py"
RUNTIME_LOCK = MAC_INSTALLER_DIR / "runtime.lock.json"
DISTRIBUTION_TEMPLATE = MAC_INSTALLER_DIR / "distribution.xml"
POSTINSTALL_SCRIPT = MAC_INSTALLER_DIR / "postinstall.sh"
UNINSTALL_SCRIPT = MAC_INSTALLER_DIR / "uninstall.sh"
BUILT_PACKAGE = REPO_ROOT / "installer" / "output" / "XSTARS-1.1.1.pkg"

_spec = importlib.util.spec_from_file_location("xstars_macos_build_pkg", BUILD_SCRIPT)
assert _spec is not None and _spec.loader is not None
build_pkg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build_pkg
_spec.loader.exec_module(build_pkg)

SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCUMENT_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CUSTOM_UI_2006_NS = "http://schemas.microsoft.com/office/2006/01/customui"
CUSTOM_UI_2006_REL = (
    "http://schemas.microsoft.com/office/2006/relationships/ui/extensibility"
)


def _write_odc_payload(path: Path, names: list[str]) -> None:
    """Write the minimal POSIX odc cpio stream needed by package-cleanup tests."""

    def write_entry(archive, name: str) -> None:
        encoded_name = name.encode("utf-8") + b"\0"
        header = (
            "070707"
            "000000"  # device
            "000001"  # inode
            "100644"  # mode
            "000000"  # uid
            "000000"  # gid
            "000001"  # links
            "000000"  # rdev
            "00000000000"  # mtime
            f"{len(encoded_name):06o}"
            "00000000000"  # file size
        ).encode("ascii")
        assert len(header) == 76
        archive.write(header)
        archive.write(encoded_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as archive:
        for name in names:
            write_entry(archive, name)
        write_entry(archive, "TRAILER!!!")


def _clean_component_payload_names() -> list[str]:
    return [
        ".",
        "./Library",
        "./Library/Application Support",
        "./Library/Application Support/XSTARS",
        "./Library/Application Support/XSTARS/XSTARS-payload.tar.gz",
    ]


def _runtime_lock_for(content: bytes, *, sha256: str | None = None):
    return build_pkg.RuntimeLock(
        url="https://example.invalid/runtime.tar.gz",
        sha256=sha256 or hashlib.sha256(content).hexdigest(),
        size=len(content),
        python_version="3.12.14",
        arch="aarch64-apple-darwin",
        variant="install_only_stripped",
        tag="20260901",
    )


def _write_lock(path: Path, content: bytes, expected_content: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "url": "https://example.invalid/runtime.tar.gz",
                "sha256": hashlib.sha256(expected_content).hexdigest(),
                "size": len(content),
                "python_version": "3.12.14",
                "arch": "aarch64-apple-darwin",
                "variant": "install_only_stripped",
                "tag": "20260901",
            }
        ),
        encoding="utf-8",
    )


def _macro_sources(office_file: Path) -> dict[str, str]:
    parser = VBA_Parser(str(office_file))
    macros: dict[str, str] = {}
    try:
        for _, _, module_name, source in parser.extract_macros():
            assert isinstance(module_name, str)
            assert isinstance(source, str)
            macros[module_name] = source
        return macros
    finally:
        parser.close()


def _macro_by_prefix(macros: dict[str, str], prefix: str) -> str:
    matches = [
        source
        for module_name, source in macros.items()
        if module_name.casefold().startswith(prefix.casefold())
    ]
    assert len(matches) == 1, f"expected one VBA module prefixed {prefix!r}"
    return matches[0]


def _normalize_vba(source: str) -> str:
    return source.replace("\r\n", "\n")


def _xml_root(content: bytes):
    assert b"<!DOCTYPE" not in content
    assert b"<!ENTITY" not in content
    parser = ET.XMLParser(resolve_entities=False, no_network=True)
    return ET.fromstring(content, parser=parser)


def _shared_strings(archive: ZipFile) -> list[str]:
    root = _xml_root(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
        for item in root.findall(f"{{{SHEET_NS}}}si")
    ]


def _worksheet_part(archive: ZipFile, sheet_name: str) -> str:
    workbook = _xml_root(archive.read("xl/workbook.xml"))
    sheet = next(
        (
            item
            for item in workbook.findall(f".//{{{SHEET_NS}}}sheet")
            if item.attrib["name"] == sheet_name
        ),
        None,
    )
    assert sheet is not None, f"missing workbook sheet {sheet_name!r}"
    relationship_id = sheet.attrib[f"{{{DOCUMENT_REL_NS}}}id"]

    relationships = _xml_root(archive.read("xl/_rels/workbook.xml.rels"))
    relationship = next(
        (
            item
            for item in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
            if item.attrib["Id"] == relationship_id
        ),
        None,
    )
    assert relationship is not None
    return str(PurePosixPath("xl") / relationship.attrib["Target"])


def _worksheet_value(archive: ZipFile, sheet_name: str, coordinate: str):
    worksheet = _xml_root(archive.read(_worksheet_part(archive, sheet_name)))
    cell = next(
        (
            item
            for item in worksheet.findall(f".//{{{SHEET_NS}}}c")
            if item.attrib["r"] == coordinate
        ),
        None,
    )
    if cell is None:
        return None
    value = cell.find(f"{{{SHEET_NS}}}v")
    if value is None or value.text is None:
        return None
    if cell.attrib.get("t") == "s":
        return _shared_strings(archive)[int(value.text)]
    return value.text


def test_read_project_version_uses_pyproject(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "9.8.7"\n', encoding="utf-8")

    assert build_pkg.read_project_version(project) == "9.8.7"


def test_missing_pyproject_fails_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_pkg, "PROJECT_FILE", tmp_path / "missing.toml")

    assert build_pkg.main(["--version"]) == 1
    assert "project metadata not found" in capsys.readouterr().err


def test_corrupt_pyproject_fails_closed(monkeypatch, tmp_path, capsys):
    project = tmp_path / "pyproject.toml"
    project.write_text("[project\nversion = broken", encoding="utf-8")
    monkeypatch.setattr(build_pkg, "PROJECT_FILE", project)

    assert build_pkg.main(["--version"]) == 1
    stderr = capsys.readouterr().err
    assert "cannot read project metadata" in stderr
    assert str(project) in stderr


def test_signing_environment_is_explicitly_ignored(monkeypatch, capsys):
    monkeypatch.setenv("XSTARS_SIGN", "1")

    assert build_pkg.main(["--version"]) == 0
    output = capsys.readouterr()
    assert output.out.strip() == "1.1.1"
    assert "intentionally unsigned" in output.err


def test_runtime_lock_contains_exact_pinned_release():
    lock = build_pkg.load_runtime_lock(RUNTIME_LOCK)

    assert lock.url == (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20260901/cpython-3.12.14%2B20260901-aarch64-apple-darwin-"
        "install_only_stripped.tar.gz"
    )
    assert lock.sha256 == (
        "81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b"
    )
    assert lock.size == 24_981_445
    assert lock.python_version == "3.12.14"
    assert lock.arch == "aarch64-apple-darwin"
    assert lock.variant == "install_only_stripped"
    assert lock.tag == "20260901"
    assert lock.filename == (
        "cpython-3.12.14+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz"
    )


def test_download_runtime_uses_injected_opener_and_reuses_verified_cache(tmp_path):
    content = b"verified standalone runtime"
    lock = _runtime_lock_for(content)
    opened_urls = []

    def fake_urlopen(url):
        opened_urls.append(url)
        return io.BytesIO(content)

    archive = build_pkg.download_runtime(lock, tmp_path, opener=fake_urlopen)
    assert archive.read_bytes() == content
    assert opened_urls == [lock.url]

    def unexpected_urlopen(_url):
        raise AssertionError("verified cache must avoid a second download")

    assert (
        build_pkg.download_runtime(lock, tmp_path, opener=unexpected_urlopen) == archive
    )


def test_sha256_mismatch_is_nonzero_and_reports_expected_actual(
    monkeypatch, tmp_path, capsys
):
    expected = b"expected"
    tampered = b"tampered"
    lock_file = tmp_path / "runtime.lock.json"
    _write_lock(lock_file, tampered, expected)
    monkeypatch.setattr(build_pkg, "LOCK_FILE", lock_file)
    monkeypatch.setattr(build_pkg, "urlopen", lambda _url: io.BytesIO(tampered))

    exit_code = build_pkg.main(
        ["--prepare-runtime", "--staging-dir", str(tmp_path / "staging")]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "runtime archive verification failed" in stderr
    assert f"expected={hashlib.sha256(expected).hexdigest()}" in stderr
    assert f"actual={hashlib.sha256(tampered).hexdigest()}" in stderr
    assert not list((tmp_path / "staging" / "downloads").glob("*.part"))


def test_extract_runtime_rejects_entries_outside_python(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    source = tmp_path / "escape.txt"
    source.write_text("no", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="not-python/escape.txt")

    with pytest.raises(build_pkg.BuildError, match="outside python"):
        build_pkg.extract_runtime(archive, tmp_path / "staging")

    assert not (tmp_path / "staging" / "python").exists()


def test_assemble_staging_installs_non_editably_and_gathers_assets(tmp_path):
    staging = tmp_path / "staging"
    python_executable = staging / "python" / "bin" / "python3"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("fake interpreter", encoding="utf-8")
    site_packages = staging / "python" / "lib" / "python3.12" / "site-packages"
    applescript = site_packages / "xlwings" / "xlwings-0.37.0.applescript"
    applescript.parent.mkdir(parents=True)
    applescript.write_text("fake AppleScript", encoding="utf-8")

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "XSTARS.xlam").write_bytes(b"xlam")
    (assets / "XSTARS_mac.xlsm").write_bytes(b"xlsm")
    repo = tmp_path / "repo"
    repo.mkdir()
    commands = []

    def fake_runner(command):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    layout = build_pkg.assemble_staging(
        staging, repo_root=repo, assets_dir=assets, runner=fake_runner
    )
    (layout.bin / "XSTARS.xlam").write_bytes(b"stale")
    layout = build_pkg.assemble_staging(
        staging, repo_root=repo, assets_dir=assets, runner=fake_runner
    )

    assert layout.python == staging / "python"
    assert layout.python_executable == python_executable
    assert {path.name for path in layout.bin.iterdir()} == {
        "XSTARS.xlam",
        "XSTARS_mac.xlsm",
        "xlwings.applescript",
    }
    assert (layout.bin / "XSTARS.xlam").read_bytes() == b"xlam"
    assert (layout.bin / "XSTARS_mac.xlsm").read_bytes() == b"xlsm"
    assert (layout.bin / "xlwings.applescript").read_text() == "fake AppleScript"
    assert commands[0] == (str(python_executable), "-m", "pip", "--version")
    install = commands[1]
    assert install == (
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        str(repo),
    )
    assert "-e" not in install
    assert all("dev" not in argument for argument in install)
    assert commands[2:] == commands[:2]


def test_xlam_has_2006_custom_ui_and_root_relationship():
    artifact = ASSETS_DIR / "XSTARS.xlam"
    with ZipFile(artifact) as archive:
        names = archive.namelist()
        assert "customUI/customUI.xml" in names
        assert all("customui14" not in name.casefold() for name in names)
        custom_ui = _xml_root(archive.read("customUI/customUI.xml"))
        assert custom_ui.tag == f"{{{CUSTOM_UI_2006_NS}}}customUI"
        assert all(
            "insertAfterMso" not in element.attrib for element in custom_ui.iter()
        )

        relationships = _xml_root(archive.read("_rels/.rels"))
        custom_relationships = [
            relationship
            for relationship in relationships.findall(
                f"{{{PACKAGE_REL_NS}}}Relationship"
            )
            if relationship.attrib.get("Type") == CUSTOM_UI_2006_REL
        ]
        assert len(custom_relationships) == 1
        assert custom_relationships[0].attrib["Target"] == ("customUI/customUI.xml")


def test_office_artifacts_embed_expected_normalized_vba_sources():
    repository_callbacks = (REPO_ROOT / "ribbon" / "ribbon_callbacks.bas").read_text(
        encoding="utf-8"
    )
    xlam_macros = _macro_sources(ASSETS_DIR / "XSTARS.xlam")
    workbook_macros = _macro_sources(ASSETS_DIR / "XSTARS_mac.xlsm")

    assert _normalize_vba(
        _macro_by_prefix(xlam_macros, "RibbonCallbacks")
    ) == _normalize_vba(repository_callbacks)
    assert _normalize_vba(
        _macro_by_prefix(workbook_macros, "RibbonCallbacks")
    ) == _normalize_vba(repository_callbacks)

    assert xlwings.__version__ == "0.37.0"
    wheel_xlwings_bas = (
        Path(xlwings.__file__).with_name("xlwings.bas").read_text(encoding="utf-8")
    )
    assert _normalize_vba(
        _macro_by_prefix(workbook_macros, "xlwings")
    ) == _normalize_vba(wheel_xlwings_bas)
    assert _macro_by_prefix(workbook_macros, "Dictionary").strip()


@pytest.mark.parametrize("artifact_name", ["XSTARS.xlam", "XSTARS_mac.xlsm"])
def test_office_artifact_xlwings_interpreter_is_empty(artifact_name):
    with ZipFile(ASSETS_DIR / artifact_name) as archive:
        assert _worksheet_value(archive, "xlwings.conf", "A1") == "Interpreter"
        assert _worksheet_value(archive, "xlwings.conf", "B1") in (None, "")


@pytest.mark.parametrize("artifact_name", ["XSTARS.xlam", "XSTARS_mac.xlsm"])
def test_office_artifact_has_no_private_path_metadata(artifact_name):
    with ZipFile(ASSETS_DIR / artifact_name) as archive:
        assert "xl/vbaProject.bin" in archive.namelist()
        assert archive.getinfo("xl/vbaProject.bin").file_size > 0
        for member in archive.infolist():
            if member.is_dir():
                continue
            content = archive.read(member)
            assert b"x15ac:absPath" not in content, member.filename
            assert b"C:\\Users\\" not in content, member.filename
            assert b"/Users/frank" not in content, member.filename
            assert not re.search(rb"/Users/(?!<User>)", content), member.filename


def _write_office_stub(path: Path, content: bytes = b"clean") -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/vbaProject.bin", content)


def _fake_m2_staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    python_executable = staging / "python" / "bin" / "python3"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("fake interpreter", encoding="utf-8")
    bin_dir = staging / "bin"
    bin_dir.mkdir()
    _write_office_stub(bin_dir / "XSTARS.xlam")
    _write_office_stub(bin_dir / "XSTARS_mac.xlsm")
    (bin_dir / "xlwings.applescript").write_text("script", encoding="utf-8")
    return staging


def test_distribution_template_and_rendering(tmp_path):
    raw = _xml_root(DISTRIBUTION_TEMPLATE.read_bytes())
    domains = raw.find("domains")
    options = raw.find("options")
    os_version = raw.find("./volume-check/allowed-os-versions/os-version")

    assert domains is not None
    assert domains.attrib == {"enable_currentUserHome": "true"}
    assert options is not None
    assert options.attrib["hostArchitectures"] == "arm64"
    assert options.attrib["customize"] == "never"
    assert options.attrib["require-scripts"] == "true"
    assert "rootVolumeOnly" not in options.attrib
    assert os_version is not None and os_version.attrib["min"] == "12.0"

    rendered = build_pkg.render_distribution(
        DISTRIBUTION_TEMPLATE, tmp_path / "distribution.xml", "9.8.7"
    )
    content = rendered.read_text(encoding="utf-8")
    assert "__VERSION__" not in content
    pkg_ref = _xml_root(rendered.read_bytes()).find("pkg-ref")
    assert pkg_ref is not None
    assert pkg_ref.attrib["version"] == "9.8.7"
    assert pkg_ref.text == "XSTARS-component.pkg"


def test_render_distribution_requires_version_placeholder(tmp_path):
    template = tmp_path / "distribution.xml"
    template.write_text("<installer-gui-script/>", encoding="utf-8")

    with pytest.raises(build_pkg.BuildError, match="does not contain __VERSION__"):
        build_pkg.render_distribution(template, tmp_path / "out.xml", "1.2.3")


def test_assemble_install_tree_layout_and_optional_uninstaller(tmp_path):
    staging = _fake_m2_staging(tmp_path)
    uninstall = tmp_path / "uninstall.sh"
    uninstall.write_text("#!/bin/sh\n", encoding="utf-8")
    destination = build_pkg.assemble_install_tree(
        staging, tmp_path / "install-tree", uninstall_script=uninstall
    )

    assert destination.relative_to(tmp_path / "install-tree").as_posix() == (
        "Library/Application Support/XSTARS"
    )
    assert (destination / "python/bin/python3").is_file()
    assert (destination / "bin/XSTARS.xlam").is_file()
    assert (destination / "bin/xlwings.applescript").is_file()
    assert (destination / "bin/XSTARS_mac.xlsm").is_file()
    assert (destination / "Templates/XSTARS_mac.xlsm").is_file()
    assert (destination / "uninstall.sh").is_file()

    without_uninstaller = build_pkg.assemble_install_tree(
        staging,
        tmp_path / "install-tree-without-uninstaller",
        uninstall_script=tmp_path / "not-yet-created.sh",
    )
    assert not (without_uninstaller / "uninstall.sh").exists()


def test_payload_scan_rejects_appledouble_finder_and_abspath(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    forbidden = root / "._python"
    forbidden.write_bytes(b"metadata")
    with pytest.raises(build_pkg.BuildError, match="forbidden macOS metadata"):
        build_pkg.scan_payload_tree(root)
    forbidden.unlink()

    finder = root / ".DS_Store"
    finder.write_bytes(b"metadata")
    with pytest.raises(build_pkg.BuildError, match="forbidden macOS metadata"):
        build_pkg.scan_payload_tree(root)
    finder.unlink()

    artifact = root / "XSTARS.xlam"
    _write_office_stub(artifact, b"x15ac:absPath")
    with pytest.raises(build_pkg.BuildError, match="forbidden x15ac:absPath"):
        build_pkg.scan_payload_tree(root)


def test_payload_tar_command_strips_macos_metadata(tmp_path):
    install_tree = tmp_path / "install-tree" / "XSTARS"
    archive = tmp_path / "component-root" / "XSTARS-payload.tar.gz"

    command = build_pkg.payload_tar_command(install_tree, archive)

    assert command[:3] == [
        "/usr/bin/env",
        "COPYFILE_DISABLE=1",
        "/usr/bin/tar",
    ]
    assert "--no-xattrs" in command
    assert "--no-mac-metadata" in command
    assert command[-3:] == ["-C", str(install_tree.parent), "XSTARS"]


def test_package_commands_are_unsigned_user_domain_and_one_way(tmp_path):
    scripts = tmp_path / "scripts"
    pkg_command = build_pkg.pkgbuild_command(
        tmp_path / "root",
        tmp_path / "XSTARS-component.pkg",
        "1.1.1",
        scripts_dir=scripts,
    )
    product_command = build_pkg.productbuild_command(
        tmp_path / "distribution.xml",
        tmp_path,
        tmp_path / "XSTARS-1.1.1.pkg",
    )

    assert pkg_command == [
        "pkgbuild",
        "--root",
        str(tmp_path / "root"),
        "--identifier",
        "com.frank-sysu.xstars",
        "--version",
        "1.1.1",
        "--install-location",
        "/",
        "--scripts",
        str(scripts),
        str(tmp_path / "XSTARS-component.pkg"),
    ]
    assert product_command == [
        "productbuild",
        "--distribution",
        str(tmp_path / "distribution.xml"),
        "--package-path",
        str(tmp_path),
        str(tmp_path / "XSTARS-1.1.1.pkg"),
    ]
    combined = " ".join((*pkg_command, *product_command)).casefold()
    assert "sign" not in combined
    assert "notar" not in combined


def test_render_xlwings_conf_creates_canonical_interpreter_entry():
    assert build_pkg.render_xlwings_conf() == (
        '"INTERPRETER_MAC","$HOME/Library/Application Support/'
        'XSTARS/python/bin/python3"\n'
    )


def test_render_xlwings_conf_preserves_unknown_keys_and_replaces_only_interpreter():
    existing = (
        '"LOG_FILE","$HOME/xlwings.log"\n'
        '"INTERPRETER_MAC","/old/python"\n'
        '"USE_UDF_SERVER","False"\n'
    )

    rendered = build_pkg.render_xlwings_conf(existing)

    assert rendered == (
        '"LOG_FILE","$HOME/xlwings.log"\n'
        '"INTERPRETER_MAC","$HOME/Library/Application Support/'
        'XSTARS/python/bin/python3"\n'
        '"USE_UDF_SERVER","False"\n'
    )
    assert build_pkg.render_xlwings_conf(rendered) == rendered


def test_postinstall_has_user_domain_deployment_and_fail_closed_runtime_contract():
    script = POSTINSTALL_SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/bin/bash\n")
    assert "\nset -e\n" in script
    assert 'INSTALL_ROOT="$USER_HOME/Library/Application Support/XSTARS"' in script
    assert (
        'EXCEL_STARTUP="$USER_HOME/Library/Group Containers/'
        'UBF8T346G9.Office/User Content.localized/Startup.localized/Excel"'
    ) in script
    assert (
        'APP_SCRIPTS_DIR="$USER_HOME/Library/Application Scripts/com.microsoft.Excel"'
    ) in script
    assert (
        'XLWINGS_CONF_DIR="$USER_HOME/Library/Containers/com.microsoft.Excel/Data"'
    ) in script
    assert (
        '"INTERPRETER_MAC","$HOME/Library/Application Support/'
        'XSTARS/python/bin/python3"'
    ) in script
    assert "COPYFILE_DISABLE=1 /usr/bin/tar -xzf" in script
    assert 'elif [ ! -x "$PYTHON_EXECUTABLE" ]; then' in script
    assert script.count("exit 1") >= 2
    assert script.rstrip().endswith("exit 0")

    assert "/usr/bin/stat -f '%Su' /dev/console" in script
    assert "show State:/Users/ConsoleUser" in script
    assert "/usr/bin/dscl . -read" in script
    assert "/usr/sbin/chown -R" in script
    assert "copy_user_file" in script and "|| true" in script
    assert "merge_xlwings_conf || true" in script
    assert "rm -rf" not in script
    assert "~/.xstars" not in script


def test_stage_package_scripts_includes_executable_postinstall_and_conf(tmp_path):
    source = tmp_path / "postinstall.sh"
    source.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    scripts_dir = build_pkg.stage_package_scripts(tmp_path / "work", source)

    assert scripts_dir == tmp_path / "work" / "scripts"
    postinstall = scripts_dir / "postinstall"
    assert postinstall.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert postinstall.stat().st_mode & 0o111
    assert (scripts_dir / "xlwings.conf").read_text(encoding="utf-8") == (
        build_pkg.render_xlwings_conf()
    )


def test_component_package_cleanup_removes_outer_and_nested_appledouble(tmp_path):
    component = tmp_path / "XSTARS-component.pkg"
    component.write_bytes(b"original")
    commands = []
    expansion_count = 0

    def fake_runner(command):
        nonlocal expansion_count
        commands.append(tuple(command))
        if command[0] == "pkgutil" and command[1] == "--expand":
            expansion_count += 1
            destination = Path(command[-1])
            scripts = destination / "Scripts"
            scripts.mkdir(parents=True)
            (scripts / "postinstall").write_text("script", encoding="utf-8")
            payload_names = _clean_component_payload_names()
            if expansion_count == 1:
                (scripts / "._postinstall").write_bytes(b"appledouble")
                payload_names.append("./Library/._Application Support")
            _write_odc_payload(destination / "Payload", payload_names)
        elif command[0] == "/usr/bin/env" and "-xzf" in command:
            extracted = Path(command[command.index("-C") + 1])
            nested = extracted / "Library" / "Application Support" / "XSTARS"
            nested.mkdir(parents=True)
            (nested / "XSTARS-payload.tar.gz").write_bytes(b"payload")
            (nested / "._XSTARS-payload.tar.gz").write_bytes(b"appledouble")
        elif command[0] == "/usr/bin/env" and "-czf" in command:
            _write_odc_payload(
                Path(command[command.index("-czf") + 1]),
                _clean_component_payload_names(),
            )
        elif command[0] == "pkgutil" and command[1] == "--flatten":
            expanded = Path(command[2])
            assert not build_pkg.find_macos_metadata(expanded)
            build_pkg.validate_component_payload(expanded / "Payload")
            Path(command[-1]).write_bytes(b"cleaned")
        return subprocess.CompletedProcess(command, 0)

    build_pkg.sanitize_component_package(
        component,
        tmp_path,
        runner=fake_runner,
    )

    assert component.read_bytes() == b"cleaned"
    assert [(command[0], command[1]) for command in commands] == [
        ("pkgutil", "--expand"),
        ("/usr/bin/env", "COPYFILE_DISABLE=1"),
        ("/usr/bin/env", "COPYFILE_DISABLE=1"),
        ("pkgutil", "--flatten"),
        ("pkgutil", "--expand"),
    ]
    assert not (tmp_path / "component-expanded").exists()
    assert not (tmp_path / "component-verification").exists()
    assert not (tmp_path / "payload-expanded").exists()


def test_assemble_package_uses_injected_commands(tmp_path):
    staging = _fake_m2_staging(tmp_path)
    work = tmp_path / "work"
    output = tmp_path / "output"
    commands = []

    def fake_runner(command):
        commands.append(tuple(command))
        if command[0] == "/usr/bin/env":
            archive = Path(command[command.index("-czf") + 1])
            source_parent = Path(command[command.index("-C") + 1])
            source_name = command[-1]
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source_parent / source_name, arcname=source_name)
        elif command[0] in {"pkgbuild", "productbuild"}:
            Path(command[-1]).write_bytes(b"fake package")
        return subprocess.CompletedProcess(command, 0)

    layout = build_pkg.assemble_package(
        staging,
        work,
        output,
        "1.1.1",
        distribution_template=DISTRIBUTION_TEMPLATE,
        runner=fake_runner,
    )

    assert layout.final_package == output / "XSTARS-1.1.1.pkg"
    assert layout.final_package.read_bytes() == b"fake package"
    assert layout.payload.archive.is_file()
    assert [command[0] for command in commands] == [
        "/usr/bin/env",
        "pkgbuild",
        "productbuild",
    ]
    assert "--scripts" not in commands[1]
    assert "__VERSION__" not in layout.distribution.read_text(encoding="utf-8")
    build_pkg.validate_payload_archive(layout.payload.archive)


def test_uninstall_has_backup_registration_cleanup_and_data_safety_contract():
    script = UNINSTALL_SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/bin/bash\n")
    assert 'MODE="dry-run"' in script
    assert '--apply) MODE="apply"' in script
    assert "no files will be changed" in script
    assert '/usr/bin/pgrep -x "$process"' in script
    assert '"Microsoft Excel" "WPS Office" "wpsoffice"' in script
    assert "xstars_launch.scpt" in script
    assert "xlwings.applescript" in script
    assert "INTERPRETER_MAC" in script
    assert 'BACKUP_PARENT="$USER_HOME/Documents/XSTARS-uninstall-backups"' in script
    assert '/bin/cp -p "$database" "$backup"' in script
    assert "HKEY_CURRENT_USER_values" in script
    assert "OPEN[0-9]*" in script
    assert "XSTARS.XLAM" in script
    assert "PRAGMA integrity_check;" in script
    assert "restore_registration_backup" in script
    assert "/usr/sbin/pkgutil --forget" in script
    assert 'PACKAGE_IDENTIFIER="com.frank-sysu.xstars"' in script
    assert "~/.xstars" not in script.casefold()
    assert "$user_home/.xstars" not in script.casefold()
    deletion_lines = [line for line in script.splitlines() if "/bin/rm" in line]
    assert all(".xstars" not in line.casefold() for line in deletion_lines)


def test_uninstall_help_and_default_dry_run_do_not_write(tmp_path):
    environment = {**os.environ, "HOME": str(tmp_path), "TMPDIR": str(tmp_path)}
    before = tuple(tmp_path.iterdir())

    help_result = subprocess.run(
        [str(UNINSTALL_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    dry_run = subprocess.run(
        [str(UNINSTALL_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert help_result.returncode == 0
    assert "Usage: uninstall.sh" in help_result.stdout
    assert dry_run.returncode == 0
    assert "dry-run only" in dry_run.stdout
    assert "would remove Excel Startup add-in" in dry_run.stdout
    assert "would require PRAGMA integrity_check" in dry_run.stdout
    assert tuple(tmp_path.iterdir()) == before


def test_component_payload_validation_rejects_nested_appledouble(tmp_path):
    payload = tmp_path / "Payload"
    names = _clean_component_payload_names()
    names.append("./Library/Application Support/._XSTARS")
    _write_odc_payload(payload, names)

    with pytest.raises(
        build_pkg.BuildError,
        match="component payload contains forbidden macOS metadata",
    ):
        build_pkg.validate_component_payload(payload)


@pytest.mark.skipif(
    sys.platform != "darwin" or not BUILT_PACKAGE.is_file(),
    reason="requires the locally built macOS package",
)
def test_built_pkg_is_unsigned_product_archive(tmp_path):
    listing = subprocess.run(
        ["xar", "-tf", str(BUILT_PACKAGE)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "Distribution" in listing
    assert "XSTARS-component.pkg" in listing

    expanded = tmp_path / "expanded-package"
    subprocess.run(
        ["pkgutil", "--expand", str(BUILT_PACKAGE), str(expanded)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not build_pkg.find_macos_metadata(expanded)
    component_dir = expanded / "XSTARS-component.pkg"
    build_pkg.validate_component_payload(component_dir / "Payload")
    component_payload = tmp_path / "component-payload"
    component_payload.mkdir()
    subprocess.run(
        [
            "/usr/bin/env",
            "COPYFILE_DISABLE=1",
            "/usr/bin/tar",
            "--no-xattrs",
            "--no-mac-metadata",
            "-xzf",
            str(component_dir / "Payload"),
            "-C",
            str(component_payload),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload_archive = (
        component_payload
        / "Library"
        / "Application Support"
        / "XSTARS"
        / build_pkg.PAYLOAD_ARCHIVE_NAME
    )
    with tarfile.open(payload_archive, "r:gz") as archive:
        archive_names = {PurePosixPath(name) for name in archive.getnames()}
    assert PurePosixPath("XSTARS/uninstall.sh") in archive_names

    scripts_dir = component_dir / "Scripts"
    assert (scripts_dir / "postinstall").is_file()
    assert (scripts_dir / "postinstall").stat().st_mode & 0o111
    assert (scripts_dir / "xlwings.conf").read_text(encoding="utf-8") == (
        build_pkg.render_xlwings_conf()
    )

    signature = subprocess.run(
        ["pkgutil", "--check-signature", str(BUILT_PACKAGE)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert signature.returncode != 0
    assert "no signature" in (signature.stdout + signature.stderr).casefold()
