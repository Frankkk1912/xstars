#!/usr/bin/env python3
"""Build the unsigned, user-domain XSTARS macOS installer.

The build is a one-way pipeline: prepare the relocatable runtime, assemble a
metadata-free payload archive, then invoke ``pkgbuild`` and ``productbuild``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_FILE = REPO_ROOT / "pyproject.toml"
LOCK_FILE = SCRIPT_DIR / "runtime.lock.json"
ASSETS_DIR = SCRIPT_DIR / "assets"
DEFAULT_STAGING_DIR = SCRIPT_DIR / "staging"
DISTRIBUTION_TEMPLATE = SCRIPT_DIR / "distribution.xml"
OUTPUT_DIR = REPO_ROOT / "installer" / "output"
PACKAGE_IDENTIFIER = "com.frank-sysu.xstars"
COMPONENT_PACKAGE_NAME = "XSTARS-component.pkg"
PAYLOAD_ARCHIVE_NAME = "XSTARS-payload.tar.gz"
USER_INSTALL_LOCATION = "/"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

CommandRunner = Callable[[Sequence[str]], object]
UrlOpener = Callable[[str], Any]


class BuildError(RuntimeError):
    """A fail-closed installer assembly error suitable for CLI display."""


@dataclass(frozen=True)
class RuntimeLock:
    """Pinned python-build-standalone artifact metadata."""

    url: str
    sha256: str
    size: int
    python_version: str
    arch: str
    variant: str
    tag: str

    @property
    def filename(self) -> str:
        """Return the decoded release asset filename from the pinned URL."""
        filename = PurePosixPath(unquote(urlparse(self.url).path)).name
        if not filename:
            raise BuildError("runtime lock URL does not contain an asset filename")
        return filename


@dataclass(frozen=True)
class StagingLayout:
    """Paths produced by the runtime staging pipeline."""

    root: Path
    python: Path
    python_executable: Path
    bin: Path


@dataclass(frozen=True)
class PayloadLayout:
    """Expanded install tree and the single-file component payload."""

    install_tree: Path
    component_root: Path
    archive: Path


@dataclass(frozen=True)
class PackageLayout:
    """Paths produced by the unsigned package build pipeline."""

    component_package: Path
    distribution: Path
    final_package: Path
    payload: PayloadLayout


def read_project_version(project_file: Path = PROJECT_FILE) -> str:
    """Read the canonical project version from pyproject.toml, fail closed."""
    try:
        with project_file.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise BuildError(f"project metadata not found: {project_file}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BuildError(f"cannot read project metadata {project_file}: {exc}") from exc

    try:
        version = document["project"]["version"]
    except (KeyError, TypeError) as exc:
        raise BuildError(
            f"project.version is missing from project metadata: {project_file}"
        ) from exc
    if not isinstance(version, str) or not version.strip():
        raise BuildError(
            f"project.version must be a non-empty string in project metadata: "
            f"{project_file}"
        )
    return version


def load_runtime_lock(lock_file: Path = LOCK_FILE) -> RuntimeLock:
    """Load and validate the pinned runtime manifest."""
    try:
        document = json.loads(lock_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"runtime lock not found: {lock_file}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read runtime lock {lock_file}: {exc}") from exc

    fields = (
        "url",
        "sha256",
        "size",
        "python_version",
        "arch",
        "variant",
        "tag",
    )
    missing = [field for field in fields if field not in document]
    if missing:
        raise BuildError(
            f"runtime lock {lock_file} is missing fields: {', '.join(missing)}"
        )

    lock = RuntimeLock(**{field: document[field] for field in fields})
    text_fields = (
        lock.url,
        lock.python_version,
        lock.arch,
        lock.variant,
        lock.tag,
    )
    if not all(isinstance(value, str) and value for value in text_fields):
        raise BuildError(f"runtime lock {lock_file} contains empty/non-string fields")
    if (
        not isinstance(lock.sha256, str)
        or len(lock.sha256) != 64
        or any(character not in "0123456789abcdef" for character in lock.sha256)
    ):
        raise BuildError(f"runtime lock {lock_file} contains an invalid SHA256")
    if not isinstance(lock.size, int) or isinstance(lock.size, bool) or lock.size <= 0:
        raise BuildError(f"runtime lock {lock_file} contains an invalid size")
    return lock


def verify_runtime_archive(
    archive: Path,
    lock: RuntimeLock,
    *,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> None:
    """Verify a runtime archive's pinned size and SHA256 in one streaming pass."""
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with archive.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                actual_size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise BuildError(f"cannot read runtime archive {archive}: {exc}") from exc

    actual_sha256 = digest.hexdigest()
    mismatches = []
    if actual_size != lock.size:
        mismatches.append(f"size expected={lock.size} actual={actual_size}")
    if actual_sha256 != lock.sha256:
        mismatches.append(f"sha256 expected={lock.sha256} actual={actual_sha256}")
    if mismatches:
        raise BuildError(
            f"runtime archive verification failed for {archive}: "
            + "; ".join(mismatches)
        )


def download_runtime(
    lock: RuntimeLock,
    downloads_dir: Path,
    *,
    opener: UrlOpener | None = None,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> Path:
    """Download the pinned runtime, reusing only a verified cached archive."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    destination = downloads_dir / lock.filename
    if destination.exists():
        try:
            verify_runtime_archive(destination, lock, chunk_size=chunk_size)
        except BuildError as exc:
            print(f"Discarding invalid cached runtime: {exc}", file=sys.stderr)
            destination.unlink()
        else:
            print(f"Reusing verified runtime archive: {destination}")
            return destination

    open_url = opener or urlopen
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading pinned runtime: {lock.url}")
    try:
        with open_url(lock.url) as response, temporary.open("wb") as output:
            while chunk := response.read(chunk_size):
                output.write(chunk)
        verify_runtime_archive(temporary, lock, chunk_size=chunk_size)
        temporary.replace(destination)
    except BuildError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise BuildError(f"runtime download failed for {lock.url}: {exc}") from exc
    return destination


def extract_runtime(archive: Path, staging_dir: Path) -> Path:
    """Safely replace ``staging/python`` with the archive's runtime tree."""
    python_dir = staging_dir / "python"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            members = bundle.getmembers()
            invalid = [
                member.name
                for member in members
                if not PurePosixPath(member.name).parts
                or PurePosixPath(member.name).parts[0] != "python"
            ]
            if invalid:
                raise BuildError(
                    "runtime archive contains entries outside python/: "
                    + ", ".join(invalid[:3])
                )
            bundle.extractall(staging_dir, members=members, filter="data")
    except BuildError:
        shutil.rmtree(python_dir, ignore_errors=True)
        raise
    except (OSError, tarfile.TarError) as exc:
        shutil.rmtree(python_dir, ignore_errors=True)
        raise BuildError(f"cannot extract runtime archive {archive}: {exc}") from exc

    python_executable = python_dir / "bin" / "python3"
    if not python_executable.is_file():
        raise BuildError(
            f"runtime archive did not provide expected interpreter: {python_executable}"
        )
    return python_dir


def default_command_runner(command: Sequence[str]) -> object:
    """Run an external command without shell expansion or recursion."""
    printable = " ".join(str(item) for item in command)
    print(f"$ {printable}")
    return subprocess.run([str(item) for item in command], check=True, text=True)


def _run_checked(
    command: Sequence[str],
    runner: CommandRunner,
    description: str,
) -> object:
    try:
        return runner(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildError(f"{description} failed: {exc}") from exc


def ensure_runtime_pip(
    python_executable: Path,
    *,
    runner: CommandRunner = default_command_runner,
) -> None:
    """Ensure pip exists in the standalone runtime."""
    check_command = [str(python_executable), "-m", "pip", "--version"]
    try:
        runner(check_command)
    except subprocess.CalledProcessError:
        _run_checked(
            [str(python_executable), "-m", "ensurepip", "--upgrade"],
            runner,
            "pip bootstrap",
        )
        _run_checked(check_command, runner, "pip availability check")
    except OSError as exc:
        raise BuildError(f"pip availability check failed: {exc}") from exc


def install_project(
    python_executable: Path,
    repo_root: Path,
    *,
    runner: CommandRunner = default_command_runner,
) -> None:
    """Install XSTARS and runtime dependencies non-editably into the runtime."""
    _run_checked(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            str(repo_root),
        ],
        runner,
        "runtime dependency installation",
    )


def find_xlwings_applescript(python_dir: Path) -> Path:
    """Locate the versioned AppleScript shipped by the installed xlwings wheel."""
    candidates = {
        candidate.resolve()
        for site_packages in python_dir.glob("lib/python*/site-packages")
        for candidate in site_packages.rglob("xlwings*.applescript")
        if candidate.is_file()
    }
    if not candidates:
        raise BuildError(
            f"installed xlwings did not provide xlwings*.applescript under {python_dir}"
        )
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates))
        raise BuildError(f"multiple xlwings AppleScript files found: {rendered}")
    return candidates.pop()


def assemble_staging(
    staging_dir: Path,
    repo_root: Path = REPO_ROOT,
    assets_dir: Path = ASSETS_DIR,
    *,
    runner: CommandRunner = default_command_runner,
) -> StagingLayout:
    """Install into an extracted runtime and gather M1 binary assets."""
    python_dir = staging_dir / "python"
    python_executable = python_dir / "bin" / "python3"
    if not python_executable.is_file():
        raise BuildError(f"runtime interpreter is missing: {python_executable}")

    ensure_runtime_pip(python_executable, runner=runner)
    install_project(python_executable, repo_root, runner=runner)
    applescript = find_xlwings_applescript(python_dir)

    bin_dir = staging_dir / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True)
    for asset_name in ("XSTARS.xlam", "XSTARS_mac.xlsm"):
        source = assets_dir / asset_name
        if not source.is_file():
            raise BuildError(f"required installer asset is missing: {source}")
        shutil.copy2(source, bin_dir / asset_name)
    shutil.copy2(applescript, bin_dir / "xlwings.applescript")

    return StagingLayout(
        root=staging_dir,
        python=python_dir,
        python_executable=python_executable,
        bin=bin_dir,
    )


def prepare_runtime(
    staging_dir: Path,
    *,
    lock_file: Path = LOCK_FILE,
    repo_root: Path = REPO_ROOT,
    assets_dir: Path = ASSETS_DIR,
    opener: UrlOpener | None = None,
    runner: CommandRunner = default_command_runner,
) -> StagingLayout:
    """Download, verify, extract, install, and gather the M1 staging tree."""
    lock = load_runtime_lock(lock_file)
    archive = download_runtime(lock, staging_dir / "downloads", opener=opener)
    extract_runtime(archive, staging_dir)
    return assemble_staging(
        staging_dir,
        repo_root=repo_root,
        assets_dir=assets_dir,
        runner=runner,
    )


def staging_layout(staging_dir: Path) -> StagingLayout:
    """Validate and describe an existing runtime staging tree."""
    python_dir = staging_dir / "python"
    python_executable = python_dir / "bin" / "python3"
    bin_dir = staging_dir / "bin"
    required = (
        python_executable,
        bin_dir / "XSTARS.xlam",
        bin_dir / "XSTARS_mac.xlsm",
        bin_dir / "xlwings.applescript",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError(
            "runtime staging is incomplete; run --prepare-runtime first; missing: "
            + ", ".join(missing)
        )
    return StagingLayout(
        root=staging_dir,
        python=python_dir,
        python_executable=python_executable,
        bin=bin_dir,
    )


def scan_office_metadata(artifact: Path) -> None:
    """Fail if an Office artifact contains build-machine path metadata."""
    try:
        with ZipFile(artifact) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                content = archive.read(member)
                if b"x15ac:absPath" in content:
                    raise BuildError(
                        f"forbidden x15ac:absPath metadata in "
                        f"{artifact}:{member.filename}"
                    )
    except (OSError, BadZipFile) as exc:
        raise BuildError(f"cannot inspect Office artifact {artifact}: {exc}") from exc


def scan_payload_tree(root: Path) -> None:
    """Reject AppleDouble/Finder debris and leaked Office path metadata."""
    if not root.is_dir():
        raise BuildError(f"payload tree is missing: {root}")
    forbidden = [
        path
        for path in root.rglob("*")
        if path.name == ".DS_Store" or path.name.startswith("._")
    ]
    if forbidden:
        raise BuildError(
            "payload contains forbidden macOS metadata: "
            + ", ".join(str(path) for path in forbidden[:5])
        )
    office_artifact_names = {"XSTARS.xlam", "XSTARS_mac.xlsm"}
    for artifact in root.rglob("*"):
        if artifact.is_file() and artifact.name in office_artifact_names:
            scan_office_metadata(artifact)


def assemble_install_tree(
    staging_dir: Path,
    install_tree_root: Path,
    *,
    uninstall_script: Path | None = None,
) -> Path:
    """Assemble the final ``Library/Application Support/XSTARS`` tree."""
    staging = staging_layout(staging_dir)
    if install_tree_root.exists():
        shutil.rmtree(install_tree_root)

    destination = install_tree_root / "Library" / "Application Support" / "XSTARS"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging.python, destination / "python", symlinks=True)

    bin_dir = destination / "bin"
    templates_dir = destination / "Templates"
    bin_dir.mkdir()
    templates_dir.mkdir()
    for asset_name in ("XSTARS.xlam", "xlwings.applescript", "XSTARS_mac.xlsm"):
        shutil.copy2(staging.bin / asset_name, bin_dir / asset_name)
    shutil.copy2(staging.bin / "XSTARS_mac.xlsm", templates_dir / "XSTARS_mac.xlsm")
    if uninstall_script is not None and uninstall_script.is_file():
        shutil.copy2(uninstall_script, destination / "uninstall.sh")

    scan_payload_tree(install_tree_root)
    return destination


def payload_tar_command(install_tree: Path, archive: Path) -> list[str]:
    """Return the metadata-stripping tar command used for the payload."""
    return [
        "/usr/bin/env",
        "COPYFILE_DISABLE=1",
        "/usr/bin/tar",
        "--no-xattrs",
        "--no-mac-metadata",
        "-czf",
        str(archive),
        "-C",
        str(install_tree.parent),
        install_tree.name,
    ]


def validate_payload_archive(archive: Path) -> None:
    """Validate payload archive structure without extracting it."""
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            names = [PurePosixPath(member.name) for member in bundle.getmembers()]
    except (OSError, tarfile.TarError) as exc:
        raise BuildError(f"cannot inspect payload archive {archive}: {exc}") from exc

    forbidden = [
        str(path)
        for path in names
        if any(part == ".DS_Store" or part.startswith("._") for part in path.parts)
    ]
    if forbidden:
        raise BuildError(
            "payload archive contains forbidden macOS metadata: "
            + ", ".join(forbidden[:5])
        )
    required = {
        PurePosixPath("XSTARS/python"),
        PurePosixPath("XSTARS/bin/XSTARS.xlam"),
        PurePosixPath("XSTARS/bin/xlwings.applescript"),
        PurePosixPath("XSTARS/bin/XSTARS_mac.xlsm"),
        PurePosixPath("XSTARS/Templates/XSTARS_mac.xlsm"),
    }
    missing = sorted(str(path) for path in required if path not in names)
    if missing:
        raise BuildError(
            f"payload archive {archive} is missing entries: {', '.join(missing)}"
        )


def create_payload_archive(
    install_tree: Path,
    component_root: Path,
    *,
    runner: CommandRunner = default_command_runner,
) -> Path:
    """Store the install tree as one metadata-free component payload file."""
    archive = (
        component_root
        / "Library"
        / "Application Support"
        / "XSTARS"
        / PAYLOAD_ARCHIVE_NAME
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        payload_tar_command(install_tree, archive),
        runner,
        "payload archive creation",
    )
    if not archive.is_file():
        raise BuildError(f"payload archive command did not create {archive}")
    validate_payload_archive(archive)
    scan_payload_tree(component_root)
    return archive


def assemble_payload(
    staging_dir: Path,
    work_dir: Path,
    *,
    uninstall_script: Path | None = None,
    runner: CommandRunner = default_command_runner,
) -> PayloadLayout:
    """Assemble and archive the user-domain payload under a work directory."""
    install_tree_root = work_dir / "install-tree"
    install_tree = assemble_install_tree(
        staging_dir,
        install_tree_root,
        uninstall_script=uninstall_script,
    )
    component_root = work_dir / "component-root"
    if component_root.exists():
        shutil.rmtree(component_root)
    archive = create_payload_archive(
        install_tree,
        component_root,
        runner=runner,
    )
    return PayloadLayout(
        install_tree=install_tree,
        component_root=component_root,
        archive=archive,
    )


def render_distribution(
    template: Path,
    destination: Path,
    version: str,
) -> Path:
    """Render the product distribution template with the canonical version."""
    try:
        source = template.read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildError(
            f"cannot read distribution template {template}: {exc}"
        ) from exc
    if "__VERSION__" not in source:
        raise BuildError(
            f"distribution template does not contain __VERSION__: {template}"
        )
    rendered = source.replace("__VERSION__", version)
    if "__VERSION__" in rendered:
        raise BuildError(f"distribution version replacement failed: {template}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise BuildError(f"cannot write distribution {destination}: {exc}") from exc
    return destination


def pkgbuild_command(
    component_root: Path,
    component_package: Path,
    version: str,
    *,
    scripts_dir: Path | None = None,
) -> list[str]:
    """Return the unsigned user-domain component-package command."""
    command = [
        "pkgbuild",
        "--root",
        str(component_root),
        "--identifier",
        PACKAGE_IDENTIFIER,
        "--version",
        version,
        "--install-location",
        USER_INSTALL_LOCATION,
    ]
    if scripts_dir is not None:
        command.extend(("--scripts", str(scripts_dir)))
    command.append(str(component_package))
    return command


def productbuild_command(
    distribution: Path,
    package_path: Path,
    final_package: Path,
) -> list[str]:
    """Return the unsigned product archive command."""
    return [
        "productbuild",
        "--distribution",
        str(distribution),
        "--package-path",
        str(package_path),
        str(final_package),
    ]


def stage_package_scripts(work_dir: Path, source: Path | None) -> Path | None:
    """Stage postinstall when M3 has provided it; otherwise omit scripts."""
    if source is None or not source.is_file():
        return None
    scripts_dir = work_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    destination = scripts_dir / "postinstall"
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o111)
    return scripts_dir


def assemble_package(
    staging_dir: Path,
    work_dir: Path,
    output_dir: Path,
    version: str,
    *,
    distribution_template: Path = DISTRIBUTION_TEMPLATE,
    postinstall_script: Path | None = None,
    uninstall_script: Path | None = None,
    runner: CommandRunner = default_command_runner,
) -> PackageLayout:
    """Assemble payload and invoke the one-way unsigned package toolchain."""
    payload = assemble_payload(
        staging_dir,
        work_dir,
        uninstall_script=uninstall_script,
        runner=runner,
    )
    component_package = work_dir / COMPONENT_PACKAGE_NAME
    scripts_dir = stage_package_scripts(work_dir, postinstall_script)
    _run_checked(
        pkgbuild_command(
            payload.component_root,
            component_package,
            version,
            scripts_dir=scripts_dir,
        ),
        runner,
        "component package build",
    )
    if not component_package.is_file():
        raise BuildError(f"pkgbuild did not create {component_package}")

    distribution = render_distribution(
        distribution_template,
        work_dir / "distribution.xml",
        version,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_package = output_dir / f"XSTARS-{version}.pkg"
    final_package.unlink(missing_ok=True)
    _run_checked(
        productbuild_command(distribution, work_dir, final_package),
        runner,
        "product package build",
    )
    if not final_package.is_file():
        raise BuildError(f"productbuild did not create {final_package}")
    return PackageLayout(
        component_package=component_package,
        distribution=distribution,
        final_package=final_package,
        payload=payload,
    )


def build_package(
    staging_dir: Path,
    version: str,
    *,
    output_dir: Path = OUTPUT_DIR,
    runner: CommandRunner = default_command_runner,
) -> Path:
    """Build a final package while keeping only the requested output."""
    staging_layout(staging_dir)
    with TemporaryDirectory(prefix="xstars-pkg-") as temporary:
        layout = assemble_package(
            staging_dir,
            Path(temporary),
            output_dir,
            version,
            postinstall_script=SCRIPT_DIR / "postinstall.sh",
            uninstall_script=SCRIPT_DIR / "uninstall.sh",
            runner=runner,
        )
        print(f"Unsigned package ready: {layout.final_package}")
        return layout.final_package


def build_parser() -> argparse.ArgumentParser:
    """Build the installer command-line contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--prepare-runtime",
        action="store_true",
        help="download and assemble the relocatable Python runtime staging tree",
    )
    action.add_argument(
        "--build-pkg",
        action="store_true",
        help="build the unsigned user-domain package from prepared staging",
    )
    action.add_argument(
        "--version",
        action="store_true",
        help="print the version from pyproject.toml and exit",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help=f"staging directory (default: {DEFAULT_STAGING_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed installer build CLI."""
    args = build_parser().parse_args(argv)
    if "XSTARS_SIGN" in os.environ:
        print(
            "XSTARS_SIGN is ignored: this installer is intentionally unsigned.",
            file=sys.stderr,
        )

    try:
        version = read_project_version(PROJECT_FILE)
        if args.version:
            print(version)
            return 0
        if args.prepare_runtime:
            print(f"Preparing XSTARS {version} runtime in {args.staging_dir}")
            layout = prepare_runtime(
                args.staging_dir,
                lock_file=LOCK_FILE,
                repo_root=REPO_ROOT,
                assets_dir=ASSETS_DIR,
            )
            print(f"Runtime staging ready: {layout.root}")
            return 0
        if args.build_pkg:
            print(f"Building unsigned XSTARS {version} package")
            build_package(args.staging_dir, version)
            return 0
        raise BuildError("no build action selected")
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
