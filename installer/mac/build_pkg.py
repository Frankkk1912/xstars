#!/usr/bin/env python3
"""Build the unsigned, user-domain XSTARS macOS installer.

Milestone M1 implements the relocatable Python runtime staging pipeline. Later
milestones extend this module with payload and ``.pkg`` assembly.
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
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import tomllib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_FILE = REPO_ROOT / "pyproject.toml"
LOCK_FILE = SCRIPT_DIR / "runtime.lock.json"
ASSETS_DIR = SCRIPT_DIR / "assets"
DEFAULT_STAGING_DIR = SCRIPT_DIR / "staging"
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
    """Paths produced by the M1 runtime staging pipeline."""

    root: Path
    python: Path
    python_executable: Path
    bin: Path


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
        mismatches.append(
            f"sha256 expected={lock.sha256} actual={actual_sha256}"
        )
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
            f"runtime archive did not provide expected interpreter: "
            f"{python_executable}"
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
            f"installed xlwings did not provide xlwings*.applescript under "
            f"{python_dir}"
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


def build_parser() -> argparse.ArgumentParser:
    """Build the M1 command-line contract."""
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
        help="build the unsigned package (implemented in Milestone M2)",
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
    """Run the fail-closed M1 CLI."""
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
        raise BuildError(
            "--build-pkg is reserved for Milestone M2 and is not available yet"
        )
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
