# XSTARS macOS Installer

This page documents the standalone, per-user XSTARS package for Apple Silicon.
Installation and unsigned-package release guidance will be completed in M5.

## Installation

> M5 documentation placeholder: system requirements and package installation steps.

## Opening the unsigned package

> M5 documentation placeholder: macOS Privacy & Security and command-line options.

## Uninstall XSTARS

Quit Microsoft Excel and WPS before uninstalling. Run the installed script as
the signed-in user; do not use `sudo`:

```bash
"$HOME/Library/Application Support/XSTARS/uninstall.sh"
```

The default is a **dry run**. It prints every planned action and does not change
files. Review that output, then explicitly apply the uninstall:

```bash
"$HOME/Library/Application Support/XSTARS/uninstall.sh" --apply
```

Use `--help` for the complete command summary. The applied uninstall removes:

- `XSTARS.xlam` from Excel's per-user Startup folder;
- the current `xlwings.applescript` and legacy `xstars_launch.scpt` launchers;
- only the `INTERPRETER_MAC` row from `xlwings.conf` (other keys are kept);
- stale Excel `OPEN`/`OPEN1`/… registration values that point to
  `XSTARS.xlam`;
- `~/Library/Application Support/XSTARS/` and the package receipt.

### Registration database backup and recovery

Before changing any Office registration database, the script copies every
`MicrosoftRegistrationDB/*.reg` file to a timestamped directory outside the
installation root:

```text
~/Documents/XSTARS-uninstall-backups/uninstall-backup-<UTC timestamp>-<pid>/
```

After removing XSTARS `OPENn` values, the script runs SQLite
`PRAGMA integrity_check`. It requires the exact result `ok`. If an edit or the
integrity check fails, the script automatically restores every `.reg` file
from that backup and exits with an error.

Keep the backup until Excel has started normally several times. With Excel and
WPS closed, a manual restoration uses the timestamped directory reported by
the script:

```bash
cp -p "$HOME/Documents/XSTARS-uninstall-backups/uninstall-backup-<timestamp>-<pid>/"*.reg \
  "$HOME/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB/"
```

### Preserved user data

Uninstalling deliberately preserves `~/.xstars/`, including `settings.json`
and generated artifacts. Reinstalling XSTARS can reuse this data. Do not remove
that directory unless you independently decide to discard your settings and
artifacts.
