# XSTARS macOS Installer

The unsigned XSTARS `.pkg` is a standalone, per-user installer for Microsoft
Excel on Apple Silicon. It includes Python 3.12, XSTARS, xlwings, and the other
runtime dependencies; users do not need to install Python, create a virtual
environment, or import VBA modules manually.

## System requirements

- Apple Silicon Mac (`arm64`); Intel Macs are not supported by this package.
- macOS 14 Sonoma or later.
- Microsoft Excel for Mac 2016 or later.
- A logged-in graphical user session. Each macOS account that needs XSTARS
  installs its own copy.

WPS for Mac is not supported.

## Installation

Download `XSTARS-1.1.1.pkg` from the project release and quit Excel before
installing. Use one of these methods:

1. Double-click the package and follow Installer. The package selects the
   current-user domain; it does not install a system-wide Python.
2. From Terminal, install explicitly into the current user's home domain:

   ```bash
   sudo installer -pkg "/path/to/XSTARS-1.1.1.pkg" \
     -target CurrentUserHomeDirectory
   ```

The `CurrentUserHomeDirectory` form is the target used by the real-package
layout validation. With the package's `/` install location and relative
payload layout, it places the runtime under the signed-in user's
`~/Library/Application Support/XSTARS/`, not in a system Python. The Terminal
form invokes `sudo`, so macOS asks for an administrator password; normal
graphical installation remains per-user.

After installation, the package deploys the bundled runtime, the XSTARS Excel
startup add-in, the xlwings AppleScript bridge, and the xlwings
`INTERPRETER_MAC` setting. Start Excel and confirm that the **XSTARS** tab is
available. The workbook template remains available inside the XSTARS install
directory under `Templates/XSTARS_mac.xlsm`.

## Opening the unsigned package

This package is intentionally unsigned and not notarized. Only continue when
the package came from a release source you trust. On current macOS releases,
including Sequoia, use one of these three release paths:

1. **Privacy & Security:** after macOS blocks the first launch, open
   **System Settings → Privacy & Security**, find the blocked XSTARS package,
   choose **Open Anyway**, and authenticate with an administrator password.
2. **Remove quarantine from this package:** inspect the path, then run:

   ```bash
   xattr -d com.apple.quarantine "/path/to/XSTARS-1.1.1.pkg"
   ```

   Double-click the package again after the command succeeds.
3. **Install from Terminal:** run either `installer` command from the
   installation section, for example:

   ```bash
   sudo installer -pkg "/path/to/XSTARS-1.1.1.pkg" \
     -target CurrentUserHomeDirectory
   ```

Removing quarantine or using `installer` does not provide signature
verification. Do not use these commands for an untrusted package.

## Known limitations

- The package is Apple Silicon `arm64` only; it is not universal2.
- WPS for Mac is not supported.
- Installation, upgrades, and uninstall preserve `~/.xstars/`, including user
  settings and chart rebuild artifacts.
- The bundled upstream xlwings wheel contains its own `quickstart.xlsm` with
  upstream build-path metadata. This third-party sample is not an XSTARS
  workbook and is not modified during packaging.
- The package is unsigned and not notarized, so transferred downloads can be
  quarantined as described above.

## Uninstall XSTARS

Quit Microsoft Excel and WPS before uninstalling. Run the installed script as
the signed-in user; do not use `sudo`:

```bash
"$HOME/Library/Application Support/XSTARS/uninstall.sh"
```

The command shown above defaults to a **dry run**: it prints every planned
action and does not change files. Review that output, then rerun the same
installed script with `--apply` to perform the uninstall:

```bash
"$HOME/Library/Application Support/XSTARS/uninstall.sh" --apply
```

Use `--help` for the complete command summary. The applied uninstall removes:

- `XSTARS.xlam` from Excel's per-user Startup folder;
- the legacy `xstars_launch.scpt` launcher;
- stale Excel `OPEN`/`OPEN1`/… registration values that point to
  `XSTARS.xlam`;
- `~/Library/Application Support/XSTARS/` and the package receipt.

The shared xlwings files use the pre-install state recorded by the installer:

- when a backup exists, uninstall restores the original
  `xlwings.applescript` or `xlwings.conf` exactly;
- when an absent marker records that the file did not exist before XSTARS,
  uninstall removes `xlwings.applescript` or filters only the
  `INTERPRETER_MAC` row from `xlwings.conf` (preserving other keys);
- when neither a backup nor an absent marker exists, the pre-install state is
  unknown, so uninstall prints a warning and leaves that shared file unchanged.

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
