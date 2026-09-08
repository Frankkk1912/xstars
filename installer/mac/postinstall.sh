#!/bin/bash
# Deploy the per-user XSTARS runtime and Excel integration files.
# Package scripts may run as root even for CurrentUserHomeDirectory installs,
# so all user-facing files are resolved through the active console account.

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONF_TEMPLATE="$SCRIPT_DIR/xlwings.conf"
EXPECTED_CONF_LINE='"INTERPRETER_MAC","$HOME/Library/Application Support/XSTARS/python/bin/python3"'

warn() {
    echo "postinstall: warning: $*" >&2
}

console_user() {
    local user
    user=$(/usr/bin/stat -f '%Su' /dev/console 2>/dev/null || true)
    case "$user" in
    "" | root | loginwindow | _mbsetupuser)
        user=$(
            /usr/sbin/scutil 2>/dev/null <<'EOF' |
show State:/Users/ConsoleUser
EOF
                /usr/bin/awk '/Name :/ { print $3; exit }'
        )
        ;;
    esac
    case "$user" in
    "" | root | loginwindow | _mbsetupuser) return 1 ;;
    *) printf '%s\n' "$user" ;;
    esac
}

if ! ACTIVE_USER=$(console_user); then
    warn "no active GUI user; skipping per-user deployment (install while logged in)"
    exit 0
fi

USER_HOME=$(/usr/bin/dscl . -read "/Users/$ACTIVE_USER" NFSHomeDirectory 2>/dev/null |
    /usr/bin/sed 's/^NFSHomeDirectory: //')
case "$USER_HOME" in
/*) ;;
*)
    warn "cannot resolve home directory for $ACTIVE_USER; skipping per-user deployment"
    exit 0
    ;;
esac

INSTALL_ROOT="$USER_HOME/Library/Application Support/XSTARS"
PAYLOAD_ARCHIVE="$INSTALL_ROOT/XSTARS-payload.tar.gz"
PYTHON_EXECUTABLE="$INSTALL_ROOT/python/bin/python3"
EXCEL_STARTUP="$USER_HOME/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel"
APP_SCRIPTS_DIR="$USER_HOME/Library/Application Scripts/com.microsoft.Excel"
XLWINGS_CONF_DIR="$USER_HOME/Library/Containers/com.microsoft.Excel/Data"
XLWINGS_CONF="$XLWINGS_CONF_DIR/xlwings.conf"
PREINSTALL_BACKUP_DIR="$INSTALL_ROOT/preinstall-backup"
APPLESCRIPT_BACKUP="$PREINSTALL_BACKUP_DIR/xlwings.applescript"
APPLESCRIPT_ABSENT_MARKER="$PREINSTALL_BACKUP_DIR/.xlwings.applescript.absent"
CONF_BACKUP="$PREINSTALL_BACKUP_DIR/xlwings.conf"
CONF_ABSENT_MARKER="$PREINSTALL_BACKUP_DIR/.xlwings.conf.absent"

run_as_user() {
    if [ "$(/usr/bin/id -u)" -eq 0 ]; then
        /usr/bin/sudo -u "$ACTIVE_USER" "$@"
    else
        "$@"
    fi
}

# The component payload is a tarball to prevent Sequoia provenance xattrs from
# becoming AppleDouble companions. Reinstalling overlays the same paths.
if [ -f "$PAYLOAD_ARCHIVE" ]; then
    /bin/mkdir -p "$USER_HOME/Library/Application Support"
    COPYFILE_DISABLE=1 /usr/bin/tar -xzf "$PAYLOAD_ARCHIVE" \
        -C "$USER_HOME/Library/Application Support"
    /bin/rm -f "$PAYLOAD_ARCHIVE"
elif [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "postinstall: payload and runtime are missing under $INSTALL_ROOT" >&2
    exit 1
fi

if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "postinstall: runtime extraction did not create $PYTHON_EXECUTABLE" >&2
    exit 1
fi

# Root-context package scripts must not leave the user-owned runtime root:wheel.
if [ "$(/usr/bin/id -u)" -eq 0 ]; then
    ACTIVE_GROUP=$(/usr/bin/id -gn "$ACTIVE_USER" 2>/dev/null || printf 'staff')
    if ! /usr/sbin/chown -R "$ACTIVE_USER:$ACTIVE_GROUP" "$INSTALL_ROOT"; then
        warn "could not correct ownership under $INSTALL_ROOT"
    fi
fi

copy_user_file() {
    local source=$1
    local destination_dir=$2
    local destination_name=$3
    local label=$4
    local destination="$destination_dir/$destination_name"

    if [ ! -f "$source" ]; then
        warn "$label source is missing: $source"
        return 1
    fi
    if ! run_as_user /bin/mkdir -p "$destination_dir"; then
        warn "cannot create $label directory: $destination_dir"
        return 1
    fi
    if ! run_as_user /bin/cp -f "$source" "$destination"; then
        warn "cannot deploy $label to $destination"
        return 1
    fi
    run_as_user /bin/chmod 644 "$destination" || warn "cannot chmod $destination"
    if [ "$(/usr/bin/id -u)" -eq 0 ]; then
        /usr/sbin/chown "$ACTIVE_USER:$ACTIVE_GROUP" "$destination" ||
            warn "cannot chown $destination"
    fi
    echo "postinstall: deployed $label to $destination"
    return 0
}

backup_user_file_once() {
    local target=$1
    local backup=$2
    local absent_marker=$3
    local label=$4

    if [ -f "$backup" ] || [ -f "$absent_marker" ]; then
        return 0
    fi
    if ! run_as_user /bin/mkdir -p "$PREINSTALL_BACKUP_DIR"; then
        warn "cannot create pre-install backup directory: $PREINSTALL_BACKUP_DIR"
        return 1
    fi
    if [ -f "$target" ]; then
        if ! run_as_user /bin/cp -p "$target" "$backup"; then
            warn "cannot back up existing $label: $target"
            return 1
        fi
        echo "postinstall: backed up existing $label to $backup"
    elif ! run_as_user /usr/bin/touch "$absent_marker"; then
        warn "cannot record that $label was absent before installation"
        return 1
    fi
    return 0
}

# Both integrations are best-effort: sandbox/TCC denial must not corrupt an
# otherwise usable runtime installation. Preserve shared xlwings state once,
# so an uninstall can restore the exact files that predated XSTARS.
copy_user_file \
    "$INSTALL_ROOT/bin/XSTARS.xlam" \
    "$EXCEL_STARTUP" \
    "XSTARS.xlam" \
    "Excel add-in" || true
if backup_user_file_once \
    "$APP_SCRIPTS_DIR/xlwings.applescript" \
    "$APPLESCRIPT_BACKUP" \
    "$APPLESCRIPT_ABSENT_MARKER" \
    "xlwings AppleScript"; then
    copy_user_file \
        "$INSTALL_ROOT/bin/xlwings.applescript" \
        "$APP_SCRIPTS_DIR" \
        "xlwings.applescript" \
        "xlwings AppleScript" || true
else
    warn "leaving the existing xlwings AppleScript unchanged"
fi

merge_xlwings_conf() {
    local configured_line
    local temporary

    if [ ! -f "$CONF_TEMPLATE" ]; then
        warn "generated xlwings configuration is missing: $CONF_TEMPLATE"
        return 1
    fi
    configured_line=$(/bin/cat "$CONF_TEMPLATE")
    if [ "$configured_line" != "$EXPECTED_CONF_LINE" ]; then
        warn "generated xlwings configuration has unexpected content"
        return 1
    fi
    if ! run_as_user /bin/mkdir -p "$XLWINGS_CONF_DIR"; then
        warn "cannot create xlwings configuration directory: $XLWINGS_CONF_DIR"
        return 1
    fi

    if ! temporary=$(/usr/bin/mktemp "${TMPDIR:-/tmp}/xstars-xlwings-conf.XXXXXX"); then
        warn "cannot create temporary xlwings configuration"
        return 1
    fi
    if [ -f "$XLWINGS_CONF" ]; then
        if ! run_as_user /usr/bin/awk -v replacement="$configured_line" '
            function is_interpreter(line) {
                return toupper(line) ~ /^[[:space:]]*"?INTERPRETER_MAC"?[[:space:]]*,/
            }
            BEGIN { replaced = 0 }
            is_interpreter($0) {
                if (!replaced) {
                    print replacement
                    replaced = 1
                }
                next
            }
            { print }
            END {
                if (!replaced) print replacement
            }
        ' "$XLWINGS_CONF" >"$temporary"; then
            /bin/rm -f "$temporary"
            warn "cannot merge existing xlwings configuration"
            return 1
        fi
    else
        if ! /usr/bin/printf '%s\n' "$configured_line" >"$temporary"; then
            /bin/rm -f "$temporary"
            warn "cannot render xlwings configuration"
            return 1
        fi
    fi
    /bin/chmod 644 "$temporary"
    if ! run_as_user /bin/cp -f "$temporary" "$XLWINGS_CONF"; then
        /bin/rm -f "$temporary"
        warn "cannot deploy xlwings configuration to $XLWINGS_CONF"
        return 1
    fi
    /bin/rm -f "$temporary"
    if [ "$(/usr/bin/id -u)" -eq 0 ]; then
        /usr/sbin/chown "$ACTIVE_USER:$ACTIVE_GROUP" "$XLWINGS_CONF" ||
            warn "cannot chown $XLWINGS_CONF"
    fi
    echo "postinstall: updated INTERPRETER_MAC in $XLWINGS_CONF"
    return 0
}

if backup_user_file_once \
    "$XLWINGS_CONF" \
    "$CONF_BACKUP" \
    "$CONF_ABSENT_MARKER" \
    "xlwings configuration"; then
    merge_xlwings_conf || true
else
    warn "leaving the existing xlwings configuration unchanged"
fi

echo "postinstall: done"
exit 0
