#!/bin/bash
# Remove the per-user XSTARS runtime and Excel integration files.
# The default mode is a non-destructive preview; pass --apply to make changes.

set -eu

MODE="dry-run"
MODE_OPTION=""
PACKAGE_IDENTIFIER="com.frank-sysu.xstars"

usage() {
    cat <<'EOF'
Usage: uninstall.sh [--dry-run | --apply] [--help]

By default, print the uninstall actions without changing any files.
  --dry-run  Preview every cleanup step (default).
  --apply    Perform the cleanup. Quit Excel and WPS before using this option.
  --help     Show this help text.

The uninstall preserves XSTARS settings and generated artifacts. Before editing
MicrosoftRegistrationDB, --apply stores a timestamped backup under Documents.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
    --dry-run | --apply)
        if [ -n "$MODE_OPTION" ]; then
            echo "uninstall: specify --dry-run or --apply only once" >&2
            usage >&2
            exit 2
        fi
        MODE_OPTION=$1
        if [ "$1" = "--apply" ]; then
            MODE="apply"
        fi
        ;;
    --help | -h)
        usage
        exit 0
        ;;
    *)
        echo "uninstall: unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
done

USER_HOME=${HOME:?"HOME is not set"}
INSTALL_ROOT="$USER_HOME/Library/Application Support/XSTARS"
EXCEL_STARTUP_ADDIN="$USER_HOME/Library/Group Containers/UBF8T346G9.Office/User Content.localized/Startup.localized/Excel/XSTARS.xlam"
APP_SCRIPTS_DIR="$USER_HOME/Library/Application Scripts/com.microsoft.Excel"
LEGACY_LAUNCH_SCRIPT="$APP_SCRIPTS_DIR/xstars_launch.scpt"
XLWINGS_APPLESCRIPT="$APP_SCRIPTS_DIR/xlwings.applescript"
XLWINGS_CONF="$USER_HOME/Library/Containers/com.microsoft.Excel/Data/xlwings.conf"
PREINSTALL_BACKUP_DIR="$INSTALL_ROOT/preinstall-backup"
APPLESCRIPT_BACKUP="$PREINSTALL_BACKUP_DIR/xlwings.applescript"
CONF_BACKUP="$PREINSTALL_BACKUP_DIR/xlwings.conf"
REGISTRATION_DIR="$USER_HOME/Library/Group Containers/UBF8T346G9.Office/MicrosoftRegistrationDB"
BACKUP_PARENT="$USER_HOME/Documents/XSTARS-uninstall-backups"

log() {
    echo "uninstall: $*"
}

warn() {
    echo "uninstall: warning: $*" >&2
}

applications_running() {
    local process
    for process in "Microsoft Excel" "WPS Office" "wpsoffice"; do
        if pgrep -x "$process" >/dev/null 2>&1; then
            warn "$process is running; quit it before uninstalling"
            return 0
        fi
    done
    return 1
}

print_dry_run() {
    log "dry-run only; no files will be changed (pass --apply to proceed)"
    log "would remove Excel Startup add-in: $EXCEL_STARTUP_ADDIN"
    log "would remove legacy launch script: $LEGACY_LAUNCH_SCRIPT"
    log "would restore pre-install xlwings AppleScript when backed up; otherwise remove: $XLWINGS_APPLESCRIPT"
    log "would restore pre-install xlwings configuration when backed up; otherwise remove only INTERPRETER_MAC from: $XLWINGS_CONF"
    log "would back up MicrosoftRegistrationDB .reg files under: $BACKUP_PARENT"
    log "would delete Excel OPEN/OPENn values that point to XSTARS.xlam"
    log "would require PRAGMA integrity_check to return ok; otherwise restore backup"
    log "would remove installed runtime: $INSTALL_ROOT"
    log "would run: pkgutil --forget $PACKAGE_IDENTIFIER"
    log "settings and generated artifacts are preserved"
}

if [ "$MODE" = "dry-run" ]; then
    print_dry_run
    exit 0
fi

if [ "$(/usr/bin/id -u)" -eq 0 ]; then
    echo "uninstall: do not run --apply with sudo; run as the installed user" >&2
    exit 1
fi

if applications_running; then
    echo "uninstall: refusing to continue while Excel or WPS is running" >&2
    exit 1
fi

remove_file_if_present() {
    local path=$1
    local label=$2
    if [ -e "$path" ] || [ -L "$path" ]; then
        log "removing $label: $path"
        /bin/rm -f "$path"
    else
        log "$label not present: $path"
    fi
}

remove_file_if_present "$EXCEL_STARTUP_ADDIN" "Excel Startup add-in"
remove_file_if_present "$LEGACY_LAUNCH_SCRIPT" "legacy launch script"

restore_xlwings_applescript() {
    if [ -f "$APPLESCRIPT_BACKUP" ]; then
        /bin/mkdir -p "$APP_SCRIPTS_DIR"
        /bin/cp -p "$APPLESCRIPT_BACKUP" "$XLWINGS_APPLESCRIPT"
        log "restored pre-install xlwings AppleScript"
    else
        remove_file_if_present "$XLWINGS_APPLESCRIPT" "xlwings AppleScript"
    fi
}

restore_xlwings_applescript

remove_interpreter_mac() {
    local temporary
    if [ ! -f "$XLWINGS_CONF" ]; then
        log "xlwings configuration not present: $XLWINGS_CONF"
        return 0
    fi

    temporary=$(/usr/bin/mktemp "${TMPDIR:-/tmp}/xstars-uninstall-conf.XXXXXX")
    if ! /usr/bin/awk '
        toupper($0) !~ /^[[:space:]]*"?INTERPRETER_MAC"?[[:space:]]*,/ { print }
    ' "$XLWINGS_CONF" >"$temporary"; then
        /bin/rm -f "$temporary"
        echo "uninstall: cannot filter $XLWINGS_CONF" >&2
        return 1
    fi

    if /usr/bin/grep -q '[^[:space:]]' "$temporary"; then
        /bin/chmod 600 "$temporary"
        /bin/mv -f "$temporary" "$XLWINGS_CONF"
        log "removed INTERPRETER_MAC and preserved other xlwings settings"
    else
        /bin/rm -f "$temporary" "$XLWINGS_CONF"
        log "removed empty xlwings configuration after deleting INTERPRETER_MAC"
    fi
}

restore_xlwings_conf() {
    if [ -f "$CONF_BACKUP" ]; then
        /bin/mkdir -p "$(/usr/bin/dirname "$XLWINGS_CONF")"
        /bin/cp -p "$CONF_BACKUP" "$XLWINGS_CONF"
        log "restored pre-install xlwings configuration"
    else
        remove_interpreter_mac
    fi
}

restore_xlwings_conf

BACKUP_DIR=""
restore_registration_backup() {
    local backup
    [ -n "$BACKUP_DIR" ] || return 0
    warn "restoring MicrosoftRegistrationDB from $BACKUP_DIR"
    for backup in "$BACKUP_DIR"/*.reg; do
        [ -f "$backup" ] || continue
        /bin/cp -p "$backup" "$REGISTRATION_DIR/$(/usr/bin/basename "$backup")"
    done
}

clean_registration_database() {
    local database
    local backup
    local table_present
    local result
    local integrity
    local found=false
    local timestamp

    for database in "$REGISTRATION_DIR"/*.reg; do
        [ -f "$database" ] || continue
        found=true
        break
    done
    if [ "$found" = false ]; then
        log "MicrosoftRegistrationDB .reg files not present: $REGISTRATION_DIR"
        return 0
    fi
    if ! command -v sqlite3 >/dev/null 2>&1; then
        echo "uninstall: sqlite3 is required to clean Excel OPEN/OPENn values" >&2
        return 1
    fi

    timestamp=$(/bin/date -u '+%Y%m%dT%H%M%SZ')
    BACKUP_DIR="$BACKUP_PARENT/uninstall-backup-$timestamp-$$"
    /bin/mkdir -p "$BACKUP_DIR"
    for database in "$REGISTRATION_DIR"/*.reg; do
        [ -f "$database" ] || continue
        backup="$BACKUP_DIR/$(/usr/bin/basename "$database")"
        /bin/cp -p "$database" "$backup"
    done
    log "backed up MicrosoftRegistrationDB to: $BACKUP_DIR"

    for database in "$REGISTRATION_DIR"/*.reg; do
        [ -f "$database" ] || continue
        if ! table_present=$(sqlite3 "$database" \
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='HKEY_CURRENT_USER_values';"); then
            restore_registration_backup
            echo "uninstall: cannot inspect registration database: $database" >&2
            return 1
        fi
        if [ "$table_present" != "1" ]; then
            log "registration table absent; leaving unchanged: $database"
            continue
        fi

        # OPEN/OPENn are Excel Options auto-load values. The XSTARS.xlam value
        # match (including common BLOB encodings) prevents unrelated OPEN values
        # from being removed from the shared Office registration database.
        if ! result=$(
            sqlite3 "$database" <<'SQL'
.bail on
BEGIN IMMEDIATE;
DELETE FROM HKEY_CURRENT_USER_values
 WHERE (UPPER(name) = 'OPEN' OR UPPER(name) GLOB 'OPEN[0-9]*')
   AND (
       UPPER(CAST(value AS TEXT)) LIKE '%XSTARS.XLAM%'
       OR UPPER(HEX(value)) LIKE '%5853544152532E584C414D%'
       OR UPPER(HEX(value)) LIKE '%5800530054004100520053002E0058004C0041004D00%'
   );
COMMIT;
PRAGMA integrity_check;
SQL
        ); then
            restore_registration_backup
            echo "uninstall: registration cleanup failed; backup restored" >&2
            return 1
        fi
        integrity=$(printf '%s\n' "$result" | /usr/bin/tail -n 1)
        if [ "$integrity" != "ok" ]; then
            restore_registration_backup
            echo "uninstall: integrity_check returned '$integrity'; backup restored" >&2
            return 1
        fi
        log "cleaned XSTARS Excel OPEN/OPENn values; integrity_check ok: $database"
    done
}

clean_registration_database

if [ -d "$INSTALL_ROOT" ]; then
    log "removing installed runtime: $INSTALL_ROOT"
    /bin/rm -rf "$INSTALL_ROOT"
else
    log "installed runtime not present: $INSTALL_ROOT"
fi

if /usr/sbin/pkgutil --pkg-info "$PACKAGE_IDENTIFIER" >/dev/null 2>&1; then
    log "forgetting package receipt: $PACKAGE_IDENTIFIER"
    /usr/sbin/pkgutil --forget "$PACKAGE_IDENTIFIER"
else
    log "package receipt not present: $PACKAGE_IDENTIFIER"
fi

log "uninstall complete"
if [ -n "$BACKUP_DIR" ]; then
    log "registration backup retained at: $BACKUP_DIR"
fi
exit 0
