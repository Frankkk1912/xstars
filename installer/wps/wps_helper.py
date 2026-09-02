#!/usr/bin/env python3
"""XSTARS WPS Standalone Installer Helper.

Provides stdlib-only utilities for:
1. bootstrap: ensure cryptographically secure per-install secret and render config.js
2. backup: backup existing WPS jsaddons before installation/modification
3. sync-config: sync rendered config.js with local secret into installed WPS add-in directories
4. install-page: serve the official offline publish page on 127.0.0.1:3890, launch the browser,
   and watch the WPS jsaddons directory to auto-sync config.js as soon as the add-in is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import os
import secrets
import shutil
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "1.0"
DEFAULT_SERVICE_PORT = 3892
DEFAULT_PUBLISH_PORT = 3890


def get_default_config_path() -> Path:
    return Path.home() / ".xstars" / "wps_service.json"


def get_default_jsaddons_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "kingsoft" / "wps" / "jsaddons"
    return Path.home() / "AppData" / "Roaming" / "kingsoft" / "wps" / "jsaddons"


def get_default_backup_base_dir() -> Path:
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        return Path(localappdata) / "XSTARS-WPS" / "backup"
    return Path.home() / "AppData" / "Local" / "XSTARS-WPS" / "backup"


def ensure_service_config(
    config_path: Path, default_port: int = DEFAULT_SERVICE_PORT
) -> tuple[str, int]:
    """Load existing token/port or atomically generate a cryptographically secure token."""
    config_path = config_path.resolve()
    token = None
    port = default_port

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                existing_token = data.get("token")
                if isinstance(existing_token, str) and len(existing_token) >= 32:
                    token = existing_token
                existing_port = data.get("port")
                if isinstance(existing_port, int) and 1 <= existing_port <= 65535:
                    port = existing_port
        except Exception:
            token = None

    if not token:
        token = secrets.token_urlsafe(32)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "token": token,
        "port": port,
    }
    tmp_file = config_path.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_file.replace(config_path)

    return token, port


def render_config_js(template_path: Path, token: str, port: int) -> str:
    """Render config.template.js with port and JSON-encoded token."""
    template_text = template_path.read_text(encoding="utf-8")
    if "<port>" not in template_text or '"<token>"' not in template_text:
        raise ValueError(
            f"Template {template_path} is missing '<port>' or '\"<token>\"' placeholder"
        )
    rendered = template_text.replace("<port>", str(port)).replace(
        '"<token>"', json.dumps(token)
    )
    if "<port>" in rendered or "<token>" in rendered:
        raise ValueError(
            f"Template {template_path} has unresolved placeholders after replacement"
        )
    return rendered


def cmd_bootstrap(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else get_default_config_path()
    port = args.port if args.port else DEFAULT_SERVICE_PORT
    token, port = ensure_service_config(config_path, default_port=port)
    print(f"[BOOTSTRAP] Service config ready at {config_path} (port={port})")

    template_path = Path(args.template) if args.template else None
    if template_path and template_path.is_file():
        rendered = render_config_js(template_path, token, port)
        if args.out:
            for out_str in args.out:
                out_path = Path(out_str)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_out = out_path.with_suffix(f".tmp.{os.getpid()}")
                tmp_out.write_text(rendered, encoding="utf-8")
                tmp_out.replace(out_path)
                print(f"[BOOTSTRAP] Rendered {out_path}")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    jsaddons_dir = (
        Path(args.jsaddons_dir) if args.jsaddons_dir else get_default_jsaddons_dir()
    )
    backup_base = (
        Path(args.backup_base_dir)
        if args.backup_base_dir
        else get_default_backup_base_dir()
    )

    if not jsaddons_dir.is_dir():
        print(
            f"[BACKUP] No WPS jsaddons directory found at {jsaddons_dir}, skipping backup."
        )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_backup = backup_base / f"backup_{timestamp}"
    target_backup.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in jsaddons_dir.iterdir():
        dest = target_backup / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            copied += 1
        elif item.is_file():
            shutil.copy2(item, dest)
            copied += 1

    print(f"[BACKUP] Backed up {copied} item(s) from {jsaddons_dir} to {target_backup}")
    return 0


def sync_rendered_config_to_dirs(
    rendered_content: str,
    jsaddons_dir: Path,
    addon_dir: Path | None = None,
) -> list[Path]:
    """Sync rendered config.js to installed xstars add-in directories."""
    synced_files: list[Path] = []
    target_dirs: list[Path] = []

    if addon_dir:
        if addon_dir.is_dir():
            target_dirs.append(addon_dir)
    elif jsaddons_dir.is_dir():
        for sub in jsaddons_dir.iterdir():
            if sub.is_dir() and (
                sub.name.startswith("xstars-wps-addon")
                or sub.name.startswith("xstars-wps")
            ):
                target_dirs.append(sub)

    for target in target_dirs:
        config_file = target / "config.js"
        needs_write = True
        if config_file.is_file():
            try:
                current_text = config_file.read_text(encoding="utf-8")
                if current_text == rendered_content:
                    needs_write = False
            except Exception:
                needs_write = True

        if needs_write:
            tmp_file = target / f"config.js.tmp.{os.getpid()}"
            tmp_file.write_text(rendered_content, encoding="utf-8")
            tmp_file.replace(config_file)
            synced_files.append(config_file)

    return synced_files


def cmd_sync_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else get_default_config_path()
    jsaddons_dir = (
        Path(args.jsaddons_dir) if args.jsaddons_dir else get_default_jsaddons_dir()
    )
    addon_dir = Path(args.addon_dir) if args.addon_dir else None
    template_path = Path(args.template) if args.template else None

    if not config_path.is_file():
        print(f"[ERROR] Service config not found at {config_path}")
        return 1

    token, port = ensure_service_config(config_path)

    if not template_path or not template_path.is_file():
        # Fallback to inline template if template path not supplied
        inline_template = (
            "(function exposeInjectedConfig(root) {\n"
            "  root.XSTARS_WPS_CONFIG = Object.freeze({\n"
            '    port: Number("<port>"),\n'
            '    token: "<token>",\n'
            "    healthRetries: 3,\n"
            "    retryIntervalMs: 150,\n"
            "  });\n"
            '})(typeof window === "undefined" ? globalThis : window);\n'
        )
        rendered = inline_template.replace("<port>", str(port)).replace(
            '"<token>"', json.dumps(token)
        )
    else:
        rendered = render_config_js(template_path, token, port)

    synced = sync_rendered_config_to_dirs(rendered, jsaddons_dir, addon_dir)
    if synced:
        for f in synced:
            print(f"[SYNC] Synced config.js to {f}")
    else:
        print(
            "[SYNC] All installed xstars add-in configurations are already up-to-date."
        )
    return 0


class LoopbackDeployHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files exclusively from deploy directory and restrict to loopback."""

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Keep console output clean
        pass


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def cmd_install_page(args: argparse.Namespace) -> int:
    deploy_dir = Path(args.dir).resolve()
    if not deploy_dir.is_dir():
        print(f"[ERROR] Deploy directory does not exist: {deploy_dir}")
        return 1

    publish_html = deploy_dir / "publish.html"
    if not publish_html.is_file():
        print(f"[ERROR] publish.html not found in deploy directory: {deploy_dir}")
        return 1

    port = args.port if args.port else DEFAULT_PUBLISH_PORT
    if not is_port_available(port, "127.0.0.1"):
        print(
            f"[ERROR] Port {port} is already in use on 127.0.0.1. Cannot start publish server."
        )
        return 2

    config_path = Path(args.config) if args.config else get_default_config_path()
    token, service_port = ensure_service_config(config_path)

    template_path = (
        Path(args.template)
        if args.template
        else deploy_dir.parent / "config.template.js"
    )
    if template_path.is_file():
        rendered = render_config_js(template_path, token, service_port)
    else:
        inline_template = (
            "(function exposeInjectedConfig(root) {\n"
            "  root.XSTARS_WPS_CONFIG = Object.freeze({\n"
            '    port: Number("<port>"),\n'
            '    token: "<token>",\n'
            "    healthRetries: 3,\n"
            "    retryIntervalMs: 150,\n"
            "  });\n"
            '})(typeof window === "undefined" ? globalThis : window);\n'
        )
        rendered = inline_template.replace("<port>", str(service_port)).replace(
            '"<token>"', json.dumps(token)
        )

    jsaddons_dir = (
        Path(args.jsaddons_dir) if args.jsaddons_dir else get_default_jsaddons_dir()
    )

    # Initial sync in case add-in is already present
    initial_synced = sync_rendered_config_to_dirs(rendered, jsaddons_dir)
    for f in initial_synced:
        print(f"[SYNC] Initial config sync to: {f}")

    handler_class = functools.partial(
        LoopbackDeployHTTPRequestHandler,
        directory=str(deploy_dir.resolve()),
    )

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    except Exception as exc:
        print(f"[ERROR] Failed to bind HTTP server to 127.0.0.1:{port}: {exc}")
        return 2

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}/publish.html"
    print("=" * 60)
    print("  XSTARS for WPS - 加载项离线发布与管理服务")
    print("=" * 60)
    print(f"发布页面 URL  : {url}")
    print(f"部署文件目录  : {deploy_dir}")
    print(f"WPS插件目录   : {jsaddons_dir}")
    print(f"本地服务端口  : {service_port}")
    print("")
    print("说明:")
    print(" 1. 默认浏览器已自动打开 WPS 加载项管理页面。")
    print(" 2. 在页面中找到 'xstars-wps-addon'，点击【安装】或【升级】。")
    print(" 3. 本服务将在后台自动检测安装并注入您本机的安全令牌配置。")
    print(" 4. 安装完成后，重启 WPS 表格即可看到 XSTARS 功能区。")
    print("")
    print(">>> 保持此窗口开启以支持加载项安装；完成后按 Enter 键或 Ctrl+C 退出 <<<")
    print("=" * 60)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[NOTE] Could not launch browser automatically: {exc}")

    stop_event = threading.Event()

    def watcher_loop():
        while not stop_event.is_set():
            try:
                synced = sync_rendered_config_to_dirs(rendered, jsaddons_dir)
                for f in synced:
                    print(f"\n[SYNC] 成功自动同步本地配置到加载项: {f}")
            except Exception as exc:
                print(f"[DEBUG] Watcher sync error: {exc}", file=sys.stderr)
            time.sleep(1.0)

    watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
    watcher_thread.start()

    try:
        # Wait for user input or interrupt
        if sys.stdin and sys.stdin.isatty():
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                line = sys.stdin.readline()
                if not line:  # stdin reached EOF
                    while not stop_event.is_set():
                        time.sleep(0.5)
        else:
            while not stop_event.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
        print("\n[EXIT] 发布服务已停止。")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xstars-wps-helper",
        description="XSTARS WPS Standalone Installer Helper",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_bootstrap = subparsers.add_parser(
        "bootstrap", help="Ensure secret token and render config.js"
    )
    p_bootstrap.add_argument("--config", type=str, help="Path to wps_service.json")
    p_bootstrap.add_argument("--template", type=str, help="Path to config.template.js")
    p_bootstrap.add_argument(
        "--out", type=str, nargs="+", help="Output paths for rendered config.js"
    )
    p_bootstrap.add_argument(
        "--port", type=int, default=DEFAULT_SERVICE_PORT, help="WPS service port"
    )

    # backup
    p_backup = subparsers.add_parser(
        "backup", help="Backup existing WPS jsaddons directory"
    )
    p_backup.add_argument(
        "--jsaddons-dir", type=str, help="Path to WPS jsaddons directory"
    )
    p_backup.add_argument(
        "--backup-base-dir", type=str, help="Base directory to store backups"
    )

    # sync-config
    p_sync = subparsers.add_parser(
        "sync-config", help="Sync config.js into installed WPS add-in directories"
    )
    p_sync.add_argument("--config", type=str, help="Path to wps_service.json")
    p_sync.add_argument("--template", type=str, help="Path to config.template.js")
    p_sync.add_argument(
        "--jsaddons-dir", type=str, help="Path to WPS jsaddons directory"
    )
    p_sync.add_argument(
        "--addon-dir", type=str, help="Specific add-in directory to sync"
    )

    # install-page
    p_install = subparsers.add_parser(
        "install-page", help="Serve publish page on 127.0.0.1:3890 and auto-sync config"
    )
    p_install.add_argument(
        "--dir", type=str, required=True, help="Path to addin deploy directory"
    )
    p_install.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PUBLISH_PORT,
        help="Port to serve publish page",
    )
    p_install.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically",
    )
    p_install.add_argument("--config", type=str, help="Path to wps_service.json")
    p_install.add_argument("--template", type=str, help="Path to config.template.js")
    p_install.add_argument(
        "--jsaddons-dir", type=str, help="Path to WPS jsaddons directory"
    )

    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        return cmd_bootstrap(args)
    if args.command == "backup":
        return cmd_backup(args)
    if args.command == "sync-config":
        return cmd_sync_config(args)
    if args.command == "install-page":
        return cmd_install_page(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
