"""CLI entry point for frozen (PyInstaller) distribution.

VBA calls via Shell:
    Shell ExePath() & " run_quick " & Chr(34) & ActiveWorkbook.FullName & Chr(34), vbHide

This module:
  1. Reads the workbook path from argv and sets it as the mock caller
  2. Dispatches the command to the corresponding function in main.py
"""

from __future__ import annotations

import sys
from importlib import import_module


def _run_serve_mode(arguments: list[str]) -> int:
    import argparse

    service = import_module("xstars.wps_service")

    parser = argparse.ArgumentParser(prog="xstars serve")
    parser.add_argument("--port", type=int, default=service.DEFAULT_PORT)
    options = parser.parse_args(arguments)
    return service.serve(options.port)


def _run_worker_mode(arguments: list[str]) -> int:
    import argparse

    worker = import_module("xstars.application.worker")

    parser = argparse.ArgumentParser(prog="xstars worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    options = parser.parse_args(arguments)
    return worker.run_worker(options.request, options.result)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "serve":
        raise SystemExit(_run_serve_mode(sys.argv[2:]))
    if mode == "worker":
        raise SystemExit(_run_worker_mode(sys.argv[2:]))

    if len(sys.argv) < 3:
        raise SystemExit("Usage: xstars.exe <command> <workbook_path>")

    command = sys.argv[1].strip().strip('"').strip("'")
    workbook_path = sys.argv[2].strip().strip('"').strip("'")

    # Connect to the calling workbook and register it as Book.caller()
    xw = import_module("xlwings")

    book = xw.Book(workbook_path)
    book.set_mock_caller()

    from xstars import main as ep_main

    func = getattr(ep_main, command, None)
    if func is None:
        raise SystemExit(f"Unknown command: {command}")
    func()


if __name__ == "__main__":
    main()
