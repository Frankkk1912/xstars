"""CLI entry point for frozen (PyInstaller) distribution.

VBA calls via Shell:
    Shell ExePath() & " run_quick " & Chr(34) & ActiveWorkbook.FullName & Chr(34), vbHide

This module:
  1. Reads the workbook path from argv and sets it as the mock caller
  2. Dispatches the command to the corresponding function in main.py
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: xstars.exe <command> <workbook_path>"
        )

    command = sys.argv[1].strip().strip('"').strip("'")
    workbook_path = sys.argv[2].strip().strip('"').strip("'")

    # Connect to the calling workbook and register it as Book.caller()
    import xlwings as xw

    book = xw.Book(workbook_path)
    book.set_mock_caller()

    from xstars import main as ep_main

    func = getattr(ep_main, command, None)
    if func is None:
        raise SystemExit(f"Unknown command: {command}")
    func()


if __name__ == "__main__":
    main()
