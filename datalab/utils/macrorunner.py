# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Run an editor macro inside a Python or standalone DataLab child process."""

from __future__ import annotations

import linecache
import sys
import traceback
from pathlib import Path


def run(path: str, filename: str) -> None:
    """Execute the UTF-8 editor text in the launcher's actual main module."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        source = Path(path).read_text(encoding="utf-8-sig")
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        code = compile(source, filename, "exec", dont_inherit=True)
        exec(code, vars(sys.modules["__main__"]))
    except (Exception, KeyboardInterrupt) as exc:
        # A windowed PyInstaller bootloader otherwise opens a native error dialog.
        # Report the macro's frames, not this runner's frame, on the captured pipe.
        traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        raise SystemExit(1) from None
