# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Parent-side driver for ``frozen_smoke_test.py macro`` (runs inside DataLab.exe)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def probe(expected_exe: Path) -> dict:
    """Run macros through the real panel and verify both processes and their output."""
    from guidata.qthelpers import qt_app_context
    from qtpy.QtCore import QCoreApplication, QEventLoop

    import datalab
    from datalab.config.core import ensure_initialized
    from datalab.gui.panel.macro import MacroPanel

    root = Path(sys._MEIPASS).resolve()
    parent = {
        "executable": sys.executable,
        "pid": os.getpid(),
        "frozen": sys.frozen,
        "module": datalab.__file__,
    }
    assert parent["frozen"] and Path(sys.executable).resolve() == expected_exe
    assert root.is_relative_to(expected_exe.parent)
    assert Path(datalab.__file__).resolve().is_relative_to(root)

    def wait(predicate) -> None:
        deadline = time.monotonic() + 45
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError("macro did not reach the expected state")
            QCoreApplication.processEvents(QEventLoop.AllEvents, 25)
            time.sleep(0.01)

    with qt_app_context():
        ensure_initialized(load_user_config=False)
        panel = MacroPanel(None)
        macro = panel.add_macro()
        macro.title = "L'essai été"
        terminal = []
        macro.FINISHED.connect(lambda: terminal.append(macro.get_exit_code()))
        directory = Path(tempfile.gettempdir()).resolve()
        assert directory == Path(os.environ["TEMP"]).resolve()

        def launch(source: str) -> Path:
            macro.set_code(source)
            panel.run_macro()
            path = Path(macro._temp_script_path).resolve()
            assert path.parent == directory and path.is_file()
            return path

        def finish(path: Path, count: int) -> str:
            wait(lambda: len(terminal) == count)
            assert not macro.is_running()
            assert not path.exists() and not list(directory.glob("datalab_macro_*.py"))
            return panel.console.toPlainText()

        payload = "élève '\"\\ →✓ " * 4000
        digest = hashlib.sha256(payload.encode()).hexdigest()
        source = (
            "\ufeff# coding: latin-1\nfrom __future__ import annotations\n"
            "import datalab, hashlib, json, os, pickle\n"
            "class Record: pass\n"
            "assert globals() is vars(sys.modules['__main__'])\n"
            "assert type(pickle.loads(pickle.dumps(Record()))) is Record\n"
            f"payload = {payload!r}\n"
            "print('CHILD ' + json.dumps(dict(pid=os.getpid(), ppid=os.getppid(), "
            "frozen=sys.frozen, executable=sys.executable, module=datalab.__file__, "
            "file=__file__, digest=hashlib.sha256(payload.encode()).hexdigest())))"
        )
        try:
            assert len(source) > 40000
            console = finish(launch(source), 1)
            assert terminal == [0], console
            line = next(
                line for line in console.splitlines() if line.startswith("CHILD ")
            )
            child = json.loads(line.removeprefix("CHILD "))
            assert child["pid"] != parent["pid"] and child["ppid"] == parent["pid"]
            assert (
                child["frozen"] and Path(child["executable"]).resolve() == expected_exe
            )
            assert Path(child["module"]).resolve().is_relative_to(root)
            assert Path(child["file"]).resolve().parent == root
            assert Path(child["file"]).stem == "start" and child["digest"] == digest

            instruction = "raise ValueError('frozen_line_three')"
            console = finish(launch("a = 1\nb = 2\n" + instruction), 2)
            frame = f'File "<macro:{macro.title}>", line 3, in <module>'
            assert terminal[-1] == 1 and frame in console, console
            assert instruction in console and "ValueError: frozen_line_three" in console

            path = launch(
                "import time\nprint('STOP_READY', flush=True)\ntime.sleep(90)"
            )
            wait(lambda: "STOP_READY" in panel.console.toPlainText())
            panel.stop_macro()
            finish(path, 3)
            assert terminal[-1] not in (0, None)
        finally:
            panel.remove_all_objects()
        return {
            "parent": parent,
            "child": child,
            "source_characters": len(source),
            "payload_sha256": digest,
            "traceback_frame": frame,
            "terminal_exit_codes": terminal,
            "tempdir": str(directory),
            "remaining_scripts": list(map(str, directory.glob("datalab_macro_*.py"))),
        }


def main(executable: str, report: str) -> None:
    """Always leave a diagnostic for this windowed executable."""
    try:
        result = probe(Path(executable).resolve())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
    Path(report).write_text(json.dumps(result, indent=2), encoding="utf-8")
