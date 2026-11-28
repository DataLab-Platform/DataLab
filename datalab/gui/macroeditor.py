# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro editor
============

The :mod:`datalab.gui.macroeditor` module provides the macro editor widget for DataLab.

.. autoclass:: Macro
"""

from __future__ import annotations

import abc
import os
import os.path as osp
import re
import sys
import tempfile
import time
import uuid

from guidata.io import BaseIOHandler
from guidata.utils.misc import to_string
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

import datalab
from datalab.config import _
from datalab.env import execenv
from datalab.gui import ObjItf
from datalab.utils import macrorecovery
from datalab.widgets.codecompleter import MacroCodeEditor

UNTITLED_NB = 0

# Debounce delay before persisting a modified macro to the recovery cache.
_AUTOSAVE_DELAY_MS = 500


class MacroMeta(type(QC.QObject), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class Macro(QC.QObject, ObjItf, metaclass=MacroMeta):
    """Object representing a macro: editor, path, open/save actions, etc.

    Args:
        console: Python shell widget
        name: Macro name. Defaults to None.
    """

    PREFIX = "m"

    STARTED = QC.Signal()
    FINISHED = QC.Signal()
    MODIFIED = QC.Signal()
    FILE_HEADER = os.linesep.join(
        [
            "# -*- coding: utf-8 -*-",
            "",
            '''"""
DataLab Macro: "%s"
-------------

This file is a DataLab macro. It can be executed from DataLab's Macro Panel, or
from any Python environment, provided that the ``datalab`` package is installed.

Please do not modify this file header. It is used to identify the file as a
DataLab macro, and to store the macro's title.
"""''',
            "",
            "",
        ]
    )
    MACRO_TITLE = _("Macro simple example")
    MACRO_SAMPLE = f"""# {MACRO_TITLE}

import numpy as np

from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.calc("fft")

print("All done!")
"""

    def __init__(self, console: PythonShellWidget, title: str | None = None) -> None:
        super().__init__()
        self.console = console
        self.title = self.get_untitled_title() if title is None else title
        self.uid = uuid.uuid4().hex
        self.editor = MacroCodeEditor()
        self.completer = self.editor.completer
        self.editor.setLineWrapMode(QW.QPlainTextEdit.NoWrap)
        self.set_code(self.MACRO_SAMPLE)
        self.editor.modificationChanged.connect(self.modification_changed)
        self._autosave_timer = QC.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(_AUTOSAVE_DELAY_MS)
        self._autosave_timer.timeout.connect(self._autosave_pending)
        self.process: QC.QProcess | None = None
        self._stopping = False
        self._temp_script_path: str | None = None
        self.__last_exit_code = None

    @property
    def title(self) -> str:
        """Return object title"""
        return self.objectName()

    @title.setter
    def title(self, title: str) -> None:
        """Set object title"""
        self.setObjectName(title)

    def get_code(self) -> str:
        """Return code to be executed"""
        text = self.editor.toPlainText()
        return os.linesep.join(text.splitlines(False))

    def set_code(self, code: str) -> None:
        """Set code to be executed

        Args:
            code: Code to be executed
        """
        self.editor.setPlainText(code)
        # ``setPlainText`` leaves the document in the *modified* state, which
        # would suppress the very first ``modificationChanged(True)`` emission
        # on a real user edit and thus break the autosave/recovery cache.
        self.editor.document().setModified(False)

    def serialize(self, writer: BaseIOHandler) -> None:
        """Serialize this macro

        Args:
            writer: Writer
        """
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("contents"):
            writer.write(self.get_code())
        with writer.group("uid"):
            writer.write(self.uid)

    def deserialize(self, reader: BaseIOHandler) -> None:
        """Deserialize this macro

        Args:
            reader: Reader
        """
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("contents"):
            self.set_code(reader.read_any())
        # Backward-compat: ``uid`` was added in v1.4 (DataLab-Web writes none);
        # fall back to a fresh UUID. The section is closed even when missing.
        reader.begin("uid")
        try:
            stored_uid = reader.read_any()
        except Exception:  # pylint: disable=broad-except
            stored_uid = None
        finally:
            reader.end("uid")
        if stored_uid:
            self.uid = stored_uid

    def to_file(self, filename: str) -> None:
        """Save macro to file

        Args:
            filename: File name
        """
        code = self.FILE_HEADER % self.title + self.get_code()
        with open(filename, "wb") as fdesc:
            fdesc.write(code.encode("utf-8"))

    def from_file(self, filename: str) -> None:
        """Load macro from file

        Args:
            filename: File name
        """
        with open(filename, "rb") as fdesc:
            code = to_string(fdesc.read()).strip()

        # Retrieve title from header:
        lines = code.splitlines()
        for line in lines:
            # Match a line exactly like 'DataLab Macro: "Macro title"':
            if re.match(r"DataLab Macro: \".*\"", line):
                self.title = line.split('"')[1]
                break
        else:
            self.title = osp.basename(filename)

        # Remove header:
        header = (self.FILE_HEADER % self.title).strip()
        if code.startswith(header):
            code = code[len(header) :].strip()

        # Set code:
        self.set_code(code)

    @staticmethod
    def get_untitled_title() -> str:
        """Increment untitled number and return untitled macro title

        Returns:
            Untitled macro title
        """
        global UNTITLED_NB  # pylint: disable=global-statement
        UNTITLED_NB += 1
        return f"macro_{UNTITLED_NB:02d}"

    @staticmethod
    def set_untitled_number(number: int) -> None:
        """Set the untitled number counter

        This is useful when loading macros from HDF5 or files to ensure
        that the next untitled macro has a unique name.

        Args:
            number: New untitled number
        """
        global UNTITLED_NB  # pylint: disable=global-statement
        UNTITLED_NB = number

    def modification_changed(self, state: bool) -> None:
        """Method called when macro's editor modification state changed

        Args:
            state: Modification state
        """
        if state:
            self.MODIFIED.emit()
            self._autosave_timer.start()

    def _autosave_pending(self) -> None:
        """Persist the current code to the recovery cache (best-effort)."""
        try:
            macrorecovery.save_pending(self.uid, self.title, self.get_code())
        except OSError:
            pass

    def flush_autosave(self) -> None:
        """Force an immediate write to the recovery cache (if pending)."""
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        self._autosave_pending()

    def clear_autosave(self) -> None:
        """Remove this macro from the recovery cache."""
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        try:
            macrorecovery.clear_pending(self.uid)
        except OSError:
            pass

    @staticmethod
    def transcode(bytearr: QC.QByteArray) -> str:
        """Transcode bytes to locale str

        Args:
            bytearr: Byte array

        Returns:
            Locale str
        """
        # Python 3 outputs UTF-8 by default, so we need to decode as UTF-8
        # instead of using the system locale codec (which might be cp1252 on Windows)
        return bytes(bytearr).decode("utf-8", errors="replace")

    def get_stdout(self) -> str:
        """Return standard output str

        Returns:
            Standard output str
        """
        assert self.process is not None
        return self.transcode(self.process.readAllStandardOutput())

    def get_stderr(self) -> str:
        """Return standard error str

        Returns:
            Standard error str
        """
        assert self.process is not None
        return self.transcode(self.process.readAllStandardError())

    def write_output(self) -> None:
        """Write text as standard output"""
        self.console.write(self.get_stdout())

    def write_error(self) -> None:
        """Write text as standard error"""
        self.console.write_error(self.get_stderr())

    def print(self, text, error=False, eol_before=True) -> None:
        """Print text in console, with line separator

        Args:
            text: Text to be printed
            error: Print as error. Defaults to False.
        """
        msg = f"---({time.ctime()})---[{text}]{os.linesep}"
        if eol_before:
            msg = os.linesep + msg
        self.console.write(msg, flush=error, error=error, prompt=not error)

    def _cleanup_temp_script(self) -> None:
        """Remove the temporary script file created by ``run`` if it still exists."""
        path = self._temp_script_path
        if path is None:
            return
        for attempt in range(2):
            try:
                os.remove(path)
                break
            except FileNotFoundError:
                break
            except OSError as exc:
                # Windows scanners may briefly keep the closed file open.
                if attempt == 0:
                    time.sleep(0.02)
                else:
                    self.print(
                        _("Unable to remove temporary macro file '%s': %s")
                        % (path, exc),
                        error=True,
                    )
        self._temp_script_path = None

    def _finish_run(self, process: QC.QProcess) -> None:
        """Release this run before notifying slots that may start another one."""
        if self.process is not process:
            return
        self._cleanup_temp_script()
        process.deleteLater()
        self.process = None
        self.FINISHED.emit()

    def run(self) -> None:
        """Run macro"""
        if self.process is not None or self._stopping:
            self.print(_("Macro '%s' is already running") % self.title, error=True)
            return
        process = self.process = QC.QProcess(self)
        self.__last_exit_code = None
        self.flush_autosave()
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix="datalab_macro_", suffix=".py", delete=False
            ) as fdesc:
                self._temp_script_path = fdesc.name
                fdesc.write(self.get_code().encode("utf-8"))
        except (OSError, UnicodeError) as exc:
            message = _("Unable to prepare '%s' macro: %s") % (self.title, exc)
            self.print(message, error=True)
            QW.QMessageBox.critical(self.console, _("Error"), message)
            self._finish_run(process)
            return
        datalab_path = osp.abspath(osp.join(osp.dirname(datalab.__file__), os.pardir))
        filename = "<macro:" + " ".join(self.title[:128].splitlines()) + ">"
        bootstrap = (
            f"import sys; sys.path.append({datalab_path!r}); "
            "from datalab.utils.macrorunner import run as _run_macro; "
            f"_run_macro({self._temp_script_path!r}, {filename!r})"
        )
        env = QC.QProcessEnvironment.systemEnvironment()
        env.insert(execenv.XMLRPCPORT_ENV, str(execenv.xmlrpcport))
        process.readyReadStandardOutput.connect(self.write_output)
        process.readyReadStandardError.connect(self.write_error)
        process.started.connect(self.started)
        process.errorOccurred.connect(self.error_occurred)
        process.finished.connect(self.finished)
        process.setProcessEnvironment(env)
        args = ["-c", bootstrap]
        # The process starts asynchronously, without a timeout: its start may be
        # slow, e.g. when an antivirus scans the standalone executable.
        process.start(sys.executable, args)
        # On Windows, a start failure is reported (and FINISHED emitted) by
        # `error_occurred` before `start` returns
        if self.process is process:
            self.STARTED.emit()

    def started(self) -> None:
        """Process has started"""
        self.print(_("# ==> Running '%s' macro...") % self.title)

    def error_occurred(self, error: QC.QProcess.ProcessError) -> None:
        """Process error has occurred

        Args:
            error: Process error
        """
        # Other errors (e.g. a crash) are followed by the `finished` signal
        if error != QC.QProcess.FailedToStart:
            return
        process = self.process
        assert process is not None
        reason = process.errorString()
        self.print(
            _("# ==> Unable to run '%s' macro") % self.title + f": {reason}",
            error=True,
        )
        QW.QMessageBox.critical(
            self.console,
            _("Error"),
            _("Macro Python interpreter failed to start!")
            + f"\n\n{reason}\n\n"
            + _("Executable: %s") % process.program(),
        )
        self._finish_run(process)

    def is_running(self) -> bool:
        """Is macro running?

        Returns:
            True if macro is running
        """
        return self.process is not None

    def kill(self) -> None:
        """Kill process associated to macro"""
        process = self.process
        if process is not None:
            self.print(_("Terminating '%s' macro") % self.title, error=True)
            if self.process is process:
                process.kill()

    def stop_and_wait(self) -> None:
        """Stop the child before its tab or its transient owner is discarded."""
        process = self.process
        if process is not None and process.state() != QC.QProcess.NotRunning:
            self._stopping = True
            try:
                self.kill()
                if self.process is not process:
                    return
                # The finished signal may release the process during the wait.
                if not process.waitForFinished(5000) and self.process is process:
                    raise RuntimeError(
                        _("Unable to stop '%s' macro: %s")
                        % (self.title, process.errorString())
                    )
            finally:
                self._stopping = False

    # pylint: disable=unused-argument
    def finished(self, exit_code: int, exit_status: QC.QProcess.ExitStatus) -> None:
        """Process has finished

        Args:
            exit_code: Exit code
            exit_status: Exit status
        """
        process = self.process
        assert process is not None
        self.__last_exit_code = exit_code
        for read, error in ((self.get_stdout, False), (self.get_stderr, True)):
            text = read()
            if text:
                self.console.write(text, error=error, flush=True)
        self.print(_("# <== '%s' macro has finished") % self.title, eol_before=False)
        self._finish_run(process)

    def get_exit_code(self) -> int | None:
        """Return last exit code of the macro process

        Returns:
            Last exit code or None if process has not finished yet
        """
        return self.__last_exit_code
