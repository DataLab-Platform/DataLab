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
import time

from guidata.io import BaseIOHandler
from guidata.utils.misc import to_string
from guidata.widgets.codeeditor import CodeEditor
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

import datalab
from datalab.config import _
from datalab.env import execenv
from datalab.gui import ObjItf

UNTITLED_NB = 0


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

from datalab.proxy import RemoteProxy

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
        self.editor = CodeEditor(language="python")
        self.editor.setLineWrapMode(QW.QPlainTextEdit.NoWrap)
        self.set_code(self.MACRO_SAMPLE)
        self.editor.modificationChanged.connect(self.modification_changed)
        self.process = None

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

    def serialize(self, writer: BaseIOHandler) -> None:
        """Serialize this macro

        Args:
            writer: Writer
        """
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("contents"):
            writer.write(self.get_code())

    def deserialize(self, reader: BaseIOHandler) -> None:
        """Deserialize this macro

        Args:
            reader: Reader
        """
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("contents"):
            self.set_code(reader.read_any())

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
        untitled = _("Untitled")
        return f"{untitled} {UNTITLED_NB:02d}"

    def modification_changed(self, state: bool) -> None:
        """Method called when macro's editor modification state changed

        Args:
            state: Modification state
        """
        if state:
            self.MODIFIED.emit()

    @staticmethod
    def transcode(bytearr: QC.QByteArray) -> str:
        """Transcode bytes to locale str

        Args:
            bytearr: Byte array

        Returns:
            Locale str
        """
        locale_codec = QC.QTextCodec.codecForLocale()
        return locale_codec.toUnicode(bytearr.data())

    def get_stdout(self) -> str:
        """Return standard output str

        Returns:
            Standard output str
        """
        self.process.setReadChannel(QC.QProcess.StandardOutput)
        bytearr = QC.QByteArray()
        while self.process.bytesAvailable():
            bytearr += self.process.readAllStandardOutput()
        return self.transcode(bytearr)

    def get_stderr(self) -> str:
        """Return standard error str

        Returns:
            Standard error str
        """
        self.process.setReadChannel(QC.QProcess.StandardError)
        bytearr = QC.QByteArray()
        while self.process.bytesAvailable():
            bytearr += self.process.readAllStandardError()
        return self.transcode(bytearr)

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
        self.console.write(msg, error=error, prompt=not error)

    def run(self) -> None:
        """Run macro"""
        self.process = QC.QProcess()
        code = self.get_code().replace('"', "'")
        datalab_path = osp.abspath(osp.join(osp.dirname(datalab.__file__), os.pardir))
        code = f"import sys; sys.path.append(r'{datalab_path}'){os.linesep}{code}"
        env = QC.QProcessEnvironment()
        env.insert(execenv.XMLRPCPORT_ENV, str(execenv.xmlrpcport))
        sysenv = env.systemEnvironment()
        for key in sysenv.keys():
            env.insert(key, sysenv.value(key))
        self.process.readyReadStandardOutput.connect(self.write_output)
        self.process.readyReadStandardError.connect(self.write_error)
        self.process.finished.connect(self.finished)
        self.process.setProcessEnvironment(env)
        args = ["-c", code]
        self.process.start(sys.executable, args)
        running = self.process.waitForStarted(3000)
        if not running:
            self.print(_("# ==> Unable to run '%s' macro") % self.title, error=True)
            QW.QMessageBox.critical(
                self, _("Error"), _("Macro Python interpreter failed to start!")
            )
        else:
            self.print(_("# ==> Running '%s' macro...") % self.title)
            self.STARTED.emit()

    def is_running(self) -> bool:
        """Is macro running?

        Returns:
            True if macro is running
        """
        if self.process is not None:
            return self.process.state() == QC.QProcess.Running
        return False

    def kill(self) -> None:
        """Kill process associated to macro"""
        if self.process is not None:
            self.print(_("Terminating '%s' macro") % self.title, error=True)
            self.process.kill()

    # pylint: disable=unused-argument
    def finished(self, exit_code: int, exit_status: QC.QProcess.ExitStatus) -> None:
        """Process has finished

        Args:
            exit_code: Exit code
            exit_status: Exit status
        """
        self.print(_("# <== '%s' macro has finished") % self.title, eol_before=False)
        self.FINISHED.emit()
        self.process = None
