# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing DataLab Macro editor widget
"""

from __future__ import annotations

import abc
import os
import os.path as osp
import sys
import time

from guidata.userconfigio import BaseIOHandler
from guidata.widgets.codeeditor import CodeEditor
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

import cdl
from cdl.config import _
from cdl.core.gui import ObjItf
from cdl.env import execenv
from cdl.utils.misc import to_string

UNTITLED_NB = 0


class MacroMeta(type(QC.QObject), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class Macro(QC.QObject, ObjItf, metaclass=MacroMeta):
    """Object representing a macro: editor, path, open/save actions, etc.

    Args:
        console (PythonShellWidget): Python shell widget
        name (str, optional): Macro name. Defaults to None.
    """

    PREFIX = "m"

    STARTED = QC.Signal()
    FINISHED = QC.Signal()
    MODIFIED = QC.Signal()
    FILE_HEADER = os.linesep.join(
        ["# -*- coding: utf-8 -*-", "", '"""DataLab Macro"""', "", ""]
    )
    MACRO_TITLE = _("Macro simple example")
    MACRO_SAMPLE = f"""# {MACRO_TITLE}

import numpy as np

from cdl.remotecontrol import RemoteClient

remote = RemoteClient()
remote.connect()

z = np.random.rand(20, 20)
remote.add_image("toto", z)
remote.compute_fft()

print("All done!")
"""

    def __init__(self, console: PythonShellWidget, title: str | None = None) -> None:
        super().__init__()
        self.console = console
        self.title = self.get_untitled_title() if title is None else title
        self.editor = CodeEditor(language="python")
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
            code (str): Code to be executed
        """
        self.editor.setPlainText(code)

    def serialize(self, writer: BaseIOHandler) -> None:
        """Serialize this macro

        Args:
            writer (BaseIOHandler): Writer
        """
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("contents"):
            writer.write(self.get_code())

    def deserialize(self, reader: BaseIOHandler) -> None:
        """Deserialize this macro

        Args:
            reader (BaseIOHandler): Reader
        """
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("contents"):
            self.set_code(reader.read_any())

    def to_file(self, filename: str) -> None:
        """Save macro to file

        Args:
            filename (str): File name
        """
        code = self.FILE_HEADER + self.get_code()
        with open(filename, "wb") as fdesc:
            fdesc.write(code.encode("utf-8"))

    def from_file(self, filename: str) -> None:
        """Load macro from file

        Args:
            filename (str): File name
        """
        with open(filename, "rb") as fdesc:
            code = to_string(fdesc.read()).strip()
        header = self.FILE_HEADER.strip()
        if code.startswith(header):
            code = code[len(header) :].strip()
        self.set_code(code)

    @staticmethod
    def get_untitled_title() -> str:
        """Increment untitled number and return untitled macro title

        Returns:
            str: Untitled macro title
        """
        global UNTITLED_NB  # pylint: disable=global-statement
        UNTITLED_NB += 1
        untitled = _("Untitled")
        return f"{untitled} {UNTITLED_NB:02d}"

    def modification_changed(self, state: bool) -> None:
        """Method called when macro's editor modification state changed

        Args:
            state (bool): Modification state
        """
        if state:
            self.MODIFIED.emit()

    @staticmethod
    def transcode(bytearr: QC.QByteArray) -> str:
        """Transcode bytes to locale str

        Args:
            bytearr (QByteArray): Byte array

        Returns:
            str: Locale str
        """
        locale_codec = QC.QTextCodec.codecForLocale()
        return locale_codec.toUnicode(bytearr.data())

    def get_stdout(self) -> str:
        """Return standard output str

        Returns:
            str: Standard output str
        """
        self.process.setReadChannel(QC.QProcess.StandardOutput)
        bytearr = QC.QByteArray()
        while self.process.bytesAvailable():
            bytearr += self.process.readAllStandardOutput()
        return self.transcode(bytearr)

    def get_stderr(self) -> str:
        """Return standard error str

        Returns:
            str: Standard error str
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
            text (str): Text to be printed
            error (bool, optional): Print as error. Defaults to False.
        """
        msg = f"---({time.ctime()})---[{text}]{os.linesep}"
        if eol_before:
            msg = os.linesep + msg
        self.console.write(msg, error=error, prompt=not error)

    def run(self) -> None:
        """Run macro"""
        self.process = QC.QProcess()
        code = self.get_code().replace('"', "'")
        cdl_path = osp.abspath(osp.join(osp.dirname(cdl.__file__), os.pardir))
        code = f"import sys; sys.path.append(r'{cdl_path}'){os.linesep}{code}"
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
            bool: True if macro is running
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
    def finished(self, exit_code, exit_status) -> None:
        """Process has finished

        Args:
            exit_code (int): Exit code
            exit_status (QC.QProcess.ExitStatus): Exit status
        """
        self.print(_("# <== '%s' macro has finished") % self.title, eol_before=False)
        self.FINISHED.emit()
        self.process = None
