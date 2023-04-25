# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module providing DataLab Macro editor widget
"""

import os
import sys
import time
from uuid import uuid4

from guidata.userconfigio import BaseIOHandler
from guidata.widgets.codeeditor import CodeEditor
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.core.gui import ObjItf
from cdl.env import execenv

UNTITLED_NB = 0


class Macro(QC.QObject, ObjItf):
    """Object representing a macro: editor, path, open/save actions, etc."""

    PREFIX = "m"

    STARTED = QC.Signal()
    FINISHED = QC.Signal()
    MODIFIED = QC.Signal()
    MACRO_TITLE = _("Macro simple example")
    MACRO_SAMPLE = f"""# {MACRO_TITLE}

import numpy as np

from cdl.remotecontrol import RemoteClient

remote = RemoteClient()
remote.try_and_connect()

z = np.random.rand(20, 20)
remote.add_image("toto", z)

print("All done!")
"""

    def __init__(self, console: PythonShellWidget, name: str = None) -> None:
        super().__init__()
        self.uuid = str(uuid4())
        self.console = console
        self.setObjectName(self.get_untitled_title() if name is None else name)
        self.editor = CodeEditor(language="python")
        self.editor.setPlainText(self.MACRO_SAMPLE)
        self.editor.modificationChanged.connect(self.modification_changed)
        self.process = None

    @property
    def short_id(self):
        """Short macro ID"""
        return self.PREFIX + self.uuid[:3]

    @property
    def title(self) -> str:
        """Return object title"""
        return self.objectName()

    def serialize(self, writer: BaseIOHandler) -> None:
        """Serialize this macro"""
        with writer.group("name"):
            writer.write(self.objectName())
        with writer.group("contents"):
            writer.write(self.editor.toPlainText())

    def deserialize(self, reader: BaseIOHandler) -> None:
        """Deserialize this macro"""
        with reader.group("name"):
            self.setObjectName(reader.read_any())
        with reader.group("contents"):
            self.editor.setPlainText(reader.read_any())

    @staticmethod
    def get_untitled_title() -> str:
        """Increment untitled number and return untitled macro title"""
        global UNTITLED_NB  # pylint: disable=global-statement
        UNTITLED_NB += 1
        untitled = _("Untitled")
        return f"{untitled} {UNTITLED_NB:02d}"

    def modification_changed(self, state: bool) -> None:
        """Method called when macro's editor modification state changed"""
        if state:
            self.MODIFIED.emit()

    @staticmethod
    def transcode(bytearr) -> str:
        """Transcode bytes to locale str"""
        locale_codec = QC.QTextCodec.codecForLocale()
        return locale_codec.toUnicode(bytearr.data())

    def get_stdout(self) -> str:
        """Return standard output str"""
        self.process.setReadChannel(QC.QProcess.StandardOutput)
        bytearr = QC.QByteArray()
        while self.process.bytesAvailable():
            bytearr += self.process.readAllStandardOutput()
        return self.transcode(bytearr)

    def get_stderr(self) -> str:
        """Return standard error str"""
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
        """Print text in console, with line separator"""
        msg = f"---({time.ctime()})---[{text}]{os.linesep}"
        if eol_before:
            msg = os.linesep + msg
        self.console.write(msg, error=error, prompt=not error)

    def run(self) -> None:
        """Run macro"""
        self.process = QC.QProcess()
        text = self.editor.toPlainText()
        code = os.linesep.join(text.splitlines(False)).replace('"', "'")
        env = QC.QProcessEnvironment()
        env.insert(execenv.XMLRPCPORT_ENV, str(execenv.port))
        sysenv = env.systemEnvironment()
        for key in sysenv.keys():
            env.insert(key, sysenv.value(key))
        # env = [str(_path) for _path in self.process.systemEnvironment()]
        self.process.readyReadStandardOutput.connect(self.write_output)
        self.process.readyReadStandardError.connect(self.write_error)
        self.process.finished.connect(self.finished)
        self.process.setProcessEnvironment(env)
        args = ["-c", code]
        self.process.start(sys.executable, args)
        running = self.process.waitForStarted(3000)
        name = self.objectName()
        if not running:
            self.print(_("# ==> Unable to run '%s' macro") % name, error=True)
            QW.QMessageBox.critical(
                self, _("Error"), _("Macro Python interpreter failed to start!")
            )
        else:
            self.print(_("# ==> Running '%s' macro...") % name)
            self.STARTED.emit()

    def is_running(self) -> bool:
        """Is macro running?"""
        if self.process is not None:
            return self.process.state() == QC.QProcess.Running
        return False

    def kill(self) -> None:
        """Kill process associated to macro"""
        if self.process is not None:
            self.print(_("Terminating '%s' macro") % self.objectName(), error=True)
            self.process.kill()

    def finished(
        self, exit_code, exit_status
    ) -> None:  # pylint: disable=unused-argument
        """Process has finished"""
        self.print(
            _("# <== '%s' macro has finished") % self.objectName(), eol_before=False
        )
        self.FINISHED.emit()
        self.process = None
