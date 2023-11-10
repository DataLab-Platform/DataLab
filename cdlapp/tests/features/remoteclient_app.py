# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Remote GUI-based client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

# guitest: show

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from contextlib import contextmanager

from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdlapp import app
from cdlapp.config import _
from cdlapp.env import execenv
from cdlapp.proxy import RemoteCDLProxy
from cdlapp.tests.features import embedded1_unit
from cdlapp.tests.features.remoteclient_unit import multiple_commands
from cdlapp.tests.features.utilities.logview_app import exec_script
from cdlapp.utils.qthelpers import qt_app_context, qt_wait

APP_NAME = _("Remote client test")


def try_send_command():
    """Try and send command to DataLab application remotely"""

    def try_send_command_decorator(func):
        """Try... except... decorator"""

        @functools.wraps(func)
        def method_wrapper(*args, **kwargs):
            """Decorator wrapper function"""
            self: HostWindow = args[0]  # extracting 'self' from method arguments
            output = None
            try:
                output = func(*args, **kwargs)
            except ConnectionRefusedError:
                self.cdlapp = None
                message = "🔥 Connection refused 🔥 (server is not ready?)"
                self.host.log(message)
                QW.QMessageBox.critical(self, APP_NAME, message)
            return output

        return method_wrapper

    return try_send_command_decorator


class DataLabConnectionThread(QC.QThread):
    """DataLab Connection thread"""

    SIG_CONNECTION_OK = QC.Signal()
    SIG_CONNECTION_KO = QC.Signal()

    def __init__(self, connect_callback: Callable, parent: QC.QObject = None) -> None:
        super().__init__(parent)
        self.connect_callback = connect_callback

    def run(self) -> None:
        """Run thread"""
        try:
            self.connect_callback()
            self.SIG_CONNECTION_OK.emit()
        except ConnectionRefusedError:
            self.SIG_CONNECTION_KO.emit()


class DataLabConnectionDialog(QW.QDialog):
    """DataLab Connection dialog

    Args:
        connect_callback: Callback function to connect to DataLab server
        parent: Parent widget. Defaults to None.
    """

    def __init__(self, connect_callback: Callable, parent: QW.QWidget = None) -> None:
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.host_label = QW.QLabel("Host:")
        self.progress_bar = QW.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.status_label = QW.QLabel("Waiting for connection...")
        layout = QW.QVBoxLayout()
        layout.addWidget(self.host_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.thread = DataLabConnectionThread(connect_callback)
        self.thread.SIG_CONNECTION_OK.connect(self.on_connection_successful)
        self.thread.SIG_CONNECTION_KO.connect(self.on_connection_failed)
        button_box = QW.QDialogButtonBox(QW.QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def exec(self) -> int:
        """Execute dialog"""
        self.connect_to_server()
        return super().exec()

    def connect_to_server(self) -> None:
        """Connect to server"""
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Connecting to server...")
        self.thread.start()

    def on_connection_successful(self) -> None:
        """Connection successful"""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText("Connection successful!")
        self.accept()

    def on_connection_failed(self) -> None:
        """Connection failed"""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText("Connection failed.")
        self.reject()


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to DataLab.")
    INIT_BUTTON_LABEL = _("Connect to DataLab")

    def init_cdl(self):
        """Open DataLab test"""
        if self.cdlapp is None:
            self.cdlapp = RemoteCDLProxy(autoconnect=False)
            connect_dlg = DataLabConnectionDialog(self.cdlapp.connect, self)
            connect_dlg.host_label.setText(f"Host: DataLab server")
            ok = connect_dlg.exec()
            if ok:
                self.host.log("✨ Initialized DataLab connection ✨")
                self.host.log(f"  Communication port: {self.cdlapp.port}")
                self.host.log("  List of exposed methods:")
                for name in self.cdlapp.get_method_list():
                    self.host.log(f"    {name}")
            else:
                self.cdlapp = None
                self.host.log("🔥 Connection refused 🔥 (server is not ready?)")

    @try_send_command()
    def close_cdl(self):
        """Close DataLab window"""
        if self.cdlapp is not None:
            self.cdlapp.close_application()
            self.host.log("🎬 Closed DataLab!")
            self.cdlapp = None

    def add_additional_buttons(self):
        """Add additional buttons"""
        add_btn = self.host.add_button
        add_btn(_("Execute multiple commands"), self.exec_multiple_cmd, 10)
        add_btn(_("Get object titles"), self.get_object_titles, 10)
        add_btn(_("Get object uuids"), self.get_object_uuids, 10)
        add_btn(_("Get object"), self.get_object)

    @try_send_command()
    def exec_multiple_cmd(self):
        """Execute multiple commands in DataLab"""
        if self.cdlapp is not None:
            self.host.log("Starting command sequence...")
            multiple_commands(self.cdlapp)
            self.host.log("...end")

    @try_send_command()
    def get_object_titles(self):
        """Get object (signal/image) titles for current panel"""
        if self.cdlapp is not None:
            self.host.log("Object titles:")
            titles = self.cdlapp.get_object_titles()
            if titles:
                for name in titles:
                    self.host.log(f"  {name}")
            else:
                self.host.log("  Empty.")

    @try_send_command()
    def get_object_uuids(self):
        """Get object (signal/image) uuids for current panel"""
        if self.cdlapp is not None:
            self.host.log("Object uuids:")
            uuids = self.cdlapp.get_object_uuids()
            if uuids:
                for uuid in uuids:
                    self.host.log(f"  {uuid}")
            else:
                self.host.log("  Empty.")

    @try_send_command()
    def get_object(self):
        """Get object (signal/image) at index for current panel"""
        if self.cdlapp is not None:
            titles = self.cdlapp.get_object_titles()
            if titles:
                obj = self.cdlapp.get_object()
                self.host.log(f"Object '{obj.title}'")
                self.host.log(str(obj))
            else:
                self.host.log("🏴‍☠️ Object list is empty!")

    @try_send_command()
    def add_object(self, obj):
        """Add object to DataLab"""
        super().add_object(obj)

    @try_send_command()
    def remove_all(self):
        """Remove all objects from DataLab"""
        if self.cdlapp is not None:
            self.cdlapp.reset_all()
            self.host.log("Removed all objects")


@contextmanager
def qt_wait_print(dt: float, message: str, parent=None):
    """Wait and print message"""
    qt_wait(dt, show_message=True, parent=parent)
    execenv.print(f"{message}...", end="")
    yield
    execenv.print("OK")


def test_remote_client():
    """Remote client test"""
    env = os.environ.copy()
    env[execenv.DONOTQUIT_ENV] = "1"
    exec_script(app.__file__, wait=False, env=env)
    with qt_app_context(exec_loop=True, enable_logs=False):
        window = HostWindow()
        window.resize(800, 800)
        window.show()
        dt = 1
        if execenv.unattended:
            qt_wait(10, show_message=True, parent=window)
            window.init_cdl()
            with qt_wait_print(dt, "Executing multiple commands"):
                window.exec_multiple_cmd()
            with qt_wait_print(dt, "Getting object titles"):
                window.get_object_titles()
            with qt_wait_print(dt, "Getting object uuids"):
                window.get_object_uuids()
            with qt_wait_print(dt, "Getting object"):
                window.get_object()
            with qt_wait_print(dt, "Adding signals"):
                window.add_signals()
            with qt_wait_print(dt, "Adding images"):
                window.add_images()
            with qt_wait_print(dt, "Removing all objects"):
                window.remove_all()
            with qt_wait_print(dt, "Closing DataLab"):
                window.close_cdl()


if __name__ == "__main__":
    test_remote_client()
