# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Remote GUI-based client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

# guitest: show

from __future__ import annotations

import functools
import os
from contextlib import contextmanager

from guidata.qthelpers import qt_app_context, qt_wait
from qtpy import QtWidgets as QW

from cdl import app
from cdl.config import _
from cdl.env import execenv
from cdl.proxy import RemoteProxy
from cdl.tests.features.control import embedded1_unit
from cdl.tests.features.control.remoteclient_unit import multiple_commands
from cdl.tests.features.utilities.logview_app import exec_script
from cdl.utils.qthelpers import bring_to_front
from cdl.widgets.connection import ConnectionDialog

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
            except ConnectionRefusedError as exc:
                self.cdl = None
                message = "🔥 Connection refused 🔥 (server is not ready?)"
                self.host.log(message)
                if execenv.unattended:
                    raise ConnectionRefusedError(
                        "Connection refused (server is not ready?)"
                    ) from exc
                QW.QMessageBox.critical(self, APP_NAME, message)
            return output

        return method_wrapper

    return try_send_command_decorator


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to DataLab.")
    INIT_BUTTON_LABEL = _("Connect to DataLab")

    def init_cdl(self):
        """Open DataLab test"""
        if self.cdl is None:
            if execenv.unattended:
                self.cdl = RemoteProxy()
                ok = True
            else:
                self.cdl = RemoteProxy(autoconnect=False)
                connect_dlg = ConnectionDialog(self.cdl.connect, self)
                ok = connect_dlg.exec()
            if ok:
                self.host.log("✨ Initialized DataLab connection ✨")
                self.host.log(f"  Communication port: {self.cdl.port}")
                self.host.log("  List of exposed methods:")
                for name in self.cdl.get_method_list():
                    self.host.log(f"    {name}")
            else:
                self.cdl = None
                self.host.log("🔥 Connection refused 🔥 (server is not ready?)")
                if execenv.unattended:
                    raise ConnectionRefusedError(
                        "Connection refused (server is not ready?)"
                    )
        else:
            self.host.log("=> Already connected to DataLab")

    @try_send_command()
    def close_cdl(self):
        """Close DataLab window"""
        if self.cdl is not None:
            self.cdl.close_application()
            self.host.log("🎬 Closed DataLab!")
            self.cdl = None

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
        if self.cdl is not None:
            self.host.log("Starting command sequence...")
            multiple_commands(self.cdl)
            self.host.log("...end")

    @try_send_command()
    def get_object_titles(self):
        """Get object (signal/image) titles for current panel"""
        if self.cdl is not None:
            self.host.log("Object titles:")
            titles = self.cdl.get_object_titles()
            if titles:
                for name in titles:
                    self.host.log(f"  {name}")
            else:
                self.host.log("  Empty.")

    @try_send_command()
    def get_object_uuids(self):
        """Get object (signal/image) uuids for current panel"""
        if self.cdl is not None:
            self.host.log("Object uuids:")
            uuids = self.cdl.get_object_uuids()
            if uuids:
                for uuid in uuids:
                    self.host.log(f"  {uuid}")
            else:
                self.host.log("  Empty.")

    @try_send_command()
    def get_object(self):
        """Get object (signal/image) at index for current panel"""
        if self.cdl is not None:
            titles = self.cdl.get_object_titles()
            if titles:
                obj = self.cdl.get_object()
                if obj is not None:
                    self.host.log(f"Object '{obj.title}'")
                    self.host.log(str(obj))
                else:
                    self.host.log("🏴‍☠️ Object is None! (no selection)")
            else:
                self.host.log("🏴‍☠️ Object list is empty!")

    @try_send_command()
    def add_object(self, obj):
        """Add object to DataLab"""
        super().add_object(obj)

    @try_send_command()
    def remove_all(self):
        """Remove all objects from DataLab"""
        if self.cdl is not None:
            self.cdl.reset_all()
            self.host.log("Removed all objects")


@contextmanager
def qt_wait_print(dt: float, message: str, parent=None):
    """Wait and print message"""
    qt_wait(dt, show_message=True, parent=parent)
    execenv.print(f"{message}...", end="")
    yield
    execenv.print("OK")


def test_remote_client():
    """Remote client application test"""
    env = os.environ.copy()
    env[execenv.DO_NOT_QUIT_ENV] = "1"
    exec_script(app.__file__, wait=False, env=env)
    with qt_app_context(exec_loop=True):
        window = HostWindow()
        window.resize(800, 800)
        window.show()
        dt = 1
        if execenv.unattended:
            qt_wait(10, show_message=True, parent=window)
            window.init_cdl()
            with qt_wait_print(dt, "Executing multiple commands"):
                window.exec_multiple_cmd()
            bring_to_front(window)
            with qt_wait_print(dt, "Raising DataLab window"):
                window.raise_cdl()
            with qt_wait_print(dt, "Import macro"):
                window.import_macro()
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
            with qt_wait_print(dt, "Run macro"):
                window.run_macro()
            with qt_wait_print(dt * 2, "Stop macro"):
                window.stop_macro()
            with qt_wait_print(dt, "Removing all objects"):
                window.remove_all()
            with qt_wait_print(dt, "Closing DataLab"):
                window.close_cdl()


if __name__ == "__main__":
    test_remote_client()
