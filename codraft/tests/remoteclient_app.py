# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Remote GUI-based client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code


from random import randint

from codraft import app
from codraft.config import _
from codraft.remotecontrol import CodraFTConnectionError, RemoteClient
from codraft.tests import embedded1_unit
from codraft.tests.logview_app import exec_script
from codraft.tests.remoteclient_unit import multiple_commands

SHOW = True  # Show test in GUI-based test launcher


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to CodraFT.")
    INIT_BUTTON_LABEL = _("Connect to CodraFT")

    def init_codraft(self):
        """Open CodraFT test"""
        if self.codraft is None:
            self.codraft = RemoteClient()
            try:
                self.codraft.connect()
                self.host.log("✨ Initialized CodraFT connection ✨")
                self.host.log(f"  Communication port: {self.codraft.port}")
                self.host.log("  List of exposed methods:")
                for name in self.codraft.serverproxy.system.listMethods():
                    self.host.log(f"    {name}")
            except CodraFTConnectionError:
                self.codraft = None
                self.host.log("🔥 Connection refused 🔥 (server is not ready?)")

    def add_additional_buttons(self):
        """Add additional buttons"""
        add_btn = self.host.add_button
        add_btn(_("Execute multiple commands"), self.exec_multiple_cmd, 10)
        add_btn(_("Get object list"), self.get_object_list, 10)
        add_btn(_("Get object"), self.get_object)

    def exec_multiple_cmd(self):
        """Execute multiple commands in CodraFT"""
        if self.codraft is not None:
            self.host.log("Starting command sequence...")
            multiple_commands(self.codraft)
            self.host.log("...end")

    def get_object_list(self):
        """Get object (signal/image) list for current panel"""
        if self.codraft is not None:
            self.host.log("Object list:")
            titles = self.codraft.get_object_list()
            if titles:
                for name in titles:
                    self.host.log(f"  {name}")
            else:
                self.host.log("  Empty.")

    def get_object(self):
        """Get object (signal/image) at index for current panel"""
        if self.codraft is not None:
            titles = self.codraft.get_object_list()
            if titles:
                index = randint(0, len(titles) - 1)
                obj = self.codraft.get_object(index)
                self.host.log(f"Object '{titles[index]}'")
                self.host.log(str(obj))
            else:
                self.host.log("🏴‍☠️ Object list is empty!")

    def add_object(self, obj):
        """Add object to CodraFT"""
        self.codraft.add_object(obj)

    def remove_all(self):
        """Remove all objects from CodraFT"""
        if self.codraft is not None:
            self.codraft.reset_all()
            self.host.log("Removed all objects")

    def close_codraft(self):
        """Close CodraFT window"""
        self.codraft.close_application()
        self.host.log("🎬 Closed CodraFT!")


if __name__ == "__main__":
    exec_script(app.__file__, wait=False)
    embedded1_unit.test_embedded_feature(HostWindow)
