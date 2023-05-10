# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Remote GUI-based client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code


from random import randint

from cdl import app
from cdl.config import _
from cdl.remotecontrol import CDLConnectionError, RemoteClient
from cdl.tests import embedded1_unit
from cdl.tests.logview_app import exec_script
from cdl.tests.remoteclient_unit import multiple_commands

SHOW = True  # Show test in GUI-based test launcher


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to DataLab.")
    INIT_BUTTON_LABEL = _("Connect to DataLab")

    def init_cdl(self):
        """Open DataLab test"""
        if self.cdl is None:
            self.cdl = RemoteClient()
            try:
                self.cdl.connect()
                self.host.log("✨ Initialized DataLab connection ✨")
                self.host.log(f"  Communication port: {self.cdl.port}")
                self.host.log("  List of exposed methods:")
                for name in self.cdl.serverproxy.system.listMethods():
                    self.host.log(f"    {name}")
            except CDLConnectionError:
                self.cdl = None
                self.host.log("🔥 Connection refused 🔥 (server is not ready?)")

    def add_additional_buttons(self):
        """Add additional buttons"""
        add_btn = self.host.add_button
        add_btn(_("Execute multiple commands"), self.exec_multiple_cmd, 10)
        add_btn(_("Get object titles"), self.get_object_titles, 10)
        add_btn(_("Get object uuids"), self.get_object_uuids, 10)
        add_btn(_("Get object"), self.get_object)

    def exec_multiple_cmd(self):
        """Execute multiple commands in DataLab"""
        if self.cdl is not None:
            self.host.log("Starting command sequence...")
            multiple_commands(self.cdl)
            self.host.log("...end")

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

    def get_object(self):
        """Get object (signal/image) at index for current panel"""
        if self.cdl is not None:
            titles = self.cdl.get_object_titles()
            if titles:
                index = randint(0, len(titles) - 1)
                oid = self.cdl.get_object_uuids()[index]
                obj = self.cdl.get_object_from_uuid(oid)
                self.host.log(f"Object '{titles[index]}'")
                self.host.log(str(obj))
            else:
                self.host.log("🏴‍☠️ Object list is empty!")

    def add_object(self, obj):
        """Add object to DataLab"""
        self.cdl.add_object(obj)

    def remove_all(self):
        """Remove all objects from DataLab"""
        if self.cdl is not None:
            self.cdl.reset_all()
            self.host.log("Removed all objects")

    def close_cdl(self):
        """Close DataLab window"""
        self.cdl.close_application()
        self.host.log("🎬 Closed DataLab!")


if __name__ == "__main__":
    exec_script(app.__file__, wait=False)
    embedded1_unit.test_embedded_feature(HostWindow)
