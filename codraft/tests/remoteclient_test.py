# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Remote server/client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import json
import os.path as osp
import time
from typing import List
from xmlrpc.client import Binary, ServerProxy

import numpy as np
from guidata.dataset import datatypes as gdt
from guidata.jsonio import JSONWriter

from codraft.config import Conf, _, initialize
from codraft.core.gui.processor.signal import XYCalibrateParam
from codraft.core.model.signal import SignalParam, create_signal
from codraft.tests import embedded1_unit
from codraft.tests.data import create_2d_gaussian, create_test_signal1
from codraft.tests.logview_app import exec_script
from codraft.tests.remoteserver_test import array_to_rpcbinary, rpcbinary_to_array
from codraft.utils.tests import temporary_directory

SHOW = True  # Show test in GUI-based test launcher

# === Python 2.7 client side:
# import xmlrpclib
# import numpy as np
# def array_to_binary(data):
#     """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
#     dbytes = BytesIO()
#     np.save(dbytes, data, allow_pickle=False)
#     return xmlrpc.Binary(dbytes.getvalue())
# s = xmlrpclib.ServerProxy("http://127.0.0.1:8000")
# data = np.array([[3, 4, 5], [7, 8, 0]], dtype=np.uint16)
# s.add_image("toto", array_to_binary(data))


class CodraFTRemote:
    """Object representing a proxy/client to CodraFT XML-RPC server"""

    def __init__(self):
        self.port = None
        self.serverproxy = None

    def connect(self, port=None):
        """Connect to CodraFT XML-RPC server"""
        if port is None:
            initialize()
            port = Conf.main.rpc_server_port.get()
        self.port = port
        self.serverproxy = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        self.get_version()  # Will raise a ConnectionRefusedError if connection failed

    # === Following methods should match the register functions in XML-RPC server

    def get_version(self):
        """Return CodraFT version"""
        return self.serverproxy.get_version()

    def close_application(self):
        """Close CodraFT application"""
        self.serverproxy.close_application()

    def switch_to_signal_panel(self):
        """Switch to signal panel"""
        self.serverproxy.switch_to_signal_panel()

    def switch_to_image_panel(self):
        """Switch to image panel"""
        self.serverproxy.switch_to_image_panel()

    def reset_all(self):
        """Reset all application data"""
        self.serverproxy.reset_all()

    def save_to_h5_file(self, filename: str):
        """Save to a CodraFT HDF5 file"""
        self.serverproxy.save_to_h5_file(filename)

    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ):
        """Open a CodraFT HDF5 file or import from any other HDF5 file"""
        self.serverproxy.open_h5_files(h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool = None):
        """Open CodraFT HDF5 browser to Import HDF5 file"""
        self.serverproxy.import_h5_file(filename, reset_all)

    def open_object(self, filename: str) -> None:
        """Open object from file in current panel (signal/image)"""
        self.serverproxy.open_object(filename)

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str = None,
        yunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
    ):
        """Add signal data to CodraFT"""
        xbinary = array_to_rpcbinary(xdata)
        ybinary = array_to_rpcbinary(ydata)
        p = self.serverproxy
        return p.add_signal(title, xbinary, ybinary, xunit, yunit, xlabel, ylabel)

    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str = None,
        yunit: str = None,
        zunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
    ):  # pylint: disable=too-many-arguments
        """Add image data to CodraFT"""
        zbinary = array_to_rpcbinary(data)
        p = self.serverproxy
        return p.add_image(title, zbinary, xunit, yunit, zunit, xlabel, ylabel, zlabel)

    def calc(self, name: str, param: gdt.DataSet = None):
        """Call compute function `name` in current panel's processor"""
        p = self.serverproxy
        if param is None:
            return p.calc(name)
        writer = JSONWriter()
        param.serialize(writer)
        param_json = writer.get_json()
        klass = param.__class__
        return p.calc(name, [klass.__module__, klass.__name__, param_json])

    def add_object(self, obj):
        """Add object to CodraFT"""
        p = self.serverproxy
        if isinstance(obj, SignalParam):
            p.add_signal(
                obj.title,
                array_to_rpcbinary(obj.x),
                array_to_rpcbinary(obj.y),
                obj.xunit,
                obj.yunit,
                obj.xlabel,
                obj.ylabel,
            )
        else:
            p.add_image(
                obj.title,
                array_to_rpcbinary(obj.data),
                obj.xunit,
                obj.yunit,
                obj.zunit,
                obj.xlabel,
                obj.ylabel,
                obj.zlabel,
            )


def multiple_commands(remote: CodraFTRemote):
    """Execute multiple XML-RPC commands"""
    with temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        print(f"CodraFT version: {remote.get_version()}")

        x, y = create_test_signal1().get_data()
        print(remote.add_signal("tutu", x, y))

        z = create_2d_gaussian(2000, np.uint16)
        print(remote.add_image("toto", z))

        remote.save_to_h5_file(fname)
        remote.reset_all()
        remote.open_h5_files([fname], True, False)
        remote.import_h5_file(fname, True)
        remote.switch_to_signal_panel()
        remote.calc("log10")

        param = XYCalibrateParam()
        param.a, param.b = 1.2, 0.1
        remote.calc("calibrate", param)

        print(remote.serverproxy.system.listMethods())
        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


def test(port=None):
    """Remote client test"""
    remote = CodraFTRemote()
    remote.connect(port=port)
    multiple_commands(remote)


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to CodraFT.")
    INIT_BUTTON_LABEL = _("Connect to CodraFT")

    def init_codraft(self):
        """Open CodraFT test"""
        if self.codraft is None:
            self.codraft = CodraFTRemote()
            try:
                self.codraft.connect()
                self.host.log("✨ Initialized CodraFT connection ✨")
            except ConnectionRefusedError:
                self.codraft = None
                self.host.log("🔥 Connection refused 🔥 (server is not ready?)")

    def add_additional_buttons(self):
        """Add additional buttons"""
        add_btn = self.host.add_button
        add_btn(_("Execute multiple commands"), self.exec_multiple_cmd, 10)

    def exec_multiple_cmd(self):
        """Execute multiple commands in CodraFT"""
        if self.codraft is not None:
            self.host.log("Starting command sequence...")
            multiple_commands(self.codraft)
            self.host.log("...end")

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
    exec_script(osp.join(osp.dirname(__file__), "remoteserver_test.py"), wait=False)
    embedded1_unit.test_embedded_feature(HostWindow)
