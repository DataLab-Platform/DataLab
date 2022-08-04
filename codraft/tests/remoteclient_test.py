# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Remote server/client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import os.path as osp
import time
from xmlrpc.client import ServerProxy

import numpy as np

from codraft.config import Conf, _, initialize
from codraft.core.model.signal import SignalParam
from codraft.tests import embedded1_unit
from codraft.tests.data import create_2d_gaussian, create_test_signal1
from codraft.tests.logview_app import exec_script
from codraft.tests.remoteserver_test import array_to_rpcbinary
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


def connect_to_codraft(port=None) -> ServerProxy:
    """Connect to CodraFT XML-RPC server and return `ServerProxy` instance"""
    if port is None:
        initialize()
        port = Conf.main.rpc_server_port.get()
    return ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)


def multiple_commands(proxy: ServerProxy):
    """Execute multiple XML-RPC commands"""
    with temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        print(f"CodraFT version: {proxy.get_version()}")
        x, y = create_test_signal1().get_data()
        print(proxy.add_signal("tutu", array_to_rpcbinary(x), array_to_rpcbinary(y)))
        z = create_2d_gaussian(2000, np.uint16)
        print(proxy.add_image("toto", array_to_rpcbinary(z)))
        proxy.save_to_h5_file(fname)
        proxy.reset_all()
        proxy.open_h5_files([fname], True, False)
        proxy.import_h5_file(fname, True)
        proxy.switch_to_signal_panel()
        proxy.calc("log10")
        print(proxy.system.listMethods())
        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


def test(port=None):
    """Remote client test"""
    proxy = connect_to_codraft(port=port)
    multiple_commands(proxy)


class HostWindow(embedded1_unit.AbstractClientWindow):
    """Test main view"""

    PURPOSE = _("This the client application, which connects to CodraFT.")
    INIT_BUTTON_LABEL = _("Connect to CodraFT")

    def init_codraft(self):
        """Open CodraFT test"""
        if self.codraft is None:
            self.codraft = connect_to_codraft()
            self.host.log("✨ Initialized CodraFT connection")

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
        if isinstance(obj, SignalParam):
            self.codraft.add_signal(
                obj.title,
                array_to_rpcbinary(obj.x),
                array_to_rpcbinary(obj.y),
                obj.xunit,
                obj.yunit,
                obj.xlabel,
                obj.ylabel,
            )
        else:
            self.codraft.add_image(
                obj.title,
                array_to_rpcbinary(obj.data),
                obj.xunit,
                obj.yunit,
                obj.zunit,
                obj.xlabel,
                obj.ylabel,
                obj.zlabel,
            )

    def remove_all(self):
        """Remove all objects from CodraFT"""
        if self.codraft is not None:
            self.codraft.reset_all()
            self.host.log("Removed all objects")

    def close_codraft(self):
        """Close CodraFT window"""
        self.host.log("🛠 Not implemented yet")


if __name__ == "__main__":
    exec_script(osp.join(osp.dirname(__file__), "remoteserver_test.py"), wait=False)
    embedded1_unit.test_embedded_feature(HostWindow)
