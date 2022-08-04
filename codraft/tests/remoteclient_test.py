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

from codraft.config import Conf
from codraft.tests.data import create_2d_gaussian, create_test_signal1
from codraft.tests.remoteserver_test import array_to_rpcbinary
from codraft.utils.tests import temporary_directory

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


def test(port=None):
    """Remote client test"""
    if port is None:
        port = Conf.main.rpc_server_port.get()
    with temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        s = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        print(f"CodraFT version: {s.get_version()}")
        x, y = create_test_signal1().get_data()
        print(s.add_signal("tutu", array_to_rpcbinary(x), array_to_rpcbinary(y)))
        z = create_2d_gaussian(2000, np.uint16)
        print(s.add_image("toto", array_to_rpcbinary(z)))
        s.save_to_h5_file(fname)
        s.reset_all()
        s.open_h5_files([fname], True, False)
        s.import_h5_file(fname, True)
        s.switch_to_signal_panel()
        s.calc("log10")
        print(s.system.listMethods())
        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


if __name__ == "__main__":
    test()
