# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Remote server/client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import abc
import os
import os.path as osp
import time
from io import BytesIO
from typing import List
from xmlrpc.client import Binary, ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from qtpy import QtCore as QC

from codraft import __version__
from codraft.core.gui.main import CodraFTMainWindow
from codraft.core.model.image import create_image
from codraft.core.model.signal import create_signal
from codraft.tests import codraft_app_context
from codraft.tests.data import create_2d_gaussian, create_test_signal1

# from codraft.utils.qthelpers import qt_app_context
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


def array_to_rpcbinary(data: np.ndarray) -> Binary:
    """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
    dbytes = BytesIO()
    np.save(dbytes, data, allow_pickle=False)
    return Binary(dbytes.getvalue())


def rpcbinary_to_array(binary: Binary) -> np.ndarray:
    """Convert XML-RPC binary to NumPy array"""
    dbytes = BytesIO(binary.data)
    return np.load(dbytes, allow_pickle=False)


class DummyCodraFTWindow:
    """Dummy CodraFT window, for test only"""

    def switch_to_signal_panel(self):
        """Switch to signal panel"""
        print(self.switch_to_signal_panel.__doc__)

    def switch_to_image_panel(self):
        """Switch to image panel"""
        print(self.switch_to_image_panel.__doc__)

    def reset_all(self):
        """Reset all application data"""
        print(self.reset_all.__doc__)

    def save_to_h5_file(self, filename=None):
        """Save to a CodraFT HDF5 file"""
        print(self.save_to_h5_file.__doc__, filename)

    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ) -> None:
        """Open a CodraFT HDF5 file or import from any other HDF5 file"""
        print(self.open_h5_files.__doc__, h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool = None) -> None:
        """Open CodraFT HDF5 browser to Import HDF5 file"""
        print(self.import_h5_file.__doc__, filename, reset_all)

    def add_object(self, obj, refresh=True):
        """Add object - signal or image"""
        print(self.add_object.__doc__, obj, refresh)


class BaseRPCServer(abc.ABC):
    """Base XML-RPC server mixin"""

    def __init__(self):
        self.port = None

    def serve(self):
        """Start server and serve forever"""
        with SimpleXMLRPCServer(
            ("127.0.0.1", 0), logRequests=False, allow_none=True
        ) as server:
            server.register_introspection_functions()
            server.register_function(self.get_version)
            self.register_functions(server)
            self.port = server.server_address[1]
            self.notify_port(self.port)
            server.serve_forever()

    @staticmethod
    def get_version():
        """Return CodraFT version"""
        return __version__

    @abc.abstractmethod
    def notify_port(self, port: int):
        """Notify automatically attributed port"""

    @abc.abstractmethod
    def register_functions(self, server: SimpleXMLRPCServer):
        """Register functions"""


class RPCServerThreadMeta(type(QC.QThread), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class RPCServerThread(QC.QThread, BaseRPCServer, metaclass=RPCServerThreadMeta):
    """XML-RPC server QThread"""

    SIG_SERVER_PORT = QC.Signal(int)
    SIG_ADD_OBJECT = QC.Signal(object)
    SIG_SWITCH_TO_SIGNAL_PANEL = QC.Signal()
    SIG_SWITCH_TO_IMAGE_PANEL = QC.Signal()
    SIG_RESET_ALL = QC.Signal()
    SIG_SAVE_TO_H5 = QC.Signal(str)
    SIG_OPEN_H5 = QC.Signal(list, bool, bool)
    SIG_IMPORT_H5 = QC.Signal(str, bool)

    def __init__(self, win: CodraFTMainWindow):
        QC.QThread.__init__(self)
        BaseRPCServer.__init__(self)
        self.SIG_ADD_OBJECT.connect(win.add_object)
        self.SIG_SWITCH_TO_SIGNAL_PANEL.connect(win.switch_to_signal_panel)
        self.SIG_SWITCH_TO_IMAGE_PANEL.connect(win.switch_to_image_panel)
        self.SIG_RESET_ALL.connect(win.reset_all)
        self.SIG_SAVE_TO_H5.connect(win.save_to_h5_file)
        self.SIG_OPEN_H5.connect(win.open_h5_files)
        self.SIG_IMPORT_H5.connect(win.import_h5_file)

    def notify_port(self, port: int):
        """Notify automatically attributed port"""
        self.SIG_SERVER_PORT.emit(port)

    def register_functions(self, server: SimpleXMLRPCServer):
        """Register functions"""
        server.register_function(self.swith_to_signal_panel)
        server.register_function(self.swith_to_image_panel)
        server.register_function(self.add_signal)
        server.register_function(self.add_image)
        server.register_function(self.reset_all)
        server.register_function(self.save_to_h5_file)
        server.register_function(self.open_h5_files)
        server.register_function(self.import_h5_file)

    def run(self):
        """Thread execution method"""
        self.serve()

    def swith_to_signal_panel(self):
        """Swith to signal panel"""
        self.SIG_SWITCH_TO_SIGNAL_PANEL.emit()

    def swith_to_image_panel(self):
        """Swith to image panel"""
        self.SIG_SWITCH_TO_IMAGE_PANEL.emit()

    def reset_all(self):
        """Reset all application data"""
        self.SIG_RESET_ALL.emit()

    def save_to_h5_file(self, filename: str):
        """Save to a CodraFT HDF5 file"""
        self.SIG_SAVE_TO_H5.emit(filename)

    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ):
        """Open a CodraFT HDF5 file or import from any other HDF5 file"""
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool = None):
        """Open CodraFT HDF5 browser to Import HDF5 file"""
        self.SIG_IMPORT_H5.emit(filename, reset_all)

    def add_signal(
        self,
        title: str,
        xbinary: Binary,
        ybinary: Binary,
        xunit: str = None,
        yunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
    ):
        """Add signal data to CodraFT"""
        xdata = rpcbinary_to_array(xbinary)
        ydata = rpcbinary_to_array(ybinary)
        signal = create_signal(title, xdata, ydata)
        signal.xunit = xunit
        signal.yunit = yunit
        signal.xlabel = xlabel
        signal.ylabel = ylabel
        self.SIG_ADD_OBJECT.emit(signal)
        return True

    def add_image(
        self,
        title: str,
        zbinary: Binary,
        xunit: str = None,
        yunit: str = None,
        zunit: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
    ):
        """Add image data to CodraFT"""
        data = rpcbinary_to_array(zbinary)
        image = create_image(title, data)
        image.xunit = xunit
        image.yunit = yunit
        image.zunit = zunit
        image.xlabel = xlabel
        image.ylabel = ylabel
        image.zlabel = zlabel
        self.SIG_ADD_OBJECT.emit(image)
        return True


def test():
    """Remote server/client test"""
    with temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        win = DummyCodraFTWindow()
        # with qt_app_context() as qapp:
        with codraft_app_context(console=False) as win:
            server_thread = RPCServerThread(win)
            server_thread.start()
            while server_thread.port is None:
                time.sleep(0.1)
            port = server_thread.port
            print(f"Port: {port}")
            # qapp.exec()
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
            print(s.system.listMethods())
        os.remove(fname)


if __name__ == "__main__":
    test()
