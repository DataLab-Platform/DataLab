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
import functools
import time
from io import BytesIO
from typing import List
from xmlrpc.client import Binary
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from guidata.dataset import datatypes as gdt
from qtpy import QtCore as QC

from codraft import __version__
from codraft.config import Conf
from codraft.core.gui.main import CodraFTMainWindow
from codraft.core.model.image import create_image
from codraft.core.model.signal import create_signal
from codraft.tests import codraft_app_context


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


def remote_call(func):
    """Decorator for method calling CodraFT main window remotely"""

    @functools.wraps(func)
    def method_wrapper(*args, **kwargs):
        """Decorator wrapper function"""
        self = args[0]  # extracting 'self' from method arguments
        self.is_ready = False
        output = func(*args, **kwargs)
        while not self.is_ready:
            QC.QCoreApplication.processEvents()
            time.sleep(0.05)
        return output

    return method_wrapper


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
    SIG_CALC = QC.Signal(str, object)

    def __init__(self, win: CodraFTMainWindow):
        QC.QThread.__init__(self)
        BaseRPCServer.__init__(self)
        self.is_ready = True
        self.win = win
        win.SIG_READY.connect(self.codraft_is_ready)
        self.SIG_ADD_OBJECT.connect(win.add_object)
        self.SIG_SWITCH_TO_SIGNAL_PANEL.connect(win.switch_to_signal_panel)
        self.SIG_SWITCH_TO_IMAGE_PANEL.connect(win.switch_to_image_panel)
        self.SIG_RESET_ALL.connect(win.reset_all)
        self.SIG_SAVE_TO_H5.connect(win.save_to_h5_file)
        self.SIG_OPEN_H5.connect(win.open_h5_files)
        self.SIG_IMPORT_H5.connect(win.import_h5_file)
        self.SIG_CALC.connect(win.calc)

    def notify_port(self, port: int):
        """Notify automatically attributed port"""
        self.SIG_SERVER_PORT.emit(port)

    def register_functions(self, server: SimpleXMLRPCServer):
        """Register functions"""
        server.register_function(self.switch_to_signal_panel)
        server.register_function(self.switch_to_image_panel)
        server.register_function(self.add_signal)
        server.register_function(self.add_image)
        server.register_function(self.reset_all)
        server.register_function(self.save_to_h5_file)
        server.register_function(self.open_h5_files)
        server.register_function(self.import_h5_file)
        server.register_function(self.calc)

    def run(self):
        """Thread execution method"""
        self.serve()

    def codraft_is_ready(self):
        """Called when CodraFT is ready to process new requests"""
        self.is_ready = True

    @remote_call
    def switch_to_signal_panel(self):
        """Switch to signal panel"""
        self.SIG_SWITCH_TO_SIGNAL_PANEL.emit()

    @remote_call
    def switch_to_image_panel(self):
        """Switch to image panel"""
        self.SIG_SWITCH_TO_IMAGE_PANEL.emit()

    @remote_call
    def reset_all(self):
        """Reset all application data"""
        self.SIG_RESET_ALL.emit()

    @remote_call
    def save_to_h5_file(self, filename: str):
        """Save to a CodraFT HDF5 file"""
        self.SIG_SAVE_TO_H5.emit(filename)

    @remote_call
    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ):
        """Open a CodraFT HDF5 file or import from any other HDF5 file"""
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    @remote_call
    def import_h5_file(self, filename: str, reset_all: bool = None):
        """Open CodraFT HDF5 browser to Import HDF5 file"""
        self.SIG_IMPORT_H5.emit(filename, reset_all)

    @remote_call
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

    @remote_call
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
    ):  # pylint: disable=too-many-arguments
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

    @remote_call
    def calc(self, name: str, param: gdt.DataSet = None):
        """Call compute function `name` in current panel's processor"""
        self.SIG_CALC.emit(name, param)


def test():
    """Remote server test"""
    # win = DummyCodraFTWindow()
    # with qt_app_context() as qapp:
    with codraft_app_context(console=False) as win:
        server_thread = RPCServerThread(win)
        server_thread.start()
        while server_thread.port is None:
            time.sleep(0.1)
        port = server_thread.port
        Conf.main.rpc_server_port.set(port)
        print(f"Port: {port}")


if __name__ == "__main__":
    test()
