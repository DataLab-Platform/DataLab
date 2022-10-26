# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT remote controlling utilities
"""

import abc
import functools
import importlib
import time
from io import BytesIO
from typing import List
from xmlrpc.client import Binary, ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from guidata.dataset import datatypes as gdt
from qtpy import QtCore as QC

from codraft import __version__
from codraft.config import Conf, initialize
from codraft.core.io.base import NativeJSONReader, NativeJSONWriter
from codraft.core.model.image import create_image
from codraft.core.model.signal import SignalParam, create_signal

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code


def array_to_rpcbinary(data: np.ndarray) -> Binary:
    """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
    dbytes = BytesIO()
    np.save(dbytes, data, allow_pickle=False)
    return Binary(dbytes.getvalue())


def rpcbinary_to_array(binary: Binary) -> np.ndarray:
    """Convert XML-RPC binary to NumPy array"""
    dbytes = BytesIO(binary.data)
    return np.load(dbytes, allow_pickle=False)


def dataset_to_json(param: gdt.DataSet) -> List[str]:
    """Convert guidata DataSet to JSON data"""
    writer = NativeJSONWriter()
    param.serialize(writer)
    param_json = writer.get_json()
    klass = param.__class__
    return [klass.__module__, klass.__name__, param_json]


def json_to_dataset(param_data: List[str]) -> gdt.DataSet:
    """Convert JSON data to guidata DataSet"""
    param_module, param_clsname, param_json = param_data
    mod = importlib.__import__(param_module, fromlist=[param_clsname])
    klass = getattr(mod, param_clsname)
    param = klass()
    reader = NativeJSONReader(param_json)
    param.deserialize(reader)
    return param


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


class RemoteServerMeta(type(QC.QThread), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class RemoteServer(QC.QThread, BaseRPCServer, metaclass=RemoteServerMeta):
    """XML-RPC server QThread"""

    SIG_SERVER_PORT = QC.Signal(int)
    SIG_CLOSE_APP = QC.Signal()
    SIG_ADD_OBJECT = QC.Signal(object)
    SIG_OPEN_OBJECT = QC.Signal(str)
    SIG_SWITCH_TO_SIGNAL_PANEL = QC.Signal()
    SIG_SWITCH_TO_IMAGE_PANEL = QC.Signal()
    SIG_RESET_ALL = QC.Signal()
    SIG_SAVE_TO_H5 = QC.Signal(str)
    SIG_OPEN_H5 = QC.Signal(list, bool, bool)
    SIG_IMPORT_H5 = QC.Signal(str, bool)
    SIG_CALC = QC.Signal(str, object)

    def __init__(self, win):
        QC.QThread.__init__(self)
        BaseRPCServer.__init__(self)
        self.is_ready = True
        self.win = win
        win.SIG_READY.connect(self.codraft_is_ready)
        self.SIG_CLOSE_APP.connect(win.close)
        self.SIG_ADD_OBJECT.connect(win.add_object)
        self.SIG_OPEN_OBJECT.connect(win.open_object)
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
        server.register_function(self.close_application)
        server.register_function(self.switch_to_signal_panel)
        server.register_function(self.switch_to_image_panel)
        server.register_function(self.add_signal)
        server.register_function(self.add_image)
        server.register_function(self.reset_all)
        server.register_function(self.save_to_h5_file)
        server.register_function(self.open_h5_files)
        server.register_function(self.import_h5_file)
        server.register_function(self.open_object)
        server.register_function(self.calc)
        server.register_function(self.get_object_list)
        server.register_function(self.get_object)

    def run(self):
        """Thread execution method"""
        self.serve()

    def codraft_is_ready(self):
        """Called when CodraFT is ready to process new requests"""
        self.is_ready = True

    def close_application(self):
        """Close CodraFT application"""
        self.SIG_CLOSE_APP.emit()

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
    def open_object(self, filename: str) -> None:
        """Open object from file in current panel (signal/image)"""
        self.SIG_OPEN_OBJECT.emit(filename)

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
    def calc(self, name: str, param_data: List[str] = None):
        """Call compute function `name` in current panel's processor"""
        if param_data is None:
            param = None
        else:
            param = json_to_dataset(param_data)
        self.SIG_CALC.emit(name, param)
        return True

    def get_object_list(self) -> List[str]:
        """Get object (signal/image) list for current panel"""
        return self.win.get_object_list()

    def get_object(self, index: str) -> List[str]:
        """Get object (signal/image) at index for current panel"""
        return dataset_to_json(self.win.get_object(index))


# === Python 2.7 client side:
#
# # See doc/codraft_remotecontrol_py27.py for an almost complete Python 2.7
# # implementation of RemoteClient class
#
# import io
# from xmlrpclib import ServerProxy, Binary
# import numpy as np
# def array_to_binary(data):
#     """Convert NumPy array to XML-RPC Binary object, with shape and dtype"""
#     dbytes = io.BytesIO()
#     np.save(dbytes, data, allow_pickle=False)
#     return Binary(dbytes.getvalue())
# s = ServerProxy("http://127.0.0.1:8000")
# data = np.array([[3, 4, 5], [7, 8, 0]], dtype=np.uint16)
# s.add_image("toto", array_to_binary(data))


class CodraFTConnectionError(Exception):
    """Error when trying to connect to CodraFT XML-RPC server"""


def get_codraft_xmlrpc_port():
    """Return CodraFT current XML-RPC port"""
    #  The following is valid only when using Python 3.8+ with CodraFT installed on the
    #  client side. In any other situation, please use the `get_codraft_xmlrpc_port`
    #  function from doc/codraft_remotecontrol_py27.py.
    initialize()
    try:
        return Conf.main.rpc_server_port.get()
    except RuntimeError:
        raise CodraFTConnectionError("CodraFT has not yet been executed")


class RemoteClient:
    """Object representing a proxy/client to CodraFT XML-RPC server"""

    def __init__(self):
        self.port = None
        self.serverproxy = None

    def connect(self, port=None):
        """Connect to CodraFT XML-RPC server"""
        if port is None:
            port = get_codraft_xmlrpc_port()
        self.port = port
        self.serverproxy = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        try:
            self.get_version()
        except ConnectionRefusedError:
            raise CodraFTConnectionError("CodraFT is currently not running")

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
        return p.calc(name, dataset_to_json(param))

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

    def get_object_list(self) -> List[str]:
        """Get object (signal/image) list for current panel"""
        return self.serverproxy.get_object_list()

    def get_object(self, index: str):
        """Get object (signal/image) at index for current panel"""
        param_data = self.serverproxy.get_object(index)
        return json_to_dataset(param_data)
