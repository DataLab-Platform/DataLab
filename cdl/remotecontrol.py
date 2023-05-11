# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab remote controlling utilities
"""

from __future__ import annotations

import abc
import functools
import importlib
import time
from collections.abc import Callable
from io import BytesIO
from typing import TYPE_CHECKING
from xmlrpc.client import Binary, ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from guidata.dataset import datatypes as gdt
from qtpy import QtCore as QC

from cdl import __version__
from cdl.config import Conf, initialize
from cdl.core.io.native import NativeJSONReader, NativeJSONWriter
from cdl.core.model.image import ImageParam, create_image
from cdl.core.model.signal import SignalParam, create_signal
from cdl.env import execenv

if TYPE_CHECKING:
    from cdl.core.gui.main import CDLMainWindow

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


def dataset_to_json(param: gdt.DataSet) -> list[str]:
    """Convert guidata DataSet to JSON data"""
    writer = NativeJSONWriter()
    param.serialize(writer)
    param_json = writer.get_json()
    klass = param.__class__
    return [klass.__module__, klass.__name__, param_json]


def json_to_dataset(param_data: list[str]) -> gdt.DataSet:
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

    def __init__(self) -> None:
        self.port: int = None

    def serve(self) -> None:
        """Start server and serve forever"""
        with SimpleXMLRPCServer(
            ("127.0.0.1", 0), logRequests=False, allow_none=True
        ) as server:
            server.register_introspection_functions()
            server.register_function(self.get_version)
            self.register_functions(server)
            self.port = server.server_address[1]
            self.notify_port(self.port)
            execenv.port = self.port
            server.serve_forever()

    @staticmethod
    def get_version() -> str:
        """Return DataLab version"""
        return __version__

    @abc.abstractmethod
    def notify_port(self, port: int) -> None:
        """Notify automatically attributed port"""

    @abc.abstractmethod
    def register_functions(self, server: SimpleXMLRPCServer) -> None:
        """Register functions"""


def remote_call(func: Callable) -> object:
    """Decorator for method calling DataLab main window remotely"""

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

    def __init__(self, win: CDLMainWindow) -> None:
        QC.QThread.__init__(self)
        BaseRPCServer.__init__(self)
        self.is_ready = True
        self.win = win
        win.SIG_READY.connect(self.cdl_is_ready)
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

    def notify_port(self, port: int) -> None:
        """Notify automatically attributed port"""
        self.SIG_SERVER_PORT.emit(port)

    def register_functions(self, server: SimpleXMLRPCServer) -> None:
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
        server.register_function(self.get_object_titles)
        server.register_function(self.get_object_uuids)
        server.register_function(self.get_object_from_uuid)

    def run(self) -> None:
        """Thread execution method"""
        self.serve()

    def cdl_is_ready(self) -> None:
        """Called when DataLab is ready to process new requests"""
        self.is_ready = True

    def close_application(self) -> None:
        """Close DataLab application"""
        self.SIG_CLOSE_APP.emit()

    @remote_call
    def switch_to_signal_panel(self) -> None:
        """Switch to signal panel"""
        self.SIG_SWITCH_TO_SIGNAL_PANEL.emit()

    @remote_call
    def switch_to_image_panel(self) -> None:
        """Switch to image panel"""
        self.SIG_SWITCH_TO_IMAGE_PANEL.emit()

    @remote_call
    def reset_all(self) -> None:
        """Reset all application data"""
        self.SIG_RESET_ALL.emit()

    @remote_call
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file"""
        self.SIG_SAVE_TO_H5.emit(filename)

    @remote_call
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file"""
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    @remote_call
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file"""
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
        xunit: str | None = None,
        yunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab"""
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
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab"""
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
    def calc(self, name: str, param_data: list[str] | None = None) -> bool:
        """Call compute function `name` in current panel's processor"""
        if param_data is None:
            param = None
        else:
            param = json_to_dataset(param_data)
        self.SIG_CALC.emit(name, param)
        return True

    def get_object_titles(self) -> list[str]:
        """Get object (signal/image) list for current panel"""
        return self.win.get_object_titles()

    def get_object_uuids(self) -> list[str]:
        """Get object (signal/image) list for current panel"""
        return self.win.get_object_uuids()

    def get_object_from_uuid(self, oid: str) -> list[str]:
        """Get object (signal/image) from uuid"""
        return dataset_to_json(self.win.get_object_from_uuid(oid))


# === Python 2.7 client side:
#
# # See doc/remotecontrol_py27.py for an almost complete Python 2.7
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


class CDLConnectionError(Exception):
    """Error when trying to connect to DataLab XML-RPC server"""


def get_cdl_xmlrpc_port():
    """Return DataLab current XML-RPC port"""
    #  The following is valid only when using Python 3.8+ with DataLab
    #  installed on the client side. In any other situation, please use the
    #  `get_cdl_xmlrpc_port` function from doc/remotecontrol_py27.py.
    initialize()
    try:
        return Conf.main.rpc_server_port.get()
    except RuntimeError as exc:
        raise CDLConnectionError("DataLab has not yet been executed") from exc


class RemoteClient:
    """Object representing a proxy/client to DataLab XML-RPC server"""

    def __init__(self) -> None:
        self.port: str = None
        self.serverproxy: ServerProxy = None

    def connect(self, port: str | None = None) -> None:
        """Connect to DataLab XML-RPC server"""
        if port is None:
            port = execenv.port
            if port is None:
                port = get_cdl_xmlrpc_port()
        self.port = port
        self.serverproxy = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        try:
            self.get_version()
        except ConnectionRefusedError as exc:
            raise CDLConnectionError("DataLab is currently not running") from exc

    def try_and_connect(self, port: str | None = None, timeout: int = 5) -> None:
        """Try (10 times over timeout in s.)
        and connect to DataLab XML-RPC server"""
        execenv.print("Connecting to DataLab XML-RPC server...", end="")
        retries = 10
        for _index in range(retries):
            try:
                self.connect(port=port)
                break
            except CDLConnectionError:
                time.sleep(timeout / retries)
        else:
            raise CDLConnectionError("Unable to connect to DataLab")
        execenv.print(f"OK (port: {self.port})")

    # === Following methods should match the register functions in XML-RPC server

    def get_version(self) -> str:
        """Return DataLab version"""
        return self.serverproxy.get_version()

    def close_application(self) -> None:
        """Close DataLab application"""
        self.serverproxy.close_application()

    def switch_to_signal_panel(self) -> None:
        """Switch to signal panel"""
        self.serverproxy.switch_to_signal_panel()

    def switch_to_image_panel(self) -> None:
        """Switch to image panel"""
        self.serverproxy.switch_to_image_panel()

    def reset_all(self) -> None:
        """Reset all application data"""
        self.serverproxy.reset_all()

    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file"""
        self.serverproxy.save_to_h5_file(filename)

    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file"""
        self.serverproxy.open_h5_files(h5files, import_all, reset_all)

    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file"""
        self.serverproxy.import_h5_file(filename, reset_all)

    def open_object(self, filename: str) -> None:
        """Open object from file in current panel (signal/image)"""
        self.serverproxy.open_object(filename)

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab"""
        xbinary = array_to_rpcbinary(xdata)
        ybinary = array_to_rpcbinary(ydata)
        p = self.serverproxy
        return p.add_signal(title, xbinary, ybinary, xunit, yunit, xlabel, ylabel)

    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab"""
        zbinary = array_to_rpcbinary(data)
        p = self.serverproxy
        return p.add_image(title, zbinary, xunit, yunit, zunit, xlabel, ylabel, zlabel)

    def calc(self, name: str, param: gdt.DataSet | None = None) -> gdt.DataSet:
        """Call compute function `name` in current panel's processor"""
        p = self.serverproxy
        if param is None:
            return p.calc(name)
        return p.calc(name, dataset_to_json(param))

    def add_object(self, obj: SignalParam | ImageParam) -> None:
        """Add object to DataLab"""
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

    def get_object_titles(self) -> list[str]:
        """Get object (signal/image) list for current panel"""
        return self.serverproxy.get_object_titles()

    def get_object_uuids(self) -> list[str]:
        """Get object (signal/image) list for current panel"""
        return self.serverproxy.get_object_uuids()

    def get_object_from_uuid(self, oid: str) -> SignalParam | ImageParam:
        """Get object (signal/image) from its UUID"""
        param_data = self.serverproxy.get_object_from_uuid(oid)
        return json_to_dataset(param_data)
