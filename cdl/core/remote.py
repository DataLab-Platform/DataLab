# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab remote control
----------------------

This module provides utilities to control DataLab from a Python script (e.g. with
Spyder) or from a Jupyter notebook.

The :class:`RemoteClient` class provides the main interface to DataLab XML-RPC server.

.. autoclass:: RemoteClient
    :members:
"""

from __future__ import annotations

import functools
import importlib
import sys
import threading
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
from cdl.core.baseproxy import BaseProxy
from cdl.core.io.native import NativeJSONReader, NativeJSONWriter
from cdl.core.model.image import ImageObj, create_image
from cdl.core.model.signal import SignalObj, create_signal
from cdl.env import execenv

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.gui.main import CDLMainWindow

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code


def array_to_rpcbinary(data: np.ndarray) -> Binary:
    """Convert NumPy array to XML-RPC Binary object, with shape and dtype.

    The array is converted to a binary string using NumPy's native binary
    format.

    Args:
        data: NumPy array to convert

    Returns:
        XML-RPC Binary object
    """
    dbytes = BytesIO()
    np.save(dbytes, data, allow_pickle=False)
    return Binary(dbytes.getvalue())


def rpcbinary_to_array(binary: Binary) -> np.ndarray:
    """Convert XML-RPC binary to NumPy array.

    Args:
        binary: XML-RPC Binary object

    Returns:
        NumPy array
    """
    dbytes = BytesIO(binary.data)
    return np.load(dbytes, allow_pickle=False)


def dataset_to_json(param: gdt.DataSet) -> list[str]:
    """Convert guidata DataSet to JSON data.

    The JSON data is a list of three elements:

    - The first element is the module name of the DataSet class
    - The second element is the class name of the DataSet class
    - The third element is the JSON data of the DataSet instance

    Args:
        param: guidata DataSet to convert

    Returns:
        JSON data
    """
    writer = NativeJSONWriter()
    param.serialize(writer)
    param_json = writer.get_json()
    klass = param.__class__
    return [klass.__module__, klass.__name__, param_json]


def json_to_dataset(param_data: list[str]) -> gdt.DataSet:
    """Convert JSON data to guidata DataSet.

    Args:
        param_data: JSON data

    Returns:
        guidata DataSet
    """
    param_module, param_clsname, param_json = param_data
    mod = importlib.__import__(param_module, fromlist=[param_clsname])
    klass = getattr(mod, param_clsname)
    param = klass()
    reader = NativeJSONReader(param_json)
    param.deserialize(reader)
    return param


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


# Note: RemoteServer can't inherit from AbstractCDLControl because it is a QThread
# and most of the methods are not returning expected data types


class RemoteServer(QC.QThread):
    """XML-RPC server QThread"""

    SIG_SERVER_PORT = QC.Signal(int)
    SIG_CLOSE_APP = QC.Signal()
    SIG_ADD_OBJECT = QC.Signal(object)
    SIG_OPEN_OBJECT = QC.Signal(str)
    SIG_SWITCH_TO_PANEL = QC.Signal(str)
    SIG_SWITCH_TO_IMAGE_PANEL = QC.Signal()
    SIG_RESET_ALL = QC.Signal()
    SIG_SAVE_TO_H5 = QC.Signal(str)
    SIG_OPEN_H5 = QC.Signal(list, bool, bool)
    SIG_IMPORT_H5 = QC.Signal(str, bool)
    SIG_CALC = QC.Signal(str, object)

    def __init__(self, win: CDLMainWindow) -> None:
        QC.QThread.__init__(self)
        self.port: int = None
        self.is_ready = True
        self.win = win
        win.SIG_READY.connect(self.cdl_is_ready)
        self.SIG_CLOSE_APP.connect(win.close)
        self.SIG_ADD_OBJECT.connect(win.add_object)
        self.SIG_OPEN_OBJECT.connect(win.open_object)
        self.SIG_SWITCH_TO_PANEL.connect(win.switch_to_panel)
        self.SIG_RESET_ALL.connect(win.reset_all)
        self.SIG_SAVE_TO_H5.connect(win.save_to_h5_file)
        self.SIG_OPEN_H5.connect(win.open_h5_files)
        self.SIG_IMPORT_H5.connect(win.import_h5_file)
        self.SIG_CALC.connect(win.calc)

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
            execenv.xmlrpcport = self.port
            server.serve_forever()

    def notify_port(self, port: int) -> None:
        """Notify automatically attributed port.

        This method is called after the server port has been automatically
        attributed. It notifies the port number to the main window.

        Args:
            port: Server port number
        """
        self.SIG_SERVER_PORT.emit(port)

    def register_functions(self, server: SimpleXMLRPCServer) -> None:
        """Register functions"""
        server.register_function(self.close_application)
        server.register_function(self.switch_to_panel)
        server.register_function(self.add_signal)
        server.register_function(self.add_image)
        server.register_function(self.reset_all)
        server.register_function(self.save_to_h5_file)
        server.register_function(self.open_h5_files)
        server.register_function(self.import_h5_file)
        server.register_function(self.open_object)
        server.register_function(self.calc)
        server.register_function(self.get_object_titles)
        server.register_function(self.get_object_from_title)
        server.register_function(self.get_object)
        server.register_function(self.get_object_uuids)
        server.register_function(self.get_object_from_uuid)

    def run(self) -> None:
        """Thread execution method"""
        if "coverage" in sys.modules:
            # The following is required to make coverage work with threading
            # pylint: disable=protected-access
            sys.settrace(threading._trace_hook)
        self.serve()

    def cdl_is_ready(self) -> None:
        """Called when DataLab is ready to process new requests"""
        self.is_ready = True

    @staticmethod
    def get_version() -> str:
        """Return DataLab version"""
        return __version__

    def close_application(self) -> None:
        """Close DataLab application"""
        self.SIG_CLOSE_APP.emit()

    @remote_call
    def switch_to_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel (str): Panel name (valid values: 'signal', 'image', 'macro')
        """
        self.SIG_SWITCH_TO_PANEL.emit(panel)

    @remote_call
    def reset_all(self) -> None:
        """Reset all application data"""
        self.SIG_RESET_ALL.emit()

    @remote_call
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename (str): HDF5 file name (with extension .h5)
        """
        self.SIG_SAVE_TO_H5.emit(filename)

    @remote_call
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files (list[str], optional): HDF5 file names. Defaults to None.
            import_all (bool, optional): Import all objects from HDF5 file.
                Defaults to None.
            reset_all (bool, optional): Reset all application data. Defaults to None.
        """
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    @remote_call
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename (str): HDF5 file name
            reset_all (bool, optional): Reset all application data. Defaults to None.
        """
        self.SIG_IMPORT_H5.emit(filename, reset_all)

    @remote_call
    def open_object(self, filename: str) -> None:
        """Open object from file in current panel (signal/image).

        Args:
            filename (str): File name
        """
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
        """Add signal data to DataLab.

        Args:
            title (str): Signal title
            xbinary (Binary): X data
            ybinary (Binary): Y data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.

        Returns:
            bool: True if successful
        """
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
        """Add image data to DataLab.

        Args:
            title (str): Image title
            zbinary (Binary): Z data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            zunit (str, optional): Z unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.
            zlabel (str, optional): Z label. Defaults to None.

        Returns:
            bool: True if successful
        """
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
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param_data (list[str], optional): Compute function parameters.
                Defaults to None.

        Returns:
            bool: True if successful
        """
        if param_data is None:
            param = None
        else:
            param = json_to_dataset(param_data)
        self.SIG_CALC.emit(name, param)
        return True

    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.

        Args:
            panel (str, optional): Panel name. Defaults to None.

        Returns:
            list[str]: Object titles
        """
        return self.win.get_object_titles(panel)

    def get_object_from_title(self, title: str, panel: str | None = None) -> list[str]:
        """Get object (signal/image) from title.

        Args:
            title (str): Object title
            panel (str, optional): Panel name. Defaults to None.

        Returns:
            list[str]: Object data
        """
        return dataset_to_json(self.win.get_object_from_title(title, panel))

    def get_object(
        self,
        index: int | None = None,
        group_index: int | None = None,
        panel: str | None = None,
    ) -> list[str]:
        """Get object (signal/image) from index.

        Args:
            index (int): Object index in current panel. Defaults to None.
            group_index (int, optional): Group index. Defaults to None.
            panel (str, optional): Panel name. Defaults to None.

        If ``index`` is not specified, returns the currently selected object.
        If ``group_index`` is not specified, return an object from the current group.
        If ``panel`` is not specified, return an object from the current panel.

        Returns:
            list[str]: Object data

        Raises:
            IndexError: if object not found
        """
        return dataset_to_json(self.win.get_object(index, group_index, panel))

    def get_object_uuids(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.

        Args:
            panel (str, optional): Panel name. Defaults to None.

        Returns:
            list[str]: Object uuids
        """
        return self.win.get_object_uuids(panel)

    def get_object_from_uuid(self, oid: str, panel: str | None = None) -> list[str]:
        """Get object (signal/image) from uuid.

        Args:
            oid (str): Object uuid
            panel (str, optional): Panel name. Defaults to None.

        Returns:
            list[str]: Object data
        """
        return dataset_to_json(self.win.get_object_from_uuid(oid, panel))


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
    #  ``get_cdl_xmlrpc_port`` function from doc/remotecontrol_py27.py.
    initialize()
    try:
        return Conf.main.rpc_server_port.get()
    except RuntimeError as exc:
        raise CDLConnectionError("DataLab has not yet been executed") from exc


class RemoteClient(BaseProxy):
    """Object representing a proxy/client to DataLab XML-RPC server.
    This object is used to call DataLab functions from a Python script.

    Examples:
        Here is a simple example of how to use RemoteClient in a Python script
        or in a Jupyter notebook:

        >>> from cdl.remotecontrol import RemoteClient
        >>> cdl = RemoteClient()
        >>> cdl.connect()
        >>> cdl.get_version()
        '1.0.0'
        >>> cdl.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> cdl.get_object_titles()
        ['toto']
        >>> cdl.get_object_from_title("toto")
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> cdl.get_object(0)
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> cdl.get_object(0).data
        array([1., 2., 3.])
    """

    def __init__(self) -> None:
        super().__init__()
        self.port: str = None
        self._cdl: ServerProxy

    def __connect_to_server(self, port: str | None = None) -> None:
        """Connect to DataLab XML-RPC server.

        Args:
            port (str, optional): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.

        Raises:
            CDLConnectionError: DataLab is currently not running
        """
        if port is None:
            port = execenv.xmlrpcport
            if port is None:
                port = get_cdl_xmlrpc_port()
        self.port = port
        self._cdl = ServerProxy(f"http://127.0.0.1:{port}", allow_none=True)
        try:
            self.get_version()
        except ConnectionRefusedError as exc:
            raise CDLConnectionError("DataLab is currently not running") from exc

    def connect(
        self, port: str | None = None, timeout: float = 5.0, retries: int = 10
    ) -> None:
        """Try to connect to DataLab XML-RPC server.

        Args:
            port (str, optional): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.
            timeout (float, optional): Timeout in seconds. Defaults to 5.0.
            retries (int, optional): Number of retries. Defaults to 10.

        Raises:
            CDLConnectionError: Unable to connect to DataLab
            ValueError: Invalid timeout (must be >= 0.0)
            ValueError: Invalid number of retries (must be >= 1)
        """
        if timeout < 0.0:
            raise ValueError("timeout must be >= 0.0")
        if retries < 1:
            raise ValueError("retries must be >= 1")
        execenv.print("Connecting to DataLab XML-RPC server...", end="")
        for _index in range(retries):
            try:
                self.__connect_to_server(port=port)
                break
            except CDLConnectionError:
                time.sleep(timeout / retries)
        else:
            raise CDLConnectionError("Unable to connect to DataLab")
        execenv.print(f"OK (port: {self.port})")

    def disconnect(self) -> None:
        """Disconnect from DataLab XML-RPC server."""
        # This is not mandatory with XML-RPC, but if we change protocol in the
        # future, it may be useful to have a disconnect method.
        self._cdl = None

    def get_method_list(self) -> list[str]:
        """Return list of available methods."""
        return self._cdl.system.listMethods()

    # === Following methods should match the register functions in XML-RPC server

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
        """Add signal data to DataLab.

        Args:
            title (str): Signal title
            xdata (np.ndarray): X data
            ydata (np.ndarray): Y data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.

        Returns:
            bool: True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """
        obj = SignalObj()
        obj.set_xydata(xdata, ydata)
        obj.check_data()
        xbinary = array_to_rpcbinary(xdata)
        ybinary = array_to_rpcbinary(ydata)
        return self._cdl.add_signal(
            title, xbinary, ybinary, xunit, yunit, xlabel, ylabel
        )

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
        """Add image data to DataLab.

        Args:
            title (str): Image title
            data (np.ndarray): Image data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            zunit (str, optional): Z unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.
            zlabel (str, optional): Z label. Defaults to None.

        Returns:
            bool: True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        obj = ImageObj()
        obj.data = data
        obj.check_data()
        zbinary = array_to_rpcbinary(data)
        return self._cdl.add_image(
            title, zbinary, xunit, yunit, zunit, xlabel, ylabel, zlabel
        )

    def calc(self, name: str, param: gdt.DataSet | None = None) -> gdt.DataSet:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param (gdt.DataSet, optional): Compute function parameter. Defaults to None.

        Returns:
            gdt.DataSet: Compute function result
        """
        if param is None:
            return self._cdl.calc(name)
        return self._cdl.calc(name, dataset_to_json(param))

    def get_object_from_title(
        self, title: str, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from title

        Args:
            title (str): object
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            ValueError: if object not found
            ValueError: if panel not found
        """
        param_data = self._cdl.get_object_from_title(title, panel)
        return json_to_dataset(param_data)

    def get_object(
        self,
        index: int | None = None,
        group_index: int | None = None,
        panel: str | None = None,
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            index (int): Object index in current panel. Defaults to None.
            group_index (int, optional): Group index. Defaults to None.
            panel (str, optional): Panel name. Defaults to None.

        If ``index`` is not specified, returns the currently selected object.
        If ``group_index`` is not specified, return an object from the current group.
        If ``panel`` is not specified, return an object from the current panel.

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            IndexError: if object not found
        """
        param_data = self._cdl.get_object(index, group_index, panel)
        return json_to_dataset(param_data)

    def get_object_from_uuid(
        self, oid: str, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from uuid

        Args:
            oid (str): object uuid
            panel (str | None): panel name (valid values: "signal", "image").

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            ValueError: if object not found
            ValueError: if panel not found
        """
        param_data = self._cdl.get_object_from_uuid(oid, panel)
        return json_to_dataset(param_data)
