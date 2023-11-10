# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab remote control
----------------------

This module provides utilities to control DataLab from a Python script (e.g. with
Spyder) or from a Jupyter notebook.

The :class:`RemoteClient` class provides the main interface to DataLab XML-RPC server.
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

import guidata.dataset as gds
import numpy as np
from guidata.dataset.io import JSONReader, JSONWriter
from qtpy import QtCore as QC

from cdlapp import __version__
from cdlapp.config import Conf, initialize
from cdlapp.core.baseproxy import AbstractCDLControl, BaseProxy
from cdlapp.core.model.base import items_to_json, json_to_items
from cdlapp.core.model.image import ImageObj, create_image
from cdlapp.core.model.signal import SignalObj, create_signal
from cdlapp.env import execenv

if TYPE_CHECKING:  # pragma: no cover
    from cdlapp.core.gui.main import CDLMainWindow

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


def dataset_to_json(param: gds.DataSet) -> list[str]:
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
    writer = JSONWriter()
    param.serialize(writer)
    param_json = writer.get_json()
    klass = param.__class__
    return [klass.__module__, klass.__name__, param_json]


def json_to_dataset(param_data: list[str]) -> gds.DataSet:
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
    reader = JSONReader(param_json)
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
    SIG_SELECT_OBJECTS = QC.Signal(list, int, str)
    SIG_SELECT_GROUPS = QC.Signal(list, str)
    SIG_DELETE_METADATA = QC.Signal(bool)
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
        self.SIG_SELECT_OBJECTS.connect(win.select_objects)
        self.SIG_SELECT_GROUPS.connect(win.select_groups)
        self.SIG_DELETE_METADATA.connect(win.delete_metadata)
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

    @classmethod
    def check_remote_functions(cls) -> None:
        """Check if all AbstractCDLControl methods are implemented in RemoteServer"""
        mlist = []
        for method in AbstractCDLControl.get_public_methods():
            if not hasattr(cls, method):
                mlist.append(method)
        if mlist:
            raise RuntimeError(f"{cls} is missing some methods: {','.join(mlist)}")

    def register_functions(self, server: SimpleXMLRPCServer) -> None:
        """Register functions"""
        for name in AbstractCDLControl.get_public_methods():
            server.register_function(getattr(self, name))

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
            h5files (list[str] | None): HDF5 file names. Defaults to None.
            import_all (bool | None): Import all objects from HDF5 file.
                Defaults to None.
            reset_all (bool | None): Reset all application data. Defaults to None.
        """
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    @remote_call
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename (str): HDF5 file name
            reset_all (bool | None): Reset all application data. Defaults to None.
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
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.

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
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            zunit (str | None): Z unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.
            zlabel (str | None): Z label. Defaults to None.

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
    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """
        return self.win.get_sel_object_uuids(include_groups)

    @remote_call
    def select_objects(
        self,
        selection: list[int | str],
        group_num: int | None = None,
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection (list[int|str]): List of object indices or object uuids to select
            group_num (int | None): Group number. Defaults to None.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used. Defaults to None.
        """
        self.SIG_SELECT_OBJECTS.emit(selection, group_num, panel)

    @remote_call
    def select_groups(
        self, selection: list[int | str], panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection (list[int|str]): List of group indices or group uuids to select
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used. Defaults to None.
        """
        self.SIG_SELECT_GROUPS.emit(selection, panel)

    @remote_call
    def delete_metadata(self, refresh_plot: bool = True) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot (bool | None): Refresh plot. Defaults to True.
        """
        self.SIG_DELETE_METADATA.emit(refresh_plot)

    @remote_call
    def calc(self, name: str, param_data: list[str] | None = None) -> bool:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param_data (list[str] | None): Compute function parameters.
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

    @remote_call
    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.

        Args:
            panel (str | None): Panel name. Defaults to None.

        Returns:
            list[str]: Object titles
        """
        return self.win.get_object_titles(panel)

    @remote_call
    def get_object_from_title(self, title: str, panel: str | None = None) -> list[str]:
        """Get object (signal/image) from title.

        Args:
            title (str): Object title
            panel (str | None): Panel name. Defaults to None.

        Returns:
            list[str]: Object data
        """
        return dataset_to_json(self.win.get_object_from_title(title, panel))

    @remote_call
    def get_object(
        self,
        index: int | None = None,
        group_index: int | None = None,
        panel: str | None = None,
    ) -> list[str]:
        """Get object (signal/image) from index.

        Args:
            index (int): Object index in current panel. Defaults to None.
            group_index (int | None): Group index. Defaults to None.
            panel (str | None): Panel name. Defaults to None.

        If ``index`` is not specified, returns the currently selected object.
        If ``group_index`` is not specified, return an object from the current group.
        If ``panel`` is not specified, return an object from the current panel.

        Returns:
            list[str]: Object data

        Raises:
            IndexError: if object not found
        """
        return dataset_to_json(self.win.get_object(index, group_index, panel))

    @remote_call
    def get_object_uuids(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.

        Args:
            panel (str | None): Panel name. Defaults to None.

        Returns:
            list[str]: Object uuids
        """
        return self.win.get_object_uuids(panel)

    @remote_call
    def get_object_from_uuid(self, oid: str, panel: str | None = None) -> list[str]:
        """Get object (signal/image) from uuid.

        Args:
            oid (str): Object uuid
            panel (str | None): Panel name. Defaults to None.

        Returns:
            list[str]: Object data
        """
        return dataset_to_json(self.win.get_object_from_uuid(oid, panel))

    @remote_call
    def add_annotations_from_items(
        self, items_json: str, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items_json (str): JSON string of annotation items
            refresh_plot (bool | None): refresh plot. Defaults to True.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        items = json_to_items(items_json)
        if items:
            self.win.add_annotations_from_items(items, refresh_plot, panel)

    @remote_call
    def add_label_with_title(
        self, title: str | None = None, panel: str | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title (str | None): Label title. Defaults to None.
                If None, the title is the object title.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        self.win.add_label_with_title(title, panel)


RemoteServer.check_remote_functions()


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


def get_cdl_xmlrpc_port() -> str:
    """Return DataLab current XML-RPC port

    Returns:
        XML-RPC port

    Raises:
        ConnectionRefusedError: DataLab has not yet been executed
    """
    #  The following is valid only when using Python 3.8+ with DataLab
    #  installed on the client side. In any other situation, please use the
    #  ``get_cdl_xmlrpc_port`` function from doc/remotecontrol_py27.py.
    initialize()
    try:
        return Conf.main.rpc_server_port.get()
    except RuntimeError as exc:
        raise ConnectionRefusedError("DataLab has not yet been executed") from exc


class RemoteClient(BaseProxy):
    """Object representing a proxy/client to DataLab XML-RPC server.
    This object is used to call DataLab functions from a Python script.

    Examples:
        Here is a simple example of how to use RemoteClient in a Python script
        or in a Jupyter notebook:

        >>> from cdlapp.remotecontrol import RemoteClient
        >>> proxy = RemoteClient()
        >>> proxy.connect()
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy.get_object_from_title("toto")
        <cdlapp.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy.get_object(0)
        <cdlapp.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy.get_object(0).data
        array([1., 2., 3.])
    """

    def __init__(self) -> None:
        super().__init__()
        self.port: str = None
        self._cdl: ServerProxy

    def __connect_to_server(self, port: str | None = None) -> None:
        """Connect to DataLab XML-RPC server.

        Args:
            port (str | None): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.

        Raises:
            ConnectionRefusedError: DataLab is currently not running
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
            raise ConnectionRefusedError("DataLab is currently not running") from exc

    def connect(
        self,
        port: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        """Try to connect to DataLab XML-RPC server.

        Args:
            port (str | None): XML-RPC port to connect to. If not specified,
                the port is automatically retrieved from DataLab configuration.
            timeout (float | None): Timeout in seconds. Defaults to 5.0.
            retries (int | None): Number of retries. Defaults to 10.

        Raises:
            ConnectionRefusedError: Unable to connect to DataLab
            ValueError: Invalid timeout (must be >= 0.0)
            ValueError: Invalid number of retries (must be >= 1)
        """
        timeout = 5.0 if timeout is None else timeout
        retries = 10 if retries is None else retries
        if timeout < 0.0:
            raise ValueError("timeout must be >= 0.0")
        if retries < 1:
            raise ValueError("retries must be >= 1")
        execenv.print("Connecting to DataLab XML-RPC server...", end="")
        for _index in range(retries):
            try:
                self.__connect_to_server(port=port)
                break
            except ConnectionRefusedError:
                time.sleep(timeout / retries)
        else:
            execenv.print("KO")
            raise ConnectionRefusedError("Unable to connect to DataLab")
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
            xdata (numpy.ndarray): X data
            ydata (numpy.ndarray): Y data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.

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
            data (numpy.ndarray): Image data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            zunit (str | None): Z unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.
            zlabel (str | None): Z label. Defaults to None.

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

    def calc(self, name: str, param: gds.DataSet | None = None) -> gds.DataSet:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param (guidata.dataset.DataSet | None): Compute function
             parameter. Defaults to None.

        Returns:
            guidata.dataset.DataSet: Compute function result
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
            group_index (int | None): Group index. Defaults to None.
            panel (str | None): Panel name. Defaults to None.

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

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items (list): annotation plot items
            refresh_plot (bool | None): refresh plot. Defaults to True.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        items_json = items_to_json(items)
        if items_json is not None:
            self._cdl.add_annotations_from_items(items_json, refresh_plot, panel)

    # ----- Proxy specific methods ------------------------------------------------
    # (not available symetrically in AbstractCDLControl)

    def add_object(self, obj: SignalObj | ImageObj) -> None:
        """Add object to DataLab.

        Args:
            obj (SignalObj | ImageObj): Signal or image object
        """

        # TODO [P1]: Would it be better to use directly a remote "add_object" method?
        #     This would require to implement the add_object method in the
        #     XML-RPC server. And first of all, to check if performance is
        #     really better or not. This is equivalent to comparing the performance
        #     between JSON transfer (using "json_to_dataset") and binary transfer
        #     (using "array_to_rpcbinary") through XML-RPC.
        #
        #     If it is better, then here is what should be done:
        #     - Implement add_object method in AbstractCDLProcessor instead of in
        #       BaseProxy
        #     - Implement add_object method in XML-RPC server
        #     - Remove add_object method from BaseProxy
        #     - Rewrite add_object method in CDLProxy and RemoteClient to use
        #       the remote method

        if isinstance(obj, SignalObj):
            self.add_signal(
                obj.title, obj.x, obj.y, obj.xunit, obj.yunit, obj.xlabel, obj.ylabel
            )
        elif isinstance(obj, ImageObj):
            self.add_image(
                obj.title,
                obj.data,
                obj.xunit,
                obj.yunit,
                obj.zunit,
                obj.xlabel,
                obj.ylabel,
                obj.zlabel,
            )
