# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
import warnings
from collections.abc import Callable
from io import BytesIO
from typing import TYPE_CHECKING
from xmlrpc.client import Binary, ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

import guidata.dataset as gds
import numpy as np
from guidata.io import JSONReader, JSONWriter
from packaging.version import Version
from qtpy import QtCore as QC

import cdl
from cdl.adapters_plotpy import items_to_json, json_to_items
from cdl.baseproxy import AbstractCDLControl, BaseProxy
from cdl.config import Conf, initialize
from cdl.env import execenv
from sigima_.obj.image import ImageObj, create_image
from sigima_.obj.signal import SignalObj, create_signal

if TYPE_CHECKING:
    from cdl.gui.main import CDLMainWindow

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
    SIG_RAISE_WINDOW = QC.Signal()
    SIG_ADD_OBJECT = QC.Signal(object, str, bool)
    SIG_ADD_GROUP = QC.Signal(str, str, bool)
    SIG_LOAD_FROM_FILES = QC.Signal(list)
    SIG_LOAD_FROM_DIRECTORY = QC.Signal(str)
    SIG_SELECT_OBJECTS = QC.Signal(list, str)
    SIG_SELECT_GROUPS = QC.Signal(list, str)
    SIG_SELECT_ALL_GROUPS = QC.Signal(str)
    SIG_DELETE_METADATA = QC.Signal(bool, bool)
    SIG_SWITCH_TO_PANEL = QC.Signal(str)
    SIG_TOGGLE_AUTO_REFRESH = QC.Signal(bool)
    SIG_TOGGLE_SHOW_TITLES = QC.Signal(bool)
    SIG_RESET_ALL = QC.Signal()
    SIG_SAVE_TO_H5 = QC.Signal(str)
    SIG_OPEN_H5 = QC.Signal(list, bool, bool)
    SIG_IMPORT_H5 = QC.Signal(str, bool)
    SIG_CALC = QC.Signal(str, object)
    SIG_RUN_MACRO = QC.Signal(str)
    SIG_STOP_MACRO = QC.Signal(str)
    SIG_IMPORT_MACRO_FROM_FILE = QC.Signal(str)

    def __init__(self, win: CDLMainWindow) -> None:
        QC.QThread.__init__(self)
        self.port: int = None
        self.is_ready = True
        self.server: SimpleXMLRPCServer | None = None
        self.win = win
        win.SIG_READY.connect(self.cdl_is_ready)
        win.SIG_CLOSING.connect(self.shutdown_server)
        self.SIG_CLOSE_APP.connect(win.close)
        self.SIG_RAISE_WINDOW.connect(win.raise_window)
        self.SIG_ADD_OBJECT.connect(win.add_object)
        self.SIG_ADD_GROUP.connect(win.add_group)
        self.SIG_LOAD_FROM_FILES.connect(win.load_from_files)
        self.SIG_LOAD_FROM_DIRECTORY.connect(win.load_from_directory)
        self.SIG_SELECT_OBJECTS.connect(win.select_objects)
        self.SIG_SELECT_GROUPS.connect(win.select_groups)
        self.SIG_SELECT_ALL_GROUPS.connect(lambda panel: win.select_groups(None, panel))
        self.SIG_DELETE_METADATA.connect(win.delete_metadata)
        self.SIG_SWITCH_TO_PANEL.connect(win.set_current_panel)
        self.SIG_TOGGLE_AUTO_REFRESH.connect(win.toggle_auto_refresh)
        self.SIG_TOGGLE_SHOW_TITLES.connect(win.toggle_show_titles)
        self.SIG_RESET_ALL.connect(win.reset_all)
        self.SIG_SAVE_TO_H5.connect(win.save_to_h5_file)
        self.SIG_OPEN_H5.connect(win.open_h5_files)
        self.SIG_IMPORT_H5.connect(win.import_h5_file)
        self.SIG_CALC.connect(win.calc)
        self.SIG_RUN_MACRO.connect(win.run_macro)
        self.SIG_STOP_MACRO.connect(win.stop_macro)
        self.SIG_IMPORT_MACRO_FROM_FILE.connect(win.import_macro_from_file)

    def serve(self) -> None:
        """Start server and serve forever"""
        with SimpleXMLRPCServer(
            ("127.0.0.1", 0), logRequests=False, allow_none=True
        ) as server:
            self.server = server
            server.register_introspection_functions()
            self.register_functions(server)
            self.port = server.server_address[1]
            self.notify_port(self.port)
            with execenv.context(xmlrpcport=self.port):
                server.serve_forever()

    def shutdown_server(self) -> None:
        """Shutdown server"""
        if self.server is not None:
            self.server.shutdown()
            self.server = None

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
        """Return DataLab public version"""
        return cdl.__version__

    def close_application(self) -> None:
        """Close DataLab application"""
        self.SIG_CLOSE_APP.emit()

    def raise_window(self) -> None:
        """Raise DataLab window"""
        self.SIG_RAISE_WINDOW.emit()

    @remote_call
    def get_current_panel(self) -> str:
        """Return current panel name.

        Returns:
            Panel name (valid values: 'signal', 'image', 'macro')
        """
        return self.win.get_current_panel()

    @remote_call
    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel: Panel name (valid values: 'signal', 'image', 'macro')
        """
        self.SIG_SWITCH_TO_PANEL.emit(panel)

    @remote_call
    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh state.

        Args:
            state: True to enable auto refresh, False to disable it
        """
        self.SIG_TOGGLE_AUTO_REFRESH.emit(state)

    @remote_call
    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show titles state.

        Args:
            state: True to enable show titles, False to disable it
        """
        self.SIG_TOGGLE_SHOW_TITLES.emit(state)

    @remote_call
    def reset_all(self) -> None:
        """Reset all application data"""
        self.SIG_RESET_ALL.emit()

    @remote_call
    def save_to_h5_file(self, filename: str) -> None:
        """Save to a DataLab HDF5 file.

        Args:
            filename: HDF5 file name (with extension .h5)
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
            h5files: HDF5 file names. Defaults to None.
            import_all: Import all objects from HDF5 file. Defaults to None.
            reset_all: Reset all application data. Defaults to None.
        """
        import_all = True if import_all is None else import_all
        reset_all = False if reset_all is None else reset_all
        self.SIG_OPEN_H5.emit(h5files, import_all, reset_all)

    @remote_call
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Open DataLab HDF5 browser to Import HDF5 file.

        Args:
            filename: HDF5 file name
            reset_all: Reset all application data. Defaults to None.
        """
        reset_all = False if reset_all is None else reset_all
        self.SIG_IMPORT_H5.emit(filename, reset_all)

    @remote_call
    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images).

        Args:
            filenames: list of file names
        """
        self.SIG_LOAD_FROM_FILES.emit(filenames)

    @remote_call
    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """
        self.SIG_LOAD_FROM_DIRECTORY.emit(path)

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
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title: Signal title
            xbinary: X data
            ybinary: Y data
            xunit: X unit. Defaults to None
            yunit: Y unit. Defaults to None
            xlabel: X label. Defaults to None
            ylabel: Y label. Defaults to None
            group_id: group id in which to add the signal. Defaults to ""
            set_current: if True, set the added signal as current

        Returns:
            True if successful
        """
        xdata = rpcbinary_to_array(xbinary)
        ydata = rpcbinary_to_array(ybinary)
        signal = create_signal(title, xdata, ydata)
        signal.xunit = xunit
        signal.yunit = yunit
        signal.xlabel = xlabel
        signal.ylabel = ylabel
        self.SIG_ADD_OBJECT.emit(signal, group_id, set_current)
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
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title: Image title
            zbinary: Z data
            xunit: X unit. Defaults to None
            yunit: Y unit. Defaults to None
            zunit: Z unit. Defaults to None
            xlabel: X label. Defaults to None
            ylabel: Y label. Defaults to None
            zlabel: Z label. Defaults to None
            group_id: group id in which to add the image. Defaults to ""
            set_current: if True, set the added image as current

        Returns:
            True if successful
        """
        data = rpcbinary_to_array(zbinary)
        image = create_image(title, data)
        image.xunit = xunit
        image.yunit = yunit
        image.zunit = zunit
        image.xlabel = xlabel
        image.ylabel = ylabel
        image.zlabel = zlabel
        self.SIG_ADD_OBJECT.emit(image, group_id, set_current)
        return True

    @remote_call
    def add_object(
        self, obj_data: list[str], group_id: str = "", set_current: bool = True
    ) -> bool:
        """Add object to DataLab.

        Args:
            obj_data: Object data
            group_id: Group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current

        Returns:
            True if successful
        """
        obj = json_to_dataset(obj_data)
        self.SIG_ADD_OBJECT.emit(obj, group_id, set_current)
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
    def add_group(
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """
        self.SIG_ADD_GROUP.emit(title, panel, select)

    @remote_call
    def select_objects(
        self,
        selection: list[int | str],
        panel: str | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """
        self.SIG_SELECT_OBJECTS.emit(selection, panel)

    @remote_call
    def select_groups(
        self, selection: list[int | str] | None = None, panel: str | None = None
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used. Defaults to None.
        """
        if selection is None:
            self.SIG_SELECT_ALL_GROUPS.emit(panel)
        else:
            self.SIG_SELECT_GROUPS.emit(selection, panel)

    @remote_call
    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """
        self.SIG_DELETE_METADATA.emit(refresh_plot, keep_roi)

    @remote_call
    def calc(self, name: str, param_data: list[str] | None = None) -> bool:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name: Compute function name
            param_data: Compute function parameters. Defaults to None.

        Returns:
            True if successful, False otherwise
        """
        if param_data is None:
            param = None
        else:
            param = json_to_dataset(param_data)
        self.SIG_CALC.emit(name, param)
        return True

    @remote_call
    def get_group_titles_with_object_infos(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Groups titles, lists of inner objects uuids and titles
        """
        return self.win.get_group_titles_with_object_infos()

    @remote_call
    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

        Returns:
            List of object titles
        """
        return self.win.get_object_titles(panel)

    @remote_call
    def get_object(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list[str]:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """
        obj = self.win.get_object(nb_id_title, panel)
        if obj is None:
            return None
        return dataset_to_json(obj)

    @remote_call
    def get_object_uuids(
        self, panel: str | None = None, group: int | str | None = None
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            Object uuids
        """
        return self.win.get_object_uuids(panel, group)

    @remote_call
    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """
        items = self.win.get_object_shapes(nb_id_title, panel)
        return items_to_json(items)

    @remote_call
    def add_annotations_from_items(
        self, items_json: str, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items_json: JSON string of annotation items
            refresh_plot: refresh plot. Defaults to True.
            panel: panel name (valid values: "signal", "image").
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
            title: Label title. Defaults to None.
             If None, the title is the object title.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
        """
        self.win.add_label_with_title(title, panel)

    @remote_call
    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (run
             current macro, or does nothing if there is no macro).
        """
        self.SIG_RUN_MACRO.emit(number_or_title)

    @remote_call
    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (stop
             current macro, or does nothing if there is no macro).
        """
        self.SIG_STOP_MACRO.emit(number_or_title)

    @remote_call
    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """
        self.SIG_IMPORT_MACRO_FROM_FILE.emit(filename)


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
    #  The following is valid only when using Python 3.9+ with DataLab
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

        >>> from cdl.core.remote import RemoteClient
        >>> proxy = RemoteClient()
        >>> proxy.connect()
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy["toto"]
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1]
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1].data
        array([1., 2., 3.])
    """

    def __init__(self) -> None:
        super().__init__()
        self.port: str | None = None
        self._cdl: ServerProxy

    def set_port(self, port: str | None = None) -> None:
        """Set XML-RPC port to connect to.

        Args:
            port: XML-RPC port to connect to. If None, the port is automatically
             retrieved from DataLab configuration.
        """
        execenv.print(f"Setting XML-RPC port... [input:{port}] ", end="")
        port_str = ""
        if port is None:
            port = execenv.xmlrpcport
            port_str = f"→[execenv.xmlrpcport:{port}] "
            if port is None:
                port = get_cdl_xmlrpc_port()
                port_str = f"→[Conf.main.rpc_server_port:{port}] "
        execenv.print(port_str, end="")
        self.port = port
        if port is None:
            execenv.print("KO")
            raise ConnectionRefusedError("DataLab XML-RPC port is not set")
        execenv.print("OK")

    def __connect_to_server(self) -> None:
        """Connect to DataLab XML-RPC server.

        Args:
            port: XML-RPC port to connect to.

        Raises:
            ConnectionRefusedError: DataLab is currently not running
        """
        self._cdl = ServerProxy(f"http://127.0.0.1:{self.port}", allow_none=True)
        try:
            version = self.get_version()
        except ConnectionRefusedError as exc:
            raise ConnectionRefusedError("DataLab is currently not running") from exc
        # If DataLab version is not compatible with this client, show a warning
        server_ver = Version(version)
        client_ver = Version(cdl.__version__)
        if server_ver < client_ver:
            warnings.warn(
                f"DataLab server version ({server_ver}) may not be fully compatible "
                f"with this DataLab client version ({client_ver}).\n"
                f"Please upgrade the server to {client_ver} or higher."
            )

    def connect(
        self,
        port: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        """Try to connect to DataLab XML-RPC server.

        Args:
            port: XML-RPC port to connect to. If not specified,
             the port is automatically retrieved from DataLab configuration.
            timeout: Timeout in seconds. Defaults to 5.0.
            retries: Number of retries. Defaults to 10.

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
        self.set_port(port)
        execenv.print(f"Connecting to DataLab XML-RPC server... [port:{port}] ", end="")
        for _index in range(retries):
            try:
                self.__connect_to_server()
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

    def is_connected(self) -> bool:
        """Return True if connected to DataLab XML-RPC server."""
        if self._cdl is not None:
            try:
                self.get_version()
                return True
            except ConnectionRefusedError:
                self._cdl = None
        return False

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
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title: Signal title
            xdata: X data
            ydata: Y data
            xunit: X unit. Defaults to None
            yunit: Y unit. Defaults to None
            xlabel: X label. Defaults to None
            ylabel: Y label. Defaults to None
            group_id: group id in which to add the signal. Defaults to ""
            set_current: if True, set the added signal as current

        Returns:
            True if signal was added successfully, False otherwise

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
            title, xbinary, ybinary, xunit, yunit, xlabel, ylabel, group_id, set_current
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
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title: Image title
            data: Image data
            xunit: X unit. Defaults to None
            yunit: Y unit. Defaults to None
            zunit: Z unit. Defaults to None
            xlabel: X label. Defaults to None
            ylabel: Y label. Defaults to None
            zlabel: Z label. Defaults to None
            group_id: group id in which to add the image. Defaults to ""
            set_current: if True, set the added image as current

        Returns:
            True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        obj = ImageObj()
        obj.data = data
        obj.check_data()
        zbinary = array_to_rpcbinary(data)
        return self._cdl.add_image(
            title,
            zbinary,
            xunit,
            yunit,
            zunit,
            xlabel,
            ylabel,
            zlabel,
            group_id,
            set_current,
        )

    def add_object(
        self,
        obj: SignalObj | ImageObj,
        group_id: str = "",
        set_current: bool = True,
    ) -> None:
        """Add object to DataLab.

        Args:
            obj: Signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current
        """
        obj_data = dataset_to_json(obj)
        self._cdl.add_object(obj_data, group_id, set_current)

    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """
        if param is None:
            return self._cdl.calc(name)
        return self._cdl.calc(name, dataset_to_json(param))

    def get_object(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """
        param_data = self._cdl.get_object(nb_id_title, panel)
        if param_data is None:
            return None
        return json_to_dataset(param_data)

    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """
        items_json = self._cdl.get_object_shapes(nb_id_title, panel)
        return json_to_items(items_json)

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
            refresh_plot: refresh plot. Defaults to True.
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.
        """
        items_json = items_to_json(items)
        if items_json is not None:
            self._cdl.add_annotations_from_items(items_json, refresh_plot, panel)
