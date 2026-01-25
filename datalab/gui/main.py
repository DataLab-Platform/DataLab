# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Main window
===========

The :mod:`datalab.gui.main` module provides the main window of the
DataLab project.

.. autoclass:: DLMainWindow
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import base64
import functools
import os
import os.path as osp
import sys
import time
import webbrowser
from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from guidata import qthelpers as guidata_qth
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.console import DockableConsole
from plotpy import config as plotpy_config
from plotpy.builder import make
from plotpy.constants import PlotType
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilenames, getsavefilename
from sigima.config import options as sigima_options
from sigima.objects import ImageObj, SignalObj, create_image, create_signal

import datalab
from datalab import __docurl__, __homeurl__, __supporturl__, env
from datalab.adapters_metadata.common import have_geometry_results
from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import (
    APP_DESC,
    APP_NAME,
    DATAPATH,
    DEBUG,
    TEST_SEGFAULT_ERROR,
    Conf,
    _,
)
from datalab.control.baseproxy import AbstractDLControl
from datalab.control.remote import RemoteServer
from datalab.env import execenv
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.docks import DockablePlotWidget
from datalab.gui.h5io import H5InputOutput
from datalab.gui.panel import base, image, macro, signal
from datalab.gui.settings import edit_settings
from datalab.objectmodel import ObjectGroup
from datalab.plugins import PluginRegistry, discover_plugins, discover_v020_plugins
from datalab.utils import qthelpers as qth
from datalab.utils.qthelpers import (
    add_corner_menu,
    bring_to_front,
    configure_menu_about_to_show,
)
from datalab.webapi import WEBAPI_AVAILABLE, get_webapi_controller
from datalab.webapi.actions import WebApiActions
from datalab.widgets import instconfviewer, logviewer, status
from datalab.widgets.warningerror import go_to_error

if TYPE_CHECKING:
    from typing import Literal

    from datalab.gui.panel.base import AbstractPanel, BaseDataPanel
    from datalab.gui.panel.image import ImagePanel
    from datalab.gui.panel.macro import MacroPanel
    from datalab.gui.panel.signal import SignalPanel
    from datalab.plugins import PluginBase


def remote_controlled(func):
    """Decorator for remote-controlled methods"""

    @functools.wraps(func)
    def method_wrapper(*args, **kwargs):
        """Decorator wrapper function"""
        win = args[0]  # extracting 'self' from method arguments
        already_busy = not win.ready_flag
        win.ready_flag = False
        try:
            output = func(*args, **kwargs)
        finally:
            if not already_busy:
                win.SIG_READY.emit()
                win.ready_flag = True
            QW.QApplication.processEvents()
        return output

    return method_wrapper


class DLMainWindowMeta(type(QW.QMainWindow), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class DLMainWindow(QW.QMainWindow, AbstractDLControl, metaclass=DLMainWindowMeta):
    """DataLab main window

    Args:
        console: enable internal console
        hide_on_close: True to hide window on close
    """

    __instance = None

    SIG_READY = QC.Signal()
    SIG_SEND_OBJECT = QC.Signal(object)
    SIG_SEND_OBJECTLIST = QC.Signal(object)
    SIG_CLOSING = QC.Signal()

    @staticmethod
    def get_instance(console=None, hide_on_close=False):
        """Return singleton instance"""
        if DLMainWindow.__instance is None:
            return DLMainWindow(console, hide_on_close)
        return DLMainWindow.__instance

    def __init__(self, console=None, hide_on_close=False):
        """Initialize main window"""
        DLMainWindow.__instance = self
        super().__init__()
        self.setObjectName(APP_NAME)
        self.setWindowIcon(get_icon("DataLab.svg"))

        execenv.log(self, "Starting initialization")

        self.ready_flag = True

        self.hide_on_close = hide_on_close
        self.__old_size: tuple[int, int] | None = None
        self.__memory_warning = False
        self.memorystatus: status.MemoryStatus | None = None
        self.webapistatus: status.WebAPIStatus | None = None

        self.consolestatus: status.ConsoleStatus | None = None
        self.console: DockableConsole | None = None
        self.macropanel: MacroPanel | None = None

        self.main_toolbar: QW.QToolBar | None = None
        self.signalpanel_toolbar: QW.QToolBar | None = None
        self.imagepanel_toolbar: QW.QToolBar | None = None
        self.signalpanel: SignalPanel | None = None
        self.imagepanel: ImagePanel | None = None
        self.signalview: DockablePlotWidget | None = None
        self.imageview: DockablePlotWidget | None = None
        self.tabwidget: QW.QTabWidget | None = None
        self.tabmenu: QW.QMenu | None = None
        self.docks: dict[AbstractPanel | DockableConsole, QW.QDockWidget] | None = None
        self.h5inputoutput = H5InputOutput(self)
        self.webapi_actions: WebApiActions | None = None

        self.openh5_action: QW.QAction | None = None
        self.saveh5_action: QW.QAction | None = None
        self.browseh5_action: QW.QAction | None = None
        self.settings_action: QW.QAction | None = None
        self.quit_action: QW.QAction | None = None
        self.autorefresh_action: QW.QAction | None = None
        self.showfirstonly_action: QW.QAction | None = None
        self.showlabel_action: QW.QAction | None = None

        self.file_menu: QW.QMenu | None = None
        self.create_menu: QW.QMenu | None = None
        self.edit_menu: QW.QMenu | None = None
        self.roi_menu: QW.QMenu | None = None
        self.operation_menu: QW.QMenu | None = None
        self.processing_menu: QW.QMenu | None = None
        self.analysis_menu: QW.QMenu | None = None
        self.plugins_menu: QW.QMenu | None = None
        self.view_menu: QW.QMenu | None = None
        self.help_menu: QW.QMenu | None = None

        self.__update_color_mode(startup=True)

        self.__is_modified = False
        self.set_modified(False)

        # Starting XML-RPC server thread
        self.remote_server = RemoteServer(self)
        if Conf.main.rpc_server_enabled.get():
            self.remote_server.SIG_SERVER_PORT.connect(self.xmlrpc_server_started)
            self.remote_server.start()

        # Setup actions and menus
        if console is None:
            console = Conf.console.console_enabled.get()
        self.setup(console)

        self.__restore_pos_and_size()
        execenv.log(self, "Initialization done")

    # ------API related to XML-RPC remote control
    @staticmethod
    def xmlrpc_server_started(port):
        """XML-RPC server has started, writing comm port in configuration file"""
        Conf.main.rpc_server_port.set(port)

    def __get_current_basedatapanel(self) -> BaseDataPanel:
        """Return the current BaseDataPanel,
        or the signal panel if macro panel is active

        Returns:
            BaseDataPanel: current panel
        """
        panel = self.tabwidget.currentWidget()
        if not isinstance(panel, base.BaseDataPanel):
            panel = self.signalpanel
        return panel

    def __get_datapanel(
        self, panel: Literal["signal", "image"] | None
    ) -> BaseDataPanel:
        """Return a specific BaseDataPanel.

        Args:
            panel: panel name. If None, current panel is used.

        Returns:
            Panel widget

        Raises:
            ValueError: if panel is unknown
        """
        if not panel:
            return self.__get_current_basedatapanel()
        if panel == "signal":
            return self.signalpanel
        if panel == "image":
            return self.imagepanel
        raise ValueError(f"Unknown panel: {panel}")

    @remote_controlled
    def get_group_titles_with_object_info(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Groups titles, lists of inner objects uuids and titles
        """
        panel = self.__get_current_basedatapanel()
        return panel.objmodel.get_group_titles_with_object_info()

    @remote_controlled
    def get_object_titles(
        self, panel: Literal["signal", "image", "macro"] | None = None
    ) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name. If None, current data panel is used (i.e. signal or
             image panel).

        Returns:
            List of object titles

        Raises:
            ValueError: if panel is unknown
        """
        if not panel or panel in ("signal", "image"):
            return self.__get_datapanel(panel).objmodel.get_object_titles()
        if panel == "macro":
            return self.macropanel.get_macro_titles()
        raise ValueError(f"Unknown panel: {panel}")

    @remote_controlled
    def get_object(
        self,
        nb_id_title: int | str | None = None,
        panel: Literal["signal", "image"] | None = None,
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
            TypeError: if index_id_title type is invalid
        """
        panelw = self.__get_datapanel(panel)
        if nb_id_title is None:
            return panelw.objview.get_current_object()
        if isinstance(nb_id_title, int):
            return panelw.objmodel.get_object_from_number(nb_id_title)
        if isinstance(nb_id_title, str):
            try:
                return panelw.objmodel[nb_id_title]
            except KeyError:
                try:
                    return panelw.objmodel.get_object_from_title(nb_id_title)
                except KeyError as exc:
                    raise KeyError(
                        f"Invalid object index, id or title: {nb_id_title}"
                    ) from exc
        raise TypeError(f"Invalid index_id_title type: {type(nb_id_title)}")

    def find_object_by_uuid(
        self, uuid: str
    ) -> SignalObj | ImageObj | ObjectGroup | None:
        """Find an object by UUID, searching across all panels.

        This method searches for an object in both signal and image panels,
        making it suitable for cross-panel operations (e.g., radial profile that
        takes an ImageObj and produces a SignalObj).

        Difference from get_object():
        - get_object() requires specifying a panel and accepts number/id/title
        - find_object_by_uuid() searches all panels automatically using only UUID

        Args:
            uuid: UUID of the object to find

        Returns:
            The object if found in any panel, None otherwise
        """
        for panel in (self.signalpanel, self.imagepanel):
            if panel is not None:
                try:
                    return panel.objmodel[uuid]
                except KeyError:
                    continue
        return None

    @remote_controlled
    def get_object_uuids(
        self,
        panel: Literal["signal", "image"] | None = None,
        group: int | str | None = None,
    ) -> list[str]:
        """Get object (signal/image) uuid list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name. If None, current panel is used.
            group: Group number, or group id, or group title.
             Defaults to None (all groups).

        Returns:
            List of object uuids

        Raises:
            ValueError: if panel is unknown
        """
        objmodel = self.__get_datapanel(panel).objmodel
        if group is None:
            return objmodel.get_object_ids()
        if isinstance(group, int):
            grp = objmodel.get_group_from_number(group)
        else:
            try:
                grp = objmodel.get_group(group)
            except KeyError:
                grp = objmodel.get_group_from_title(group)
        if grp is None:
            raise KeyError(f"Invalid group index, id or title: {group}")
        return grp.get_object_ids()

    @remote_controlled
    def get_sel_object_uuids(self, include_groups: bool = False) -> list[str]:
        """Return selected objects uuids.

        Args:
            include_groups: If True, also return objects from selected groups.

        Returns:
            List of selected objects uuids.
        """
        panel = self.__get_current_basedatapanel()
        return panel.objview.get_sel_object_uuids(include_groups)

    @remote_controlled
    def add_group(
        self,
        title: str,
        panel: Literal["signal", "image"] | None = None,
        select: bool = False,
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name. Defaults to None.
            select: Select the group after creation. Defaults to False.
        """
        self.__get_datapanel(panel).add_group(title, select)

    @remote_controlled
    def select_objects(
        self,
        selection: list[int | str],
        panel: Literal["signal", "image"] | None = None,
    ) -> None:
        """Select objects in current panel.

        Args:
            selection: List of object numbers (1 to N) or uuids to select
            panel: panel name. If None, current panel is used. Defaults to None.
        """
        panel = self.__get_datapanel(panel)
        panel.objview.select_objects(selection)

    @remote_controlled
    def select_groups(
        self,
        selection: list[int | str] | None = None,
        panel: Literal["signal", "image"] | None = None,
    ) -> None:
        """Select groups in current panel.

        Args:
            selection: List of group numbers (1 to N), or list of group uuids,
             or None to select all groups. Defaults to None.
            panel: panel name. If None, current panel is used. Defaults to None.
        """
        panel = self.__get_datapanel(panel)
        panel.objview.select_groups(selection)

    @remote_controlled
    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool = False
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep ROI. Defaults to False.
        """
        panel = self.__get_current_basedatapanel()
        panel.delete_metadata(refresh_plot, keep_roi)

    @remote_controlled
    def call_method(
        self,
        method_name: str,
        *args,
        panel: Literal["signal", "image"] | None = None,
        **kwargs,
    ):
        """Call a public method on a panel or main window.

        This generic method allows calling any public method that is not explicitly
        exposed in the proxy API. The method resolution follows this order:

        1. If panel is specified: call method on that specific panel
        2. If panel is None:
           a. Try to call method on main window (DLMainWindow)
           b. If not found, try to call method on current panel (BaseDataPanel)

        This makes it convenient to call panel methods without specifying the panel
        parameter when working on the current panel.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments to pass to the method
            panel: Panel name ("signal", "image", or None for auto-detection).
             Defaults to None.
            **kwargs: Keyword arguments to pass to the method

        Returns:
            The return value of the called method

        Raises:
            AttributeError: If the method does not exist or is not public
            ValueError: If the panel name is invalid

        Examples:
            >>> # Call remove_object on current panel (auto-detected)
            >>> win.call_method("remove_object", force=True)
            >>> # Call a signal panel method specifically
            >>> win.call_method("delete_all_objects", panel="signal")
            >>> # Call main window method
            >>> win.call_method("get_current_panel")
        """
        # Security check: only allow public methods (not starting with _)
        if method_name.startswith("_"):
            raise AttributeError(
                f"Cannot call private method '{method_name}' through proxy"
            )

        # If panel is specified, use that panel directly
        if panel is not None:
            target = self.__get_datapanel(panel)
            if not hasattr(target, method_name):
                raise AttributeError(
                    f"Method '{method_name}' does not exist on {panel} panel"
                )
            method = getattr(target, method_name)
            if not callable(method):
                raise AttributeError(f"'{method_name}' is not a callable method")
            return method(*args, **kwargs)

        # Panel is None: try main window first, then current panel
        # Try main window first
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return method(*args, **kwargs)

        # Method not found on main window, try current panel
        current_panel = self.__get_current_basedatapanel()
        if hasattr(current_panel, method_name):
            method = getattr(current_panel, method_name)
            if callable(method):
                return method(*args, **kwargs)

        # Method not found anywhere
        raise AttributeError(
            f"Method '{method_name}' does not exist on main window or current panel"
        )

    @remote_controlled
    def call_method_slot(
        self,
        method_name: str,
        args: list,
        panel: Literal["signal", "image"] | None,
        kwargs: dict,
    ) -> None:
        """Slot to call a method from RemoteServer thread in GUI thread.

        This slot receives signals from RemoteServer and executes the method in
        the GUI thread, avoiding thread-safety issues with Qt widgets and dialogs.

        Args:
            method_name: Name of the method to call
            args: Positional arguments as a list
            panel: Panel name or None for auto-detection
            kwargs: Keyword arguments as a dict
        """
        # Call the method and store result in RemoteServer
        try:
            result = self.call_method(method_name, *args, panel=panel, **kwargs)
            # Store result in RemoteServer for retrieval by XML-RPC thread
            self.remote_server.result = result
            self.remote_server.exception = None  # Clear any previous exception
        except Exception as exc:  # pylint: disable=broad-except
            # Store exception for re-raising in XML-RPC thread
            self.remote_server.result = None
            self.remote_server.exception = exc

    @remote_controlled
    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: Literal["signal", "image"] | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """
        obj = self.get_object(nb_id_title, panel)
        return list(create_adapter_from_object(obj).iterate_shape_items(editable=False))

    @remote_controlled
    def add_annotations_from_items(
        self,
        items: list,
        refresh_plot: bool = True,
        panel: Literal["signal", "image"] | None = None,
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
            refresh_plot: refresh plot. Defaults to True.
            panel: panel name. If None, current panel is used.
        """
        panel = self.__get_datapanel(panel)
        panel.add_annotations_from_items(items, refresh_plot)

    @remote_controlled
    def add_label_with_title(
        self, title: str | None = None, panel: Literal["signal", "image"] | None = None
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title: Label title. Defaults to None.
             If None, the title is the object title.
            panel: panel name. If None, current panel is used.
        """
        self.__get_datapanel(panel).add_label_with_title(title)

    @remote_controlled
    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run macro.

        Args:
             number: Number of the macro (starting at 1). Defaults to None (run
              current macro, or does nothing if there is no macro).
        """
        self.macropanel.run_macro(number_or_title)

    @remote_controlled
    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop macro.

        Args:
            number: Number of the macro (starting at 1). Defaults to None (stop
             current macro, or does nothing if there is no macro).
        """
        self.macropanel.stop_macro(number_or_title)

    @remote_controlled
    def import_macro_from_file(self, filename: str) -> None:
        """Import macro from file

        Args:
            filename: Filename.
        """
        self.macropanel.import_macro_from_file(filename)

    # ------WebAPI control
    @remote_controlled
    def start_webapi_server(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> dict:
        """Start the Web API server.

        Args:
            host: Host address to bind to. Defaults to "127.0.0.1".
            port: Port number. Defaults to auto-detect available port.

        Returns:
            Dictionary with "url" and "token" keys.

        Raises:
            RuntimeError: If Web API deps not installed or server already running.
        """
        if not WEBAPI_AVAILABLE:
            raise RuntimeError(
                "Web API dependencies not installed. "
                "Install with: pip install datalab-platform[webapi]"
            )

        controller = get_webapi_controller()
        controller.set_main_window(self)
        url, token = controller.start(host=host, port=port)
        return {"url": url, "token": token}

    @remote_controlled
    def stop_webapi_server(self) -> None:
        """Stop the Web API server."""
        if not WEBAPI_AVAILABLE:
            return

        controller = get_webapi_controller()
        controller.stop()

    @remote_controlled
    def get_webapi_status(self) -> dict:
        """Get Web API server status.

        Returns:
            Dictionary with "running", "url", and "token" keys.
        """
        if not WEBAPI_AVAILABLE:
            return {"running": False, "url": None, "token": None, "available": False}

        controller = get_webapi_controller()
        info = controller.get_connection_info()
        info["available"] = True
        return info

    # ------Misc.
    @property
    def panels(self) -> tuple[AbstractPanel, ...]:
        """Return the tuple of implemented panels (signal, image)

        Returns:
            Tuple of panels
        """
        return (self.signalpanel, self.imagepanel, self.macropanel)

    def __set_low_memory_state(self, state: bool) -> None:
        """Set memory warning state"""
        self.__memory_warning = state

    def __show_webapi_info(self) -> None:
        """Show Web API connection info when status widget is clicked."""
        if self.webapi_actions is not None:
            self.webapi_actions.show_connection_info()

    def __start_webapi_server(self) -> None:
        """Start Web API server when status widget is clicked."""
        if self.webapi_actions is not None:
            self.webapi_actions.start_server_from_status_widget()

    def confirm_memory_state(self) -> bool:  # pragma: no cover
        """Check memory warning state and eventually show a warning dialog

        Returns:
            True if memory state is ok
        """
        if not env.execenv.unattended and self.__memory_warning:
            threshold = Conf.main.available_memory_threshold.get()
            answer = QW.QMessageBox.critical(
                self,
                _("Warning"),
                _("Available memory is below %d MB.<br><br>Do you want to continue?")
                % threshold,
                QW.QMessageBox.Yes | QW.QMessageBox.No,
            )
            return answer == QW.QMessageBox.Yes
        return True

    def check_stable_release(self) -> None:  # pragma: no cover
        """Check if this is a stable release"""
        if datalab.__version__.replace(".", "").isdigit():
            # This is a stable release
            return
        if "b" in datalab.__version__:
            # This is a beta release
            rel = _(
                "This software is in the <b>beta stage</b> of its release cycle. "
                "The focus of beta testing is providing a feature complete "
                "software for users interested in trying new features before "
                "the final release. However, <u>beta software may not behave as "
                "expected and will probably have more bugs or performance issues "
                "than completed software</u>."
            )
        else:
            # This is an alpha release
            rel = _(
                "This software is in the <b>alpha stage</b> of its release cycle. "
                "The focus of alpha testing is providing an incomplete software "
                "for early testing of specific features by users. "
                "Please note that <u>alpha software was not thoroughly tested</u> "
                "by the developer before it is released."
            )
        txtlist = [
            f"<b>{APP_NAME}</b> v{datalab.__version__}:",
            "",
            _("<i>This is not a stable release.</i>"),
            "",
            rel,
        ]
        if not env.execenv.unattended:
            QW.QMessageBox.warning(
                self, APP_NAME, "<br>".join(txtlist), QW.QMessageBox.Ok
            )

    def check_for_previous_crash(self) -> None:  # pragma: no cover
        """Check for previous crash"""
        if execenv.unattended and not execenv.do_not_quit:
            # Showing the log viewer for testing purpose (unattended mode) but only
            # if option 'do_not_quit' is not set, to avoid blocking the test suite
            self.__show_logviewer()
        elif execenv.do_not_quit:
            # If 'do_not_quit' is set, we do not show any message box to avoid blocking
            # the test suite
            return
        elif Conf.main.faulthandler_log_available.get(
            False
        ) or Conf.main.traceback_log_available.get(False):
            txt = "<br>".join(
                [
                    logviewer.get_log_prompt_message(),
                    "",
                    _("Do you want to see available log files?"),
                ]
            )
            btns = QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No
            choice = QW.QMessageBox.warning(self, APP_NAME, txt, btns)
            if choice == QW.QMessageBox.StandardButton.Yes:
                self.__show_logviewer()

    def check_for_v020_plugins(self) -> None:  # pragma: no cover
        """Check for v0.20 plugins and warn user if any are found"""
        if Conf.main.v020_plugins_warning_ignore.get(False):
            return

        v020_plugins = discover_v020_plugins()
        if execenv.unattended or not v020_plugins:
            return

        # Build plugin list with clickable directory paths
        plugin_items = []
        for name, directory_path in v020_plugins:
            if directory_path:
                # Create clickable file:// link to directory
                dir_url = QC.QUrl.fromLocalFile(directory_path).toString()
                plugin_items.append(
                    f'<li>{name} (<a href="{dir_url}">{directory_path}</a>)</li>'
                )
            else:
                plugin_items.append(f"<li>{name}</li>")
        plugin_list = "<ul>" + "".join(plugin_items) + "</ul>"

        txtlist = [
            "<b>" + _("DataLab v0.20 plugins detected") + "</b>",
            "",
            _("The following plugins are using the old DataLab v0.20 format:"),
            plugin_list,
            _(
                "These plugins will <b>not be loaded</b> in DataLab v1.0 because "
                "they are not compatible with the new architecture."
            ),
            "",
            _(
                "To use these plugins with DataLab v1.0, you need to update them. "
                "Please refer to the migration guide on the DataLab website "
            )
            + '(<a href="https://datalab-platform.com/en/features/advanced/'
            'migration_v020_to_v100.html">Migration guide</a>)'
            + _(" or in the PDF documentation."),
            "",
            _("Choosing to ignore this message will prevent it from appearing again."),
        ]

        answer = QW.QMessageBox.question(
            self,
            APP_NAME,
            "<br>".join(txtlist),
            QW.QMessageBox.Ok | QW.QMessageBox.Ignore,
        )

        if answer == QW.QMessageBox.Ignore:
            Conf.main.v020_plugins_warning_ignore.set(True)

    def execute_post_show_actions(self) -> None:
        """Execute post-show actions"""
        self.check_stable_release()
        self.check_for_previous_crash()
        self.check_for_v020_plugins()
        tour = Conf.main.tour_enabled.get()
        if tour:
            Conf.main.tour_enabled.set(False)
            self.show_tour()

    def take_screenshot(self, name: str) -> None:  # pragma: no cover
        """Take main window screenshot"""
        # For esthetic reasons, we set the central widget width to a lower value:
        old_width = self.tabwidget.maximumWidth()
        self.tabwidget.setMaximumWidth(500)
        # To avoid having screenshot depending on memory status, we set demo mode ON:
        self.memorystatus.set_demo_mode(True)
        qth.grab_save_window(self, f"{name}")
        # Restore previous state:
        self.memorystatus.set_demo_mode(False)
        self.tabwidget.setMaximumWidth(old_width)

    def take_menu_screenshots(self) -> None:  # pragma: no cover
        """Take menu screenshots"""
        for panel in self.panels:
            if isinstance(panel, base.BaseDataPanel):
                self.tabwidget.setCurrentWidget(panel)
                for name in (
                    "file",
                    "create",
                    "edit",
                    "roi",
                    "view",
                    "operation",
                    "processing",
                    "analysis",
                    "help",
                ):
                    menu = getattr(self, f"{name}_menu")
                    menu.popup(self.pos())
                    qth.grab_save_window(menu, f"{panel.objectName()}_{name}")
                    menu.close()
                if panel in (self.signalpanel, self.imagepanel):
                    panel: BaseDataPanel
                    # Take screenshots of Edit menu submenus (Metadata and Annotations)
                    for submenu, suffix in (
                        (panel.acthandler.metadata_submenu, "_edit_metadata"),
                        (panel.acthandler.annotations_submenu, "_edit_annotations"),
                    ):
                        submenu.popup(self.pos())
                        qth.grab_save_window(submenu, f"{panel.objectName()}{suffix}")
                        submenu.close()

    # ------GUI setup
    def __restore_pos_and_size(self) -> None:
        """Restore main window position and size from configuration"""
        pos = Conf.main.window_position.get(None)
        if pos is not None:
            posx, posy = pos
            self.move(QC.QPoint(posx, posy))
        size = Conf.main.window_size.get(None)
        if size is None:
            sgeo = self.screen().availableGeometry()
            sw, sh = sgeo.width(), sgeo.height()
            w = max(1200, min(1800, int(sw * 0.8)))
            h = max(700, min(1100, int(sh * 0.8)))
            size = (w, h)
            if pos is None:
                cx = sgeo.x() + (sw - w) // 2
                cy = sgeo.y() + (sh - h) // 2
                self.move(QC.QPoint(cx, cy))
        width, height = size
        self.resize(QC.QSize(width, height))
        if pos is not None and size is not None:
            sgeo = self.screen().availableGeometry()
            out_inf = posx < -int(0.9 * width) or posy < -int(0.9 * height)
            out_sup = posx > int(0.9 * sgeo.width()) or posy > int(0.9 * sgeo.height())
            if len(QW.QApplication.screens()) == 1 and (out_inf or out_sup):
                #  Main window is offscreen
                posx = min(max(posx, 0), sgeo.width() - width)
                posy = min(max(posy, 0), sgeo.height() - height)
                self.move(QC.QPoint(posx, posy))

    def __restore_state(self) -> None:
        """Restore main window state from configuration"""
        state = Conf.main.window_state.get(None)
        if state is not None:
            state = base64.b64decode(state)
            self.restoreState(QC.QByteArray(state))
            for widget in self.children():
                if isinstance(widget, QW.QDockWidget):
                    self.restoreDockWidget(widget)

    def __save_pos_size_and_state(self) -> None:
        """Save main window position, size and state to configuration"""
        is_maximized = self.windowState() == QC.Qt.WindowMaximized
        Conf.main.window_maximized.set(is_maximized)
        if not is_maximized:
            size = self.size()
            Conf.main.window_size.set((size.width(), size.height()))
            pos = self.pos()
            Conf.main.window_position.set((pos.x(), pos.y()))
        # Encoding window state into base64 string to avoid sending binary data
        # to the configuration file:
        state = base64.b64encode(self.saveState().data()).decode("ascii")
        Conf.main.window_state.set(state)

    def setup(self, console: bool = False) -> None:
        """Setup main window

        Args:
            console: True to setup console
        """
        self.__register_plugins()
        self.__configure_statusbar(console)
        self.__setup_global_actions()
        self.__add_signal_image_panels()
        self.__create_plugins_actions()
        self.__setup_central_widget()
        self.__add_menus()
        self.__setup_webapi()
        if console:
            self.__setup_console()
        self.__update_actions(update_other_data_panel=True)
        self.__add_macro_panel()
        self.__configure_panels()
        # Now that everything is set up, we can restore the window state:
        self.__restore_state()

    def __setup_webapi(self) -> None:
        """Setup Web API actions."""
        self.webapi_actions = WebApiActions(self)
        # Note: Menu is added in __update_view_menu since view_menu is cleared each show

    def __register_plugins(self) -> None:
        """Register plugins"""
        with qth.try_or_log_error("Discovering plugins"):
            # Discovering plugins
            plugin_nb = len(discover_plugins())
            execenv.log(self, f"{plugin_nb} plugin(s) found")
        for plugin_class in PluginRegistry.get_plugin_classes():
            with qth.try_or_log_error(f"Instantiating plugin {plugin_class.__name__}"):
                # Instantiating plugin
                plugin: PluginBase = plugin_class()
            with qth.try_or_log_error(f"Registering plugin {plugin.info.name}"):
                # Registering plugin
                plugin.register(self)

    def __create_plugins_actions(self) -> None:
        """Create plugins actions"""
        with self.signalpanel.acthandler.new_category(ActionCategory.PLUGINS):
            with self.imagepanel.acthandler.new_category(ActionCategory.PLUGINS):
                for plugin in PluginRegistry.get_plugins():
                    with qth.try_or_log_error(f"Create actions for {plugin.info.name}"):
                        plugin.create_actions()

    @staticmethod
    def __unregister_plugins() -> None:
        """Unregister plugins"""
        with qth.try_or_log_error("Unregistering plugins"):
            PluginRegistry.unregister_all_plugins()

    def __configure_statusbar(self, console: bool) -> None:
        """Configure status bar

        Args:
            console: True if console is enabled
        """
        self.statusBar().showMessage(_("Welcome to %s!") % APP_NAME, 5000)
        if console:
            # Console status
            self.consolestatus = status.ConsoleStatus()
            self.statusBar().addPermanentWidget(self.consolestatus)
        # Plugin status
        pluginstatus = status.PluginStatus()
        self.statusBar().addPermanentWidget(pluginstatus)
        # XML-RPC server status
        xmlrpcstatus = status.XMLRPCStatus()
        xmlrpcstatus.set_port(self.remote_server.port)
        self.statusBar().addPermanentWidget(xmlrpcstatus)
        # Web API server status
        self.webapistatus = status.WebAPIStatus()
        self.webapistatus.SIG_SHOW_INFO.connect(self.__show_webapi_info)
        self.webapistatus.SIG_START_SERVER.connect(self.__start_webapi_server)
        self.statusBar().addPermanentWidget(self.webapistatus)
        # Memory status
        threshold = Conf.main.available_memory_threshold.get()
        self.memorystatus = status.MemoryStatus(threshold)
        self.memorystatus.SIG_MEMORY_ALARM.connect(self.__set_low_memory_state)
        self.statusBar().addPermanentWidget(self.memorystatus)

    def __add_toolbar(
        self, title: str, position: Literal["top", "bottom", "left", "right"], name: str
    ) -> QW.QToolBar:
        """Add toolbar to main window

        Args:
            title: toolbar title
            position: toolbar position
            name: toolbar name (Qt object name)
        """
        toolbar = QW.QToolBar(title, self)
        toolbar.setObjectName(name)
        area = getattr(QC.Qt, f"{position.capitalize()}ToolBarArea")
        self.addToolBar(area, toolbar)
        return toolbar

    def __setup_global_actions(self) -> None:
        """Setup global actions"""
        self.openh5_action = create_action(
            self,
            _("Open HDF5 files..."),
            icon=get_icon("fileopen_h5.svg"),
            tip=_("Open one or more HDF5 files"),
            triggered=lambda checked=False: self.open_h5_files(import_all=True),
        )
        self.saveh5_action = create_action(
            self,
            _("Save to HDF5 file..."),
            icon=get_icon("filesave_h5.svg"),
            tip=_("Save to HDF5 file"),
            triggered=self.save_to_h5_file,
        )
        self.browseh5_action = create_action(
            self,
            _("Browse HDF5 file..."),
            icon=get_icon("h5browser.svg"),
            tip=_("Browse an HDF5 file"),
            triggered=lambda checked=False: self.open_h5_files(import_all=None),
        )
        self.settings_action = create_action(
            self,
            _("Settings..."),
            icon=get_icon("libre-gui-settings.svg"),
            tip=_("Open settings dialog"),
            triggered=self.__edit_settings,
        )
        self.main_toolbar = self.__add_toolbar(
            _("Main Toolbar"), "left", "main_toolbar"
        )
        add_actions(
            self.main_toolbar,
            [
                self.openh5_action,
                self.saveh5_action,
                self.browseh5_action,
                None,
                self.settings_action,
            ],
        )
        # Quit action for "File menu" (added when populating menu on demand)
        if self.hide_on_close:
            quit_text = _("Hide window")
            quit_tip = _("Hide DataLab window")
        else:
            quit_text = _("Quit")
            quit_tip = _("Quit application")
        if sys.platform != "darwin":
            # On macOS, the "Quit" action is automatically added to the application menu
            self.quit_action = create_action(
                self,
                quit_text,
                shortcut=QG.QKeySequence(QG.QKeySequence.Quit),
                icon=get_icon("libre-gui-close.svg"),
                tip=quit_tip,
                triggered=self.close,
            )
        # View menu actions
        self.autorefresh_action = create_action(
            self,
            _("Auto-refresh"),
            icon=get_icon("refresh-auto.svg"),
            tip=_("Auto-refresh plot when object is modified, added or removed"),
            toggled=self.handle_autorefresh_action,
        )
        self.showfirstonly_action = create_action(
            self,
            _("Show first object only"),
            icon=get_icon("show_first.svg"),
            tip=_("Show only the first selected object (signal or image)"),
            toggled=self.toggle_show_first_only,
        )
        self.showlabel_action = create_action(
            self,
            _("Show graphical object titles"),
            icon=get_icon("show_titles.svg"),
            tip=_("Show or hide ROI and other graphical object titles or subtitles"),
            toggled=self.toggle_show_titles,
        )

    def __add_signal_panel(self) -> None:
        """Setup signal toolbar, widgets and panel"""
        self.signalpanel_toolbar = self.__add_toolbar(
            _("Signal Panel Toolbar"), "left", "signalpanel_toolbar"
        )
        dpw = DockablePlotWidget(self, PlotType.CURVE)
        self.signalpanel = signal.SignalPanel(self, dpw, self.signalpanel_toolbar)
        self.signalpanel.SIG_STATUS_MESSAGE.connect(self.statusBar().showMessage)
        plot = dpw.get_plot()
        plot.add_item(make.legend("TR"))
        plot.SIG_ITEM_PARAMETERS_CHANGED.connect(
            self.signalpanel.plot_item_parameters_changed
        )
        plot.SIG_ITEM_MOVED.connect(self.signalpanel.plot_item_moved)
        return dpw

    def __add_image_panel(self) -> None:
        """Setup image toolbar, widgets and panel"""
        self.imagepanel_toolbar = self.__add_toolbar(
            _("Image Panel Toolbar"), "left", "imagepanel_toolbar"
        )
        dpw = DockablePlotWidget(self, PlotType.IMAGE)
        self.imagepanel = image.ImagePanel(self, dpw, self.imagepanel_toolbar)
        # -----------------------------------------------------------------------------
        # # Before eventually disabling the "peritem" mode by default, wait for the
        # # plotpy bug to be fixed (peritem mode is not compatible with multiple image
        # # items):
        # for cspanel in (
        #     self.imagepanel.plotwidget.get_xcs_panel(),
        #     self.imagepanel.plotwidget.get_ycs_panel(),
        # ):
        #     cspanel.peritem_ac.setChecked(False)
        # -----------------------------------------------------------------------------
        self.imagepanel.SIG_STATUS_MESSAGE.connect(self.statusBar().showMessage)
        plot = dpw.get_plot()
        plot.SIG_ITEM_PARAMETERS_CHANGED.connect(
            self.imagepanel.plot_item_parameters_changed
        )
        plot.SIG_ITEM_MOVED.connect(self.imagepanel.plot_item_moved)
        plot.SIG_LUT_CHANGED.connect(self.imagepanel.plot_lut_changed)
        return dpw

    def __update_tab_menu(self) -> None:
        """Update tab menu"""
        current_panel: BaseDataPanel = self.tabwidget.currentWidget()
        add_actions(self.tabmenu, current_panel.get_context_menu().actions())

    def __add_signal_image_panels(self) -> None:
        """Add signal and image panels"""
        self.tabwidget = QW.QTabWidget()
        self.tabmenu = add_corner_menu(self.tabwidget)
        configure_menu_about_to_show(self.tabmenu, self.__update_tab_menu)
        self.signalview = self.__add_signal_panel()
        self.imageview = self.__add_image_panel()
        sdock = self.__add_dockwidget(self.signalview, title=_("Signal View"))
        idock = self.__add_dockwidget(self.imageview, title=_("Image View"))
        self.tabifyDockWidget(sdock, idock)
        self.docks = {self.signalpanel: sdock, self.imagepanel: idock}
        self.tabwidget.currentChanged.connect(self.__tab_index_changed)
        self.signalpanel.SIG_OBJECT_ADDED.connect(
            lambda: self.set_current_panel("signal")
        )
        self.imagepanel.SIG_OBJECT_ADDED.connect(
            lambda: self.set_current_panel("image")
        )
        for panel in (self.signalpanel, self.imagepanel):
            panel.setup_panel()

    def __setup_central_widget(self) -> None:
        """Setup central widget (main panel)"""
        self.tabwidget.setMaximumWidth(600)
        s_idx = self.tabwidget.addTab(
            self.signalpanel, get_icon("signal.svg"), _("Signal Panel")
        )
        i_idx = self.tabwidget.addTab(
            self.imagepanel, get_icon("image.svg"), _("Image Panel")
        )
        self.tabwidget.setTabToolTip(
            s_idx, _("1D Signals: Manage and process one-dimensional data")
        )
        self.tabwidget.setTabToolTip(
            i_idx, _("2D Images: Manage and process two-dimensional data")
        )

        # Apply enhanced tab bar styling
        tab_bar = self.tabwidget.tabBar()
        font = tab_bar.font()
        font.setPointSize(10)
        tab_bar.setFont(font)
        # Use QTimer to ensure tab bar is properly sized first
        QC.QTimer.singleShot(0, self.__update_tab_icon_size)

        self.setCentralWidget(self.tabwidget)

    def __update_tab_icon_size(self) -> None:
        """Update tab icon size based on tab bar height"""
        tab_bar = self.tabwidget.tabBar()
        if tab_bar.height() > 0:
            # Use approximately 80% of tab height for icon size
            icon_size = int(tab_bar.height() * 0.8)
            self.tabwidget.setIconSize(QC.QSize(icon_size, icon_size))

    @staticmethod
    def __get_local_doc_path() -> str | None:
        """Return local documentation path, if it exists"""
        locale = QC.QLocale.system().name()
        for suffix in ("_" + locale[:2], "_en"):
            path = osp.join(DATAPATH, "doc", f"{APP_NAME}{suffix}.pdf")
            if osp.isfile(path):
                return path
        return None

    def __add_menus(self) -> None:
        """Adding menus"""
        self.file_menu = self.menuBar().addMenu(_("&File"))
        configure_menu_about_to_show(self.file_menu, self.__update_file_menu)
        self.create_menu = self.menuBar().addMenu(_("&Create"))
        self.edit_menu = self.menuBar().addMenu(_("&Edit"))
        self.roi_menu = self.menuBar().addMenu(_("ROI"))
        self.operation_menu = self.menuBar().addMenu(_("Operations"))
        self.processing_menu = self.menuBar().addMenu(_("Processing"))
        self.analysis_menu = self.menuBar().addMenu(_("Analysis"))
        self.plugins_menu = self.menuBar().addMenu(_("Plugins"))
        self.view_menu = self.menuBar().addMenu(_("&View"))
        configure_menu_about_to_show(self.view_menu, self.__update_view_menu)
        self.help_menu = self.menuBar().addMenu("?")
        for menu in (
            self.create_menu,
            self.edit_menu,
            self.roi_menu,
            self.operation_menu,
            self.processing_menu,
            self.analysis_menu,
            self.plugins_menu,
        ):
            configure_menu_about_to_show(menu, self.__update_generic_menu)
        help_menu_actions = [
            create_action(
                self,
                _("Online documentation"),
                icon=get_icon("libre-gui-help.svg"),
                triggered=lambda: webbrowser.open(__docurl__),
            ),
        ]
        localdocpath = self.__get_local_doc_path()
        if localdocpath is not None:
            help_menu_actions += [
                create_action(
                    self,
                    _("PDF documentation"),
                    icon=get_icon("help_pdf.svg"),
                    triggered=lambda: webbrowser.open(localdocpath),
                ),
            ]
        help_menu_actions += [
            create_action(
                self,
                _("Tour") + "...",
                icon=get_icon("tour.svg"),
                triggered=self.show_tour,
            ),
            create_action(
                self,
                _("Demo") + "...",
                icon=get_icon("play_demo.svg"),
                triggered=self.play_demo,
            ),
            None,
        ]
        if TEST_SEGFAULT_ERROR:
            help_menu_actions += [
                create_action(
                    self,
                    _("Test segfault/Python error"),
                    triggered=self.test_segfault_error,
                )
            ]
        help_menu_actions += [
            create_action(
                self,
                _("Log files") + "...",
                icon=get_icon("logs.svg"),
                triggered=self.__show_logviewer,
            ),
            create_action(
                self,
                _("Installation and configuration") + "...",
                icon=get_icon("libre-toolbox.svg"),
                triggered=lambda: instconfviewer.exec_datalab_installconfig_dialog(
                    self
                ),
            ),
            None,
            create_action(
                self,
                _("Project home page"),
                icon=get_icon("libre-gui-globe.svg"),
                triggered=lambda: webbrowser.open(__homeurl__),
            ),
            create_action(
                self,
                _("Bug report or feature request"),
                icon=get_icon("libre-gui-globe.svg"),
                triggered=lambda: webbrowser.open(__supporturl__),
            ),
            create_action(
                self,
                _("About..."),
                icon=get_icon("libre-gui-about.svg"),
                triggered=self.__about,
            ),
        ]
        add_actions(self.help_menu, help_menu_actions)

    def __update_console_show_mode(self) -> None:
        """Update console show mode from configuration option

        Console show mode is whether the console is shown or not when an error occurs.
        """
        if self.console is not None:
            state = Conf.console.show_console_on_error.get()
            cdock = self.docks[self.console]
            if not state and cdock.isVisible():
                cdock.hide()
            if state:
                self.console.exception_occurred.connect(self.console.show_console)
            else:
                self.console.exception_occurred.disconnect(self.console.show_console)

    def __setup_console(self) -> None:
        """Add an internal console"""
        ns = {
            "dl": self,
            "np": np,
            "sps": sps,
            "spi": spi,
            "os": os,
            "sys": sys,
            "osp": osp,
            "time": time,
        }
        msg = _(
            "Welcome to DataLab console!\n"
            "---------------------------\n"
            "You can access the main window with the 'dl' variable.\n"
            "Example:\n"
            "  o = dl.get_object()  # returns currently selected object\n"
            "  o = dl[1]  # returns object number 1\n"
            "  o = dl['My image']  # returns object which title is 'My image'\n"
            "  o.data  # returns object data\n"
            "Modules imported at startup: "
            "os, sys, os.path as osp, time, "
            "numpy as np, scipy.signal as sps, scipy.ndimage as spi"
        )
        self.console = DockableConsole(self, namespace=ns, message=msg, debug=DEBUG)
        self.console.setMaximumBlockCount(Conf.console.max_line_count.get(5000))
        self.console.go_to_error.connect(go_to_error)
        cdock = self.__add_dockwidget(self.console, _("Console"))
        self.docks[self.console] = cdock
        cdock.hide()
        self.console.interpreter.widget_proxy.sig_new_prompt.connect(
            lambda txt: self.repopulate_panel_trees()
        )
        self.__update_console_show_mode()
        self.console.exception_occurred.connect(self.consolestatus.exception_occurred)
        cdock.visibilityChanged.connect(self.consolestatus.console_visibility_changed)
        self.consolestatus.SIG_SHOW_CONSOLE.connect(self.console.show_console)

    def __add_macro_panel(self) -> None:
        """Add macro panel"""
        self.macropanel = macro.MacroPanel(self)
        mdock = self.__add_dockwidget(self.macropanel, _("Macro Panel"))
        self.docks[self.macropanel] = mdock
        self.tabifyDockWidget(self.docks[self.imagepanel], mdock)
        self.docks[self.signalpanel].raise_()

    def __configure_panels(self) -> None:
        """Configure panels"""
        # Connectings signals
        for panel in self.panels:
            panel.SIG_OBJECT_ADDED.connect(self.set_modified)
            panel.SIG_OBJECT_REMOVED.connect(self.set_modified)
        self.macropanel.SIG_OBJECT_MODIFIED.connect(self.set_modified)
        # Initializing common panel actions
        self.autorefresh_action.setChecked(Conf.view.auto_refresh.get(True))
        self.showfirstonly_action.setChecked(Conf.view.show_first_only.get(False))
        self.showlabel_action.setChecked(Conf.view.show_label.get(False))
        # Restoring current tab from last session
        tab_idx = Conf.main.current_tab.get(None)
        if tab_idx is not None:
            self.tabwidget.setCurrentIndex(tab_idx)
        # Set focus on current panel, so that keyboard shortcuts work (Fixes #10)
        self.tabwidget.currentWidget().setFocus()

    def set_process_isolation_enabled(self, state: bool) -> None:
        """Enable/disable process isolation

        Args:
            state: True to enable process isolation
        """
        for processor in (self.imagepanel.processor, self.signalpanel.processor):
            processor.set_process_isolation_enabled(state)

    # ------Remote control
    @remote_controlled
    def get_current_panel(self) -> str:
        """Return current panel name

        Returns:
            Panel name (valid values: "signal", "image", "macro")
        """
        panel = self.tabwidget.currentWidget()
        dock = self.docks[panel]
        if panel is self.signalpanel and dock.isVisible():
            return "signal"
        if panel is self.imagepanel and dock.isVisible():
            return "image"
        return "macro"

    @remote_controlled
    def set_current_panel(
        self, panel: Literal["signal", "image", "macro"] | BaseDataPanel
    ) -> None:
        """Switch to panel.

        Args:
            panel: panel name or panel instance

        Raises:
            ValueError: unknown panel
        """
        if not isinstance(panel, str):
            if panel not in self.panels:
                raise ValueError(f"Unknown panel {panel}")
            panel = (
                "signal"
                if panel is self.signalpanel
                else "image"
                if panel is self.imagepanel
                else "macro"
            )
        if self.get_current_panel() == panel:
            if panel in ("signal", "image"):
                # Force tab index changed event to be sure that the dock associated
                # to the current panel is raised
                self.__tab_index_changed(self.tabwidget.currentIndex())
            return
        if panel == "signal":
            self.tabwidget.setCurrentWidget(self.signalpanel)
        elif panel == "image":
            self.tabwidget.setCurrentWidget(self.imagepanel)
        elif panel == "macro":
            self.docks[self.macropanel].raise_()
        else:
            raise ValueError(f"Unknown panel {panel}")

    @remote_controlled
    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call computation feature ``name``

        .. note::

            This calls either the processor's ``compute_<name>`` method (if it exists),
            or the processor's ``<name>`` computation feature (if it is registered,
            using the ``run_feature`` method).
            It looks for the function in all panels, starting with the current one.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """
        panels = [self.tabwidget.currentWidget()]
        panels.extend(self.panels)
        for panel in panels:
            if isinstance(panel, base.BaseDataPanel):
                name = name.removeprefix("compute_")
                panel: base.BaseDataPanel
                # Some computation features are wrapped in a method with a
                # "compute_" prefix, so we check for this first:
                func = getattr(panel.processor, f"compute_{name}", None)
                if func is not None:
                    if param is None:
                        func()
                    else:
                        func(param)
                    return
                # If the function is not wrapped, we check if it is a
                # registered feature:
                try:
                    feature = panel.processor.get_feature(name)
                    panel.processor.run_feature(feature, param)
                    return
                except ValueError:
                    continue
        raise ValueError(f"Unknown computation function {name}")

    # ------GUI refresh
    def has_objects(self) -> bool:
        """Return True if sig/ima panels have any object"""
        return sum(len(panel) for panel in self.panels) > 0

    def set_modified(self, state: bool = True) -> None:
        """Set mainwindow modified state"""
        state = state and self.has_objects()
        self.__is_modified = state
        title = APP_NAME + ("*" if state else "")
        if not datalab.__version__.replace(".", "").isdigit():
            title += f" [{datalab.__version__}]"
        self.setWindowTitle(title)

    def is_modified(self) -> bool:
        """Return True if mainwindow is modified"""
        return self.__is_modified

    def __add_dockwidget(self, child, title: str) -> QW.QDockWidget:
        """Add QDockWidget and toggleViewAction"""
        dockwidget, location = child.create_dockwidget(title)
        dockwidget.setObjectName(title)
        self.addDockWidget(location, dockwidget)
        return dockwidget

    def repopulate_panel_trees(self) -> None:
        """Repopulate all panel trees"""
        for panel in self.panels:
            if isinstance(panel, base.BaseDataPanel):
                panel.objview.populate_tree()

    def __update_actions(self, update_other_data_panel: bool = False) -> None:
        """Update selection dependent actions

        Args:
            update_other_data_panel: True to update other data panel actions
             (i.e. if the current panel is the signal panel, also update the image
             panel actions, and vice-versa)
        """
        is_signal = self.tabwidget.currentWidget() is self.signalpanel
        panel = self.signalpanel if is_signal else self.imagepanel
        other_panel = self.imagepanel if is_signal else self.signalpanel
        if update_other_data_panel:
            other_panel.selection_changed()
        panel.selection_changed()
        self.signalpanel_toolbar.setVisible(is_signal)
        self.imagepanel_toolbar.setVisible(not is_signal)
        if self.plugins_menu is not None:
            plugin_actions = panel.get_category_actions(ActionCategory.PLUGINS)
            self.plugins_menu.setEnabled(len(plugin_actions) > 0)

    def __tab_index_changed(self, index: int) -> None:
        """Switch from signal to image mode, or vice-versa"""
        dock = self.docks[self.tabwidget.widget(index)]
        dock.raise_()
        self.__update_actions()

    def __update_generic_menu(self, menu: QW.QMenu | None = None) -> None:
        """Update menu before showing up -- Generic method"""
        if menu is None:
            menu = self.sender()
        menu.clear()
        panel = self.tabwidget.currentWidget()
        category = {
            self.file_menu: ActionCategory.FILE,
            self.create_menu: ActionCategory.CREATE,
            self.edit_menu: ActionCategory.EDIT,
            self.roi_menu: ActionCategory.ROI,
            self.view_menu: ActionCategory.VIEW,
            self.operation_menu: ActionCategory.OPERATION,
            self.processing_menu: ActionCategory.PROCESSING,
            self.analysis_menu: ActionCategory.ANALYSIS,
            self.plugins_menu: ActionCategory.PLUGINS,
        }[menu]
        actions = panel.get_category_actions(category)
        add_actions(menu, actions)

    def __update_file_menu(self) -> None:
        """Update file menu before showing up"""
        self.saveh5_action.setEnabled(self.has_objects())
        self.__update_generic_menu(self.file_menu)
        add_actions(
            self.file_menu,
            [
                None,
                self.openh5_action,
                self.saveh5_action,
                self.browseh5_action,
                None,
                self.settings_action,
            ],
        )
        # Add Web API submenu
        if self.webapi_actions is not None:
            self.file_menu.addSeparator()
            self.webapi_actions.create_menu(self.file_menu)
        if self.quit_action is not None:
            add_actions(self.file_menu, [self.quit_action])

    def __update_view_menu(self) -> None:
        """Update view menu before showing up"""
        self.__update_generic_menu(self.view_menu)
        add_actions(self.view_menu, [None] + self.createPopupMenu().actions())

    @remote_controlled
    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show annotations option

        Args:
            state: state
        """
        Conf.view.show_label.set(state)
        for datapanel in (self.signalpanel, self.imagepanel):
            for obj in datapanel.objmodel:
                obj.set_metadata_option("showlabel", state)
            datapanel.refresh_plot("selected", True, False)

    def handle_autorefresh_action(self, state: bool) -> None:
        """Handle auto-refresh action from UI (with confirmation dialog)

        Args:
            state: desired state
        """
        # If disabling auto-refresh, show confirmation dialog
        if not state:
            txtlist = [
                "<b>" + _("Disable auto-refresh?") + "</b>",
                "",
                _(
                    "When auto-refresh is disabled, the plot view will not "
                    "automatically update when objects are modified, added or removed."
                ),
                "",
                _(
                    "You will need to manually click the refresh button to update "
                    "the view."
                ),
                "",
                _("Are you sure you want to disable auto-refresh?"),
            ]

            answer = QW.QMessageBox.question(
                self,
                APP_NAME,
                "<br>".join(txtlist),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
                QW.QMessageBox.No,
            )

            if answer == QW.QMessageBox.No:
                # User cancelled, restore the action's checked state
                self.autorefresh_action.blockSignals(True)
                self.autorefresh_action.setChecked(True)
                self.autorefresh_action.blockSignals(False)
                return

        # Apply the change
        self.toggle_auto_refresh(state)

    @remote_controlled
    def toggle_auto_refresh(self, state: bool) -> None:
        """Toggle auto refresh option

        Args:
            state: state
        """
        Conf.view.auto_refresh.set(state)
        for datapanel in (self.signalpanel, self.imagepanel):
            datapanel.plothandler.set_auto_refresh(state)

    @remote_controlled
    def toggle_show_first_only(self, state: bool) -> None:
        """Toggle show first only option

        Args:
            state: state
        """
        Conf.view.show_first_only.set(state)
        for datapanel in (self.signalpanel, self.imagepanel):
            datapanel.plothandler.set_show_first_only(state)

    # ------Common features
    @remote_controlled
    def reset_all(self) -> None:
        """Reset all application data"""
        for panel in self.panels:
            if panel is not None:
                panel.remove_all_objects()

    @remote_controlled
    def remove_object(self, force: bool = False) -> None:
        """Remove current object from current panel.

        Args:
            force: if True, remove object without confirmation. Defaults to False.
        """
        panel = self.__get_current_basedatapanel()
        panel.remove_object(force)

    @staticmethod
    def __check_h5file(filename: str, operation: str) -> str:
        """Check HDF5 filename"""
        filename = osp.abspath(osp.normpath(filename))
        bname = osp.basename(filename)
        if operation == "load" and not osp.isfile(filename):
            raise IOError(f'File not found "{bname}"')
        Conf.main.base_dir.set(filename)
        return filename

    @remote_controlled
    def save_to_h5_file(self, filename=None) -> None:
        """Save to a DataLab HDF5 file

        Args:
            filename: HDF5 filename. If None, a file dialog is opened.

        Raises:
            IOError: if filename is invalid or file cannot be saved.
        """
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with qth.save_restore_stds():
                filename, _fl = getsavefilename(
                    self,
                    _("Save"),
                    basedir,
                    "HDF5 (*.h5 *.hdf5 *.hdf *.he5);;All files (*)",
                )
            if not filename:
                return
        with qth.qt_try_loadsave_file(self, filename, "save"):
            self.save_h5_workspace(filename)

    @remote_controlled
    def open_h5_files(
        self,
        h5files: list[str] | None = None,
        import_all: bool | None = None,
        reset_all: bool | None = None,
    ) -> None:
        """Open a DataLab HDF5 file or import from any other HDF5 file.

        Args:
            h5files: HDF5 filenames (optionally with dataset name, separated by ":")
            import_all: Import all datasets from HDF5 files
            reset_all: Reset all application data before importing
        """
        if not self.confirm_memory_state():
            return
        if reset_all is None:
            # When workspace is empty, always preserve UUIDs (reset_all=True)
            # since there's no risk of conflicts
            if not self.has_objects():
                reset_all = True
            else:
                reset_all = Conf.io.h5_clear_workspace.get()
                if Conf.io.h5_clear_workspace_ask.get():
                    # Build message with optional note for native workspace import
                    msg = _(
                        "Do you want to clear current workspace "
                        "(signals and images) before importing data from "
                        "HDF5 files?"
                    )
                    # Only show the UUID conflict note when importing native DataLab
                    # workspace files (import_all=True), not when using HDF5 browser
                    if import_all:
                        msg += "<br><br>" + _(
                            "<u>Note:</u> If you choose <i>No</i>, when importing "
                            "DataLab workspace files, objects with conflicting "
                            "identifiers will have their processing history lost "
                            "(features like 'Show source' and 'Recompute' will not "
                            "work for those objects). Non-conflicting objects will "
                            "preserve their processing history."
                        )
                    msg += "<br><br>" + _(
                        "Choosing to ignore this message will prevent it "
                        "from being displayed again, and will use the "
                        "current setting (%s)."
                    ) % (_("Yes") if reset_all else _("No"))
                    answer = QW.QMessageBox.question(
                        self,
                        _("Warning"),
                        msg,
                        QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Ignore,
                    )
                    if answer == QW.QMessageBox.Yes:
                        reset_all = True
                    elif answer == QW.QMessageBox.No:
                        reset_all = False
                    elif answer == QW.QMessageBox.Ignore:
                        Conf.io.h5_clear_workspace_ask.set(False)
        if h5files is None:
            basedir = Conf.main.base_dir.get()
            with qth.save_restore_stds():
                h5files, _fl = getopenfilenames(
                    self,
                    _("Open"),
                    basedir,
                    _("HDF5 files (*.h5 *.hdf5 *.hdf *.he5);;All files (*)"),
                )
        if not h5files:
            return
        filenames, dsetnames = [], []
        for fname_with_dset in h5files:
            if "," in fname_with_dset:
                filename, dsetname = fname_with_dset.split(",")
                dsetnames.append(dsetname)
            else:
                filename = fname_with_dset
                dsetnames.append(None)
            filenames.append(filename)
        if import_all is None and all(dsetname is None for dsetname in dsetnames):
            self.browse_h5_files(filenames, reset_all)
            return
        for filename, dsetname in zip(filenames, dsetnames):
            if import_all is None and dsetname is None:
                self.import_h5_file(filename, reset_all)
            else:
                with qth.qt_try_loadsave_file(self, filename, "load"):
                    filename = self.__check_h5file(filename, "load")
                    if dsetname is None:
                        self.h5inputoutput.open_file(filename, import_all, reset_all)
                    else:
                        self.h5inputoutput.import_dataset_from_file(filename, dsetname)
            reset_all = False

    def browse_h5_files(self, filenames: list[str], reset_all: bool) -> None:
        """Browse HDF5 files

        Args:
            filenames: HDF5 filenames
            reset_all: Reset all application data before importing
        """
        for filename in filenames:
            self.__check_h5file(filename, "load")
        self.h5inputoutput.import_files(filenames, False, reset_all)

    @remote_controlled
    def load_h5_workspace(self, h5files: list[str], reset_all: bool = False) -> None:
        """Load native DataLab HDF5 workspace files without any GUI elements.

        This method can be safely called from the internal console as it does not
        create any Qt widgets, dialogs, or progress bars. It is designed for
        programmatic use when loading DataLab workspace files.

        .. warning::

            This method only supports native DataLab HDF5 files. For importing
            arbitrary HDF5 files (non-native), use the GUI menu or macros with
            :class:`datalab.control.proxy.RemoteProxy`.

        Args:
            h5files: List of native DataLab HDF5 filenames
            reset_all: Reset all application data before importing. Defaults to False.

        Raises:
            ValueError: If a file is not a valid native DataLab HDF5 file
        """
        for idx, filename in enumerate(h5files):
            filename = self.__check_h5file(filename, "load")
            success = self.h5inputoutput.open_file_headless(
                filename, reset_all=(reset_all and idx == 0)
            )
            if not success:
                raise ValueError(
                    f"File '{filename}' is not a native DataLab HDF5 file. "
                    f"Use the GUI menu or a macro with RemoteProxy to import "
                    f"arbitrary HDF5 files."
                )
        # Refresh panel trees after loading
        self.repopulate_panel_trees()

    @remote_controlled
    def save_h5_workspace(self, filename: str) -> None:
        """Save current workspace to a native DataLab HDF5 file without GUI elements.

        This method can be safely called from the internal console as it does not
        create any Qt widgets, dialogs, or progress bars. It is designed for
        programmatic use when saving DataLab workspace files.

        Args:
            filename: HDF5 filename to save to

        Raises:
            IOError: If file cannot be saved
        """
        filename = self.__check_h5file(filename, "save")
        self.h5inputoutput.save_file(filename)
        self.set_modified(False)

    @remote_controlled
    def import_h5_file(self, filename: str, reset_all: bool | None = None) -> None:
        """Import HDF5 file into DataLab

        Args:
            filename: HDF5 filename (optionally with dataset name,
            separated by ":")
            reset_all: Delete all DataLab signals/images before importing data
        """
        with qth.qt_try_loadsave_file(self, filename, "load"):
            filename = self.__check_h5file(filename, "load")
            self.h5inputoutput.import_files([filename], False, reset_all)

    # This method is intentionally *not* remote controlled
    # (see TODO regarding RemoteClient.add_object method)
    #  @remote_controlled
    def add_object(
        self, obj: SignalObj | ImageObj, group_id: str = "", set_current=True
    ) -> None:
        """Add object - signal or image

        Args:
            obj: object to add (signal or image)
            group_id: group ID (optional)
            set_current: True to set the object as current object
        """
        if self.confirm_memory_state():
            if isinstance(obj, SignalObj):
                self.signalpanel.add_object(obj, group_id, set_current)
            elif isinstance(obj, ImageObj):
                self.imagepanel.add_object(obj, group_id, set_current)
            else:
                raise TypeError(f"Unsupported object type {type(obj)}")

    @remote_controlled
    def load_from_files(self, filenames: list[str]) -> None:
        """Open objects from files in current panel (signals/images)

        Args:
            filenames: list of filenames
        """
        panel = self.__get_current_basedatapanel()
        panel.load_from_files(filenames)

    @remote_controlled
    def load_from_directory(self, path: str) -> None:
        """Open objects from directory in current panel (signals/images).

        Args:
            path: directory path
        """
        panel = self.__get_current_basedatapanel()
        panel.load_from_directory(path)

    # ------Other methods related to AbstractDLControl interface
    def get_version(self) -> str:
        """Return DataLab public version.

        Returns:
            DataLab version
        """
        return datalab.__version__

    def close_application(self) -> None:  # Implementing AbstractDLControl interface
        """Close DataLab application"""
        self.close()

    def raise_window(self) -> None:  # Implementing AbstractDLControl interface
        """Raise DataLab window"""
        bring_to_front(self)

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title: Signal title
            xdata: X data
            ydata: Y data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            group_id: group id in which to add the signal. Defaults to ""
            set_current: if True, set the added signal as current

        Returns:
            True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """
        obj = create_signal(
            title,
            xdata,
            ydata,
            units=(xunit, yunit),
            labels=(xlabel, ylabel),
        )
        self.add_object(obj, group_id, set_current)
        return True

    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        zunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title: Image title
            data: Image data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            zunit: Z unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            zlabel: Z label. Defaults to ""
            group_id: group id in which to add the image. Defaults to ""
            set_current: if True, set the added image as current

        Returns:
            True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        obj = create_image(
            title,
            data,
            units=(xunit, yunit, zunit),
            labels=(xlabel, ylabel, zlabel),
        )
        self.add_object(obj, group_id, set_current)
        return True

    # ------?
    def __about(self) -> None:  # pragma: no cover
        """About dialog box"""
        self.check_stable_release()
        if self.remote_server.port is None:
            xrpcstate = '<font color="red">' + _("not started") + "</font>"
        else:
            xrpcstate = _("started (port %s)") % self.remote_server.port
            xrpcstate = f"<font color='green'>{xrpcstate}</font>"
        if Conf.main.process_isolation_enabled.get():
            pistate = "<font color='green'>" + _("enabled") + "</font>"
        else:
            pistate = "<font color='red'>" + _("disabled") + "</font>"
        adv_conf = "<br>".join(
            [
                "<i>" + _("Advanced configuration:") + "</i>",
                "• " + _("XML-RPC server:") + " " + xrpcstate,
                "• " + _("Process isolation:") + " " + pistate,
            ]
        )
        created_by = _("Created by")
        dev_by = _("Developed and maintained by %s open-source project team") % APP_NAME
        cprght = "2023 DataLab Platform Developers"
        QW.QMessageBox.about(
            self,
            _("About") + " " + APP_NAME,
            f"""<b>{APP_NAME}</b> v{datalab.__version__}<br>{APP_DESC}
              <p>{created_by} Pierre Raybaut<br>{dev_by}<br>Copyright &copy; {cprght}
              <p>{adv_conf}""",
        )

    def __update_color_mode(self, startup: bool = False) -> None:
        """Update color mode

        Args:
            startup: True if method is called during application startup (in that case,
             color theme is applied only if mode != "auto")
        """
        mode = Conf.main.color_mode.get()
        if startup and mode == "auto":
            guidata_qth.win32_fix_title_bar_background(self)
            return

        # Prevent Qt from refreshing the window when changing the color mode:
        self.setUpdatesEnabled(False)

        plotpy_config.set_plotpy_color_mode(mode)

        if self.console is not None:
            self.console.update_color_mode()
        if self.macropanel is not None:
            self.macropanel.update_color_mode()
        if self.docks is not None:
            for dock in self.docks.values():
                widget = dock.widget()
                if isinstance(widget, DockablePlotWidget):
                    widget.update_color_mode()

        # Allow Qt to refresh the window:
        self.setUpdatesEnabled(True)

    def __edit_settings(self) -> None:
        """Edit settings"""
        changed_options = edit_settings(self)
        sigima_options.fft_shift_enabled.set(Conf.proc.fft_shift_enabled.get())
        sigima_options.auto_normalize_kernel.set(Conf.proc.auto_normalize_kernel.get())
        refresh_signal_panel = refresh_image_panel = False

        # Handling changes to shape/marker parameters
        s_view_result_param = (
            "sig_shape_param" in changed_options
            or "sig_marker_param" in changed_options
        ) and have_geometry_results(self.signalpanel.objview.get_sel_objects(True))
        i_view_result_param = (
            "ima_shape_param" in changed_options
            or "ima_marker_param" in changed_options
        ) and have_geometry_results(self.imagepanel.objview.get_sel_objects(True))
        if (s_view_result_param or i_view_result_param) and (
            QW.QMessageBox.question(
                self,
                _("Apply settings to existing results?"),
                _(
                    "Visualization settings for annotated shapes and "
                    "markers have been modified.\n\n"
                    "Do you want to apply these settings to existing results "
                    "in the workspace?"
                ),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
                QW.QMessageBox.No,
            )
            == QW.QMessageBox.Yes
        ):
            if s_view_result_param:
                self.signalpanel.plothandler.refresh_all_shape_items()
            if i_view_result_param:
                self.imagepanel.plothandler.refresh_all_shape_items()

        for option in changed_options:
            if option in (
                "max_shapes_to_draw",
                "max_cells_in_label",
                "max_cols_in_label",
            ):
                refresh_signal_panel = refresh_image_panel = True
            if option == "show_result_label":
                for panel in (self.signalpanel, self.imagepanel):
                    panel.acthandler.show_label_action.setChecked(
                        Conf.view.show_result_label.get()
                    )
            if option == "color_mode":
                self.__update_color_mode()
            if option == "show_console_on_error":
                self.__update_console_show_mode()
            if option == "plot_toolbar_position":
                for dock in self.docks.values():
                    widget = dock.widget()
                    if isinstance(widget, DockablePlotWidget):
                        widget.update_toolbar_position()
            if option.startswith(("sig_autodownsampling", "sig_linewidth")):
                refresh_signal_panel = True
            if option == "sig_autoscale_margin_percent":
                # Update signal plot widget autoscale margin
                sig_margin = Conf.view.sig_autoscale_margin_percent.get()
                for dock in self.docks.values():
                    widget: DockablePlotWidget | QW.QWidget = dock.widget()
                    if isinstance(widget, DockablePlotWidget):
                        plot = widget.get_plot()
                        if (
                            hasattr(plot, "options")
                            and plot.options.type == PlotType.CURVE
                        ):
                            plot.set_autoscale_margin_percent(sig_margin)
            if option == "ima_autoscale_margin_percent":
                # Update image plot widget autoscale margin
                ima_margin = Conf.view.ima_autoscale_margin_percent.get()
                for dock in self.docks.values():
                    widget: DockablePlotWidget | QW.QWidget = dock.widget()
                    if isinstance(widget, DockablePlotWidget):
                        plot = widget.get_plot()
                        if (
                            hasattr(plot, "options")
                            and plot.options.type == PlotType.IMAGE
                        ):
                            plot.set_autoscale_margin_percent(ima_margin)
            if option == "ima_defaults" and len(self.imagepanel) > 0:
                answer = QW.QMessageBox.question(
                    self,
                    _("Visualization settings"),
                    _(
                        "Default visualization settings have changed.<br><br>"
                        "Do you want to update all active %s objects?"
                    )
                    % _("image"),
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                )
                if answer == QW.QMessageBox.Yes:
                    self.imagepanel.update_metadata_view_settings()
            if option == "ima_aspect_ratio_1_1":
                refresh_image_panel = True
        if refresh_signal_panel:
            self.signalpanel.manual_refresh()
        if refresh_image_panel:
            self.imagepanel.manual_refresh()

    def __show_logviewer(self) -> None:
        """Show error logs"""
        logviewer.exec_datalab_logviewer_dialog(self)

    def play_demo(self) -> None:
        """Play demo"""
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from datalab.tests.scenarios import demo

        demo.play_demo(self)

    def show_tour(self) -> None:
        """Show tour"""
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from datalab.gui import tour

        tour.start(self)

    @staticmethod
    def test_segfault_error() -> None:
        """Generate errors (both fault and traceback)"""
        import ctypes  # pylint: disable=import-outside-toplevel

        ctypes.string_at(0)
        raise RuntimeError("!!! Testing RuntimeError !!!")

    def show(self) -> None:
        """Reimplement QMainWindow method"""
        super().show()
        if self.__old_size is not None:
            self.resize(self.__old_size)

    # ------Close window
    def close_properly(self) -> bool:
        """Close properly

        Returns:
            True if closed properly, False otherwise
        """
        if not env.execenv.unattended and self.is_modified():
            answer = QW.QMessageBox.warning(
                self,
                _("Quit"),
                _(
                    "Do you want to save all signals and images "
                    "to an HDF5 file before quitting DataLab?"
                ),
                QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Cancel,
            )
            if answer == QW.QMessageBox.Yes:
                self.save_to_h5_file()
                if self.is_modified():
                    return False
            elif answer == QW.QMessageBox.Cancel:
                return False
        self.hide()  # Avoid showing individual widgets closing one after the other
        for panel in self.panels:
            if panel is not None:
                panel.close()
        if self.console is not None:
            try:
                self.console.close()
            except RuntimeError:
                # TODO: [P3] Investigate further why the following error occurs when
                # restarting the mainwindow (this is *not* a production case):
                # "RuntimeError: wrapped C/C++ object of type DockableConsole
                #  has been deleted".
                # Another solution to avoid this error would be to really restart
                # the application (run each unit test in a separate process), but
                # it would represent too much effort for an error occuring in test
                # configurations only.
                pass
        if self.webapi_actions is not None:
            self.webapi_actions.cleanup()
        self.reset_all()
        self.__save_pos_size_and_state()
        self.__unregister_plugins()

        # Saving current tab for next session
        Conf.main.current_tab.set(self.tabwidget.currentIndex())

        execenv.log(self, "closed properly")
        return True

    def closeEvent(self, event: QG.QCloseEvent) -> None:
        """Reimplement QMainWindow method"""
        if self.hide_on_close:
            self.__old_size = self.size()
            self.hide()
        else:
            if self.close_properly():
                self.SIG_CLOSING.emit()
                event.accept()
            else:
                event.ignore()
