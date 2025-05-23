# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Main window
===========

The :mod:`cdl.core.gui.main` module provides the main window of the
DataLab (CDL) project.

.. autoclass:: CDLMainWindow
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

import cdl
from cdl import __docurl__, __homeurl__, __supporturl__, env
from cdl.config import (
    APP_DESC,
    APP_NAME,
    DATAPATH,
    DEBUG,
    IS_FROZEN,
    TEST_SEGFAULT_ERROR,
    Conf,
    _,
)
from cdl.core.baseproxy import AbstractCDLControl
from cdl.core.gui.actionhandler import ActionCategory
from cdl.core.gui.docks import DockablePlotWidget
from cdl.core.gui.h5io import H5InputOutput
from cdl.core.gui.panel import base, image, macro, signal
from cdl.core.gui.settings import edit_settings
from cdl.core.model.image import ImageObj, create_image
from cdl.core.model.signal import SignalObj, create_signal
from cdl.core.remote import RemoteServer
from cdl.env import execenv
from cdl.plugins import PluginRegistry, discover_plugins
from cdl.utils import dephash
from cdl.utils import qthelpers as qth
from cdl.utils.misc import go_to_error
from cdl.utils.qthelpers import (
    add_corner_menu,
    bring_to_front,
    configure_menu_about_to_show,
)
from cdl.widgets import instconfviewer, logviewer, status

if TYPE_CHECKING:
    from typing import Literal

    from cdl.core.gui.panel.base import AbstractPanel, BaseDataPanel
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.macro import MacroPanel
    from cdl.core.gui.panel.signal import SignalPanel
    from cdl.plugins import PluginBase


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


class CDLMainWindowMeta(type(QW.QMainWindow), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class CDLMainWindow(QW.QMainWindow, AbstractCDLControl, metaclass=CDLMainWindowMeta):
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
        if CDLMainWindow.__instance is None:
            return CDLMainWindow(console, hide_on_close)
        return CDLMainWindow.__instance

    def __init__(self, console=None, hide_on_close=False):
        """Initialize main window"""
        CDLMainWindow.__instance = self
        super().__init__()
        self.setObjectName(APP_NAME)
        self.setWindowIcon(get_icon("DataLab.svg"))

        execenv.log(self, "Starting initialization")

        self.ready_flag = True

        self.hide_on_close = hide_on_close
        self.__old_size: tuple[int, int] | None = None
        self.__memory_warning = False
        self.memorystatus: status.MemoryStatus | None = None

        self.console: DockableConsole | None = None
        self.macropanel: MacroPanel | None = None

        self.main_toolbar: QW.QToolBar | None = None
        self.signalpanel_toolbar: QW.QToolBar | None = None
        self.imagepanel_toolbar: QW.QToolBar | None = None
        self.signalpanel: SignalPanel | None = None
        self.imagepanel: ImagePanel | None = None
        self.tabwidget: QW.QTabWidget | None = None
        self.tabmenu: QW.QMenu | None = None
        self.docks: dict[AbstractPanel, QW.QDockWidget] | None = None
        self.h5inputoutput = H5InputOutput(self)

        self.openh5_action: QW.QAction | None = None
        self.saveh5_action: QW.QAction | None = None
        self.browseh5_action: QW.QAction | None = None
        self.settings_action: QW.QAction | None = None
        self.quit_action: QW.QAction | None = None
        self.autorefresh_action: QW.QAction | None = None
        self.showfirstonly_action: QW.QAction | None = None
        self.showlabel_action: QW.QAction | None = None

        self.file_menu: QW.QMenu | None = None
        self.edit_menu: QW.QMenu | None = None
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

    def __get_datapanel(self, panel: str | None) -> BaseDataPanel:
        """Return a specific BaseDataPanel.

        Args:
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used.

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
    def get_group_titles_with_object_infos(
        self,
    ) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """Return groups titles and lists of inner objects uuids and titles.

        Returns:
            Groups titles, lists of inner objects uuids and titles
        """
        panel = self.__get_current_basedatapanel()
        return panel.objmodel.get_group_titles_with_object_infos()

    @remote_controlled
    def get_object_titles(self, panel: str | None = None) -> list[str]:
        """Get object (signal/image) list for current panel.
        Objects are sorted by group number and object index in group.

        Args:
            panel: panel name (valid values: "signal", "image", "macro").
             If None, current data panel is used (i.e. signal or image panel).

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

    @remote_controlled
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
        self, title: str, panel: str | None = None, select: bool = False
    ) -> None:
        """Add group to DataLab.

        Args:
            title: Group title
            panel: Panel name (valid values: "signal", "image"). Defaults to None.
            select: Select the group after creation. Defaults to False.
        """
        self.__get_datapanel(panel).add_group(title, select)

    @remote_controlled
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
        panel = self.__get_datapanel(panel)
        panel.objview.select_objects(selection)

    @remote_controlled
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
        obj = self.get_object(nb_id_title, panel)
        return list(obj.iterate_shape_items(editable=False))

    @remote_controlled
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
        panel = self.__get_datapanel(panel)
        panel.add_annotations_from_items(items, refresh_plot)

    @remote_controlled
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
        if cdl.__version__.replace(".", "").isdigit():
            # This is a stable release
            return
        if "b" in cdl.__version__:
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
            f"<b>{APP_NAME}</b> v{cdl.__version__}:",
            "",
            _("<i>This is not a stable release.</i>"),
            "",
            rel,
        ]
        if not env.execenv.unattended:
            QW.QMessageBox.warning(
                self, APP_NAME, "<br>".join(txtlist), QW.QMessageBox.Ok
            )

    def __check_dependencies(self) -> None:  # pragma: no cover
        """Check dependencies"""
        if IS_FROZEN or execenv.unattended:
            # No need to check dependencies if DataLab has been frozen, or if
            # the user has chosen to ignore this check, or if we are in unattended mode
            # (i.e. running automated tests)

            if IS_FROZEN:
                QW.QMessageBox.information(
                    self,
                    _("Information"),
                    _(
                        "The dependency check feature is not relevant for the "
                        "standalone version of DataLab."
                    ),
                    QW.QMessageBox.Ok,
                )
            return
        try:
            state = dephash.check_dependencies_hash(DATAPATH)
            bad_deps = [name for name in state if not state[name]]
            if not bad_deps:
                # Everything is OK
                QW.QMessageBox.information(
                    self,
                    _("Information"),
                    _(
                        "All critical dependencies of DataLab have been qualified "
                        "on this operating system."
                    ),
                    QW.QMessageBox.Ok,
                )
                return
        except IOError:
            bad_deps = None
        txt0 = _("Non-compliant dependency:")
        if bad_deps is None or len(bad_deps) > 1:
            txt0 = _("Non-compliant dependencies:")
        if bad_deps is None:
            txtlist = [
                _("DataLab has not yet been qualified on your operating system."),
            ]
        else:
            txtlist = [
                "<u>" + txt0 + "</u> " + ", ".join(bad_deps),
                "",
                _(
                    "At least one dependency does not comply with DataLab "
                    "qualification standard reference (wrong dependency version "
                    "has been installed, or dependency source code has been "
                    "modified, or the application has not yet been qualified "
                    "on your operating system)."
                ),
            ]
        txtlist += [
            "",
            _(
                "This means that the application has not been officially qualified "
                "in this context and may not behave as expected."
            ),
        ]
        txt = "<br>".join(txtlist)
        QW.QMessageBox.warning(self, APP_NAME, txt, QW.QMessageBox.Ok)

    def check_for_previous_crash(self) -> None:  # pragma: no cover
        """Check for previous crash"""
        if execenv.unattended and not execenv.do_not_quit:
            # Showing the log viewer for testing purpose (unattended mode) but only
            # if option 'do_not_quit' is not set, to avoid blocking the test suite
            self.__show_logviewer()
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

    def execute_post_show_actions(self) -> None:
        """Execute post-show actions"""
        self.check_stable_release()
        self.check_for_previous_crash()
        tour = Conf.main.tour_enabled.get()
        if tour:
            Conf.main.tour_enabled.set(False)
            self.show_tour()

    def take_screenshot(self, name: str) -> None:  # pragma: no cover
        """Take main window screenshot"""
        self.memorystatus.set_demo_mode(True)
        qth.grab_save_window(self, f"{name}")
        self.memorystatus.set_demo_mode(False)

    def take_menu_screenshots(self) -> None:  # pragma: no cover
        """Take menu screenshots"""
        for panel in self.panels:
            if isinstance(panel, base.BaseDataPanel):
                self.tabwidget.setCurrentWidget(panel)
                for name in (
                    "file",
                    "edit",
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

    # ------GUI setup
    def __restore_pos_and_size(self) -> None:
        """Restore main window position and size from configuration"""
        pos = Conf.main.window_position.get(None)
        if pos is not None:
            posx, posy = pos
            self.move(QC.QPoint(posx, posy))
        size = Conf.main.window_size.get(None)
        if size is None:
            size = 1200, 700
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
        self.__configure_statusbar()
        self.__setup_global_actions()
        self.__add_signal_image_panels()
        self.__create_plugins_actions()
        self.__setup_central_widget()
        self.__add_menus()
        if console:
            self.__setup_console()
        self.__update_actions(update_other_data_panel=True)
        self.__add_macro_panel()
        self.__configure_panels()
        # Now that everything is set up, we can restore the window state:
        self.__restore_state()

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
        while PluginRegistry.get_plugins():
            # Unregistering plugin
            plugin = PluginRegistry.get_plugins()[-1]
            with qth.try_or_log_error(f"Unregistering plugin {plugin.info.name}"):
                plugin.unregister()

    def __configure_statusbar(self) -> None:
        """Configure status bar"""
        self.statusBar().showMessage(_("Welcome to %s!") % APP_NAME, 5000)
        # Plugin status
        pluginstatus = status.PluginStatus()
        self.statusBar().addPermanentWidget(pluginstatus)
        # XML-RPC server status
        xmlrpcstatus = status.XMLRPCStatus()
        xmlrpcstatus.set_port(self.remote_server.port)
        self.statusBar().addPermanentWidget(xmlrpcstatus)
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
            tip=_("Open one or several HDF5 files"),
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
            toggled=self.toggle_auto_refresh,
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
        cdock = self.__add_dockwidget(self.__add_signal_panel(), title=_("Signal View"))
        idock = self.__add_dockwidget(self.__add_image_panel(), title=_("Image View"))
        self.tabifyDockWidget(cdock, idock)
        self.docks = {self.signalpanel: cdock, self.imagepanel: idock}
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
        self.tabwidget.setMaximumWidth(500)
        self.tabwidget.addTab(
            self.signalpanel, get_icon("signal.svg"), _("Signal Panel")
        )
        self.tabwidget.addTab(self.imagepanel, get_icon("image.svg"), _("Image Panel"))
        self.setCentralWidget(self.tabwidget)

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
        self.file_menu = self.menuBar().addMenu(_("File"))
        configure_menu_about_to_show(self.file_menu, self.__update_file_menu)
        self.edit_menu = self.menuBar().addMenu(_("&Edit"))
        self.operation_menu = self.menuBar().addMenu(_("Operations"))
        self.processing_menu = self.menuBar().addMenu(_("Processing"))
        self.analysis_menu = self.menuBar().addMenu(_("Analysis"))
        self.plugins_menu = self.menuBar().addMenu(_("Plugins"))
        self.view_menu = self.menuBar().addMenu(_("&View"))
        configure_menu_about_to_show(self.view_menu, self.__update_view_menu)
        self.help_menu = self.menuBar().addMenu("?")
        for menu in (
            self.edit_menu,
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
                triggered=lambda: instconfviewer.exec_cdl_installconfig_dialog(self),
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
                _("Check critical dependencies..."),
                triggered=self.__check_dependencies,
            ),
            create_action(
                self,
                _("About..."),
                icon=get_icon("libre-gui-about.svg"),
                triggered=self.__about,
            ),
        ]
        add_actions(self.help_menu, help_menu_actions)

    def __setup_console(self) -> None:
        """Add an internal console"""
        ns = {
            "cdl": self,
            "np": np,
            "sps": sps,
            "spi": spi,
            "os": os,
            "sys": sys,
            "osp": osp,
            "time": time,
        }
        msg = (
            "Welcome to DataLab console!\n"
            "---------------------------\n"
            "You can access the main window with the 'cdl' variable.\n"
            "Example:\n"
            "  o = cdl.get_object()  # returns currently selected object\n"
            "  o = cdl[1]  # returns object number 1\n"
            "  o = cdl['My image']  # returns object which title is 'My image'\n"
            "  o.data  # returns object data\n"
            "Modules imported at startup: "
            "os, sys, os.path as osp, time, "
            "numpy as np, scipy.signal as sps, scipy.ndimage as spi"
        )
        self.console = DockableConsole(self, namespace=ns, message=msg, debug=DEBUG)
        self.console.setMaximumBlockCount(Conf.console.max_line_count.get(5000))
        self.console.go_to_error.connect(go_to_error)
        console_dock = self.__add_dockwidget(self.console, _("Console"))
        console_dock.hide()
        self.console.interpreter.widget_proxy.sig_new_prompt.connect(
            lambda txt: self.repopulate_panel_trees()
        )

    def __add_macro_panel(self) -> None:
        """Add macro panel"""
        self.macropanel = macro.MacroPanel()
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
    def set_current_panel(self, panel: str) -> None:
        """Switch to panel.

        Args:
            panel: panel name (valid values: "signal", "image", "macro")

        Raises:
            ValueError: unknown panel
        """
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
        """Call compute function ``name`` in current panel's processor.

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
                for funcname in (name, f"compute_{name}"):
                    func = getattr(panel.processor, funcname, None)
                    if func is not None:
                        if param is None:
                            func()
                        else:
                            func(param)
                        return
        raise ValueError(f"Unknown function {name}")

    # ------GUI refresh
    def has_objects(self) -> bool:
        """Return True if sig/ima panels have any object"""
        return sum(len(panel) for panel in self.panels) > 0

    def set_modified(self, state: bool = True) -> None:
        """Set mainwindow modified state"""
        state = state and self.has_objects()
        self.__is_modified = state
        title = APP_NAME + ("*" if state else "")
        if not cdl.__version__.replace(".", "").isdigit():
            title += f" [{cdl.__version__}]"
        self.setWindowTitle(title)

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
            self.edit_menu: ActionCategory.EDIT,
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
        if self.quit_action is not None:
            add_actions(self.file_menu, [None, self.quit_action])

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
                filename, _fl = getsavefilename(self, _("Save"), basedir, "HDF5 (*.h5)")
            if not filename:
                return
        with qth.qt_try_loadsave_file(self, filename, "save"):
            filename = self.__check_h5file(filename, "save")
            self.h5inputoutput.save_file(filename)
            self.set_modified(False)

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
            reset_all = Conf.io.h5_clear_workspace.get()
            if self.has_objects() and Conf.io.h5_clear_workspace_ask.get():
                answer = QW.QMessageBox.question(
                    self,
                    _("Warning"),
                    _(
                        "Do you want to clear current workspace (signals and images) "
                        "before importing data from HDF5 files?<br><br>"
                        "Choosing to ignore this message will prevent it "
                        "from being displayed again, and will use the "
                        "current setting (%s)."
                    )
                    % (_("Yes") if reset_all else _("No")),
                    QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Ignore,
                )
                if answer == QW.QMessageBox.Yes:
                    reset_all = True
                elif answer == QW.QMessageBox.Ignore:
                    Conf.io.h5_clear_workspace_ask.set(False)
        if h5files is None:
            basedir = Conf.main.base_dir.get()
            with qth.save_restore_stds():
                h5files, _fl = getopenfilenames(
                    self, _("Open"), basedir, _("HDF5 files (*.h5 *.hdf5)")
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

    # ------Other methods related to AbstractCDLControl interface
    def get_version(self) -> str:
        """Return DataLab public version.

        Returns:
            DataLab version
        """
        return cdl.__version__

    def close_application(self) -> None:  # Implementing AbstractCDLControl interface
        """Close DataLab application"""
        self.close()

    def raise_window(self) -> None:  # Implementing AbstractCDLControl interface
        """Raise DataLab window"""
        bring_to_front(self)

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
            f"""<b>{APP_NAME}</b> v{cdl.__version__}<br>{APP_DESC}
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
        for option in changed_options:
            if option == "color_mode":
                self.__update_color_mode()
            if option == "plot_toolbar_position":
                for dock in self.docks.values():
                    widget = dock.widget()
                    if isinstance(widget, DockablePlotWidget):
                        widget.update_toolbar_position()
            if option.startswith("sig_autodownsampling"):
                self.signalpanel.refresh_plot("existing", True, False)
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

    def __show_logviewer(self) -> None:
        """Show error logs"""
        logviewer.exec_cdl_logviewer_dialog(self)

    def play_demo(self) -> None:
        """Play demo"""
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from cdl.tests.scenarios import demo

        demo.play_demo(self)

    def show_tour(self) -> None:
        """Show tour"""
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from cdl.core.gui import tour

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
        if not env.execenv.unattended and self.__is_modified:
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
                if self.__is_modified:
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
