# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT main window
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import locale
import os
import os.path as osp
import platform
import sys
import time
import webbrowser
from typing import List

import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from guidata import __version__ as guidata_ver
from guidata.configtools import get_icon, get_module_data_path, get_module_path
from guidata.qthelpers import add_actions, create_action, win32_fix_title_bar_background
from guidata.widgets.console import DockableConsole
from guiqwt import __version__ as guiqwt_ver
from guiqwt.builder import make
from guiqwt.plot import CurveWidget, ImageWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilenames, getsavefilename
from qwt import __version__ as qwt_ver

from codraft import __docurl__, __homeurl__, __supporturl__, __version__, env
from codraft.config import APP_DESC, APP_NAME, TEST_SEGFAULT_ERROR, Conf, _
from codraft.core.gui.actionhandler import ActionCategory
from codraft.core.gui.docks import DockablePlotWidget, DockableTabWidget
from codraft.core.gui.h5io import H5InputOutput
from codraft.core.gui.panel import ImagePanel, SignalPanel
from codraft.core.model.image import ImageParam
from codraft.core.model.signal import SignalParam
from codraft.env import execenv
from codraft.utils import dephash
from codraft.utils import qthelpers as qth
from codraft.widgets.instconfviewer import exec_codraft_installconfig_dialog
from codraft.widgets.logviewer import exec_codraft_logviewer_dialog
from codraft.widgets.status import MemoryStatus

DATAPATH = get_module_data_path("codraft", "data")


def get_htmlhelp():
    """Return HTML Help documentation link adapted to locale, if it exists"""
    if os.name == "nt":
        for suffix in ("_" + locale.getlocale()[0][:2], ""):
            path = osp.join(DATAPATH, f"CodraFT{suffix}.chm")
            if osp.isfile(path):
                return path
    return None


class AppProxy:
    """Proxy to CodraFT application: object used from the embedded console
    to access CodraFT internal objects"""

    def __init__(self, win):
        self.win = win
        self.s = self.win.signalpanel.objlist
        self.i = self.win.imagepanel.objlist


def is_frozen(module_name):
    """Test if module has been frozen (py2exe/cx_Freeze)"""
    datapath = get_module_path(module_name)
    parentdir = osp.normpath(osp.join(datapath, osp.pardir))
    return not osp.isfile(__file__) or osp.isfile(parentdir)  # library.zip


class CodraFTMainWindow(QW.QMainWindow):
    """CodraFT main window"""

    __instance = None

    @staticmethod
    def get_instance(console=None, hide_on_close=False):
        """Return singleton instance"""
        if CodraFTMainWindow.__instance is None:
            return CodraFTMainWindow(console, hide_on_close)
        return CodraFTMainWindow.__instance

    def __init__(self, console=None, hide_on_close=False):
        """Initialize main window"""
        CodraFTMainWindow.__instance = self
        super().__init__()
        win32_fix_title_bar_background(self)
        self.setObjectName(APP_NAME)
        self.setWindowIcon(get_icon("codraft.svg"))
        self.__restore_pos_and_size()

        self.hide_on_close = hide_on_close
        self.__old_size = None
        self.__memory_warning = False
        self.memorystatus = None

        self.console = None
        self.app_proxy = None
        self.signal_toolbar = None
        self.image_toolbar = None
        self.signalpanel = None
        self.imagepanel = None
        self.tabwidget = None
        self.signal_image_docks = None
        self.h5inputoutput = H5InputOutput(self)

        self.openh5_action = None
        self.saveh5_action = None
        self.browseh5_action = None
        self.quit_action = None

        self.file_menu = None
        self.edit_menu = None
        self.operation_menu = None
        self.processing_menu = None
        self.computing_menu = None
        self.view_menu = None
        self.help_menu = None

        self.__is_modified = None
        self.set_modified(False)

        # Setup actions and menus
        if console is None:
            console = Conf.console.enable.get(True)
        self.setup(console)

    @property
    def panels(self):
        """Return the tuple of implemented panels (signal, image)"""
        return (self.signalpanel, self.imagepanel)

    def __set_low_memory_state(self, state):
        """Set memory warning state"""
        self.__memory_warning = state

    def confirm_memory_state(self):  # pragma: no cover
        """Check memory warning state and eventually show a warning dialog"""
        if self.__memory_warning:
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

    def check_stable_release(self):  # pragma: no cover
        """Check if this is a stable release"""
        if __version__.replace(".", "").isdigit():
            # This is a stable release
            return
        if "b" in __version__:
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
            f"<b>{APP_NAME}</b> v{__version__}:",
            "",
            _("<i>This is not a stable release.</i>"),
            "",
            rel,
        ]
        QW.QMessageBox.warning(self, APP_NAME, "<br>".join(txtlist), QW.QMessageBox.Ok)

    def check_dependencies(self):  # pragma: no cover
        """Check dependencies"""
        if is_frozen("codraft") or Conf.main.ignore_dependency_check.get(False):
            # No need to check dependencies if CodraFT has been frozen
            return
        try:
            state = dephash.check_dependencies_hash(DATAPATH)
            bad_deps = [name for name in state if not state[name]]
            if not bad_deps:
                # Everything is OK
                return
        except IOError:
            bad_deps = None
        txt0 = _("Non-compliant dependency:")
        if bad_deps is None or len(bad_deps) > 1:
            txt0 = _("Non-compliant dependencies:")
        if bad_deps is None:
            txtlist = [
                _("CodraFT has not yet been qualified on your operating system."),
            ]
        else:
            txtlist = [
                "<u>" + txt0 + "</u> " + ", ".join(bad_deps),
                "",
                "",
                _(
                    "At least one dependency does not comply with CodraFT "
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
            "",
            _(
                "Please click on the Ignore button "
                "to avoid showing this message at startup."
            ),
        ]
        txt = "<br>".join(txtlist)
        btn = QW.QMessageBox.information(
            self, APP_NAME, txt, QW.QMessageBox.Ok | QW.QMessageBox.Ignore
        )
        Conf.main.ignore_dependency_check.set(btn == QW.QMessageBox.Ignore)

    def check_for_previous_crash(self):  # pragma: no cover
        """Check for previous crash"""
        if execenv.unattended:
            self.show_log_viewer()
        elif Conf.main.faulthandler_log_available.get(
            False
        ) or Conf.main.traceback_log_available.get(False):
            txt = "<br>".join(
                [
                    _("Log files were generated during last session."),
                    "",
                    _("Do you want to see available log files?"),
                ]
            )
            btns = QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No
            choice = QW.QMessageBox.warning(self, APP_NAME, txt, btns)
            if choice == QW.QMessageBox.StandardButton.Yes:
                self.show_log_viewer()

    def take_screenshot(self, name):  # pragma: no cover
        """Take main window screenshot"""
        self.memorystatus.set_demo_mode(True)
        qth.grab_save_window(self, f"{name}")
        self.memorystatus.set_demo_mode(False)

    def take_menu_screenshots(self):  # pragma: no cover
        """Take menu screenshots"""
        for panel in self.panels:
            self.tabwidget.setCurrentWidget(panel)
            for name in (
                "file",
                "edit",
                "view",
                "operation",
                "processing",
                "computing",
                "help",
            ):
                menu = getattr(self, f"{name}_menu")
                menu.popup(self.pos())
                qth.grab_save_window(menu, f"{panel.objectName()}_{name}")
                menu.close()

    # ------GUI setup
    def __restore_pos_and_size(self):
        """Restore main window position and size from configuration"""
        pos = Conf.main.window_position.get(None)
        if pos is not None:
            posx, posy = pos
            self.move(QC.QPoint(posx, posy))
        size = Conf.main.window_size.get(None)
        if size is not None:
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

    def __save_pos_and_size(self):
        """Save main window position and size to configuration"""
        is_maximized = self.windowState() == QC.Qt.WindowMaximized
        Conf.main.window_maximized.set(is_maximized)
        if not is_maximized:
            size = self.size()
            Conf.main.window_size.set((size.width(), size.height()))
            pos = self.pos()
            Conf.main.window_position.set((pos.x(), pos.y()))

    def setup(self, console):
        """Setup main window"""
        self.statusBar().showMessage(_("Welcome to %s!") % APP_NAME, 5000)
        self.memorystatus = MemoryStatus(Conf.main.available_memory_threshold.get(500))
        self.memorystatus.SIG_MEMORY_ALARM.connect(self.__set_low_memory_state)
        self.statusBar().addPermanentWidget(self.memorystatus)
        self.__setup_commmon_actions()
        curvewidget = self.__add_signal_panel()
        imagewidget = self.__add_image_panel()
        self.__add_tabwidget(curvewidget, imagewidget)
        self.__add_menus()
        if console:
            self.__setup_console()
        # Update selection dependent actions
        self.__update_actions()
        self.signal_image_docks[0].raise_()
        # Restoring current tab from last session
        tab_idx = Conf.main.current_tab.get(None)
        if tab_idx is not None:
            self.tabwidget.setCurrentIndex(tab_idx)

    def __setup_commmon_actions(self):
        """Setup common actions"""
        self.openh5_action = create_action(
            self,
            _("Open HDF5 files..."),
            icon=get_icon("h5open.svg"),
            tip=_("Open one or several HDF5 files"),
            triggered=lambda checked=False: self.open_h5_files(import_all=True),
        )
        self.saveh5_action = create_action(
            self,
            _("Save to HDF5 file..."),
            icon=get_icon("h5save.svg"),
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
        h5_toolbar = self.addToolBar(_("HDF5 I/O Toolbar"))
        add_actions(
            h5_toolbar, [self.openh5_action, self.saveh5_action, self.browseh5_action]
        )
        # Quit action for "File menu" (added when populating menu on demand)
        if self.hide_on_close:
            quit_text = _("Hide window")
            quit_tip = _("Hide CodraFT window")
        else:
            quit_text = _("Quit")
            quit_tip = _("Quit application")
        self.quit_action = create_action(
            self,
            quit_text,
            shortcut=QG.QKeySequence(QG.QKeySequence.Quit),
            icon=get_icon("libre-gui-close.svg"),
            tip=quit_tip,
            triggered=self.close,
        )

    def __add_signal_panel(self):
        """Setup signal toolbar, widgets and panel"""
        self.signal_toolbar = self.addToolBar(_("Signal Processing Toolbar"))
        curveplot_toolbar = self.addToolBar(_("Curve Plotting Toolbar"))
        curvewidget = DockablePlotWidget(self, CurveWidget, curveplot_toolbar)
        curveplot = curvewidget.get_plot()
        curveplot.add_item(make.legend("TR"))
        self.signalpanel = SignalPanel(
            self, curvewidget.plotwidget, self.signal_toolbar
        )
        self.signalpanel.SIG_STATUS_MESSAGE.connect(self.statusBar().showMessage)
        return curvewidget

    def __add_image_panel(self):
        """Setup image toolbar, widgets and panel"""
        self.image_toolbar = self.addToolBar(_("Image Processing Toolbar"))
        imagevis_toolbar = self.addToolBar(_("Image Visualization Toolbar"))
        imagewidget = DockablePlotWidget(self, ImageWidget, imagevis_toolbar)
        self.imagepanel = ImagePanel(self, imagewidget.plotwidget, self.image_toolbar)
        # -----------------------------------------------------------------------------
        # # Before eventually disabling the "peritem" mode by default, wait for the
        # # guiqwt bug to be fixed (peritem mode is not compatible with multiple image
        # # items):
        # for cspanel in (
        #     self.imagepanel.plotwidget.get_xcs_panel(),
        #     self.imagepanel.plotwidget.get_ycs_panel(),
        # ):
        #     cspanel.peritem_ac.setChecked(False)
        # -----------------------------------------------------------------------------
        self.imagepanel.SIG_STATUS_MESSAGE.connect(self.statusBar().showMessage)
        return imagewidget

    def switch_to_signal_panel(self):
        """Switch to signal panel"""
        self.tabwidget.setCurrentWidget(self.signalpanel)

    def switch_to_image_panel(self):
        """Switch to image panel"""
        self.tabwidget.setCurrentWidget(self.imagepanel)

    def __add_tabwidget(self, curvewidget, imagewidget):
        """Setup tabwidget with signals and images"""
        self.tabwidget = DockableTabWidget()
        self.tabwidget.setMaximumWidth(500)
        self.tabwidget.addTab(self.signalpanel, get_icon("signal.svg"), _("Signals"))
        self.tabwidget.addTab(self.imagepanel, get_icon("image.svg"), _("Images"))
        self.__add_dockwidget(self.tabwidget, _("Main panel"))
        curve_dock = self.__add_dockwidget(curvewidget, title=_("Curve panel"))
        image_dock = self.__add_dockwidget(imagewidget, title=_("Image panel"))
        self.tabifyDockWidget(curve_dock, image_dock)
        self.signal_image_docks = curve_dock, image_dock
        self.tabwidget.currentChanged.connect(self.__tab_index_changed)
        self.signalpanel.SIG_OBJECT_ADDED.connect(self.switch_to_signal_panel)
        self.imagepanel.SIG_OBJECT_ADDED.connect(self.switch_to_image_panel)
        for panel in self.panels:
            panel.SIG_OBJECT_ADDED.connect(self.set_modified)
            panel.SIG_OBJECT_REMOVED.connect(self.set_modified)

    def __add_menus(self):
        """Adding menus"""
        self.file_menu = self.menuBar().addMenu(_("File"))
        self.file_menu.aboutToShow.connect(self.__update_file_menu)
        self.edit_menu = self.menuBar().addMenu(_("&Edit"))
        self.operation_menu = self.menuBar().addMenu(_("Operations"))
        self.processing_menu = self.menuBar().addMenu(_("Processing"))
        self.computing_menu = self.menuBar().addMenu(_("Computing"))
        self.view_menu = self.menuBar().addMenu(_("&View"))
        self.view_menu.aboutToShow.connect(self.__update_view_menu)
        self.help_menu = self.menuBar().addMenu("?")
        for menu in (
            self.edit_menu,
            self.operation_menu,
            self.processing_menu,
            self.computing_menu,
        ):
            menu.aboutToShow.connect(self.__update_generic_menu)
        about_action = create_action(
            self,
            _("About..."),
            icon=get_icon("libre-gui-about.svg"),
            triggered=self.__about,
        )
        homepage_action = create_action(
            self,
            _("Project home page"),
            icon=get_icon("libre-gui-globe.svg"),
            triggered=lambda: webbrowser.open(__homeurl__),
        )
        issue_action = create_action(
            self,
            _("Bug report or feature request"),
            icon=get_icon("libre-gui-globe.svg"),
            triggered=lambda: webbrowser.open(__supporturl__),
        )
        onlinedoc_action = create_action(
            self,
            _("Online documentation"),
            icon=get_icon("libre-gui-help.svg"),
            triggered=lambda: webbrowser.open(__docurl__),
        )
        chmdoc_action = create_action(
            self,
            _("CHM documentation"),
            icon=get_icon("chm.svg"),
            triggered=lambda: os.startfile(get_htmlhelp()),
        )
        chmdoc_action.setVisible(get_htmlhelp() is not None)
        logv_action = create_action(
            self,
            _("Show log files..."),
            icon=get_icon("logs.svg"),
            triggered=self.show_log_viewer,
        )
        dep_action = create_action(
            self,
            _("About CodraFT installation") + "...",
            icon=get_icon("logs.svg"),
            triggered=lambda: exec_codraft_installconfig_dialog(self),
        )
        errtest_action = create_action(
            self, "Test segfault/Python error", triggered=self.test_segfault_error
        )
        errtest_action.setVisible(TEST_SEGFAULT_ERROR)
        about_action = create_action(
            self,
            _("About..."),
            icon=get_icon("libre-gui-about.svg"),
            triggered=self.__about,
        )
        add_actions(
            self.help_menu,
            (
                onlinedoc_action,
                chmdoc_action,
                None,
                errtest_action,
                logv_action,
                dep_action,
                None,
                homepage_action,
                issue_action,
                about_action,
            ),
        )

    def __setup_console(self):
        """Add an internal console"""
        self.app_proxy = AppProxy(self)
        ns = {
            "app": self.app_proxy,
            "np": np,
            "sps": sps,
            "spi": spi,
            "os": os,
            "sys": sys,
            "osp": osp,
            "time": time,
        }
        msg = (
            "Example: app.s[0] returns signal object #0\n"
            "Modules imported at startup: "
            "os, sys, os.path as osp, time, "
            "numpy as np, scipy.signal as sps, scipy.ndimage as spi"
        )
        debug = os.environ.get("DEBUG") == "1"
        self.console = DockableConsole(self, namespace=ns, message=msg, debug=debug)
        self.console.setMaximumBlockCount(Conf.console.max_line_count.get(5000))
        console_dock = self.__add_dockwidget(self.console, _("Console"))
        console_dock.hide()
        self.console.interpreter.widget_proxy.sig_new_prompt.connect(
            lambda txt: self.refresh_lists()
        )

    # ------GUI refresh
    def has_objects(self):
        """Return True if sig/ima panels have any object"""
        return sum([len(panel.objlist) for panel in self.panels]) > 0

    def set_modified(self, state=True):
        """Set mainwindow modified state"""
        state = state and self.has_objects()
        self.__is_modified = state
        self.setWindowTitle(APP_NAME + ("*" if state else ""))

    def __add_dockwidget(self, child, title):
        """Add QDockWidget and toggleViewAction"""
        dockwidget, location = child.create_dockwidget(title)
        self.addDockWidget(location, dockwidget)
        return dockwidget

    def refresh_lists(self):
        """Refresh signal/image lists"""
        for panel in self.panels:
            panel.objlist.refresh_list()

    def __update_actions(self):
        """Update selection dependent actions"""
        is_signal = self.tabwidget.currentWidget() is self.signalpanel
        panel = self.signalpanel if is_signal else self.imagepanel
        panel.selection_changed()
        self.signal_toolbar.setVisible(is_signal)
        self.image_toolbar.setVisible(not is_signal)

    def __tab_index_changed(self, index):
        """Switch from signal to image mode, or vice-versa"""
        dock = self.signal_image_docks[index]
        dock.raise_()
        self.__update_actions()

    def __update_generic_menu(self, menu=None):
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
            self.computing_menu: ActionCategory.COMPUTING,
        }[menu]
        actions = panel.get_category_actions(category)
        add_actions(menu, actions)

    def __update_file_menu(self):
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
                self.quit_action,
            ],
        )

    def __update_view_menu(self):
        """Update view menu before showing up"""
        self.__update_generic_menu(self.view_menu)
        add_actions(self.view_menu, [None] + self.createPopupMenu().actions())

    # ------Common features
    def reset_all(self):
        """Reset all application data"""
        for panel in self.panels:
            panel.remove_all_objects()

    @staticmethod
    def __check_h5file(filename, operation: str):
        """Check HDF5 filename"""
        filename = osp.abspath(osp.normpath(filename))
        bname = osp.basename(filename)
        if operation == "load" and not osp.isfile(filename):
            raise IOError(f'File not found "{bname}"')
        if not filename.endswith(".h5"):
            raise IOError(f'Invalid HDF5 file "{bname}"')
        Conf.main.base_dir.set(filename)
        return filename

    def save_to_h5_file(self, filename=None):
        """Save to a CodraFT HDF5 file"""
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with qth.save_restore_stds():
                filters = f'{_("HDF5 files")} (*.h5)'
                filename, _filter = getsavefilename(self, _("Save"), basedir, filters)
            if not filename:
                return
        with qth.qt_try_loadsave_file(self.parent(), filename, "save"):
            filename = self.__check_h5file(filename, "save")
            self.h5inputoutput.save_file(filename)
            self.set_modified(False)

    def open_h5_files(
        self,
        h5files: List[str] = None,
        import_all: bool = None,
        reset_all: bool = None,
    ) -> None:
        """Open a CodraFT HDF5 file or import from any other HDF5 file

        :param h5files: HDF5 filenames (optionally with dataset name, separated by ":")
        :param import_all: Import all HDF5 file contents
        :param reset_all: Delete all CodraFT signals and images before importing data
        """
        if not self.confirm_memory_state():
            return
        if reset_all is None:
            reset_all = False
            if self.has_objects():
                answer = QW.QMessageBox.question(
                    self,
                    _("Warning"),
                    _(
                        "Do you want to remove all signals and images "
                        "before importing data from HDF5 files?"
                    ),
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                )
                if answer == QW.QMessageBox.Yes:
                    reset_all = True
        if h5files is None:
            basedir = Conf.main.base_dir.get()
            with qth.save_restore_stds():
                filters = f'{_("HDF5 files")} (*.h5)'
                h5files, _filter = getopenfilenames(self, _("Open"), basedir, filters)
        for fname_with_dset in h5files:
            if "," in fname_with_dset:
                filename, dsetname = fname_with_dset.split(",")
            else:
                filename, dsetname = fname_with_dset, None
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

    def import_h5_file(self, filename: str, reset_all: bool = None) -> None:
        """Open CodraFT HDF5 browser to Import HDF5 file

        :param filename: HDF5 filename
        :param reset_all: Delete all CodraFT signals and images before importing data
        """
        with qth.qt_try_loadsave_file(self, filename, "load"):
            filename = self.__check_h5file(filename, "load")
            self.h5inputoutput.import_file(filename, False, reset_all)

    def add_object(self, obj, refresh=True):
        """Add object - signal or image"""
        if self.confirm_memory_state():
            if isinstance(obj, SignalParam):
                self.signalpanel.add_object(obj, refresh=refresh)
            elif isinstance(obj, ImageParam):
                self.imagepanel.add_object(obj, refresh=refresh)
            else:
                raise TypeError(f"Unsupported object type {type(obj)}")

    # ------?
    def __about(self):  # pragma: no cover
        """About dialog box"""
        self.check_stable_release()
        QW.QMessageBox.about(
            self,
            _("About ") + APP_NAME,
            f"""<b>{APP_NAME}</b> v{__version__}<br>{APP_DESC}<p>
              %s Pierre Raybaut
              <br>Copyright &copy; 2018-2022 CEA-CODRA
              <p>PythonQwt {qwt_ver}, guidata {guidata_ver},
              guiqwt {guiqwt_ver}<br>Python {platform.python_version()},
              Qt {QC.__version__}, PyQt {QC.PYQT_VERSION_STR}
               %s {platform.system()}"""
            % (_("Developped by"), _("on")),
        )

    def show_log_viewer(self):
        """Show error logs"""
        exec_codraft_logviewer_dialog(self)

    @staticmethod
    def test_segfault_error():
        """Generate errors (both fault and traceback)"""
        import ctypes  # pylint: disable=import-outside-toplevel

        ctypes.string_at(0)
        raise RuntimeError("!!! Testing RuntimeError !!!")

    def show(self):
        """Reimplement QMainWindow method"""
        super().show()
        if self.__old_size is not None:
            self.resize(self.__old_size)

    # ------Close window
    def closeEvent(self, event):
        """Reimplement QMainWindow method"""
        if self.hide_on_close:
            self.__old_size = self.size()
            self.hide()
        else:
            if not env.execenv.unattended and self.__is_modified:
                answer = QW.QMessageBox.warning(
                    self,
                    _("Quit"),
                    _(
                        "Do you want to save all signals and images "
                        "to an HDF5 file before quitting CodraFT?"
                    ),
                    QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Cancel,
                )
                if answer == QW.QMessageBox.Yes:
                    self.save_to_h5_file()
                    if self.__is_modified:
                        event.ignore()
                        return
                elif answer == QW.QMessageBox.Cancel:
                    event.ignore()
                    return
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
            self.__save_pos_and_size()

            # Saving current tab for next session
            Conf.main.current_tab.set(self.tabwidget.currentIndex())

            event.accept()
