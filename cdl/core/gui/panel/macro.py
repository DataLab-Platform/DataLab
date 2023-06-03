# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""DataLab Macro Panel"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os
import os.path as osp
from typing import TYPE_CHECKING

from guidata.config import CONF
from guidata.configtools import get_font, get_icon
from guidata.qthelpers import add_actions, create_action, is_dark_mode
from guidata.qtwidgets import DockableWidgetMixin
from guidata.widgets.console.shell import PythonShellWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getsavefilename

from cdl.config import Conf, _
from cdl.core.gui.macroeditor import Macro
from cdl.core.gui.panel.base import AbstractPanel
from cdl.env import execenv
from cdl.utils.qthelpers import qt_try_loadsave_file, save_restore_stds

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer


class MacroTabs(QW.QTabWidget):
    """Macro tabwidget

    Args:
        parent (QWidget): Parent widget
    """

    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)

    def contextMenuEvent(self, event):  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())


class MacroPanel(AbstractPanel, DockableWidgetMixin):
    """Macro manager widget

    Args:
        parent (QWidget): Parent widget
    """

    LOCATION = QC.Qt.LeftDockWidgetArea
    PANEL_STR = _("Macro panel")

    H5_PREFIX = "DataLab_Mac"

    SIG_OBJECT_MODIFIED = QC.Signal()

    FILE_FILTERS = f"{_('Python files')} (*.py)"

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Macro manager"))
        self.setWindowIcon(get_icon("libre-gui-cogs.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.context_menu = QW.QMenu()

        self.console = PythonShellWidget(self)
        self.console.set_light_background(not is_dark_mode())
        self.console.setMaximumBlockCount(5000)
        font = get_font(CONF, "console")
        font.setPointSize(10)
        self.console.set_font(font)
        self.console.write(_("-***- Macro Console -***-"), prompt=True)

        self.tabwidget = MacroTabs(self)
        self.tabwidget.tabBarDoubleClicked.connect(self.rename_macro)
        self.tabwidget.tabCloseRequested.connect(self.remove_macro)
        self.tabwidget.currentChanged.connect(self.current_macro_changed)

        for widget in (self.tabwidget, self.console):
            self.addWidget(widget)

        self.run_action = None
        self.stop_action = None
        self.obj_actions: list[QW.QAction] = []  # Object-dependent actions
        self.__macros: dict[str, Macro] = {}

        self.setup_actions()

    # ------AbstractPanel interface-----------------------------------------------------
    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file

        Args:
            writer (NativeH5Writer): HDF5 writer
        """
        with writer.group(self.H5_PREFIX):
            for obj in self.__macros.values():
                self.serialize_object_to_hdf5(obj, writer)

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file

        Args:
            reader (NativeH5Reader): HDF5 reader
        """
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                #  Contrary to signal or image panels, macros are not stored
                #  in a group but directly in the root of the HDF5 file
                obj = self.deserialize_object_from_hdf5(reader, name)
                self.add_object(obj)

    @property
    def object_number(self) -> int:
        """Return object number

        Returns:
            int: Number of objects
        """
        return len(self.__macros)

    def create_object(self, title=None) -> Macro:
        """Create object.

        Args:
            title (str): Title of the object

        Returns:
            Macro: Macro object
        """
        macro = Macro(self.console, title)
        macro.objectNameChanged.connect(self.macro_name_changed)
        macro.STARTED.connect(
            lambda orig_macro=macro: self.macro_state_changed(orig_macro, True)
        )
        macro.FINISHED.connect(
            lambda orig_macro=macro: self.macro_state_changed(orig_macro, False)
        )
        macro.MODIFIED.connect(self.macro_contents_changed)
        return macro

    def add_object(self, obj: Macro) -> None:
        """Add object.

        Args:
            obj (Macro): Macro object
        """
        index = self.tabwidget.addTab(obj.editor, obj.title)
        self.__macros[obj.uuid] = obj
        self.SIG_OBJECT_ADDED.emit()
        self.tabwidget.setCurrentIndex(index)

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        while self.tabwidget.count() > 0:
            self.tabwidget.removeTab(0)
        super().remove_all_objects()

    # ---- Macro panel API -------------------------------------------------------------
    def setup_actions(self) -> None:
        """Setup macro menu actions"""
        self.run_action = create_action(
            self,
            _("Run macro"),
            icon=get_icon("libre-camera-flash-on.svg"),
            triggered=self.run_macro,
            shortcut="Ctrl+F5",
        )
        self.stop_action = create_action(
            self,
            _("Stop macro"),
            icon=get_icon("libre-camera-flash-off.svg"),
            triggered=self.stop_macro,
            shortcut="Shift+F5",
        )
        self.stop_action.setDisabled(True)
        add_act = create_action(
            self,
            _("New macro"),
            icon=get_icon("libre-gui-add.svg"),
            triggered=self.add_macro,
        )
        ren_act = create_action(
            self,
            _("Rename macro"),
            icon=get_icon("libre-gui-pencil.svg"),
            triggered=self.rename_macro,
        )
        exp_act = create_action(
            self,
            _("Export macro to file"),
            icon=get_icon("export.svg"),
            triggered=self.export_macro_to_file,
        )
        imp_act = create_action(
            self,
            _("Import macro from file"),
            icon=get_icon("import.svg"),
            triggered=self.import_macro_from_file,
        )
        rem_act = create_action(
            self,
            _("Remove macro"),
            icon=get_icon("libre-gui-action-delete.svg"),
            triggered=self.remove_macro,
        )
        actions = (
            self.run_action,
            self.stop_action,
            None,
            add_act,
            ren_act,
            exp_act,
            imp_act,
            None,
            rem_act,
        )
        self.obj_actions += [
            self.run_action,
            self.stop_action,
            ren_act,
            exp_act,
            rem_act,
        ]

        self.tabwidget.SIG_CONTEXT_MENU.connect(self.__popup_contextmenu)

        toolbar = QW.QToolBar(_("Macro editor toolbar"), self)
        menu_button = QW.QPushButton(get_icon("libre-gui-menu.svg"), "", self)
        menu_button.setFlat(True)
        menu_button.setMenu(self.context_menu)
        self.context_menu.aboutToShow.connect(self.__update_actions)
        toolbar.addWidget(menu_button)
        self.tabwidget.setCornerWidget(toolbar)

        add_actions(toolbar, [self.run_action, self.stop_action, None])
        add_actions(self.context_menu, actions)

    def __update_actions(self) -> None:
        """Update actions"""
        not_empty = self.tabwidget.count() > 0
        for action in self.obj_actions:
            action.setEnabled(not_empty)
        if not_empty:
            self.current_macro_changed()

    def __popup_contextmenu(self, position: QC.QPoint) -> None:  # pragma: no cover
        """Popup context menu at position

        Args:
            position (QPoint): Position of the context menu
        """
        self.__update_actions()
        self.context_menu.popup(position)

    def get_macro(self, index: int | None = None) -> Macro | None:
        """Return macro at index (if index is None, return current macro)

        Args:
            index (int, optional): Index of the macro. Defaults to None.

        Returns:
            Macro: Macro object
        """
        if index is None:
            index = self.tabwidget.currentIndex()
        for macro in self.__macros.values():
            if self.tabwidget.widget(index) is macro.editor:
                return macro
        return None

    # pylint: disable=unused-argument
    def current_macro_changed(self) -> None:
        """Current macro has changed"""
        macro = self.get_macro()
        if macro is not None:
            state = macro.is_running()
            self.macro_state_changed(macro, state)

    def macro_contents_changed(self) -> None:
        """One of the macro contents has changed"""
        self.SIG_OBJECT_MODIFIED.emit()

    def run_macro(self, index: int | None = None) -> None:
        """Run current macro

        Args:
            index (int, optional): Index of the macro. Defaults to None.
        """
        macro = self.get_macro(index)
        assert macro is not None
        macro.run()

    def stop_macro(self, index: int | None = None) -> None:
        """Stop current macro

        Args:
            index (int, optional): Index of the macro. Defaults to None.
        """
        macro = self.get_macro(index)
        assert macro is not None
        macro.kill()

    def macro_state_changed(self, orig_macro: Macro, state: bool) -> None:
        """Macro state has changed (True: started, False: stopped)

        Args:
            orig_macro (Macro): Macro object
            state (bool): State of the macro
        """
        macro = self.get_macro()
        if macro is orig_macro:
            self.run_action.setEnabled(not state)
            self.stop_action.setEnabled(state)

    def add_macro(self, name: str | None = None) -> Macro:
        """Add macro, optionally with name

        Args:
            name (str, optional): Name of the macro. Defaults to None.
                If None, a dialog box will be opened to ask for a name.

        Returns:
            Macro: Macro object
        """
        macro = self.create_object(name)
        self.add_object(macro)
        if name is None:
            self.rename_macro()
        return macro

    def macro_name_changed(self, name: str) -> None:
        """Macro name has been changed

        Args:
            name (str): New name of the macro
        """
        index = self.indexOf(self.tabwidget)
        self.tabwidget.setTabText(index, name)

    def rename_macro(self, index: int | None = None) -> None:
        """Rename macro

        Args:
            index (int, optional): Index of the macro. Defaults to None.
        """
        macro = self.get_macro(index)
        assert macro is not None
        name, valid = QW.QInputDialog.getText(
            self,
            _("Rename"),
            _("New title:"),
            QW.QLineEdit.Normal,
            macro.objectName(),
        )
        if valid:
            macro.setObjectName(name)
            if index is not None:
                self.tabwidget.setCurrentIndex(index)

    def export_macro_to_file(
        self, index: int | None = None, filename: str | None = None
    ) -> None:
        """Export macro to file

        Args:
            index (int, optional): Index of the macro. Defaults to None.
            filename (str, optional): Filename. Defaults to None.
        """
        macro = self.get_macro(index)
        assert macro is not None
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Save as"), basedir, self.FILE_FILTERS
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                macro.to_file(filename)

    def import_macro_from_file(self, filename: str | None = None) -> Macro | None:
        """Import macro from file

        Args:
            filename (str, optional): Filename. Defaults to None.

        Returns:
            Macro: Macro object or None
        """
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getopenfilename(
                    self, _("Open"), basedir, self.FILE_FILTERS
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                Conf.main.base_dir.set(filename)
                macro = self.add_macro(osp.basename(filename))
                macro.from_file(filename)
            return macro
        return None

    def remove_macro(self, index: int | None = None) -> None:
        """Remove macro

        Args:
            index (int, optional): Index of the macro. Defaults to None.
        """
        if index is None:
            index = self.tabwidget.currentIndex()
        txt = "<br>".join(
            [
                _(
                    "When closed, the macro is <u>permanently destroyed</u>, "
                    "unless it has been exported first."
                ),
                "",
                _("Do you want to continue?"),
            ]
        )
        btns = QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No
        if execenv.unattended:
            choice = QW.QMessageBox.StandardButton.Yes
        else:
            choice = QW.QMessageBox.warning(self, self.windowTitle(), txt, btns)
        if choice == QW.QMessageBox.StandardButton.Yes:
            self.tabwidget.removeTab(index)
            self.SIG_OBJECT_REMOVED.emit()
