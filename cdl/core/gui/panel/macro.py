# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Macro panel (see parent package :mod:`cdl.core.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp
import re
from typing import TYPE_CHECKING

from guidata.config import CONF
from guidata.configtools import get_font, get_icon
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.console.shell import PythonShellWidget
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getsavefilename

from cdl.config import Conf, _
from cdl.core.gui.macroeditor import Macro
from cdl.core.gui.panel.base import AbstractPanel
from cdl.env import execenv
from cdl.utils.qthelpers import (
    create_menu_button,
    qt_try_loadsave_file,
    save_restore_stds,
)

if TYPE_CHECKING:
    from guidata.widgets.codeeditor import CodeEditor

    from cdl.core.io.native import NativeH5Reader, NativeH5Writer


class MacroTabs(QW.QTabWidget):
    """Macro tabwidget

    Args:
        parent (QWidget): Parent widget
    """

    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)
    SIG_RENAME = QC.Signal(int)
    SIG_REMOVE = QC.Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabBarDoubleClicked.connect(self.__rename)
        self.tabCloseRequested.connect(self.__remove)
        self.__titles: list[str] = []

    def clear(self) -> None:
        """Override Qt method"""
        super().clear()
        self.__titles.clear()

    def contextMenuEvent(self, event):  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())

    def __rename(self, index: int) -> None:
        """Rename tab

        Args:
            index: Index of the tab
        """
        self.SIG_RENAME.emit(index + 1)

    def __remove(self, index: int) -> None:
        """Remove tab

        Args:
            index: Index of the tab
        """
        self.SIG_REMOVE.emit(index + 1)

    def __update_tab_titles(self) -> None:
        """Update tab titles"""
        for number, title in enumerate(self.__titles, 1):
            self.setTabText(number - 1, f"{number:02d}: {title}")

    def add_tab(self, macro: Macro) -> int:
        """Add tab

        Args:
            macro: Macro object

        Returns:
            int: Number of the tab (starting at 1)
        """
        self.__titles.append(macro.title)
        index = self.addTab(macro.editor, "")
        self.__update_tab_titles()
        return index + 1  # Numbering starts at 1

    def remove_tab(self, number: int) -> None:
        """Remove tab

        Args:
            number: Number of the tab (starting at 1)
        """
        self.removeTab(number - 1)
        self.__titles.pop(number - 1)
        self.__update_tab_titles()

    def get_widget(self, number: int) -> CodeEditor:
        """Return macro editor widget at number

        Args:
            number: Number of the tab (starting at 1)

        Returns:
            Macro editor widget
        """
        return self.widget(number - 1)

    def set_current_number(self, number: int) -> None:
        """Set current tab number

        Args:
            number: Number of the tab (starting at 1)
        """
        self.setCurrentIndex(number - 1)

    def get_current_number(self) -> int:
        """Return current tab number

        Returns:
            int: Number of the tab (starting at 1)
        """
        return self.currentIndex() + 1

    def set_tab_title(self, number: int, name: str) -> None:
        """Set tab title

        Args:
            number: Number of the tab (starting at 1)
            name: Macro name
        """
        self.__titles[number - 1] = name
        self.__update_tab_titles()


class MacroPanel(AbstractPanel, DockableWidgetMixin):
    """Macro Panel widget

    Args:
        parent (QWidget): Parent widget
    """

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("Macro panel")

    H5_PREFIX = "DataLab_Mac"

    SIG_OBJECT_MODIFIED = QC.Signal()

    FILE_FILTERS = f"{_('Python files')} (*.py)"

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.PANEL_STR)
        self.setWindowIcon(get_icon("libre-gui-cogs.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.context_menu = QW.QMenu()
        self.tabwidget_tb = QW.QToolBar(self)
        self.tabwidget_tb.setOrientation(QC.Qt.Vertical)

        self.console = PythonShellWidget(self, read_only=True)
        self.console.setMaximumBlockCount(5000)
        font = get_font(CONF, "console")
        font.setPointSize(10)
        self.console.set_font(font)
        self.console.write(_("-***- Macro Console -***-"), prompt=True)

        self.tabwidget = MacroTabs(self)
        self.tabwidget.SIG_RENAME.connect(self.rename_macro)
        self.tabwidget.SIG_REMOVE.connect(self.remove_macro)
        self.tabwidget.currentChanged.connect(self.__update_actions)

        tabwidget_with_tb = QW.QWidget(self)
        tabwidget_with_tb.setLayout(QW.QHBoxLayout())
        tabwidget_with_tb.layout().addWidget(self.tabwidget_tb)
        tabwidget_with_tb.layout().addWidget(self.tabwidget)

        # Put console in a groupbox to have a title
        console_groupbox = QW.QGroupBox(_("Console"), self)
        console_groupbox.setLayout(QW.QHBoxLayout())
        console_groupbox.layout().addWidget(self.console)
        # Put console groupbox in a frame to have a nice margin
        console_frame = QW.QFrame(self)
        console_frame.setLayout(QW.QHBoxLayout())
        console_frame.layout().addWidget(console_groupbox)

        for widget in (tabwidget_with_tb, console_frame):
            self.addWidget(widget)
        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 1)

        self.run_action = None
        self.stop_action = None
        self.obj_actions: list[QW.QAction] = []  # Object-dependent actions
        self.__macros: list[Macro] = []

        self.setup_actions()

    def update_color_mode(self) -> None:
        """Update color mode according to the current theme"""
        self.console.update_color_mode()
        for macro in self.__macros:
            macro.editor.update_color_mode()

    # ------AbstractPanel interface-----------------------------------------------------
    # pylint: disable=unused-argument
    def get_serializable_name(self, obj: Macro) -> str:
        """Return serializable name of object"""
        title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
        name = f"{obj.PREFIX}{(self.__macros.index(obj) + 1):03d}: {title}"
        return name

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file

        Args:
            writer: HDF5 writer
        """
        with writer.group(self.H5_PREFIX):
            for obj in self.__macros:
                self.serialize_object_to_hdf5(obj, writer)

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file

        Args:
            reader: HDF5 reader
        """
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                #  Contrary to signal or image panels, macros are not stored
                #  in a group but directly in the root of the HDF5 file
                obj = self.deserialize_object_from_hdf5(reader, name)
                self.add_object(obj)

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self.__macros)

    def __getitem__(self, nb: int) -> Macro:
        """Return object from its number (1 to N)"""
        return self.__macros[nb - 1]

    def __iter__(self):
        """Iterate over objects"""
        return iter(self.__macros)

    def create_object(self) -> Macro:
        """Create object.

        Returns:
            Macro object
        """
        macro = Macro(self.console)
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
            obj: Macro object
        """
        self.__macros.append(obj)
        number = self.tabwidget.add_tab(obj)
        self.SIG_OBJECT_ADDED.emit()
        self.tabwidget.set_current_number(number)

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        self.tabwidget.clear()
        self.__macros.clear()
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
        imp_act = create_action(
            self,
            _("Import macro from file"),
            icon=get_icon("fileopen_py.svg"),
            triggered=self.import_macro_from_file,
        )
        exp_act = create_action(
            self,
            _("Export macro to file"),
            icon=get_icon("filesave_py.svg"),
            triggered=self.export_macro_to_file,
        )
        rem_act = create_action(
            self,
            _("Remove macro"),
            icon=get_icon("libre-gui-action-delete.svg"),
            triggered=self.remove_macro,
        )
        self.obj_actions += [
            self.run_action,
            self.stop_action,
            ren_act,
            exp_act,
            rem_act,
        ]

        self.tabwidget.SIG_CONTEXT_MENU.connect(self.context_menu.popup)

        tabwidget_corner = QW.QToolBar(_("Macro editor toolbar"), self)
        self.context_menu.aboutToShow.connect(self.__update_actions)
        tabwidget_menu_btn = create_menu_button(self, self.context_menu)
        tabwidget_corner.addWidget(tabwidget_menu_btn)
        self.tabwidget.setCornerWidget(tabwidget_corner)

        main_actions = [
            self.run_action,
            self.stop_action,
            None,
            add_act,
            ren_act,
            imp_act,
            exp_act,
        ]
        add_actions(tabwidget_corner, [self.run_action, self.stop_action])
        add_actions(self.tabwidget_tb, main_actions)
        add_actions(self.context_menu, main_actions + [None, rem_act])

        self.__update_actions()

    def __update_actions(self) -> None:
        """Update actions"""
        not_empty = self.tabwidget.count() > 0
        for action in self.obj_actions:
            action.setEnabled(not_empty)
        if not_empty:
            macro = self.get_macro()
            if macro is not None:
                macro: Macro
                self.macro_state_changed(macro, macro.is_running())

    def get_macro(self, number_or_title: int | str | None = None) -> Macro | None:
        """Return macro at number (if number is None, return current macro)

        Args:
            number: Number of the macro (starting at 1) or title of the macro.
             Defaults to None (current macro).

        Returns:
            Macro object or None (if not found)
        """
        if number_or_title is None:
            number_or_title = self.tabwidget.get_current_number()
        if isinstance(number_or_title, str):
            return self.get_macro(self.get_number_from_title(number_or_title))
        for macro in self.__macros:
            if self.tabwidget.get_widget(number_or_title) is macro.editor:
                return macro
        return None

    def get_number_from_title(self, title: str) -> int | None:
        """Return macro number from title

        Args:
            title: Title of the macro

        Returns:
            Number of the macro (starting at 1) or None (if not found)
        """
        for number in range(1, self.tabwidget.count() + 1):
            if self.tabwidget.tabText(number - 1).endswith(title):
                return number
        return None

    def get_number_from_macro(self, macro: Macro) -> int | None:
        """Return macro number from macro object

        Args:
            macro: Macro object

        Returns:
            Number of the macro (starting at 1) or None (if not found)
        """
        for number in range(1, self.tabwidget.count() + 1):
            if self.tabwidget.get_widget(number) is macro.editor:
                return number
        return None

    def get_macro_titles(self) -> list[str]:
        """Return list of macro titles"""
        return [macro.title for macro in self.__macros]

    def macro_contents_changed(self) -> None:
        """One of the macro contents has changed"""
        self.SIG_OBJECT_MODIFIED.emit()

    def run_macro(self, number_or_title: int | str | None = None) -> None:
        """Run current macro

        Args:
            number: Number of the macro (starting at 1). Defaults to None (run
             current macro, or does nothing if there is no macro).
        """
        macro = self.get_macro(number_or_title)
        if macro is not None:
            macro: Macro
            macro.run()

    def stop_macro(self, number_or_title: int | str | None = None) -> None:
        """Stop current macro

        Args:
            number: Number of the macro (starting at 1). Defaults to None (run
             current macro, or does nothing if there is no macro).
        """
        macro = self.get_macro(number_or_title)
        if macro is not None:
            macro: Macro
            macro.kill()

    def macro_state_changed(self, orig_macro: Macro, state: bool) -> None:
        """Macro state has changed (True: started, False: stopped)

        Args:
            orig_macro: Macro object
            state: State of the macro
        """
        macro = self.get_macro()
        if macro is orig_macro:
            self.run_action.setEnabled(not state)
            self.stop_action.setEnabled(state)

    def add_macro(self) -> Macro:
        """Add macro, optionally with name

        Returns:
            Macro object
        """
        macro = self.create_object()
        self.add_object(macro)
        if not macro.title:
            self.rename_macro()
        return macro

    def macro_name_changed(self, name: str) -> None:
        """Macro name has been changed

        Args:
            name: New name of the macro
        """
        number = self.get_number_from_macro(self.sender())
        if number is not None:
            self.tabwidget.set_tab_title(number, name)

    def rename_macro(self, number: int | None = None, title: str | None = None) -> None:
        """Rename macro

        Args:
            number: Number of the macro (starting at 1). Defaults to None.
            title: Title of the macro. Defaults to None.
        """
        macro = self.get_macro(number)
        assert isinstance(macro, Macro)
        if title is None:
            title, valid = QW.QInputDialog.getText(
                self,
                _("Rename"),
                _("New title:"),
                QW.QLineEdit.Normal,
                macro.title,
            )
            title = title if valid else None
        if title:
            macro.title = title
            if number is not None:
                self.tabwidget.set_current_number(number)

    def export_macro_to_file(
        self, number_or_title: int | str | None = None, filename: str | None = None
    ) -> None:
        """Export macro to file

        Args:
            number_or_title: Number of the macro (starting at 1) or title of the macro.
             Defaults to None.
            filename: Filename. Defaults to None.

        Raises:
            ValueError: If title is not found
        """
        macro = self.get_macro(number_or_title)
        assert isinstance(macro, Macro)
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Save as"), basedir, self.FILE_FILTERS
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                macro.title = osp.basename(filename)
                macro.to_file(filename)

    def import_macro_from_file(self, filename: str | None = None) -> int:
        """Import macro from file

        Args:
            filename: Filename. Defaults to None.

        Returns:
            Number of the macro (starting at 1)
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
                macro = self.add_macro()
                macro.from_file(filename)
            return self.get_number_from_macro(macro)
        return -1

    def remove_macro(self, number_or_title: int | str | None = None) -> None:
        """Remove macro

        Args:
            number_or_title: Number of the macro (starting at 1) or title of the macro.
             Defaults to None.
        """
        if number_or_title is None:
            number_or_title = self.tabwidget.get_current_number()
        if isinstance(number_or_title, str):
            number_or_title = self.get_number_from_title(number_or_title)
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
            self.tabwidget.remove_tab(number_or_title)
            self.__macros.pop(number_or_title - 1)
            self.SIG_OBJECT_REMOVED.emit()
