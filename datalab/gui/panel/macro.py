# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Macro panel (see parent package :mod:`datalab.gui.panel`)
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

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui.macroeditor import Macro
from datalab.gui.macros_templates import MacroTemplate, list_templates
from datalab.gui.panel.base import AbstractPanel
from datalab.utils import macrorecovery, recentmacros
from datalab.utils.qthelpers import (
    create_menu_button,
    qt_try_loadsave_file,
    save_restore_stds,
)
from datalab.widgets.codecompleter import PythonCompleter

if TYPE_CHECKING:
    from guidata.widgets.codeeditor import CodeEditor

    from datalab.h5.native import NativeH5Reader, NativeH5Writer


class _RecentMacrosDialog(QW.QDialog):
    """Dialog listing recent macros cached across sessions."""

    def __init__(self, panel: "MacroPanel") -> None:
        super().__init__(panel)
        self._panel = panel
        self.setWindowTitle(_("Recent macros"))
        self.resize(560, 360)

        self.listwidget = QW.QListWidget(self)
        self.listwidget.itemDoubleClicked.connect(self._open_selected)

        self.open_btn = QW.QPushButton(_("Open"), self)
        self.open_btn.clicked.connect(self._open_selected)
        self.remove_btn = QW.QPushButton(_("Remove"), self)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn = QW.QPushButton(_("Clear all"), self)
        self.clear_btn.clicked.connect(self._clear_all)
        close_btn = QW.QPushButton(_("Close"), self)
        close_btn.clicked.connect(self.accept)

        btn_layout = QW.QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)

        layout = QW.QVBoxLayout(self)
        layout.addWidget(self.listwidget)
        layout.addLayout(btn_layout)

        self._refresh()

    def _refresh(self) -> None:
        """Reload the list from the recent cache."""
        self.listwidget.clear()
        entries = recentmacros.list_recent()
        for entry in entries:
            ts = entry.get("last_seen")
            try:
                from datetime import datetime

                when = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError):
                when = "?"
            label = f"{entry.get('title', '?')}  —  {when}"
            source = entry.get("source")
            if source:
                label += f"  [{source}]"
            item = QW.QListWidgetItem(label, self.listwidget)
            item.setData(QC.Qt.UserRole, entry.get("uid"))
        self.open_btn.setEnabled(bool(entries))
        self.remove_btn.setEnabled(bool(entries))
        self.clear_btn.setEnabled(bool(entries))

    def _selected_uid(self) -> str | None:
        """Return the uid of the selected entry, or None."""
        item = self.listwidget.currentItem()
        if item is None:
            return None
        return item.data(QC.Qt.UserRole)

    def _open_selected(self) -> None:
        """Open the selected recent macro as a new tab."""
        uid = self._selected_uid()
        if uid is None:
            return
        entry = recentmacros.get_recent(uid)
        if entry is None:
            self._refresh()
            return
        self._panel.add_macro_with_code(
            entry.get("title", _("Untitled")),
            entry.get("code", ""),
            source="recent",
        )
        self.accept()

    def _remove_selected(self) -> None:
        """Remove the selected entry from the recent cache."""
        uid = self._selected_uid()
        if uid is None:
            return
        recentmacros.remove_recent(uid)
        self._refresh()

    def _clear_all(self) -> None:
        """Clear the recent cache after user confirmation."""
        if execenv.unattended:
            choice = QW.QMessageBox.StandardButton.Yes
        else:
            choice = QW.QMessageBox.question(
                self,
                self.windowTitle(),
                _("Clear all recent macros?"),
                QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No,
            )
        if choice == QW.QMessageBox.StandardButton.Yes:
            recentmacros.clear_recent()
            self._refresh()


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

    def __init__(self, parent: QW.QMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.PANEL_STR)
        self.setWindowIcon(get_icon("libre-gui-cogs.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.context_menu = QW.QMenu()
        self.tabwidget_tb = QW.QToolBar(self)
        self.tabwidget_tb.setOrientation(QC.Qt.Vertical)

        self.console = PythonShellWidget(self, read_only=True)
        try:
            max_lines = int(Conf.macro.console_max_lines.get(5000))
        except (TypeError, ValueError):
            max_lines = 5000
        self.console.setMaximumBlockCount(max_lines)
        # Add a "Clear console" entry at the top of the shell's context menu
        # (the shell already provides "Save history log..." for exporting).
        clear_console_action = create_action(
            self.console,
            _("Clear console"),
            icon=get_icon("libre-gui-action-delete.svg"),
            triggered=self.console.clear,
        )
        shell_menu = getattr(self.console, "menu", None)
        if shell_menu is not None:
            first = shell_menu.actions()[0] if shell_menu.actions() else None
            shell_menu.insertAction(first, clear_console_action)
            if first is not None:
                shell_menu.insertSeparator(first)
        font = get_font(CONF, "console")
        font.setPointSize(10)
        self.console.set_font(font)
        self.console.write(_("-***- Macro Console -***-"), prompt=True)

        self.tabwidget = MacroTabs(self)
        self.tabwidget.SIG_RENAME.connect(self.rename_macro)
        self.tabwidget.SIG_REMOVE.connect(self.remove_macro)
        self.tabwidget.currentChanged.connect(self.__update_actions)
        self.tabwidget.currentChanged.connect(self._persist_active_tab)

        tabwidget_with_tb = QW.QWidget(self)
        tabwidget_with_tb.setLayout(QW.QHBoxLayout())
        tabwidget_with_tb.layout().addWidget(self.tabwidget_tb)

        # Editor area: tabs + Find/Replace bar stacked vertically.
        editor_column = QW.QWidget(self)
        editor_column_layout = QW.QVBoxLayout(editor_column)
        editor_column_layout.setContentsMargins(0, 0, 0, 0)
        editor_column_layout.addWidget(self.tabwidget)
        from datalab.widgets.findreplace import FindReplaceBar

        self.find_bar = FindReplaceBar(
            lambda: self.get_macro().editor if self.get_macro() else None,
            shortcut_parent=self,
            parent=editor_column,
        )
        editor_column_layout.addWidget(self.find_bar)
        tabwidget_with_tb.layout().addWidget(editor_column)

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
        # Set initial sizes: give more space to editor (70%) than console (30%)
        # This ensures proper layout on first open
        total_height = 600  # Default reasonable height
        self.setSizes([int(total_height * 0.7), int(total_height * 0.3)])

        self.run_action = None
        self.stop_action = None
        self.obj_actions: list[QW.QAction] = []  # Object-dependent actions
        self.status_label: QW.QLabel | None = None
        self.__macros: list[Macro] = []
        self._active_tab_restored = False

        self.setup_actions()

        # Restore splitter state and trigger recovery prompt after the event
        # loop is up so QMessageBox is properly parented and modal.
        self._restore_splitter_state()
        QC.QTimer.singleShot(0, self._check_recovery)

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

    def deserialize_from_hdf5(
        self, reader: NativeH5Reader, reset_all: bool = False
    ) -> None:
        """Deserialize whole panel from a HDF5 file

        Args:
            reader: HDF5 reader
            reset_all: If True, preserve original UUIDs (workspace reload).
                      If False, regenerate UUIDs (importing objects).
        """
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                #  Contrary to signal or image panels, macros are not stored
                #  in a group but directly in the root of the HDF5 file
                obj = self.deserialize_object_from_hdf5(reader, name, reset_all)
                self.add_object(obj)
        # Update untitled number counter to prevent duplicate names
        self.update_untitled_counter()

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
        # Attach Python autocompletion to the editor of this new macro.
        # Stored on the macro itself to keep it alive for the editor's lifetime.
        obj.completer = PythonCompleter(obj.editor)  # type: ignore[attr-defined]
        self.SIG_OBJECT_ADDED.emit()
        self.tabwidget.set_current_number(number)
        # Try to restore the previously-active tab once its macro is loaded.
        if not self._active_tab_restored:
            self._restore_active_tab()
            try:
                stored_uid = Conf.macro.active_tab_uid.get(None)
            except Exception:  # pylint: disable=broad-except
                stored_uid = None
            if stored_uid:
                # If the target uid is present, mark restoration complete.
                if any(m.uid == stored_uid for m in self.__macros):
                    self._active_tab_restored = True

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        for macro in self.__macros:
            macro.clear_autosave()
        self.tabwidget.clear()
        self.__macros.clear()
        super().remove_all_objects()
        # Reset untitled counter when clearing all macros
        Macro.set_untitled_number(0)

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
        self.new_action = add_act
        ren_act = create_action(
            self,
            _("Rename macro"),
            icon=get_icon("libre-gui-pencil.svg"),
            triggered=self.rename_macro,
        )
        dup_act = create_action(
            self,
            _("Duplicate macro"),
            icon=get_icon("duplicate.svg"),
            triggered=self.duplicate_macro,
        )
        self.duplicate_action = dup_act
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
            icon=self.style().standardIcon(QW.QStyle.SP_TitleBarCloseButton),
            triggered=self.remove_macro,
        )
        recent_act = create_action(
            self,
            _("Recent macros..."),
            icon=get_icon("history.svg"),
            triggered=self.show_recent_dialog,
        )
        self.recent_action = recent_act
        self.obj_actions += [
            self.run_action,
            self.stop_action,
            ren_act,
            dup_act,
            exp_act,
            rem_act,
        ]

        self.tabwidget.SIG_CONTEXT_MENU.connect(self.context_menu.popup)

        tabwidget_corner = QW.QToolBar(_("Macro editor toolbar"), self)
        self.context_menu.aboutToShow.connect(self.__update_actions)
        tabwidget_menu_btn = create_menu_button(self, self.context_menu)
        tabwidget_corner.addWidget(tabwidget_menu_btn)

        self.status_label = QW.QLabel("", self)
        self.status_label.setStyleSheet("padding: 0 6px;")
        self._update_status_label(False)
        tabwidget_corner.addWidget(self.status_label)

        self.tabwidget.setCornerWidget(tabwidget_corner)

        # "New macro" tool buttons with template dropdown (one per toolbar)
        new_btn_corner = self._build_new_macro_button(add_act)
        new_btn_tb = self._build_new_macro_button(add_act)

        tabwidget_corner.addWidget(new_btn_corner)
        add_actions(tabwidget_corner, [self.run_action, self.stop_action])
        add_actions(
            self.tabwidget_tb,
            [self.run_action, self.stop_action, None],
        )
        self.tabwidget_tb.addWidget(new_btn_tb)
        add_actions(
            self.tabwidget_tb,
            [dup_act, ren_act, recent_act, imp_act, exp_act],
        )
        add_actions(
            self.context_menu,
            [
                self.run_action,
                self.stop_action,
                None,
                add_act,
                dup_act,
                ren_act,
                recent_act,
                imp_act,
                exp_act,
                None,
                rem_act,
            ],
        )

        self.__update_actions()

    def _build_new_macro_button(self, default_action: QW.QAction) -> QW.QToolButton:
        """Return a tool button whose default action is *default_action* and
        whose dropdown menu offers a blank macro plus the bundled templates."""
        button = QW.QToolButton(self)
        button.setDefaultAction(default_action)
        button.setPopupMode(QW.QToolButton.MenuButtonPopup)
        menu = QW.QMenu(button)
        blank_act = menu.addAction(_("Blank macro (empty)"))
        blank_act.setToolTip(_("Create a new empty macro"))
        blank_act.triggered.connect(self.add_blank_macro)
        menu.addSeparator()
        templates_header = menu.addAction(_("Templates"))
        templates_header.setEnabled(False)
        for template in list_templates():
            tpl_act = menu.addAction(template.title)
            tpl_act.setToolTip(template.description or template.title)
            tpl_act.triggered.connect(
                lambda checked=False, t=template: self.add_macro_from_template(t)
            )
        button.setMenu(menu)
        return button

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
            self._update_status_label(state)

    def _update_status_label(self, running: bool) -> None:
        """Refresh the running-status pill in the corner toolbar.

        Args:
            running: True if the current macro is running
        """
        if self.status_label is None:
            return
        if running:
            self.status_label.setText(_("● Macro running"))
            self.status_label.setStyleSheet(
                "padding: 0 6px; color: #2e7d32; font-weight: bold;"
            )
        else:
            self.status_label.setText(_("○ Idle"))
            self.status_label.setStyleSheet("padding: 0 6px; color: gray;")

    def _restore_splitter_state(self) -> None:
        """Restore splitter sizes from persisted configuration."""
        try:
            from qtpy.QtCore import QByteArray

            raw = Conf.macro.splitter_state.get(None)
            if raw:
                ba = QByteArray.fromBase64(raw.encode("ascii"))
                self.restoreState(ba)
        except Exception:  # pylint: disable=broad-except
            pass

    def _save_splitter_state(self) -> None:
        """Persist current splitter sizes."""
        try:
            raw = bytes(self.saveState().toBase64()).decode("ascii")
            Conf.macro.splitter_state.set(raw)
        except Exception:  # pylint: disable=broad-except
            pass

    def _persist_active_tab(self, _index: int = -1) -> None:
        """Persist the uid of the currently active macro tab."""
        macro = self.get_macro()
        if macro is None:
            return
        try:
            Conf.macro.active_tab_uid.set(macro.uid)
        except Exception:  # pylint: disable=broad-except
            pass

    def _restore_active_tab(self) -> None:
        """Restore the active tab from the persisted uid (best-effort)."""
        try:
            uid = Conf.macro.active_tab_uid.get(None)
        except Exception:  # pylint: disable=broad-except
            uid = None
        if not uid:
            return
        for index, macro in enumerate(self.__macros, start=1):
            if macro.uid == uid:
                self.tabwidget.set_current_number(index)
                return

    def closeEvent(self, event) -> None:  # noqa: D401, N802
        """Persist splitter state on close."""
        self._save_splitter_state()
        super().closeEvent(event)

    def _check_recovery(self) -> None:
        """Offer to restore auto-saved macros from a previous session."""
        try:
            pending = macrorecovery.load_pending()
        except OSError:
            return
        if not pending:
            return
        # Filter out macros already in the workspace by uid
        known_uids = {m.uid for m in self.__macros}
        pending = {
            uid: entry for uid, entry in pending.items() if uid not in known_uids
        }
        if not pending:
            return
        if execenv.unattended:
            macrorecovery.clear_pending()
            return
        choice = QW.QMessageBox.question(
            self,
            _("Restore unsaved macros"),
            _(
                "%d unsaved macro(s) were found from a previous session.\n"
                "Would you like to restore them?"
            )
            % len(pending),
            QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No,
        )
        if choice == QW.QMessageBox.StandardButton.Yes:
            for entry in pending.values():
                macro = self.add_macro_with_code(
                    entry.get("title", _("Recovered macro")),
                    entry.get("code", ""),
                    source="recovery",
                )
                # Preserve the original UID so subsequent autosaves overwrite
                # the same recovery entry rather than creating a new one.
                uid = entry.get("uid")
                if uid and macro is not None:
                    macro.uid = uid
        macrorecovery.clear_pending()

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

    def add_macro_with_code(
        self, title: str, code: str, source: str | None = None
    ) -> Macro:
        """Add a macro with a predefined title and code, without prompting.

        This helper is used by the AI assistant to programmatically inject
        a generated macro into the panel.

        Args:
            title: Macro title (displayed in the tab).
            code: Python source code of the macro.
            source: Optional origin marker (e.g. ``"import"``, ``"ai"``,
             ``"template:simple_macro"``) recorded in the recent cache.

        Returns:
            The newly created macro.
        """
        macro = self.create_object()
        macro.title = title
        macro.set_code(code)
        self.add_object(macro)
        try:
            recentmacros.record_recent(title, code, source=source)
        except OSError:
            pass
        return macro

    def add_blank_macro(self) -> Macro:
        """Add a new empty macro and return it."""
        macro = self.create_object()
        macro.set_code("")
        self.add_object(macro)
        return macro

    def add_macro_from_template(self, template: MacroTemplate) -> Macro:
        """Add a new macro initialised from *template*.

        Args:
            template: Macro template description.

        Returns:
            The newly created macro.
        """
        return self.add_macro_with_code(
            template.title, template.code, source=f"template:{template.name}"
        )

    def duplicate_macro(self, number_or_title: int | str | None = None) -> Macro | None:
        """Duplicate the given (or current) macro.

        Args:
            number_or_title: Number of the macro (starting at 1) or title.
             Defaults to None (current macro).

        Returns:
            The duplicated macro, or None if no source macro was found.
        """
        src = self.get_macro(number_or_title)
        if src is None:
            return None
        new_title = _("%s (copy)") % src.title
        return self.add_macro_with_code(new_title, src.get_code(), source="duplicate")

    def show_recent_dialog(self) -> None:
        """Open a dialog listing recent macros and allow reopening one."""
        dialog = _RecentMacrosDialog(self)
        dialog.exec_()

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
            with qt_try_loadsave_file(self.parentWidget(), filename, "save"):
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
            with qt_try_loadsave_file(self.parentWidget(), filename, "load"):
                Conf.main.base_dir.set(filename)
                macro = self.add_macro()
                macro.from_file(filename)
                try:
                    recentmacros.record_recent(
                        macro.title, macro.get_code(), source="import"
                    )
                except OSError:
                    pass
            # Update untitled number counter to prevent duplicate names
            self.update_untitled_counter()
            return self.get_number_from_macro(macro)
        return -1

    def update_untitled_counter(self) -> None:
        """Update the untitled counter based on existing macro titles

        This scans all macro titles to find the highest "macro_XX" number
        and updates the global counter to prevent duplicate names.
        """
        max_untitled = 0
        for macro in self.__macros:
            # Match titles like "macro_01", "macro_02", etc.
            match = re.match(r"macro_(\d+)", macro.title)
            if match:
                number = int(match.group(1))
                max_untitled = max(max_untitled, number)
        # Set the counter to the highest found number
        Macro.set_untitled_number(max_untitled)

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
            macro = self.get_macro(number_or_title)
            if macro is not None:
                macro.clear_autosave()
            self.tabwidget.remove_tab(number_or_title)
            self.__macros.pop(number_or_title - 1)
            self.SIG_OBJECT_REMOVED.emit()
