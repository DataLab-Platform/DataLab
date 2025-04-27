# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. History panel (see parent package :mod:`cdl.core.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import functools
import inspect
import os
from typing import TYPE_CHECKING, Any, Callable, Generator
from uuid import uuid4

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.config import _
from cdl.core.gui import ObjItf
from cdl.core.gui.panel.base import AbstractPanel

if TYPE_CHECKING:
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.gui.panel.base import BaseDataPanel
    from cdl.core.gui.processor.base import BaseProcessor
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer


def get_datetime_str() -> str:
    """Return current date and time as a string"""
    return QC.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")


def add_to_history(kwargs_names: list[str] = [], title: str | None = None):
    """Method decorator to add the method call to the history panel

    Args:
        kwargs_names: List of keyword arguments to add to the history action.
         Defaults to [].
        title: Title of the history action. Defaults to None.
    """

    def add_to_history_decorator(func):
        """Decorator function"""

        @functools.wraps(func)
        def method_wrapper(*args, **kwargs):
            """Decorator wrapper function"""
            self: BaseDataPanel | BaseProcessor = args[0]
            history: HistoryPanel = self.mainwindow.historypanel
            histkwargs = {k: kwargs[k] for k in kwargs_names if k in kwargs}
            history.add_entry(
                kwargs.get("title", title),
                kwargs.get("save_state", True),
                func,
                *args,
                **histkwargs,
            )
            return func(*args, **kwargs)

        return method_wrapper

    return add_to_history_decorator


class HistoryAction(ObjItf):
    """Object representing an action in the history panel.

    An action is basically a function that can be called in the same conditions as
    when it was added to the history panel. Replay an action is done by calling the
    function with the same parameters.

    Args:
        title: Title of the history action
        func: Function to call
        args: Function arguments
        kwargs: Function keyword arguments
        state: State of the workspace before the action
    """

    FUNC_EDIT_MODE = "edit"  # Name of the function parameter to enable edit mode

    def __init__(
        self,
        title: str | None = None,
        func: Callable | None = None,
        args: tuple | None = None,
        kwargs: dict[str, Any] | None = None,
        state: WorkspaceState | None = None,
    ) -> None:
        """Create a new action"""
        super().__init__()
        self.__title = "" if title is None else title
        if func is None:

            def default_func():
                """Default function"""
                # This function is used to create a default action when the
                # function is not provided.
                pass

            func = default_func
        else:
            if not callable(func):
                raise TypeError("func must be callable")
            self.func = func
        self.args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.state = WorkspaceState() if state is None else state
        self.dtstr: str = get_datetime_str()
        self.uuid: str = str(uuid4())

    def regenerate_uuid(self):
        """Regenerate UUID

        This method is used to regenerate UUID after loading the object from a file.
        This is required to avoid UUID conflicts when loading objects from file
        without clearing the workspace first.
        """
        # No UUID to regenerate for history action

    @property
    def title(self) -> str:
        """Return object title"""
        return self.__title

    @property
    def description(self) -> str:
        """Return object description (string representing function parameters)"""
        desc = ""
        no_parameters = True
        for kwname in self.kwargs:
            if kwname.endswith("param"):
                param = self.kwargs[kwname]
                # Note: `param` can't be None because we removed None values from kwargs
                if desc:
                    desc += os.linesep
                desc += str(param)
                no_parameters = False
        if desc or no_parameters:
            return desc
        if len(self.args) >= 2 and isinstance(self.args[1], Callable):
            doc: str = self.args[1].__doc__
            return doc.splitlines()[0] if doc else ""
        return self.func.__doc__ or ""

    def is_current_state_compatible(
        self, mainwindow: CDLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the selection before checking the state

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        return self.state.is_current_state_compatible(mainwindow, restore_selection)

    def restore(self, mainwindow: CDLMainWindow) -> None:
        """Restore the associated workspace state

        Args:
            mainwindow: DataLab's main window
        """
        self.state.restore(mainwindow)

    def replay(
        self, mainwindow: CDLMainWindow, restore_selection: bool, edit: bool
    ) -> None:
        """Replay the action

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the workspace selection before replaying
            edit: if True, always open the dialog boxes to edit parameters, if False,
             use the parameters passed when creating the action
        """
        if restore_selection:
            self.state.restore(mainwindow)
        sig = inspect.signature(self.func)
        if self.FUNC_EDIT_MODE in sig.parameters:
            self.kwargs[self.FUNC_EDIT_MODE] = edit
        self.func(*self.args, **self.kwargs)

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this action

        Args:
            writer: Writer
        """
        with writer.group("func"):
            writer.write(self.func)
        with writer.group("args"):
            writer.write(self.args)
        with writer.group("kwargs"):
            writer.write_dict(self.kwargs)
        with writer.group("state"):
            self.state.serialize(writer)
        with writer.group("dtstr"):
            writer.write(self.dtstr)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this action

        Args:
            reader: Reader
        """
        with reader.group("func"):
            self.func = reader.read_any()
        with reader.group("args"):
            self.args = reader.read_any()
        with reader.group("kwargs"):
            self.kwargs = reader.read_dict()
        with reader.group("state"):
            self.state.deserialize(reader)
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()


class WorkspaceState:
    """Object representing the workspace state at a given time.

    The workspace state is the state of the workspace at a given time. It contains
    the list of objects in the workspace and the selection of objects. Instead of
    storing the objects themselves, the workspace state only stores what is relevant
    to the history panel, i.e. the object data shape and the object title (even
    if the latter is purely informative).
    """

    def __init__(self) -> None:
        """Create a new workspace state"""
        # The selection is stored as a dictionary where the key is the panel name
        # and the value is the list of selected object numbers (1 to N).
        self.selection: dict[str, list[int]] = {}
        # The states are stored as a dictionary where the key is the panel name
        # and the value is the list of states (str) of the objects in the panel. The
        # state is a string containing the object data shape (for now, only the shape,
        # but we could add more information if needed). The idea is that two objects
        # have the same state, we can apply the same action (processing, operation, ...)
        # to both objects.
        self.states: dict[str, list[str]] = {}
        # The titles are stored as a dictionary where the key is the panel name and the
        # value is the list of titles of the objects in the panel. The title is only
        # informative and is not used to determine if two objects have the same state.
        self.titles: dict[str, list[str]] = {}

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this workspace state

        Args:
            writer: Writer
        """
        with writer.group("selection"):
            writer.write_dict(self.selection)
        with writer.group("states"):
            writer.write_dict(self.states)
        with writer.group("titles"):
            writer.write_dict(self.titles)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this workspace state

        Args:
            reader: Reader
        """
        with reader.group("selection"):
            self.selection = reader.read_dict()
        with reader.group("states"):
            self.states = reader.read_dict()
        with reader.group("titles"):
            self.titles = reader.read_dict()

    def get_current_selection(self, mainwindow: CDLMainWindow) -> dict[str, list[int]]:
        """Get the current selection in the workspace

        Args:
            mainwindow: DataLab's main window

        Returns:
            dict[str, list[int]]: Current selection in the workspace
        """
        selection: dict[str, list[int]] = {}
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            selection[panel.PANEL_STR] = [
                obj.number for obj in panel.objview.get_sel_objects(include_groups=True)
            ]
        return selection

    def save(self, mainwindow: CDLMainWindow) -> None:
        """Save the current workspace state

        Args:
            mainwindow: DataLab's main window
        """
        self.selection = self.get_current_selection(mainwindow)
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            selection = self.selection[panel.PANEL_STR]
            self.states[panel.PANEL_STR] = [
                str(obj.data.shape) for obj in panel.objmodel if obj.number in selection
            ]
            self.titles[panel.PANEL_STR] = [
                obj.title for obj in panel.objmodel if obj.number in selection
            ]

    def is_current_state_compatible(
        self, mainwindow: CDLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the selection before checking the state

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        # A compatible state is a state where the selected objects are the same as the
        # saved selected objects in terms of position in the list of objects and of
        # data shape (title is not relevant).
        # To check this, we have to try to restore the selection (without restoring it)
        # and compare the current selection with the saved selection in terms of
        # position in the list of objects and of data shape.
        if self.states == {}:
            return True
        current_states: dict[str, list[str]] = {}
        selection = self.selection
        if not restore_selection:
            selection = self.get_current_selection(mainwindow)
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            numbers = selection[panel.PANEL_STR]
            current_states[panel.PANEL_STR] = [
                str(obj.data.shape) for obj in panel.objmodel if obj.number in numbers
            ]
        return current_states == self.states

    def restore(self, mainwindow: CDLMainWindow) -> None:
        """Restore the workspace state

        Only the selection is restored, not the objects themselves because they are
        not stored in the workspace state. Before restoring the selection, we may
        check if the current workspace objects are compatible with the saved
        workspace state. If not, we raise a `ValueError`.

        Args:
            mainwindow: DataLab's main window

        Raises:
            ValueError: If the current workspace state is not compatible with the
             saved state
        """
        if self.selection == {}:
            return
        if not self.is_current_state_compatible(mainwindow, False):
            raise ValueError(
                "Current workspace state is not compatible with saved state"
            )
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            numbers = self.selection[panel.PANEL_STR]
            panel.objview.select_objects(numbers)


class HistorySession:
    """Object representing a history session, i.e. a list of actions.

    A history session is a list of actions that can be replayed in the same order
    as they were added to the history session. The history session can be saved to
    a file and loaded from a file.

    Args:
        title: Title of the history session
        number: Number of the history session
    """

    def __init__(self, title: str = "", number: int = 0) -> None:
        """Create a new history session"""
        prefix = _("Session")
        self.title = title if title else f"{prefix} {number:03d}"
        self.number = number
        self.dtstr: str = get_datetime_str()
        self.actions: list[HistoryAction] = []

    def add_action(self, action: HistoryAction) -> None:
        """Add an action to the history session

        Args:
            action: Action to add
        """
        self.actions.append(action)

    def is_current_state_compatible(
        self, mainwindow: CDLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the selection before checking the state

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        if self.actions:
            return self.actions[0].is_current_state_compatible(
                mainwindow, restore_selection
            )
        return True

    def restore(self, mainwindow: CDLMainWindow) -> None:
        """Restore the state of the workspace associated to the first action of session

        Args:
            mainwindow: DataLab's main window
        """
        if self.actions:
            self.actions[0].restore(mainwindow)

    def replay(
        self, mainwindow: CDLMainWindow, restore_selection: bool, edit: bool
    ) -> None:
        """Replay the history session

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the workspace selection before replaying
            edit: if True, always open the dialog boxes to edit parameters, if False,
             use the parameters passed when creating the action
        """
        for action in self.actions[:]:
            action.replay(mainwindow, restore_selection=restore_selection, edit=edit)

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this history session

        Args:
            writer: Writer
        """
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("number"):
            writer.write(self.number)
        with writer.group("dtstr"):
            writer.write(self.dtstr)
        writer.write_object_list(self.actions, "actions")

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this history session

        Args:
            reader: Reader
        """
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("number"):
            self.number = reader.read_any()
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()
        self.actions = reader.read_object_list("actions", HistoryAction)

    def remove_action(self, action: HistoryAction) -> None:
        """Remove an action from the history session

        This implies removing all subsequent actions. If action is not found, this
        fails silently.

        Args:
            action: Action to remove
        """
        if action in self.actions:
            index = self.actions.index(action)
            self.actions = self.actions[:index]


class HistoryTree(QW.QTreeWidget):
    """Tree widget for the history panel"""

    def __init__(self, parent: QW.QWidget) -> None:
        """Create a new history tree widget"""
        super().__init__(parent)
        self.setHeaderLabels([_("Title"), _("Date and time"), _("Description")])
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)
        self.setSelectionMode(QW.QAbstractItemView.ContiguousSelection)

    @staticmethod
    def action_to_tree_item(action: HistoryAction) -> QW.QTreeWidgetItem:
        """Convert an action to a tree item

        Args:
            action: Action to convert

        Returns:
            QW.QTreeWidgetItem: Tree item
        """
        item = QW.QTreeWidgetItem([action.title, action.dtstr, action.description])
        item.setData(0, QC.Qt.UserRole, action.uuid)
        return item

    def populate_tree(self, history_sessions: list[HistorySession]) -> None:
        """Populate the history tree widget

        Args:
            history_sessions: List of history sessions
        """
        self.clear()
        for session in history_sessions:
            ritem = QW.QTreeWidgetItem([session.title, session.dtstr])
            self.addTopLevelItem(ritem)
            for action in session.actions:
                ritem.addChild(self.action_to_tree_item(action))
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def rearrange_tree(self) -> None:
        """Rearrange the history tree widget"""
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def add_action_to_tree(self, action: HistoryAction) -> None:
        """Add an action to the history tree widget

        Args:
            action: Action to add
        """
        item = self.action_to_tree_item(action)
        ritem = self.topLevelItem(self.topLevelItemCount() - 1)
        ritem.addChild(item)

    def get_action_from_uuid(
        self, uuid: str, history_sessions: list[HistorySession]
    ) -> HistoryAction:
        """Get the action from its UUID

        Args:
            uuid: Action UUID
            history_sessions: List of history sessions

        Returns:
            HistoryAction: Action
        """
        for session in history_sessions:
            for action in session.actions:
                if action.uuid == uuid:
                    return action
        raise ValueError("Action not found")

    def get_selected_actions_or_sessions(
        self, history_sessions: list[HistorySession]
    ) -> list[HistoryAction | HistorySession]:
        """Get the selected actions or sessions

        Args:
            history_sessions: List of history sessions

        Returns:
            list[HistoryAction | HistorySession]: List of selected actions or sessions
        """
        selected: list[HistoryAction | HistorySession] = []
        for item in self.selectedItems():
            if item.parent() is None:
                index = self.indexOfTopLevelItem(item)
                selected.append(history_sessions[index])
            else:
                uuid = item.data(0, QC.Qt.UserRole)
                selected.append(self.get_action_from_uuid(uuid, history_sessions))
        return selected

    def get_selected_actions(
        self, history_sessions: list[HistorySession]
    ) -> list[HistoryAction]:
        """Get the selected actions

        Args:
            history_sessions: List of history sessions

        Returns:
            list[HistoryAction]: List of selected actions
        """
        selected: list[HistoryAction] = []
        for item in self.selectedItems():
            if item.parent() is not None:
                uuid = item.data(0, QC.Qt.UserRole)
                selected.append(self.get_action_from_uuid(uuid, history_sessions))
        return selected


class HistoryPanel(AbstractPanel, DockableWidgetMixin):
    """History panel"""

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("History panel")

    H5_PREFIX = "DataLab_His"

    SIG_OBJECT_MODIFIED = QC.Signal()

    FILE_FILTERS = f"{_('History files')} (*.cdlhist)"

    def __init__(self, parent: CDLMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.PANEL_STR)
        self.setWindowIcon(get_icon("history.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.__record_mode = False
        self.__edit_mode = False
        self.__menu_actions: list[QW.QAction] = self.__create_menu_actions()

        self.mainwindow = parent
        self.tree = HistoryTree(self)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.replay_restore_actions)

        toolbar = QW.QToolBar(self)
        add_actions(toolbar, self.__menu_actions)
        widget = QW.QWidget(self)
        layout = QW.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        self.addWidget(widget)

        self.__history_sessions: list[HistorySession] = []
        self.__session_increment = 0

    def __create_menu_actions(self) -> list[QW.QAction]:
        """Create menu actions for the history panel

        Returns:
            list[QW.QAction]: List of menu actions
        """
        edit_action = create_action(
            self,
            _("Edit mode"),
            toggled=self.toggle_edit_mode,
            icon=get_icon("edit_mode.svg"),
        )
        edit_action.setChecked(self.__edit_mode)
        record_action = create_action(
            self,
            _("Record mode"),
            toggled=self.toggle_record_mode,
            icon=get_icon("record.svg"),
        )
        record_action.setChecked(self.__record_mode)
        return [
            record_action,
            None,
            create_action(
                self,
                _("Replay"),
                lambda: self.replay_restore_actions(restore_selection=False),
                icon=get_icon("replay.svg"),
            ),
            create_action(
                self,
                _("Restore selection"),
                lambda: self.replay_restore_actions(
                    restore_selection=True, replay=False
                ),
                icon=get_icon("restore_selection.svg"),
            ),
            create_action(
                self,
                _("Restore selection and replay"),
                self.replay_restore_actions,
                icon=get_icon("restore_and_replay.svg"),
            ),
            edit_action,
            None,
            create_action(
                self,
                _("Delete"),
                self.delete_actions,
                icon=get_icon("delete.svg"),
            ),
        ]

    def toggle_edit_mode(self, checked: bool) -> None:
        """Toggle edit mode

        Args:
            checked: True if the edit mode is checked, False otherwise
        """
        self.__edit_mode = checked

    def toggle_record_mode(self, checked: bool) -> None:
        """Toggle record mode

        Args:
            checked: True if the record mode is checked, False otherwise
        """
        self.__record_mode = checked

    def show_context_menu(self, pos: QC.QPoint) -> None:
        """Show the context menu

        Args:
            pos: Position of the context menu
        """
        menu = QW.QMenu()
        add_actions(menu, self.__menu_actions)
        menu.exec_(self.tree.mapToGlobal(pos))

    def get_action_from_uuid(self, uuid: str) -> HistoryAction:
        """Get the action from its UUID

        Args:
            uuid: Action UUID

        Returns:
            HistoryAction: Action
        """
        for session in self.__history_sessions:
            for action in session.actions:
                if action.uuid == uuid:
                    return action
        raise ValueError("Action not found")

    def replay_restore_actions(
        self, replay: bool = True, restore_selection: bool = True
    ) -> None:
        """Replay and/or restore selection for the selected actions"""
        for session_or_action in self.tree.get_selected_actions_or_sessions(
            self.__history_sessions
        ):
            if not session_or_action.is_current_state_compatible(
                self.mainwindow, restore_selection=restore_selection
            ):
                QW.QMessageBox.critical(
                    self.mainwindow,
                    _("Error"),
                    _("The current workspace state is not compatible with the action."),
                )
                return
            if replay:
                session_or_action.replay(
                    self.mainwindow,
                    restore_selection=restore_selection,
                    edit=self.__edit_mode,
                )
            elif restore_selection:
                session_or_action.restore(self.mainwindow)

    def delete_actions(self) -> None:
        """Delete the selected actions"""
        # Ask for confirmation as this will delete the action and all subsequent actions
        reply = QW.QMessageBox.question(
            self.mainwindow,
            _("Delete actions"),
            _(
                "Do you really want to delete the selected action "
                "and all the next ones?"
            ),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if reply == QW.QMessageBox.Yes:
            for action in self.tree.get_selected_actions(self.__history_sessions):
                for session in self.__history_sessions:
                    if action in session.actions:
                        session.remove_action(action)
            self.tree.populate_tree(self.__history_sessions)

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file

        Args:
            writer: HDF5 writer
        """
        writer.write_object_list(self.__history_sessions, self.H5_PREFIX)

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file

        Args:
            reader: HDF5 reader
        """
        self.__history_sessions: list[HistorySession] = reader.read_object_list(
            self.H5_PREFIX, HistorySession
        )
        if self.__history_sessions:
            self.__session_increment = self.__history_sessions[-1].number
        self.tree.populate_tree(self.__history_sessions)

    def __len__(self) -> int:
        """Return number of objects"""
        return sum(len(session.actions) for session in self.__history_sessions)

    def __getitem__(self, nb: int) -> HistoryAction:
        """Return object from its number (1 to N)"""
        for session in self.__history_sessions:
            if nb <= len(session.actions):
                return session.actions[nb - 1]
            nb -= len(session.actions)
        raise IndexError("Index out of range")

    def __iter__(self) -> Generator[HistoryAction, None, None]:
        """Iterate over objects"""
        for session in self.__history_sessions:
            for action in session.actions:
                yield action

    def create_new_session(self) -> None:
        """Create a new history list"""
        self.__session_increment += 1
        session = HistorySession(number=self.__session_increment)
        self.__history_sessions.append(session)
        self.tree.populate_tree(self.__history_sessions)

    def add_entry(
        self,
        action_title: str,
        save_state: bool,
        func: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Add an entry to the current history list

        Args:
            action_title: Title of the history action
            save_state: If True, the current workspace state is saved before adding the
             action (this is the most common case). If False, the action is added
             without saving the state: this may be useful when the action is not
             related to the current workspace state (e.g. when creating a new object).
            func: Function to call
            args: Function arguments
            kwargs: Function keyword arguments

        .. warning::

            Action will **not** be added to history if *record mode* is disabled.
        """
        assert isinstance(action_title, str), "action_title must be a string"
        assert isinstance(save_state, bool), "save_state must be a boolean"
        assert callable(func), "func must be callable"
        if not self.__record_mode:
            return
        if save_state:
            state = WorkspaceState()
            state.save(self.mainwindow)
        else:
            state = None
        obj = HistoryAction(action_title, func, args, kwargs, state)
        self.add_object(obj)

    # ------ AbstractPanel interface ---------------------------------------------------
    def create_object(self) -> HistoryAction:
        """Create and return object"""
        return HistoryAction()

    def add_object(self, obj: HistoryAction) -> None:
        """Add object to panel"""
        if not self.__history_sessions:
            self.create_new_session()
        self.__history_sessions[-1].add_action(obj)
        self.tree.add_action_to_tree(obj)
        self.tree.rearrange_tree()

    def remove_all_objects(self):
        """Remove all objects"""
        super().remove_all_objects()
