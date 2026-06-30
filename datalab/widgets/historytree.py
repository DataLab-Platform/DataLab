# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History tree widget used by the History panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.history import HistoryAction, HistorySession
from datalab.widgets.historydescription import CollapsibleDescriptionWidget

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


class HistoryTree(QW.QTreeWidget):
    """Tree widget for the history panel"""

    DESCRIPTION_COLUMN = 2
    COMPATIBILITY_ROLE = QC.Qt.UserRole + 1

    def __init__(self, parent: QW.QWidget) -> None:
        """Create a new history tree widget"""
        super().__init__(parent)
        self.setHeaderLabels([_("Title"), _("Date and time"), _("Description")])
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)
        self.setSelectionMode(QW.QAbstractItemView.ContiguousSelection)
        self.setUniformRowHeights(False)
        header = self.header()
        header.setSectionResizeMode(self.DESCRIPTION_COLUMN, QW.QHeaderView.Stretch)
        # Per-action expanded state, preserved across repopulate (delete/replay).
        self.__expanded_state: dict[str, bool] = {}

    def on_description_toggled(self, uuid: str, expanded: bool) -> None:
        """Remember the expanded state of a description cell."""
        self.__expanded_state[uuid] = expanded
        # Force the tree to recompute row heights now that the label content
        # has changed.
        self.scheduleDelayedItemsLayout()

    def install_description_widget(
        self, item: QW.QTreeWidgetItem, action: HistoryAction
    ) -> None:
        """Attach the collapsible description widget to ``item`` (column 2).

        The item must already be inserted in the tree before calling this.
        """
        expanded = self.__expanded_state.get(action.uuid, False)
        widget = CollapsibleDescriptionWidget(
            action.description_summary,
            action.description_html,
            expanded=expanded,
            parent=self,
        )
        widget.toggled.connect(
            lambda checked, uuid=action.uuid: self.on_description_toggled(uuid, checked)
        )
        # Clear any text the item may carry for that column to avoid double
        # rendering behind the widget.
        item.setText(self.DESCRIPTION_COLUMN, "")
        self.setItemWidget(item, self.DESCRIPTION_COLUMN, widget)

    @classmethod
    def action_to_tree_item(cls, action: HistoryAction) -> QW.QTreeWidgetItem:
        """Convert an action to a tree item

        Args:
            action: Action to convert

        Returns:
            QW.QTreeWidgetItem: Tree item
        """
        # Description column is left empty: a CollapsibleDescriptionWidget is
        # installed by ``HistoryTree`` once the item is inserted in the tree.
        item = QW.QTreeWidgetItem([action.title, action.dtstr, ""])
        item.setData(0, QC.Qt.UserRole, action.uuid)
        item.setData(0, cls.COMPATIBILITY_ROLE, True)
        return item

    def update_compatibility_states(
        self, history_sessions: list[HistorySession], mainwindow: DLMainWindow
    ) -> None:
        """Update action item visual state from workspace compatibility."""
        default_brush = QG.QBrush()
        disabled_brush = QG.QBrush(
            self.palette().color(QG.QPalette.Disabled, QG.QPalette.Text)
        )
        compatible_tip = _("Action is compatible with the current workspace state.")
        incompatible_tip = _(
            "Action is not compatible with the current workspace state."
        )
        for i in range(self.topLevelItemCount()):
            session_item = self.topLevelItem(i)
            for j in range(session_item.childCount()):
                child = session_item.child(j)
                uuid = child.data(0, QC.Qt.UserRole)
                action = self.get_action_from_uuid(uuid, history_sessions)
                compatible = action.is_current_state_compatible(
                    mainwindow, restore_selection=True
                )
                child.setData(0, self.COMPATIBILITY_ROLE, compatible)
                brush = default_brush if compatible else disabled_brush
                icon = get_icon("apply.svg") if compatible else get_icon("delete.svg")
                child.setIcon(0, icon)
                for col in range(self.columnCount()):
                    child.setForeground(col, brush)
                    child.setToolTip(
                        col, compatible_tip if compatible else incompatible_tip
                    )

    def forget_orphan_expanded_states(
        self, history_sessions: list[HistorySession]
    ) -> None:
        """Drop expanded-state entries for actions that no longer exist."""
        live_uuids = {
            action.uuid for session in history_sessions for action in session.actions
        }
        self.__expanded_state = {
            uuid: state
            for uuid, state in self.__expanded_state.items()
            if uuid in live_uuids
        }

    def populate_tree(self, history_sessions: list[HistorySession]) -> None:
        """Populate the history tree widget

        Args:
            history_sessions: List of history sessions
        """
        self.forget_orphan_expanded_states(history_sessions)
        self.clear()
        for session in history_sessions:
            ritem = QW.QTreeWidgetItem([session.title, session.dtstr])
            ritem.setData(0, self.COMPATIBILITY_ROLE, True)
            self.addTopLevelItem(ritem)
            for action in session.actions:
                child = self.action_to_tree_item(action)
                ritem.addChild(child)
                self.install_description_widget(child, action)
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def rearrange_tree(self) -> None:
        """Rearrange the history tree widget"""
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def add_action_to_tree(
        self, action: HistoryAction, session_index: int | None = None
    ) -> None:
        """Add an action under the session item at ``session_index``.

        Args:
            action: Action to add.
            session_index: Top-level session item index. Defaults to the last
                session (backward-compatible).
        """
        item = self.action_to_tree_item(action)
        if session_index is None:
            session_index = self.topLevelItemCount() - 1
        ritem = self.topLevelItem(session_index)
        if ritem is None:
            return
        ritem.addChild(item)
        self.install_description_widget(item, action)

    def refresh_action_item(self, action: HistoryAction) -> None:
        """Refresh the tree item corresponding to ``action``.

        Re-installs the description widget so it reflects the current
        ``action.kwargs`` (e.g. after the user edited a ``param`` via the
        Processing tab of the Signal/Image panel). Also applies a light
        orange background when ``action.is_stale`` is True, to signal that
        the action is currently being recomputed in a cascade.
        """
        target_uuid = action.uuid
        stale_brush = QG.QBrush(QG.QColor(255, 220, 150))  # light orange
        normal_brush = QG.QBrush()
        iterator = QW.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QC.Qt.UserRole) == target_uuid:
                # Remove and re-install the collapsible description widget so
                # it reflects the mutated ``action.kwargs``.
                self.removeItemWidget(item, self.DESCRIPTION_COLUMN)
                self.install_description_widget(item, action)
                brush = stale_brush if action.is_stale else normal_brush
                for col in range(self.columnCount()):
                    item.setBackground(col, brush)
                self.scheduleDelayedItemsLayout()
                return
            iterator += 1

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
