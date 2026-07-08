# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History tree widget used by the History panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.gui.panel.history.chainmodel import build_session_chains
from datalab.history import HistoryAction, HistorySession
from datalab.widgets.historydescription import CollapsibleDescriptionWidget

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.history.panel import HistoryPanel


class HistoryTree(QW.QTreeWidget):
    """Tree widget for the history panel"""

    DESCRIPTION_COLUMN = 2
    COMPATIBILITY_ROLE = QC.Qt.UserRole + 1
    SESSION_NUMBER_ROLE = QC.Qt.UserRole + 2
    ITEM_KIND_ROLE = QC.Qt.UserRole + 3
    ITEM_SESSION = "session"
    ITEM_CHAIN = "chain"
    ITEM_ACTION = "action"

    def __init__(self, parent: QW.QWidget) -> None:
        """Create a new history tree widget"""
        super().__init__(parent)
        self._panel: HistoryPanel = parent
        self.setHeaderLabels([_("Title"), _("Date and time"), _("Description")])
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)
        self.setSelectionMode(QW.QAbstractItemView.ContiguousSelection)
        self.setUniformRowHeights(False)
        header = self.header()
        header.setSectionResizeMode(self.DESCRIPTION_COLUMN, QW.QHeaderView.Stretch)
        # Per-action expanded state, preserved across repopulate (delete/replay).
        self.__expanded_state: dict[str, bool] = {}
        # Session numbers currently flagged as active recording sessions,
        # mapped to their panel id ('signal'/'image'). Used to highlight the
        # active session(s) and survive tree repopulation.
        self.__active_session_numbers: dict[int, str] = {}

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
        item.setData(0, cls.ITEM_KIND_ROLE, cls.ITEM_ACTION)
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
        actions_by_uuid = {
            action.uuid: action
            for session in history_sessions
            for action in session.actions
        }
        iterator = QW.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.data(0, self.ITEM_KIND_ROLE) == self.ITEM_ACTION:
                uuid = item.data(0, QC.Qt.UserRole)
                action = actions_by_uuid.get(uuid)
                # The tree can transiently reference an action that was just
                # removed from the model (e.g. mid-cascade during
                # reconnect_chain_after_removal, before the final repopulate).
                # Skip such stale items instead of crashing.
                if action is None:
                    iterator += 1
                    continue
                compatible = action.is_current_state_compatible(
                    mainwindow, restore_selection=True
                )
                item.setData(0, self.COMPATIBILITY_ROLE, compatible)
                brush = default_brush if compatible else disabled_brush
                icon = get_icon("apply.svg") if compatible else get_icon("delete.svg")
                item.setIcon(0, icon)
                for col in range(self.columnCount()):
                    item.setForeground(col, brush)
                    item.setToolTip(
                        col, compatible_tip if compatible else incompatible_tip
                    )
            iterator += 1

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
            ritem.setData(0, self.SESSION_NUMBER_ROLE, session.number)
            ritem.setData(0, self.ITEM_KIND_ROLE, self.ITEM_SESSION)
            self.addTopLevelItem(ritem)
            self._build_session_children(ritem, session)
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)
        self.__apply_active_highlight()

    def _build_session_children(
        self, session_item: QW.QTreeWidgetItem, session: HistorySession
    ) -> None:
        """(Re)build the chain/action subtree under ``session_item``.

        Args:
            session_item: Top-level tree item for ``session``.
            session: History session whose actions are grouped into chains.
        """
        session_item.takeChildren()
        chains = build_session_chains(self._panel, session)
        for chain in chains:
            chain_item = QW.QTreeWidgetItem([chain.root.title, ""])
            chain_item.setData(0, self.ITEM_KIND_ROLE, self.ITEM_CHAIN)
            chain_item.setData(0, self.COMPATIBILITY_ROLE, True)
            font = chain_item.font(0)
            font.setBold(True)
            chain_item.setFont(0, font)
            chain_item.setToolTip(
                0, _("Processing chain \u2014 %d step(s)") % len(chain.actions)
            )
            session_item.addChild(chain_item)
            for action in chain.actions:
                child = self.action_to_tree_item(action)
                chain_item.addChild(child)
                self.install_description_widget(child, action)

    def set_active_sessions(self, active_session_numbers: dict[int, str]) -> None:
        """Flag the active recording session(s) by session number and panel.

        Args:
            active_session_numbers: Mapping ``{session.number: panel_str}`` of
                the active recording session for each panel.
        """
        self.__active_session_numbers = dict(active_session_numbers)
        self.__apply_active_highlight()

    def __apply_active_highlight(self) -> None:
        """Bold + tint the top-level items of the active recording sessions."""
        hl = self.palette().color(QG.QPalette.Highlight)
        hl.setAlpha(60)
        active_brush = QG.QBrush(hl)
        normal_brush = QG.QBrush()
        tip = _("Active recording session ({panel}).")
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            number = item.data(0, self.SESSION_NUMBER_ROLE)
            panel_str = self.__active_session_numbers.get(number)
            is_active = panel_str is not None
            font = item.font(0)
            font.setBold(is_active)
            for col in (0, 1):
                item.setFont(col, font)
                item.setBackground(col, active_brush if is_active else normal_brush)
                item.setToolTip(col, tip.format(panel=panel_str) if is_active else "")

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
        if session_index is None:
            session_index = self.topLevelItemCount() - 1
        ritem = self.topLevelItem(session_index)
        if ritem is None:
            return
        session = self._panel.history_sessions[session_index]
        self._build_session_children(ritem, session)
        ritem.setExpanded(True)

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
            elif item.data(0, self.ITEM_KIND_ROLE) == self.ITEM_ACTION:
                uuid = item.data(0, QC.Qt.UserRole)
                try:
                    selected.append(self.get_action_from_uuid(uuid, history_sessions))
                except ValueError:
                    continue
            # chain-header items are containers only: ignored here.
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
            if item.parent() is not None and (
                item.data(0, self.ITEM_KIND_ROLE) == self.ITEM_ACTION
            ):
                uuid = item.data(0, QC.Qt.UserRole)
                try:
                    selected.append(self.get_action_from_uuid(uuid, history_sessions))
                except ValueError:
                    continue
        return selected
