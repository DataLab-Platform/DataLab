# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive (dialog-driven) replay helpers for the History panel."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import guidata.dataset as gds
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import recompute as hrec
from datalab.history import HistoryAction, HistorySession

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel

_logger = logging.getLogger(__name__)


def replay_restore_actions(
    panel: HistoryPanel, replay: bool = True, restore_selection: bool = True
) -> None:
    """Replay and/or restore selection for the selected actions."""
    panel.refresh_compatibility_items()
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        if not panel.history_sessions:
            return
        selected = [panel.history_sessions[-1]]
    edit_actions: list[HistoryAction] = []
    for session_or_action in selected:
        if isinstance(session_or_action, HistoryAction) and session_or_action.is_stale:
            hrec.recompute_cascade(panel, session_or_action)
            continue
        if not session_or_action.is_current_state_compatible(
            panel.mainwindow, restore_selection=restore_selection
        ):
            if not execenv.unattended:
                QW.QMessageBox.critical(
                    panel.mainwindow,
                    _("Error"),
                    _("The current workspace state is not compatible with the action."),
                )
            return
        if replay:
            if panel.edit_mode and isinstance(session_or_action, HistoryAction):
                # Defer: edit only the selected actions, no automatic cascade
                edit_actions.append(session_or_action)
            else:
                # Scope decision: clicking a session in edit mode now replays it
                # WITH parameter dialogs (view-only session replay disabled).
                with panel.replaying(), panel.output_suppressed():
                    session_or_action.replay(
                        panel.mainwindow,
                        restore_selection=restore_selection,
                        edit=panel.edit_mode,
                    )
        elif restore_selection:
            if panel.edit_mode or panel.has_any_pending_edits():
                restore_action_params(panel, session_or_action)
            else:
                session_or_action.restore(panel.mainwindow)
    if edit_actions:
        edit_mode_replay_actions(panel, edit_actions)


def prompt_edit_action_params(
    panel: HistoryPanel, action: HistoryAction
) -> bool | None:
    """Open the parameter dialog for *action* according to its pattern."""
    if (
        action.kind == HistoryAction.KIND_UI
        and action.method_name in HistoryAction.UI_CREATION_METHODS
    ):
        param = action.kwargs.get("param")
        if param is None:
            return None
        edited = copy.deepcopy(param)
        if not edited.edit(parent=panel.mainwindow):
            return False
        action.snapshot_kwargs()
        action.kwargs["param"] = edited
        return True
    pattern = action.pattern
    if pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
        param = action.kwargs.get("param")
        if param is None:
            return None
        edited = copy.deepcopy(param)
        dialog_target: gds.DataSet | gds.DataSetGroup = edited
        new_kwargs = {"param": edited}
    elif pattern == "1_to_n":
        params = action.kwargs.get("params") or []
        if not params:
            return None
        edited_params = [copy.deepcopy(p) for p in params]
        dialog_target = gds.DataSetGroup(edited_params, title=_("Parameters"))
        new_kwargs = {"params": edited_params}
    else:
        return None
    if not dialog_target.edit(parent=panel.mainwindow):
        return False
    action.snapshot_kwargs()
    action.kwargs.update(new_kwargs)
    return True


def edit_mode_replay_actions(panel: HistoryPanel, actions: list[HistoryAction]) -> None:
    """Edit and recompute only the selected actions, in session order.

    Each selected action gets exactly one parameter dialog; non-selected
    downstream actions are left untouched (no automatic cascade). A
    re-entrance guard prevents nested prompt loops.
    """
    if panel.edit_replay_in_progress:
        return
    # Deduplicate and sort the selected actions in their session order
    ordered = order_selected_actions(panel, actions)
    if not ordered:
        return
    panel.edit_replay_in_progress = True
    try:
        edited_actions: list[HistoryAction] = []
        recomputable: list[HistoryAction] = []
        for action in ordered:
            is_creation = (
                action.kind == HistoryAction.KIND_UI
                and action.method_name in HistoryAction.UI_CREATION_METHODS
            )
            is_compute = (
                action.kind == HistoryAction.KIND_COMPUTE and action.pattern is not None
            )
            if not is_creation and not is_compute:
                with panel.replaying(), panel.output_suppressed():
                    action.replay(panel.mainwindow, restore_selection=True, edit=True)
                continue
            result = prompt_edit_action_params(panel, action)
            if result is False:
                for done in edited_actions:
                    done.restore_kwargs()
                    panel.tree.refresh_action_item(done)
                return
            if result is True:
                edited_actions.append(action)
            recomputable.append(action)

        for action in edited_actions:
            panel.tree.refresh_action_item(action)
        for action in recomputable:
            hrec.recompute_action_in_place(panel, action)
            panel.tree.refresh_action_item(action)
        if edited_actions:
            hrec.recompute_cascade(panel, edited_actions[0])
        QW.QApplication.processEvents()
    finally:
        panel.edit_replay_in_progress = False


def order_selected_actions(
    panel: HistoryPanel, actions: list[HistoryAction]
) -> list[HistoryAction]:
    """Deduplicate ``actions`` and sort them by (session, position) order."""
    rank: dict[str, int] = {}
    pos = 0
    for session in panel.history_sessions:
        for action in session.actions:
            rank[action.uuid] = pos
            pos += 1
    seen: set[str] = set()
    unique: list[HistoryAction] = []
    for action in actions:
        if action.uuid in seen:
            continue
        seen.add(action.uuid)
        unique.append(action)
    unique.sort(key=lambda a: rank.get(a.uuid, 0))
    return unique


def restore_action_params(
    panel: HistoryPanel, item: HistoryAction | HistorySession
) -> None:
    """Restore original kwargs from snapshot and recompute in-place."""
    actions: list[HistoryAction]
    if isinstance(item, HistorySession):
        actions = [
            a
            for a in item.actions
            if a.kind == HistoryAction.KIND_COMPUTE
            or (
                a.kind == HistoryAction.KIND_UI
                and a.method_name in HistoryAction.UI_CREATION_METHODS
            )
        ]
    else:
        actions = [item]
    for action in actions:
        if not action.has_pending_edits:
            continue
        action.restore_kwargs()
        panel.tree.refresh_action_item(action)
        hrec.recompute_action_in_place(panel, action)
        hrec.recompute_cascade(panel, action)
    panel.update_actions_state()
