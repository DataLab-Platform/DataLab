# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive (dialog-driven) replay helpers for the History panel."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import recompute as hrec
from datalab.history import HistoryAction, HistorySession
from datalab.history.core import copy_history_value

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel

_logger = logging.getLogger(__name__)


@dataclass
class ActionParamEdit:
    """Parameter dialog target and action kwargs to update after acceptance."""

    dialog_target: gds.DataSet | gds.DataSetGroup
    new_kwargs: dict[str, Any]


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
            if panel.runtime.execution.edit_mode and isinstance(
                session_or_action, HistoryAction
            ):
                # Defer: edit only the selected actions, no automatic cascade
                edit_actions.append(session_or_action)
            else:
                # Scope decision: clicking a session in edit mode now replays it
                # WITH parameter dialogs (view-only session replay disabled).
                with panel.replaying(), panel.output_suppressed():
                    session_or_action.replay(
                        panel.mainwindow,
                        restore_selection=restore_selection,
                        edit=panel.runtime.execution.edit_mode,
                    )
        elif restore_selection:
            if panel.runtime.execution.edit_mode or any(
                action.has_pending_edits
                for session in panel.history_sessions
                for action in session.actions
            ):
                restore_action_params(panel, session_or_action)
            else:
                session_or_action.restore(panel.mainwindow)
    if edit_actions:
        edit_mode_replay_actions(panel, edit_actions)


def prepare_action_param_edit(action: HistoryAction) -> ActionParamEdit | None:
    """Prepare the editable parameter copy for ``action``."""
    result = None
    if (
        action.kind == HistoryAction.KIND_UI
        and action.method_name in HistoryAction.UI_CREATION_METHODS
    ):
        param = action.kwargs.get("param")
        if param is not None:
            edited = copy.deepcopy(param)
            result = ActionParamEdit(edited, {"param": edited})
    elif action.pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
        param = action.kwargs.get("param")
        if param is not None:
            edited = copy.deepcopy(param)
            result = ActionParamEdit(edited, {"param": edited})
    elif action.pattern == "1_to_n":
        params = action.kwargs.get("params") or []
        if params:
            edited_params = [copy.deepcopy(p) for p in params]
            dialog_target = gds.DataSetGroup(edited_params, title=_("Parameters"))
            result = ActionParamEdit(dialog_target, {"params": edited_params})
    return result


def prompt_edit_action_params(
    panel: HistoryPanel, action: HistoryAction
) -> bool | None:
    """Open the parameter dialog for *action* according to its pattern."""
    edit = prepare_action_param_edit(action)
    if edit is None:
        return None
    if not edit.dialog_target.edit(parent=panel.mainwindow):
        return False
    action.snapshot_kwargs()
    action.kwargs.update(edit.new_kwargs)
    return True


def edit_mode_replay_actions(panel: HistoryPanel, actions: list[HistoryAction]) -> None:
    """Edit selected actions and recompute their affected branches once.

    Each selected action gets exactly one parameter dialog. Recomputable
    selected actions are always included, while accepted parameter edits also
    include all downstream dependent actions. The resulting global plan is
    deduplicated and executed in session order. A re-entrance guard prevents
    nested prompt loops.
    """
    # Deduplicate and sort the selected actions in their session order
    ordered = order_selected_actions(panel, actions)
    if not ordered:
        return
    with panel.runtime.execution.replaying_edits() as started:
        if not started:
            return
        entry_states = {
            action.uuid: (
                copy_history_value(action.kwargs),
                copy_history_value(action.saved_kwargs),
            )
            for action in ordered
        }
        edited_actions: list[HistoryAction] = []
        recomputable: list[HistoryAction] = []
        deferred_actions: list[HistoryAction] = []
        for action in ordered:
            is_creation = (
                action.kind == HistoryAction.KIND_UI
                and action.method_name in HistoryAction.UI_CREATION_METHODS
            )
            is_compute = (
                action.kind == HistoryAction.KIND_COMPUTE and action.pattern is not None
            )
            if not is_creation and not is_compute:
                deferred_actions.append(action)
                continue
            result = prompt_edit_action_params(panel, action)
            if result is False:
                for selected_action in ordered:
                    kwargs, saved_kwargs = entry_states[selected_action.uuid]
                    selected_action.kwargs = kwargs
                    selected_action.saved_kwargs = saved_kwargs
                    panel.tree.refresh_action_item(selected_action)
                return
            if result is True:
                edited_actions.append(action)
            recomputable.append(action)

        for action in edited_actions:
            panel.tree.refresh_action_item(action)
        planned = list(recomputable)
        for action in edited_actions:
            planned.extend(hchain.get_downstream_actions(panel, action))
        planned = order_selected_actions(panel, planned)
        execution_plan = order_selected_actions(panel, deferred_actions + planned)
        for action in planned:
            action.is_stale = True
            panel.tree.refresh_action_item(action)
        QW.QApplication.processEvents()
        blocked_outputs: set[str] = set()
        try:
            for action in execution_plan:
                if action in deferred_actions:
                    with panel.replaying(), panel.output_suppressed():
                        action.replay(
                            panel.mainwindow, restore_selection=True, edit=True
                        )
                    continue
                if hchain.action_consumes_any(action, blocked_outputs):
                    blocked_outputs.update(
                        panel.runtime.objects.action_output_uuids.get(action.uuid, [])
                    )
                    continue
                success = hrec.recompute_action_in_place(panel, action)
                action.is_stale = not success
                panel.tree.refresh_action_item(action)
                if not success:
                    blocked_outputs.update(
                        panel.runtime.objects.action_output_uuids.get(action.uuid, [])
                    )
        finally:
            hrec.flush_cascade_warnings(panel)
        QW.QApplication.processEvents()


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
        success = hrec.recompute_action_in_place(panel, action)
        action.is_stale = not success
        panel.tree.refresh_action_item(action)
        if not success:
            break
        if not isinstance(item, HistorySession):
            hrec.recompute_cascade(panel, action)
    panel.ui.update_actions_state()
