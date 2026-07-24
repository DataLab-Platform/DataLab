# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Action↔output chain helpers for the History panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history.chainmodel import (
    ReconnectionPlan,
    ReconnectionTarget,
    action_input_uuids,
    remap_processing_parameters,
)
from datalab.gui.processor.base import (
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.history import HistoryAction, HistorySession
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history.panel import HistoryPanel

_logger = logging.getLogger(__name__)


def find_parent_session(
    panel: HistoryPanel, action: HistoryAction
) -> HistorySession | None:
    """Return the session that contains ``action``, or None."""
    for session in panel.history_sessions:
        if action in session.actions:
            return session
    return None


def action_panel_target(action: HistoryAction) -> str | None:
    """Return the main-window data-panel attribute targeted by ``action``."""
    if action.kind == HistoryAction.KIND_UI:
        if action.method_name in HistoryAction.UI_CREATION_METHODS:
            return action.target
        return None
    return {"signal": "signalpanel", "image": "imagepanel"}.get(action.panel_str)


def resolve_panel_for_action(
    panel: HistoryPanel, action: HistoryAction
) -> BaseDataPanel | None:
    """Return the data panel targeted by ``action``, or ``None``."""
    panels = {
        "signalpanel": panel.mainwindow.signalpanel,
        "imagepanel": panel.mainwindow.imagepanel,
    }
    return panels.get(action_panel_target(action))


def find_output_object_uuid(
    panel: HistoryPanel, panel_data: BaseDataPanel, action: HistoryAction
) -> str | None:
    """Find the UUID of the output object produced by ``action`` in ``panel_data``.

    Primary path: consult the bijective ``action_output_uuids`` mapping.
    Fallback path: legacy heuristic on ``processing_parameters`` metadata.
    """
    registered = panel.runtime.objects.action_output_uuids.get(action.uuid)
    if registered:
        existing_ids = set(panel_data.objmodel.get_object_ids())
        for out_uuid in registered:
            if out_uuid in existing_ids:
                return out_uuid
    if action.func_name is None:
        return None
    recorded_uuids = set(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    if not recorded_uuids:
        return None
    for obj in panel_data.objmodel:
        pp = extract_processing_parameters(obj)
        if pp is None or pp.func_name != action.func_name:
            continue
        if pp.source_uuid is not None and pp.source_uuid in recorded_uuids:
            return get_uuid(obj)
        if pp.source_uuids is not None and recorded_uuids.intersection(pp.source_uuids):
            return get_uuid(obj)
    return None


def find_action_for_output(
    panel: HistoryPanel, output_uuid: str, func_name: str
) -> HistoryAction | None:
    """Find the :class:`HistoryAction` that produced ``output_uuid``."""
    if not panel.history_sessions:
        return None
    action_uuid = panel.runtime.objects.output_to_action.get(output_uuid)
    if action_uuid is not None:
        mapped = next(
            (
                action
                for session in panel.history_sessions
                for action in session.actions
                if action.uuid == action_uuid
            ),
            None,
        )
        if mapped is not None:
            return mapped if mapped.func_name == func_name else None
    panel_data: BaseDataPanel | None = None
    output_obj = None
    for p in (panel.mainwindow.signalpanel, panel.mainwindow.imagepanel):
        if p.objmodel.has_uuid(output_uuid):
            output_obj = p.objmodel[output_uuid]
            panel_data = p
            break
    if panel_data is None or output_obj is None:
        return None
    pp = extract_processing_parameters(output_obj)
    if pp is None or pp.func_name != func_name or pp.source_uuid is None:
        return None
    target_source_uuid = pp.source_uuid
    for current_session in reversed(panel.history_sessions):
        for action in reversed(current_session.actions):
            if action.kind != HistoryAction.KIND_COMPUTE:
                continue
            if action.func_name != func_name:
                continue
            if action.panel_str != panel_data.PANEL_STR_ID:
                continue
            captured = action.state.selection.get(panel_data.PANEL_STR_ID, [])
            if captured and captured[0] == target_source_uuid:
                return action
    return None


def find_creation_action_for_output(
    panel: HistoryPanel, output_uuid: str
) -> HistoryAction | None:
    """Find the creation (``new_object``) action that produced ``output_uuid``.

    Creation actions are ``KIND_UI`` entries without a ``func_name`` so the
    standard :func:`find_action_for_output` lookup cannot match them. The
    bijective ``output_to_action`` mapping is consulted first; if no mapping
    exists, a fallback scan looks for a creation action whose registered
    output UUIDs include ``output_uuid``.
    """
    if not panel.history_sessions:
        return None
    action_uuid = panel.runtime.objects.output_to_action.get(output_uuid)
    if action_uuid is not None:
        mapped = next(
            (
                action
                for session in panel.history_sessions
                for action in session.actions
                if action.uuid == action_uuid
            ),
            None,
        )
        if mapped is not None and mapped.kind == HistoryAction.KIND_UI:
            return mapped
    for session in reversed(panel.history_sessions):
        for action in reversed(session.actions):
            if (
                action.kind == HistoryAction.KIND_UI
                and action.method_name in HistoryAction.UI_CREATION_METHODS
                and output_uuid
                in panel.runtime.objects.action_output_uuids.get(action.uuid, [])
            ):
                return action
    return None


def find_analysis_action(
    panel: HistoryPanel, obj_uuid: str, func_name: str
) -> HistoryAction | None:
    """Find the 1-to-0 analysis action for ``obj_uuid`` with ``func_name``.

    Analysis operations (1-to-0) do not produce a new output object: they
    write their result to the input object's metadata. The matching action is
    therefore identified by its input UUID and function name.

    Args:
        panel: The history panel providing the sessions.
        obj_uuid: UUID of the analyzed object.
        func_name: Sigima analysis feature name.

    Returns:
        The matching :class:`HistoryAction`, or ``None`` if not found.
    """
    for session in reversed(panel.history_sessions):
        for action in reversed(session.actions):
            if action.kind != HistoryAction.KIND_COMPUTE:
                continue
            if action.func_name != func_name:
                continue
            if obj_uuid in action_input_uuids(action):
                return action
    return None


def get_session_of(panel: HistoryPanel, action: HistoryAction) -> HistorySession | None:
    """Return the session that contains ``action``, or None."""
    for session in panel.history_sessions:
        if action in session.actions:
            return session
    return None


def action_output_uuid(panel: HistoryPanel, action: HistoryAction) -> str | None:
    """Return the UUID of the object produced by ``action``, or ``None``."""
    panel_data = resolve_panel_for_action(panel, action)
    if panel_data is None:
        return None
    return find_output_object_uuid(panel, panel_data, action)


def action_consumes_any(action: HistoryAction, uuids: set[str]) -> bool:
    """Return True if ``action``'s input UUIDs intersect ``uuids``."""
    if action.kind != HistoryAction.KIND_COMPUTE:
        return False
    return bool(action_input_uuids(action) & uuids)


def get_downstream_actions(
    panel: HistoryPanel, action: HistoryAction
) -> list[HistoryAction]:
    """Return the actions of the current session that depend on ``action``."""
    if not panel.history_sessions:
        return []
    current = get_session_of(panel, action)
    if current is None:
        return []
    root_out = action_output_uuid(panel, action)
    if root_out is None:
        return []
    closure: set[str] = {root_out}
    downstream: list[HistoryAction] = []
    idx = current.actions.index(action)
    for candidate in current.actions[idx + 1 :]:
        if candidate.kind != HistoryAction.KIND_COMPUTE:
            continue
        if not action_consumes_any(candidate, closure):
            continue
        downstream.append(candidate)
        out_uuid = action_output_uuid(panel, candidate)
        if out_uuid is not None:
            closure.add(out_uuid)
    return downstream


def resolve_target_outputs(
    panel: HistoryPanel, panel_data: BaseDataPanel, action: HistoryAction
) -> tuple[list[str], list[str]]:
    """Return ``(existing, missing)`` UUIDs registered for ``action``."""
    registered = list(panel.runtime.objects.action_output_uuids.get(action.uuid, []))
    existing_ids = set(panel_data.objmodel.get_object_ids())
    existing: list[str] = [u for u in registered if u in existing_ids]
    missing: list[str] = [u for u in registered if u not in existing_ids]
    return existing, missing


def existing_input_uuids(panel_data: BaseDataPanel, action: HistoryAction) -> list[str]:
    """Return recorded input UUIDs that still exist in ``panel_data``."""
    recorded = action.state.selection.get(panel_data.PANEL_STR_ID, [])
    return [uuid for uuid in recorded if panel_data.objmodel.has_uuid(uuid)]


def prune_output_mapping(panel: HistoryPanel) -> None:
    """Drop entries of :attr:`output_to_action` whose object no longer exists."""
    panel.runtime.objects.prune_output_mapping()


def rewrite_action_source(
    action: HistoryAction,
    pstr: str,
    old_uuid: str,
    new_uuid: str,
) -> None:
    """Replace ``old_uuid`` with ``new_uuid`` in an action's recorded inputs."""
    sel = action.state.selection.get(pstr)
    if sel:
        action.state.selection[pstr] = [new_uuid if u == old_uuid else u for u in sel]
    obj2 = action.kwargs.get("obj2_uuids")
    if isinstance(obj2, str):
        if obj2 == old_uuid:
            action.kwargs["obj2_uuids"] = new_uuid
    elif obj2:
        action.kwargs["obj2_uuids"] = [new_uuid if u == old_uuid else u for u in obj2]


def remove_single_action(panel: HistoryPanel, action: HistoryAction) -> None:
    """Remove a single action from its session (splice, not truncate)."""
    for session in panel.history_sessions:
        if action in session.actions:
            session.actions.remove(action)
            panel.runtime.objects.remove_action_outputs(action)
            if not session.actions:
                panel.history_sessions.remove(session)
            break


def find_reconnection_source(
    panel: HistoryPanel, panel_str: str, output_uuid: str
) -> tuple[HistoryAction | None, str | None]:
    """Return the action and source UUID behind a removed output."""
    action_uuid = panel.runtime.objects.output_to_action.get(output_uuid)
    if action_uuid is None:
        return None, None
    for session in panel.history_sessions:
        for action in session.actions:
            if action.uuid == action_uuid:
                selection = action.state.selection.get(panel_str, [])
                source_uuid = selection[0] if selection else None
                return action, source_uuid
    return None, None


def plan_reconnection(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    removed_uuid: str,
) -> ReconnectionPlan:
    """Build a reconnection plan without mutating history or data objects."""
    panel_str = panel_data.PANEL_STR_ID
    producer_action, source_uuid = find_reconnection_source(
        panel, panel_str, removed_uuid
    )
    plan = ReconnectionPlan(
        panel_str=panel_str,
        removed_uuid=removed_uuid,
        source_uuid=source_uuid,
        producer_action=producer_action,
    )
    for obj in panel_data.objmodel:
        parameters = extract_processing_parameters(obj)
        if parameters is None:
            continue
        consumes_removed = parameters.source_uuid == removed_uuid or (
            parameters.source_uuids and removed_uuid in parameters.source_uuids
        )
        if not consumes_removed:
            continue
        action = None
        if parameters.func_name:
            action = find_action_for_output(panel, get_uuid(obj), parameters.func_name)
        plan.targets.append(ReconnectionTarget(get_uuid(obj), parameters, action))
    if not plan.targets:
        return plan
    alive_ids = set(panel_data.objmodel.get_object_ids())
    if source_uuid is None or source_uuid not in alive_ids:
        label = removed_uuid
        if producer_action is not None:
            label = producer_action.title or producer_action.func_name or removed_uuid
        plan.warning = (
            _(
                "“%s” has dependent operations but no valid source to "
                "reconnect to — downstream results are left unchanged."
            )
            % label
        )
        return plan
    if producer_action is not None:
        outputs = panel.runtime.objects.action_output_uuids.get(
            producer_action.uuid, []
        )
        plan.remove_producer = not any(output in alive_ids for output in outputs)
    return plan


def apply_reconnection_plan(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    plan: ReconnectionPlan,
    roots_to_recompute: list[HistoryAction],
) -> None:
    """Apply object and action source rewrites described by ``plan``."""
    if plan.warning is not None or plan.source_uuid is None:
        return
    for target in plan.targets:
        if not panel_data.objmodel.has_uuid(target.object_uuid):
            continue
        obj = panel_data.objmodel[target.object_uuid]
        insert_processing_parameters(
            obj,
            remap_processing_parameters(
                target.parameters, {plan.removed_uuid: plan.source_uuid}
            ),
        )
        if target.action is not None:
            rewrite_action_source(
                target.action,
                plan.panel_str,
                plan.removed_uuid,
                plan.source_uuid,
            )
            if target.action not in roots_to_recompute:
                roots_to_recompute.append(target.action)
    if plan.remove_producer and plan.producer_action is not None:
        remove_single_action(panel, plan.producer_action)


def reconnect_single_removed(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    removed_uuid: str,
    warnings: list[str],
    roots_to_recompute: list[HistoryAction],
) -> None:
    """Plan and reconnect consumers of one deleted object."""
    plan = plan_reconnection(panel, panel_data, removed_uuid)
    apply_reconnection_plan(panel, panel_data, plan, roots_to_recompute)
    if plan.warning is not None:
        warnings.append(plan.warning)


def show_reconnection_warnings(panel: HistoryPanel, warnings: list[str]) -> None:
    """Show reconnection warnings at the GUI boundary."""
    if warnings and not execenv.unattended:
        QW.QMessageBox.warning(
            panel.mainwindow,
            _("Delete"),
            _("Some operations could not be reconnected after deletion:")
            + "\n\n• "
            + "\n• ".join(warnings),
        )


def refresh_reconnected_history(panel: HistoryPanel) -> None:
    """Refresh the history tree after applying reconnection plans."""
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()
