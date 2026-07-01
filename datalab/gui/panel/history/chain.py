# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Action↔output chain helpers for the History panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.panel.history.chainmodel import action_input_uuids
from datalab.gui.processor.base import (
    ProcessingParameters,
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


def resolve_panel_for_action(
    panel: HistoryPanel, action: HistoryAction
) -> BaseDataPanel | None:
    """Return the data panel targeted by ``action``, or ``None``."""
    if action.kind == HistoryAction.KIND_UI:
        if action.method_name not in HistoryAction.UI_CREATION_METHODS:
            return None
        if action.target == "signalpanel":
            return panel.mainwindow.signalpanel
        if action.target == "imagepanel":
            return panel.mainwindow.imagepanel
        return None
    if action.panel_str == "signal":
        return panel.mainwindow.signalpanel
    if action.panel_str == "image":
        return panel.mainwindow.imagepanel
    return None


def find_output_object_uuid(
    panel: HistoryPanel, panel_data: BaseDataPanel, action: HistoryAction
) -> str | None:
    """Find the UUID of the output object produced by ``action`` in ``panel_data``.

    Primary path: consult the bijective ``action_output_uuids`` mapping.
    Fallback path: legacy heuristic on ``processing_parameters`` metadata.
    """
    registered = panel.action_output_uuids.get(action.uuid)
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
    action_uuid = panel.output_to_action.get(output_uuid)
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
    action_uuid = panel.output_to_action.get(output_uuid)
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
                and output_uuid in panel.action_output_uuids.get(action.uuid, [])
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
    pstr = action.panel_str or ""
    captured: set[str] = set(action.state.selection.get(pstr, []))
    obj2 = action.kwargs.get("obj2_uuids")
    if obj2:
        if isinstance(obj2, str):
            captured.add(obj2)
        else:
            captured.update(obj2)
    return bool(captured & uuids)


def collect_downstream_uuids(panel: HistoryPanel, action: HistoryAction) -> set[str]:
    """Return the transitive closure of output UUIDs descending from ``action``."""
    if not panel.history_sessions:
        return set()
    current = get_session_of(panel, action)
    if current is None:
        return set()
    root_out = action_output_uuid(panel, action)
    if root_out is None:
        return set()
    closure: set[str] = {root_out}
    idx = current.actions.index(action)
    for downstream in current.actions[idx + 1 :]:
        if downstream.kind != HistoryAction.KIND_COMPUTE:
            continue
        if not action_consumes_any(downstream, closure):
            continue
        out_uuid = action_output_uuid(panel, downstream)
        if out_uuid is not None:
            closure.add(out_uuid)
    closure.discard(root_out)
    return closure


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
    registered = list(panel.action_output_uuids.get(action.uuid, []))
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
    if not panel.output_to_action:
        return
    alive: set[str] = set()
    for pdata in (panel.mainwindow.signalpanel, panel.mainwindow.imagepanel):
        alive.update(pdata.objmodel.get_object_ids())
    stale = [u for u in panel.output_to_action if u not in alive]
    for u in stale:
        panel.output_to_action.pop(u, None)


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
            outs = panel.action_output_uuids.pop(action.uuid, [])
            for out_uuid in outs:
                if panel.output_to_action.get(out_uuid) == action.uuid:
                    panel.output_to_action.pop(out_uuid, None)
            if not session.actions:
                panel.history_sessions.remove(session)
            break


def reconnect_single_removed(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    x_uuid: str,
    warnings: list[str],
    roots_to_recompute: list[HistoryAction],
) -> None:
    """Reconnect consumers of a single deleted object ``x_uuid``."""
    pstr = panel_data.PANEL_STR_ID
    action_a = None
    action_a_uuid = panel.output_to_action.get(x_uuid)
    if action_a_uuid is not None:
        for session in panel.history_sessions:
            for a in session.actions:
                if a.uuid == action_a_uuid:
                    action_a = a
                    break
            if action_a is not None:
                break
    consumers: list[tuple[Any, Any]] = []
    for obj in panel_data.objmodel:
        pp = extract_processing_parameters(obj)
        if pp is None:
            continue
        if pp.source_uuid == x_uuid or (pp.source_uuids and x_uuid in pp.source_uuids):
            consumers.append((obj, pp))
    if not consumers:
        return
    s_uuid: str | None = None
    if action_a is not None:
        sel = action_a.state.selection.get(pstr, [])
        if sel:
            s_uuid = sel[0]
    alive_ids = set(panel_data.objmodel.get_object_ids())
    if s_uuid is None or s_uuid not in alive_ids:
        label = action_a.title or action_a.func_name if action_a is not None else x_uuid
        warnings.append(
            _(
                "“%s” has dependent operations but no valid source to "
                "reconnect to — downstream results are left unchanged."
            )
            % label
        )
        return
    for obj, pp in consumers:
        new_source_uuid = s_uuid if pp.source_uuid == x_uuid else pp.source_uuid
        new_source_uuids = pp.source_uuids
        if pp.source_uuids and x_uuid in pp.source_uuids:
            new_source_uuids = [s_uuid if u == x_uuid else u for u in pp.source_uuids]
        insert_processing_parameters(
            obj,
            ProcessingParameters(
                func_name=pp.func_name,
                pattern=pp.pattern,
                param=pp.param,
                source_uuid=new_source_uuid,
                source_uuids=new_source_uuids,
            ),
        )
        if pp.func_name:
            action_b = find_action_for_output(panel, get_uuid(obj), pp.func_name)
            if action_b is not None:
                rewrite_action_source(action_b, pstr, x_uuid, s_uuid)
                if action_b not in roots_to_recompute:
                    roots_to_recompute.append(action_b)
    if action_a is not None:
        outs = panel.action_output_uuids.get(action_a.uuid, [])
        if not any(o in alive_ids for o in outs):
            remove_single_action(panel, action_a)


def reconnect_chain_after_removal(
    panel: HistoryPanel, panel_data: BaseDataPanel
) -> None:
    """Reconnect the processing chain after object(s) were deleted from a data panel."""
    pstr = panel_data.PANEL_STR_ID
    previous = panel.obj_ids_snapshot.get(pstr, set())
    current = set(panel_data.objmodel.get_object_ids())
    removed = previous - current
    if not removed or panel.reconnecting:
        return
    panel.reconnecting = True
    try:
        warnings: list[str] = []
        roots_to_recompute: list[HistoryAction] = []
        for x_uuid in removed:
            reconnect_single_removed(
                panel, panel_data, x_uuid, warnings, roots_to_recompute
            )
        for action in roots_to_recompute:
            hrec.recompute_action_in_place(panel, action)
            hrec.recompute_cascade(panel, action)
        if warnings and not execenv.unattended:
            QW.QMessageBox.warning(
                panel.mainwindow,
                _("Delete"),
                _("Some operations could not be reconnected after deletion:")
                + "\n\n• "
                + "\n• ".join(warnings),
            )
        panel.tree.populate_tree(panel.history_sessions)
        panel.refresh_compatibility_items()
        panel.update_actions_state()
    finally:
        panel.reconnecting = False
        panel.refresh_obj_ids_snapshot()
