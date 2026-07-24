# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Helpers for History panel session tools."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history.chainmodel import (
    ChainSelectionPlan,
    DeletionPlan,
    DeletionResult,
    DuplicatedSession,
    ProcessingChain,
    UuidCloneRegistry,
    action_input_uuids,
    build_session_chains,
    remap_processing_parameters,
)
from datalab.gui.processor.base import (
    PROCESSING_PARAMETERS_OPTION,
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.history import HistoryAction, HistorySession
from datalab.history.workspace_state import WorkspaceState
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history import HistoryPanel


def action_panel_str(action: HistoryAction) -> str:
    """Return the panel an action operates on, using target as fallback."""
    if action.panel_str:
        return action.panel_str
    return {"imagepanel": "image", "signalpanel": "signal"}.get(action.target, "signal")


def make_initial_state_head(pstr: str, clone_uuid: str, title: str) -> HistoryAction:
    """Return a synthetic creation-root action for an operation-rooted chain.

    A *Cas B* chain starts from an operation whose input object was created
    outside the chain (e.g. imported or added programmatically). To make the
    duplicated chain self-contained and replayable, a synthetic ``new_object``
    UI action is prepended, standing in for that missing object creation. Its
    empty workspace state mirrors a real ``new_object`` recorded with
    ``save_state=False`` (hence always compatible), and its ``new_object``
    method name places it in :attr:`HistoryAction.UI_CREATION_METHODS` so
    :func:`build_session_chains` treats it as a genuine chain root.

    Args:
        pstr: Panel string of the created object (``"signal"``/``"image"``).
        clone_uuid: UUID of the cloned source object produced as head output.
        title: Title shown for the synthetic action in the tree.

    Returns:
        A new :class:`HistoryAction` describing the synthetic creation root.
    """
    target = "signalpanel" if pstr == "signal" else "imagepanel"
    head = HistoryAction(
        title=title,
        kind=HistoryAction.KIND_UI,
        target=target,
        method_name="new_object",
        kwargs={},
        state=WorkspaceState(),
        panel_str=None,
    )
    head.output_uuids = [clone_uuid]
    return head


def append_action_chain(
    action: HistoryAction,
    all_chains: list[ProcessingChain],
    seen_chains: set[int],
    chains: list[ProcessingChain],
) -> None:
    """Append the unseen processing chain containing the selected action."""
    for chain in all_chains:
        if any(item is action or item.uuid == action.uuid for item in chain.actions):
            if id(chain) not in seen_chains:
                seen_chains.add(id(chain))
                chains.append(chain)
            break


def resolve_chain_selection(panel: HistoryPanel) -> list[ChainSelectionPlan]:
    """Resolve tree selection to ordered processing chains per source session."""
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    session_by_id: dict[int, HistorySession] = {}
    full_session_ids: set[int] = set()
    actions_by_session: dict[int, list[HistoryAction]] = {}
    for item in selected:
        if isinstance(item, HistorySession):
            session_by_id[id(item)] = item
            full_session_ids.add(id(item))
        else:
            session = hchain.find_parent_session(panel, item)
            if session is None:
                continue
            session_by_id[id(session)] = session
            actions_by_session.setdefault(id(session), []).append(item)

    # Preserve source-session order (iterate panel.history_sessions).
    plans: list[ChainSelectionPlan] = []
    for session in panel.history_sessions:
        sid = id(session)
        if sid not in session_by_id:
            continue
        all_chains = build_session_chains(session)
        if sid in full_session_ids:
            chains = all_chains
        else:
            chains = []
            seen_chains: set[int] = set()
            for action in actions_by_session.get(sid, []):
                append_action_chain(action, all_chains, seen_chains, chains)
        if chains:
            plans.append(ChainSelectionPlan(session, chains))
    return plans


def collect_referenced_uuids(chains: list[ProcessingChain]) -> dict[str, set[str]]:
    """Collect object UUIDs referenced or produced by selected chains."""
    uuids_by_panel: dict[str, set[str]] = {}
    for chain in chains:
        for action in chain.actions:
            for panel_str, uuids in action.state.selection.items():
                uuids_by_panel.setdefault(panel_str, set()).update(uuids)
            for panel_str, metadata in action.state.object_metadata.items():
                uuids_by_panel.setdefault(panel_str, set()).update(metadata)
            obj2_uuids = action.kwargs.get("obj2_uuids")
            if obj2_uuids:
                panel_str = action_panel_str(action)
                if isinstance(obj2_uuids, str):
                    obj2_uuids = [obj2_uuids]
                uuids_by_panel.setdefault(panel_str, set()).update(obj2_uuids)
            if action.output_uuids:
                panel_str = action_panel_str(action)
                uuids_by_panel.setdefault(panel_str, set()).update(action.output_uuids)
    return uuids_by_panel


def set_object_uuid(obj: Any, new_uuid: str) -> None:
    """Set a cloned data object's UUID through its supported storage API."""
    try:
        obj.set_metadata_option("uuid", new_uuid)
    except AttributeError:
        obj.uuid = new_uuid


def clone_referenced_objects(
    panel: HistoryPanel, plan: ChainSelectionPlan, copy_suffix: str
) -> UuidCloneRegistry:
    """Clone a selection plan's objects and register source-to-clone UUIDs."""
    registry = UuidCloneRegistry()
    group_title = f"{copy_suffix} - {plan.source_session.title}"
    for panel_str, referenced in collect_referenced_uuids(plan.chains).items():
        data_panel = data_panel_for(panel, panel_str)
        if data_panel is None:
            continue
        ordered_uuids = [
            obj_uuid
            for obj_uuid in data_panel.objmodel.get_object_ids()
            if obj_uuid in referenced
        ]
        clones: list[Any] = []
        for old_uuid in ordered_uuids:
            clone = deepcopy(data_panel.objmodel[old_uuid])
            new_uuid = str(uuid4())
            set_object_uuid(clone, new_uuid)
            registry.register(panel_str, old_uuid, new_uuid, clone)
            clones.append(clone)
        if clones:
            group_id = get_uuid(data_panel.add_group(group_title))
            for clone in clones:
                data_panel.add_object(clone, group_id=group_id)
    return registry


def remap_cloned_object_sources(registry: UuidCloneRegistry) -> None:
    """Rewrite processing-parameter source UUIDs in all cloned objects."""
    for panel_str, clones in registry.clones_by_panel.items():
        panel_remap = registry.uuid_remap.get(panel_str, {})
        for clone in clones:
            try:
                parameters_dict = clone.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
            except (AttributeError, ValueError):
                continue
            if not parameters_dict:
                continue
            try:
                parameters = ProcessingParameters.from_dict(parameters_dict)
            except (TypeError, ValueError, AttributeError):
                continue
            remapped = remap_processing_parameters(parameters, panel_remap)
            if remapped == parameters:
                continue
            try:
                clone.set_metadata_option(
                    PROCESSING_PARAMETERS_OPTION, remapped.to_dict()
                )
            except (AttributeError, ValueError):
                continue


def chain_root_inputs(action: HistoryAction) -> list[tuple[str, str]]:
    """Return panel-qualified source UUIDs consumed by a chain root."""
    inputs = [
        (panel_str, old_uuid)
        for panel_str, uuids in action.state.selection.items()
        for old_uuid in uuids
    ]
    obj2_uuids = action.kwargs.get("obj2_uuids")
    if isinstance(obj2_uuids, str):
        obj2_uuids = [obj2_uuids]
    if obj2_uuids:
        panel_str = action_panel_str(action)
        inputs.extend((panel_str, old_uuid) for old_uuid in obj2_uuids)
    return inputs


def make_synthetic_heads(
    panel: HistoryPanel, chain: ProcessingChain, registry: UuidCloneRegistry
) -> list[HistoryAction]:
    """Create one independent creation head per cloned external root input."""
    heads: list[HistoryAction] = []
    seen_clones: set[str] = set()
    for panel_str, old_uuid in chain_root_inputs(chain.root):
        clone_uuid = registry.resolve(panel_str, old_uuid)
        if clone_uuid is None or clone_uuid in seen_clones:
            continue
        seen_clones.add(clone_uuid)
        head_title = _("Initial state")
        data_panel = data_panel_for(panel, panel_str)
        if data_panel is not None:
            try:
                head_title = data_panel.objmodel[clone_uuid].title
            except (KeyError, AttributeError):
                pass
        heads.append(make_initial_state_head(panel_str, clone_uuid, head_title))
    return heads


def assemble_duplicated_actions(
    panel: HistoryPanel, plan: ChainSelectionPlan, registry: UuidCloneRegistry
) -> list[HistoryAction]:
    """Copy selected actions and add heads for operation-rooted chains."""
    new_actions: list[HistoryAction] = []
    for chain in plan.chains:
        is_creation_root = (
            chain.root.kind == HistoryAction.KIND_UI
            and chain.root.method_name in HistoryAction.UI_CREATION_METHODS
        )
        if not is_creation_root:
            new_actions.extend(make_synthetic_heads(panel, chain, registry))
        new_actions.extend(
            action.copy_with_uuid_remap(registry.uuid_remap) for action in chain.actions
        )
    return new_actions


def register_session_outputs(panel: HistoryPanel, session: HistorySession) -> None:
    """Register action-to-output mappings for one assembled session."""
    for action in session.actions:
        if not action.output_uuids:
            continue
        panel.runtime.objects.register_action_outputs(action, action.output_uuids)


def duplicate_chain_plan(
    panel: HistoryPanel, plan: ChainSelectionPlan, copy_suffix: str
) -> DuplicatedSession:
    """Clone objects and assemble one independent duplicated session."""
    registry = clone_referenced_objects(panel, plan, copy_suffix)
    remap_cloned_object_sources(registry)
    panel.navigation.session_increment += 1
    new_session = HistorySession(
        title=f"{plan.source_session.title} {copy_suffix}",
        number=panel.navigation.session_increment,
    )
    new_session.actions = assemble_duplicated_actions(panel, plan, registry)
    register_session_outputs(panel, new_session)
    return DuplicatedSession(plan.source_session, new_session)


def insert_duplicated_sessions(
    panel: HistoryPanel, duplicated_sessions: list[DuplicatedSession]
) -> None:
    """Insert duplicates after their sources and refresh/select the tree."""
    for duplicated in reversed(duplicated_sessions):
        source_index = panel.history_sessions.index(duplicated.source_session)
        panel.history_sessions.insert(source_index + 1, duplicated.new_session)
    panel.tree.populate_tree(panel.history_sessions)
    panel.navigation.select_sessions([item.new_session for item in duplicated_sessions])
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()


def duplicate_selected_entries(panel: HistoryPanel) -> None:
    """Duplicate selected processing chains into new independent sessions."""
    selection_plans = resolve_chain_selection(panel)
    if not selection_plans:
        return

    copy_suffix = _("Copy")
    duplicated_sessions = [
        duplicate_chain_plan(panel, plan, copy_suffix) for plan in selection_plans
    ]
    insert_duplicated_sessions(panel, duplicated_sessions)


def data_panel_for(panel: HistoryPanel, panel_str: str) -> BaseDataPanel | None:
    """Return the data panel matching ``panel_str`` (``"signal"``/``"image"``)."""
    if panel_str == "signal":
        return panel.mainwindow.signalpanel
    if panel_str == "image":
        return panel.mainwindow.imagepanel
    return None


def strip_source_links(obj) -> None:
    """Turn ``obj`` into a parentless creation root (drop source references)."""
    pp = extract_processing_parameters(obj)
    if pp is None:
        return
    insert_processing_parameters(
        obj,
        remap_processing_parameters(pp, {}, clear_sources=True),
    )


def remap_object_source(obj, old_uuid: str, new_uuid: str) -> None:
    """Replace ``old_uuid`` with ``new_uuid`` in ``obj``'s source references."""
    pp = extract_processing_parameters(obj)
    if pp is None:
        return
    changed = False
    if pp.source_uuid == old_uuid:
        pp.source_uuid = new_uuid
        changed = True
    if pp.source_uuids and old_uuid in pp.source_uuids:
        pp.source_uuids = [new_uuid if u == old_uuid else u for u in pp.source_uuids]
        changed = True
    if changed:
        insert_processing_parameters(obj, pp)


def first_alive_output(
    panel: HistoryPanel, panel_str: str, output_uuids: list[str]
) -> str | None:
    """Return the first ``output_uuids`` entry still present in its data panel."""
    data_panel = data_panel_for(panel, panel_str)
    if data_panel is None:
        return None
    for out_uuid in output_uuids:
        if data_panel.objmodel.has_uuid(out_uuid):
            return out_uuid
    return None


def split_chain_on_action_delete(
    panel: HistoryPanel, action: HistoryAction
) -> str | None:
    """Splice ``action`` out of its session and split its processing chain.

    The action is removed from its session (splice, not truncate). If it had
    downstream compute steps, the first downstream action becomes the head of a
    new, independent chain: the deleted action's now-orphaned output object is
    deep-copied into a new ``Chain copy`` group as a parentless creation root,
    and the downstream head is rewired to consume that copy.

    Args:
        panel: The history panel owning sessions and the output registry.
        action: The action to delete.

    Returns:
        The UUID of the deleted action's output object if it remains present
        (now truly orphaned) in its data panel, otherwise ``None``.
    """
    panel_str = action.panel_str or ""
    # Compute downstream + captured output UUIDs BEFORE removing the action.
    downstream = hchain.get_downstream_actions(panel, action)
    output_uuids = list(panel.runtime.objects.action_output_uuids.get(action.uuid, []))
    # Splice the action out (does not truncate the rest of the session).
    hchain.remove_single_action(panel, action)
    if not downstream:
        return first_alive_output(panel, panel_str, output_uuids)
    first = downstream[0]
    data_panel = data_panel_for(panel, first.panel_str or "")
    if data_panel is None:
        return first_alive_output(panel, panel_str, output_uuids)
    # Locate the orphaned output object that ``first`` still consumes.
    first_inputs = action_input_uuids(first)
    orphan_uuid = next((u for u in output_uuids if u in first_inputs), None)
    if orphan_uuid is None or not data_panel.objmodel.has_uuid(orphan_uuid):
        return first_alive_output(panel, panel_str, output_uuids)
    # §8.2 — autonomy via COPY: clone the orphan as a parentless creation root.
    orphan_obj = data_panel.objmodel[orphan_uuid]
    clone = deepcopy(orphan_obj)
    new_uuid = str(uuid4())
    try:
        clone.set_metadata_option("uuid", new_uuid)
    except AttributeError:
        clone.uuid = new_uuid
    strip_source_links(clone)
    group_id = get_uuid(data_panel.add_group(_("Chain copy")))
    data_panel.add_object(clone, group_id=group_id)
    # Rewire ALL downstream actions that directly consume the orphan onto the copy.
    for d in downstream:
        if orphan_uuid not in action_input_uuids(d):
            continue
        hchain.rewrite_action_source(d, d.panel_str or "", orphan_uuid, new_uuid)
        for out_uuid in panel.runtime.objects.action_output_uuids.get(d.uuid, []):
            if data_panel.objmodel.has_uuid(out_uuid):
                remap_object_source(
                    data_panel.objmodel[out_uuid], orphan_uuid, new_uuid
                )
    # The orphan is now consumed by no surviving action: report it as orphaned.
    return orphan_uuid if data_panel.objmodel.has_uuid(orphan_uuid) else None


def remove_data_object(data_panel: BaseDataPanel, obj_uuid: str) -> None:
    """Remove a single object from ``data_panel`` without recording history."""
    obj = data_panel.objmodel[obj_uuid]
    data_panel.plothandler.remove_item(obj_uuid)
    data_panel.objview.remove_item(obj_uuid, refresh=False)
    data_panel.objmodel.remove_object(obj)


def plan_deletion(
    panel: HistoryPanel, selected: list[HistoryAction | HistorySession]
) -> DeletionPlan:
    """Classify selected history entities without mutating panel state."""
    plan = DeletionPlan()
    for item in selected:
        if isinstance(item, HistorySession):
            plan.session_ids.add(id(item))
            continue
        plan.actions.append(item)
        if plan.affected_session is None:
            plan.affected_session = hchain.find_parent_session(panel, item)
    return plan


def confirm_deletion(panel: HistoryPanel, plan: DeletionPlan) -> bool:
    """Ask the user to confirm a planned history deletion."""
    if plan.actions:
        msg = _(
            "Do you really want to delete the selected items?\n\n"
            "Note: deleting an intermediate action splits its processing "
            "chain; downstream steps become an independent chain."
        )
    else:
        msg = _("Do you really want to delete the selected items?")
    reply = (
        QW.QMessageBox.Yes
        if execenv.unattended
        else QW.QMessageBox.question(
            panel.mainwindow,
            _("Delete"),
            msg,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
    )
    return reply == QW.QMessageBox.Yes


def apply_deletion(panel: HistoryPanel, plan: DeletionPlan) -> DeletionResult:
    """Delete planned actions/sessions and return orphan cleanup state."""
    orphan_refs: list[tuple[str, str]] = []
    for action in plan.actions:
        orphan_uuid = split_chain_on_action_delete(panel, action)
        if orphan_uuid is not None:
            orphan_refs.append((action.panel_str or "", orphan_uuid))
    for session in panel.history_sessions:
        if id(session) in plan.session_ids:
            for action in session.actions:
                panel.runtime.objects.remove_action_outputs(action)
    panel.history_sessions = [
        session
        for session in panel.history_sessions
        if id(session) not in plan.session_ids
    ]
    return DeletionResult(
        plan.affected_session,
        plan.session_ids,
        orphan_refs,
    )


def collect_alive_orphans(
    panel: HistoryPanel, orphan_refs: list[tuple[str, str]]
) -> list[tuple[BaseDataPanel, str]]:
    """Resolve orphan references that are still present in data panels."""
    alive_orphans: list[tuple[BaseDataPanel, str]] = []
    for panel_str, orphan_uuid in orphan_refs:
        data_panel = data_panel_for(panel, panel_str)
        if data_panel is not None and data_panel.objmodel.has_uuid(orphan_uuid):
            alive_orphans.append((data_panel, orphan_uuid))
    return alive_orphans


def confirm_orphan_removal(
    panel: HistoryPanel, alive_orphans: list[tuple[BaseDataPanel, str]]
) -> bool:
    """Ask whether surviving output objects should also be removed."""
    if not alive_orphans or execenv.unattended:
        return False
    answer = QW.QMessageBox.question(
        panel.mainwindow,
        _("Delete"),
        _(
            "The deleted action(s) produced object(s) still present in the "
            "workspace. Do you want to remove the associated object(s) as well?"
        ),
        QW.QMessageBox.Yes | QW.QMessageBox.No,
        QW.QMessageBox.No,
    )
    return answer == QW.QMessageBox.Yes


def remove_orphan_objects(
    panel: HistoryPanel, alive_orphans: list[tuple[BaseDataPanel, str]]
) -> None:
    """Remove confirmed orphan objects and refresh affected data panels."""
    touched: dict[int, BaseDataPanel] = {}
    for data_panel, orphan_uuid in alive_orphans:
        if data_panel.objmodel.has_uuid(orphan_uuid):
            remove_data_object(data_panel, orphan_uuid)
            touched[id(data_panel)] = data_panel
    for data_panel in touched.values():
        data_panel.objview.update_tree()
        data_panel.selection_changed(update_items=True)
    panel.runtime.objects.refresh_obj_ids_snapshot()


def refresh_history_after_deletion(panel: HistoryPanel) -> None:
    """Refresh history presentation after applying a deletion."""
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()


def select_after_deletion(panel: HistoryPanel, result: DeletionResult) -> None:
    """Select the last action of the affected surviving session, or a fallback."""
    target_item = None
    affected_session = result.affected_session
    if (
        affected_session is not None
        and id(affected_session) not in result.removed_session_ids
    ):
        try:
            session_idx = panel.history_sessions.index(affected_session)
        except ValueError:
            session_idx = -1
        if session_idx >= 0:
            top = panel.tree.topLevelItem(session_idx)
            if top is not None:
                last_action_item = None
                iterator = QW.QTreeWidgetItemIterator(top)
                while iterator.value():
                    node = iterator.value()
                    if (
                        node.data(0, panel.tree.ITEM_KIND_ROLE)
                        == panel.tree.ITEM_ACTION
                    ):
                        last_action_item = node
                    iterator += 1
                target_item = last_action_item if last_action_item is not None else top
    if target_item is None and panel.tree.topLevelItemCount() > 0:
        target_item = panel.tree.topLevelItem(panel.tree.topLevelItemCount() - 1)
    if target_item is not None:
        panel.tree.setCurrentItem(target_item)
        target_item.setSelected(True)


def delete_selected(panel: HistoryPanel) -> None:
    """Delete selected actions or sessions through explicit GUI/mutation phases."""
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        return
    plan = plan_deletion(panel, selected)
    if not confirm_deletion(panel, plan):
        return
    result = apply_deletion(panel, plan)
    alive_orphans = collect_alive_orphans(panel, result.orphan_refs)
    if confirm_orphan_removal(panel, alive_orphans):
        remove_orphan_objects(panel, alive_orphans)
    refresh_history_after_deletion(panel)
    select_after_deletion(panel, result)


def remove_incompatible_actions(panel: HistoryPanel) -> None:
    """Remove all actions whose workspace state is incompatible.

    Shows a confirmation dialog listing how many actions will be removed,
    then purges them from their sessions. Empty sessions are also removed.
    """
    incompatible: list[tuple[HistorySession, HistoryAction]] = []
    for session in panel.history_sessions:
        for action in session.actions:
            if not action.is_current_state_compatible(
                panel.mainwindow, restore_selection=True
            ):
                incompatible.append((session, action))
    if not incompatible:
        if not execenv.unattended:
            QW.QMessageBox.information(
                panel.mainwindow,
                _("Remove incompatible"),
                _("All actions are compatible with the current workspace."),
            )
        return
    reply = (
        QW.QMessageBox.Yes
        if execenv.unattended
        else QW.QMessageBox.question(
            panel.mainwindow,
            _("Remove incompatible"),
            _("%d incompatible action(s) will be removed. Continue?")
            % len(incompatible),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
    )
    if reply != QW.QMessageBox.Yes:
        return
    for session, action in incompatible:
        if action in session.actions:
            panel.runtime.objects.remove_action_outputs(action)
            session.actions.remove(action)
    # Remove empty sessions
    panel.history_sessions = [s for s in panel.history_sessions if s.actions]
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()
