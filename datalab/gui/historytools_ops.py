# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Helpers for History panel session tools."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from uuid import uuid4

from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history.chainmodel import (
    ProcessingChain,
    build_session_chains,
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


def duplicate_selected_entries(panel: HistoryPanel) -> None:
    """Duplicate selected processing chains into new independent sessions.

    Selection is resolved to *processing chains* (see
    :func:`build_session_chains`): a session **is** a single linear processing
    chain, so selecting a session duplicates its chain, and selecting an action
    duplicates the chain of its session. For each source session, exactly one
    new session is produced containing the duplicate of its chain, with all
    referenced data objects deep-copied into a new group and every UUID
    reference rewritten to the clones.

    Two chain shapes are handled:

    * **Cas A** -- creation-rooted chain (root is a ``new_object`` UI action):
      the chain is copied as-is (root included), no synthetic head is added.
    * **Cas B** -- operation-rooted chain (root consumes an external object):
      one synthetic ``new_object`` head is prepended per distinct remapped
      source object so the duplicated chain remains self-contained and
      replayable.

    The result is an independent, editable and replayable session.
    """
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        return

    # 1. Resolve selection to a per-session set of chains.
    session_by_id: dict[int, HistorySession] = {}
    full_session_ids: set[int] = set()
    actions_by_session: dict[int, list[HistoryAction]] = {}
    for item in selected:
        if isinstance(item, HistorySession):
            session_by_id[id(item)] = item
            full_session_ids.add(id(item))
        else:
            session = panel.find_parent_session(item)
            if session is None:
                continue
            session_by_id[id(session)] = session
            actions_by_session.setdefault(id(session), []).append(item)

    # Preserve source-session order (iterate panel.history_sessions).
    ordered: list[tuple[HistorySession, list[ProcessingChain]]] = []
    for session in panel.history_sessions:
        sid = id(session)
        if sid not in session_by_id:
            continue
        all_chains = build_session_chains(panel, session)
        if sid in full_session_ids:
            chains = all_chains
        else:
            chains = []
            seen_chains: set[int] = set()
            for action in actions_by_session.get(sid, []):
                for chain in all_chains:
                    if any(a is action or a.uuid == action.uuid for a in chain.actions):
                        if id(chain) not in seen_chains:
                            seen_chains.add(id(chain))
                            chains.append(chain)
                        break
        if chains:
            ordered.append((session, chains))
    if not ordered:
        return

    copy_suffix = _("Copy")
    new_sessions: list[HistorySession] = []
    panel_map = {
        "signal": panel.mainwindow.signalpanel,
        "image": panel.mainwindow.imagepanel,
    }

    for session, chains in ordered:
        # 2. Collect all UUIDs referenced by the SELECTED chains only.
        uuids_by_panel: dict[str, set[str]] = {}
        for chain in chains:
            for action in chain.actions:
                for pstr, uuids in action.state.selection.items():
                    uuids_by_panel.setdefault(pstr, set()).update(uuids)
                for pstr, metadata in action.state.object_metadata.items():
                    uuids_by_panel.setdefault(pstr, set()).update(metadata.keys())
                obj2 = action.kwargs.get("obj2_uuids")
                if obj2:
                    pstr = action_panel_str(action)
                    if isinstance(obj2, str):
                        obj2 = [obj2]
                    uuids_by_panel.setdefault(pstr, set()).update(obj2)
                # Output UUIDs produced by this action (e.g. result of a
                # compute step). Without this, the last action's outputs
                # would be missing because no subsequent state captures them.
                if action.output_uuids:
                    pstr = action_panel_str(action)
                    uuids_by_panel.setdefault(pstr, set()).update(action.output_uuids)

        # 3. Clone objects and build uuid_remap.
        uuid_remap: dict[str, dict[str, str]] = {}
        clones_by_pstr: dict[str, list] = {}
        group_title = f"{copy_suffix} - {session.title}"
        for pstr, uuids in uuids_by_panel.items():
            data_panel = panel_map.get(pstr)
            if data_panel is None:
                continue
            uuid_remap[pstr] = {}
            existing_ids = set(data_panel.objmodel.get_object_ids())
            clones = []
            # Iterate in panel order (not set order) to preserve
            # the topological object ordering in the duplicated group.
            ordered_ids = [
                u for u in data_panel.objmodel.get_object_ids() if u in uuids
            ]
            for old_uuid in ordered_ids:
                if old_uuid not in existing_ids:
                    continue
                obj = data_panel.objmodel[old_uuid]
                clone = deepcopy(obj)
                new_uuid = str(uuid4())
                # SignalObj/ImageObj store UUID via metadata option
                try:
                    clone.set_metadata_option("uuid", new_uuid)
                except AttributeError:
                    clone.uuid = new_uuid
                uuid_remap[pstr][old_uuid] = new_uuid
                clones.append(clone)
            clones_by_pstr[pstr] = clones
            if clones:
                group_id = get_uuid(data_panel.add_group(group_title))
                for clone in clones:
                    data_panel.add_object(clone, group_id=group_id)

        # Second pass: remap source UUIDs in cloned objects'
        # processing_parameters so reprocessing in the Processing tab
        # uses the cloned source, not the original.
        for pstr_inner, clones_inner in clones_by_pstr.items():
            pmap = uuid_remap.get(pstr_inner, {})
            if not pmap:
                continue
            for clone in clones_inner:
                try:
                    pp_dict = clone.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
                except (AttributeError, ValueError):
                    continue
                if not pp_dict:
                    continue
                try:
                    pp = ProcessingParameters.from_dict(pp_dict)
                except (TypeError, ValueError, AttributeError):
                    continue
                changed = False
                if pp.source_uuid is not None and pp.source_uuid in pmap:
                    pp.source_uuid = pmap[pp.source_uuid]
                    changed = True
                if pp.source_uuids is not None:
                    new_src = [pmap.get(u, u) for u in pp.source_uuids]
                    if new_src != pp.source_uuids:
                        pp.source_uuids = new_src
                        changed = True
                if changed:
                    try:
                        clone.set_metadata_option(
                            PROCESSING_PARAMETERS_OPTION, pp.to_dict()
                        )
                    except (AttributeError, ValueError):
                        pass

        # 4. Build the new session's actions per chain (Cas A / Cas B).
        new_actions: list[HistoryAction] = []
        for chain in chains:
            is_creation_root = (
                chain.root.kind == HistoryAction.KIND_UI
                and chain.root.method_name in HistoryAction.UI_CREATION_METHODS
            )
            if is_creation_root:
                # Cas A: creation-rooted chain, copy as-is (root included).
                new_actions.extend(
                    action.copy_with_uuid_remap(uuid_remap) for action in chain.actions
                )
                continue
            # Cas B: operation-rooted chain, synthesize one creation head per
            # distinct remapped source object of the chain root.
            root_inputs: list[tuple[str, str]] = []
            for pstr, uuids in chain.root.state.selection.items():
                for old_uuid in uuids:
                    root_inputs.append((pstr, old_uuid))
            obj2 = chain.root.kwargs.get("obj2_uuids")
            if obj2:
                pstr = action_panel_str(chain.root)
                if isinstance(obj2, str):
                    obj2 = [obj2]
                for old_uuid in obj2:
                    root_inputs.append((pstr, old_uuid))
            heads: list[HistoryAction] = []
            seen_clones: set[str] = set()
            for pstr, old_uuid in root_inputs:
                clone_uuid = uuid_remap.get(pstr, {}).get(old_uuid)
                if clone_uuid is None or clone_uuid in seen_clones:
                    # Source object deleted / not clonable, or already headed.
                    continue
                seen_clones.add(clone_uuid)
                head_title = _("Initial state")
                data_panel = panel_map.get(pstr)
                if data_panel is not None:
                    try:
                        head_title = data_panel.objmodel[clone_uuid].title
                    except (KeyError, AttributeError):
                        head_title = _("Initial state")
                head = make_initial_state_head(pstr, clone_uuid, head_title)
                heads.append(head)
                panel.action_output_uuids[head.uuid] = [clone_uuid]
                panel.output_to_action[clone_uuid] = head.uuid
            new_actions.extend(heads)
            new_actions.extend(
                action.copy_with_uuid_remap(uuid_remap) for action in chain.actions
            )

        # 5. Assemble the new session and register output mappings.
        panel.session_increment += 1
        title = f"{session.title} {copy_suffix}"
        new_session = HistorySession(title=title, number=panel.session_increment)
        new_session.actions = new_actions
        new_sessions.append(new_session)

        # Register output mappings for cloned actions so that
        # resolve_target_outputs / get_downstream_actions work on
        # the duplicated session (same logic as deserialize_from_hdf5).
        for action in new_session.actions:
            if action.output_uuids:
                panel.action_output_uuids[action.uuid] = list(action.output_uuids)
                for out_uuid in action.output_uuids:
                    panel.output_to_action[out_uuid] = action.uuid

    # Insert each duplicated session immediately after its original.
    offset = 0
    source_sessions = [session for session, _chains in ordered]
    for original_session, new_session in zip(source_sessions, new_sessions):
        idx = panel.history_sessions.index(original_session)
        panel.history_sessions.insert(idx + 1 + offset, new_session)
        offset += 1
    panel.tree.populate_tree(panel.history_sessions)
    panel.select_sessions(new_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()


def data_panel_for(panel: HistoryPanel, panel_str: str) -> BaseDataPanel | None:
    """Return the data panel matching ``panel_str`` (``"signal"``/``"image"``)."""
    if panel_str == "signal":
        return panel.mainwindow.signalpanel
    if panel_str == "image":
        return panel.mainwindow.imagepanel
    return None


def action_input_uuids(action: HistoryAction) -> set[str]:
    """Return the set of object UUIDs consumed as inputs by ``action``."""
    captured: set[str] = set(action.state.selection.get(action.panel_str or "", []))
    obj2 = action.kwargs.get("obj2_uuids")
    if obj2:
        if isinstance(obj2, str):
            captured.add(obj2)
        else:
            captured.update(obj2)
    return captured


def strip_source_links(obj) -> None:
    """Turn ``obj`` into a parentless creation root (drop source references)."""
    pp = extract_processing_parameters(obj)
    if pp is None:
        return
    insert_processing_parameters(
        obj,
        ProcessingParameters(
            func_name=pp.func_name,
            pattern=pp.pattern,
            param=pp.param,
            source_uuid=None,
            source_uuids=None,
        ),
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
    output_uuids = list(panel.action_output_uuids.get(action.uuid, []))
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
        for out_uuid in panel.action_output_uuids.get(d.uuid, []):
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


def delete_selected(panel: HistoryPanel) -> None:
    """Delete the selected actions or sessions (with confirmation).

    When a top-level session is selected, the entire session is deleted.
    When individual actions are selected, each action is spliced out of its
    session and its processing chain is split: downstream steps become an
    independent chain rooted at a copy of the deleted action's output. After
    deletion, the last action leaf in the affected session is selected.
    """
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        return
    has_individual_actions = any(isinstance(item, HistoryAction) for item in selected)
    if has_individual_actions:
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
    if reply != QW.QMessageBox.Yes:
        return
    # Memorize affected session for post-deletion selection
    affected_session: HistorySession | None = None
    for item in selected:
        if isinstance(item, HistoryAction):
            for session in panel.history_sessions:
                if item in session.actions:
                    affected_session = session
                    break
            if affected_session is not None:
                break

    sessions_to_remove: set[int] = set()
    orphan_refs: list[tuple[str, str]] = []
    for item in selected:
        if isinstance(item, HistorySession):
            sessions_to_remove.add(id(item))
        elif isinstance(item, HistoryAction):
            # Individual action: splice it out and split its chain (with copy).
            orphan_uuid = split_chain_on_action_delete(panel, item)
            if orphan_uuid is not None:
                orphan_refs.append((item.panel_str or "", orphan_uuid))
    panel.history_sessions = [
        s for s in panel.history_sessions if id(s) not in sessions_to_remove
    ]
    # §8.3 — opt-in: offer to also remove the now-orphaned output object(s).
    alive_orphans: list[tuple[BaseDataPanel, str]] = []
    for panel_str, orphan_uuid in orphan_refs:
        data_panel = data_panel_for(panel, panel_str)
        if data_panel is not None and data_panel.objmodel.has_uuid(orphan_uuid):
            alive_orphans.append((data_panel, orphan_uuid))
    if alive_orphans and not execenv.unattended:
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
        if answer == QW.QMessageBox.Yes:
            touched: dict[int, BaseDataPanel] = {}
            for data_panel, orphan_uuid in alive_orphans:
                if data_panel.objmodel.has_uuid(orphan_uuid):
                    remove_data_object(data_panel, orphan_uuid)
                    touched[id(data_panel)] = data_panel
            for data_panel in touched.values():
                data_panel.objview.update_tree()
                data_panel.selection_changed(update_items=True)
            panel.refresh_obj_ids_snapshot()
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()
    # Auto-select an item in the same session to avoid panel switch
    target_item = None
    if affected_session is not None and id(affected_session) not in sessions_to_remove:
        # Find the tree item for the affected session (same index in list)
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
        # Fallback: last top-level item (least likely to switch panels)
        last_top = panel.tree.topLevelItem(panel.tree.topLevelItemCount() - 1)
        target_item = last_top
    if target_item is not None:
        panel.tree.setCurrentItem(target_item)
        target_item.setSelected(True)


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
            session.actions.remove(action)
    # Remove empty sessions
    panel.history_sessions = [s for s in panel.history_sessions if s.actions]
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()
