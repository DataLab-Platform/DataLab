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
from datalab.gui.processor.base import (
    PROCESSING_PARAMETERS_OPTION,
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.history import HistoryAction, HistorySession
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history import HistoryPanel


def duplicate_selected_entries(panel: HistoryPanel) -> None:
    """Duplicate selected sessions (with their data) into new independent sessions.

    For each selected session (or the parent session of a selected action),
    all referenced data objects are deep-copied into a new group and the
    session is duplicated with all UUID references rewritten to the clones.
    The result is an independent, editable and replayable session.
    """
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        return
    # Normalise: resolve individual actions to their parent session, deduplicate.
    sessions_to_dup: list[HistorySession] = []
    seen: set[int] = set()
    for item in selected:
        if isinstance(item, HistorySession):
            session = item
        else:
            session = panel.find_parent_session(item)
            if session is None:
                continue
        if id(session) not in seen:
            seen.add(id(session))
            sessions_to_dup.append(session)

    copy_suffix = _("Copy")
    new_sessions: list[HistorySession] = []
    panel_map = {
        "signal": panel.mainwindow.signalpanel,
        "image": panel.mainwindow.imagepanel,
    }

    for session in sessions_to_dup:
        # 1. Collect all UUIDs referenced by this session
        uuids_by_panel: dict[str, set[str]] = {}
        for action in session.actions:
            for pstr, uuids in action.state.selection.items():
                uuids_by_panel.setdefault(pstr, set()).update(uuids)
            for pstr, metadata in action.state.object_metadata.items():
                uuids_by_panel.setdefault(pstr, set()).update(metadata.keys())
            obj2 = action.kwargs.get("obj2_uuids")
            if obj2:
                pstr = action.panel_str or ""
                if isinstance(obj2, str):
                    obj2 = [obj2]
                uuids_by_panel.setdefault(pstr, set()).update(obj2)
            # Output UUIDs produced by this action (e.g. result of a
            # compute step). Without this, the last action's outputs
            # would be missing because no subsequent state captures them.
            if action.output_uuids:
                pstr = action.panel_str or ""
                uuids_by_panel.setdefault(pstr, set()).update(action.output_uuids)

        # 2. Clone objects and build uuid_remap
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

        # 3. Build the new session with remapped UUIDs
        panel.session_increment += 1
        title = f"{session.title} {copy_suffix}"
        new_session = session.copy_with_uuid_remap(title=title, uuid_remap=uuid_remap)
        new_session.number = panel.session_increment
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
    for original_session, new_session in zip(sessions_to_dup, new_sessions):
        idx = panel.history_sessions.index(original_session)
        panel.history_sessions.insert(idx + 1 + offset, new_session)
        offset += 1
    panel.tree.populate_tree(panel.history_sessions)
    panel.select_sessions(new_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()


def generate_macro(panel: HistoryPanel) -> None:
    """Generate a standalone Python script from selected history entries.

    The generated script uses sigima functions directly with proper variable
    chaining.  Object references (UUIDs) are resolved to variable names so
    that 2-to-1 operations reference the correct intermediate result.
    The script is copied to the clipboard and the user is notified.
    """
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    actions: list[HistoryAction] = []
    if not selected:
        for session in panel.history_sessions:
            actions.extend(session.actions)
    else:
        for item in selected:
            if isinstance(item, HistorySession):
                actions.extend(item.actions)
            else:
                actions.append(item)
    if not actions:
        return

    # Filter to compute-only actions for the pipeline
    compute_actions = [a for a in actions if a.kind == HistoryAction.KIND_COMPUTE]
    if not compute_actions:
        if not execenv.unattended:
            QW.QMessageBox.information(
                panel.mainwindow,
                _("Generate macro"),
                _("No compute actions to export."),
            )
        return

    # Determine input type from first action
    first_panel = compute_actions[0].panel_str
    if first_panel == "signal":
        obj_type = "SignalObj"
        obj_import = "from sigima.objects import SignalObj"
    else:
        obj_type = "ImageObj"
        obj_import = "from sigima.objects import ImageObj"

    imports: set[str] = set()
    imports.add(obj_import)
    body_lines: list[str] = []

    # UUID → variable mapping for resolving object references.
    # Populated with input UUIDs ("src", "src_2", ...) and enriched
    # with each step's output UUID after code generation.
    uuid_to_var: dict[str, str] = {}

    # Extra input parameters discovered during generation (second
    # operands that are not produced by any previous step).
    extra_inputs: list[str] = []

    # Seed the mapping with the first action's input selection.
    first_sel = compute_actions[0].state.selection.get(compute_actions[0].panel_str, [])
    for i, uuid in enumerate(first_sel):
        var = "src" if i == 0 else f"src_{i + 1}"
        uuid_to_var[uuid] = var

    step = 0
    current_var = "src"

    for action in compute_actions:
        step += 1

        # Resolve input variable from the action's selection UUIDs.
        sel_uuids = action.state.selection.get(action.panel_str or "", [])
        if sel_uuids and sel_uuids[0] in uuid_to_var:
            input_var = uuid_to_var[sel_uuids[0]]
        else:
            input_var = current_var

        # Resolve second operand for 2-to-1 patterns.
        obj2_var: str | None = None
        if action.pattern == "2_to_1":
            obj2_uuids = action.kwargs.get("obj2_uuids", [])
            if isinstance(obj2_uuids, str):
                obj2_uuids = [obj2_uuids]
            if obj2_uuids:
                obj2_uuid = obj2_uuids[0]
                if obj2_uuid in uuid_to_var:
                    obj2_var = uuid_to_var[obj2_uuid]
                else:
                    # External input — add as function parameter.
                    obj2_var = f"obj2_{step}"
                    uuid_to_var[obj2_uuid] = obj2_var
                    extra_inputs.append(obj2_var)

        code_lines, output_var = action.to_macro_code(
            step, input_var, imports, obj2_var=obj2_var
        )
        body_lines.extend(code_lines)
        body_lines.append("")

        if output_var is not None:
            current_var = output_var
            # Map the output UUID so subsequent steps can reference it.
            output_uuid = panel.action_output_uuid(action)
            if output_uuid:
                uuid_to_var[output_uuid] = output_var
            # Also register any new UUIDs from the action's selection
            # that we haven't seen yet (secondary selections).
            for uuid in sel_uuids[1:]:
                if uuid not in uuid_to_var:
                    uuid_to_var[uuid] = input_var

    # Build the function signature with extra inputs.
    params_str = f"src: {obj_type}"
    for extra in extra_inputs:
        params_str += f", {extra}: {obj_type}"

    # Assemble the full script
    sorted_imports = sorted(imports)
    script_lines: list[str] = [
        '"""',
        "DataLab — standalone processing pipeline",
        f"Generated from history ({len(compute_actions)} steps)",
        '"""',
        "",
    ]
    script_lines.extend(sorted_imports)
    script_lines.append("")
    script_lines.append("")
    script_lines.append(f"def process({params_str}) -> {obj_type}:")
    script_lines.append('    """Apply the recorded processing pipeline."""')
    for line in body_lines:
        script_lines.append(f"    {line}" if line else "")
    script_lines.append(f"    return {current_var}")
    script_lines.append("")
    script_lines.append("")
    script_lines.append('if __name__ == "__main__":')
    script_lines.append("    # Standalone execution: run from DataLab's Macro panel.")
    script_lines.append("    # Operates on the current object of the target panel.")
    script_lines.append("    from datalab.control.proxy import RemoteProxy")
    script_lines.append("")
    script_lines.append("    proxy = RemoteProxy()")
    panel_str = compute_actions[0].panel_str or (
        "signal" if obj_type == "SignalObj" else "image"
    )
    script_lines.append(f'    proxy.set_current_panel("{panel_str}")')
    if extra_inputs:
        n_extra = len(extra_inputs)
        script_lines.append("    _uuids = proxy.get_sel_object_uuids()")
        script_lines.append(f"    if len(_uuids) < {n_extra + 1}:")
        script_lines.append("        raise RuntimeError(")
        script_lines.append(
            f'            "Pipeline needs {n_extra + 1} selected'
            f' object(s): 1 source + {n_extra} extra"'
        )
        script_lines.append("        )")
        script_lines.append(f'    src = proxy.get_object(_uuids[0], "{panel_str}")')
        script_lines.append("    if src is None:")
        script_lines.append(
            f'        raise RuntimeError("No current object in panel: {panel_str}")'
        )
        for idx, extra in enumerate(extra_inputs):
            script_lines.append(
                f'    {extra} = proxy.get_object(_uuids[{idx + 1}], "{panel_str}")'
            )
    else:
        script_lines.append("    src = proxy.get_object()")
        script_lines.append("    if src is None:")
        script_lines.append(
            f'        raise RuntimeError("No current object in panel: {panel_str}")'
        )
    extra_args = "".join(f", {e}" for e in extra_inputs)
    script_lines.append(f"    result = process(src{extra_args})")
    script_lines.append("    proxy.add_object(result)")
    script_lines.append('    print(f"Pipeline applied: {result.title}")')
    script_lines.append("")

    script = "\n".join(script_lines)
    QW.QApplication.clipboard().setText(script)
    if not execenv.unattended:
        QW.QMessageBox.information(
            panel.mainwindow,
            _("Generate macro"),
            _("Macro script copied to clipboard (%d actions).") % len(compute_actions),
        )


def _data_panel_for(panel: HistoryPanel, panel_str: str) -> BaseDataPanel | None:
    """Return the data panel matching ``panel_str`` (``"signal"``/``"image"``)."""
    if panel_str == "signal":
        return panel.mainwindow.signalpanel
    if panel_str == "image":
        return panel.mainwindow.imagepanel
    return None


def _action_input_uuids(action: HistoryAction) -> set[str]:
    """Return the set of object UUIDs consumed as inputs by ``action``."""
    captured: set[str] = set(action.state.selection.get(action.panel_str or "", []))
    obj2 = action.kwargs.get("obj2_uuids")
    if obj2:
        if isinstance(obj2, str):
            captured.add(obj2)
        else:
            captured.update(obj2)
    return captured


def _strip_source_links(obj) -> None:
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


def _remap_object_source(obj, old_uuid: str, new_uuid: str) -> None:
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


def _first_alive_output(
    panel: HistoryPanel, panel_str: str, output_uuids: list[str]
) -> str | None:
    """Return the first ``output_uuids`` entry still present in its data panel."""
    data_panel = _data_panel_for(panel, panel_str)
    if data_panel is None:
        return None
    for out_uuid in output_uuids:
        if data_panel.objmodel.has_uuid(out_uuid):
            return out_uuid
    return None


def _split_chain_on_action_delete(
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
        return _first_alive_output(panel, panel_str, output_uuids)
    first = downstream[0]
    data_panel = _data_panel_for(panel, first.panel_str or "")
    if data_panel is None:
        return _first_alive_output(panel, panel_str, output_uuids)
    # Locate the orphaned output object that ``first`` still consumes.
    first_inputs = _action_input_uuids(first)
    orphan_uuid = next((u for u in output_uuids if u in first_inputs), None)
    if orphan_uuid is None or not data_panel.objmodel.has_uuid(orphan_uuid):
        return _first_alive_output(panel, panel_str, output_uuids)
    # §8.2 — autonomy via COPY: clone the orphan as a parentless creation root.
    orphan_obj = data_panel.objmodel[orphan_uuid]
    clone = deepcopy(orphan_obj)
    new_uuid = str(uuid4())
    try:
        clone.set_metadata_option("uuid", new_uuid)
    except AttributeError:
        clone.uuid = new_uuid
    _strip_source_links(clone)
    group_id = get_uuid(data_panel.add_group(_("Chain copy")))
    data_panel.add_object(clone, group_id=group_id)
    # Rewire ALL downstream actions that directly consume the orphan onto the copy.
    for d in downstream:
        if orphan_uuid not in _action_input_uuids(d):
            continue
        hchain.rewrite_action_source(d, d.panel_str or "", orphan_uuid, new_uuid)
        for out_uuid in panel.action_output_uuids.get(d.uuid, []):
            if data_panel.objmodel.has_uuid(out_uuid):
                _remap_object_source(
                    data_panel.objmodel[out_uuid], orphan_uuid, new_uuid
                )
    # The orphan is now consumed by no surviving action: report it as orphaned.
    return orphan_uuid if data_panel.objmodel.has_uuid(orphan_uuid) else None


def _remove_data_object(data_panel: BaseDataPanel, obj_uuid: str) -> None:
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
            orphan_uuid = _split_chain_on_action_delete(panel, item)
            if orphan_uuid is not None:
                orphan_refs.append((item.panel_str or "", orphan_uuid))
    panel.history_sessions = [
        s for s in panel.history_sessions if id(s) not in sessions_to_remove
    ]
    # §8.3 — opt-in: offer to also remove the now-orphaned output object(s).
    alive_orphans: list[tuple[BaseDataPanel, str]] = []
    for panel_str, orphan_uuid in orphan_refs:
        data_panel = _data_panel_for(panel, panel_str)
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
                    _remove_data_object(data_panel, orphan_uuid)
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
