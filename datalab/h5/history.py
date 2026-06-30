# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel HDF5 import/export and persistence helpers."""

from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from qtpy.compat import getopenfilename, getsavefilename

from datalab.config import Conf, _
from datalab.gui.processor.base import (
    PROCESSING_PARAMETERS_OPTION,
    ProcessingParameters,
)
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history import HistorySession
from datalab.objectmodel import get_uuid
from datalab.utils.qthelpers import qt_try_loadsave_file, save_restore_stds

if TYPE_CHECKING:
    from datalab.gui.panel.history import HistoryPanel


def save_to_dlhist_file(panel: HistoryPanel, filename: str | None = None) -> bool:
    """Save the History Panel content to a standalone ``.dlhist`` file.

    Args:
        filename: History filename. If None, a file dialog is opened.

    Returns:
        True if the history was saved, False if the operation was canceled.
    """
    if filename is None:
        basedir = Conf.main.base_dir.get()
        with save_restore_stds():
            filename, _filt = getsavefilename(
                panel, _("Save history file"), basedir, panel.FILE_FILTERS
            )
    if not filename:
        return False
    if osp.splitext(filename)[1] == "":
        filename += ".dlhist"
    with qt_try_loadsave_file(panel.parentWidget(), filename, "save"):
        Conf.main.base_dir.set(filename)
        with NativeH5Writer(filename) as writer:
            # Make the .dlhist file panel-contained: store the signal and
            # image panel objects (all of them) alongside the history, so
            # that reopening restores both the data objects and the history
            # that references them. Each section is read back by its own
            # H5_PREFIX key, so the write order is not significant.
            panel.mainwindow.signalpanel.serialize_to_hdf5(writer)
            panel.mainwindow.imagepanel.serialize_to_hdf5(writer)
            panel.serialize_to_hdf5(writer)
    return True


def open_dlhist_file(panel: HistoryPanel, filename: str | None = None) -> bool:
    """Open a standalone ``.dlhist`` file into the History Panel.

    Args:
        filename: History filename. If None, a file dialog is opened.

    Returns:
        True if the history was loaded, False if the operation was canceled.
    """
    if filename is None:
        basedir = Conf.main.base_dir.get()
        with save_restore_stds():
            filename, _filt = getopenfilename(
                panel, _("Open history file"), basedir, panel.FILE_FILTERS
            )
    if not filename:
        return False
    with qt_try_loadsave_file(panel.parentWidget(), filename, "load"):
        Conf.main.base_dir.set(filename)
        with NativeH5Reader(filename) as reader:
            # A panel-contained .dlhist file stores the signal and image
            # panel objects in addition to the history sessions. The way
            # they are restored depends on whether the workspace is already
            # in use (data objects OR history): a pristine workspace is
            # loaded directly while preserving UUIDs, otherwise the file
            # is imported as new groups/sessions.
            workspace_in_use = (
                panel.mainwindow.signalpanel.objmodel.get_object_ids()
                or panel.mainwindow.imagepanel.objmodel.get_object_ids()
                or bool(panel.history_sessions)
            )
            if workspace_in_use:
                # Workspace not empty: import the objects into new groups
                # with fresh UUIDs and append the history as new sessions
                # whose references are remapped to the imported objects.
                panel.import_dlhist_into_new_session(reader)
            else:
                # Workspace empty: load directly, preserving original UUIDs
                # (reset_all=True) so that history references stay valid.
                panel.mainwindow.signalpanel.deserialize_from_hdf5(
                    reader, reset_all=True
                )
                panel.mainwindow.imagepanel.deserialize_from_hdf5(
                    reader, reset_all=True
                )
                panel.deserialize_from_hdf5(reader)
    return True


def import_dlhist_into_new_session(panel: HistoryPanel, reader: NativeH5Reader) -> None:
    """Import a ``.dlhist`` file into new groups and new history sessions.

    Used when the workspace already contains objects: the file's signal and
    image objects are imported into fresh groups with regenerated UUIDs, and
    the history sessions are appended as new independent sessions whose action
    references are remapped to the freshly imported objects.

    Args:
        reader: HDF5 reader positioned on a ``.dlhist`` file.
    """
    panel_map = {
        "signal": panel.mainwindow.signalpanel,
        "image": panel.mainwindow.imagepanel,
    }
    uuid_remap: dict[str, dict[str, str]] = {}
    imported_by_pstr: dict[str, list] = {}
    # 1. Import objects from each panel (each panel is read by its own
    #    H5_PREFIX key). Read each object preserving its original UUID to
    #    capture the old->new mapping, then assign a fresh UUID so that the
    #    imported objects keep an independent identity.
    for pstr, data_panel in panel_map.items():
        uuid_remap[pstr] = {}
        imported: list = []
        imported_by_pstr[pstr] = imported
        if data_panel.H5_PREFIX not in reader.h5:
            continue
        with reader.group(data_panel.H5_PREFIX):
            for name in reader.h5.get(data_panel.H5_PREFIX, []):
                with reader.group(name):
                    group = data_panel.add_group("")
                    with reader.group("title"):
                        group.title = reader.read_str()
                    for obj_name in reader.h5.get(f"{data_panel.H5_PREFIX}/{name}", []):
                        obj = data_panel.deserialize_object_from_hdf5(
                            reader, obj_name, reset_all=True
                        )
                        old_uuid = get_uuid(obj)
                        new_uuid = str(uuid4())
                        # SignalObj/ImageObj store UUID via metadata option
                        try:
                            obj.set_metadata_option("uuid", new_uuid)
                        except AttributeError:
                            obj.uuid = new_uuid
                        uuid_remap[pstr][old_uuid] = new_uuid
                        data_panel.add_object(obj, get_uuid(group), set_current=False)
                        imported.append(obj)
                    data_panel.selection_changed()
    # 2. Remap source UUIDs in imported objects' processing_parameters so
    #    that reprocessing in the Processing tab uses the imported sources,
    #    not the originals (same logic as duplicate_selected_entries).
    for pstr, objs in imported_by_pstr.items():
        pmap = uuid_remap.get(pstr, {})
        if not pmap:
            continue
        for obj in objs:
            try:
                pp_dict = obj.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
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
                    obj.set_metadata_option(PROCESSING_PARAMETERS_OPTION, pp.to_dict())
                except (AttributeError, ValueError):
                    pass
    # 3. Import history sessions as new independent sessions whose captured
    #    UUIDs are remapped to the imported objects.
    if panel.H5_PREFIX not in reader.h5:
        return
    sessions = reader.read_object_list(panel.H5_PREFIX, HistorySession) or []
    imported_suffix = _("Imported")
    new_sessions: list[HistorySession] = []
    for session in sessions:
        panel.session_increment += 1
        title = f"{session.title} {imported_suffix}"
        new_session = session.copy_with_uuid_remap(title=title, uuid_remap=uuid_remap)
        new_session.number = panel.session_increment
        new_sessions.append(new_session)
        # Register output mappings for imported actions so that
        # resolve_target_outputs / get_downstream_actions work.
        for action in new_session.actions:
            if action.output_uuids:
                panel.action_output_uuids[action.uuid] = list(action.output_uuids)
                for out_uuid in action.output_uuids:
                    panel.output_to_action[out_uuid] = action.uuid
    panel.history_sessions.extend(new_sessions)
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()


def refresh_compatibility_items(panel: HistoryPanel, *args: Any) -> None:
    """Refresh action item compatibility markers in the tree."""
    del args
    panel.tree.update_compatibility_states(panel.history_sessions, panel.mainwindow)


def serialize_to_hdf5(panel: HistoryPanel, writer: NativeH5Writer) -> None:
    """Serialize whole panel to a HDF5 file

    Args:
        writer: HDF5 writer
    """
    writer.write_object_list(panel.history_sessions, panel.H5_PREFIX)


def deserialize_from_hdf5(
    panel: HistoryPanel, reader: NativeH5Reader, reset_all: bool = False
) -> None:
    """Deserialize whole panel from a HDF5 file

    Args:
        reader: HDF5 reader
        reset_all: Unused (kept for compatibility with panel API)
    """
    del reset_all  # required by the polymorphic panel API; unused here
    if panel.H5_PREFIX not in reader.h5:
        panel.history_sessions = []
        panel.session_increment = 0
        panel.tree.populate_tree(panel.history_sessions)
        panel.update_actions_state()
        return
    panel.history_sessions = (
        reader.read_object_list(panel.H5_PREFIX, HistorySession) or []
    )
    if panel.history_sessions:
        panel.session_increment = panel.history_sessions[-1].number
    # Rebuild the bijective mapping from the loaded actions. Legacy
    # (v1) actions have empty ``output_uuids`` and contribute nothing
    # to the index — the heuristic fallback handles them.
    panel.action_output_uuids = {}
    panel.output_to_action = {}
    for session in panel.history_sessions:
        for action in session.actions:
            if action.output_uuids:
                panel.action_output_uuids[action.uuid] = list(action.output_uuids)
                for out_uuid in action.output_uuids:
                    panel.output_to_action[out_uuid] = action.uuid
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.update_actions_state()
