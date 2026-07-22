# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel HDF5 import/export and persistence helpers."""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass
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
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history import HistoryPanel


@dataclass
class HistoryImportRegistry:
    """Imported objects and their old-to-new UUID mappings."""

    panel_map: dict[str, BaseDataPanel]
    uuid_remap: dict[str, dict[str, str]]
    imported_by_pstr: dict[str, list[Any]]


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


def create_import_registry(panel: HistoryPanel) -> HistoryImportRegistry:
    """Create empty per-panel object and UUID registries."""
    panel_map = {
        "signal": panel.mainwindow.signalpanel,
        "image": panel.mainwindow.imagepanel,
    }
    return HistoryImportRegistry(
        panel_map=panel_map,
        uuid_remap={panel_str: {} for panel_str in panel_map},
        imported_by_pstr={panel_str: [] for panel_str in panel_map},
    )


def assign_imported_uuid(obj: Any) -> tuple[str, str]:
    """Assign a fresh UUID to an imported object and return both UUIDs."""
    old_uuid = get_uuid(obj)
    new_uuid = str(uuid4())
    try:
        obj.set_metadata_option("uuid", new_uuid)
    except AttributeError:
        obj.uuid = new_uuid
    return old_uuid, new_uuid


def read_imported_group(
    reader: NativeH5Reader,
    data_panel: BaseDataPanel,
    panel_str: str,
    group_name: str,
    registry: HistoryImportRegistry,
) -> None:
    """Read one object group and register its regenerated UUIDs."""
    with reader.group(group_name):
        group = data_panel.add_group("")
        with reader.group("title"):
            group.title = reader.read_str()
        path = f"{data_panel.H5_PREFIX}/{group_name}"
        for object_name in reader.h5.get(path, []):
            obj = data_panel.deserialize_object_from_hdf5(
                reader, object_name, reset_all=True
            )
            old_uuid, new_uuid = assign_imported_uuid(obj)
            registry.uuid_remap[panel_str][old_uuid] = new_uuid
            data_panel.add_object(obj, get_uuid(group), set_current=False)
            registry.imported_by_pstr[panel_str].append(obj)
        data_panel.selection_changed()


def read_imported_objects(
    reader: NativeH5Reader, registry: HistoryImportRegistry
) -> None:
    """Read signal and image payloads into fresh object groups."""
    for panel_str, data_panel in registry.panel_map.items():
        if data_panel.H5_PREFIX not in reader.h5:
            continue
        with reader.group(data_panel.H5_PREFIX):
            for group_name in reader.h5.get(data_panel.H5_PREFIX, []):
                read_imported_group(reader, data_panel, panel_str, group_name, registry)


def remap_imported_object_sources(obj: Any, uuid_remap: dict[str, str]) -> None:
    """Remap processing source UUIDs stored on one imported object."""
    try:
        parameters_dict = obj.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
    except (AttributeError, ValueError):
        return
    if not parameters_dict:
        return
    try:
        parameters = ProcessingParameters.from_dict(parameters_dict)
    except (TypeError, ValueError, AttributeError):
        return
    changed = False
    if parameters.source_uuid is not None and parameters.source_uuid in uuid_remap:
        parameters.source_uuid = uuid_remap[parameters.source_uuid]
        changed = True
    if parameters.source_uuids is not None:
        new_sources = [uuid_remap.get(uuid, uuid) for uuid in parameters.source_uuids]
        if new_sources != parameters.source_uuids:
            parameters.source_uuids = new_sources
            changed = True
    if changed:
        try:
            obj.set_metadata_option(PROCESSING_PARAMETERS_OPTION, parameters.to_dict())
        except (AttributeError, ValueError):
            pass


def remap_imported_sources(registry: HistoryImportRegistry) -> None:
    """Remap processing sources for all imported objects."""
    for panel_str, objects in registry.imported_by_pstr.items():
        uuid_remap = registry.uuid_remap.get(panel_str, {})
        if not uuid_remap:
            continue
        for obj in objects:
            remap_imported_object_sources(obj, uuid_remap)


def assemble_imported_sessions(
    panel: HistoryPanel,
    reader: NativeH5Reader,
    registry: HistoryImportRegistry,
) -> list[HistorySession] | None:
    """Read history payload and clone sessions with remapped UUIDs."""
    if panel.H5_PREFIX not in reader.h5:
        return None
    sessions = reader.read_object_list(panel.H5_PREFIX, HistorySession) or []
    imported_suffix = _("Imported")
    new_sessions: list[HistorySession] = []
    for session in sessions:
        panel.navigation.session_increment += 1
        title = f"{session.title} {imported_suffix}"
        new_session = session.copy_with_uuid_remap(
            title=title, uuid_remap=registry.uuid_remap
        )
        new_session.number = panel.navigation.session_increment
        new_sessions.append(new_session)
    return new_sessions


def register_imported_outputs(
    panel: HistoryPanel, sessions: list[HistorySession]
) -> None:
    """Register action output mappings for imported sessions."""
    for session in sessions:
        for action in session.actions:
            if action.output_uuids:
                panel.runtime.objects.register_action_outputs(
                    action, action.output_uuids
                )


def update_imported_history_ui(
    panel: HistoryPanel, sessions: list[HistorySession]
) -> None:
    """Append imported sessions and refresh history presentation."""
    panel.history_sessions.extend(sessions)
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()


def import_dlhist_into_new_session(panel: HistoryPanel, reader: NativeH5Reader) -> None:
    """Import a ``.dlhist`` file into new groups and new history sessions.

    Args:
        reader: HDF5 reader positioned on a ``.dlhist`` file.
    """
    registry = create_import_registry(panel)
    read_imported_objects(reader, registry)
    remap_imported_sources(registry)
    sessions = assemble_imported_sessions(panel, reader, registry)
    if sessions is None:
        return
    register_imported_outputs(panel, sessions)
    update_imported_history_ui(panel, sessions)


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
    panel.runtime.objects.clear_output_mappings()
    if panel.H5_PREFIX not in reader.h5:
        panel.history_sessions = []
        panel.navigation.session_increment = 0
        panel.tree.populate_tree(panel.history_sessions)
        panel.ui.update_actions_state()
        return
    panel.history_sessions = (
        reader.read_object_list(panel.H5_PREFIX, HistorySession) or []
    )
    if panel.history_sessions:
        panel.navigation.session_increment = panel.history_sessions[-1].number
    # Rebuild the bijective mapping from the loaded actions. Legacy
    # (v1) actions have empty ``output_uuids`` and contribute nothing
    # to the index — the heuristic fallback handles them.
    for session in panel.history_sessions:
        for action in session.actions:
            if action.output_uuids:
                panel.runtime.objects.register_action_outputs(
                    action, action.output_uuids
                )
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()
