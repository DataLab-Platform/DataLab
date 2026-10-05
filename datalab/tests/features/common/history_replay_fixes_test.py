# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Regression tests for History panel replay fixes.

Covers cross-panel output routing, synthetic session heads, file-save
confirmation and load-action replay (plan ``history-replay-fixes``).
"""

from __future__ import annotations

import os
import os.path as osp
from unittest.mock import patch

import numpy as np
import sigima.objects
import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal, create_sincos_image

from datalab.config import _
from datalab.env import execenv
from datalab.gui import historytools_ops as htools
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    add_paracetamol_signals,
    select_tree_entry,
    select_tree_session,
)


def record_line_profile(win) -> tuple:
    """Record an image line-profile compute and return (action, output_uuid).

    Args:
        win: DataLab main window with history recording enabled

    Returns:
        Recorded cross-panel compute action and its output signal UUID
    """
    history, ipanel = win.historypanel, win.imagepanel
    ipanel.add_object(create_sincos_image())
    ipanel.objview.select_objects([1])
    param = sigima.params.LineProfileParam.create(
        direction="horizontal", row=100, col=100
    )
    ipanel.processor.run_feature("line_profile", param)
    action = history[len(history)]
    return action, action.output_uuids[0]


def test_replay_cross_panel_output_stays_in_destination_panel() -> None:
    """Keep cross-panel compute outputs in the destination panel on replay."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        action, output_uuid = record_line_profile(win)
        spanel, ipanel = win.signalpanel, win.imagepanel
        assert spanel.objmodel.has_uuid(output_uuid)
        assert not ipanel.objmodel.has_uuid(output_uuid)
        image_count = len(ipanel.objmodel)
        signal_count = len(spanel.objmodel)
        for _replay_nb in range(2):
            hireplay.replay_actions(history, [action], prompt=False)
            assert action.is_stale is False
            assert spanel.objmodel.has_uuid(output_uuid)
            assert not ipanel.objmodel.has_uuid(output_uuid)
            assert len(ipanel.objmodel) == image_count
            assert len(spanel.objmodel) == signal_count


def test_replay_recreates_deleted_cross_panel_output_in_destination_panel() -> None:
    """Re-create a deleted cross-panel output in the destination panel."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        action, output_uuid = record_line_profile(win)
        spanel, ipanel = win.signalpanel, win.imagepanel
        expected_data = spanel.objmodel[output_uuid].xydata.copy()
        image_count = len(ipanel.objmodel)
        spanel.objview.select_objects([output_uuid])
        spanel.remove_object(force=True)
        assert not spanel.objmodel.has_uuid(output_uuid)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, [action], prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert action.is_stale is False
        assert spanel.objmodel.has_uuid(output_uuid)
        assert not ipanel.objmodel.has_uuid(output_uuid)
        assert len(ipanel.objmodel) == image_count
        recreated = spanel.objmodel[output_uuid]
        assert np.array_equal(recreated.xydata, expected_data)


def test_failed_compute_replay_leaves_no_temporary_objects() -> None:
    """Detach fresh temporaries when a compute raises mid-batch on replay."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 2)
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create()
        )
        action = history[len(history)]
        object_count = len(panel.objmodel)
        original_execute = hrec.execute_compute_via_ui

        def failing_execute(panel_data, act, obj2_uuids) -> None:
            """Insert the whole batch of fresh outputs, then fail: simulates
            a compute raising after some objects were already created."""
            original_execute(panel_data, act, obj2_uuids)
            raise RuntimeError("simulated mid-batch failure")

        with patch.object(hrec, "execute_compute_via_ui", failing_execute):
            assert hrec.recompute_action_in_place(history, action) is False

        assert action.is_stale is True
        assert len(panel.objmodel) == object_count
        history.runtime.execution.cascade_warnings.clear()


def build_synthetic_head_chain(win) -> list:
    """Record a creation + two computes, then clear the head kwargs.

    Mirrors the synthetic session head produced by *Duplicate chain*
    (``new_object`` UI action with empty kwargs).

    Args:
        win: DataLab main window with history recording enabled

    Returns:
        The three recorded actions in session order (head first)
    """
    history, panel = win.historypanel, win.signalpanel
    panel.new_object(edit=False)
    head = history[len(history)]
    assert head.method_name == "new_object"
    panel.objview.select_objects(head.output_uuids)
    panel.processor.run_feature(
        sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
    )
    first = history[len(history)]
    panel.objview.select_objects(first.output_uuids)
    panel.processor.run_feature(sips.derivative)
    second = history[len(history)]
    head.kwargs.clear()  # Simulate the Duplicate-chain synthetic head
    return [head, first, second]


def test_replay_synthetic_head_without_param_is_noop_success() -> None:
    """Treat a parameterless creation head with live outputs as a no-op."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        actions = build_synthetic_head_chain(win)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, actions, prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert all(not action.is_stale for action in actions)


def test_replay_synthetic_head_with_deleted_output_warns() -> None:
    """Warn and keep the chain stale when the head object was deleted."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        actions = build_synthetic_head_chain(win)
        head = actions[0]
        deleted_uuids = [
            output_uuid for action in actions for output_uuid in action.output_uuids
        ]
        panel.objview.select_objects(deleted_uuids)
        panel.remove_object(force=True)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, actions, prompt=False)

        warnings = history.runtime.execution.cascade_warnings
        assert warnings
        assert any(head.title in warning for warning in warnings)
        assert all(action.is_stale for action in actions)
        history.runtime.execution.cascade_warnings.clear()


def record_file_save(win, filename: str):
    """Record a ``save_to_files`` action writing one signal to ``filename``.

    Args:
        win: DataLab main window with history recording enabled
        filename: Destination file name

    Returns:
        Recorded file-save UI action
    """
    history, panel = win.historypanel, win.signalpanel
    add_paracetamol_signals(panel, 1)
    panel.objview.select_objects([1])
    panel.save_to_files([filename])
    action = history[len(history)]
    assert action.method_name == "save_to_files"
    assert osp.isfile(filename)
    return action


def test_replay_skips_file_save_without_confirmation(tmp_path) -> None:
    """Skip file-save actions on unattended replay without accept_dialogs."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        filename = str(tmp_path / "signal.csv")
        action = record_file_save(win, filename)
        os.remove(filename)
        assert execenv.unattended and not execenv.accept_dialogs

        hireplay.replay_actions(history, [action], prompt=False)

        assert not osp.isfile(filename)
        assert action.is_stale is False


def test_replay_file_save_with_accept_dialogs(tmp_path) -> None:
    """Replay file-save actions when accept_dialogs is enabled."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        filename = str(tmp_path / "signal.csv")
        action = record_file_save(win, filename)
        os.remove(filename)
        saved_accept_dialogs = execenv.accept_dialogs
        execenv.accept_dialogs = True
        try:
            hireplay.replay_actions(history, [action], prompt=False)
        finally:
            execenv.accept_dialogs = saved_accept_dialogs

        assert osp.isfile(filename)
        assert action.is_stale is False


def populate_signal_directory(win, directory, layout: dict[str, list[str]]) -> None:
    """Write round-trip-compatible signal files into ``directory``.

    Signals are saved through the panel I/O registry so the written files are
    guaranteed to load back. No history entry is recorded (record mode off).

    Args:
        win: DataLab main window (record mode must be disabled)
        directory: Base directory (``pathlib.Path``)
        layout: Mapping of subdirectory name ("" for the base directory) to
         file names
    """
    panel = win.signalpanel
    assert not win.historypanel.record_mode_enabled
    add_paracetamol_signals(panel, 1)
    panel.objview.select_objects([1])
    for subdir_name, filenames in layout.items():
        subdir = directory / subdir_name if subdir_name else directory
        subdir.mkdir(exist_ok=True)
        for filename in filenames:
            panel.save_to_files([str(subdir / filename)])
    panel.remove_object(force=True)
    assert len(panel.objmodel) == 0


def test_replay_load_from_directory_reloads_deleted_objects(tmp_path) -> None:
    """Reload deleted objects (and groups) by replaying a directory load."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(
            win,
            tmp_path,
            {"suba": ["s1.csv", "s2.csv"], "subb": ["s3.csv", "s4.csv"]},
        )
        history.toggle_record_mode(True)
        objs = panel.load_from_directory(str(tmp_path))
        assert len(objs) == 4
        action = history[len(history)]
        assert action.method_name == "load_from_directory"
        assert len(action.output_uuids) == 4
        group_titles = [group.title for group in panel.objmodel.get_groups()]
        assert "suba" in group_titles and "subb" in group_titles
        # Delete every loaded object
        history.toggle_record_mode(False)
        panel.objview.select_objects(panel.objmodel.get_object_ids())
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0
        history.toggle_record_mode(True)

        hireplay.replay_actions(history, [action], prompt=False)

        assert len(panel.objmodel) == 4
        assert action.is_stale is False
        # Outputs re-bound to the freshly loaded objects
        assert len(action.output_uuids) == 4
        assert all(
            panel.objmodel.has_uuid(output_uuid) for output_uuid in action.output_uuids
        )
        # Group structure (one group per subdirectory) is re-created
        reloaded_groups = {
            group.title: len(group.get_object_ids())
            for group in panel.objmodel.get_groups()
            if group.get_object_ids()
        }
        assert reloaded_groups.get("suba") == 2
        assert reloaded_groups.get("subb") == 2


def test_replay_load_reloads_after_recorded_deletion(tmp_path) -> None:
    """Reload deleted objects when the deletion itself was recorded.

    Faithful GUI scenario: record mode stays ON during the deletion (so a
    ``remove_object`` UI action is recorded and the reconnection machinery
    runs), and the replay goes through the real GUI entry point
    ``replay_restore_actions`` (Replay button / double-click) with the load
    action selected in the tree.
    """
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv", "s2.csv"]})
        history.toggle_record_mode(True)
        fnames = [str(tmp_path / "s1.csv"), str(tmp_path / "s2.csv")]
        objs = panel.load_from_files(fnames)
        assert len(objs) == 2
        action = history[len(history)]
        assert action.method_name == "load_from_files"
        # Record mode stays ON: the deletion is recorded as a UI action
        panel.objview.select_objects(panel.objmodel.get_object_ids())
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0
        remove_action = history[len(history)]
        assert remove_action.method_name == "remove_object"

        select_tree_entry(history, action.uuid)
        history.replay_restore_actions(restore_selection=False)

        assert len(panel.objmodel) == 2
        assert action.is_stale is False
        assert len(action.output_uuids) == 2
        assert all(
            panel.objmodel.has_uuid(output_uuid) for output_uuid in action.output_uuids
        )


def test_replay_directory_load_reloads_after_recorded_deletion(tmp_path) -> None:
    """Directory-load variant of the recorded-deletion replay scenario."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(
            win, tmp_path, {"suba": ["s1.csv", "s2.csv"], "subb": ["s3.csv"]}
        )
        history.toggle_record_mode(True)
        objs = panel.load_from_directory(str(tmp_path))
        assert len(objs) == 3
        action = history[len(history)]
        assert action.method_name == "load_from_directory"
        # Record mode stays ON: the deletion is recorded as a UI action
        panel.objview.select_objects(panel.objmodel.get_object_ids())
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0

        select_tree_entry(history, action.uuid)
        history.replay_restore_actions(restore_selection=False)

        assert len(panel.objmodel) == 3
        assert action.is_stale is False
        assert len(action.output_uuids) == 3
        assert all(
            panel.objmodel.has_uuid(output_uuid) for output_uuid in action.output_uuids
        )


def test_replay_session_fallback_after_recorded_deletion(tmp_path) -> None:
    """Replay with no tree selection after a recorded deletion.

    With nothing selected in the tree, ``replay_restore_actions`` targets the
    last session, which contains both the load action and the recorded
    ``remove_object`` action. The load must be replayed and the destructive
    action skipped (its captured UUIDs no longer exist), leaving the reloaded
    objects in place.
    """
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv", "s2.csv"]})
        history.toggle_record_mode(True)
        fnames = [str(tmp_path / "s1.csv"), str(tmp_path / "s2.csv")]
        objs = panel.load_from_files(fnames)
        assert len(objs) == 2
        action = history[len(history)]
        # Record mode stays ON: the deletion is recorded as a UI action
        panel.objview.select_objects(panel.objmodel.get_object_ids())
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0

        history.tree.clearSelection()
        history.replay_restore_actions(restore_selection=False)

        assert len(panel.objmodel) == 2
        assert all(
            panel.objmodel.has_uuid(output_uuid) for output_uuid in action.output_uuids
        )


def test_replay_load_skipped_when_outputs_still_exist(tmp_path) -> None:
    """Skip a load-action replay when every loaded object still exists."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv", "s2.csv"]})
        history.toggle_record_mode(True)
        objs = panel.load_from_directory(str(tmp_path))
        assert len(objs) == 2
        action = history[len(history)]
        assert action.method_name == "load_from_directory"
        object_count = len(panel.objmodel)
        group_count = len(panel.objmodel.get_groups())
        output_uuids_before = list(action.output_uuids)

        hireplay.replay_actions(history, [action], prompt=False)

        assert len(panel.objmodel) == object_count
        assert len(panel.objmodel.get_groups()) == group_count
        assert action.output_uuids == output_uuids_before
        assert action.is_stale is False


def test_replay_load_legacy_add_objects_false_self_heals(tmp_path) -> None:
    """Self-heal legacy load entries recorded with ``add_objects=False``."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv", "s2.csv"]})
        history.toggle_record_mode(True)
        fnames = [str(tmp_path / "s1.csv"), str(tmp_path / "s2.csv")]
        objs = panel.load_from_files(fnames)
        assert len(objs) == 2
        action = history[len(history)]
        assert action.method_name == "load_from_files"
        # Simulate a legacy entry recorded by the old ``load_from_directory``
        action.kwargs["add_objects"] = False
        # Delete every loaded object
        history.toggle_record_mode(False)
        panel.objview.select_objects(panel.objmodel.get_object_ids())
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0
        history.toggle_record_mode(True)

        hireplay.replay_actions(history, [action], prompt=False)

        assert len(panel.objmodel) == 2
        assert action.is_stale is False
        assert action.kwargs["add_objects"] is True
        assert all(
            panel.objmodel.has_uuid(output_uuid) for output_uuid in action.output_uuids
        )


def write_unreadable_files(directory, names: list[str]) -> list[str]:
    """Write files that no signal reader can load into ``directory``.

    Args:
        directory: Base directory (``pathlib.Path``)
        names: File names to create

    Returns:
        Full paths of the created files
    """
    fnames = []
    for name in names:
        path = directory / name
        path.write_bytes(b"\x89PNG\r\n\x1a\nnot actually loadable")
        fnames.append(str(path))
    return fnames


def test_load_from_files_without_loadable_file_records_no_entry(tmp_path) -> None:
    """Discard the load entry when no file produced any object."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        fnames = write_unreadable_files(tmp_path, ["i1.png", "i2.png"])
        objs = panel.load_from_files(fnames, ignore_errors=True)
        assert objs == []
        assert len(history) == 0


def test_load_from_directory_without_loadable_file_records_no_entry(tmp_path) -> None:
    """Discard the directory-load entry when nothing could be loaded."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        write_unreadable_files(tmp_path, ["i1.png", "i2.png"])
        objs = panel.load_from_directory(str(tmp_path))
        assert objs == []
        assert len(history) == 0


def test_load_from_files_partial_failure_updates_entry(tmp_path) -> None:
    """Reflect the actually loaded files in the recorded load entry."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv", "s2.csv"]})
        bad = write_unreadable_files(tmp_path, ["i1.png", "i2.png"])
        good = [str(tmp_path / "s1.csv"), str(tmp_path / "s2.csv")]
        history.toggle_record_mode(True)
        objs = panel.load_from_files(sorted(good + bad), ignore_errors=True)
        assert len(objs) == 2
        assert len(history) == 1
        action = history[len(history)]
        assert action.method_name == "load_from_files"
        assert action.title == _("Load from %d files") % 2
        assert action.kwargs["filenames"] == sorted(good)
        assert len(action.output_uuids) == 2


def test_load_from_files_single_success_updates_entry(tmp_path) -> None:
    """Use the single-file title when only one file could be loaded."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        populate_signal_directory(win, tmp_path, {"": ["s1.csv"]})
        bad = write_unreadable_files(tmp_path, ["i1.png"])
        good = str(tmp_path / "s1.csv")
        history.toggle_record_mode(True)
        objs = panel.load_from_files(sorted([good] + bad), ignore_errors=True)
        assert len(objs) == 1
        assert len(history) == 1
        action = history[len(history)]
        assert action.title == _('Load "%s"') % "s1.csv"
        assert action.kwargs["filenames"] == [good]


def record_group_gaussian_filter(win) -> tuple:
    """Record a 1-to-1 compute applied to a whole image group.

    Args:
        win: DataLab main window with history recording enabled

    Returns:
        Recorded compute action and the list of source image UUIDs
    """
    history, panel = win.historypanel, win.imagepanel
    for _index in range(3):
        panel.add_object(create_sincos_image())
    source_uuids = panel.objmodel.get_object_ids()[-3:]
    group = panel.objmodel.get_group_from_object(panel.objmodel[source_uuids[0]])
    panel.objview.select_groups([get_uuid(group)])
    panel.processor.run_feature(
        "gaussian_filter", sigima.params.GaussianParam.create(sigma=2.0)
    )
    action = history[len(history)]
    assert action.pattern == "1_to_1"
    assert len(action.output_uuids) == 3
    return action, source_uuids


def test_replay_recreates_single_deleted_1_to_1_output() -> None:
    """Re-create one deleted output of a group-wide 1-to-1 compute."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.imagepanel
        history.toggle_record_mode(True)
        action, _source_uuids = record_group_gaussian_filter(win)
        recorded = list(action.output_uuids)
        deleted_uuid = recorded[1]
        survivors = [uuid for uuid in recorded if uuid != deleted_uuid]
        survivor_group_id = panel.objmodel.get_object_group_id(
            panel.objmodel[survivors[0]]
        )
        expected_data = panel.objmodel[deleted_uuid].data.copy()
        object_count = len(panel.objmodel)
        survivor_identities = {uuid: id(panel.objmodel[uuid]) for uuid in survivors}
        panel.objview.select_objects([deleted_uuid])
        panel.remove_object(force=True)
        assert not panel.objmodel.has_uuid(deleted_uuid)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, [action], prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert action.is_stale is False
        assert len(panel.objmodel) == object_count
        assert action.output_uuids == recorded
        # The deleted output is re-created under its recorded UUID, next to
        # its surviving siblings
        assert panel.objmodel.has_uuid(deleted_uuid)
        recreated = panel.objmodel[deleted_uuid]
        assert np.array_equal(recreated.data, expected_data)
        assert panel.objmodel.get_object_group_id(recreated) == survivor_group_id
        # Surviving outputs are updated in place (identity preserved)
        for uuid in survivors:
            assert id(panel.objmodel[uuid]) == survivor_identities[uuid]


def test_replay_recreates_all_deleted_1_to_1_outputs() -> None:
    """Re-create every deleted output of a group-wide 1-to-1 compute."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.imagepanel
        history.toggle_record_mode(True)
        action, source_uuids = record_group_gaussian_filter(win)
        recorded = list(action.output_uuids)
        expected_data = [panel.objmodel[uuid].data.copy() for uuid in recorded]
        object_count = len(panel.objmodel)
        panel.objview.select_objects(recorded)
        panel.remove_object(force=True)
        assert all(not panel.objmodel.has_uuid(uuid) for uuid in recorded)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, [action], prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert action.is_stale is False
        assert len(panel.objmodel) == object_count
        assert action.output_uuids == recorded
        for uuid, data in zip(recorded, expected_data):
            assert panel.objmodel.has_uuid(uuid)
            assert np.array_equal(panel.objmodel[uuid].data, data)
        # Sources are untouched
        assert all(panel.objmodel.has_uuid(uuid) for uuid in source_uuids)


def test_replay_recreates_deleted_pairwise_n_to_1_output() -> None:
    """Re-create a deleted output of a pairwise n-to-1 compute."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        group_a = panel.add_group("group_a")
        group_b = panel.add_group("group_b")
        for group in (group_a, group_b):
            for _index in range(2):
                panel.add_object(create_paracetamol_signal(), group_id=get_uuid(group))
        panel.objview.select_groups([get_uuid(group_a), get_uuid(group_b)])
        feature = panel.processor.get_feature("average")
        panel.processor.compute_n_to_1(feature.function, edit=False, pairwise=True)
        action = history[len(history)]
        assert action.pattern == "n_to_1"
        assert action.kwargs.get("pairwise") is True
        recorded = list(action.output_uuids)
        assert len(recorded) == 2
        deleted_uuid = recorded[0]
        expected_data = panel.objmodel[deleted_uuid].xydata.copy()
        object_count = len(panel.objmodel)
        panel.objview.select_objects([deleted_uuid])
        panel.remove_object(force=True)
        assert not panel.objmodel.has_uuid(deleted_uuid)

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, [action], prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert action.is_stale is False
        assert len(panel.objmodel) == object_count
        assert action.output_uuids == recorded
        assert panel.objmodel.has_uuid(deleted_uuid)
        assert np.array_equal(panel.objmodel[deleted_uuid].xydata, expected_data)


def assert_no_original_reference(action, original_uuids: set[str]) -> None:
    """Assert that a duplicated action references no original object UUID.

    Args:
        action: Duplicated history action to inspect
        original_uuids: UUIDs of the original (source) chain objects
    """
    captured = set(action.state.selection.get("signal", []))
    assert not captured & original_uuids, action.title
    obj2 = action.kwargs.get("obj2_uuids") or []
    if isinstance(obj2, str):
        obj2 = [obj2]
    assert not set(obj2) & original_uuids, action.title
    assert not set(action.output_uuids) & original_uuids, action.title


def test_duplicated_chain_independent_after_original_group_deletion() -> None:
    """Keep a duplicated chain fully independent after deleting the originals.

    User scenario: build a chain (recorded creation + two computes), duplicate
    it from the History panel, re-run the duplicated chain with different
    parameters, then delete the ORIGINAL objects group from the Signal panel.
    Replaying each session afterwards must only touch that session's own
    objects: the duplicate is a fully standalone deep copy.
    """
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        # Original chain: recorded creation + two 1-to-1 computes
        panel.new_object(param=sigima.objects.GaussParam(), edit=False)
        source_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
        )
        gaussian = history[len(history)]
        panel.objview.select_objects(gaussian.output_uuids)
        panel.processor.run_feature(sips.derivative)
        original = history.history_sessions[-1]
        original_uuids = [
            uuid for action in original.actions for uuid in action.output_uuids
        ]
        assert original_uuids[0] == source_uuid
        # Duplicate the whole session (History panel "Duplicate" command)
        select_tree_session(history, original)
        htools.duplicate_selected_entries(history)
        duplicate = history.history_sessions[-1]
        assert duplicate is not original
        clone_uuids = [
            uuid for action in duplicate.actions for uuid in action.output_uuids
        ]
        assert set(clone_uuids).isdisjoint(original_uuids)
        # No duplicated action nor clone object references an original UUID
        for action in duplicate.actions:
            assert_no_original_reference(action, set(original_uuids))
        for uuid in clone_uuids:
            parameters = extract_processing_parameters(panel.objmodel[uuid])
            if parameters is None:
                continue
            sources = list(parameters.source_uuids or [])
            if parameters.source_uuid is not None:
                sources.append(parameters.source_uuid)
            assert not set(sources) & set(original_uuids), uuid
        # Re-run the duplicated chain with a different parameter
        dup_gaussian = next(
            action
            for action in duplicate.actions
            if action.func_name == "gaussian_filter"
        )
        dup_gaussian.kwargs["param"] = sigima.params.GaussianParam.create(sigma=4.0)
        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, list(duplicate.actions), prompt=False)
        # Delete the ORIGINAL objects group from the Signal panel
        original_group_id = panel.objmodel.get_object_group_id(
            panel.objmodel[source_uuid]
        )
        panel.objview.select_groups([original_group_id])
        panel.remove_object(force=True)
        assert all(not panel.objmodel.has_uuid(uuid) for uuid in original_uuids)
        clone_data = {uuid: panel.objmodel[uuid].xydata.copy() for uuid in clone_uuids}
        # The deletion must not rewire the duplicated session onto originals
        for action in duplicate.actions:
            if action.kind == action.KIND_COMPUTE:
                assert_no_original_reference(action, set(original_uuids))
        # Replay the ORIGINAL session: it must only touch its own objects
        select_tree_session(history, original)
        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_restore_actions(history)
        for uuid in clone_uuids:
            assert panel.objmodel.has_uuid(uuid), "clone deleted by original replay"
            assert np.array_equal(panel.objmodel[uuid].xydata, clone_data[uuid]), (
                "original session replay modified a clone object"
            )
        for uuid in original_uuids:
            assert panel.objmodel.has_uuid(uuid), (
                "original session replay did not recreate its own objects"
            )
        original_data = {
            uuid: panel.objmodel[uuid].xydata.copy() for uuid in original_uuids
        }
        # Replay the DUPLICATED session: original objects must stay untouched
        select_tree_session(history, duplicate)
        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_restore_actions(history)
        for uuid in original_uuids:
            assert panel.objmodel.has_uuid(uuid), (
                "duplicate session replay deleted an original object"
            )
            assert np.array_equal(panel.objmodel[uuid].xydata, original_data[uuid]), (
                "duplicate session replay modified an original object"
            )
        for uuid in clone_uuids:
            assert np.array_equal(panel.objmodel[uuid].xydata, clone_data[uuid])


def test_replay_skips_recorded_deletion_of_another_session_objects() -> None:
    """Skip a recorded deletion targeting objects owned by another session.

    Variant of the duplication scenario where the duplicated session is the
    active recording session when the ORIGINAL objects group is deleted: the
    ``remove_object`` action lands in the duplicated session but captures the
    original session's objects. Replaying the duplicated session must never
    delete the original session's (recreated) objects.
    """
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        panel.new_object(param=sigima.objects.GaussParam(), edit=False)
        source_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
        )
        original = history.history_sessions[-1]
        original_uuids = [
            uuid for action in original.actions for uuid in action.output_uuids
        ]
        select_tree_session(history, original)
        htools.duplicate_selected_entries(history)
        duplicate = history.history_sessions[-1]
        # The duplicated session is the active recording session when the
        # original group is deleted: the remove action lands in it.
        history.navigation.set_active_session(duplicate)
        original_group_id = panel.objmodel.get_object_group_id(
            panel.objmodel[source_uuid]
        )
        panel.objview.select_groups([original_group_id])
        panel.remove_object(force=True)
        remove_action = duplicate.actions[-1]
        assert remove_action.method_name == "remove_object"
        # Replay the ORIGINAL session: its objects are recreated
        select_tree_session(history, original)
        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_restore_actions(history)
        assert all(panel.objmodel.has_uuid(uuid) for uuid in original_uuids)
        # Replay the DUPLICATED session: the recorded deletion of the original
        # session's objects must be skipped
        select_tree_session(history, duplicate)
        history.runtime.execution.cascade_warnings.clear()
        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_restore_actions(history)
        assert all(panel.objmodel.has_uuid(uuid) for uuid in original_uuids), (
            "duplicate session replay deleted another session's objects"
        )
        # The skip must come from guard 3 (objects produced by another session)
        remove_name = (
            remove_action.title or remove_action.method_name or remove_action.uuid
        )
        guard3_message = (
            _("Action %s targets objects produced by another session and was skipped.")
            % remove_name
        )
        assert guard3_message in history.runtime.execution.cascade_warnings, (
            "recorded deletion was not skipped by the cross-session guard"
        )


def build_broken_signal_chain(win) -> tuple:
    """Record a compute chain, then delete its root input (record mode OFF).

    The root signals are created with record mode disabled so no producer
    action is registered: deleting ``root1`` afterwards cannot be reconnected
    and the recorded chain is genuinely broken. The session records three
    computes: a Gaussian filter and its derivative (both depending on
    ``root1``) plus an independent normalize on ``root2``.

    Args:
        win: DataLab main window

    Returns:
        Tuple (session, broken actions, still-valid action)
    """
    history, panel = win.historypanel, win.signalpanel
    root1, root2 = add_paracetamol_signals(panel, 2)
    history.toggle_record_mode(True)
    panel.objview.select_objects([root1])
    panel.processor.run_feature(
        sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
    )
    gaussian = history[len(history)]
    panel.objview.select_objects(gaussian.output_uuids)
    panel.processor.run_feature(sips.derivative)
    derivative = history[len(history)]
    panel.objview.select_objects([root2])
    panel.processor.run_feature(sips.normalize, sigima.params.NormalizeParam.create())
    normalize = history[len(history)]
    history.toggle_record_mode(False)
    panel.objview.select_objects([root1])
    panel.remove_object(force=True)
    history.toggle_record_mode(True)
    session = history.history_sessions[-1]
    assert not session.is_current_state_compatible(win)
    return session, [gaussian, derivative], normalize


def test_replay_broken_chain_repair_and_continue() -> None:
    """Repair a broken chain on replay, then replay the remaining actions.

    With ``accept_dialogs`` enabled, the broken-chain resolution dialog
    defaults to "Repair and continue": the incompatible actions and their
    downstream dependents are pruned from the history, the tree is refreshed
    and the remaining valid action of the selection is replayed.
    """
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        session, broken_actions, valid_action = build_broken_signal_chain(win)
        select_tree_session(history, session)
        saved_accept_dialogs = execenv.accept_dialogs
        execenv.accept_dialogs = True
        try:
            with patch.object(hrec, "flush_cascade_warnings"):
                history.replay_restore_actions(restore_selection=False)
        finally:
            execenv.accept_dialogs = saved_accept_dialogs
        # Broken actions and their downstream dependents were pruned
        assert session in history.history_sessions
        assert session.actions == [valid_action]
        for action in broken_actions:
            assert all(
                action not in other.actions for other in history.history_sessions
            )
        # The tree reflects the pruned history
        assert history.tree.topLevelItemCount() == len(history.history_sessions)
        # The remaining valid action was replayed
        assert valid_action.is_stale is False
        history.runtime.execution.cascade_warnings.clear()


def test_replay_broken_chain_cancel_keeps_history_intact() -> None:
    """Cancel the broken-chain dialog: nothing is modified, nothing replayed.

    In unattended mode without ``accept_dialogs``, the resolution dialog
    defaults to "Cancel": the history is left untouched and no action is
    replayed. The dialog will simply reappear on the next replay attempt.
    """
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        session, broken_actions, valid_action = build_broken_signal_chain(win)
        actions_before = list(session.actions)
        select_tree_session(history, session)
        assert execenv.unattended and not execenv.accept_dialogs
        with patch.object(hireplay, "replay_actions") as replay_mock:
            history.replay_restore_actions(restore_selection=False)
        replay_mock.assert_not_called()
        assert session in history.history_sessions
        assert session.actions == actions_before
        assert all(action in session.actions for action in broken_actions)
        assert valid_action in session.actions
