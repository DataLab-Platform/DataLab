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
import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_sincos_image

from datalab.env import execenv
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history import recompute as hrec
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import add_paracetamol_signals


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

        assert history.runtime.execution.cascade_warnings == []
        assert action.is_stale is False
        assert spanel.objmodel.has_uuid(output_uuid)
        assert not ipanel.objmodel.has_uuid(output_uuid)
        assert len(ipanel.objmodel) == image_count
        recreated = spanel.objmodel[output_uuid]
        assert np.array_equal(recreated.xydata, expected_data)


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

        assert history.runtime.execution.cascade_warnings == []
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
