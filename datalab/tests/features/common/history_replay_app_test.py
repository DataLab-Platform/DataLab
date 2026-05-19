# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History panel replay & persistence tests

Covers replay/restore edge cases and HDF5 persistence paths:

1. Workspace ``.h5`` round-trip embeds the history panel and survives
   save → reset → reload (``MainWindow.save_h5_workspace`` /
   ``load_h5_workspace``).
2. Standalone ``HistoryPanel.serialize_to_hdf5`` /
   ``deserialize_from_hdf5`` round-trip on a populated panel (mirrors
   the ``.dlhist`` save/open menu paths).
3. ``1_to_n`` replay (extract ROI) reproduces the same number of
   children and same source UUID.
4. ``multiple_1_to_1`` replay surfaces a clear ``NotImplementedError``.
5. Replay of a ``2_to_1`` action whose secondary source UUID has been
   deleted raises ``ValueError`` instead of silently failing.
6. Public panel API ``replay_restore_actions(replay=True)`` produces
   a new object when a tree item is selected.
7. Public panel API ``replay_restore_actions(replay=False)`` only
   restores the workspace selection (no new object).
8. ``n_to_1`` replay forces the captured selection so a chained
   ``[New, ..., New, average]`` session does not aggregate over a
   stale single-object selection.
9. ``n_to_1`` replay falls back to the current selection when the
   captured UUIDs no longer exist (typical of a full-session replay
   in a fresh workspace), so the failure surfaces as the native
   processor error rather than an opaque ``WorkspaceState``
   incompatibility.
10. Full ``HistorySession`` replay on an empty workspace correctly
    re-runs ``[New, New, New, average]`` end-to-end: the per-session
    UUID remap translates the captured ``n_to_1`` selection to the
    freshly-created signals.
11. Full ``HistorySession`` replay with an intermediate
    ``[New, New, New, Remove, average]`` correctly drops the
    vanished UUID from the remap queue so ``average`` aggregates
    over the right two surviving signals.
"""

# pylint: disable=invalid-name

# guitest: skip

import os

import pytest
import sigima.objects
import sigima.params
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from sigima.objects.signal.creation import NewSignalParam
from sigima.tests import helpers
from sigima.tests.data import create_paracetamol_signal

from datalab.gui.panel.history import HistoryAction, WorkspaceState
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def _select_tree_item_for(history, entry: HistoryAction) -> None:
    """Select the tree item matching ``entry`` in the history tree."""
    tree = history.tree
    for i in range(tree.topLevelItemCount()):
        sess_item = tree.topLevelItem(i)
        for j in range(sess_item.childCount()):
            child = sess_item.child(j)
            if child.data(0, QC.Qt.UserRole) == entry.uuid:
                tree.clearSelection()
                tree.setCurrentItem(child)
                child.setSelected(True)
                return
    raise AssertionError(f"No tree item found for entry {entry.uuid}")


# --- 1) Workspace .h5 round-trip ------------------------------------------


def test_workspace_h5_roundtrip_with_history():
    """Saving + reloading the workspace ``.h5`` preserves the history panel."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())

        norm_param = sigima.params.NormalizeParam.create(method="maximum")
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.normalize, norm_param)

        recorded_titles = [a.title for a in history]
        recorded_func_names = [a.func_name for a in history]
        recorded_kinds = [a.kind for a in history]
        n_recorded = len(history)
        assert n_recorded >= 1

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            path = os.path.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(path)

            # Wipe everything (objects + history) then reload.
            win.reset_all()
            assert len(panel.objmodel) == 0

            win.load_h5_workspace([path], reset_all=True)

            # Note: ``save_h5_workspace`` itself records a UI history entry
            # ("Save to HDF5 file") *before* writing — that entry is part of
            # what gets saved. The reloaded history must therefore contain
            # at least the originally-recorded actions.
            reloaded = win.historypanel
            reloaded_titles = [a.title for a in reloaded]
            for title in recorded_titles:
                assert title in reloaded_titles, (
                    f"Title {title!r} missing from reloaded history"
                )
            for func_name in recorded_func_names:
                if func_name is not None:
                    assert func_name in [a.func_name for a in reloaded]
            for kind in recorded_kinds:
                assert kind in [a.kind for a in reloaded]


# --- 2) Standalone .dlhist round-trip on full panel -----------------------


def test_history_panel_dlhist_roundtrip():
    """``HistoryPanel.serialize_to_hdf5`` / ``deserialize_from_hdf5`` round-trip."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )

        original_titles = [a.title for a in history]
        original_func_names = [a.func_name for a in history]
        assert len(original_titles) >= 2

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            path = os.path.join(tmpdir, "history_panel.dlhist")
            with NativeH5Writer(path) as writer:
                history.serialize_to_hdf5(writer)
            with NativeH5Reader(path) as reader:
                history.deserialize_from_hdf5(reader)

        assert [a.title for a in history] == original_titles
        assert [a.func_name for a in history] == original_func_names


# --- 3) 1-to-n replay (extract ROI) ---------------------------------------


def test_replay_1_to_n_extract_roi():
    """``extract_roi`` (1-to-n) is recorded and replayable."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        sig = create_paracetamol_signal()
        sig.roi = sigima.objects.create_signal_roi([[26, 41], [125, 146]], indices=True)
        panel.add_object(sig)
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        n_objects_before = len(panel.objmodel)
        panel.objview.select_objects([1])
        panel.processor.run_feature("extract_roi", params=sig.roi.to_params(sig))
        n_added_first = len(panel.objmodel) - n_objects_before
        assert n_added_first >= 1

        entry = history[len(history)]
        assert entry.kind == HistoryAction.KIND_COMPUTE
        assert entry.pattern == "1_to_n"
        assert entry.func_name == "extract_roi"
        assert entry.state.selection.get(panel.PANEL_STR) == [src_uuid]

        # Replay against the original source.
        n_before_replay = len(panel.objmodel)
        entry.replay(win, restore_selection=True, edit=False)
        n_added_replay = len(panel.objmodel) - n_before_replay
        assert n_added_replay == n_added_first, (
            "Replay must produce the same number of children as the original"
        )


# --- 4) multiple_1_to_1 NotImplementedError --------------------------------


def test_multiple_1_to_1_replay_raises_not_implemented():
    """``multiple_1_to_1`` replay raises a clear ``NotImplementedError``."""
    with datalab_test_app_context() as win:
        action = HistoryAction(
            title="dummy multiple_1_to_1",
            kind=HistoryAction.KIND_COMPUTE,
            panel_str="signal",
            func_name="some_compound_op",
            pattern="multiple_1_to_1",
            state=WorkspaceState(),
        )
        with pytest.raises(NotImplementedError):
            action.replay(win, restore_selection=False, edit=False)


# --- 5) 2-to-1 replay with vanished secondary source -----------------------


def test_replay_2_to_1_with_vanished_obj2_raises():
    """Replaying a 2-to-1 action whose obj2 was deleted raises ``ValueError``."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        obj2 = panel.objmodel.get_object_from_number(2)

        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.difference, obj2)
        diff_entry = history[len(history)]
        assert diff_entry.pattern == "2_to_1"

        # Delete obj2 so its UUID is no longer resolvable.
        panel.objview.set_current_object(obj2)
        panel.remove_object(force=True)

        with pytest.raises(ValueError):
            # restore_selection=False so we hit the obj2 lookup path,
            # not the WorkspaceState compatibility check.
            diff_entry.replay(win, restore_selection=False, edit=False)


# --- 6) Public API replay_restore_actions(replay=True) ---------------------


def test_replay_via_panel_api_creates_new_object():
    """``HistoryPanel.replay_restore_actions`` replays the selected entry."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]

        _select_tree_item_for(history, deriv_entry)

        n_before = len(panel.objmodel)
        history.replay_restore_actions(replay=True, restore_selection=True)
        assert len(panel.objmodel) == n_before + 1


# --- 7) Public API replay_restore_actions(replay=False) -------------------


def test_restore_selection_only_via_panel_api():
    """``replay_restore_actions(replay=False)`` only restores selection."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]

        # Move selection elsewhere (the freshly-added derivative).
        derived = panel.objmodel.get_object_from_number(2)
        panel.objview.set_current_object(derived)
        derived_uuid = get_uuid(derived)
        assert panel.objview.get_sel_object_uuids() == [derived_uuid]

        _select_tree_item_for(history, deriv_entry)

        n_before = len(panel.objmodel)
        history.replay_restore_actions(replay=False, restore_selection=True)
        # No new object created.
        assert len(panel.objmodel) == n_before
        # Selection restored to the originally-recorded source.
        assert panel.objview.get_sel_object_uuids() == [src_uuid]


# --- 8) n_to_1 replay ignores restore_selection=False --------------------


def test_replay_n_to_1_forces_captured_selection():
    """``n_to_1`` replay must restore the captured multi-object selection
    even when ``restore_selection=False`` (otherwise an aggregator such as
    ``average`` would be applied to the single object the previous action
    left selected and fail with ``src_list must be a list of at least 2
    objects``)."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())

        # Average over the three signals (n_to_1 aggregator).
        panel.objview.select_objects([1, 2, 3])
        panel.processor.run_feature(sips.average)
        avg_entry = history[len(history)]
        assert avg_entry.pattern == "n_to_1"

        # Drift selection to a single object (mimics the state left by a
        # preceding "New signal" UI action in a chained session replay).
        panel.objview.select_objects([1])
        assert len(panel.objview.get_sel_object_uuids()) == 1

        n_before = len(panel.objmodel)
        # restore_selection=False on purpose: compute actions must still
        # restore their captured selection internally.
        avg_entry.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 1


# --- 9) n_to_1 replay tolerates vanished captured UUIDs -------------------


def test_replay_n_to_1_falls_back_when_captured_uuids_gone():
    """When the captured selection refers to UUIDs that no longer exist
    (full-session replay creates fresh objects with new UUIDs), the
    compute replay must fall back to the current selection rather than
    raising the opaque ``WorkspaceState`` incompatibility error."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1, 2, 3])
        panel.processor.run_feature(sips.average)
        avg_entry = history[len(history)]
        assert avg_entry.pattern == "n_to_1"

        # Wipe the workspace and recreate three signals: same shape,
        # different UUIDs (mimics the state mid-way through a full
        # session replay). The captured selection is now stale.
        win.reset_all()
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1, 2, 3])
        assert not avg_entry.state.is_current_state_compatible(win, False)

        n_before = len(panel.objmodel)
        # Must not raise ValueError("... not compatible with saved state");
        # falls back to the current selection (the 3 freshly-added signals).
        avg_entry.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 1


# --- 10) Full session replay with UUID remap ------------------------------


def test_full_session_replay_remaps_uuids_for_n_to_1():
    """A full ``HistorySession.replay`` on an empty workspace correctly
    re-runs ``[New signal, New signal, New signal, average]``: the
    per-session UUID remap translates ``average``'s captured selection
    to the freshly-created signals, instead of aggregating over the
    single object the last ``New signal`` left selected."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        # Record [New, New, New, average].
        for _i in range(3):
            panel.new_object(param=NewSignalParam(), edit=False)
        n_after_creations = len(panel.objmodel)
        assert n_after_creations == 3
        panel.objview.select_objects([1, 2, 3])
        panel.processor.run_feature(sips.average)
        assert len(panel.objmodel) == 4

        # pylint: disable=protected-access
        session = history._HistoryPanel__history_sessions[-1]  # noqa: SLF001

        # Reset to an empty workspace, then replay the whole session.
        win.reset_all()
        assert len(panel.objmodel) == 0
        session.replay(win, restore_selection=False, edit=False)
        # 3 fresh signals + 1 average = 4 objects.
        assert len(panel.objmodel) == 4


# --- 11) Full session replay with intermediate removal --------------------


def test_full_session_replay_with_intermediate_removal():
    """A full ``HistorySession.replay`` with ``[New, New, New, Remove,
    average]`` correctly drops the removed UUID from the unclaimed
    queue and the reverse remap, so ``average`` aggregates over the
    two surviving signals (and not, e.g., over a stale single-object
    selection or a removed one)."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        # Record [New, New, New, Remove #3, average of remaining 2].
        for _i in range(3):
            panel.new_object(param=NewSignalParam(), edit=False)
        assert len(panel.objmodel) == 3
        panel.objview.select_objects([3])
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 2
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(sips.average)
        assert len(panel.objmodel) == 3

        # pylint: disable=protected-access
        session = history._HistoryPanel__history_sessions[-1]  # noqa: SLF001

        # Reset to an empty workspace, then replay the whole session.
        win.reset_all()
        assert len(panel.objmodel) == 0
        session.replay(win, restore_selection=False, edit=False)
        # 3 created − 1 removed + 1 average = 3 objects.
        assert len(panel.objmodel) == 3


if __name__ == "__main__":
    test_workspace_h5_roundtrip_with_history()
    test_history_panel_dlhist_roundtrip()
    test_replay_1_to_n_extract_roi()
    test_multiple_1_to_1_replay_raises_not_implemented()
    test_replay_2_to_1_with_vanished_obj2_raises()
    test_replay_via_panel_api_creates_new_object()
    test_restore_selection_only_via_panel_api()
    test_replay_n_to_1_forces_captured_selection()
    test_replay_n_to_1_falls_back_when_captured_uuids_gone()
    test_full_session_replay_remaps_uuids_for_n_to_1()
    test_full_session_replay_with_intermediate_removal()
