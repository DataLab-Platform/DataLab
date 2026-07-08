# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History panel — grouped exhaustive tests.

The History-panel behaviour is exercised by exactly five grouped tests, each
bundling many closely-related scenarios that previously lived in dedicated
per-behaviour tests:

* ``test_history_persistence`` — schema versioning, HDF5/.dlhist round-trips
  and ``HistoryAction`` compatibility (UUID/shape/legacy fallback).
* ``test_history_recording_and_replay`` — recording of compute + UI actions,
  per-pattern replay (1_to_1, 1_to_n, n_to_1, 2_to_1), session replay,
  interactive-fit replay and ``capture_outputs`` funnel behaviour.
* ``test_history_duplication`` — action/session duplication, UUID remapping
  and the creation-root vs synthetic-head duplication cases.
* ``test_history_editing_and_cascade`` — processing-tab edit propagation,
  step-by-step launch, cascade recompute and recompute invariants.
* ``test_history_navigation_and_chains`` — stepping/selection sync, chain
  reconnection/grouping/splitting, new-session prompt and active-session
  routing/highlight.

Each scenario is delimited by a ``# --- scenario: <name> ---`` comment.
Scenarios sharing GUI state run inside a single ``datalab_test_app_context``
block; truly independent pure-Python scenarios (HDF5 round-trip,
``NotImplementedError`` smoke) run outside or in a nested block.

Two GUI smoke tests are intentionally kept in their own modules:
``history_app_test.py::test_history_app`` and
``history_panel_app_test.py::test_history_panel``.
"""

# guitest: skip

import os
import shutil
import tempfile

import numpy as np
import pytest
import sigima.objects
import sigima.params
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import create_signal_roi
from sigima.objects.base import BaseROI
from sigima.objects.signal.creation import NewSignalParam
from sigima.tests import helpers
from sigima.tests.data import create_paracetamol_signal, create_sincos_image

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.base import AddMetadataParam, BaseDataPanel
from datalab.gui.panel.history import (
    HISTORY_ACTION_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    HistoryAction,
    HistorySession,
    HistoryTree,
    WorkspaceState,
)
from datalab.gui.panel.history.chainmodel import (
    ProcessingChain,
    build_processing_chains,
    build_session_chains,
)
from datalab.gui.processor.base import extract_processing_parameters
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delete_hdf5_items_by_name(group, item_name: str) -> None:
    """Delete HDF5 attributes/groups named ``item_name`` recursively."""
    if item_name in group.attrs:
        del group.attrs[item_name]
    if not hasattr(group, "keys"):
        return
    for key in list(group.keys()):
        if key == item_name:
            del group[key]
        else:
            _delete_hdf5_items_by_name(group[key], item_name)


def _create_serializable_history_session() -> HistorySession:
    """Create a history session requiring no application startup."""
    state = WorkspaceState()
    state.selection = {"signal": ["source-uuid"]}
    state.states = {"signal": ["(10,)"]}
    state.titles = {"signal": ["source"]}
    state.object_metadata = {"signal": {"source-uuid": {"shape": [10], "ndim": 1}}}
    action = HistoryAction(
        title="Rename",
        kind=HistoryAction.KIND_UI,
        target="signalpanel",
        method_name="set_current_object_title",
        kwargs={"title": "renamed"},
        state=state,
    )
    session = HistorySession(number=1)
    session.add_action(action)
    return session


def _session_action_counts(history) -> list[int]:
    """Return the number of recorded actions in each history session."""
    return [len(session.actions) for session in history.history_sessions]


def _get_tree_item_for(history, entry: HistoryAction):
    """Return the tree item matching ``entry`` in the history tree."""
    tree = history.tree
    iterator = QW.QTreeWidgetItemIterator(tree)
    while iterator.value():
        item = iterator.value()
        if item.data(0, QC.Qt.UserRole) == entry.uuid:
            return item
        iterator += 1
    raise AssertionError(f"No tree item found for entry {entry.uuid}")


def _select_tree_item_for(history, entry: HistoryAction) -> None:
    """Select the tree item matching ``entry`` in the history tree."""
    child = _get_tree_item_for(history, entry)
    history.tree.clearSelection()
    history.tree.setCurrentItem(child)
    child.setSelected(True)


def _select_tree_session(history, session) -> None:
    """Select the tree item matching ``session`` in the history tree."""
    sessions = history.history_sessions
    index = sessions.index(session)
    item = history.tree.topLevelItem(index)
    history.tree.clearSelection()
    history.tree.setCurrentItem(item)
    item.setSelected(True)


def _record_three_action_session(win):
    """Helper: record [add_signal + normalize + derivative] in one session."""
    history = win.historypanel
    history.toggle_record_mode(True)
    panel = win.signalpanel
    panel.add_object(create_paracetamol_signal())
    panel.objview.select_objects([1])
    panel.processor.run_feature(
        sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
    )
    panel.objview.select_objects([2])
    panel.processor.run_feature(sips.derivative)
    return panel, history


def _build_cascade_chain(panel, history):
    """Build chain s001 -> gaussian -> s002 -> derivative -> s003 -> mavg -> s004.

    Returns ``(action_a, action_b, action_c, output_b, output_c)``.
    """
    panel.add_object(create_paracetamol_signal())
    panel.objview.select_objects([1])
    panel.processor.run_feature(
        sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
    )
    action_a = history[len(history)]

    panel.objview.select_objects([2])
    panel.processor.run_feature(sips.derivative)
    action_b = history[len(history)]
    output_b = panel.objmodel.get_object_from_number(3)

    panel.objview.select_objects([3])
    mavg = sigima.params.MovingAverageParam()
    mavg.n = 3
    panel.processor.run_feature(sips.moving_average, mavg)
    action_c = history[len(history)]
    output_c = panel.objmodel.get_object_from_number(4)
    return action_a, action_b, action_c, output_b, output_c


# ---------------------------------------------------------------------------
# 1) Persistence: schema, HDF5/.dlhist round-trips, compatibility
# ---------------------------------------------------------------------------


def test_history_persistence():
    """Schema, HDF5/.dlhist round-trips and ``HistoryAction`` compatibility."""
    # --- scenario: schema_version is persisted ---
    session = _create_serializable_history_session()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history_schema.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        with NativeH5Reader(path) as reader:
            restored_sessions = reader.read_object_list(
                "history_session", HistorySession
            )
    assert len(restored_sessions) == 1
    restored = restored_sessions[0]
    assert restored.schema_version == HISTORY_SCHEMA_VERSION
    assert restored.actions[0].schema_version == HISTORY_ACTION_SCHEMA_VERSION
    assert restored.actions[0].kwargs == {"title": "renamed"}
    assert restored.actions[0].state.object_metadata == {
        "signal": {"source-uuid": {"shape": [10], "ndim": 1}}
    }

    # --- scenario: missing schema_version defaults to current ---
    session = _create_serializable_history_session()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history_schema_missing.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
            _delete_hdf5_items_by_name(writer.h5, "schema_version")
        with NativeH5Reader(path) as reader:
            restored_sessions = reader.read_object_list(
                "history_session", HistorySession
            )
    restored_action = restored_sessions[0].actions[0]
    assert restored_sessions[0].schema_version == HISTORY_SCHEMA_VERSION
    assert restored_action.schema_version == HISTORY_SCHEMA_VERSION
    assert restored_action.title == "Rename"

    # --- scenario: ROI kwargs survive HDF5 round-trip ---
    roi = create_signal_roi([[26, 41]], indices=True)
    state = WorkspaceState()
    state.selection = {"signal": ["dst-uuid"]}
    state.states = {"signal": ["(100,)"]}
    state.titles = {"signal": ["dst"]}
    state.object_metadata = {"signal": {"dst-uuid": {"shape": [100], "ndim": 1}}}
    action = HistoryAction(
        title="Paste ROI",
        kind=HistoryAction.KIND_UI,
        target="signalpanel",
        method_name="paste_roi",
        kwargs={"roi_data": roi},
        state=state,
    )
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roi_roundtrip.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        with NativeH5Reader(path) as reader:
            restored_sessions = reader.read_object_list(
                "history_session", HistorySession
            )
    restored_roi = restored_sessions[0].actions[0].kwargs.get("roi_data")
    assert restored_roi is not None
    assert isinstance(restored_roi, BaseROI)
    assert restored_roi.get_single_roi(0).coords.tolist() == [26, 41]

    # --- scenario: legacy translated panel keys are normalized ---
    state = WorkspaceState()
    state.selection = {"Signal Panel": ["uuid-1"]}
    state.states = {"Signal Panel": ["(10,)"]}
    state.titles = {"Signal Panel": ["obj1"]}
    state.object_metadata = {"Signal Panel": {"uuid-1": {"shape": [10], "ndim": 1}}}
    legacy_action = HistoryAction(
        title="Legacy",
        kind=HistoryAction.KIND_UI,
        target="signalpanel",
        method_name="set_current_object_title",
        kwargs={"title": "renamed"},
        state=state,
    )
    legacy_session = HistorySession(number=1)
    legacy_session.add_action(legacy_action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy_keys.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([legacy_session], "history_session")
        with NativeH5Reader(path) as reader:
            restored_sessions = reader.read_object_list(
                "history_session", HistorySession
            )
    rstate = restored_sessions[0].actions[0].state
    assert "signal" in rstate.selection and "Signal Panel" not in rstate.selection
    assert rstate.selection["signal"] == ["uuid-1"]
    assert rstate.states["signal"] == ["(10,)"]
    assert rstate.titles["signal"] == ["obj1"]
    assert rstate.object_metadata["signal"] == {"uuid-1": {"shape": [10], "ndim": 1}}

    # GUI-bound HDF5 scenarios run inside one app context.
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel

        # --- scenario: deserialize from .h5 without history group does not raise ---
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_history.h5")
            with NativeH5Writer(path) as writer:
                panel.serialize_to_hdf5(writer)
            with NativeH5Reader(path) as reader:
                history.deserialize_from_hdf5(reader)
        assert len(history) == 0

        # --- scenario: HistoryAction HDF5 round-trip without pickle, then replay ---
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        norm_param = sigima.params.NormalizeParam.create(method="maximum")
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.normalize, norm_param)
        original = history[len(history)]
        ser_session = HistorySession(number=1)
        ser_session.actions.append(original)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.dlhist")
            with NativeH5Writer(path) as writer:
                writer.write_object_list([ser_session], "history_session")
            with NativeH5Reader(path) as reader:
                restored_sessions = reader.read_object_list(
                    "history_session", HistorySession
                )
        restored = restored_sessions[0].actions[0]
        assert restored.uuid == original.uuid
        assert not hasattr(restored, "func")
        assert restored.kind == HistoryAction.KIND_COMPUTE
        assert restored.func_name == "normalize"
        assert restored.pattern == "1_to_1"
        assert restored.panel_str == panel.PANEL_STR_ID
        restored_param = restored.kwargs.get("param")
        assert restored_param is not None
        assert type(restored_param).__name__ == type(norm_param).__name__
        n_before = len(panel.objmodel)
        restored.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == n_before + 1
        new_obj = panel.objmodel.get_object_from_number(len(panel.objmodel))
        new_pp = extract_processing_parameters(new_obj)
        assert new_pp is not None
        assert new_pp.source_uuid == src_uuid
        assert new_pp.func_name == "normalize"

        # --- scenario: workspace .h5 round-trip embeds history ---
        recorded_titles = [a.title for a in history]
        recorded_func_names = [a.func_name for a in history]
        recorded_kinds = [a.kind for a in history]
        assert len(history) >= 1
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            path = os.path.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(path)
            win.reset_all()
            assert len(panel.objmodel) == 0
            win.load_h5_workspace([path], reset_all=True)
            reloaded = win.historypanel
            reloaded_titles = [a.title for a in reloaded]
            for title in recorded_titles:
                assert title in reloaded_titles
            for func_name in recorded_func_names:
                if func_name is not None:
                    assert func_name in [a.func_name for a in reloaded]
            for kind in recorded_kinds:
                assert kind in [a.kind for a in reloaded]

        # --- scenario: standalone .dlhist round-trip (import path on non-empty WS) ---
        win.reset_all()
        history = win.historypanel
        panel = win.signalpanel
        history.toggle_record_mode(True)
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
        n_actions_before = len(history)
        n_signals_before = len(panel.objmodel)
        n_sessions_before = len(history.history_sessions)
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            path = os.path.join(tmpdir, "history_panel.dlhist")
            assert history.save_to_dlhist_file(path)
            assert history.open_dlhist_file(path)
        reloaded_titles = [a.title for a in history]
        reloaded_func_names = [a.func_name for a in history]
        for title in original_titles:
            assert title in reloaded_titles
        for func_name in original_func_names:
            if func_name is not None:
                assert func_name in reloaded_func_names
        assert len(history.history_sessions) > n_sessions_before
        assert len(panel.objmodel) > n_signals_before
        assert len(history) > n_actions_before

    # --- scenario: .dlhist self-contained — direct-load into fresh empty workspace ---
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.dlhist")
        with datalab_test_app_context() as win:
            history = win.historypanel
            history.toggle_record_mode(True)
            panel = win.signalpanel
            panel.add_object(create_paracetamol_signal())
            panel.objview.select_objects([1])
            panel.processor.run_feature(sips.derivative)
            original_titles = [a.title for a in history]
            original_func_names = [a.func_name for a in history]
            assert len(original_titles) >= 1
            assert history.save_to_dlhist_file(path)
        with datalab_test_app_context() as win2:
            history2 = win2.historypanel
            panel2 = win2.signalpanel
            assert len(panel2.objmodel) == 0
            assert len(history2.history_sessions) == 0
            assert history2.open_dlhist_file(path)
            reloaded_titles = [a.title for a in history2]
            reloaded_func_names = [a.func_name for a in history2]
            assert reloaded_titles == original_titles
            for func_name in original_func_names:
                if func_name is not None:
                    assert func_name in reloaded_func_names
            assert len(panel2.objmodel) >= 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- scenario: incompatible when selected UUID disappears ---
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        obj = panel.objmodel.get_object_from_number(1)
        panel.objview.set_current_object(obj)
        state = WorkspaceState()
        state.save(win)
        action = HistoryAction(title="Dummy", state=state)
        assert action.is_current_state_compatible(win, restore_selection=False)
        panel.remove_object(force=True)
        assert not action.is_current_state_compatible(win, restore_selection=False)

        # --- scenario: incompatible when selected object shape changes ---
        win.reset_all()
        panel.add_object(create_paracetamol_signal())
        obj = panel.objmodel.get_object_from_number(1)
        panel.objview.set_current_object(obj)
        state = WorkspaceState()
        state.save(win)
        action = HistoryAction(title="Dummy", state=state)
        assert action.is_current_state_compatible(win, restore_selection=False)
        obj.set_xydata(obj.x[:-1], obj.y[:-1])
        assert not action.is_current_state_compatible(win, restore_selection=False)

        # --- scenario: histories without object_metadata fall back to UUID check ---
        win.reset_all()
        panel.add_object(create_paracetamol_signal())
        obj = panel.objmodel.get_object_from_number(1)
        panel.objview.set_current_object(obj)
        state = WorkspaceState()
        state.save(win)
        action = HistoryAction(title="Dummy", state=state)
        session = HistorySession(number=1)
        session.add_action(action)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history_without_gate2_metadata.dlhist")
            with NativeH5Writer(path) as writer:
                writer.write_object_list([session], "history_session")
                _delete_hdf5_items_by_name(writer.h5, "object_metadata")
            with NativeH5Reader(path) as reader:
                restored_sessions = reader.read_object_list(
                    "history_session", HistorySession
                )
        restored_action = restored_sessions[0].actions[0]
        assert restored_action.state.object_metadata == {}
        assert restored_action.is_current_state_compatible(win, restore_selection=False)

        # --- scenario: tree marks incompatible action after source deletion ---
        win.reset_all()
        history = win.historypanel
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]
        item = _get_tree_item_for(history, deriv_entry)
        assert item.data(0, HistoryTree.COMPATIBILITY_ROLE) is True
        history.toggle_record_mode(False)
        panel.objview.select_objects([1])
        panel.remove_object(force=True)
        history.refresh_compatibility_items()
        item = _get_tree_item_for(history, deriv_entry)
        assert item.data(0, HistoryTree.COMPATIBILITY_ROLE) is False
        assert item.foreground(0).color().isValid()

    # --- scenario: action WorkspaceState is panel-scoped ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        spanel, ipanel = win.signalpanel, win.imagepanel

        # Create a signal and keep it selected in the signal panel.
        spanel.add_object(create_paracetamol_signal())
        sig_uuid = get_uuid(spanel.objmodel.get_object_from_number(1))
        spanel.objview.select_objects([1])

        # Create an image and run a 1_to_1 image compute action.
        ipanel.add_object(create_sincos_image())
        ima_uuid = get_uuid(ipanel.objmodel.get_object_from_number(1))
        ipanel.objview.select_objects([1])
        ipanel.processor.run_feature(sipi.inverse)

        # The recorded image action must NOT embed the signal-panel selection.
        entry = history[len(history)]
        assert entry.kind == HistoryAction.KIND_COMPUTE
        assert entry.panel_str == ipanel.PANEL_STR_ID
        assert entry.state.selection.get("signal", []) == []
        assert entry.state.selection.get("image") == [ima_uuid]

        # Deleting the unrelated signal must keep the image action compatible.
        history.toggle_record_mode(False)
        spanel.objview.select_objects([1])
        spanel.remove_object(force=True)
        assert sig_uuid not in spanel.objmodel.get_object_ids()
        assert entry.is_current_state_compatible(win, restore_selection=False) is True


# ---------------------------------------------------------------------------
# 2) Recording, replay patterns & session replay
# ---------------------------------------------------------------------------


def test_history_recording_and_replay(monkeypatch):
    """Recording, per-pattern replay, session replay, fit replay & capture."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        # --- scenario: compute_1_to_1 history matches ProcessingParameters ---
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        result_obj = panel.objmodel.get_object_from_number(2)
        pp = extract_processing_parameters(result_obj)
        assert pp is not None
        assert pp.func_name == "derivative"
        assert pp.source_uuid == src_uuid
        entry = history[len(history)]
        assert entry.kind == HistoryAction.KIND_COMPUTE
        assert entry.func_name == "derivative"
        assert entry.pattern == "1_to_1"
        assert entry.state.selection.get(panel.PANEL_STR_ID) == [src_uuid]

        # --- scenario: recompute_processing does not add a history entry ---
        n_before = len(history)
        derived = panel.objmodel.get_object_from_number(2)
        panel.objview.set_current_object(derived)
        panel.recompute_processing()
        assert len(history) == n_before

        # --- scenario: replay finds target by UUID after panel reorder ---
        deriv_entry = history[len(history)]
        panel.add_object(create_paracetamol_signal())
        assert get_uuid(panel.objmodel[src_uuid]) == src_uuid
        n_before_replay = len(panel.objmodel)
        deriv_entry.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == n_before_replay + 1
        new_obj = panel.objmodel.get_object_from_number(len(panel.objmodel))
        new_pp = extract_processing_parameters(new_obj)
        assert new_pp is not None
        assert new_pp.source_uuid == src_uuid

    # --- scenario: UI rename capture + replay ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        obj = panel.objmodel.get_object_from_number(1)
        original_title = obj.title
        new_title = "renamed-by-test"
        panel.objview.set_current_object(obj)
        panel.set_current_object_title(new_title)
        assert obj.title == new_title
        rename_entry = history[len(history)]
        assert rename_entry.kind == HistoryAction.KIND_UI
        assert rename_entry.target == "signalpanel"
        assert rename_entry.method_name == "set_current_object_title"
        assert rename_entry.kwargs.get("title") == new_title
        panel.set_current_object_title("transient-title")
        assert obj.title == "transient-title"
        rename_entry.replay(win, restore_selection=False, edit=False)
        assert obj.title == new_title and obj.title != original_title
        assert isinstance(rename_entry.state, WorkspaceState)

        # --- scenario: add_metadata capture ---
        obj_uuid = get_uuid(obj)
        panel.objview.select_objects([1])
        param = AddMetadataParam([obj])
        param.metadata_key = "history_gate6"
        param.value_pattern = "value_{index}"
        param.conversion = "string"
        panel.add_metadata(param)
        entry = history[len(history)]
        assert entry.kind == HistoryAction.KIND_UI
        assert entry.target == "signalpanel"
        assert entry.method_name == "add_metadata"
        captured = entry.kwargs.get("param")
        assert captured is not None and captured is not param
        assert captured.metadata_key == param.metadata_key
        assert captured.value_pattern == param.value_pattern
        assert captured.conversion == param.conversion
        assert entry.state.selection.get(panel.PANEL_STR_ID) == [obj_uuid]

    # --- scenario: add_metadata replay ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        obj = panel.objmodel.get_object_from_number(1)
        panel.objview.select_objects([1])
        param = AddMetadataParam([obj])
        param.metadata_key = "replay_test_key"
        param.value_pattern = "replay_value_{index}"
        param.conversion = "string"
        panel.add_metadata(param)
        assert obj.metadata.get("replay_test_key") == "replay_value_1"
        entry = history[len(history)]
        del obj.metadata["replay_test_key"]
        assert "replay_test_key" not in obj.metadata
        entry.replay(win, restore_selection=False, edit=False)
        assert obj.metadata.get("replay_test_key") == "replay_value_1"

    # --- scenario: ROI copy/paste capture + deterministic replay ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        src = create_paracetamol_signal()
        src.roi = create_signal_roi([[26, 41]], indices=True)
        panel.add_object(src)
        dst = create_paracetamol_signal()
        panel.add_object(dst)
        src_uuid = get_uuid(src)
        dst_uuid = get_uuid(dst)
        panel.objview.set_current_item_id(src_uuid)
        panel.copy_roi()
        copy_entry = history[len(history)]
        panel.objview.set_current_item_id(dst_uuid)
        panel.paste_roi()
        paste_entry = history[len(history)]
        assert copy_entry.kind == HistoryAction.KIND_UI
        assert copy_entry.target == "signalpanel"
        assert copy_entry.method_name == "copy_roi"
        assert "roi_data" in copy_entry.kwargs
        assert copy_entry.state.selection.get(panel.PANEL_STR_ID) == [src_uuid]
        assert paste_entry.kind == HistoryAction.KIND_UI
        assert paste_entry.target == "signalpanel"
        assert paste_entry.method_name == "paste_roi"
        assert "roi_data" in paste_entry.kwargs
        assert paste_entry.state.selection.get(panel.PANEL_STR_ID) == [dst_uuid]
        # Deterministic replay: change source ROI then replay paste.
        dst_obj = panel.objmodel.get_object_from_number(2)
        assert dst_obj.roi is not None
        src.roi = create_signal_roi([[100, 200]], indices=True)
        dst_obj.roi = None
        paste_entry.replay(win, restore_selection=False, edit=False)
        assert dst_obj.roi is not None
        assert dst_obj.roi.get_single_roi(0).coords.tolist() == [26, 41]

    # --- scenario: save_to_directory capture + replay ---
    saved_paths: list[str] = []

    def fake_save_to_file(_self, _obj, filename):
        saved_paths.append(filename)

    monkeypatch.setattr(
        BaseDataPanel, "_BaseDataPanel__save_to_file", fake_save_to_file
    )
    with datalab_test_app_context() as win:
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            history = win.historypanel
            history.toggle_record_mode(True)
            panel = win.signalpanel
            panel.add_object(create_paracetamol_signal())
            obj = panel.objmodel.get_object_from_number(1)
            obj_uuid = get_uuid(obj)
            panel.objview.select_objects([1])
            param = sigima.params.SaveToDirectoryParam.create(
                directory=tmpdir,
                basename="history_gate6_{index}",
                extension=".csv",
                overwrite=True,
            )
            panel.save_to_directory(param)
            entry = history[len(history)]
            assert entry.kind == HistoryAction.KIND_UI
            assert entry.target == "signalpanel"
            assert entry.method_name == "save_to_directory"
            captured = entry.kwargs.get("param")
            assert captured is not None and captured is not param
            assert captured.directory == param.directory
            assert captured.basename == param.basename
            assert captured.extension == param.extension
            assert captured.overwrite == param.overwrite
            assert entry.state.selection.get(panel.PANEL_STR_ID) == [obj_uuid]
            assert saved_paths == [os.path.join(tmpdir, "history_gate6_1.csv")]
            # Replay: must call save again with same parameters.
            n_before = len(saved_paths)
            entry.replay(win, restore_selection=False, edit=False)
            assert len(saved_paths) == n_before + 1
            assert saved_paths[-1] == saved_paths[-2]

    # --- scenario: multiple_1_to_1 replay raises NotImplementedError ---
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

        # --- scenario: normal processing outside replay still adds objects ---
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        n_before = len(panel.objmodel)
        panel.processor.run_feature(sips.derivative)
        assert len(panel.objmodel) == n_before + 1

    # --- scenario: 1_to_n extract_roi replay (persistent direct replay) ---
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
        assert entry.state.selection.get(panel.PANEL_STR_ID) == [src_uuid]
        n_before_replay = len(panel.objmodel)
        entry.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) - n_before_replay == n_added_first

    # --- scenario: n_to_1 forces captured selection ---
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
        panel.objview.select_objects([1])
        assert len(panel.objview.get_sel_object_uuids()) == 1
        n_before = len(panel.objmodel)
        avg_entry.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 1

        # --- scenario: n_to_1 falls back when captured UUIDs are gone ---
        win.reset_all()
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1, 2, 3])
        assert not avg_entry.state.is_current_state_compatible(win, False)
        n_before = len(panel.objmodel)
        avg_entry.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 1

    # --- scenario: n_to_1 passes recorded pairwise flag ---
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        captured: dict = {}

        def capture_compute_n_to_1(*_args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(panel.processor, "compute_n_to_1", capture_compute_n_to_1)
        action = HistoryAction(
            title="average pairwise",
            kind=HistoryAction.KIND_COMPUTE,
            panel_str=panel.PANEL_STR_ID,
            func_name="average",
            pattern="n_to_1",
            kwargs={"pairwise": True},
            state=WorkspaceState(),
        )
        action.replay_compute(win, edit=False)
        assert captured["pairwise"] is True

    # --- scenario: 2_to_1 with vanished obj2 raises ValueError ---
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
        panel.objview.set_current_object(obj2)
        panel.remove_object(force=True)
        with pytest.raises(ValueError):
            diff_entry.replay(win, restore_selection=False, edit=False)

    # --- scenario: 2_to_1 replay translates obj2 UUIDs and passes pairwise ---
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        obj2 = panel.objmodel.get_object_from_number(1)
        obj2_uuid = get_uuid(obj2)
        captured = {}

        def capture_compute_2_to_1(obj2_arg, *_args, **kwargs):
            captured["obj2"] = obj2_arg
            captured.update(kwargs)

        monkeypatch.setattr(panel.processor, "compute_2_to_1", capture_compute_2_to_1)
        action = HistoryAction(
            title="difference pairwise",
            kind=HistoryAction.KIND_COMPUTE,
            panel_str=panel.PANEL_STR_ID,
            func_name="difference",
            pattern="2_to_1",
            kwargs={"obj2_uuids": ["recorded-obj2"], "pairwise": True},
            state=WorkspaceState(),
        )
        action.replay_compute(
            win,
            edit=False,
            uuid_remap={panel.PANEL_STR_ID: {"recorded-obj2": obj2_uuid}},
        )
        assert captured["obj2"] is obj2
        assert captured["pairwise"] is True

    # scenario: replay_restore_actions(replay=True) on 1_to_1 does NOT add output
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]
        _select_tree_item_for(history, deriv_entry)
        n_signal_before = len(panel.objmodel)
        n_history_before = len(history)
        history.replay_restore_actions(replay=True, restore_selection=True)
        assert len(panel.objmodel) == n_signal_before
        assert len(history) == n_history_before

    # --- scenario: reset_all starts a new session and preserves history ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel
        win.reset_all()
        assert len(history) == 0
        assert _session_action_counts(history) == []
        history.toggle_record_mode(True)
        panel.new_object(param=sigima.objects.GaussParam(), edit=False)
        assert len(history) == 1
        assert _session_action_counts(history) == [1]
        first_title = history[1].title
        history.toggle_record_mode(False)
        win.reset_all()
        assert len(history) == 1
        assert _session_action_counts(history) == [1, 0]
        assert history[1].title == first_title
        history.toggle_record_mode(True)
        panel.new_object(param=sigima.objects.LorentzParam(), edit=False)
        assert len(history) == 2
        assert _session_action_counts(history) == [1, 1]
        assert history[1].title == first_title
        assert history[2].title == _("New signal")

    # --- scenario: full session replay on existing data ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        assert len(panel.objmodel) == 3
        panel.objview.select_objects([1, 2, 3])
        panel.processor.run_feature(sips.average)
        assert len(panel.objmodel) == 4
        session = history.history_sessions[-1]
        n_before = len(panel.objmodel)
        session.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 1

    # --- scenario: chained compute session replay (output queue remap) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        panel.objview.select_objects([3, 4])
        panel.processor.run_feature(sips.average)
        assert len(panel.objmodel) == 5
        session = history.history_sessions[-1]
        n_before = len(panel.objmodel)
        session.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == n_before + 3

    # --- scenario: direct HistoryAction.replay() does NOT record new entries ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel
        history.toggle_record_mode(True)
        panel.new_object(param=NewSignalParam(), edit=False)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        assert len(panel.objmodel) == 2
        session = history.history_sessions[-1]
        compute_action = [
            a for a in session.actions if a.kind == HistoryAction.KIND_COMPUTE
        ][0]
        n_before = sum(len(s.actions) for s in history.history_sessions)
        panel.objview.select_objects([1])
        compute_action.replay(win, restore_selection=True, edit=False)
        assert sum(len(s.actions) for s in history.history_sessions) == n_before

    # --- scenario: direct HistorySession.replay() does NOT record entries ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(sips.average)
        assert len(panel.objmodel) == 3
        session = history.history_sessions[-1]
        n_before = sum(len(s.actions) for s in history.history_sessions)
        panel.objview.select_objects([1, 2])
        session.replay(win, restore_selection=False, edit=False)
        assert sum(len(s.actions) for s in history.history_sessions) == n_before

    # --- scenario: panel API session replay skips UI-creation, no output ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.new_object(param=NewSignalParam(), edit=False)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        session = history.history_sessions[-1]
        _select_tree_session(history, session)
        n_signal_before = len(panel.objmodel)
        n_history_before = len(history)
        history.replay_restore_actions(replay=True, restore_selection=True)
        assert len(panel.objmodel) == n_signal_before
        assert len(history) == n_history_before

    # --- scenario: replay whole session when no tree selection ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        history.tree.clearSelection()
        n_before = len(panel.objmodel)
        n_history_before = len(history)
        history.replay_restore_actions(replay=True, restore_selection=False)
        assert len(panel.objmodel) == n_before
        assert len(history) == n_history_before

    # --- scenario: 2_to_1 with primary = #1, obj2 = #2 ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        obj2_for_diff = panel.objmodel.get_object_from_number(2)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.difference, obj2_for_diff)
        assert len(panel.objmodel) == 3
        original_title = panel.objmodel.get_object_from_number(3).title
        assert "s001" in original_title and "s002" in original_title
        assert original_title.index("s001") < original_title.index("s002")
        diff_entry = history[len(history)]
        _select_tree_item_for(history, diff_entry)
        n_before = len(panel.objmodel)
        history.replay_restore_actions(replay=True, restore_selection=True)
        assert len(panel.objmodel) == n_before

    # --- scenario: duplicate_object registers duplicated outputs ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.duplicate_object()
        session = history.history_sessions[-1]
        dup_action = next(
            action
            for action in session.actions
            if action.method_name == "duplicate_object"
        )
        assert dup_action.output_uuids
        first_output = dup_action.output_uuids[0]
        assert history.output_to_action[first_output] == dup_action.uuid

    # --- interactive curve-fit record + .dlhist round-trip + replay ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        panel = win.signalpanel

        # --- scenario: record a polynomial interactive fit as a UI action ---
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        src = panel.objmodel.get_object_from_number(1)
        src_uuid = get_uuid(src)
        panel.objview.select_objects([1])

        degree = 3
        coeffs = np.polyfit(src.x, src.y, degree)  # highest-first == polyval order
        fit_values = [float(c) for c in coeffs]
        expected_y = np.polyval(coeffs, src.x)
        name = _("Polynomial fit")

        result = sigima.objects.create_signal(f"{name}({src.title})", src.x, expected_y)
        action = history.add_ui_entry(
            name,
            target="signalprocessor",
            method_name="recompute_fit",
            save_state=True,
            fit_type="polynomial",
            fit_values=fit_values,
            fit_name=name,
            source_uuid=src_uuid,
        )
        assert action is not None
        with history.capture_outputs(action):
            panel.add_object(result)
        assert action.output_uuids == [get_uuid(result)]
        assert action.target == "signalprocessor"
        assert action.panel_str == panel.PANEL_STR_ID

        # --- scenario: .dlhist round-trip preserves uuid + fit kwargs ---
        ser_session = HistorySession(number=1)
        ser_session.actions.append(action)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fit.dlhist")
            with NativeH5Writer(path) as writer:
                writer.write_object_list([ser_session], "history_session")
            with NativeH5Reader(path) as reader:
                restored_sessions = reader.read_object_list(
                    "history_session", HistorySession
                )
        restored = restored_sessions[0].actions[0]
        assert restored.uuid == action.uuid
        assert restored.kind == HistoryAction.KIND_UI
        assert restored.target == "signalprocessor"
        assert restored.method_name == "recompute_fit"
        assert restored.kwargs.get("fit_type") == "polynomial"
        assert restored.kwargs.get("source_uuid") == src_uuid
        assert restored.kwargs.get("fit_values") == pytest.approx(fit_values)

        # --- scenario: deterministic replay reconstructs the fit curve ---
        n_before = len(panel.objmodel)
        restored.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == n_before + 1
        new_obj = panel.objmodel.get_object_from_number(len(panel.objmodel))
        assert np.allclose(new_obj.x, src.x)
        assert np.allclose(new_obj.y, expected_y)

    # --- capture_outputs funnel: drop failed producing computes ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])

        # --- scenario: producing compute with no output is dropped ---
        action = history.add_compute_entry(
            "Failing op",
            panel_str="signal",
            func_name="gaussian_filter",
            pattern="1_to_1",
            save_state=True,
        )
        assert action is not None
        len_before = len(history)
        with history.capture_outputs(action):
            pass  # simulate a failed compute: no object created
        assert action not in list(history)
        assert len(history) == len_before - 1
        assert action.uuid not in history.action_output_uuids

        # --- control: a UI action producing no object is preserved ---
        ui_action = history.add_ui_entry(
            "Some UI",
            target="signalpanel",
            method_name="copy_metadata",
            save_state=False,
        )
        assert ui_action is not None
        ui_len_before = len(history)
        with history.capture_outputs(ui_action):
            pass
        assert ui_action in list(history)
        assert len(history) == ui_len_before

        # --- control: a 1_to_0 analysis compute producing no object is kept ---
        analysis_action = history.add_compute_entry(
            "Analysis op",
            panel_str="signal",
            func_name="stats",
            pattern="1_to_0",
            save_state=True,
        )
        assert analysis_action is not None
        analysis_len_before = len(history)
        with history.capture_outputs(analysis_action):
            pass
        assert analysis_action in list(history)
        assert len(history) == analysis_len_before


# ---------------------------------------------------------------------------
# 3) Duplication
# ---------------------------------------------------------------------------


def test_history_duplication():
    """Action/session duplication, UUID remap, ordering + creation-root cases."""
    # --- scenario: duplicating an action creates an independent copy ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        param = sigima.params.MovingAverageParam()
        param.n = 3
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.moving_average, param)
        original = history[len(history)]
        _select_tree_item_for(history, original)
        n_before = len(panel.objmodel)
        history.duplicate_selected_entries()
        sessions = history.history_sessions
        duplicate_session = sessions[-1]
        # Operation-rooted chain (Cas B): a synthetic creation head is
        # prepended, so the compute duplicate is no longer at index 0.
        assert duplicate_session.actions[0].method_name == "new_object"
        duplicate = next(
            a for a in duplicate_session.actions if a.kind == HistoryAction.KIND_COMPUTE
        )
        assert duplicate_session.title.endswith(_("Copy"))
        assert duplicate is not original
        assert duplicate.uuid != original.uuid
        assert duplicate.kwargs["param"] is not original.kwargs["param"]
        duplicate.kwargs["param"].n = 7
        assert original.kwargs["param"].n == 3
        assert duplicate.kwargs["param"].n == 7
        assert len(panel.objmodel) > n_before

    # --- scenario: duplicating a session copies all actions independently ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        param = sigima.params.MovingAverageParam()
        param.n = 3
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.moving_average, param)
        original_session = history.history_sessions[-1]
        _select_tree_session(history, original_session)
        history.duplicate_selected_entries()
        sessions = history.history_sessions
        duplicate_session = sessions[-1]
        assert duplicate_session is not original_session
        # One synthetic creation head is prepended (Cas B).
        assert len(duplicate_session.actions) == len(original_session.actions) + 1
        assert duplicate_session.title.endswith(_("Copy"))
        orig_a = original_session.actions[-1]
        dup_a = duplicate_session.actions[-1]
        assert dup_a is not orig_a
        assert dup_a.kwargs["param"] is not orig_a.kwargs["param"]

    # --- scenario: duplicate clones data AND remaps UUIDs ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        original_session = history.history_sessions[-1]
        _select_tree_session(history, original_session)
        n_obj_before = len(panel.objmodel)
        n_sessions_before = len(history.history_sessions)
        history.duplicate_selected_entries()
        sessions = history.history_sessions
        assert len(sessions) == n_sessions_before + 1
        dup_session = sessions[-1]
        assert dup_session is not original_session
        assert dup_session.title.endswith(_("Copy"))
        assert len(panel.objmodel) > n_obj_before
        # The duplicate has an extra leading synthetic head (Cas B); compare
        # only the compute actions against the originals.
        dup_compute = [
            a for a in dup_session.actions if a.kind == HistoryAction.KIND_COMPUTE
        ]
        for orig_action, dup_action in zip(original_session.actions, dup_compute):
            for pstr in orig_action.state.selection:
                orig_uuids = set(orig_action.state.selection.get(pstr, []))
                dup_uuids = set(dup_action.state.selection.get(pstr, []))
                if orig_uuids and dup_uuids:
                    assert orig_uuids.isdisjoint(dup_uuids)

        # --- scenario: replay of duplicated session does NOT add output ---
        _select_tree_session(history, dup_session)
        n_before = len(panel.objmodel)
        n_history_before = len(history)
        history.replay_restore_actions(replay=True, restore_selection=False)
        assert len(panel.objmodel) == n_before
        assert len(history) == n_history_before

    # --- scenario: duplicated session preserves topological object order ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        _build_cascade_chain(panel, history)
        # Extra step so the moving_average output is captured in metadata.
        panel.objview.select_objects([4])
        panel.processor.run_feature(sips.derivative)
        original_ids = panel.objmodel.get_object_ids()
        original_titles = [panel.objmodel[uid].title for uid in original_ids]
        assert len(original_titles) >= 3
        sessions = history.history_sessions
        _select_tree_session(history, sessions[0])
        history.duplicate_selected_entries()
        groups = panel.objmodel.get_groups()
        assert len(groups) >= 2
        dup_group_id = get_uuid(groups[-1])
        dup_ids = [
            uid
            for uid in panel.objmodel.get_object_ids()
            if panel.objmodel.get_object_group_id(panel.objmodel[uid]) == dup_group_id
        ]
        dup_titles = [panel.objmodel[uid].title for uid in dup_ids]

        def _suffix(title: str) -> str:
            parts = title.split("|", 1)
            return parts[1].strip() if len(parts) > 1 else title.strip()

        orig_suffixes = [_suffix(t) for t in original_titles]
        dup_suffixes = [_suffix(t) for t in dup_titles]
        clonable = orig_suffixes[: len(dup_suffixes)]
        assert clonable == dup_suffixes

    # --- scenario: cas A — creation-rooted chain copies creation (no head) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        win.add_object(create_paracetamol_signal())  # records new_object creation
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        original_session = history.history_sessions[-1]
        assert original_session.actions[0].method_name == "new_object"
        _select_tree_session(history, original_session)
        n_obj_before = len(panel.objmodel)
        history.duplicate_selected_entries()
        dup_session = history.history_sessions[-1]
        assert dup_session is not original_session
        # Cas A: same number of actions (creation duplicated, NO synthetic head).
        assert len(dup_session.actions) == len(original_session.actions)
        dup_root = dup_session.actions[0]
        orig_root = original_session.actions[0]
        assert dup_root.method_name == "new_object"
        assert dup_root is not orig_root
        assert dup_root.uuid != orig_root.uuid
        # Objects were cloned.
        assert len(panel.objmodel) > n_obj_before
        # UUID remap: duplicated actions share no object uuid with originals.
        for orig_a, dup_a in zip(original_session.actions, dup_session.actions):
            for pstr in orig_a.state.selection:
                o = set(orig_a.state.selection.get(pstr, []))
                d = set(dup_a.state.selection.get(pstr, []))
                if o and d:
                    assert o.isdisjoint(d)

    # --- scenario: cas B — operation-rooted chain synthesizes head ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())  # NO creation recorded
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        original_session = history.history_sessions[-1]
        # Operation-rooted: no creation action at the head.
        assert original_session.actions[0].method_name != "new_object"
        _select_tree_session(history, original_session)
        history.duplicate_selected_entries()
        dup_session = history.history_sessions[-1]
        assert dup_session is not original_session
        # Cas B: exactly one synthetic head prepended.
        assert len(dup_session.actions) == len(original_session.actions) + 1
        head = dup_session.actions[0]
        assert head.kind == HistoryAction.KIND_UI
        assert head.method_name == "new_object"
        assert head.kwargs == {}
        assert head.state.selection == {}  # empty state (always compatible)
        assert len(head.output_uuids) == 1
        clone_uuid = head.output_uuids[0]
        # Head output registered in the panel mappings.
        assert history.action_output_uuids.get(head.uuid) == [clone_uuid]
        assert history.output_to_action.get(clone_uuid) == head.uuid
        # Clone materialized in the panel.
        assert clone_uuid in panel.objmodel.get_object_ids()
        # Head + ops re-link into a SINGLE chain on the duplicate.
        dup_chains = build_session_chains(history, dup_session)
        assert len(dup_chains) == 1
        assert dup_chains[0].root is head
        assert sum(len(c.actions) for c in dup_chains) == len(dup_session.actions)
        # Replay of the duplicate adds no new object (creations skipped under
        # suppression).
        _select_tree_session(history, dup_session)
        n_obj = len(panel.objmodel)
        n_hist = len(history)
        history.replay_restore_actions(replay=True, restore_selection=False)
        assert len(panel.objmodel) == n_obj
        assert len(history) == n_hist


# ---------------------------------------------------------------------------
# 4) Editing: processing-tab edits, cascade recompute & invariants
# ---------------------------------------------------------------------------


def test_history_editing_and_cascade():
    """Processing-tab edit propagation, step-by-step, cascade & invariants."""
    # --- scenario: edit updates current session action and refreshes html ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        param = sigima.params.GaussianParam.create(sigma=1.5)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.gaussian_filter, param)
        result_obj = panel.objmodel.get_object_from_number(2)
        action = history[len(history)]
        assert action.func_name == "gaussian_filter"
        assert action.kwargs["param"].sigma == 1.5
        html_before = action.description_html
        panel.objview.select_objects([2])
        assert panel.objprop.setup_processing_tab(result_obj, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 3.5
        report = panel.objprop.apply_processing_parameters(
            result_obj, interactive=False
        )
        assert report.success
        assert action.kwargs["param"].sigma == 3.5
        html_after = action.description_html
        assert html_before != html_after
        assert "3.5" in html_after
        win.historypanel.refresh_action(action)
        editor.dataset.sigma = 7.0
        assert action.kwargs["param"].sigma == 3.5

    # --- scenario: edit does NOT touch a past session ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.0)
        )
        past_action = history[len(history)]
        assert past_action.kwargs["param"].sigma == 1.0
        win.reset_all()
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=2.0)
        )
        new_obj = panel.objmodel.get_object_from_number(2)
        new_action = history[len(history)]
        assert new_action is not past_action
        found = history.find_action_for_output(get_uuid(new_obj), "gaussian_filter")
        assert found is new_action and found is not past_action
        panel.objview.select_objects([2])
        assert panel.objprop.setup_processing_tab(new_obj, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 4.0
        report = panel.objprop.apply_processing_parameters(new_obj, interactive=False)
        assert report.success
        assert new_action.kwargs["param"].sigma == 4.0
        assert past_action.kwargs["param"].sigma == 1.0

    # --- scenario: restore selection only (replay=False) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]
        derived = panel.objmodel.get_object_from_number(2)
        panel.objview.set_current_object(derived)
        derived_uuid = get_uuid(derived)
        assert panel.objview.get_sel_object_uuids() == [derived_uuid]
        _select_tree_item_for(history, deriv_entry)
        n_before = len(panel.objmodel)
        history.replay_restore_actions(replay=False, restore_selection=True)
        assert len(panel.objmodel) == n_before
        assert panel.objview.get_sel_object_uuids() == [src_uuid]

    # --- scenario: step-by-step launch commits edits immediately ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        norm_entry = history[len(history)]
        assert norm_entry.func_name == "normalize"
        previous_edit_mode = history.edit_mode
        _select_tree_item_for(history, norm_entry)
        # Step-by-step launch: parameter dialogs auto-accept in unattended mode.
        history.replay_step_by_step()
        assert history.edit_mode is previous_edit_mode
        assert history.has_any_pending_edits() is False

    # --- scenario: downstream detection ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        action_a, action_b, action_c, _ob, _oc = _build_cascade_chain(panel, history)
        downstream = history.get_downstream_actions(action_a)
        assert downstream == [action_b, action_c]
        assert not history.get_downstream_actions(action_c)
        assert history.get_downstream_actions(action_b) == [action_c]

    # --- scenario: cascade recompute updates downstream outputs in place ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        action_a, action_b, action_c, output_b, output_c = _build_cascade_chain(
            panel, history
        )
        uuid_b = get_uuid(output_b)
        uuid_c = get_uuid(output_c)
        data_b_before = output_b.xydata.copy()
        data_c_before = output_c.xydata.copy()
        n_objects_before = len(panel.objmodel)
        result_obj_a = panel.objmodel.get_object_from_number(2)
        panel.objview.select_objects([2])
        assert panel.objprop.setup_processing_tab(result_obj_a, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 6.0
        report = panel.objprop.apply_processing_parameters(
            result_obj_a, interactive=False
        )
        assert report.success
        assert action_a.kwargs["param"].sigma == 6.0
        assert len(panel.objmodel) == n_objects_before
        assert get_uuid(panel.objmodel[uuid_b]) == uuid_b
        assert get_uuid(panel.objmodel[uuid_c]) == uuid_c
        assert not np.array_equal(panel.objmodel[uuid_b].xydata, data_b_before)
        assert not np.array_equal(panel.objmodel[uuid_c].xydata, data_c_before)
        for a in (action_a, action_b, action_c):
            assert a.is_stale is False

    # --- scenario: play on stale action triggers cascade ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        action_a, action_b, _ac, output_b, _oc = _build_cascade_chain(panel, history)
        uuid_b = get_uuid(output_b)
        output_b.xydata = output_b.xydata * 0.0
        tampered = output_b.xydata.copy()
        action_a.is_stale = True
        _select_tree_item_for(history, action_a)
        history.replay_restore_actions(replay=True, restore_selection=False)
        assert action_a.is_stale is False
        assert action_b.is_stale is False
        assert not np.array_equal(panel.objmodel[uuid_b].xydata, tampered)

    # --- scenario: cascade in a duplicated session that is NOT [-1] ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        action_a, _ab, _ac, _ob, output_c = _build_cascade_chain(panel, history)
        uuid_c_orig = get_uuid(output_c)
        data_c_orig = output_c.xydata.copy()
        panel.objview.select_objects([4])
        panel.processor.run_feature(sips.derivative)
        sessions = history.history_sessions
        s1 = sessions[0]
        history.create_new_session()
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([6])
        panel.processor.run_feature(sips.derivative)
        assert len(sessions) == 2
        _select_tree_session(history, s1)
        history.duplicate_selected_entries()
        assert len(sessions) == 3
        dup_session = sessions[1]
        assert sessions[-1] is not dup_session
        dup_action_a = next(
            a for a in dup_session.actions if a.func_name == action_a.func_name
        )
        dup_action_c = next(
            a for a in dup_session.actions if a.func_name == "moving_average"
        )
        dup_output_c_uuid = history.action_output_uuid(dup_action_c)
        assert dup_output_c_uuid is not None
        data_dup_c_before = panel.objmodel[dup_output_c_uuid].xydata.copy()
        dup_result_obj_a_uuid = history.action_output_uuid(dup_action_a)
        assert dup_result_obj_a_uuid is not None
        dup_result_obj_a = panel.objmodel[dup_result_obj_a_uuid]
        panel.objview.select_objects([dup_result_obj_a_uuid])
        assert panel.objprop.setup_processing_tab(dup_result_obj_a, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 10.0
        report = panel.objprop.apply_processing_parameters(
            dup_result_obj_a, interactive=False
        )
        assert report.success
        assert not np.array_equal(
            panel.objmodel[dup_output_c_uuid].xydata, data_dup_c_before
        )
        assert np.array_equal(panel.objmodel[uuid_c_orig].xydata, data_c_orig)

    # --- creation-action editing (params, cascade, restore) ---
    from datalab.gui.panel.history import recompute as hrec

    # --- scenario: editing a created Gauss signal cascades to downstream ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        gparam = sigima.objects.create_signal_parameters(
            sigima.objects.SignalTypes.GAUSS, size=200, sigma=20.0
        )
        signal = panel.new_object(gparam, edit=False)
        assert signal is not None
        creation = history[len(history)]
        assert creation.kind == HistoryAction.KIND_UI
        assert creation.method_name == "new_object"
        creation_uuid = get_uuid(signal)
        assert history.action_output_uuid(creation) == creation_uuid

        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
        )
        filter_action = history[len(history)]
        output_filter = panel.objmodel.get_object_from_number(2)
        uuid_filter = get_uuid(output_filter)

        assert history.get_downstream_actions(creation) == [filter_action]

        creation_data_before = signal.xydata.copy()
        filter_data_before = output_filter.xydata.copy()
        desc_before = creation.description

        creation.snapshot_kwargs()
        creation.kwargs["param"].sigma = 60.0
        hrec.recompute_action_in_place(history, creation)
        hrec.recompute_cascade(history, creation, descendants=[filter_action])

        assert creation.has_pending_edits
        assert "60" in creation.description
        assert creation.description != desc_before
        assert not np.array_equal(signal.xydata, creation_data_before)
        assert not np.array_equal(
            panel.objmodel[uuid_filter].xydata, filter_data_before
        )

        creation.restore_kwargs()
        hrec.recompute_action_in_place(history, creation)
        hrec.recompute_cascade(history, creation, descendants=[filter_action])
        assert not creation.has_pending_edits
        assert creation.kwargs["param"].sigma == 20.0
        assert np.allclose(signal.xydata, creation_data_before)

    # --- scenario: edited snapshot survives HDF5 round-trip ---
    gparam = sigima.objects.create_signal_parameters(
        sigima.objects.SignalTypes.GAUSS, size=64, sigma=20.0
    )
    action = HistoryAction(
        title="New signal",
        kind=HistoryAction.KIND_UI,
        target="signalpanel",
        method_name="new_object",
        kwargs={"param": gparam},
        state=WorkspaceState(),
    )
    action.snapshot_kwargs()
    action.kwargs["param"].sigma = 50.0
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "creation_roundtrip.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        with NativeH5Reader(path) as reader:
            restored_sessions = reader.read_object_list(
                "history_session", HistorySession
            )
    restored_action = restored_sessions[0].actions[0]
    assert restored_action.has_pending_edits
    assert restored_action.kwargs["param"].sigma == 50.0

    # --- cascade-recompute characterization invariants (pre-P3 slimming) ---
    # --- scenario: cascade recompute call-count is bounded (no infinite loop) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        action_a, action_b, action_c, output_b, output_c = _build_cascade_chain(
            panel, history
        )
        output_a = panel.objmodel.get_object_from_number(2)
        src_obj = panel.objmodel.get_object_from_number(1)
        src_data_before = src_obj.xydata.copy()
        number_b = panel.objmodel.get_number(output_b)
        number_c = panel.objmodel.get_number(output_c)

        # Spy on the single shared signal processor recompute entry point.
        calls = {"n": 0}
        original_recompute = panel.processor.recompute_1_to_1

        def _counting_recompute(*args, **kwargs):
            calls["n"] += 1
            return original_recompute(*args, **kwargs)

        panel.processor.recompute_1_to_1 = _counting_recompute
        try:
            panel.objview.select_objects([2])
            assert panel.objprop.setup_processing_tab(output_a, reset_params=False)
            editor = panel.objprop.processing_param_editor
            assert editor is not None
            editor.dataset.sigma = 7.0
            report = panel.objprop.apply_processing_parameters(
                output_a, interactive=False
            )
            assert report.success
        finally:
            panel.processor.recompute_1_to_1 = original_recompute

        # Editing the root re-runs: the root itself (via apply) + the two
        # downstream 1-to-1 actions (via cascade). Exactly one call per
        # affected action -- a bound that fails if a replay loop reappears.
        assert calls["n"] == 3  # root apply + 2 downstream cascade recomputes

        # Identity invariants: outputs keep their UUID *and* their __number.
        assert get_uuid(panel.objmodel[get_uuid(output_b)]) == get_uuid(output_b)
        assert panel.objmodel.get_number(panel.objmodel[get_uuid(output_b)]) == number_b
        assert panel.objmodel.get_number(panel.objmodel[get_uuid(output_c)]) == number_c

        # Source object is never mutated by a downstream edit.
        assert np.array_equal(src_obj.xydata, src_data_before)

    # --- scenario: cascade recompute preserves output metadata ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        panel = win.signalpanel
        action_a, _action_b, _action_c, _output_b, output_c = _build_cascade_chain(
            panel, history
        )
        uuid_c = get_uuid(output_c)
        # Add a user metadata sentinel on the deepest output.
        panel.objmodel[uuid_c].metadata["sentinel_p3a"] = 123
        output_a = panel.objmodel.get_object_from_number(2)
        panel.objview.select_objects([2])
        assert panel.objprop.setup_processing_tab(output_a, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 9.0
        report = panel.objprop.apply_processing_parameters(output_a, interactive=False)
        assert report.success
        # P3b: the cascade now routes through the shared pre-history primitive
        # (``apply_recomputed_object_in_place``), which PRESERVES the object's
        # metadata instead of clearing it. The user sentinel survives.
        assert panel.objmodel[uuid_c].metadata.get("sentinel_p3a") == 123


# ---------------------------------------------------------------------------
# 5) Navigation, stepping & processing chains
# ---------------------------------------------------------------------------


def test_history_navigation_and_chains(monkeypatch):
    """Stepping/selection sync, chain reconnect/grouping/split & sessions."""
    # --- scenario: step_next walks forward through current session ---
    with datalab_test_app_context() as win:
        _panel, history = _record_three_action_session(win)
        sessions = history.history_sessions
        actions = sessions[-1].actions
        assert len(actions) >= 2
        history.tree.clearSelection()
        for action in actions:
            history.step_next()
            current = history.tree.currentItem()
            assert current is not None
            assert current.data(0, QC.Qt.UserRole) == action.uuid
        # End -> no-op.
        last_uuid = history.tree.currentItem().data(0, QC.Qt.UserRole)
        history.step_next()
        assert history.tree.currentItem().data(0, QC.Qt.UserRole) == last_uuid

        # --- scenario: step_prev walks backward through current session ---
        _select_tree_item_for(history, actions[-1])
        for expected in reversed(actions[:-1]):
            history.step_prev()
            current = history.tree.currentItem()
            assert current.data(0, QC.Qt.UserRole) == expected.uuid
        first_uuid = history.tree.currentItem().data(0, QC.Qt.UserRole)
        history.step_prev()
        assert history.tree.currentItem().data(0, QC.Qt.UserRole) == first_uuid

    # --- scenario: step button enabled state reflects position ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        prev_btn = history.step_prev_action
        next_btn = history.step_next_action
        history.update_actions_state()
        assert not prev_btn.isEnabled()
        assert not next_btn.isEnabled()
        _panel, history = _record_three_action_session(win)
        sessions = history.history_sessions
        actions = sessions[-1].actions
        prev_btn = history.step_prev_action
        next_btn = history.step_next_action
        _select_tree_item_for(history, actions[0])
        assert not prev_btn.isEnabled()
        assert next_btn.isEnabled()
        if len(actions) >= 3:
            _select_tree_item_for(history, actions[1])
            assert prev_btn.isEnabled()
            assert next_btn.isEnabled()
        _select_tree_item_for(history, actions[-1])
        assert prev_btn.isEnabled()
        assert not next_btn.isEnabled()

    # --- scenario: selecting a compute action selects its output ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        out_uuid = get_uuid(panel.objmodel.get_object_from_number(2))
        norm_entry = history[len(history)]
        assert norm_entry.func_name == "normalize"
        assert src_uuid in norm_entry.state.selection.get("signal", [])
        panel.objview.select_objects([src_uuid])
        _select_tree_item_for(history, norm_entry)
        assert panel.objview.get_sel_object_uuids() == [out_uuid]

        # --- scenario: deleted output -> selection falls back to input ---
        panel.objview.select_objects([2])
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 1
        panel.objview.select_groups([1])
        _select_tree_item_for(history, norm_entry)
        assert panel.objview.get_sel_object_uuids() == [src_uuid]

    # --- scenario: deleting intermediate result rewires downstream to source ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
        )
        s002_uuid = get_uuid(panel.objmodel.get_object_from_number(2))
        action_gaussian = history[len(history)]
        assert action_gaussian.func_name == "gaussian_filter"

        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        action_deriv = history[len(history)]
        assert action_deriv.func_name == "derivative"
        assert s002_uuid in action_deriv.state.selection.get("signal", [])

        panel.objview.select_objects([2])
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 2

        reconnected_uuids = action_deriv.state.selection.get("signal", [])
        assert s002_uuid not in reconnected_uuids
        assert src_uuid in reconnected_uuids

    # --- scenario: chain read-model groups actions into ordered chains ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        # Creation via the main-window proxy records a ``new_object`` action and
        # registers its output UUID. ``panel.add_object`` deliberately does not
        # record, so it is used below to exercise an externally-rooted chain.
        win.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)

        # --- invariant: grouping drops no action, per session ---
        grouped = build_processing_chains(history)
        assert len(grouped) == len(history.history_sessions)
        for sess, sess_chains in grouped:
            assert isinstance(sess, HistorySession)
            assert sum(len(c.actions) for c in sess_chains) == len(sess.actions)

        # --- creation-rooted chain: creation -> normalize -> derivative ---
        all_chains = [c for _sess, chains in grouped for c in chains]
        creation_chains = [
            c
            for c in all_chains
            if c.root.kind == HistoryAction.KIND_UI
            and c.root.method_name in HistoryAction.UI_CREATION_METHODS
        ]
        assert len(creation_chains) == 1
        chain = creation_chains[0]
        assert isinstance(chain, ProcessingChain)
        assert chain.actions[0] is chain.root
        assert [a.func_name for a in chain.actions] == [None, "normalize", "derivative"]
        assert all(a in chain.session.actions for a in chain.actions)

        # --- externally-rooted chain: a fresh session started on a
        # non-recorded object (panel.add_object records no creation) roots its
        # own (compute) chain, since a session's root is its first action ---
        history.create_new_session(panel_str="signal")
        panel.add_object(create_paracetamol_signal())
        panel.objview.select_objects([len(panel.objmodel)])
        panel.processor.run_feature(sips.derivative)
        grouped2 = build_processing_chains(history)
        for sess, sess_chains in grouped2:
            assert sum(len(c.actions) for c in sess_chains) == len(sess.actions)
        external_chains = [
            c
            for _sess, chains in grouped2
            for c in chains
            if c.root.kind == HistoryAction.KIND_COMPUTE
        ]
        assert external_chains

        # Keep build_session_chains imported — it is public API under test
        assert callable(build_session_chains)

    # --- scenario: deleting an intermediate action splits its chain ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        # Creation via the main window records a ``new_object`` creation action;
        # ``panel.add_object`` deliberately does not, so ``win.add_object`` is
        # required to obtain a creation-rooted chain.
        win.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)

        session = history.history_sessions[-1]
        assert [a.func_name for a in session.actions] == [
            None,
            "normalize",
            "derivative",
        ]
        normalize_action = session.actions[1]
        derivative_action = session.actions[2]
        s002_uuid = get_uuid(panel.objmodel.get_object_from_number(2))
        assert s002_uuid in derivative_action.state.selection.get("signal", [])
        n_objects_before = len(panel.objmodel)

        # Delete the MIDDLE action (normalize). Unattended mode auto-confirms and
        # skips the associated-object removal prompt.
        _select_tree_item_for(history, normalize_action)
        history.delete_selected()

        # ``normalize`` is spliced out; creation and ``derivative`` remain.
        assert normalize_action not in session.actions
        assert session.actions[0].method_name in HistoryAction.UI_CREATION_METHODS
        assert derivative_action in session.actions

        # A copy of ``normalize``'s output was added as a new parentless root.
        assert len(panel.objmodel) == n_objects_before + 1

        # The session remains a single linear chain (the history model does not
        # split a session on delete) and drops no action.
        chains = build_session_chains(history, session)
        assert len(chains) == 1
        assert sum(len(c.actions) for c in chains) == len(session.actions)

        # The observable "split" is a data-level rewire: ``derivative`` no longer
        # consumes ``normalize``'s original output (s002) but the parentless COPY
        # added by the delete.
        deriv_inputs = derivative_action.state.selection.get("signal", [])
        assert s002_uuid not in deriv_inputs

    # --- new-session prompt when creating on a non-empty session ---
    # --- scenario: user accepts the prompt -> a new session is opened ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)

        # First creation reuses the (empty) current session: no prompt expected.
        win.add_object(create_paracetamol_signal())
        assert len(history.history_sessions) == 1
        assert len(history.history_sessions[-1].actions) == 1

        # Headless "accept": accept_dialogs drives the prompt to "Yes".
        monkeypatch.setattr(execenv, "accept_dialogs", True)
        n_sessions = len(history.history_sessions)
        win.add_object(create_paracetamol_signal())
        assert len(history.history_sessions) == n_sessions + 1
        new_session = history.history_sessions[-1]
        assert new_session.actions
        assert new_session.actions[0].method_name == "new_object"

    # --- scenario: user declines the prompt -> no new session is created ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        win.add_object(create_paracetamol_signal())
        assert len(history.history_sessions) == 1

        # Headless "decline": accept_dialogs stays False -> prompt answered "No".
        monkeypatch.setattr(execenv, "accept_dialogs", False)
        n_sessions = len(history.history_sessions)
        win.add_object(create_paracetamol_signal())
        assert len(history.history_sessions) == n_sessions

    # --- scenario: recording routes per-panel into separate active sessions ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        spanel, ipanel = win.signalpanel, win.imagepanel

        # --- signal pipeline: one signal session ---
        spanel.add_object(create_paracetamol_signal())
        spanel.objview.select_objects([1])
        spanel.processor.run_feature(sips.derivative)
        assert len(history.history_sessions) == 1
        signal_session = history.get_active_session("signal")
        assert signal_session is not None
        assert len(signal_session.actions) == 1

        # --- image pipeline: lands in a SEPARATE session ---
        ipanel.add_object(create_sincos_image())
        ipanel.objview.select_objects([1])
        ipanel.processor.run_feature(sipi.inverse)
        image_session = history.get_active_session("image")
        assert image_session is not None
        assert image_session is not signal_session
        assert len(history.history_sessions) == 2
        assert len(image_session.actions) == 1
        # The image action did not leak into the signal session.
        assert len(signal_session.actions) == 1

        # --- back to signal: continue in the SAME signal session (no new one) ---
        spanel.objview.select_objects([2])
        spanel.processor.run_feature(sips.derivative)
        assert len(history.history_sessions) == 2  # no new session created
        assert len(signal_session.actions) == 2
        assert len(image_session.actions) == 1

        # --- resume into a SELECTED session ---
        # Create a fresh (empty) signal session, then select the ORIGINAL signal
        # session: a subsequent signal action must resume into the selected
        # session, not the newly created empty one.
        empty_session = history.create_new_session(panel_str="signal")
        assert len(history.history_sessions) == 3
        n_before = len(signal_session.actions)
        _select_tree_session(history, signal_session)
        spanel.objview.select_objects([1])
        spanel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        assert len(signal_session.actions) == n_before + 1  # resumed here
        assert len(empty_session.actions) == 0  # NOT the empty session
        assert len(image_session.actions) == 1  # image untouched

    # --- scenario: each panel's active session is highlighted (bold) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        spanel, ipanel = win.signalpanel, win.imagepanel

        spanel.add_object(create_paracetamol_signal())
        spanel.objview.select_objects([1])
        spanel.processor.run_feature(sips.derivative)
        signal_session = history.get_active_session("signal")
        assert signal_session is not None

        ipanel.add_object(create_sincos_image())
        ipanel.objview.select_objects([1])
        ipanel.processor.run_feature(sipi.inverse)
        image_session = history.get_active_session("image")
        assert image_session is not None

        def _is_bold(session):
            idx = history.history_sessions.index(session)
            item = history.tree.topLevelItem(idx)
            return item is not None and item.font(0).bold()

        # Both panels' active sessions are highlighted.
        assert _is_bold(signal_session)
        assert _is_bold(image_session)

        # Highlight survives a full repopulate.
        history.tree.populate_tree(history.history_sessions)
        assert _is_bold(signal_session)
        assert _is_bold(image_session)

        # Creating a new signal session moves the signal highlight to it.
        new_signal = history.create_new_session(panel_str="signal")
        assert _is_bold(new_signal)
        assert not _is_bold(signal_session)  # previous signal session no longer active
        assert _is_bold(image_session)  # image highlight unchanged

        # Switching panels does not crash and keeps the highlight.
        win.set_current_panel("image")
        win.set_current_panel("signal")
        assert _is_bold(new_signal)
        assert _is_bold(image_session)

    # --- scenario: compatibility refresh tolerates a stale tree (regression) ---
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        spanel = win.signalpanel
        spanel.add_object(create_paracetamol_signal())
        spanel.objview.select_objects([1])
        spanel.processor.run_feature(sips.derivative)
        spanel.objview.select_objects([2])
        spanel.processor.run_feature(sips.derivative)
        # Tree is populated and in sync here.
        session = history.history_sessions[-1]
        assert session.actions
        # Simulate a transient desync: drop the last action from the model
        # WITHOUT repopulating the tree (as happens mid-cascade).
        removed = session.actions.pop()
        # The tree still has an item for ``removed``; a compatibility refresh
        # must skip it gracefully instead of raising ValueError.
        history.tree.update_compatibility_states(
            history.history_sessions, history.mainwindow
        )
        # Restore invariant for a clean teardown.
        session.actions.append(removed)
