# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel replay and cross-panel navigation contracts."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.tests.data import create_paracetamol_signal, create_sincos_image

from datalab.gui import historytools_ops as htools
from datalab.gui.panel.history import HistoryTree
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history.ui import HistoryPanelUI
from datalab.history.action import HistoryAction
from datalab.history.session import HistorySession
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    build_signal_chain,
    get_tree_item,
    is_session_bold,
    select_tree_entry,
)


@pytest.mark.parametrize("column", (0, 2))
@pytest.mark.parametrize("selected_kind", ("action", "session"))
def test_history_tree_double_click_replays_current_selection_without_restoring(
    selected_kind: str, column: int
) -> None:
    """Replay the current action or session selection from either tree column."""
    if selected_kind == "action":
        selected_row: HistoryAction | HistorySession = HistoryAction()
        expected_actions = [selected_row]
    else:
        selected_row = HistorySession()
        selected_row.add_action(HistoryAction())
        selected_row.add_action(HistoryAction())
        expected_actions = list(selected_row.actions)
    selected_row.is_current_state_compatible = Mock(return_value=True)
    clicked_row = HistoryAction()
    tree = SimpleNamespace(
        customContextMenuRequested=Mock(),
        itemDoubleClicked=Mock(),
        itemSelectionChanged=Mock(),
        get_selected_actions_or_sessions=Mock(return_value=[selected_row]),
    )
    mainwindow = object()
    panel = SimpleNamespace(
        tree=tree,
        history_sessions=[],
        mainwindow=mainwindow,
        refresh_compatibility_items=Mock(),
        replaying=nullcontext,
        output_suppressed=nullcontext,
        runtime=SimpleNamespace(
            execution=SimpleNamespace(
                edit_mode=False,
                is_busy=lambda: False,
                is_engine_busy=lambda: False,
            )
        ),
        navigation=SimpleNamespace(
            sync_panel_selection=Mock(),
            update_state_widget=Mock(),
            set_active_session_from_selection=Mock(),
        ),
    )
    panel.replay_restore_actions = lambda **kwargs: hireplay.replay_restore_actions(
        panel, **kwargs
    )
    ui = HistoryPanelUI.__new__(HistoryPanelUI)
    ui.panel = panel

    ui.setup_connections()
    double_click_slot = tree.itemDoubleClicked.connect.call_args.args[0]
    with patch.object(hireplay, "replay_actions") as replay_actions_mock:
        double_click_slot(clicked_row, column)

    selected_row.is_current_state_compatible.assert_called_once_with(mainwindow)
    replay_actions_mock.assert_called_once_with(panel, expected_actions, prompt=False)
    assert clicked_row not in replay_actions_mock.call_args.args[1]


def test_panel_replay_restores_selection_without_outputs() -> None:
    """Use panel replay to restore selection without recording or new output."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        source_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        action = history[len(history)]
        output_uuid = action.output_uuids[0]
        select_tree_entry(history, action.uuid)
        assert panel.objview.get_sel_object_uuids() == [output_uuid]
        object_count, action_count = len(panel.objmodel), len(history)
        history.replay_restore_actions(replay=True, restore_selection=True)
        # In-place recompute keeps counts unchanged and selects the refreshed output
        assert (
            len(panel.objmodel),
            len(history),
            panel.objview.get_sel_object_uuids(),
        ) == (object_count, action_count, [output_uuid])
        assert action in history.history_sessions[-1].actions and (
            history.runtime.objects.action_output_uuids[action.uuid] == [output_uuid]
        )
        assert (
            output_uuid in panel.objmodel.get_object_ids()
            and history.runtime.objects.output_to_action[output_uuid] == action.uuid
        )
        panel.objview.select_objects([output_uuid])
        panel.remove_object(force=True)
        assert (
            action in history.history_sessions[-1].actions
            and output_uuid not in panel.objmodel.get_object_ids()
            and action.uuid not in history.runtime.objects.action_output_uuids
            and output_uuid not in history.runtime.objects.output_to_action
        )
        select_tree_entry(history, action.uuid)
        history.replay_restore_actions(replay=False, restore_selection=True)
        assert panel.objview.get_sel_object_uuids() == [source_uuid]


def test_cross_panel_sessions_navigation_and_tree_state() -> None:
    """Coordinate active sessions, navigation, tree state and selection fallback."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        signal_panel, image_panel = win.signalpanel, win.imagepanel
        history.toggle_record_mode(True)
        signal_chain = build_signal_chain(signal_panel, history)
        first_signal_action, middle_signal_action, last_signal_action = (
            signal_chain.actions
        )
        signal_uuid = first_signal_action.state.selection["signal"][0]
        signal_session = next(
            session
            for session in history.history_sessions
            if first_signal_action in session.actions
        )
        assert all(action in signal_session.actions for action in signal_chain.actions)
        navigation_states = []
        select_tree_entry(history, first_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        select_tree_entry(history, middle_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        select_tree_entry(history, last_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        assert navigation_states == [(False, True), (True, True), (True, False)]
        image_panel.add_object(create_sincos_image())
        image_panel.objview.select_objects([1])
        image_panel.processor.run_feature(sipi.inverse)
        image_action = history[len(history)]
        # Unified model: the image action is chained into the single active
        # recording session, alongside the signal actions.
        assert history.navigation.get_active_session() is signal_session
        assert image_action in signal_session.actions
        bold_before = is_session_bold(history, signal_session)
        history.tree.populate_tree(history.history_sessions)
        assert bold_before is True
        assert is_session_bold(history, signal_session) is True
        output_uuid = first_signal_action.output_uuids[0]
        select_tree_entry(history, first_signal_action.uuid)
        assert signal_panel.objview.get_sel_object_uuids() == [output_uuid]
        signal_panel.objview.select_objects([output_uuid])
        signal_panel.remove_object(force=True)
        history.toggle_record_mode(False)
        image_panel.objview.select_objects([1])
        image_panel.remove_object(force=True)
        history.refresh_compatibility_items()
        tree_action_uuids = set()
        iterator = QW.QTreeWidgetItemIterator(history.tree)
        while iterator.value():
            uuid = iterator.value().data(0, QC.Qt.UserRole)
            if uuid is not None:
                tree_action_uuids.add(uuid)
            iterator += 1
        image_item = get_tree_item(history, image_action.uuid)
        assert (
            all(
                first_signal_action not in session.actions
                for session in history.history_sessions
            )
            and middle_signal_action in signal_session.actions
            and last_signal_action in signal_session.actions
            and middle_signal_action.state.selection["signal"] == [signal_uuid]
            and first_signal_action.uuid
            not in history.runtime.objects.action_output_uuids
            and output_uuid not in history.runtime.objects.output_to_action
            and first_signal_action.uuid not in tree_action_uuids
            and {middle_signal_action.uuid, last_signal_action.uuid}.issubset(
                tree_action_uuids
            )
            and image_item.data(0, HistoryTree.COMPATIBILITY_ROLE) is False
            and image_item.foreground(0).color().isValid()
            and image_item.data(0, QC.Qt.UserRole) == image_action.uuid
        )
        # Remove-incompatible tool purges flagged actions and keeps the rest
        incompatible = [
            action
            for session in history.history_sessions
            for action in session.actions
            if not action.is_current_state_compatible(win)
        ]
        assert image_action in incompatible
        htools.remove_incompatible_actions(history)
        remaining = [
            action for session in history.history_sessions for action in session.actions
        ]
        assert not [action for action in incompatible if action in remaining]
        assert all(action.is_current_state_compatible(win) for action in remaining)
        assert all(session.actions for session in history.history_sessions)
        # Second run: everything is compatible, nothing changes
        htools.remove_incompatible_actions(history)
        assert [
            action for session in history.history_sessions for action in session.actions
        ] == remaining
