# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History application test

Exercises the History panel's full feature set:

* Record-mode toggle gating
* Entry recording for object creation, 1-to-1, 1-to-0, n-to-1 and 2-to-1
  processing patterns (covering ``BaseProcessor.compute_*`` and
  ``BaseDataPanel`` history-emitting methods)
* Workspace state attached to processing entries
* Session creation and session-aware indexing
* Replay of a recorded action
* Action deletion (cascade within a session)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.objects
import sigima.params
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from sigima.tests.data import create_paracetamol_signal

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import HistoryAction, HistorySession, WorkspaceState
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def _entry_titles(history) -> list[str]:
    """Return the list of recorded entry titles, in chronological order."""
    return [action.title for action in history]


def _session_action_counts(history) -> list[int]:
    """Return the number of recorded actions in each history session."""
    return [len(session.actions) for session in history.history_sessions]


def test_history_app():
    """Run history application test scenario"""
    with datalab_test_app_context() as win:
        history = win.historypanel
        dock = win.docks[history]
        win.addDockWidget(QC.Qt.LeftDockWidgetArea, dock)
        win.resize(int(win.width() * 1.7), win.height())
        win.move(50, 50)
        execenv.print("History application test:")

        panel = win.signalpanel

        # --- Record mode is OFF by default: nothing should be recorded ---------
        assert len(history) == 0
        panel.add_object(create_paracetamol_signal())
        panel.processor.run_feature(sips.derivative)
        assert len(history) == 0, (
            "Record mode is disabled: no entry should have been recorded"
        )

        # Reset workspace before starting the recorded scenario.
        # No history exists yet, so this must not create an empty session.
        win.reset_all()
        assert len(history) == 0
        assert _session_action_counts(history) == []

        # --- Enable record mode and start recording ----------------------------
        history.toggle_record_mode(True)

        # Pre-populate two real signals (``add_object`` does not record entries).
        panel.add_object(create_paracetamol_signal())
        panel.add_object(create_paracetamol_signal())
        assert len(history) == 0

        # 1) Object creation through the GUI path (BaseDataPanel.new_object):
        # save_state=False, no workspace state is captured.
        panel.new_object()
        assert len(history) == 1
        creation_entry = history[1]
        assert isinstance(creation_entry, HistoryAction)
        assert creation_entry.title == _("New signal")
        assert creation_entry.state.selection == {}
        assert creation_entry.state.states == {}

        # 2) 1-to-1 processing (compute_1_to_1) on signal #1: save_state=True
        panel.objview.select_objects([1])
        obj1_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.processor.run_feature(sips.derivative)
        assert len(history) == 2
        deriv_entry = history[2]
        assert deriv_entry.title  # title must be non-empty
        # Workspace state must remember the single-object selection (by UUID)
        assert deriv_entry.state.selection.get(panel.PANEL_STR_ID) == [obj1_uuid]
        assert len(deriv_entry.state.states.get(panel.PANEL_STR_ID, [])) == 1

        # 3) 1-to-1 with parameters (compute_1_to_1 + DataSet param)
        norm_param = sigima.params.NormalizeParam.create(method="maximum")
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.normalize, norm_param)
        assert len(history) == 3
        norm_entry = history[3]
        # The recorded kwargs must include the parameter (used in description)
        assert any(k.endswith("param") for k in norm_entry.kwargs)
        assert norm_entry.description  # description is built from the param

        # 4) 1-to-0 analysis (compute_1_to_0)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.fwhm, sigima.params.FWHMParam())
        assert len(history) == 4
        fwhm_entry = history[4]
        assert fwhm_entry.state.selection.get(panel.PANEL_STR_ID) == [obj1_uuid]

        # 5) n-to-1 aggregation (compute_n_to_1) on signals #1 and #2
        obj2 = panel.objmodel.get_object_from_number(2)
        obj2_uuid = get_uuid(obj2)
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(sips.average)
        assert len(history) == 5
        avg_entry = history[5]
        assert sorted(avg_entry.state.selection.get(panel.PANEL_STR_ID, [])) == sorted(
            [obj1_uuid, obj2_uuid]
        )

        # 6) 2-to-1 binary operation (compute_2_to_1)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.difference, obj2)
        assert len(history) == 6
        diff_entry = history[6]
        assert diff_entry.state.selection.get(panel.PANEL_STR_ID) == [obj1_uuid]

        # --- Iteration / indexing API ------------------------------------------
        all_titles = _entry_titles(history)
        assert len(all_titles) == 6
        assert all_titles[0] == _("New signal")
        # Indexing is 1-based; iteration order matches index order
        assert history[1] is creation_entry
        assert history[6] is diff_entry

        # --- Sessions ----------------------------------------------------------
        history.create_new_session()
        # New session does not change the action count
        assert len(history) == 6
        before = len(history)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.absolute)
        after = len(history)
        assert after == before + 1
        new_session_entry = history[after]
        assert new_session_entry.title

        # --- Replay ------------------------------------------------------------
        # Replaying the derivative entry must produce a new object without raising.
        n_objects_before_replay = len(panel.objmodel)
        deriv_entry.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) > n_objects_before_replay

        # --- Record mode OFF stops further recording ---------------------------
        count_before_off = len(history)
        history.toggle_record_mode(False)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.absolute)
        assert len(history) == count_before_off, (
            "Record mode disabled: subsequent operations must not be recorded"
        )

        # --- Workspace state types are well-formed -----------------------------
        for action in history:
            assert isinstance(action, HistoryAction)
            assert isinstance(action.state, WorkspaceState)
            assert isinstance(action.title, str)
            assert isinstance(action.dtstr, str) and action.dtstr

        # --- Delete cascade within a session -----------------------------------
        # ``delete_selected`` itself opens a confirmation dialog; we exercise the
        # underlying ``HistorySession.remove_action`` path used by it.
        target = new_session_entry
        target_session: HistorySession | None = None

        for session in history.history_sessions:
            if target in session.actions:
                target_session = session
                break
        assert target_session is not None

        n_before = len(history)
        target_session.remove_action(target)
        assert len(history) < n_before

        execenv.print("==> OK")


if __name__ == "__main__":
    test_history_app()
