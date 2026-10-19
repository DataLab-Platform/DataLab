# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Appending a workspace file that holds provenance to the current workspace.

The limitation is shown before anything changes (in the question about clearing
the workspace, or in a single warning when that question is not asked), the
current ledger is kept, appended objects never inherit what was recorded for
another object, and their later processing is recorded.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import os.path as osp
import shutil

import h5py
import pytest
import sigima.params
import sigima.proc.signal as sips
from datalab_capsule.inspection import IDENTITY_LIMIT
from datalab_capsule.recorder import APPEND_NOTICE
from qtpy import QtWidgets as QW

import datalab.gui.h5io
import datalab.gui.main
from datalab.config import Conf
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.processingcall import resolve_processing_call
from datalab.tests import helpers
from datalab.tests.features.common.provenance_inspection_unit_test import view_of
from datalab.tests.features.common.provenance_unit_test import app
from datalab.tests.features.hdf5.provenance_roundtrip_unit_test import (
    _run,
    _save_chain,
    _select,
    build_chain,
)


def _objects(win) -> list:
    return win.signalpanel.objmodel.get_all_objects()


def _without_block(path: str, target: str) -> str:
    shutil.copy(path, target)
    with h5py.File(target, "r+") as h5file:
        del h5file["DataLab_Provenance"]
    return target


class _Dialogs:
    """Record the questions and warnings shown while opening files."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answer, button: str) -> None:
        self.events: list[str] = []
        self.messages: list[str] = []

        def question(_parent, _title, message, *_args):
            self.events.append("question")
            self.messages.append(message)
            return answer

        def exec_dialog(box):
            self.events.append("warning")
            self.messages.append(box.text())
            if button == "add":
                (add,) = [
                    b
                    for b in box.buttons()
                    if box.buttonRole(b) == QW.QMessageBox.AcceptRole
                ]
                add.click()
            else:
                box.button(QW.QMessageBox.Cancel).click()
            # guidata's exec_dialog deletes the executed dialog. This answered box
            # was never shown: it refuses to close and, kept by its parent window,
            # would stop the unattended quit of later application tests. Without
            # a parent, it is deleted with its last reference.
            box.setParent(None)

        def getopenfilenames(*_args, **_kwargs):
            self.events.append("files")
            return self.files, ""

        self.files: list[str] = []
        monkeypatch.setattr(QW.QMessageBox, "question", staticmethod(question))
        monkeypatch.setattr(datalab.gui.h5io, "exec_dialog", exec_dialog)
        monkeypatch.setattr(datalab.gui.main, "getopenfilenames", getopenfilenames)
        monkeypatch.setattr(execenv, "unattended", False)


@pytest.fixture(name="ask")
def fixture_ask():
    """Ask whether to clear the workspace (restored after the test)."""
    previous = Conf.h5_clear_workspace_ask.get()
    yield
    Conf.h5_clear_workspace_ask.set(previous)


def test_append_never_attributes_recorded_provenance() -> None:
    """The appended copy of a deleted object starts a new history."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        with app() as win:
            objs = build_chain(win)
            win.save_h5_workspace(path)
            s1_uuid = get_uuid(objs["S1"])
            _select(win, objs["S1"])
            win.signalpanel.remove_object(force=True)
            activities = win.provenance.ledger.activities
            win.load_h5_workspace([path], reset_all=False)
            assert win.provenance.ledger.activities == activities
            assert win.provenance.notices[-1] == APPEND_NOTICE
            copy = win.find_object_by_uuid(s1_uuid)
            assert copy is not None and copy is not objs["S1"]
            view = win.provenance.describe_object(s1_uuid, check_data=True)
            assert view["production"] is None
            assert IDENTITY_LIMIT in view["limits"]
            # S2 was computed from the deleted object, never from its copy.
            (source,) = win.provenance.describe_object(get_uuid(objs["S2"]))[
                "production"
            ]["inputs"]
            assert source["object_present"] is False
            # Verifying S2 never uses the copy (same identifier and content).
            report = win.verify_provenance_activity(
                win.provenance.describe_object(get_uuid(objs["S2"]))["production"][
                    "activity_id"
                ]
            )
            assert report["verdict"] == "not_verified"
            assert report["eligibility"] == "missing_input"
            call = resolve_processing_call(win, objs["S2"])
            assert call.origin == "ledger" and call.inputs[0].obj is None
            assert call.missing_inputs and call.reason
            # The provenance view does not name the copy as the input of S2.
            view = view_of(win, objs["S2"])
            assert "1. source: Original input missing" in view.text()
            goto = [entry for entry in view.links if entry["action"] == "goto"]
            assert get_uuid(copy) not in [entry["payload"] for entry in goto]
            # The appended copy is processed and recorded as usual.
            _select(win, copy)
            out = _run(
                win,
                sips.normalize,
                sigima.params.NormalizeParam.create(method="maximum"),
            )
            (used,) = win.provenance.describe_object(get_uuid(out))["production"][
                "inputs"
            ]
            assert used["object_present"] is True
            assert resolve_processing_call(win, out).inputs[0].obj is copy
            win.provenance.ledger.validate()
            assert win.provenance.capture_failures == 0


@pytest.mark.usefixtures("ask")
def test_question_states_the_limitation_first(monkeypatch) -> None:
    """Files are chosen first; the question alone states the limitation."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        plain = _without_block(path, osp.join(tmpdir, "plain.h5"))
        with app() as win:
            build_chain(win)
            activities = win.provenance.ledger.activities
            count = len(_objects(win))
            Conf.h5_clear_workspace_ask.set(True)
            dialogs = _Dialogs(monkeypatch, QW.QMessageBox.No, "cancel")
            dialogs.files = [path]
            win.open_h5_files(import_all=True)
            assert dialogs.events == ["files", "question"]
            assert datalab.gui.h5io.append_provenance_notice() in dialogs.messages[0]
            assert len(_objects(win)) == 2 * count
            assert win.provenance.ledger.activities == activities
            assert win.h5inputoutput.append_informed is False
            # Without provenance in the file, the question is unchanged.
            dialogs.files = [plain]
            win.open_h5_files(import_all=True)
            assert dialogs.events[2:] == ["files", "question"]
            notice = datalab.gui.h5io.append_provenance_notice()
            assert notice not in dialogs.messages[1]


@pytest.mark.usefixtures("ask")
@pytest.mark.parametrize("button", ["cancel", "add"])
def test_single_warning_when_not_asked(monkeypatch, button) -> None:
    """Without the question, one warning comes before any change."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        with app() as win:
            build_chain(win)
            count = len(_objects(win))
            Conf.h5_clear_workspace_ask.set(False)
            dialogs = _Dialogs(monkeypatch, None, button)
            win.open_h5_files([path, path], import_all=True, reset_all=False)
            assert dialogs.events == ["warning"] * (2 if button == "cancel" else 1)
            notice = datalab.gui.h5io.append_provenance_notice()
            assert dialogs.messages[0] == notice
            expected = count if button == "cancel" else 3 * count
            assert len(_objects(win)) == expected
            assert win.h5inputoutput.append_informed is False


@pytest.mark.usefixtures("ask")
def test_answer_yes_still_warns_before_appending_the_next_files(monkeypatch) -> None:
    """Answering Yes (clear) told nothing about appending the next files."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        with app() as win:
            build_chain(win)
            Conf.h5_clear_workspace_ask.set(True)
            dialogs = _Dialogs(monkeypatch, QW.QMessageBox.Yes, "cancel")
            dialogs.files = [path, path]
            win.open_h5_files(import_all=True)
            # The first file replaces the workspace, the second asks first
            assert dialogs.events == ["files", "question", "warning"]


@pytest.mark.usefixtures("ask")
def test_cancelled_append_is_not_recorded_in_history(monkeypatch) -> None:
    """Opening a file whose appending was cancelled leaves no History entry."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        with app(history=True) as win:
            build_chain(win)
            history = win.historypanel
            history.toggle_record_mode(True)
            count = len(history)
            Conf.h5_clear_workspace_ask.set(False)
            _Dialogs(monkeypatch, None, "cancel")
            win.open_h5_files([path], import_all=True, reset_all=False)
            assert len(history) == count
            _Dialogs(monkeypatch, None, "add")
            win.open_h5_files([path], import_all=True, reset_all=False)
            assert len(history) == count + 1
            assert history[len(history)].method_name == "open_h5_files"


@pytest.mark.usefixtures("ask")
def test_answered_warning_can_be_closed(monkeypatch) -> None:
    """An answered warning accepts closing, as the unattended quit requires."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.h5")
        _save_chain(path)
        with app() as win:
            build_chain(win)
            Conf.h5_clear_workspace_ask.set(False)
            _Dialogs(monkeypatch, None, "cancel")
            win.open_h5_files([path], import_all=True, reset_all=False)
            notice = datalab.gui.h5io.append_provenance_notice()
            boxes = [
                widget
                for widget in QW.QApplication.topLevelWidgets()
                if isinstance(widget, QW.QMessageBox) and widget.text() == notice
            ]
            assert all(box.close() for box in boxes)
