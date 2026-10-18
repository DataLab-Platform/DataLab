# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace capsule dialog: the five tabs of DataLab-Web's dialog.

File dialogs are bypassed by passing file names to the tab methods.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=unsubscriptable-object  # Optional values checked by asserts
# pylint: disable=protected-access

from __future__ import annotations

import dataclasses
import json
import os.path as osp
import socket
import threading
import time

import pytest

from datalab.capsulestore import get_registry_settings, set_registry_settings
from datalab.tests import helpers
from datalab.tests.features.hdf5.capsule_unit_test import capsule_app
from datalab.tests.features.hdf5.provenance_roundtrip_unit_test import build_chain
from datalab.widgets.capsule import CapsuleDialog, short_id

NOTE = b"Bench 2.\n"


def _write(path: str, data: bytes) -> str:
    with open(path, "wb") as file:
        file.write(data)
    return path


def _read(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()


def _export(dialog: CapsuleDialog, tmpdir: str, name: str = "chain") -> str:
    tab = dialog.export_tab
    tab.name_edit.setText(name)
    tab.description_edit.setText("Operator A")
    tab.add_attachments([_write(osp.join(tmpdir, "protocol.txt"), NOTE)])
    path = osp.join(tmpdir, f"{name}.dlcapsule")
    assert tab.export(path) is not None, dialog.error
    return path


def test_export_inspect_replay_and_compare() -> None:
    """Export, inspect with supplied documents, open, replay and compare."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with capsule_app(tmpdir) as win:
            build_chain(win)
            dialog = CapsuleDialog(win)
            assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
                "Export",
                "Inspect",
                "Local replay",
                "Revisions",
                "Registry",
            ]
            # Export: no parent revision yet, a 16 MiB limit on attachments
            assert dialog.export_tab.revision_check.isHidden()
            big = osp.join(tmpdir, "big.bin")
            with open(big, "wb") as file:
                file.truncate(16 * 1024**2 + 1)
            dialog.export_tab.add_attachments([big])
            assert "16 MiB" in dialog.export_tab.attachment_error.text()
            parent = _export(dialog, tmpdir)
            # Local replay of the open workspace: no change to the workspace
            replay = dialog.replay_tab
            activity_id = replay.activity_ids()[1]
            report = replay.replay(activity_id)
            assert report["verdict"] == "exact", report
            report_path = osp.join(tmpdir, "report.json")
            assert replay.save(activity_id, report_path)
            # Inspect: integrity, attachments, documents, review writer
            inspect = dialog.inspect_tab
            assert inspect.choose(parent)
            assert inspect.intact and inspect.summary["name"] == "chain"
            saved = osp.join(tmpdir, "saved.txt")
            assert inspect.save_attachment("protocol.txt", saved)
            assert _read(saved) == NOTE
            writer = inspect.review_writer
            writer.reviewer_edit.setText("B")
            writer.statement_edit.setPlainText("Replayed: exact.")
            writer.verdict_combo.setCurrentIndex(
                writer.verdict_combo.findData("endorsed")
            )
            writer.cite_report(report_path)
            review_path = osp.join(tmpdir, "review.json")
            review = writer.save(review_path)
            assert review is not None
            assert review["verdict"] == "endorsed" and len(review["reports"]) == 1
            assert inspect.add_document("report", report_path)
            assert inspect.add_document("review", review_path)
            documents = inspect.inspection["documents"]
            assert [d["status"] for d in documents["reports"]] == ["valid"]
            assert [d["status"] for d in documents["reviews"]] == ["valid"]
            assert inspect.intact
            # Open as workspace: the next export is a revision of this capsule
            assert inspect.open(confirm=False), dialog.error
            assert len(win.signalpanel) == 5
            export = dialog.export_tab
            assert not export.revision_check.isHidden()
            assert short_id(inspect.summary["fingerprint"]) in (
                export.revision_check.text()
            )
            export.attachments.clear()
            export.refresh()
            assert "protocol.txt" in export.parent_note.text()
            child = _export(dialog, tmpdir, "child")
            parent_fp = inspect.summary["fingerprint"]
            assert inspect.choose(child)
            assert inspect.summary["revision_of"] == parent_fp
            # Revisions: compare the parent and the child
            revisions = dialog.revisions_tab
            by_name = {r.name: r for r in revisions.revisions}
            assert by_name["child"].revision_of == by_name["chain"].fingerprint
            revisions.select(by_name["chain"].fingerprint, by_name["child"].fingerprint)
            diff = revisions.compare()
            assert diff["revision_of_old"] and not diff["scientific"]
            assert "parent" in revisions.diff_view.toPlainText()
            dialog.close()


def test_inspect_reports_problems() -> None:
    """A damaged capsule, a stranger report and a referenced workspace."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with capsule_app(tmpdir) as win:
            build_chain(win)
            dialog = CapsuleDialog(win)
            inspect = dialog.inspect_tab
            assert inspect.choose(_write(osp.join(tmpdir, "bad.dlcapsule"), b"PK"))
            assert inspect.summary is None and not inspect.intact
            assert not inspect.open_button.isEnabled()
            export = dialog.export_tab
            export.referenced_check.setChecked(True)
            assert not export.export_button.isEnabled()
            export.ref_edit.setText("https://data.example.org/chain.h5")
            path = osp.join(tmpdir, "ref.dlcapsule")
            workspace = osp.join(tmpdir, "published.h5")
            assert export.export(path, workspace) is not None, dialog.error
            assert inspect.choose(path)
            assert inspect.referenced and inspect.intact
            assert not inspect.workspace_button.isHidden()
            assert not inspect.open_button.isEnabled()
            assert inspect.inspection["integrity"]["workspace"] == "not_available"
            assert inspect.choose_workspace(workspace)
            assert inspect.inspection["integrity"]["workspace"] == "verified"
            assert inspect.open_button.isEnabled()
        with capsule_app(tmpdir) as win:
            dialog = CapsuleDialog(win)
            assert not dialog.replay_tab.empty.isHidden()
            report = osp.join(tmpdir, "stranger.json")
            with open(report, "w", encoding="utf-8") as file:
                json.dump({"activity_id": "unknown"}, file)
            inspect = dialog.inspect_tab
            assert inspect.choose(path)
            assert inspect.add_document("report", report)
            assert inspect.inspection["documents"]["reports"][0]["status"] != "valid"
            assert inspect.intact
            assert inspect.open(confirm=False) is False
            assert "select the workspace file" in dialog.error
            dialog.close()


def test_inspect_refuses_files_beyond_the_policy() -> None:
    """Inspect: a capsule or a referenced workspace too large is not read."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with capsule_app(tmpdir) as win:
            build_chain(win)
            dialog = CapsuleDialog(win)
            export = dialog.export_tab
            export.referenced_check.setChecked(True)
            export.ref_edit.setText("https://data.example.org/chain.h5")
            path = osp.join(tmpdir, "ref.dlcapsule")
            workspace = osp.join(tmpdir, "published.h5")
            assert export.export(path, workspace) is not None, dialog.error
            size, ws_size = osp.getsize(path), osp.getsize(workspace)
            capsuleio = win.capsuleio
            default = capsuleio.policy
            inspect = dialog.inspect_tab
            capsuleio.policy = dataclasses.replace(default, max_archive_bytes=size - 1)
            assert not inspect.choose(path)
            assert "Capsule too large: ref.dlcapsule" in dialog.error
            assert inspect.inspection is None
            capsuleio.policy = dataclasses.replace(
                default, max_archive_bytes=size, max_workspace_bytes=ws_size - 1
            )
            assert inspect.choose(path) and inspect.intact
            assert not inspect.choose_workspace(workspace)
            assert "Referenced workspace too large: published.h5" in dialog.error
            # A new capsule that cannot be read never keeps the former inspection
            capsuleio.policy = default
            assert inspect.choose(path) and inspect.intact
            capsuleio.policy = dataclasses.replace(default, max_archive_bytes=size - 1)
            assert not inspect.choose(path)
            assert inspect.inspection is None and not inspect.intact
            assert inspect.open_button.isHidden()
            assert inspect.publish_button.isHidden()
            dialog.close()


@pytest.fixture(name="registry")
def _registry(tmp_path):
    """A real capsule registry served by uvicorn in a thread."""
    pytest.importorskip("sqlite3")
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    # pylint: disable=import-outside-toplevel
    from datalab_capsule.registry import Registry
    from datalab_capsule.registry_app import create_app

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_app(Registry(tmp_path / "registry"), token="secret")
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _index in range(100):
        if server.started:
            break
        time.sleep(0.05)
    previous = get_registry_settings()
    yield f"http://127.0.0.1:{port}"
    set_registry_settings(*previous)
    server.should_exit = True
    thread.join(5)


def test_registry(registry: str) -> None:
    """Publish a capsule and a review, list, retrieve and download."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with capsule_app(tmpdir) as win:
            build_chain(win)
            dialog = CapsuleDialog(win)
            path = _export(dialog, tmpdir)
            tab = dialog.registry_tab
            tab.address_edit.setText(registry)
            tab.token_edit.setText("wrong")
            assert tab.save()
            inspect = dialog.inspect_tab
            assert inspect.choose(path)
            assert not inspect.publish_button.isHidden()
            assert not inspect.publish()
            assert "401" in dialog.error
            tab.token_edit.setText("secret")
            assert tab.save()
            writer = inspect.review_writer
            writer.reviewer_edit.setText("B")
            writer.statement_edit.setPlainText("Checked.")
            review = osp.join(tmpdir, "review.json")
            writer.save(review)
            assert inspect.add_document("review", review)
            assert inspect.publish(), dialog.error
            assert "1 attachments and 1 reviews" in inspect.published.text()
            assert tab.list_revisions(), dialog.error
            assert "datalab-capsule" in tab.status.text()
            assert tab.table.rowCount() == 1
            fingerprint = tab.listed[0]["fingerprint"]
            assert tab.retrieve(fingerprint), dialog.error
            assert [r["status"] for r in tab.retrieved["reviews"]] == ["valid"]
            assert tab.retrieved["stored"] == {"protocol.txt"}
            downloaded = osp.join(tmpdir, "downloaded.txt")
            assert tab.save_attachment("protocol.txt", downloaded)
            assert _read(downloaded) == NOTE
            sources = {r.fingerprint: r.source for r in dialog.revisions_tab.revisions}
            assert sources[fingerprint] == "exported"
            writer = inspect.review_writer
            writer.reviewer_edit.setText("C")
            writer.statement_edit.setPlainText("Second review.")
            writer.publish()
            assert "published" in writer.published.text(), dialog.error
            dialog.close()
