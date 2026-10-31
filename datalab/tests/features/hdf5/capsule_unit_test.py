# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace capsules exported and opened by DataLab Desktop.

The chain workspace (see :mod:`provenance_roundtrip_unit_test`) is exported as a
capsule, opened again in a fresh window and replayed; a revision exported from
it names its parent. Capsules created from the DataLab-Web reference file open
and replay as well.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=unsubscriptable-object  # Optional values checked by asserts

from __future__ import annotations

import contextlib
import os
import os.path as osp
import tracemalloc
from collections.abc import Callable, Generator

import pytest
from datalab_capsule import exchange
from datalab_capsule.archive import (
    MAX_MANIFEST_BYTES,
    CapsuleError,
    SizePolicy,
    create_from_hdf5,
    read_capsule,
)
from datalab_capsule.manifest import inspect_manifest
from sigima.objects import create_signal

from datalab.capsulestore import RevisionStore
from datalab.control.proxy import LocalProxy
from datalab.gui import capsuleio
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.tests import helpers
from datalab.tests.features.common.provenance_unit_test import app
from datalab.tests.features.hdf5.provenance_roundtrip_unit_test import (
    WEB_FIXTURE,
    X,
    Y,
    build_chain,
    check_chain,
)

REF = "https://data.example.org/chain.h5"
NOTE = b"Bench 2.\n"


@contextlib.contextmanager
def capsule_app(tmpdir: str, history: bool = False) -> Generator[DLMainWindow]:
    """Return a main window whose revisions are kept in *tmpdir*."""
    with app(history=history) as win:
        win.capsuleio.revisions = RevisionStore(osp.join(tmpdir, "revisions"))
        yield win


def _note(tmpdir: str) -> str:
    path = osp.join(tmpdir, "protocol notes.txt")
    with open(path, "wb") as file:
        file.write(NOTE)
    return path


def _read(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()


def _root(path: str) -> dict:
    manifest = read_capsule(path).manifest
    return next(e for e in manifest["@graph"] if e["@id"] == "./")


def test_export_open_and_revision() -> None:
    """Export, open in a fresh window, replay, then export a revision."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        parent = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            fingerprint = win.export_capsule(
                parent,
                name="chain",
                description="Operator A",
                attachments=[_note(tmpdir)],
            )
            assert win.provenance.capsule_origin is None
            [revision] = win.capsuleio.revisions.revisions()
            assert (revision.fingerprint, revision.source) == (fingerprint, "exported")
        data = _read(parent)
        inspection = exchange.inspect_capsule(data)
        assert inspection["integrity"]["valid"], inspection["integrity"]
        summary = inspection["summary"]
        assert (summary["name"], summary["description"]) == ("chain", "Operator A")
        assert [a["name"] for a in summary["attachments"]] == ["protocol_notes.txt"]
        assert summary["attachments"][0]["media_type"] == "text/plain"
        assert exchange.capsule_attachment(data, "protocol_notes.txt") == NOTE
        child = osp.join(tmpdir, "child.dlcapsule")
        with capsule_app(tmpdir) as win:
            win.open_capsule(parent)
            check_chain(win, "desktop")
            assert win.provenance.capsule_origin == {
                "fingerprint": fingerprint,
                "name": "chain",
                "attachments": ["protocol_notes.txt"],
            }
            child_fp = win.export_capsule(child, name="child")
            assert _root(child)["dlc:revisionOf"] == fingerprint
            alone = osp.join(tmpdir, "alone.dlcapsule")
            win.export_capsule(alone, as_revision=False)
            assert "dlc:revisionOf" not in _root(alone)
            revisions = {r.fingerprint: r for r in win.capsuleio.revisions.revisions()}
            assert revisions[child_fp].revision_of == fingerprint
            diff = exchange.diff_checked_manifests(
                revisions[fingerprint].manifest, revisions[child_fp].manifest
            )
            assert diff["revision_of_old"] and not diff["scientific"]
            win.reset_all()
            assert win.provenance.capsule_origin is None


def test_revision_after_saving_the_workspace() -> None:
    """A workspace saved and opened again still exports a revision of its parent."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        parent = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            fingerprint = win.export_capsule(parent, name="chain")
        saved = osp.join(tmpdir, "work.h5")
        with capsule_app(tmpdir) as win:
            win.open_capsule(parent)
            origin = win.provenance.capsule_origin
            win.save_to_h5_file(saved)
            win.reset_all()
            assert win.provenance.capsule_origin is None
            win.load_h5_workspace([saved], reset_all=True)
            assert win.provenance.capsule_origin == origin
            # Appending another workspace keeps the parent of the current one.
            other = osp.join(tmpdir, "other.h5")
            win.provenance.capsule_origin = {**origin, "name": "other"}
            win.save_to_h5_file(other)
            win.provenance.capsule_origin = origin
            win.load_h5_workspace([other], reset_all=False)
            assert win.provenance.capsule_origin == origin
        with capsule_app(tmpdir) as win:
            win.open_h5_files([saved], import_all=True, reset_all=True)
            assert win.provenance.capsule_origin == origin
            check_chain(win, "desktop")
            child = osp.join(tmpdir, "child.dlcapsule")
            win.export_capsule(child, name="child")
            assert _root(child)["dlc:revisionOf"] == fingerprint


def test_referenced_workspace() -> None:
    """A referenced workspace is saved apart and must be supplied to open."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        capsule = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            win.export_capsule(capsule, workspace_ref=REF)
        workspace = osp.join(tmpdir, "chain.h5")
        assert osp.isfile(workspace)
        assert inspect_manifest(read_capsule(capsule).manifest)["workspace"][
            "referenced"
        ]
        with capsule_app(tmpdir) as win:
            with pytest.raises(CapsuleError, match="select the workspace file"):
                win.open_capsule(capsule)
            assert len(win.signalpanel) == 0
            win.open_capsule(capsule, workspace)
            check_chain(win, "desktop")


def test_export_never_overwrites_its_own_files() -> None:
    """The capsule, its referenced workspace and its attachments are different
    files: an export that would overwrite one of them writes nothing."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        capsule = osp.join(tmpdir, "chain.dlcapsule")
        note = _note(tmpdir)
        with open(capsule, "wb") as file:
            file.write(NOTE)
        with capsule_app(tmpdir) as win:
            build_chain(win)
            for kwargs in (
                {"workspace_ref": REF, "workspace_filename": capsule},
                {"attachments": [capsule]},
                {
                    "workspace_ref": REF,
                    "workspace_filename": note,
                    "attachments": [note],
                },
            ):
                with pytest.raises(CapsuleError, match="different files"):
                    win.export_capsule(capsule, **kwargs)
            assert _read(note) == NOTE
            assert _read(capsule) == NOTE
            assert not win.capsuleio.revisions.revisions()


def test_invalid_capsule_leaves_workspace() -> None:
    """A damaged capsule is refused before the workspace changes."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "damaged.dlcapsule")
        with open(path, "wb") as file:
            file.write(b"PK\x03\x04 not a capsule")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            activities = len(win.provenance.ledger.activities)
            with pytest.raises(Exception):  # noqa: B017
                win.open_capsule(path)
            assert len(win.signalpanel) == 5
            assert len(win.provenance.ledger.activities) == activities


def test_capsule_of_another_hdf5_file_is_refused() -> None:
    """A capsule workspace that is not a DataLab workspace is never imported."""
    import glob  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    import h5py  # pylint: disable=import-outside-toplevel

    with helpers.WorkdirRestoringTempDir() as tmpdir:
        workspace = osp.join(tmpdir, "chain.h5")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            win.save_to_h5_file(workspace)
        # Same objects and provenance, but not marked as a DataLab workspace
        with h5py.File(workspace, "a") as h5file:
            del h5file["DataLab_Version"]
        path = osp.join(tmpdir, "foreign.dlcapsule")
        with open(path, "wb") as file:
            file.write(create_from_hdf5(_read(workspace), name="foreign"))
        pattern = osp.join(tempfile.gettempdir(), "datalab-capsule-*.h5")
        leftovers = set(glob.glob(pattern))
        with capsule_app(tmpdir) as win:
            with pytest.raises(CapsuleError, match="not a DataLab workspace"):
                win.open_capsule(path)
            assert win.provenance.capsule_origin is None
            assert not win.capsuleio.revisions.revisions()
        assert set(glob.glob(pattern)) == leftovers


def test_web_capsule_opens_and_replays() -> None:
    """A capsule of a DataLab-Web workspace opens and replays on Desktop."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "web.dlcapsule")
        with open(path, "wb") as file:
            file.write(create_from_hdf5(WEB_FIXTURE, name="web chain"))
        with capsule_app(tmpdir) as win:
            win.open_capsule(path)
            check_chain(win, "web", historical=True)
            origin = win.provenance.capsule_origin
            assert origin is not None and origin["name"] == "web chain"


def test_local_proxy_and_history() -> None:
    """The proxy exports and opens capsules; opening is recorded and replayed."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir, history=True) as win:
            proxy = LocalProxy(win)
            build_chain(win)
            fingerprint = proxy.export_capsule(path, name="chain")
            assert fingerprint.startswith("sha256:")
            history = win.historypanel
            history.toggle_record_mode(True)
            proxy.open_capsule(path)
            action = history[len(history)]
            assert action.method_name == "open_capsule"
            assert len(win.signalpanel) == 5
            win.reset_all()
            assert len(win.signalpanel) == 0
            hireplay.replay_actions(history, [action], prompt=False)
            assert len(win.signalpanel) == 5
            origin = win.provenance.capsule_origin
            assert origin is not None and origin["fingerprint"] == fingerprint


def test_history_replay_asks_before_overwriting_a_capsule() -> None:
    """Replaying a capsule export overwrites a file: it is confirmed first."""
    # pylint: disable-next=import-outside-toplevel
    from datalab.env import execenv

    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir, history=True) as win:
            build_chain(win)
            history = win.historypanel
            history.toggle_record_mode(True)
            win.export_capsule(path, name="chain")
            action = history[len(history)]
            assert action.method_name == "export_capsule"
            with open(path, "wb") as file:
                file.write(b"kept")
            # Unattended: not confirmed, so not replayed
            hireplay.replay_actions(history, [action], prompt=False)
            assert _read(path) == b"kept"
            with execenv.context(accept_dialogs=True):
                hireplay.replay_actions(history, [action], prompt=False)
            assert read_capsule(path).manifest is not None


def test_remote_server() -> None:
    """The XML-RPC server exports and opens capsules, and reports refusals."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "chain.dlcapsule")
        with capsule_app(tmpdir) as win:
            server = win.remote_server
            build_chain(win)
            fingerprint = server.export_capsule(path, "chain", None, True, REF)
            assert fingerprint.startswith("sha256:")
            assert osp.isfile(osp.join(tmpdir, "chain.h5"))
            with pytest.raises(CapsuleError, match="select the workspace file"):
                server.open_capsule(path)
            server.open_capsule(path, osp.join(tmpdir, "chain.h5"))
            check_chain(win, "desktop")


def test_read_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file beyond the limit is refused before it is opened.

    Within the limit, the buffer read is the size of the file, not of the limit.
    """
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        path = osp.join(tmpdir, "data.bin")
        with open(path, "wb") as file:
            file.write(b"x" * 100)
        assert capsuleio.read_bounded(path, 100, "Capsule") == b"x" * 100
        tracemalloc.start()
        try:
            assert capsuleio.read_bounded(path, 4 * 1024**3, "Capsule") == b"x" * 100
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 1024**2, peak

        def refuse(*_args, **_kwargs):
            raise AssertionError("the file was opened")

        monkeypatch.setattr(capsuleio, "open", refuse, raising=False)
        with pytest.raises(CapsuleError, match=r"Capsule too large: data\.bin"):
            capsuleio.read_bounded(path, 99, "Capsule")


def _policy(archive: int, workspace: int) -> SizePolicy:
    return SizePolicy(
        max_archive_bytes=archive,
        max_manifest_bytes=MAX_MANIFEST_BYTES,
        max_workspace_bytes=workspace,
    )


def test_size_policy_refuses_before_reading() -> None:
    """Capsule and referenced workspace: refused beyond the policy, workspace kept.

    Exact sizes are accepted; one byte less refuses the file before it is read
    and leaves the current workspace unchanged.
    """
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        capsule = osp.join(tmpdir, "chain.dlcapsule")
        workspace = osp.join(tmpdir, "chain.h5")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            win.export_capsule(capsule, workspace_ref=REF)
            size, ws_size = osp.getsize(capsule), osp.getsize(workspace)
            activities = len(win.provenance.ledger.activities)
            for policy, message in (
                (_policy(size - 1, ws_size), "Capsule too large"),
                (_policy(size, ws_size - 1), "Referenced workspace too large"),
            ):
                win.capsuleio.policy = policy
                with pytest.raises(CapsuleError, match=message):
                    win.open_capsule(capsule, workspace)
                assert len(win.signalpanel) == 5
                assert len(win.provenance.ledger.activities) == activities
                assert win.provenance.capsule_origin is None
            win.capsuleio.policy = _policy(size, ws_size)
            win.open_capsule(capsule, workspace)
            check_chain(win, "desktop")


def test_size_policy_refuses_export() -> None:
    """An export beyond the policy writes no file."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        reference = osp.join(tmpdir, "reference.dlcapsule")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            win.export_capsule(reference)
            size = osp.getsize(reference)
            workspace = len(win.capsuleio.workspace_bytes())
            path = osp.join(tmpdir, "refused.dlcapsule")
            for policy, message in (
                (_policy(10 * size, workspace // 2), "Workspace too large"),
                (_policy(size // 2, 10 * workspace), "too large"),
            ):
                win.capsuleio.policy = policy
                with pytest.raises(CapsuleError, match=message):
                    win.export_capsule(path, workspace_ref=None)
                assert not osp.exists(path)
            assert len(win.capsuleio.revisions.revisions()) == 1


def _fail_replace(
    monkeypatch: pytest.MonkeyPatch, fails: Callable[[str, str], bool]
) -> None:
    """Make ``os.replace`` fail, as for a file in use, when *fails* returns True.

    Args:
        monkeypatch: Pytest fixture undoing the change after the test.
        fails: Called with the base names of the source and destination files.
    """
    replace = os.replace

    def failing_replace(src: str, dst: str) -> None:
        if fails(osp.basename(src), osp.basename(dst)):
            raise PermissionError(13, "File in use", dst)
        replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)


def _leftovers(dirname: str) -> list[str]:
    """Return the temporary and backup files left in *dirname*."""
    return [name for name in os.listdir(dirname) if name.endswith((".tmp", ".bak"))]


def test_write_files_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Files are flushed to disk, then replace the previous ones: a file that cannot
    be written replaces none of them, and no temporary or backup file is left."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        first, second = osp.join(tmpdir, "a.h5"), osp.join(tmpdir, "b.dlcapsule")
        capsuleio.write_files_atomically([(first, b"old a"), (second, b"old b")])
        assert (_read(first), _read(second)) == (b"old a", b"old b")
        with pytest.raises(TypeError):
            # The second file cannot be written: the first one is not replaced
            capsuleio.write_files_atomically([(first, b"new a"), (second, None)])
        assert (_read(first), _read(second)) == (b"old a", b"old b")
        assert sorted(os.listdir(tmpdir)) == ["a.h5", "b.dlcapsule"]
        fsync, synced = os.fsync, []

        def recording_fsync(fd: int) -> None:
            synced.append(fd)
            fsync(fd)

        monkeypatch.setattr(os, "fsync", recording_fsync)
        capsuleio.write_files_atomically([(first, b"new a"), (second, b"new b")])
        assert (_read(first), _read(second)) == (b"new a", b"new b")
        assert len(synced) == 2
        assert sorted(os.listdir(tmpdir)) == ["a.h5", "b.dlcapsule"]


@pytest.mark.parametrize(
    "previous, fails",
    [
        pytest.param(True, lambda src, _dst: src == "b.dlcapsule", id="capsule-in-use"),
        pytest.param(
            True,
            lambda src, dst: src.endswith(".tmp") and dst == "b.dlcapsule",
            id="capsule-not-replaced",
        ),
        pytest.param(
            False,
            lambda src, dst: src.endswith(".tmp") and dst == "b.dlcapsule",
            id="new-files",
        ),
    ],
)
def test_failed_replacement_keeps_the_previous_files(
    monkeypatch: pytest.MonkeyPatch, previous: bool, fails: Callable[[str, str], bool]
) -> None:
    """Regression: when the capsule could not be replaced, its workspace had already
    been replaced. The previous files are now put back (the new ones removed)."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        first, second = osp.join(tmpdir, "a.h5"), osp.join(tmpdir, "b.dlcapsule")
        if previous:
            capsuleio.write_files_atomically([(first, b"old a"), (second, b"old b")])
        _fail_replace(monkeypatch, fails)
        with pytest.raises(PermissionError, match="File in use"):
            capsuleio.write_files_atomically([(first, b"new a"), (second, b"new b")])
        if previous:
            assert (_read(first), _read(second)) == (b"old a", b"old b")
            assert sorted(os.listdir(tmpdir)) == ["a.h5", "b.dlcapsule"]
        else:
            assert not os.listdir(tmpdir)


def test_failed_restoration_keeps_the_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A previous file that cannot be put back is kept, and named in the error."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        first, second = osp.join(tmpdir, "a.h5"), osp.join(tmpdir, "b.dlcapsule")
        capsuleio.write_files_atomically([(first, b"old a"), (second, b"old b")])
        _fail_replace(
            monkeypatch,
            lambda src, dst: (
                (src.endswith(".tmp") and dst == "b.dlcapsule")
                or (src.endswith(".bak") and dst == "a.h5")
            ),
        )
        with pytest.raises(OSError, match="File in use") as excinfo:
            capsuleio.write_files_atomically([(first, b"new a"), (second, b"new b")])
        assert isinstance(excinfo.value.__cause__, PermissionError)
        backups = _leftovers(tmpdir)
        assert len(backups) == 1 and backups[0].startswith(".a.h5.")
        assert backups[0].endswith(".bak") and backups[0] in str(excinfo.value)
        assert _read(osp.join(tmpdir, backups[0])) == b"old a"
        assert (_read(first), _read(second)) == (b"new a", b"old b")


@pytest.mark.parametrize("failure", ["full-disk", "capsule-not-replaced"])
def test_failed_export_keeps_the_previous_capsule(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Overwriting a capsule and its workspace: on failure, both are kept (or put
    back), even when the workspace was replaced before the capsule failed."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        capsule = osp.join(tmpdir, "chain.dlcapsule")
        workspace = osp.join(tmpdir, "chain.h5")
        with capsule_app(tmpdir) as win:
            build_chain(win)
            win.export_capsule(capsule, workspace_ref=REF)
            previous = _read(capsule), _read(workspace)
            # A different workspace, so that replacing the previous one shows
            win.signalpanel.add_object(create_signal("S5", X.copy(), Y.copy()))

            def full_disk(*_args) -> None:
                raise OSError("No space left on device")

            if failure == "full-disk":
                monkeypatch.setattr(capsuleio, "set_final_permissions", full_disk)
                message = "No space left"
            else:
                _fail_replace(
                    monkeypatch,
                    lambda src, dst: src.endswith(".tmp") and dst == "chain.dlcapsule",
                )
                message = "File in use"
            with pytest.raises(OSError, match=message):
                win.export_capsule(capsule, name="other", workspace_ref=REF)
            assert (_read(capsule), _read(workspace)) == previous
            assert not _leftovers(tmpdir)
            assert len(win.capsuleio.revisions.revisions()) == 1
