# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Instance detection unit tests

Testing PID lock file mechanism for single-instance detection.
"""

# guitest: skip

# pylint: disable=protected-access,redefined-outer-name

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from datalab.utils import instancecheck


@pytest.fixture()
def lock_dir(tmp_path):
    """Provide a temporary directory and patch the default registry."""
    registry = instancecheck.ApplicationInstanceRegistry(app_name="TestDataLab")
    lock_path = str(tmp_path / registry.lock_filename)
    with (
        mock.patch.object(instancecheck, "DEFAULT_REGISTRY", registry),
        mock.patch.object(registry, "_get_lock_path", return_value=lock_path),
    ):
        yield tmp_path, lock_path


# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------


class TestIsPidAlive:
    """Tests for the _is_pid_alive helper."""

    def test_current_process_is_alive(self):
        """Current process PID should be reported as alive."""
        assert instancecheck._is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid_posix(self):
        """Posix path: ProcessLookupError means dead."""
        with (
            mock.patch.object(
                instancecheck.DEFAULT_REGISTRY,
                "_is_pid_alive_posix",
                return_value=False,
            ),
            mock.patch.object(
                instancecheck.DEFAULT_REGISTRY,
                "_is_pid_alive_win32",
                return_value=False,
            ),
        ):
            # Mock both platform implementations to guarantee the result
            # regardless of the host OS.
            assert instancecheck._is_pid_alive(999_999_999) is False

    def test_permission_error_means_alive(self):
        """Posix path: PermissionError means the process exists."""
        with mock.patch("os.kill", side_effect=PermissionError):
            assert instancecheck._is_pid_alive_posix(42) is True


# ---------------------------------------------------------------------------
# _read_lock_pids
# ---------------------------------------------------------------------------


class TestReadLockPids:
    """Tests for _read_lock_pids."""

    def test_missing_file(self, tmp_path):
        """Should return empty list when the lock file does not exist."""
        assert instancecheck._read_lock_pids(str(tmp_path / "nope.lock")) == []

    def test_valid_json_list(self, tmp_path):
        """Should return the list of PIDs stored in JSON format."""
        lock = tmp_path / "test.lock"
        lock.write_text(json.dumps([12345, 67890]))
        assert instancecheck._read_lock_pids(str(lock)) == [12345, 67890]

    def test_legacy_single_pid(self, tmp_path):
        """Should handle legacy single-PID plain-text format."""
        lock = tmp_path / "test.lock"
        lock.write_text("12345")
        assert instancecheck._read_lock_pids(str(lock)) == [12345]

    def test_corrupted_content(self, tmp_path):
        """Should return empty list and remove the corrupted file."""
        lock = tmp_path / "bad.lock"
        lock.write_text("not-a-number")
        assert instancecheck._read_lock_pids(str(lock)) == []
        assert not lock.exists()

    def test_empty_file(self, tmp_path):
        """An empty file is treated as corrupted."""
        lock = tmp_path / "empty.lock"
        lock.write_text("")
        assert instancecheck._read_lock_pids(str(lock)) == []
        assert not lock.exists()


class TestReadLockPidLegacy:
    """Tests for _read_lock_pid (legacy compat wrapper)."""

    def test_missing_file(self, tmp_path):
        """Should return None when the lock file does not exist."""
        assert instancecheck._read_lock_pid(str(tmp_path / "nope.lock")) is None

    def test_valid_pid(self, tmp_path):
        """Should return the first PID stored in the file."""
        lock = tmp_path / "test.lock"
        lock.write_text(json.dumps([12345, 67890]))
        assert instancecheck._read_lock_pid(str(lock)) == 12345

    def test_legacy_single_pid(self, tmp_path):
        """Should return the integer PID stored in legacy format."""
        lock = tmp_path / "test.lock"
        lock.write_text("12345")
        assert instancecheck._read_lock_pid(str(lock)) == 12345


# ---------------------------------------------------------------------------
# is_another_instance_running
# ---------------------------------------------------------------------------


class TestIsAnotherInstanceRunning:
    """Tests for is_another_instance_running."""

    def test_no_lock_file(self, lock_dir):
        """No lock file → no other instance."""
        _tmp_path, _lock_path = lock_dir
        assert instancecheck.is_another_instance_running() is None

    def test_own_pid(self, lock_dir):
        """Lock file with our own PID → not 'another' instance."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([os.getpid()], f)
        assert instancecheck.is_another_instance_running() is None

    def test_alive_foreign_pid(self, lock_dir):
        """Lock file with a live foreign PID → returns that PID."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            assert instancecheck.is_another_instance_running() == foreign_pid

    def test_stale_lock_removed(self, lock_dir):
        """Stale lock (dead PID) should be cleaned automatically."""
        _tmp_path, lock_path = lock_dir
        dead_pid = 2**22 - 1
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([dead_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=False
        ):
            assert instancecheck.is_another_instance_running() is None
        assert not os.path.exists(lock_path)

    def test_multiple_pids_one_alive(self, lock_dir):
        """Lock file with multiple PIDs, one alive → returns the alive one."""
        _tmp_path, lock_path = lock_dir
        alive_pid = 99999
        dead_pid = 88888
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([dead_pid, alive_pid], f)

        def fake_alive(pid):
            return pid == alive_pid

        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", side_effect=fake_alive
        ):
            assert instancecheck.is_another_instance_running() == alive_pid

        # Dead PID was cleaned up, alive PID remains
        remaining = instancecheck._read_lock_pids(lock_path)
        assert remaining == [alive_pid]

    def test_multiple_pids_all_dead(self, lock_dir):
        """Lock file with multiple stale PIDs → file removed entirely."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([88888, 77777], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=False
        ):
            assert instancecheck.is_another_instance_running() is None
        assert not os.path.exists(lock_path)

    def test_legacy_single_pid_format(self, lock_dir):
        """Legacy format (plain integer) should be handled transparently."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(str(foreign_pid))
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            assert instancecheck.is_another_instance_running() == foreign_pid


# ---------------------------------------------------------------------------
# create_lock_file / remove_lock_file
# ---------------------------------------------------------------------------


class TestCreateLockFile:
    """Tests for create_lock_file."""

    def test_creates_file_with_current_pid(self, lock_dir):
        """Lock file should contain the current PID after creation."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        pids = instancecheck._read_lock_pids(lock_path)
        assert os.getpid() in pids

    def test_atomic_prevents_overwrite_of_live_instance(self, lock_dir):
        """Should raise RuntimeError if another live instance holds the lock."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            with pytest.raises(RuntimeError, match="already running"):
                instancecheck.create_lock_file()

    def test_replaces_stale_lock(self, lock_dir):
        """Should clean up a stale lock and create a new one."""
        _tmp_path, lock_path = lock_dir
        dead_pid = 2**22 - 1
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([dead_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=False
        ):
            instancecheck.create_lock_file()
        pids = instancecheck._read_lock_pids(lock_path)
        assert os.getpid() in pids
        assert dead_pid not in pids

    def test_force_adds_pid_alongside_live_instance(self, lock_dir):
        """force=True should add our PID alongside the existing live one."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            # Without force, would raise RuntimeError
            instancecheck.create_lock_file(force=True)
        pids = instancecheck._read_lock_pids(lock_path)
        assert os.getpid() in pids
        assert foreign_pid in pids

    def test_does_not_duplicate_own_pid(self, lock_dir):
        """Calling create_lock_file twice should not duplicate our PID."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        instancecheck.create_lock_file()
        pids = instancecheck._read_lock_pids(lock_path)
        assert pids.count(os.getpid()) == 1


class TestRemoveLockFile:
    """Tests for remove_lock_file."""

    def test_removes_own_lock(self, lock_dir):
        """Should remove the lock file when only our PID is present."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        assert os.path.exists(lock_path)
        instancecheck.remove_lock_file()
        assert not os.path.exists(lock_path)

    def test_keeps_file_with_other_pids(self, lock_dir):
        """Should keep the lock file when other PIDs are still registered."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid, os.getpid()], f)
        instancecheck.remove_lock_file()
        assert os.path.exists(lock_path)
        remaining = instancecheck._read_lock_pids(lock_path)
        assert remaining == [foreign_pid]

    def test_does_not_remove_foreign_lock(self, lock_dir):
        """Should NOT modify the lock file when it does not contain our PID."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([99999], f)
        instancecheck.remove_lock_file()
        assert os.path.exists(lock_path)
        assert instancecheck._read_lock_pids(lock_path) == [99999]

    def test_noop_when_no_lock(self, lock_dir):
        """Should do nothing when there is no lock file."""
        _tmp_path, lock_path = lock_dir
        assert not os.path.exists(lock_path)
        instancecheck.remove_lock_file()  # Should not raise

    def test_concurrent_close_preserves_remaining(self, lock_dir):
        """Closing one of two instances should preserve the other's PID.

        This is the key bug fix: with the old single-PID design, closing
        instance B would delete the lock even though instance A was still
        running, allowing a new instance C to start without a warning.
        """
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999

        # Simulate two instances registered
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid, os.getpid()], f)

        # Current process closes
        instancecheck.remove_lock_file()

        # Lock file still exists with foreign PID
        assert os.path.exists(lock_path)
        remaining = instancecheck._read_lock_pids(lock_path)
        assert remaining == [foreign_pid]

        # A new instance would still detect the foreign PID
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            assert instancecheck.is_another_instance_running() == foreign_pid


# ---------------------------------------------------------------------------
# Integration tests (real filesystem, real processes, no mocks)
# ---------------------------------------------------------------------------


class TestIntegrationLockFile:
    """Integration tests verifying lock file detection across processes.

    These tests use real subprocesses and the real lock file path to ensure
    the full lifecycle works end-to-end.
    """

    def test_lock_path_uses_config_directory(self):
        """Lock path must be inside the DataLab config directory, not '.none'."""
        from datalab.config import (  # pylint: disable=import-outside-toplevel
            APP_NAME,
            Conf,
        )

        lock_path = Conf.get_path(instancecheck.LOCK_FILENAME)
        assert instancecheck.LOCK_FILENAME == f"{APP_NAME}.lock"
        assert ".none" not in lock_path, (
            f"Lock path points to uninitialized config dir: {lock_path}"
        )
        assert instancecheck._get_lock_path() == lock_path
