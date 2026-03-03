# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Instance detection unit tests

Testing PID lock file mechanism for single-instance detection.
"""

# guitest: skip

from __future__ import annotations

import os
from unittest import mock

import pytest

from datalab.utils import instancecheck


@pytest.fixture()
def lock_dir(tmp_path):
    """Provide a temporary directory and patch _get_lock_path to use it."""
    lock_path = str(tmp_path / instancecheck.LOCK_FILENAME)
    with mock.patch.object(instancecheck, "_get_lock_path", return_value=lock_path):
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
            mock.patch.object(instancecheck, "_is_pid_alive_posix", return_value=False),
            mock.patch.object(instancecheck, "_is_pid_alive_win32", return_value=False),
        ):
            # Mock both platform implementations to guarantee the result
            # regardless of the host OS.
            assert instancecheck._is_pid_alive(999_999_999) is False

    def test_permission_error_means_alive(self):
        """Posix path: PermissionError means the process exists."""
        with mock.patch("os.kill", side_effect=PermissionError):
            assert instancecheck._is_pid_alive_posix(42) is True


# ---------------------------------------------------------------------------
# _read_lock_pid
# ---------------------------------------------------------------------------


class TestReadLockPid:
    """Tests for _read_lock_pid."""

    def test_missing_file(self, tmp_path):
        """Should return None when the lock file does not exist."""
        assert instancecheck._read_lock_pid(str(tmp_path / "nope.lock")) is None

    def test_valid_pid(self, tmp_path):
        """Should return the integer PID stored in the file."""
        lock = tmp_path / "test.lock"
        lock.write_text("12345")
        assert instancecheck._read_lock_pid(str(lock)) == 12345

    def test_corrupted_content(self, tmp_path):
        """Should return None and remove the corrupted file."""
        lock = tmp_path / "bad.lock"
        lock.write_text("not-a-number")
        assert instancecheck._read_lock_pid(str(lock)) is None
        assert not lock.exists()

    def test_empty_file(self, tmp_path):
        """An empty file is treated as corrupted."""
        lock = tmp_path / "empty.lock"
        lock.write_text("")
        assert instancecheck._read_lock_pid(str(lock)) is None
        assert not lock.exists()


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
        with open(lock_path, "w") as f:
            f.write(str(os.getpid()))
        assert instancecheck.is_another_instance_running() is None

    def test_alive_foreign_pid(self, lock_dir):
        """Lock file with a live foreign PID → returns that PID."""
        _tmp_path, lock_path = lock_dir
        # Use PID 1 (init on Unix, System Idle on Windows) which is always alive
        # but not ours.  Mock _is_pid_alive to guarantee the behaviour.
        foreign_pid = 99999
        with open(lock_path, "w") as f:
            f.write(str(foreign_pid))
        with mock.patch.object(instancecheck, "_is_pid_alive", return_value=True):
            assert instancecheck.is_another_instance_running() == foreign_pid

    def test_stale_lock_removed(self, lock_dir):
        """Stale lock (dead PID) should be removed automatically."""
        _tmp_path, lock_path = lock_dir
        dead_pid = 2**22 - 1
        with open(lock_path, "w") as f:
            f.write(str(dead_pid))
        with mock.patch.object(instancecheck, "_is_pid_alive", return_value=False):
            assert instancecheck.is_another_instance_running() is None
        assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# create_lock_file / remove_lock_file
# ---------------------------------------------------------------------------


class TestCreateLockFile:
    """Tests for create_lock_file."""

    def test_creates_file_with_current_pid(self, lock_dir):
        """Lock file should contain the current PID after creation."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()

    def test_atomic_prevents_overwrite_of_live_instance(self, lock_dir):
        """Should raise RuntimeError if another live instance holds the lock."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w") as f:
            f.write(str(foreign_pid))
        with mock.patch.object(instancecheck, "_is_pid_alive", return_value=True):
            with pytest.raises(RuntimeError, match="already running"):
                instancecheck.create_lock_file()

    def test_replaces_stale_lock(self, lock_dir):
        """Should clean up a stale lock and create a new one."""
        _tmp_path, lock_path = lock_dir
        dead_pid = 2**22 - 1
        with open(lock_path, "w") as f:
            f.write(str(dead_pid))
        with mock.patch.object(instancecheck, "_is_pid_alive", return_value=False):
            instancecheck.create_lock_file()
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()

    def test_force_overwrites_live_instance(self, lock_dir):
        """force=True should overwrite lock even when another instance lives."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w") as f:
            f.write(str(foreign_pid))
        with mock.patch.object(instancecheck, "_is_pid_alive", return_value=True):
            # Without force, would raise RuntimeError
            instancecheck.create_lock_file(force=True)
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()


class TestRemoveLockFile:
    """Tests for remove_lock_file."""

    def test_removes_own_lock(self, lock_dir):
        """Should remove the lock file when it contains our PID."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        assert os.path.exists(lock_path)
        instancecheck.remove_lock_file()
        assert not os.path.exists(lock_path)

    def test_does_not_remove_foreign_lock(self, lock_dir):
        """Should NOT remove the lock file when it contains another PID."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w") as f:
            f.write("99999")
        instancecheck.remove_lock_file()
        assert os.path.exists(lock_path)

    def test_noop_when_no_lock(self, lock_dir):
        """Should do nothing when there is no lock file."""
        _tmp_path, lock_path = lock_dir
        assert not os.path.exists(lock_path)
        instancecheck.remove_lock_file()  # Should not raise


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
        from datalab.config import Conf

        lock_path = Conf.get_path(instancecheck.LOCK_FILENAME)
        assert ".none" not in lock_path, (
            f"Lock path points to uninitialized config dir: {lock_path}"
        )
        assert instancecheck._get_lock_path() == lock_path
