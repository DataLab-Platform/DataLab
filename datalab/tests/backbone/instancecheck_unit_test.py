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

    def test_nonexistent_pid(self):
        """A nonexistent PID should be reported as dead."""
        with mock.patch("psutil.pid_exists", return_value=False):
            assert instancecheck._is_pid_alive(999_999_999) is False

    def test_invalid_pid_is_dead(self):
        """Non-positive PIDs should be rejected before psutil is queried."""
        with mock.patch("psutil.pid_exists") as mock_pid_exists:
            assert instancecheck._is_pid_alive(0) is False
        mock_pid_exists.assert_not_called()


# ---------------------------------------------------------------------------
# _read_lock_pids
# ---------------------------------------------------------------------------


class TestReadLockPids:
    """Tests for _read_lock_pids."""

    @pytest.mark.parametrize(
        ("content", "expected", "removed"),
        [
            (None, [], False),
            (json.dumps([12345, 67890]), [12345, 67890], False),
            ("12345", [12345], False),
            ("not-a-number", [], True),
            ("", [], True),
        ],
    )
    def test_supported_and_invalid_formats(
        self,
        tmp_path,
        content,
        expected,
        removed,
    ):
        """Should handle missing, supported, and invalid lock file formats."""
        lock = tmp_path / "test.lock"
        if content is None:
            target = str(lock)
        else:
            lock.write_text(content)
            target = str(lock)

        assert instancecheck._read_lock_pids(target) == expected
        assert lock.exists() is (not removed and content is not None)


class TestReadLockPidLegacy:
    """Tests for _read_lock_pid (legacy compat wrapper)."""

    def test_returns_first_pid_for_supported_formats(self, tmp_path):
        """Should return the first PID in JSON and legacy formats."""
        lock = tmp_path / "json.lock"
        lock.write_text(json.dumps([12345, 67890]))
        assert instancecheck._read_lock_pid(str(lock)) == 12345

        lock = tmp_path / "entry.lock"
        lock.write_text(json.dumps([{"pid": 23456, "create_time": 1.0}]))
        assert instancecheck._read_lock_pid(str(lock)) == 23456

        lock = tmp_path / "legacy.lock"
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

    def test_mixed_pids_keep_only_live_foreign_entries(self, lock_dir):
        """Own and stale PIDs are ignored while a live foreign PID is kept."""
        _tmp_path, lock_path = lock_dir
        current_pid = os.getpid()
        alive_pid = 99999
        dead_pid = 88888
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([current_pid, dead_pid, alive_pid], f)

        def fake_alive(pid):
            return pid in (current_pid, alive_pid)

        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", side_effect=fake_alive
        ):
            assert instancecheck.is_another_instance_running() == alive_pid

        remaining = instancecheck._read_lock_pids(lock_path)
        assert remaining == [current_pid, alive_pid]

    def test_all_stale_pids_remove_file(self, lock_dir):
        """A lock file containing only stale foreign PIDs is removed."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([88888, 77777], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=False
        ):
            assert instancecheck.is_another_instance_running() is None
        assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# create_lock_file / remove_lock_file
# ---------------------------------------------------------------------------


class TestCreateLockFile:
    """Tests for create_lock_file."""

    def test_creates_single_entry_without_duplication(self, lock_dir):
        """Creating the lock twice should still keep a single current PID."""
        _tmp_path, lock_path = lock_dir
        instancecheck.create_lock_file()
        instancecheck.create_lock_file()
        assert instancecheck._read_lock_pids(lock_path) == [os.getpid()]

    def test_rejects_live_foreign_pid_unless_forced(self, lock_dir):
        """A live foreign PID blocks creation unless force=True is used."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            with pytest.raises(RuntimeError, match="already running"):
                instancecheck.create_lock_file()
            instancecheck.create_lock_file(force=True)
        assert instancecheck._read_lock_pids(lock_path) == [foreign_pid, os.getpid()]

    def test_rejects_recycled_pid_with_wrong_signature(self, lock_dir):
        """A reused PID with a different create_time is cleaned as stale."""
        _tmp_path, lock_path = lock_dir
        current_pid = os.getpid()
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"pid": current_pid, "create_time": 1.0}],
                f,
            )
        instancecheck.create_lock_file()
        assert instancecheck._read_lock_pids(lock_path) == [current_pid]

    @pytest.mark.parametrize("force", [False, True])
    def test_replaces_stale_lock(self, lock_dir, force):
        """A stale PID is cleaned before creating the current lock entry."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([99999], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=False
        ):
            instancecheck.create_lock_file(force=force)
        assert instancecheck._read_lock_pids(lock_path) == [os.getpid()]


class TestRemoveLockFile:
    """Tests for remove_lock_file."""

    def test_removes_own_pid_but_preserves_foreign_entry(self, lock_dir):
        """Removing our PID keeps the foreign one registered and detectable."""
        _tmp_path, lock_path = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid, os.getpid()], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            instancecheck.remove_lock_file()
        assert os.path.exists(lock_path)
        assert instancecheck._read_lock_pids(lock_path) == [foreign_pid]
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            assert instancecheck.is_another_instance_running() == foreign_pid

    def test_remove_lock_file_noop_without_own_pid(self, lock_dir):
        """Removing without our PID present should leave the file unchanged."""
        _tmp_path, lock_path = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([99999], f)
        with mock.patch.object(
            instancecheck.DEFAULT_REGISTRY, "_is_pid_alive", return_value=True
        ):
            instancecheck.remove_lock_file()
        assert instancecheck._read_lock_pids(lock_path) == [99999]


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
