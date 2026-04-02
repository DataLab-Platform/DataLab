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
    """Provide a temporary directory and a registry with a patched lock path."""
    registry = instancecheck.ApplicationInstanceRegistry(
        lock_filename="TestDataLab.lock"
    )
    lock_path = str(tmp_path / registry.lock_filename)
    with mock.patch.object(registry, "_get_lock_path", return_value=lock_path):
        yield tmp_path, lock_path, registry


# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------


class TestIsPidAlive:
    """Tests for the _is_pid_alive helper."""

    def test_current_process_is_alive(self):
        """Current process PID should be reported as alive."""
        registry = instancecheck.ApplicationInstanceRegistry()
        assert registry.is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        """A nonexistent PID should be reported as dead."""
        registry = instancecheck.ApplicationInstanceRegistry()
        with mock.patch("psutil.pid_exists", return_value=False):
            assert registry.is_pid_alive(999_999_999) is False

    def test_invalid_pid_is_dead(self):
        """Non-positive PIDs should be rejected before psutil is queried."""
        registry = instancecheck.ApplicationInstanceRegistry()
        with mock.patch("psutil.pid_exists") as mock_pid_exists:
            assert registry.is_pid_alive(0) is False
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

        registry = instancecheck.ApplicationInstanceRegistry()
        assert registry.read_lock_pids(target) == expected
        assert lock.exists() is (not removed and content is not None)


class TestReadLockPidLegacy:
    """Tests for _read_lock_pid (legacy compat wrapper)."""

    def test_returns_first_pid_for_supported_formats(self, tmp_path):
        """Should return the first PID in JSON and legacy formats."""
        registry = instancecheck.ApplicationInstanceRegistry()
        lock = tmp_path / "json.lock"
        lock.write_text(json.dumps([12345, 67890]))
        assert registry.read_lock_pid(str(lock)) == 12345

        lock = tmp_path / "entry.lock"
        lock.write_text(json.dumps([{"pid": 23456, "create_time": 1.0}]))
        assert registry.read_lock_pid(str(lock)) == 23456

        lock = tmp_path / "legacy.lock"
        lock.write_text("12345")
        assert registry.read_lock_pid(str(lock)) == 12345


# ---------------------------------------------------------------------------
# is_another_instance_running
# ---------------------------------------------------------------------------


class TestIsAnotherInstanceRunning:
    """Tests for is_another_instance_running."""

    def test_no_lock_file(self, lock_dir):
        """No lock file → no other instance."""
        _tmp_path, _lock_path, registry = lock_dir
        assert registry.is_another_instance_running() is None

    def test_mixed_pids_keep_only_live_foreign_entries(self, lock_dir):
        """Own and stale PIDs are ignored while a live foreign PID is kept."""
        _tmp_path, lock_path, registry = lock_dir
        current_pid = os.getpid()
        alive_pid = 99999
        dead_pid = 88888
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([current_pid, dead_pid, alive_pid], f)

        def fake_alive(pid):
            return pid in (current_pid, alive_pid)

        with mock.patch.object(registry, "_is_pid_alive", side_effect=fake_alive):
            assert registry.is_another_instance_running() == alive_pid

        remaining = registry.read_lock_pids(lock_path)
        assert remaining == [current_pid, alive_pid]

    def test_all_stale_pids_remove_file(self, lock_dir):
        """A lock file containing only stale foreign PIDs is removed."""
        _tmp_path, lock_path, registry = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([88888, 77777], f)
        with mock.patch.object(registry, "_is_pid_alive", return_value=False):
            assert registry.is_another_instance_running() is None
        assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# create_lock_file / remove_lock_file
# ---------------------------------------------------------------------------


class TestCreateLockFile:
    """Tests for create_lock_file."""

    def test_creates_single_entry_without_duplication(self, lock_dir):
        """Creating the lock twice should still keep a single current PID."""
        _tmp_path, lock_path, registry = lock_dir
        registry.create_lock_file()
        registry.create_lock_file()
        assert registry.read_lock_pids(lock_path) == [os.getpid()]

    def test_rejects_live_foreign_pid_unless_forced(self, lock_dir):
        """A live foreign PID blocks creation unless force=True is used."""
        _tmp_path, lock_path, registry = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid], f)
        with mock.patch.object(registry, "_is_pid_alive", return_value=True):
            with pytest.raises(RuntimeError, match="already running"):
                registry.create_lock_file()
            registry.create_lock_file(force=True)
        assert registry.read_lock_pids(lock_path) == [foreign_pid, os.getpid()]

    def test_rejects_recycled_pid_with_wrong_signature(self, lock_dir):
        """A reused PID with a different create_time is cleaned as stale."""
        _tmp_path, lock_path, registry = lock_dir
        current_pid = os.getpid()
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"pid": current_pid, "create_time": 1.0}],
                f,
            )
        registry.create_lock_file()
        assert registry.read_lock_pids(lock_path) == [current_pid]

    @pytest.mark.parametrize("force", [False, True])
    def test_replaces_stale_lock(self, lock_dir, force):
        """A stale PID is cleaned before creating the current lock entry."""
        _tmp_path, lock_path, registry = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([99999], f)
        with mock.patch.object(registry, "_is_pid_alive", return_value=False):
            registry.create_lock_file(force=force)
        assert registry.read_lock_pids(lock_path) == [os.getpid()]


class TestRemoveLockFile:
    """Tests for remove_lock_file."""

    def test_removes_own_pid_but_preserves_foreign_entry(self, lock_dir):
        """Removing our PID keeps the foreign one registered and detectable."""
        _tmp_path, lock_path, registry = lock_dir
        foreign_pid = 99999
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([foreign_pid, os.getpid()], f)
        with mock.patch.object(registry, "_is_pid_alive", return_value=True):
            registry.remove_lock_file()
        assert os.path.exists(lock_path)
        assert registry.read_lock_pids(lock_path) == [foreign_pid]
        with mock.patch.object(registry, "_is_pid_alive", return_value=True):
            assert registry.is_another_instance_running() == foreign_pid

    def test_remove_lock_file_noop_without_own_pid(self, lock_dir):
        """Removing without our PID present should leave the file unchanged."""
        _tmp_path, lock_path, registry = lock_dir
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump([99999], f)
        with mock.patch.object(registry, "_is_pid_alive", return_value=True):
            registry.remove_lock_file()
        assert registry.read_lock_pids(lock_path) == [99999]


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

        registry = instancecheck.ApplicationInstanceRegistry()
        lock_path = Conf.get_path(registry.lock_filename)
        assert registry.lock_filename == f"{APP_NAME}.lock"
        assert ".none" not in lock_path, (
            f"Lock path points to uninitialized config dir: {lock_path}"
        )
        assert registry.get_lock_path() == lock_path
