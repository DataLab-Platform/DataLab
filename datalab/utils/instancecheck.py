# Copyright (C) DataLab Platform Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Instance detection via PID lock file.

This module provides a class-based registry to detect whether another DataLab
instance is already running, using a PID-based lock file stored in the user
configuration directory (e.g. ``~/.DataLab_v1/DataLab.lock``).

The lock file uses a **reference-counting** approach: it stores a JSON list of
PIDs of all running DataLab instances.  Each new instance appends its PID, and
each closing instance removes its PID.  The file is only deleted when no
instances remain.  This prevents the bug where closing one of two concurrent
instances would delete the lock and allow a third instance to start without
any warning.

Cross-platform PID liveness check:

- **All platforms**: ``psutil.pid_exists(pid)``

.. note::

   The PID-based approach is subject to PID recycling: if the original
   DataLab process crashes and the OS reassigns the same PID to an
   unrelated process, the lock will be considered still held (false
   positive).  This is acceptable for an advisory lock whose sole
   purpose is to warn the user.
"""

from __future__ import annotations

import json
import logging
import os

import psutil

from datalab.config import APP_NAME, Conf

logger = logging.getLogger(__name__)


class ApplicationInstanceRegistry:
    """PID-based registry for concurrent DataLab instances."""

    def __init__(self, app_name: str | None = None) -> None:
        """Initialize registry with the target application name."""
        if app_name is None:
            app_name = APP_NAME
        self.app_name = app_name
        self.lock_filename = f"{app_name}.lock"

    def _get_lock_path(self) -> str:
        """Return the absolute path to the lock file.

        Returns:
            Absolute path to the lock file inside the configuration directory.
        """
        return Conf.get_path(self.lock_filename)

    def get_lock_path(self) -> str:
        """Return the absolute path to the lock file."""
        return self._get_lock_path()

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is still running.

        This uses ``psutil.pid_exists(pid)`` on all supported platforms.

        Args:
            pid: Process ID to check.

        Returns:
            True if the process is alive, False otherwise.
        """
        if pid <= 0:
            return False
        return psutil.pid_exists(pid)

    def is_pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        return self._is_pid_alive(pid)

    def _remove_lock_path(self, lock_path: str) -> None:
        """Remove the lock file at the given path.

        Args:
            lock_path: Absolute path to the lock file to remove.
        """
        try:
            os.remove(lock_path)
        except OSError:
            logger.warning("Could not remove lock file '%s'", lock_path)

    def remove_lock_path(self, lock_path: str) -> None:
        """Remove the lock file at the given path."""
        self._remove_lock_path(lock_path)

    def _read_lock_pids(self, lock_path: str) -> list[int]:
        """Read and return the list of PIDs stored in the lock file.

        Supports both the legacy single-PID format (plain integer) and the new
        JSON-list format.  If the file is missing, unreadable, or corrupted,
        returns an empty list and cleans up the file when appropriate.

        Args:
            lock_path: Absolute path to the lock file.

        Returns:
            List of stored PIDs (may be empty).
        """
        try:
            with open(lock_path, encoding="utf-8") as fobj:
                content = fobj.read().strip()
        except FileNotFoundError:
            return []
        except OSError:
            logger.warning("Could not read lock file '%s'", lock_path)
            return []

        if not content:
            logger.warning("Empty lock file '%s', removing", lock_path)
            self._remove_lock_path(lock_path)
            return []

        # Try JSON list format first (new format)
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [int(pid) for pid in data]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fall back to legacy single-PID format
        try:
            return [int(content)]
        except ValueError:
            logger.warning("Corrupted lock file '%s', removing", lock_path)
            self._remove_lock_path(lock_path)
            return []

    def read_lock_pids(self, lock_path: str) -> list[int]:
        """Read and return the list of PIDs stored in the lock file."""
        return self._read_lock_pids(lock_path)

    def _write_lock_pids(self, lock_path: str, pids: list[int]) -> None:
        """Write the list of PIDs to the lock file in JSON format.

        If the list is empty, the lock file is removed instead.

        Args:
            lock_path: Absolute path to the lock file.
            pids: List of PIDs to write.
        """
        if not pids:
            self._remove_lock_path(lock_path)
            return
        try:
            with open(lock_path, "w", encoding="utf-8") as fobj:
                json.dump(pids, fobj)
        except OSError:
            logger.warning("Could not write lock file '%s'", lock_path)

    def write_lock_pids(self, lock_path: str, pids: list[int]) -> None:
        """Write the list of PIDs to the lock file."""
        self._write_lock_pids(lock_path, pids)

    def _read_lock_pid(self, lock_path: str) -> int | None:
        """Read and return the PID stored in the lock file (legacy compat).

        If the file is missing, unreadable, or contains non-integer content,
        returns None and cleans up the corrupted file when appropriate.

        Args:
            lock_path: Absolute path to the lock file.

        Returns:
            The first stored PID, or None if unreadable/missing.
        """
        pids = self._read_lock_pids(lock_path)
        return pids[0] if pids else None

    def read_lock_pid(self, lock_path: str) -> int | None:
        """Read and return the first PID stored in the lock file."""
        return self._read_lock_pid(lock_path)

    def _cleanup_stale_lock_pids(self, lock_path: str) -> list[int]:
        """Remove stale PIDs from the lock file and return live entries.

        If no live PID remains, the lock file is removed.

        Args:
            lock_path: Absolute path to the lock file.

        Returns:
            List of live PIDs still registered in the lock file.
        """
        pids = self._read_lock_pids(lock_path)
        if not pids:
            return []

        my_pid = os.getpid()
        live_pids = []
        for pid in pids:
            if pid == my_pid or self._is_pid_alive(pid):
                live_pids.append(pid)
            else:
                logger.info("Removing stale PID %d from lock file", pid)

        if len(live_pids) != len(pids):
            self._write_lock_pids(lock_path, live_pids)

        return live_pids

    def is_another_instance_running(self) -> int | None:
        """Check if another DataLab instance is already running.

        Reads the lock file and checks whether any stored PID (other than the
        current process) corresponds to a live process.  Stale PIDs left by
        crashed instances are automatically cleaned up.

        Returns:
            PID of the first live foreign instance if found, None otherwise.
        """
        lock_path = self._get_lock_path()
        pids = self._cleanup_stale_lock_pids(lock_path)
        if not pids:
            return None

        my_pid = os.getpid()
        for pid in pids:
            if pid != my_pid:
                return pid
        return None

    def create_lock_file(self, *, force: bool = False) -> None:
        """Register the current process in the lock file.

        Adds the current PID to the list of running instances.  If other live
        instances exist and *force* is False, a :class:`RuntimeError` is raised.
        Stale PIDs from crashed instances are cleaned up automatically.

        Args:
            force: If True, add our PID even though another instance is running.
             This is used when the user has already been warned about another
             running instance and chose to continue anyway.

        Raises:
            RuntimeError: If another live instance already holds the lock and
             *force* is False.
        """
        lock_path = self._get_lock_path()
        pids = self._cleanup_stale_lock_pids(lock_path)
        my_pid = os.getpid()

        if not force:
            for pid in pids:
                if pid != my_pid:
                    existing_pid = pid
                    break
            else:
                existing_pid = None
            if existing_pid is not None:
                raise RuntimeError(
                    f"Another DataLab instance is already running (PID {existing_pid})"
                )

        if force:
            logger.info("Force-creating lock file (user override)")

        if my_pid not in pids:
            pids.append(my_pid)
        self._write_lock_pids(lock_path, pids)

    def remove_lock_file(self) -> None:
        """Remove the current process from the lock file.

        Removes our PID from the list of running instances.  The file is only
        deleted when no instances remain.  If other instances are still
        registered, the lock file is rewritten without our PID.
        """
        lock_path = self._get_lock_path()
        pids = self._cleanup_stale_lock_pids(lock_path)
        if not pids:
            return

        my_pid = os.getpid()
        if my_pid in pids:
            pids.remove(my_pid)
            self._write_lock_pids(lock_path, pids)
        else:
            logger.warning(
                "Lock file does not contain current PID %d — not modifying",
                my_pid,
            )


DEFAULT_REGISTRY = ApplicationInstanceRegistry()
LOCK_FILENAME = DEFAULT_REGISTRY.lock_filename


def _get_lock_path() -> str:
    """Return the default registry lock path."""
    return DEFAULT_REGISTRY.get_lock_path()


def _is_pid_alive(pid: int) -> bool:
    """Return whether the PID is alive according to the default registry."""
    return DEFAULT_REGISTRY.is_pid_alive(pid)


def _remove_lock_path(lock_path: str) -> None:
    """Remove a lock file using the default registry."""
    DEFAULT_REGISTRY.remove_lock_path(lock_path)


def _read_lock_pids(lock_path: str) -> list[int]:
    """Read lock-file PIDs using the default registry."""
    return DEFAULT_REGISTRY.read_lock_pids(lock_path)


def _write_lock_pids(lock_path: str, pids: list[int]) -> None:
    """Write lock-file PIDs using the default registry."""
    DEFAULT_REGISTRY.write_lock_pids(lock_path, pids)


def _read_lock_pid(lock_path: str) -> int | None:
    """Read the first lock-file PID using the default registry."""
    return DEFAULT_REGISTRY.read_lock_pid(lock_path)


def is_another_instance_running() -> int | None:
    """Return the PID of another running instance, if any."""
    return DEFAULT_REGISTRY.is_another_instance_running()


def create_lock_file(*, force: bool = False) -> None:
    """Register the current process in the default registry lock file."""
    DEFAULT_REGISTRY.create_lock_file(force=force)


def remove_lock_file() -> None:
    """Remove the current process from the default registry lock file."""
    DEFAULT_REGISTRY.remove_lock_file()
