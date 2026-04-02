# Copyright (C) DataLab Platform Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Instance detection via PID lock file.

This module provides a class-based registry to detect whether another DataLab
instance is already running, using a process-signature lock file stored in the
user configuration directory (e.g. ``~/.DataLab_v1/DataLab.lock``).

The lock file uses a **reference-counting** approach: it stores a JSON list of
running-instance signatures. Each new instance appends its signature, and each
closing instance removes its own entry. The file is only deleted when no
instances remain. This prevents the bug where closing one of two concurrent
instances would delete the lock and allow a third instance to start without
any warning.

Cross-platform PID liveness check:

- **All platforms**: ``psutil.pid_exists(pid)``

.. note::

    Legacy lock files storing only PIDs are still supported for backward
    compatibility. New lock entries also store process creation time to reduce
    false positives caused by PID recycling.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Union

import psutil

from datalab.config import APP_NAME, Conf

logger = logging.getLogger(__name__)


LockEntry = Dict[str, Union[int, float]]


class ApplicationInstanceRegistry:
    """PID-based registry for concurrent DataLab instances."""

    def __init__(self, lock_filename: str | None = None) -> None:
        """Initialize registry with the lock file name."""
        if lock_filename is None:
            lock_filename = f"{APP_NAME}.lock"
        self.lock_filename = lock_filename

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

    def _create_lock_entry(self, pid: int | None = None) -> LockEntry | None:
        """Create a lock entry for the given process.

        Args:
            pid: Process ID to describe. Defaults to the current process.

        Returns:
            Lock entry dictionary, or None if the process no longer exists.
        """
        if pid is None:
            pid = os.getpid()
        if pid <= 0:
            return None
        if not psutil.pid_exists(pid):
            return None

        try:
            process = psutil.Process(pid)
            return {
                "pid": pid,
                "create_time": process.create_time(),
            }
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except psutil.AccessDenied:
            return {"pid": pid}

    def is_pid_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        return self._is_pid_alive(pid)

    def _normalize_lock_entry(self, item: object) -> LockEntry:
        """Normalize a lock entry loaded from disk.

        Args:
            item: Raw decoded JSON item.

        Returns:
            Normalized lock entry.

        Raises:
            ValueError: If the item cannot be interpreted as a lock entry.
        """
        if isinstance(item, int):
            return {"pid": int(item)}

        if isinstance(item, dict) and "pid" in item:
            pid = int(item["pid"])
            entry: LockEntry = {"pid": pid}
            if "create_time" in item:
                entry["create_time"] = float(item["create_time"])
            return entry

        raise ValueError("Invalid lock entry")

    def _read_lock_entries(self, lock_path: str) -> list[LockEntry]:
        """Read and return normalized lock entries from the lock file."""
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

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [self._normalize_lock_entry(item) for item in data]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        try:
            return [self._normalize_lock_entry(int(content))]
        except ValueError:
            logger.warning("Corrupted lock file '%s', removing", lock_path)
            self._remove_lock_path(lock_path)
            return []

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
        return [int(entry["pid"]) for entry in self._read_lock_entries(lock_path)]

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

    def _write_lock_entries(self, lock_path: str, entries: list[LockEntry]) -> None:
        """Write lock entries to the lock file in JSON format."""
        if not entries:
            self._remove_lock_path(lock_path)
            return
        try:
            with open(lock_path, "w", encoding="utf-8") as fobj:
                json.dump(entries, fobj)
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

    def _is_lock_entry_alive(self, entry: LockEntry) -> bool:
        """Return whether a lock entry still matches a live process."""
        pid = int(entry["pid"])
        is_alive = False
        if not self._is_pid_alive(pid):
            return False

        if "create_time" not in entry:
            return True

        try:
            process = psutil.Process(pid)
            is_alive = True
            if "create_time" in entry:
                create_time = float(entry["create_time"])
                if abs(process.create_time() - create_time) > 1e-3:
                    is_alive = False
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            is_alive = False
        except psutil.AccessDenied:
            is_alive = True

        return is_alive

    def _cleanup_stale_lock_entries(self, lock_path: str) -> list[LockEntry]:
        """Remove stale lock entries and return live entries.

        If no live PID remains, the lock file is removed.

        Args:
            lock_path: Absolute path to the lock file.

        Returns:
            List of live lock entries still registered in the lock file.
        """
        entries = self._read_lock_entries(lock_path)
        if not entries:
            return []

        live_entries = []
        for entry in entries:
            if self._is_lock_entry_alive(entry):
                live_entries.append(entry)
            else:
                logger.info("Removing stale PID %d from lock file", int(entry["pid"]))

        if len(live_entries) != len(entries):
            self._write_lock_entries(lock_path, live_entries)

        return live_entries

    def is_another_instance_running(self) -> int | None:
        """Check if another DataLab instance is already running.

        Reads the lock file and checks whether any stored PID (other than the
        current process) corresponds to a live process.  Stale PIDs left by
        crashed instances are automatically cleaned up.

        Returns:
            PID of the first live foreign instance if found, None otherwise.
        """
        lock_path = self._get_lock_path()
        entries = self._cleanup_stale_lock_entries(lock_path)
        if not entries:
            return None

        my_pid = os.getpid()
        for entry in entries:
            pid = int(entry["pid"])
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
        entries = self._cleanup_stale_lock_entries(lock_path)
        pids = [int(entry["pid"]) for entry in entries]
        my_pid = os.getpid()
        my_entry = self._create_lock_entry(my_pid)

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

        entries = [entry for entry in entries if int(entry["pid"]) != my_pid]
        if my_entry is not None:
            entries.append(my_entry)
        self._write_lock_entries(lock_path, entries)

    def remove_lock_file(self) -> None:
        """Remove the current process from the lock file.

        Removes our PID from the list of running instances.  The file is only
        deleted when no instances remain.  If other instances are still
        registered, the lock file is rewritten without our PID.
        """
        lock_path = self._get_lock_path()
        entries = self._cleanup_stale_lock_entries(lock_path)
        if not entries:
            return

        my_pid = os.getpid()
        remaining_entries = [entry for entry in entries if int(entry["pid"]) != my_pid]
        if len(remaining_entries) != len(entries):
            self._write_lock_entries(lock_path, remaining_entries)
        else:
            logger.warning(
                "Lock file does not contain current PID %d — not modifying",
                my_pid,
            )
