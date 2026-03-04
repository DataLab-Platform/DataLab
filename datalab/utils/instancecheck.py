# Copyright (C) DataLab Platform Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Instance detection via PID lock file.

This module provides utilities to detect whether another DataLab instance is
already running, using a PID-based lock file stored in the user configuration
directory (e.g. ``~/.DataLab_v1/datalab.lock``).

The lock file uses a **reference-counting** approach: it stores a JSON list of
PIDs of all running DataLab instances.  Each new instance appends its PID, and
each closing instance removes its PID.  The file is only deleted when no
instances remain.  This prevents the bug where closing one of two concurrent
instances would delete the lock and allow a third instance to start without
any warning.

Cross-platform PID liveness check:

- **Linux / macOS**: ``os.kill(pid, 0)``
- **Windows**: ``ctypes.windll.kernel32.OpenProcess`` (because
  ``os.kill(pid, 0)`` calls ``TerminateProcess`` on Python 3.9–3.11).

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
import sys

logger = logging.getLogger(__name__)

LOCK_FILENAME = "datalab.lock"


def _get_lock_path() -> str:
    """Return the absolute path to the lock file.

    Returns:
        Absolute path to ``datalab.lock`` inside the configuration directory.
    """
    from datalab.config import Conf

    return Conf.get_path(LOCK_FILENAME)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    On Unix (Linux, macOS) this uses ``os.kill(pid, 0)``.
    On Windows this uses ``ctypes.windll.kernel32.OpenProcess`` because
    ``os.kill(pid, 0)`` calls ``TerminateProcess`` on Python < 3.12.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process is alive, False otherwise.
    """
    if sys.platform == "win32":
        return _is_pid_alive_win32(pid)
    return _is_pid_alive_posix(pid)


def _is_pid_alive_posix(pid: int) -> bool:
    """Unix implementation: ``os.kill(pid, 0)``."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except OSError:
        return False
    return True


def _is_pid_alive_win32(pid: int) -> bool:
    """Windows implementation using ``kernel32.OpenProcess``.

    Opens the process with ``PROCESS_QUERY_LIMITED_INFORMATION`` and then
    calls ``GetExitCodeProcess`` to distinguish a *running* process from
    one that has terminated but whose handle is still open (Windows keeps
    zombie process objects until all handles are closed).

    .. note::

       This only checks that *a* process with the given PID exists, not
       that it is a DataLab process.  PID recycling may cause false
       positives (see module docstring).
    """
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259  # https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == STILL_ACTIVE
        # GetExitCodeProcess failed — assume alive to be safe
        return True
    finally:
        kernel32.CloseHandle(handle)


def _remove_lock_path(lock_path: str) -> None:
    """Remove the lock file at the given path.

    Args:
        lock_path: Absolute path to the lock file to remove.
    """
    try:
        os.remove(lock_path)
    except OSError:
        logger.warning("Could not remove lock file '%s'", lock_path)


def _read_lock_pids(lock_path: str) -> list[int]:
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
        with open(lock_path) as fobj:
            content = fobj.read().strip()
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("Could not read lock file '%s'", lock_path)
        return []

    if not content:
        logger.warning("Empty lock file '%s', removing", lock_path)
        _remove_lock_path(lock_path)
        return []

    # Try JSON list format first (new format)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [int(p) for p in data]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fall back to legacy single-PID format
    try:
        return [int(content)]
    except ValueError:
        logger.warning("Corrupted lock file '%s', removing", lock_path)
        _remove_lock_path(lock_path)
        return []


def _write_lock_pids(lock_path: str, pids: list[int]) -> None:
    """Write the list of PIDs to the lock file in JSON format.

    If the list is empty, the lock file is removed instead.

    Args:
        lock_path: Absolute path to the lock file.
        pids: List of PIDs to write.
    """
    if not pids:
        _remove_lock_path(lock_path)
        return
    try:
        with open(lock_path, "w") as fobj:
            json.dump(pids, fobj)
    except OSError:
        logger.warning("Could not write lock file '%s'", lock_path)


def _read_lock_pid(lock_path: str) -> int | None:
    """Read and return the PID stored in the lock file (legacy compat).

    If the file is missing, unreadable, or contains non-integer content,
    returns None and cleans up the corrupted file when appropriate.

    Args:
        lock_path: Absolute path to the lock file.

    Returns:
        The first stored PID, or None if unreadable/missing.
    """
    pids = _read_lock_pids(lock_path)
    return pids[0] if pids else None


def is_another_instance_running() -> int | None:
    """Check if another DataLab instance is already running.

    Reads the lock file and checks whether any stored PID (other than the
    current process) corresponds to a live process.  Stale PIDs left by
    crashed instances are automatically cleaned up.

    Returns:
        PID of the first live foreign instance if found, None otherwise.
    """
    lock_path = _get_lock_path()
    pids = _read_lock_pids(lock_path)
    if not pids:
        return None

    my_pid = os.getpid()
    live_pids = []
    found_foreign = None

    for pid in pids:
        if pid == my_pid:
            live_pids.append(pid)
            continue
        if _is_pid_alive(pid):
            live_pids.append(pid)
            if found_foreign is None:
                found_foreign = pid
        else:
            logger.info("Removing stale PID %d from lock file", pid)

    # Rewrite the lock file if stale PIDs were cleaned
    if len(live_pids) != len(pids):
        _write_lock_pids(lock_path, live_pids)

    return found_foreign


def create_lock_file(*, force: bool = False) -> None:
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
    lock_path = _get_lock_path()

    if not force:
        existing_pid = is_another_instance_running()
        if existing_pid is not None:
            raise RuntimeError(
                f"Another DataLab instance is already running (PID {existing_pid})"
            )

    if force:
        logger.info("Force-creating lock file (user override)")

    # Read existing PIDs, add ours, write back
    pids = _read_lock_pids(lock_path)
    my_pid = os.getpid()
    if my_pid not in pids:
        pids.append(my_pid)
    _write_lock_pids(lock_path, pids)


def remove_lock_file() -> None:
    """Remove the current process from the lock file.

    Removes our PID from the list of running instances.  The file is only
    deleted when no instances remain.  If other instances are still
    registered, the lock file is rewritten without our PID.
    """
    lock_path = _get_lock_path()
    pids = _read_lock_pids(lock_path)
    if not pids:
        return

    my_pid = os.getpid()
    if my_pid in pids:
        pids.remove(my_pid)
        _write_lock_pids(lock_path, pids)
    else:
        logger.warning(
            "Lock file does not contain current PID %d — not modifying",
            my_pid,
        )
