# Copyright (C) DataLab Platform Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Instance detection via PID lock file.

This module provides utilities to detect whether another DataLab instance is
already running, using a PID-based lock file stored in the user configuration
directory (e.g. ``~/.DataLab_v1/datalab.lock``).

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


def _read_lock_pid(lock_path: str) -> int | None:
    """Read and return the PID stored in the lock file.

    If the file is missing, unreadable, or contains non-integer content,
    returns None and cleans up the corrupted file when appropriate.

    Args:
        lock_path: Absolute path to the lock file.

    Returns:
        The stored PID, or None if unreadable/missing.
    """
    try:
        with open(lock_path) as fobj:
            content = fobj.read().strip()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Could not read lock file '%s'", lock_path)
        return None

    try:
        return int(content)
    except ValueError:
        logger.warning("Corrupted lock file '%s', removing", lock_path)
        _remove_lock_path(lock_path)
        return None


def is_another_instance_running() -> int | None:
    """Check if another DataLab instance is already running.

    Reads the lock file and checks whether the stored PID corresponds to a
    live process.  Stale lock files left by crashed instances are
    automatically removed.

    Returns:
        PID of the running instance if alive, None otherwise.
    """
    lock_path = _get_lock_path()
    pid = _read_lock_pid(lock_path)
    if pid is None:
        return None

    if pid == os.getpid():
        # Our own lock file – not "another" instance
        return None

    if _is_pid_alive(pid):
        return pid

    # Stale lock file from a crashed instance
    logger.info("Removing stale lock file (PID %d no longer running)", pid)
    _remove_lock_path(lock_path)
    return None


def create_lock_file(*, force: bool = False) -> None:
    """Create the lock file with the current process PID.

    If the lock file already exists and contains a live PID, a
    :class:`RuntimeError` is raised (unless *force* is True).  Stale lock
    files left by crashed instances are cleaned up automatically.

    Args:
        force: If True, overwrite the lock file unconditionally.  This is
         used when the user has already been warned about another running
         instance and chose to continue anyway.

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

    with open(lock_path, "w") as fobj:
        fobj.write(str(os.getpid()))


def remove_lock_file() -> None:
    """Remove the lock file if it was created by the current process.

    Safety check: the file is only removed when its stored PID matches
    ``os.getpid()``, to avoid deleting another instance's lock.
    """
    lock_path = _get_lock_path()
    pid = _read_lock_pid(lock_path)
    if pid is None:
        return

    if pid == os.getpid():
        _remove_lock_path(lock_path)
    else:
        logger.warning(
            "Lock file contains PID %d, but current PID is %d — not removing",
            pid,
            os.getpid(),
        )
