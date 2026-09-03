# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Macro recovery cache
============================

This module provides an auto-saved "pending changes" cache for macros, used to
recover unsaved work after a crash or an unexpected shutdown.

It is independent of the HDF5 workspace serialization: every time the user
modifies a macro, the new code is written to a JSON file (debounced) so it can
be restored at the next startup.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time
from typing import Any

from sigimax.utils.conf import Configuration

PENDING_FILENAME = "recent_macros_pending.json"


def _path() -> str:
    """Return the absolute path of the pending-macros cache file."""
    return Configuration.get_path(PENDING_FILENAME)


def _load_raw() -> dict[str, dict[str, Any]]:
    """Load pending entries from disk. Return ``{}`` on missing/invalid file."""
    path = _path()
    if not osp.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fdesc:
            data = json.load(fdesc)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        uid: entry
        for uid, entry in data.items()
        if isinstance(entry, dict) and "code" in entry
    }


def _save_raw(entries: dict[str, dict[str, Any]]) -> None:
    """Write entries atomically to disk."""
    path = _path()
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fdesc:
        json.dump(entries, fdesc, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def save_pending(uid: str, title: str, code: str) -> None:
    """Persist the current state of a macro to the recovery cache."""
    entries = _load_raw()
    entries[uid] = {
        "uid": uid,
        "title": title,
        "code": code,
        "last_saved": time.time(),
    }
    _save_raw(entries)


def load_pending() -> dict[str, dict[str, Any]]:
    """Return all pending macros, keyed by uid."""
    return _load_raw()


def clear_pending(uid: str | None = None) -> None:
    """Remove a single pending entry (by uid) or all of them when ``uid`` is None."""
    if uid is None:
        _save_raw({})
        return
    entries = _load_raw()
    if uid in entries:
        del entries[uid]
        _save_raw(entries)
