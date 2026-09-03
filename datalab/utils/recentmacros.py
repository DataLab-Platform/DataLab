# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Recent Macros cache
===========================

This module provides a persistent JSON cache of recently created or imported
macros, surviving across DataLab sessions. It mirrors the cross-workspace
"recent macros" cache provided by DataLab-Web (``src/storage/recentStore.ts``).

The cache file is stored next to the DataLab user configuration (see
:func:`sigimax.utils.conf.Configuration.get_path`).
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import time
import uuid
from typing import Any

from sigimax.utils.conf import Configuration

RECENT_FILENAME = "recent_macros.json"
MAX_ENTRIES = 50


def _path() -> str:
    """Return the absolute path of the recent-macros cache file."""
    return Configuration.get_path(RECENT_FILENAME)


def _code_hash(code: str) -> str:
    """Return a stable hash of the macro source code (used for deduplication)."""
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


def _load_raw() -> list[dict[str, Any]]:
    """Load raw entries from disk. Return ``[]`` on missing/invalid file."""
    path = _path()
    if not osp.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fdesc:
            data = json.load(fdesc)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict) and "uid" in entry]


def _save_raw(entries: list[dict[str, Any]]) -> None:
    """Write entries atomically to disk."""
    path = _path()
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fdesc:
        json.dump(entries, fdesc, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def list_recent(limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
    """Return the recent macros, most recently seen first.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of dicts with keys ``uid``, ``title``, ``code``, ``last_seen``,
        ``source``.
    """
    entries = sorted(_load_raw(), key=lambda e: e.get("last_seen", 0), reverse=True)
    return entries[:limit]


def get_recent(uid: str) -> dict[str, Any] | None:
    """Return a single entry by uid, or ``None`` if not found."""
    for entry in _load_raw():
        if entry.get("uid") == uid:
            return entry
    return None


def record_recent(title: str, code: str, source: str | None = None) -> dict[str, Any]:
    """Record a macro in the recent cache.

    If an entry with the same code hash already exists, its ``last_seen`` and
    ``title`` are refreshed in place (no duplicate is created).

    Args:
        title: Macro title.
        code: Python source code.
        source: Optional origin marker (e.g. ``"import"``, ``"ai"``,
         ``"template:simple"``).

    Returns:
        The stored entry.
    """
    entries = _load_raw()
    chash = _code_hash(code)
    now = time.time()
    matched: dict[str, Any] | None = None
    for entry in entries:
        if entry.get("hash") == chash:
            matched = entry
            break
    if matched is None:
        matched = {
            "uid": uuid.uuid4().hex,
            "title": title,
            "code": code,
            "hash": chash,
            "last_seen": now,
            "source": source,
        }
        entries.append(matched)
    else:
        matched["title"] = title
        matched["last_seen"] = now
        if source is not None:
            matched["source"] = source
    # Trim to MAX_ENTRIES (drop oldest)
    entries.sort(key=lambda e: e.get("last_seen", 0), reverse=True)
    entries = entries[:MAX_ENTRIES]
    _save_raw(entries)
    return matched


def remove_recent(uid: str) -> bool:
    """Remove a recent entry by uid. Return ``True`` if it existed."""
    entries = _load_raw()
    new_entries = [e for e in entries if e.get("uid") != uid]
    if len(new_entries) == len(entries):
        return False
    _save_raw(new_entries)
    return True


def clear_recent() -> None:
    """Remove all recent entries."""
    _save_raw([])
