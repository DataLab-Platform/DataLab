# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Bash-style input history for the AI assistant chat input.

Provides up/down navigation through previously submitted prompts and
optional persistence to a plain-text file (one entry per line, with newline
characters escaped) so that history survives across DataLab sessions.
"""

from __future__ import annotations

import os
import os.path as osp


class InputHistory:
    """Persistent input history with shell-like navigation.

    Args:
        filepath: Optional path to the on-disk backing file. When ``None``,
         the history lives only in memory.
        max_size: Maximum number of entries to keep (oldest entries are
         dropped when the limit is exceeded).
    """

    def __init__(self, filepath: str | None = None, max_size: int = 500) -> None:
        self._filepath = filepath
        self._max_size = int(max_size)
        self._items: list[str] = []
        # Index of the entry currently displayed when navigating; ``None``
        # means the user is editing a fresh draft (no navigation in progress).
        self._index: int | None = None
        # Draft preserved when the user starts navigating, restored when
        # navigating past the most recent entry.
        self._draft: str = ""
        self._load()

    @staticmethod
    def _encode(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _decode(line: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                nxt = line[i + 1]
                if nxt == "n":
                    out.append("\n")
                    i += 2
                    continue
                if nxt == "\\":
                    out.append("\\")
                    i += 2
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _load(self) -> None:
        if not self._filepath or not osp.isfile(self._filepath):
            return
        try:
            with open(self._filepath, encoding="utf-8") as file:
                self._items = [
                    self._decode(line) for line in file.read().splitlines() if line
                ]
        except OSError:
            self._items = []

    def _save(self) -> None:
        if not self._filepath:
            return
        try:
            os.makedirs(osp.dirname(self._filepath), exist_ok=True)
            with open(self._filepath, "w", encoding="utf-8") as file:
                for item in self._items[-self._max_size :]:
                    file.write(self._encode(item) + "\n")
        except OSError:
            pass

    def items(self) -> list[str]:
        """Return a shallow copy of all stored entries (oldest first)."""
        return list(self._items)

    def clear(self) -> None:
        """Erase the history (in memory and on disk)."""
        self._items = []
        self.reset_navigation()
        self._save()

    def add(self, text: str) -> None:
        """Append a submitted prompt to the history.

        Empty strings are ignored. Consecutive duplicates and any earlier
        identical entry are de-duplicated to keep the history compact.
        """
        text = text.strip("\n")
        if not text.strip():
            return
        # Deduplicate: drop any earlier identical entry, then append.
        self._items = [it for it in self._items if it != text]
        self._items.append(text)
        if len(self._items) > self._max_size:
            self._items = self._items[-self._max_size :]
        self.reset_navigation()
        self._save()

    def reset_navigation(self) -> None:
        """Forget the current navigation position and any preserved draft."""
        self._index = None
        self._draft = ""

    def previous(self, current_text: str) -> str | None:
        """Return the previous entry (older), or ``None`` if unavailable.

        Saves ``current_text`` as the working draft on the first call so it
        can be restored when navigating past the most recent entry.
        """
        if not self._items:
            return None
        if self._index is None:
            self._draft = current_text
            self._index = len(self._items) - 1
        elif self._index > 0:
            self._index -= 1
        else:
            return self._items[self._index]
        return self._items[self._index]

    def next(self, current_text: str) -> str | None:  # noqa: A003
        """Return the next entry (newer), restoring the draft past the end.

        Args:
            current_text: Current input text (kept for API symmetry with
             :meth:`prev`; the working draft is captured by ``prev`` and
             restored from there).

        Returns:
            The next history entry, or ``None`` when no navigation is in
            progress.
        """
        del current_text  # API symmetry only; draft handling lives in ``prev``
        if self._index is None:
            return None
        if self._index < len(self._items) - 1:
            self._index += 1
            return self._items[self._index]
        # Past the most recent entry: restore the original draft.
        self._index = None
        draft = self._draft
        self._draft = ""
        return draft
