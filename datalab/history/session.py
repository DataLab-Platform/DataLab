# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""HistorySession: ordered list of HistoryAction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.config import _
from datalab.history.action import HistoryAction
from datalab.history.core import HISTORY_SCHEMA_VERSION, get_datetime_str

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


class HistorySession:
    """Object representing a history session, i.e. a list of actions.

    A history session groups, in chronological order, the actions forming a
    processing chain. Compute actions are recomputed in place by the History
    panel recompute engine (:meth:`HistoryAction.replay` raises
    ``NotImplementedError`` for the compute kind); UI and mutation actions
    are replayed through :meth:`HistoryAction.replay`. The session can be
    saved to a file and loaded from a file.

    Args:
        title: Title of the history session
        number: Number of the history session
    """

    def __init__(self, title: str | None = None, number: int = 0) -> None:
        """Create a new history session"""
        self.title = _("Processing") if title is None else title
        self.number = number
        self.dtstr: str = get_datetime_str()
        self.actions: list[HistoryAction] = []
        self.schema_version: int = HISTORY_SCHEMA_VERSION

    def add_action(self, action: HistoryAction) -> None:
        """Add an action to the history session

        Args:
            action: Action to add
        """
        self.actions.append(action)

    def copy(
        self, title: str | None = None, action_title_suffix: str | None = None
    ) -> HistorySession:
        """Return an independent copy of this history session."""
        session = HistorySession(title=title or self.title, number=self.number)
        session.actions = [
            action.copy(title_suffix=action_title_suffix) for action in self.actions
        ]
        return session

    def copy_with_uuid_remap(
        self, title: str, uuid_remap: dict[str, dict[str, str]]
    ) -> HistorySession:
        """Return a copy with supported UUID references rewritten via ``uuid_remap``.

        Used by the Duplicate operation to build an independent session whose
        captured object references point to the cloned data objects.

        Args:
            title: Title for the new session.
            uuid_remap: Per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``.

        Returns:
            A new :class:`HistorySession` with supported object references
             remapped.
        """
        session = HistorySession(title=title, number=self.number)
        session.actions = [
            action.copy_with_uuid_remap(uuid_remap) for action in self.actions
        ]
        return session

    def is_current_state_compatible(self, mainwindow: DLMainWindow) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        if self.actions:
            return self.actions[0].is_current_state_compatible(mainwindow)
        return True

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the state of the workspace associated to the first action of session

        Args:
            mainwindow: DataLab's main window
        """
        if self.actions:
            self.actions[0].restore(mainwindow)

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this history session

        Args:
            writer: Writer
        """
        with writer.group("schema_version"):
            writer.write(self.schema_version)
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("number"):
            writer.write(self.number)
        with writer.group("dtstr"):
            writer.write(self.dtstr)
        writer.write_object_list(self.actions, "actions")

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this history session

        Args:
            reader: Reader
        """
        self.schema_version = reader.read(
            "schema_version", default=HISTORY_SCHEMA_VERSION
        )
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("number"):
            # HDF5 readers return NumPy scalars: coerce to a plain Python int
            self.number = int(reader.read_any())
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()
        self.actions = reader.read_object_list("actions", HistoryAction)
