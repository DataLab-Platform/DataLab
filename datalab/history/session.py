# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""HistorySession: ordered list of HistoryAction with replay logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.config import _
from datalab.history.action import HistoryAction
from datalab.history.core import HISTORY_SCHEMA_VERSION, get_datetime_str
from datalab.history.replaymap import ReplayUuidMap

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


class HistorySession:
    """Object representing a history session, i.e. a list of actions.

    A history session is a list of actions that can be replayed in the same order
    as they were added to the history session. The history session can be saved to
    a file and loaded from a file.

    Args:
        title: Title of the history session
        number: Number of the history session
    """

    def __init__(self, title: str = "", number: int = 0) -> None:
        """Create a new history session"""
        prefix = _("Session")
        self.title = title if title else f"{prefix} {number:03d}"
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
        """Return a copy of this session with all UUIDs rewritten via ``uuid_remap``.

        Used by the Duplicate operation to build an independent session whose
        captured object references point to the cloned data objects.

        Args:
            title: Title for the new session.
            uuid_remap: Per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``.

        Returns:
            A new :class:`HistorySession` with all captured UUIDs remapped.
        """
        session = HistorySession(title=title, number=self.number)
        session.actions = [
            action.copy_with_uuid_remap(uuid_remap) for action in self.actions
        ]
        return session

    def is_current_state_compatible(
        self, mainwindow: DLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the selection before checking the state

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        if self.actions:
            return self.actions[0].is_current_state_compatible(
                mainwindow, restore_selection
            )
        return True

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the state of the workspace associated to the first action of session

        Args:
            mainwindow: DataLab's main window
        """
        if self.actions:
            self.actions[0].restore(mainwindow)

    def replay(
        self, mainwindow: DLMainWindow, restore_selection: bool, edit: bool
    ) -> None:
        """Replay the history session

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the workspace selection before replaying
            edit: if True, always open the dialog boxes to edit parameters, if False,
             use the parameters passed when creating the action
        """
        panels = (mainwindow.signalpanel, mainwindow.imagepanel)
        replay_map = ReplayUuidMap(panels)
        for action in self.actions[:]:
            before = replay_map.snapshot_object_ids()
            replay_map.claim_action_inputs(action)
            action.replay(
                mainwindow,
                restore_selection=restore_selection,
                edit=edit,
                uuid_remap=replay_map.mapping,
            )
            replay_map.capture_changes(action, before)

        if self.actions:
            select_last_compute_output(
                mainwindow, panels, replay_map.mapping, self.actions[-1]
            )

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
            self.number = reader.read_any()
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()
        self.actions = reader.read_object_list("actions", HistoryAction)

    def remove_action(self, action: HistoryAction) -> None:
        """Remove an action from the history session

        This implies removing all subsequent actions. If action is not found, this
        fails silently.

        Args:
            action: Action to remove
        """
        if action in self.actions:
            index = self.actions.index(action)
            self.actions = self.actions[:index]


def select_last_compute_output(
    mainwindow: DLMainWindow,
    panels: tuple,
    uuid_remap: dict[str, dict[str, str]],
    last_action: HistoryAction,
) -> None:
    """Select the output of the last compute action after a session replay.

    Visually closes the replay by highlighting the final result in its panel.
    No-op when the last action is not a compute action or its output is gone.

    Args:
        mainwindow: DataLab's main window
        panels: signal and image panels
        uuid_remap: per-panel ``{old_uuid: new_uuid}`` mapping built during replay
        last_action: last action of the replayed session
    """
    if last_action.kind != HistoryAction.KIND_COMPUTE:
        return
    hpanel = getattr(mainwindow, "historypanel", None)
    if hpanel is None:
        return
    output_uuid = hpanel.action_output_uuid(last_action)
    if not output_uuid:
        return
    panel_str = last_action.panel_str or ""
    mapped_uuid = uuid_remap.get(panel_str, {}).get(output_uuid, output_uuid)
    target_panel = next((p for p in panels if p.PANEL_STR_ID == panel_str), None)
    if target_panel is None:
        return
    try:
        target_panel.objview.select_objects([mapped_uuid])
    except KeyError:
        pass
