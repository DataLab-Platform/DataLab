# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""UUID reconciliation for history session replay."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.history.action import HistoryAction

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel


class ReplayUuidMap:
    """Reconcile recorded object UUIDs with objects created during replay.

    Args:
        panels: Data panels participating in the replay
    """

    def __init__(self, panels: tuple[BaseDataPanel, ...]) -> None:
        """Initialize an empty per-panel UUID reconciliation map."""
        self.panels = {panel.PANEL_STR_ID: panel for panel in panels}
        self.mapping: dict[str, dict[str, str]] = {
            panel_str: {} for panel_str in self.panels
        }
        self.unclaimed: dict[str, list[str]] = {
            panel_str: [] for panel_str in self.panels
        }

    def translate_uuid(self, panel_str: str, uuid: str) -> str:
        """Translate a recorded UUID to its replay UUID."""
        return self.mapping.get(panel_str, {}).get(uuid, uuid)

    def snapshot_object_ids(self) -> dict[str, set[str]]:
        """Return the current object UUIDs for every replay panel."""
        return {
            panel_str: set(panel.objmodel.get_object_ids())
            for panel_str, panel in self.panels.items()
        }

    def claim_action_inputs(self, action: HistoryAction) -> None:
        """Claim replay objects corresponding to a compute action's inputs."""
        if action.kind != HistoryAction.KIND_COMPUTE:
            return
        panel_str = action.panel_str or ""
        recorded_uuids = list(action.state.selection.get(panel_str, []))
        if action.pattern == "2_to_1":
            second_uuids = action.kwargs.get("obj2_uuids") or []
            if isinstance(second_uuids, str):
                second_uuids = [second_uuids]
            recorded_uuids = list(second_uuids) + recorded_uuids
        self.claim_inputs(panel_str, recorded_uuids, action)

    def claim_inputs(
        self, panel_str: str, recorded_uuids: list[str], action: HistoryAction
    ) -> None:
        """Claim unassigned replay UUIDs for recorded input UUIDs."""
        unmapped = self.get_unmapped(panel_str, recorded_uuids)
        if not unmapped:
            return
        panel_order = list(action.state.object_metadata.get(panel_str, {}))
        if panel_order and all(uuid in panel_order for uuid in unmapped):
            unmapped.sort(key=panel_order.index)
        queue = self.unclaimed.get(panel_str, [])
        if not queue:
            return
        assigned = self.match_exact_uuids(panel_str, unmapped, queue)
        assigned.update(self.match_titles(panel_str, unmapped, queue, action))
        assigned.update(self.match_positions(panel_str, unmapped, queue, panel_order))
        self.unclaimed[panel_str] = [uuid for uuid in queue if uuid not in assigned]

    def get_unmapped(self, panel_str: str, recorded_uuids: list[str]) -> list[str]:
        """Return distinct recorded UUIDs that have no replay mapping."""
        panel_mapping = self.mapping.get(panel_str, {})
        unmapped: list[str] = []
        for uuid in recorded_uuids:
            if uuid not in unmapped and uuid not in panel_mapping:
                unmapped.append(uuid)
        return unmapped

    def match_exact_uuids(
        self, panel_str: str, recorded_uuids: list[str], queue: list[str]
    ) -> set[str]:
        """Match recorded inputs whose UUID is present unchanged in the queue."""
        assigned = set(recorded_uuids).intersection(queue)
        panel_mapping = self.mapping.setdefault(panel_str, {})
        panel_mapping.update({uuid: uuid for uuid in assigned})
        return assigned

    def match_titles(
        self,
        panel_str: str,
        recorded_uuids: list[str],
        queue: list[str],
        action: HistoryAction,
    ) -> set[str]:
        """Match remaining inputs with the unique replay object of the same title."""
        old_titles = self.get_recorded_titles(panel_str, recorded_uuids, action)
        new_titles = self.get_replay_titles(panel_str, queue)
        panel_mapping = self.mapping.setdefault(panel_str, {})
        assigned = set(panel_mapping.values()).intersection(queue)
        for old_uuid in recorded_uuids:
            if old_uuid in panel_mapping or old_uuid not in old_titles:
                continue
            candidates = [
                new_uuid
                for new_uuid in queue
                if new_uuid not in assigned
                and new_titles.get(new_uuid) == old_titles[old_uuid]
            ]
            if len(candidates) == 1:
                panel_mapping[old_uuid] = candidates[0]
                assigned.add(candidates[0])
        return assigned

    def get_recorded_titles(
        self, panel_str: str, recorded_uuids: list[str], action: HistoryAction
    ) -> dict[str, str]:
        """Return recorded object titles indexed by UUID."""
        selected = action.state.selection.get(panel_str, [])
        titles = action.state.titles.get(panel_str, [])
        old_titles = {
            uuid: title
            for uuid, title in zip(selected, titles)
            if uuid in recorded_uuids
        }
        metadata = action.state.object_metadata.get(panel_str, {})
        for uuid in recorded_uuids:
            signature = metadata.get(uuid)
            if uuid not in old_titles and isinstance(signature, dict):
                title = signature.get("title")
                if isinstance(title, str):
                    old_titles[uuid] = title
        return old_titles

    def get_replay_titles(
        self, panel_str: str, replay_uuids: list[str]
    ) -> dict[str, str]:
        """Return titles of queued replay objects that still exist."""
        panel = self.panels.get(panel_str)
        if panel is None:
            return {}
        titles: dict[str, str] = {}
        for uuid in replay_uuids:
            try:
                titles[uuid] = panel.objmodel[uuid].title
            except KeyError:
                continue
        return titles

    def match_positions(
        self,
        panel_str: str,
        recorded_uuids: list[str],
        queue: list[str],
        panel_order: list[str],
    ) -> set[str]:
        """Match remaining inputs by absolute or relative recorded position."""
        panel_mapping = self.mapping.setdefault(panel_str, {})
        old_remaining = [uuid for uuid in recorded_uuids if uuid not in panel_mapping]
        assigned = set(panel_mapping.values()).intersection(queue)
        new_remaining = [uuid for uuid in queue if uuid not in assigned]
        if not old_remaining:
            return assigned
        free_indices = [
            index for index, uuid in enumerate(panel_order) if uuid not in panel_mapping
        ]
        if panel_order and len(new_remaining) == len(free_indices):
            self.match_absolute_positions(
                panel_mapping, old_remaining, new_remaining, panel_order, free_indices
            )
        else:
            panel_mapping.update(dict(zip(old_remaining, new_remaining)))
        return set(panel_mapping.values()).intersection(queue)

    @staticmethod
    def match_absolute_positions(
        panel_mapping: dict[str, str],
        old_uuids: list[str],
        new_uuids: list[str],
        panel_order: list[str],
        free_indices: list[int],
    ) -> None:
        """Match remaining UUIDs using their absolute recorded panel positions."""
        new_by_index = dict(zip(free_indices, new_uuids))
        for old_uuid in old_uuids:
            if old_uuid in panel_order:
                index = panel_order.index(old_uuid)
                if index in new_by_index:
                    panel_mapping[old_uuid] = new_by_index[index]

    def capture_changes(
        self, action: HistoryAction, before: dict[str, set[str]]
    ) -> None:
        """Capture objects created and removed by a replayed action."""
        for panel_str, panel in self.panels.items():
            object_ids = panel.objmodel.get_object_ids()
            current = set(object_ids)
            self.discard_removed(panel_str, before[panel_str] - current)
            created = [uuid for uuid in object_ids if uuid not in before[panel_str]]
            self.capture_created(panel_str, created, action)

    def discard_removed(self, panel_str: str, removed_uuids: set[str]) -> None:
        """Discard queued objects and mappings whose replay objects were removed."""
        if not removed_uuids:
            return
        self.unclaimed[panel_str] = [
            uuid
            for uuid in self.unclaimed.get(panel_str, [])
            if uuid not in removed_uuids
        ]
        panel_mapping = self.mapping.get(panel_str, {})
        for old_uuid in [
            uuid
            for uuid, new_uuid in panel_mapping.items()
            if new_uuid in removed_uuids
        ]:
            panel_mapping.pop(old_uuid)

    def capture_created(
        self, panel_str: str, created_uuids: list[str], action: HistoryAction
    ) -> None:
        """Map or queue objects created by a replayed action."""
        if not created_uuids:
            return
        captured = action.state.selection.get(panel_str, [])
        if action.kind == HistoryAction.KIND_UI and captured:
            self.mapping.setdefault(panel_str, {}).update(
                dict(zip(captured, created_uuids))
            )
            created_uuids = created_uuids[len(captured) :]
        self.unclaimed.setdefault(panel_str, []).extend(created_uuids)
