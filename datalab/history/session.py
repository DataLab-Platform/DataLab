# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""HistorySession: ordered list of HistoryAction with replay logic."""

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
        # Per-panel ``{old_uuid: new_uuid}`` mapping, populated as UI actions
        # create new objects. Used by compute actions to translate their
        # captured selection (and ``obj2_uuids``) into the freshly-created
        # UUIDs of the current replay, so chained ``n_to_1`` / ``2_to_1`` /
        # ``1_to_n`` actions operate on the correct inputs. Keys are
        # ``panel.PANEL_STR_ID`` (matches ``WorkspaceState.selection`` keys).
        panels = (mainwindow.signalpanel, mainwindow.imagepanel)
        uuid_remap: dict[str, dict[str, str]] = {p.PANEL_STR_ID: {} for p in panels}
        # FIFO of newly-created UUIDs not yet claimed by a remap entry --
        # required because most creation UI actions (e.g. ``new_signal``)
        # are recorded with ``save_state=False`` (empty captured selection),
        # so we cannot pair captured-vs-new UUIDs by position at UI time.
        # Subsequent compute actions claim from this queue on demand.
        unclaimed: dict[str, list[str]] = {p.PANEL_STR_ID: [] for p in panels}

        def _claim_unmapped(
            pstr: str,
            old_uuids: list[str],
            action: HistoryAction,
        ) -> None:
            """Claim unclaimed new UUIDs for *old_uuids* not yet in uuid_remap.

            Uses title matching (scanning the full unclaimed queue) followed by
            panel-order index alignment to deterministically pair old UUIDs
            to the correct new UUIDs, regardless of creation order.
            """
            # Collect unmapped UUIDs (deduplicated, preserving first-seen order).
            all_unmapped: list[str] = []
            seen: set[str] = set()
            for u in old_uuids:
                if u not in seen and u not in uuid_remap.get(pstr, {}):
                    all_unmapped.append(u)
                    seen.add(u)
            if not all_unmapped:
                return
            # Re-sort by recorded panel position when available.
            panel_order = list(action.state.object_metadata.get(pstr, {}).keys())
            if panel_order and all(u in panel_order for u in all_unmapped):
                all_unmapped.sort(key=panel_order.index)
            queue = unclaimed.get(pstr) or []
            if not queue:
                return
            # Build old UUID → title from captured state and object_metadata.
            sel_uuids = action.state.selection.get(pstr, [])
            sel_titles = action.state.titles.get(pstr, [])
            old_titles: dict[str, str] = {}
            for _u, _t in zip(sel_uuids, sel_titles):
                if _u in seen:
                    old_titles[_u] = _t
            obj_meta = action.state.object_metadata.get(pstr, {})
            for _u in all_unmapped:
                if _u not in old_titles and _u in obj_meta:
                    meta = obj_meta[_u]
                    if isinstance(meta, dict) and "title" in meta:
                        old_titles[_u] = meta["title"]
            # Build new UUID → title from the live panel (full queue).
            new_titles: dict[str, str] = {}
            panel_obj = None
            for p in panels:
                if p.PANEL_STR_ID == pstr:
                    panel_obj = p
                    break
            if panel_obj is not None:
                for nu in queue:
                    try:
                        new_titles[nu] = panel_obj.objmodel[nu].title
                    except KeyError:
                        pass
            # Phase 1: title matching against the FULL queue.
            assigned_old: set[str] = set()
            assigned_new: set[str] = set()
            for ou in all_unmapped:
                if ou not in old_titles:
                    continue
                title = old_titles[ou]
                candidates = [
                    nu
                    for nu in queue
                    if nu not in assigned_new and new_titles.get(nu) == title
                ]
                if len(candidates) == 1:
                    uuid_remap.setdefault(pstr, {})[ou] = candidates[0]
                    assigned_old.add(ou)
                    assigned_new.add(candidates[0])
            # Phase 2: positional fallback using panel-order alignment.
            # Two modes depending on whether the remaining queue covers all
            # free recorded panel slots:
            #
            # A) Absolute index alignment (len(rem_queue) == len(free_indices)):
            #    Each free panel_order index maps 1-to-1 to a queue slot.
            #    This ensures e.g. the second-created object maps to the
            #    second queue entry even when only a subset of old UUIDs
            #    needs claiming.
            #
            # B) Relative order fallback (queue is a strict subset):
            #    The queue only contains later compute-created objects while
            #    earlier full-panel entries are absent.  Absolute alignment
            #    would leave non-first old UUIDs unmapped.  Instead, zip
            #    rem_old (already sorted by panel order) with rem_queue
            #    sequentially.
            rem_old = [u for u in all_unmapped if u not in assigned_old]
            if rem_old and panel_order:
                rem_queue = [u for u in queue if u not in assigned_new]
                # Find which panel_order indices are "free" (unclaimed).
                free_indices: list[int] = []
                for idx, po_uuid in enumerate(panel_order):
                    if po_uuid not in uuid_remap.get(pstr, {}):
                        if po_uuid not in assigned_old:
                            free_indices.append(idx)
                if len(rem_queue) == len(free_indices):
                    # Mode A: absolute index alignment.
                    idx_to_new: dict[int, str] = {}
                    for qi, fi in enumerate(free_indices):
                        if qi < len(rem_queue):
                            idx_to_new[fi] = rem_queue[qi]
                    for ou in rem_old:
                        if ou in panel_order:
                            idx = panel_order.index(ou)
                            if idx in idx_to_new:
                                nu = idx_to_new[idx]
                                uuid_remap.setdefault(pstr, {})[ou] = nu
                                assigned_new.add(nu)
                else:
                    # Mode B: relative order fallback.
                    for ou, nu in zip(rem_old, rem_queue):
                        uuid_remap.setdefault(pstr, {})[ou] = nu
                        assigned_new.add(nu)
            elif rem_old:
                # No panel_order available: sequential fallback.
                rem_queue = [u for u in queue if u not in assigned_new]
                for ou, nu in zip(rem_old, rem_queue):
                    uuid_remap.setdefault(pstr, {})[ou] = nu
                    assigned_new.add(nu)
            # Remove all assigned new UUIDs from the unclaimed queue.
            if assigned_new:
                unclaimed[pstr] = [u for u in queue if u not in assigned_new]

        for action in self.actions[:]:
            before = {p.PANEL_STR_ID: set(p.objmodel.get_object_ids()) for p in panels}
            if action.kind == HistoryAction.KIND_COMPUTE:
                # Lazy-resolve any captured UUIDs missing from the remap by
                # claiming from ``unclaimed`` (deterministic: title + panel-order).
                pstr = action.panel_str or ""
                captured = action.state.selection.get(pstr, [])
                if action.pattern == "2_to_1":
                    # For 2_to_1: collect ALL unmapped old UUIDs from both
                    # captured selection and obj2_uuids in one batch so
                    # operand order is preserved by the helper.
                    obj2 = action.kwargs.get("obj2_uuids") or []
                    if isinstance(obj2, str):
                        obj2 = [obj2]
                    _claim_unmapped(pstr, list(obj2) + list(captured), action)
                else:
                    # For all other compute patterns (1_to_1, n_to_1, etc.):
                    # use the same deterministic helper.
                    _claim_unmapped(pstr, list(captured), action)
            action.replay(
                mainwindow,
                restore_selection=restore_selection,
                edit=edit,
                uuid_remap=uuid_remap,
            )
            # Post-action bookkeeping: track new/removed UUIDs for *every*
            # action kind so that later actions consuming compute-created
            # outputs can resolve them through ``uuid_remap`` / ``unclaimed``.
            for panel in panels:
                pstr = panel.PANEL_STR_ID
                current_ids = set(panel.objmodel.get_object_ids())
                new_uuids = [
                    u for u in panel.objmodel.get_object_ids() if u not in before[pstr]
                ]
                # Drop vanished UUIDs from the unclaimed queue and the
                # reverse remap entries (e.g. ``Remove selected objects``):
                # this keeps the FIFO claim in sync with the live panel
                # contents during chained creation/removal replays.
                removed_uuids = before[pstr] - current_ids
                if removed_uuids:
                    unclaimed[pstr] = [
                        u for u in unclaimed.get(pstr, []) if u not in removed_uuids
                    ]
                    panel_map = uuid_remap.get(pstr, {})
                    for old_key in [
                        k for k, v in panel_map.items() if v in removed_uuids
                    ]:
                        panel_map.pop(old_key, None)
                if not new_uuids:
                    continue
                if action.kind == HistoryAction.KIND_UI:
                    captured = action.state.selection.get(pstr, [])
                    if captured:
                        # Captured post-action selection available: pair
                        # captured UUIDs with new UUIDs by position.
                        for old_uuid, new_uuid in zip(captured, new_uuids):
                            uuid_remap.setdefault(pstr, {})[old_uuid] = new_uuid
                        # Any extra newly-created UUIDs go to the queue.
                        unclaimed.setdefault(pstr, []).extend(
                            new_uuids[len(captured) :]
                        )
                    else:
                        # No captured selection (typical of ``new_signal``):
                        # queue all new UUIDs for lazy claiming.
                        unclaimed.setdefault(pstr, []).extend(new_uuids)
                else:
                    # Compute actions: queue all newly-created UUIDs so
                    # later actions can lazily claim them.  Do NOT map
                    # captured input UUIDs to output UUIDs — compute
                    # inputs and outputs are semantically different.
                    unclaimed.setdefault(pstr, []).extend(new_uuids)

        # Visually close the replay: select the output of the last compute
        # action so the user sees the final result highlighted in the panel.
        # Without this, the very last action's output is never selected
        # (intermediate actions are implicitly "closed" by the next
        # iteration's input restore).
        if self.actions:
            last = self.actions[-1]
            if last.kind == HistoryAction.KIND_COMPUTE:
                hpanel = getattr(mainwindow, "historypanel", None)
                if hpanel is not None:
                    output_uuid = hpanel._action_output_uuid(last)
                    if output_uuid:
                        panel_str = last.panel_str or ""
                        panel_map = uuid_remap.get(panel_str, {})
                        mapped_uuid = panel_map.get(output_uuid, output_uuid)
                        for panel in panels:
                            if panel.PANEL_STR_ID == panel_str:
                                try:
                                    panel.objview.select_objects([mapped_uuid])
                                except KeyError:
                                    pass
                                break

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
