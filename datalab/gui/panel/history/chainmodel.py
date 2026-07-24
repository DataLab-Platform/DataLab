# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Derived processing-chain read-model for the History panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from datalab.gui.processor.base import ProcessingParameters
from datalab.history import HistoryAction, HistorySession

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel


@dataclass
class ProcessingChain:
    """A processing chain: a creation/external root action and its descendants.

    Attributes:
        root: The action that starts the chain (creation action or external
            root compute/UI action).
        actions: Ordered actions belonging to the chain, ``root`` first,
            followed by descendants in session order.
        session: The :class:`HistorySession` that contains the chain.
    """

    root: HistoryAction
    session: HistorySession
    actions: list[HistoryAction] = field(default_factory=list)


@dataclass
class ChainSelectionPlan:
    """Processing chains selected from one source session."""

    source_session: HistorySession
    chains: list[ProcessingChain]


@dataclass
class UuidCloneRegistry:
    """Objects cloned for duplication and their UUID remapping by panel."""

    uuid_remap: dict[str, dict[str, str]] = field(default_factory=dict)
    clones_by_panel: dict[str, list[Any]] = field(default_factory=dict)

    def register(
        self,
        panel_str: str,
        old_uuid: str,
        new_uuid: str,
        clone: Any,
    ) -> None:
        """Register a cloned object and its source-to-clone UUID mapping."""
        self.uuid_remap.setdefault(panel_str, {})[old_uuid] = new_uuid
        self.clones_by_panel.setdefault(panel_str, []).append(clone)

    def resolve(self, panel_str: str, old_uuid: str) -> str | None:
        """Return the cloned UUID corresponding to a source UUID."""
        return self.uuid_remap.get(panel_str, {}).get(old_uuid)


@dataclass
class DuplicatedSession:
    """A duplicated session paired with the source that determines insertion."""

    source_session: HistorySession
    new_session: HistorySession


@dataclass
class DeletionPlan:
    """Selected history entities grouped by their deletion behavior."""

    actions: list[HistoryAction] = field(default_factory=list)
    session_ids: set[int] = field(default_factory=set)
    affected_session: HistorySession | None = None


@dataclass
class DeletionResult:
    """State needed for orphan cleanup and post-deletion selection."""

    affected_session: HistorySession | None
    removed_session_ids: set[int]
    orphan_refs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ReconnectionTarget:
    """A surviving object and history action consuming a removed UUID."""

    object_uuid: str
    parameters: ProcessingParameters
    action: HistoryAction | None


@dataclass
class ReconnectionPlan:
    """Planned source rewrites after one data object has been removed."""

    panel_str: str
    removed_uuid: str
    source_uuid: str | None
    producer_action: HistoryAction | None
    targets: list[ReconnectionTarget] = field(default_factory=list)
    warning: str | None = None
    remove_producer: bool = False


def action_input_uuids(action: HistoryAction) -> set[str]:
    """Return the set of input object UUIDs captured by ``action``.

    Combines the recorded selection for the action's panel with any
    ``obj2_uuids`` second-operand references (2-to-1 pattern).

    Args:
        action: The history action whose inputs are extracted.

    Returns:
        The set of object UUIDs that the action consumed as inputs.
    """
    captured: set[str] = set(action.state.selection.get(action.panel_str or "", []))
    obj2 = action.kwargs.get("obj2_uuids")
    if obj2:
        if isinstance(obj2, str):
            captured.add(obj2)
        else:
            captured.update(obj2)
    return captured


def remap_processing_parameters(
    parameters: ProcessingParameters,
    uuid_remap: dict[str, str],
    clear_sources: bool = False,
) -> ProcessingParameters:
    """Rebuild processing parameters with remapped source UUIDs."""
    source_uuid = None if clear_sources else parameters.source_uuid
    if source_uuid is not None:
        source_uuid = uuid_remap.get(source_uuid, source_uuid)
    source_uuids = None if clear_sources else parameters.source_uuids
    if source_uuids is not None:
        source_uuids = [uuid_remap.get(uuid, uuid) for uuid in source_uuids]
    return ProcessingParameters(
        func_name=parameters.func_name,
        pattern=parameters.pattern,
        param=parameters.param,
        source_uuid=source_uuid,
        source_uuids=source_uuids,
        plugin_origin=parameters.plugin_origin,
    )


def build_session_chains(session: HistorySession) -> list[ProcessingChain]:
    """Return the session's single processing chain.

    In DataLab's history model a session **is** a single linear processing
    chain: every action of the session belongs to one chain, the first action
    being its root. Session boundaries are decided at recording time (the
    "start a new history session?" prompt shown on object creation), so no
    per-creation splitting is performed here.

    Args:
        session: The session whose actions form the chain.

    Returns:
        A single-element list with the session's chain, or an empty list when
        the session has no actions.
    """
    if not session.actions:
        return []
    chain = ProcessingChain(root=session.actions[0], session=session)
    chain.actions = list(session.actions)
    return [chain]


def build_processing_chains(
    panel: HistoryPanel,
) -> list[tuple[HistorySession, list[ProcessingChain]]]:
    """Return, for each session (in order), its ordered list of processing chains.

    Args:
        panel: The history panel owning sessions and output registry.

    Returns:
        A list of (session, chains) tuples in session order.
    """
    return [
        (session, build_session_chains(session)) for session in panel.history_sessions
    ]
