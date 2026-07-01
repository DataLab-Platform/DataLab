# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Derived processing-chain read-model for the History panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


def build_session_chains(
    panel: HistoryPanel, session: HistorySession
) -> list[ProcessingChain]:
    """Group the actions of a single session into ordered processing chains.

    Args:
        panel: The history panel providing the action→output registry.
        session: The session whose actions are grouped.

    Returns:
        The session's processing chains, in creation order. Each chain lists
        its actions in session order, the root first.
    """
    uuid_to_chain: dict[str, ProcessingChain] = {}
    chains: list[ProcessingChain] = []
    for action in session.actions:
        is_creation = (
            action.kind == HistoryAction.KIND_UI
            and action.method_name in HistoryAction.UI_CREATION_METHODS
        )
        if is_creation:
            chain = ProcessingChain(root=action, session=session)
            chains.append(chain)
            chain.actions.append(action)
        else:
            inputs = action_input_uuids(action)
            chain = None
            for uuid in sorted(inputs):
                if uuid in uuid_to_chain:
                    chain = uuid_to_chain[uuid]
                    break
            if chain is None:
                chain = ProcessingChain(root=action, session=session)
                chains.append(chain)
            chain.actions.append(action)
        for out in panel.action_output_uuids.get(action.uuid, []):
            uuid_to_chain[out] = chain
    return chains


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
        (session, build_session_chains(panel, session))
        for session in panel.history_sessions
    ]
