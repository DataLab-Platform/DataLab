# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""GUI-boundary orchestration for reconnecting removed history objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import recompute as hrec
from datalab.history import HistoryAction

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history.panel import HistoryPanel


def reconnect_chain_after_removal(
    panel: HistoryPanel, panel_data: BaseDataPanel
) -> None:
    """Reconnect the processing chain after objects are deleted from a data panel."""
    panel_str = panel_data.PANEL_STR_ID
    previous = panel.runtime.objects.obj_ids_snapshot.get(panel_str, set())
    current = set(panel_data.objmodel.get_object_ids())
    removed = previous - current
    if not removed:
        return
    with panel.runtime.objects.reconnecting_objects() as started:
        if not started:
            return
        plans = [
            hchain.plan_reconnection(panel, panel_data, object_uuid)
            for object_uuid in removed
        ]
        roots_to_recompute: list[HistoryAction] = []
        for plan in plans:
            hchain.apply_reconnection_plan(panel, panel_data, plan, roots_to_recompute)
        for action in roots_to_recompute:
            success = hrec.recompute_action_in_place(panel, action)
            action.is_stale = not success
            panel.tree.refresh_action_item(action)
            if success:
                hrec.recompute_cascade(panel, action)
        hchain.show_reconnection_warnings(
            panel, [plan.warning for plan in plans if plan.warning is not None]
        )
        hchain.refresh_reconnected_history(panel)
