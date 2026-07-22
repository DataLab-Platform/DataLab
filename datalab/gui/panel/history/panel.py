# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. History panel (see parent package :mod:`datalab.gui.panel`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

from guidata.configtools import get_icon
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui import historysession_ops as hsess
from datalab.gui.panel.base import AbstractPanel
from datalab.gui.panel.history.facade import (
    HistoryPersistenceFacadeMixin,
    HistoryRecordingFacadeMixin,
    HistoryReplayFacadeMixin,
    HistoryRuntimeFacadeMixin,
)
from datalab.gui.panel.history.navigation import HistoryNavigation
from datalab.gui.panel.history.runtime import HistoryRuntime
from datalab.gui.panel.history.ui import HistoryPanelUI
from datalab.history import HistoryAction, HistorySession
from datalab.widgets.historytree import HistoryTree

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


class HistoryPanel(
    HistoryRuntimeFacadeMixin,
    HistoryReplayFacadeMixin,
    HistoryRecordingFacadeMixin,
    HistoryPersistenceFacadeMixin,
    AbstractPanel,
    DockableWidgetMixin,
):
    """History panel"""

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("History panel")

    H5_PREFIX = "DataLab_His"

    SIG_OBJECT_MODIFIED = QC.Signal()

    FILE_FILTERS = f"{_('History files')} (*.dlhist)"

    def __init__(self, parent: DLMainWindow) -> None:
        super().__init__(parent)
        self.mainwindow = parent
        self.setWindowTitle(self.PANEL_STR)
        self.setWindowIcon(get_icon("history.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.history_sessions: list[HistorySession] = []
        self.tree = HistoryTree(self)
        self.runtime = HistoryRuntime(self, self.reconnect_chain_after_removal)
        self.navigation = HistoryNavigation(self)
        self.ui = HistoryPanelUI(self)
        self.set_tracking_enabled(True)
        self.runtime.objects.refresh_obj_ids_snapshot()
        self.ui.update_actions_state()
        self.refresh_compatibility_items()
        if not execenv.unattended and Conf.proc.history_auto_record.get(False):
            self.ui.actions["record"].setChecked(True)
            self.create_new_session()

    def __len__(self) -> int:
        """Return number of objects."""
        return sum(len(session.actions) for session in self.history_sessions)

    def __getitem__(self, nb: int) -> HistoryAction:
        """Return object from its number (1 to N)."""
        for session in self.history_sessions:
            if nb <= len(session.actions):
                return session.actions[nb - 1]
            nb -= len(session.actions)
        raise IndexError("Index out of range")

    def __iter__(self) -> Generator[HistoryAction, None, None]:
        """Iterate over objects."""
        for session in self.history_sessions:
            yield from session.actions

    # ------ AbstractPanel interface ---------------------------------------------------
    def create_object(self) -> HistoryAction:
        """Create and return object."""
        return HistoryAction()

    def add_object(self, obj: HistoryAction) -> None:
        """Add an object to the history."""
        return hsess.add_object(self, obj)

    def remove_all_objects(self) -> None:
        """Remove all objects."""
        super().remove_all_objects()
        self.runtime.objects.clear_output_mappings()
