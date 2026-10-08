# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace state display widget used by the History panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import _

if TYPE_CHECKING:
    from datalab.history import WorkspaceState


class WorkspaceStateWidget(QW.QWidget):
    """Side-by-side tables showing the workspace state captured by a history action.

    Left table: signals (title + data shape).
    Right table: images (title + data shape/dimensions).
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self._signal_table = QW.QTableWidget(0, 2, self)
        self._signal_table.setHorizontalHeaderLabels([_("Signal"), _("Shape")])
        self._signal_table.horizontalHeader().setStretchLastSection(True)
        self._signal_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self._signal_table.setSelectionMode(QW.QAbstractItemView.NoSelection)
        self._signal_table.verticalHeader().hide()

        self._image_table = QW.QTableWidget(0, 2, self)
        self._image_table.setHorizontalHeaderLabels([_("Image"), _("Dimensions")])
        self._image_table.horizontalHeader().setStretchLastSection(True)
        self._image_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self._image_table.setSelectionMode(QW.QAbstractItemView.NoSelection)
        self._image_table.verticalHeader().hide()

        splitter = QW.QSplitter(QC.Qt.Horizontal, self)
        splitter.addWidget(self._signal_table)
        splitter.addWidget(self._image_table)
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def update_from_state(self, state: WorkspaceState | None) -> None:
        """Populate tables from a WorkspaceState."""
        self._signal_table.setRowCount(0)
        self._image_table.setRowCount(0)
        if state is None:
            return
        self.populate_table(self._signal_table, state, "signal")
        self.populate_table(self._image_table, state, "image")

    @staticmethod
    def populate_table(
        table: QW.QTableWidget, state: WorkspaceState, panel_key: str
    ) -> None:
        """Fill a table from the state for a given panel key."""
        titles = state.titles.get(panel_key, [])
        shapes = state.states.get(panel_key, [])
        metadata = state.object_metadata.get(panel_key, {})
        uuids = state.selection.get(panel_key, [])
        # Use metadata keyed by UUID when available
        rows: list[tuple[str, str]] = []
        for i, uuid in enumerate(uuids):
            title = titles[i] if i < len(titles) else uuid[:8]
            meta = metadata.get(uuid, {})
            shape = meta.get("shape")
            if shape is not None:
                shape_str = " × ".join(str(s) for s in shape)
            elif i < len(shapes):
                shape_str = shapes[i]
            else:
                shape_str = "—"
            rows.append((title, shape_str))
        table.setRowCount(len(rows))
        for row_idx, (title, shape_str) in enumerate(rows):
            table.setItem(row_idx, 0, QW.QTableWidgetItem(title))
            table.setItem(row_idx, 1, QW.QTableWidgetItem(shape_str))
