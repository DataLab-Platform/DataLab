# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Collapsible description widget used by the History panel."""

from __future__ import annotations

import html

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import _


class CollapsibleDescriptionWidget(QW.QWidget):
    """Compact, expandable cell widget for the history Description column.

    Shows a single-line summary by default; a chevron toggle reveals the full
    HTML description (mirroring the *Properties* tab rendering).
    """

    toggled = QC.Signal(bool)

    def __init__(
        self,
        summary: str,
        html_text: str,
        expanded: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._html = html_text
        self._expanded = expanded

        self._toggle = QW.QToolButton(self)
        self._toggle.setAutoRaise(True)
        self._toggle.setCheckable(True)
        self._toggle.setFocusPolicy(QC.Qt.NoFocus)
        self._toggle.setArrowType(QC.Qt.RightArrow)
        self._toggle.setToolTip(_("Show details"))

        self._label = QW.QLabel(self)
        self._label.setTextFormat(QC.Qt.RichText)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self._label.setAlignment(QC.Qt.AlignTop | QC.Qt.AlignLeft)

        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle, 0, QC.Qt.AlignTop)
        layout.addWidget(self._label, 1)

        # Hide the toggle when there is nothing more to show than the summary.
        if not self._html or self.html_matches_summary():
            self._toggle.setVisible(False)

        self._toggle.toggled.connect(self.on_toggled)
        self.refresh_widget()

    def html_matches_summary(self) -> bool:
        """Return True when the HTML rendering would not add information."""
        return self._html.strip() == html.escape(self._summary).strip()

    def on_toggled(self, checked: bool) -> None:
        self._expanded = checked
        self.refresh_widget()
        self.toggled.emit(checked)

    def refresh_widget(self) -> None:
        if self._expanded:
            self._toggle.setArrowType(QC.Qt.DownArrow)
            self._toggle.setToolTip(_("Hide details"))
            self._label.setText(self._html or html.escape(self._summary))
        else:
            self._toggle.setArrowType(QC.Qt.RightArrow)
            self._toggle.setToolTip(_("Show details"))
            self._label.setText(html.escape(self._summary))
        self.updateGeometry()

    def is_expanded(self) -> bool:
        """Return current expanded state."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically set the expanded state."""
        if expanded == self._expanded:
            return
        self._toggle.setChecked(expanded)
