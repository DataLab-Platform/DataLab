# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Collapsible description widget used by the History panel."""

from __future__ import annotations

import html

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _

#: Maximum number of text lines shown at once by an expanded description
#: (longer descriptions are scrolled inside the cell)
DETAILS_MAX_LINES = 12


class ElidedLabel(QW.QLabel):
    """Single-line plain-text label eliding its text to the available width.

    The full text is shown as tooltip when it does not fit.
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self.__full_text = ""
        self.setTextFormat(QC.Qt.PlainText)
        self.setWordWrap(False)
        self.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignVCenter)
        # The text never imposes its width: it is elided instead
        self.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Preferred)

    def full_text(self) -> str:
        """Return the full (non-elided) text."""
        return self.__full_text

    def set_full_text(self, text: str) -> None:
        """Set the full text, displayed elided to the label width.

        Args:
            text: Single-line plain text
        """
        self.__full_text = text
        self.update_elided_text()

    def update_elided_text(self) -> None:
        """Elide the full text to the current label width."""
        width = max(self.contentsRect().width(), 0)
        elided = self.fontMetrics().elidedText(
            self.__full_text, QC.Qt.ElideRight, width
        )
        self.setText(elided)
        self.setToolTip(self.__full_text if elided != self.__full_text else "")

    def minimumSizeHint(self) -> QC.QSize:  # pylint: disable=invalid-name
        """Return the minimum size hint: the text may be elided to any width."""
        return QC.QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QC.QSize:  # pylint: disable=invalid-name
        """Return the size hint: one line, whatever the text length."""
        return QC.QSize(0, super().sizeHint().height())

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Elide the text again to the new width."""
        super().resizeEvent(event)
        self.update_elided_text()


class CollapsibleDescriptionWidget(QW.QWidget):
    """Compact, expandable cell widget for the history Description column.

    Shows a single-line summary by default (elided to the cell width, the full
    summary being available as tooltip); a chevron toggle reveals the full HTML
    description (mirroring the *Properties* tab rendering), scrolled inside the
    cell beyond :data:`DETAILS_MAX_LINES` lines so that the row height stays
    bounded.
    """

    toggled = QC.Signal(bool)
    #: Emitted when the preferred height changes without toggling (expanded
    #: details wrapped again after a width change): the tree must lay out its
    #: rows again, since it does not track the size hint of its cell widgets.
    size_hint_changed = QC.Signal()

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
        self.__last_height: int | None = None

        self._toggle = QW.QToolButton(self)
        self._toggle.setAutoRaise(True)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setFocusPolicy(QC.Qt.NoFocus)
        self._toggle.setArrowType(QC.Qt.RightArrow)
        self._toggle.setToolTip(_("Show details"))

        self._summary_label = ElidedLabel(self)
        self._summary_label.set_full_text(summary)

        self._details_label = QW.QLabel()
        self._details_label.setTextFormat(QC.Qt.RichText)
        self._details_label.setWordWrap(True)
        self._details_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self._details_label.setAlignment(QC.Qt.AlignTop | QC.Qt.AlignLeft)

        self._details = QW.QScrollArea(self)
        self._details.setFrameShape(QW.QFrame.NoFrame)
        self._details.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        self._details.setWidgetResizable(True)
        self._details.setWidget(self._details_label)
        # Keep the row background (selection, stale tint) visible behind the text
        self._details.viewport().setAutoFillBackground(False)
        self._details_label.setAutoFillBackground(False)

        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle, 0, QC.Qt.AlignTop)
        layout.addWidget(self._summary_label, 1)
        layout.addWidget(self._details, 1)

        # Hide the toggle when there is nothing more to show than the summary.
        if not self._html or self.html_matches_summary():
            self._toggle.setVisible(False)

        self._toggle.toggled.connect(self.on_toggled)
        self.refresh_widget()

    def html_matches_summary(self) -> bool:
        """Return True when the HTML rendering would not add information."""
        return self._html.strip() == html.escape(self._summary).strip()

    def on_toggled(self, checked: bool) -> None:
        """Handle the expand/collapse toggle being toggled."""
        self._expanded = checked
        self.refresh_widget()
        self.toggled.emit(checked)

    def refresh_widget(self) -> None:
        """Refresh the widget contents to match the current expanded state."""
        if self._expanded:
            self._toggle.setArrowType(QC.Qt.DownArrow)
            self._toggle.setToolTip(_("Hide details"))
            self._details_label.setText(self._html or html.escape(self._summary))
        else:
            self._toggle.setArrowType(QC.Qt.RightArrow)
            self._toggle.setToolTip(_("Show details"))
        self._summary_label.setVisible(not self._expanded)
        self._details.setVisible(self._expanded)
        self.__last_height = self.preferred_height()
        self.updateGeometry()

    def details_max_height(self) -> int:
        """Return the maximum height of the expanded details."""
        line_spacing = self._details_label.fontMetrics().lineSpacing()
        return DETAILS_MAX_LINES * line_spacing + 2 * self._details.frameWidth()

    def __details_width(self) -> int:
        """Return the width available to the expanded details."""
        layout = self.layout()
        margins = layout.contentsMargins()
        width = self.width() - margins.left() - margins.right()
        if not self._toggle.isHidden():
            width -= self._toggle.sizeHint().width() + layout.spacing()
        return max(width - 2 * self._details.frameWidth(), 1)

    def preferred_height(self) -> int:
        """Return the preferred height of the widget in its current state.

        Collapsed, the widget is one line high. Expanded, it fits the details
        up to :meth:`details_max_height`.
        """
        toggle_height = 0
        if not self._toggle.isHidden():
            toggle_height = self._toggle.sizeHint().height()
        if not self._expanded:
            return max(toggle_height, self._summary_label.sizeHint().height())
        height = self._details_label.heightForWidth(self.__details_width())
        if height < 0:
            height = self._details_label.sizeHint().height()
        height += 2 * self._details.frameWidth()
        return max(toggle_height, min(height, self.details_max_height()))

    def sizeHint(self) -> QC.QSize:  # pylint: disable=invalid-name
        """Return the size hint, whose height bounds the tree row height."""
        return QC.QSize(super().sizeHint().width(), self.preferred_height())

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Notify a change of the preferred height (details wrapped again)."""
        super().resizeEvent(event)
        height = self.preferred_height()
        if height != self.__last_height:
            self.__last_height = height
            self.size_hint_changed.emit()

    def is_expanded(self) -> bool:
        """Return current expanded state."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically set the expanded state."""
        if expanded == self._expanded:
            return
        self._toggle.setChecked(expanded)
