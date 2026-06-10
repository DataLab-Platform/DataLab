# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Expandable text widget
-------------------------------

Reusable widget that displays text with an expand/collapse toggle.
Clips text to a configurable number of lines when collapsed and provides
a scrollable expanded view.  Internally uses a read-only ``QTextBrowser``
which provides native word-wrap and scroll support, eliminating the need
for manual ``QTextLayout`` line counting.
"""

from __future__ import annotations

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _

# --- Default constants -----------------------------------------------------------

#: Default number of visible text lines in collapsed mode
DEFAULT_COLLAPSED_LINE_COUNT: int = 3

#: Ratio of the parent window height used as maximum expanded height
DEFAULT_EXPANDED_HEIGHT_RATIO: float = 0.35

#: Minimum expansion factor relative to collapsed height
DEFAULT_EXPANDED_MIN_FACTOR: int = 2

#: Default left indent in pixels
DEFAULT_INDENT: int = 20


# --- Palette helpers (public — reused by other modules) --------------------------


def apply_palette_color(widget: QW.QWidget, color: QG.QColor) -> None:
    """Apply a foreground color to *widget* via its palette (theme-safe).

    Args:
        widget: Target widget
        color: Foreground color to apply
    """
    palette = widget.palette()
    palette.setColor(QG.QPalette.WindowText, color)
    widget.setPalette(palette)


def apply_subdued_color(widget: QW.QWidget) -> None:
    """Apply a subdued/secondary text color on *widget*.

    Uses the ``QPalette.PlaceholderText`` role (Qt 5.12+) which provides a
    theme-native "dimmed" color.  Sets both ``WindowText`` (for ``QLabel``)
    and ``Text`` (for ``QTextBrowser`` / ``QTextEdit``) roles.

    Args:
        widget: Target widget
    """
    subdued = QW.QApplication.instance().palette().color(QG.QPalette.PlaceholderText)
    palette = widget.palette()
    palette.setColor(QG.QPalette.WindowText, subdued)
    palette.setColor(QG.QPalette.Text, subdued)
    widget.setPalette(palette)


# --- Internal helpers ------------------------------------------------------------


def _create_toggle_button(callback, text: str) -> QW.QPushButton:
    """Create a theme-aware toggle button.

    Args:
        callback: Slot connected to the button's ``clicked`` signal
        text: Initial button label

    Returns:
        Configured push button
    """
    button = QW.QPushButton(text)
    button.setFlat(True)
    button.setCursor(QC.Qt.PointingHandCursor)
    link_color = QW.QApplication.instance().palette().color(QG.QPalette.Link)
    button.setStyleSheet(
        f"QPushButton {{ color: {link_color.name()}; border: none; "
        "text-align: left; padding: 0; } "
        "QPushButton:hover { text-decoration: underline; }"
    )
    toggle_font = button.font()
    toggle_font.setPointSize(toggle_font.pointSize() - 1)
    button.setFont(toggle_font)
    button.clicked.connect(callback)
    return button


# --- Main widget -----------------------------------------------------------------


class ExpandableTextWidget(QW.QWidget):
    """Widget displaying text with an expand/collapse toggle.

    Clips to *collapsed_line_count* lines when collapsed and provides a
    scrollable expanded view whose height is governed by *expanded_height_ratio*
    and *expanded_min_factor*.

    Internally backed by a read-only ``QTextBrowser`` that natively handles
    word-wrap and scrolling, exposed as both ``label`` and ``scroll_area``
    for backward compatibility.

    Args:
        description: Text content to display
        parent: Parent widget
        collapsed_line_count: Visible lines when collapsed
        expanded_height_ratio: Ratio of parent window height for expanded max
        expanded_min_factor: Minimum multiplier of collapsed height for expansion
        indent: Left margin indentation in pixels
        subdued: Whether to apply a subdued (semi-transparent) text color
        text_interaction_flags: Qt text interaction flags
        label_font: Custom font
        show_more_text: Custom "show more" label (default: translated)
        show_less_text: Custom "show less" label (default: translated)
    """

    toggled = QC.Signal(bool)

    def __init__(
        self,
        description: str,
        parent: QW.QWidget = None,
        *,
        collapsed_line_count: int = DEFAULT_COLLAPSED_LINE_COUNT,
        expanded_height_ratio: float = DEFAULT_EXPANDED_HEIGHT_RATIO,
        expanded_min_factor: int = DEFAULT_EXPANDED_MIN_FACTOR,
        indent: int = DEFAULT_INDENT,
        subdued: bool = True,
        text_interaction_flags: QC.Qt.TextInteractionFlags = QC.Qt.NoTextInteraction,
        label_font: QG.QFont | None = None,
        show_more_text: str | None = None,
        show_less_text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._description = description
        self._expanded = False

        # Store configuration
        self.collapsed_line_count = collapsed_line_count
        self.expanded_height_ratio = expanded_height_ratio
        self.expanded_min_factor = expanded_min_factor
        self._show_more_text = (
            show_more_text if show_more_text is not None else _("Show more")
        )
        self._show_less_text = (
            show_less_text if show_less_text is not None else _("Show less")
        )

        # Layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(indent, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # QTextBrowser — native word-wrap, scroll, and height measurement.
        # readOnly is True by default for QTextBrowser.
        browser = QW.QTextBrowser()
        browser.setFrameShape(QW.QFrame.NoFrame)
        browser.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        browser.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        browser.setOpenLinks(False)
        browser.setStyleSheet("QTextBrowser { background: transparent; }")
        browser.setSizePolicy(QW.QSizePolicy.Expanding, QW.QSizePolicy.Fixed)
        browser.setTextInteractionFlags(text_interaction_flags)
        if label_font is not None:
            browser.setFont(label_font)
        if subdued:
            apply_subdued_color(browser)

        browser.setPlainText(description)
        layout.addWidget(browser)

        # Public attributes — ``label`` and ``scroll_area`` both point to the
        # QTextBrowser for backward compatibility (it IS a QAbstractScrollArea).
        self.label = browser
        self.scroll_area = browser

        # Toggle button
        self.toggle_button = _create_toggle_button(
            self._toggle_description,
            "\u25bc " + self._show_more_text,
        )
        layout.addWidget(self.toggle_button)

        QC.QTimer.singleShot(0, self.refresh_description)

    # --- Qt event overrides ------------------------------------------------------

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Refresh truncation when the available width changes."""
        super().resizeEvent(event)
        self.refresh_description()

    def showEvent(self, event: QG.QShowEvent) -> None:  # pylint: disable=invalid-name
        """Refresh after first layout pass to use the actual widget width."""
        super().showEvent(event)
        QC.QTimer.singleShot(0, self.refresh_description)

    # --- Internal helpers --------------------------------------------------------

    def _get_text_width(self) -> int:
        """Return the effective text width inside the browser viewport."""
        margins = self.layout().contentsMargins()
        margin_width = margins.left() + margins.right()

        # When the widget has an explicit fixed width, derive the text
        # width from it.  The viewport may not yet reflect this constraint
        # before the first layout pass (e.g. PyQt6 offscreen platform).
        min_w = self.minimumWidth()
        max_w = self.maximumWidth()
        if 0 < min_w == max_w:
            return max(min_w - margin_width, 0)

        width = self.label.viewport().width()
        if width <= 0:
            widget_width = self.width()
            if widget_width <= 0:
                widget_width = min_w
            width = widget_width - margin_width
        return max(width, 0)

    def _get_collapsed_height(self) -> int:
        """Return the pixel height that fits *collapsed_line_count* lines."""
        fm = self.label.fontMetrics()
        doc_margin = int(self.label.document().documentMargin())
        return fm.lineSpacing() * self.collapsed_line_count + 2 * doc_margin

    def _content_height_for_width(self, width: int) -> int:
        """Return the natural content height for a given *width*.

        Temporarily sets the document text-width and reads back the computed
        size, then restores the previous width to avoid side-effects.
        """
        doc = self.label.document()
        old_width = doc.textWidth()
        doc.setTextWidth(width)
        height = int(doc.size().height())
        doc.setTextWidth(old_width)
        return height

    def _get_expanded_max_height(self, collapsed_height: int) -> int:
        """Return the maximum height allocated to expanded descriptions."""
        minimum_height = max(
            collapsed_height * self.expanded_min_factor,
            collapsed_height,
        )
        window = self.window()
        if window is None or window.height() <= 0:
            return minimum_height
        proportional_height = int(window.height() * self.expanded_height_ratio)
        return max(minimum_height, proportional_height)

    def _toggle_description(self) -> None:
        """Toggle between collapsed and expanded states."""
        self._expanded = not self._expanded
        self.refresh_description()
        self.toggled.emit(self._expanded)

    # --- Public API --------------------------------------------------------------

    def refresh_description(self) -> None:
        """Update toggle visibility, height constraint, and scroll policy."""
        width = self._get_text_width()
        content_height = self._content_height_for_width(width)
        collapsed_height = self._get_collapsed_height()
        needs_toggle = content_height > collapsed_height

        self.toggle_button.setVisible(needs_toggle)

        if not needs_toggle:
            self._expanded = False
            self.label.setFixedHeight(content_height)
            self.label.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            self.updateGeometry()
            return

        if self._expanded:
            max_height = self._get_expanded_max_height(collapsed_height)
            self.label.setFixedHeight(min(content_height, max_height))
            if content_height > max_height:
                self.label.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAsNeeded)
            else:
                self.label.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            self.label.verticalScrollBar().setValue(0)
            self.toggle_button.setText("\u25b2 " + self._show_less_text)
        else:
            self.label.setFixedHeight(collapsed_height)
            self.label.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            self.label.verticalScrollBar().setValue(0)
            self.toggle_button.setText("\u25bc " + self._show_more_text)

        self.updateGeometry()

    def needs_toggle_for_width(self, width: int) -> bool:
        """Return whether the text needs an expand/collapse toggle at *width*.

        Args:
            width: Available text width in pixels
        """
        return self._content_height_for_width(width) > self._get_collapsed_height()

    def is_expanded(self) -> bool:
        """Return the current expanded state."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Set the expanded state and refresh the widget."""
        self._expanded = expanded
        self.refresh_description()
