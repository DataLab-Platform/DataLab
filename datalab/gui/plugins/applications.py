# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Applications panel: application plugins and their declared workflows."""

from __future__ import annotations

import html
import webbrowser
from collections.abc import Callable, Iterable
from math import ceil
from typing import TYPE_CHECKING

from guidata.configtools import get_icon, get_image_file_path
from guidata.qthelpers import win32_fix_title_bar_background
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.gui.plugins.recipe_inputs import (
    format_diagnostic,
    format_readiness,
    format_slot_kind,
)
from datalab.plugins import PluginCapability, PluginRegistry
from datalab.plugins.recipe_binding import RecipeReadiness, RecipeReadinessStatus
from datalab.plugins.recipes import RecipeOutcome
from datalab.plugins.resources import resolve_package_resource
from datalab.utils.qthelpers import qt_handle_error_message, try_or_log_error
from datalab.widgets.expandabletext import apply_subdued_color

if TYPE_CHECKING:
    from datalab.plugins import PluginBase
    from datalab.plugins.examples import PluginExample
    from datalab.plugins.recipes import RecipeDescriptor


__all__ = [
    "READINESS_COLORS",
    "ApplicationPage",
    "ApplicationsPanel",
    "CollapsibleSection",
    "FlashOverlay",
    "RecipeDetails",
    "RecipeRow",
    "get_application_plugins",
    "get_declared_metadata_keys",
    "get_plugin_icon",
    "record_application_use",
    "set_application_hidden",
    "set_application_pinned",
    "sort_welcome_applications",
]


CATALOG_TITLE_ROLE = QC.Qt.UserRole + 1
CATALOG_DESCRIPTION_ROLE = QC.Qt.UserRole + 2
CATALOG_VERSION_ROLE = QC.Qt.UserRole + 3

#: Number of recently used applications remembered to order the welcome page
MAX_RECENT_APPLICATIONS = 20


def get_plugin_icon(icon: str | None) -> QG.QIcon:
    """Return a plugin icon

    Args:
        icon: ``package:path`` resource (SVG or bitmap), ``module:filename`` image
         registered by a module with guidata (e.g. ``"datalab:delete.svg"``),
         DataLab icon file name, or None

    Returns:
        Plugin icon, or the generic plugin icon if ``icon`` is None or cannot be
         loaded
    """
    if icon is not None:
        with try_or_log_error(f"Loading plugin icon {icon!r}"):
            if ":" not in icon:
                return get_icon(icon)
            try:
                resource = resolve_package_resource(icon, "Plugin icon")
            except FileNotFoundError:
                # Image registered by the module with guidata ("module:filename")
                get_image_file_path(icon, default=None)
                return get_icon(icon)
            buffer = QC.QBuffer()
            buffer.setData(resource.read_bytes())
            reader = QG.QImageReader(buffer)
            if icon.lower().endswith(".svg"):
                # Render vector icons at twice their size for high-DPI screens
                reader.setScaledSize(reader.size() * 2)
            image = reader.read()
            if image.isNull():
                raise ValueError(
                    f"Invalid plugin icon {icon!r}: {reader.errorString()}"
                )
            return QG.QIcon(QG.QPixmap.fromImage(image))
    return get_icon("libre-gui-plugin.svg")


class CatalogItemDelegate(QW.QStyledItemDelegate):
    """Render a catalog entry with an optional icon, a title and a description."""

    HORIZONTAL_MARGIN = 8
    VERTICAL_MARGIN = 6
    DESCRIPTION_SPACING = 2
    ICON_SIZE = 32
    ICON_SPACING = 8

    @staticmethod
    def _blend(foreground: QG.QColor, background: QG.QColor) -> QG.QColor:
        """Return a subdued color that remains legible on its background."""
        return QG.QColor(
            (3 * foreground.red() + background.red()) // 4,
            (3 * foreground.green() + background.green()) // 4,
            (3 * foreground.blue() + background.blue()) // 4,
        )

    @staticmethod
    def _midpoint(first: QG.QColor, second: QG.QColor) -> QG.QColor:
        """Return the midpoint between two colors."""
        return QG.QColor(
            (first.red() + second.red()) // 2,
            (first.green() + second.green()) // 2,
            (first.blue() + second.blue()) // 2,
        )

    @staticmethod
    def _is_light_theme(palette: QG.QPalette) -> bool:
        """Return whether a palette uses a light base color."""
        return palette.color(QG.QPalette.Base).lightness() > 128

    @classmethod
    def _foreground_colors(
        cls, option: QW.QStyleOptionViewItem
    ) -> tuple[QG.QColor, QG.QColor]:
        """Return title and secondary colors for the current item state."""
        palette = option.palette
        selected = bool(option.state & QW.QStyle.State_Selected)
        if not selected:
            return (
                palette.color(QG.QPalette.Text),
                palette.color(QG.QPalette.Disabled, QG.QPalette.Text),
            )

        text_color = palette.color(QG.QPalette.HighlightedText)
        background_color = palette.color(QG.QPalette.Highlight)
        if not cls._is_light_theme(palette):
            if not option.state & QW.QStyle.State_HasFocus:
                return (
                    palette.color(QG.QPalette.Text),
                    palette.color(QG.QPalette.Disabled, QG.QPalette.Text),
                )
            return text_color, cls._blend(text_color, background_color)
        if option.state & QW.QStyle.State_HasFocus:
            text_color = palette.color(QG.QPalette.Text)
            return text_color, text_color
        return (
            palette.color(QG.QPalette.Text),
            palette.color(QG.QPalette.Disabled, QG.QPalette.Text),
        )

    def _document(
        self,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
        width: int,
    ) -> QG.QTextDocument:
        """Build the wrapped text document for one catalog entry."""
        text_color, subdued_color = self._foreground_colors(option)

        document = QG.QTextDocument()
        document.setDefaultFont(option.font)
        document.setDocumentMargin(0)
        text_option = document.defaultTextOption()
        text_option.setWrapMode(QG.QTextOption.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(text_option)

        cursor = QG.QTextCursor(document)
        title_format = QG.QTextCharFormat()
        title_format.setFontWeight(QG.QFont.Bold)
        title_format.setForeground(text_color)
        title = index.data(CATALOG_TITLE_ROLE) or index.data(QC.Qt.DisplayRole) or ""
        cursor.insertText(title, title_format)

        secondary_format = QG.QTextCharFormat()
        secondary_format.setForeground(subdued_color)
        point_size = option.font.pointSizeF()
        if point_size > 1:
            secondary_format.setFontPointSize(point_size - 1)
        version = index.data(CATALOG_VERSION_ROLE) or ""
        if version:
            cursor.insertText(f"  v{version}", secondary_format)

        description = index.data(CATALOG_DESCRIPTION_ROLE) or ""
        if description:
            cursor.insertBlock()
            block_format = cursor.blockFormat()
            block_format.setTopMargin(self.DESCRIPTION_SPACING)
            cursor.setBlockFormat(block_format)
            cursor.insertText(description, secondary_format)

        document.setTextWidth(max(1, width))
        return document

    @classmethod
    def _content_width(cls, option: QW.QStyleOptionViewItem) -> int:
        """Return the available text width for a view item."""
        width = option.rect.width()
        if width <= 0 and isinstance(option.widget, QW.QAbstractItemView):
            width = option.widget.viewport().width()
        return max(1, width - 2 * cls.HORIZONTAL_MARGIN)

    @classmethod
    def _icon_offset(cls, index: QC.QModelIndex) -> int:
        """Return the horizontal space taken by the item icon, if any."""
        icon = index.data(QC.Qt.DecorationRole)
        if isinstance(icon, QG.QIcon) and not icon.isNull():
            return cls.ICON_SIZE + cls.ICON_SPACING
        return 0

    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Paint the native item background, the icon and the formatted text."""
        styled_option = QW.QStyleOptionViewItem(option)
        self.initStyleOption(styled_option, index)
        styled_option.text = ""
        # The icon is painted next to the formatted text, not by the style
        styled_option.icon = QG.QIcon()
        selected_without_focus = (
            styled_option.state & QW.QStyle.State_Selected
            and not styled_option.state & QW.QStyle.State_HasFocus
        )
        if selected_without_focus and not self._is_light_theme(styled_option.palette):
            inactive_highlight = self._midpoint(
                styled_option.palette.color(QG.QPalette.Highlight),
                styled_option.palette.color(QG.QPalette.Base),
            )
            styled_option.palette.setColor(QG.QPalette.Highlight, inactive_highlight)
        style = (
            styled_option.widget.style()
            if styled_option.widget is not None
            else QW.QApplication.style()
        )
        style.drawControl(
            QW.QStyle.CE_ItemViewItem,
            styled_option,
            painter,
            styled_option.widget,
        )

        text_rect = option.rect.adjusted(
            self.HORIZONTAL_MARGIN,
            self.VERTICAL_MARGIN,
            -self.HORIZONTAL_MARGIN,
            -self.VERTICAL_MARGIN,
        )
        icon_offset = self._icon_offset(index)
        if icon_offset:
            icon_rect = QC.QRect(
                text_rect.topLeft(), QC.QSize(self.ICON_SIZE, self.ICON_SIZE)
            )
            index.data(QC.Qt.DecorationRole).paint(painter, icon_rect)
            text_rect.setLeft(text_rect.left() + icon_offset)
        document = self._document(option, index, text_rect.width())
        painter.save()
        painter.translate(text_rect.topLeft())
        context = QG.QAbstractTextDocumentLayout.PaintContext()
        context.clip = QC.QRectF(0, 0, text_rect.width(), text_rect.height())
        document.documentLayout().draw(painter, context)
        painter.restore()

    def sizeHint(
        self,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> QC.QSize:
        """Return the height required by the icon and the wrapped catalog text."""
        icon_offset = self._icon_offset(index)
        width = max(1, self._content_width(option) - icon_offset)
        document = self._document(option, index, width)
        content_height = ceil(document.size().height())
        if icon_offset:
            content_height = max(content_height, self.ICON_SIZE)
        height = content_height + 2 * self.VERTICAL_MARGIN
        return QC.QSize(width + icon_offset + 2 * self.HORIZONTAL_MARGIN, height)


def _configure_application_selector(combo: QW.QComboBox) -> None:
    """Configure a combo box listing applications with their description.

    The closed combo box shows the icon and the name of the application; its
    list also shows the version and the description of each application.
    """
    combo.setIconSize(QC.QSize(24, 24))
    # Let the panel be narrower than the longest application name
    combo.setSizeAdjustPolicy(QW.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    combo.setSizePolicy(QW.QSizePolicy.Expanding, QW.QSizePolicy.Fixed)
    combo.setItemDelegate(CatalogItemDelegate(combo))
    view = combo.view()
    view.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
    view.setWordWrap(True)


def _add_application_item(combo: QW.QComboBox, plugin: PluginBase) -> None:
    """Add an application plugin to an application selector."""
    info = plugin.info
    combo.addItem(get_plugin_icon(info.icon), info.name, plugin.plugin_id)
    index = combo.count() - 1
    combo.setItemData(index, info.name, CATALOG_TITLE_ROLE)
    combo.setItemData(index, info.description, CATALOG_DESCRIPTION_ROLE)
    combo.setItemData(index, info.version, CATALOG_VERSION_ROLE)
    combo.setItemData(
        index,
        _("Plugin ID: %s") % plugin.plugin_id + "\n" + _("Version: %s") % info.version,
        QC.Qt.ToolTipRole,
    )


def get_application_plugins() -> tuple[PluginBase, ...]:
    """Return active application plugins in stable display order."""
    plugins = (
        plugin
        for plugin in PluginRegistry.get_plugins()
        if PluginCapability.APPLICATION in plugin.info.capabilities
    )
    return tuple(sorted(plugins, key=lambda plugin: plugin.info.name.casefold()))


def get_declared_metadata_keys(object_type: str) -> list[tuple[str, str]]:
    """Return the metadata keys expected by the methods of application plugins

    Args:
        object_type: type of the objects carrying the metadata ("signal" or
         "image")

    Returns:
        ``(key, description)`` pairs, in declaration order, without duplicates
    """
    keys: dict[str, str] = {}
    for plugin in get_application_plugins():
        for recipe in plugin.get_recipes():
            for slot in recipe.inputs:
                if slot.object_type.value != object_type:
                    continue
                for requirement in slot.metadata:
                    keys.setdefault(requirement.key, requirement.description)
    return list(keys.items())


def record_application_use(plugin_id: str) -> None:
    """Remember an application as the most recently used one

    Args:
        plugin_id: ID of the application plugin
    """
    recent = [
        other for other in Conf.welcome_recent_applications.get() if other != plugin_id
    ]
    Conf.welcome_recent_applications.set([plugin_id, *recent][:MAX_RECENT_APPLICATIONS])


def set_application_pinned(plugin_id: str, pinned: bool) -> None:
    """Pin an application to the top of the welcome page, or unpin it

    Pinning an application also shows it again if it was hidden.

    Args:
        plugin_id: ID of the application plugin
        pinned: True to pin the application, False to unpin it
    """
    pinned_ids = [
        other for other in Conf.welcome_pinned_applications.get() if other != plugin_id
    ]
    if pinned:
        pinned_ids.append(plugin_id)
        set_application_hidden(plugin_id, False)
    Conf.welcome_pinned_applications.set(pinned_ids)


def set_application_hidden(plugin_id: str, hidden: bool) -> None:
    """Hide an application from the welcome page, or show it again

    Hiding an application also unpins it.

    Args:
        plugin_id: ID of the application plugin
        hidden: True to hide the application, False to show it
    """
    hidden_ids = [
        other for other in Conf.welcome_hidden_applications.get() if other != plugin_id
    ]
    if hidden:
        hidden_ids.append(plugin_id)
        set_application_pinned(plugin_id, False)
    Conf.welcome_hidden_applications.set(hidden_ids)


def sort_welcome_applications(plugins: Iterable[PluginBase]) -> list[PluginBase]:
    """Return the applications shown on the welcome page, in display order

    Pinned applications come first (in pinning order), then recently used ones
    (most recent first), then the others by name. Hidden applications are
    excluded.

    Args:
        plugins: Application plugins

    Returns:
        Visible application plugins, in display order
    """
    hidden = set(Conf.welcome_hidden_applications.get())
    pinned = Conf.welcome_pinned_applications.get()
    recent = Conf.welcome_recent_applications.get()

    def sort_key(plugin: PluginBase) -> tuple[int, int, str]:
        if plugin.plugin_id in pinned:
            return 0, pinned.index(plugin.plugin_id), ""
        if plugin.plugin_id in recent:
            return 1, recent.index(plugin.plugin_id), ""
        return 2, 0, plugin.info.name.casefold()

    visible = (plugin for plugin in plugins if plugin.plugin_id not in hidden)
    return sorted(visible, key=sort_key)


def _command_button(icon_name: str, text: str) -> QW.QPushButton:
    """Create a catalog command button that never acts as the dialog default."""
    button = QW.QPushButton(get_icon(icon_name), text)
    button.setAutoDefault(False)
    button.setDefault(False)
    return button


def _status_icon(color: str) -> QG.QIcon:
    """Return a round status icon of the given color."""
    pixmap = QG.QPixmap(32, 32)
    pixmap.fill(QC.Qt.transparent)
    painter = QG.QPainter(pixmap)
    painter.setRenderHint(QG.QPainter.Antialiasing)
    painter.setPen(QC.Qt.NoPen)
    painter.setBrush(QG.QColor(color))
    painter.drawEllipse(6, 6, 20, 20)
    painter.end()
    return QG.QIcon(pixmap)


def _entry_layout(
    title: str,
    description: str,
    button: QW.QPushButton,
    icon: QG.QIcon | None = None,
) -> QW.QHBoxLayout:
    """Return a row presenting a catalog entry and its command."""
    layout = QW.QHBoxLayout()
    if icon is not None:
        icon_label = QW.QLabel()
        icon_label.setPixmap(icon.pixmap(24, 24))
        layout.addWidget(icon_label, 0, QC.Qt.AlignTop)
    text = f"<b>{html.escape(title)}</b>"
    if description:
        text += f"<br>{html.escape(description)}"
    label = QW.QLabel(text)
    label.setTextFormat(QC.Qt.RichText)
    label.setWordWrap(True)
    layout.addWidget(label, 1)
    layout.addWidget(button, 0, QC.Qt.AlignTop)
    return layout


def _format_outcome_html(outcome: RecipeOutcome) -> str:
    """Return the number of created objects and the diagnostics of a run."""
    lines = [
        html.escape(_("Created %d objects") % len(outcome.objects)),
        *(
            f"• {html.escape(format_diagnostic(diagnostic))}"
            for diagnostic in outcome.diagnostics
        ),
    ]
    return "<br>".join(lines)


def _outcome_label() -> QW.QLabel:
    """Create the hidden label showing the outcome of a run."""
    label = QW.QLabel()
    label.setTextFormat(QC.Qt.RichText)
    label.setWordWrap(True)
    label.hide()
    return label


def _separator() -> QW.QFrame:
    """Create a horizontal line separating two entries."""
    line = QW.QFrame()
    line.setFrameShape(QW.QFrame.HLine)
    line.setFrameShadow(QW.QFrame.Sunken)
    return line


class ClickableLabel(QW.QLabel):
    """Label emitting a signal when it is clicked with the left mouse button."""

    #: Emitted when the label is clicked
    clicked = QC.Signal()

    def __init__(self, text: str = "", parent: QW.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QC.Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event: QG.QMouseEvent) -> None:  # pylint: disable=invalid-name
        """Emit :attr:`clicked` on a left click."""
        super().mouseReleaseEvent(event)
        if event.button() == QC.Qt.LeftButton:
            self.clicked.emit()


class DisclosureButton(QW.QToolButton):
    """Arrow button showing content (pointing down) or hiding it (pointing right).

    The button is not checkable, so that the style does not draw it pressed
    while the content is shown.

    Args:
        show_tip: tooltip when the content is hidden
        hide_tip: tooltip when the content is shown
        parent: parent widget
    """

    #: Emitted with True when the content is to be shown, False otherwise
    expanded_changed = QC.Signal(bool)

    def __init__(
        self, show_tip: str = "", hide_tip: str = "", parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.__tips = (show_tip, hide_tip)
        self.__expanded = False
        self.setAutoRaise(True)
        self.clicked.connect(self.toggle_expanded)
        self.__update_arrow()

    def __update_arrow(self) -> None:
        """Point the arrow down when the content is shown, right otherwise."""
        expanded = self.__expanded
        self.setArrowType(QC.Qt.DownArrow if expanded else QC.Qt.RightArrow)
        tip = self.__tips[int(expanded)]
        self.setToolTip(tip)
        self.setAccessibleName(tip)

    def is_expanded(self) -> bool:
        """Return whether the content is shown."""
        return self.__expanded

    def set_expanded(self, expanded: bool) -> None:
        """Show the content, or hide it.

        Args:
            expanded: True to show the content, False to hide it
        """
        if expanded != self.__expanded:
            self.__expanded = expanded
            self.__update_arrow()
            self.expanded_changed.emit(expanded)

    def toggle_expanded(self) -> None:
        """Hide the content if it is shown, show it otherwise."""
        self.set_expanded(not self.__expanded)


#: Colors of the readiness dots, by readiness status
READINESS_COLORS = {
    RecipeReadinessStatus.READY: "#2e7d32",
    RecipeReadinessStatus.WARNINGS: "#b26a00",
    RecipeReadinessStatus.NEEDS_ASSIGNMENT: "#1565c0",
    RecipeReadinessStatus.NOT_READY: "#c62828",
    RecipeReadinessStatus.NO_INPUT: "#808080",
}

#: Keys of the sections of an application page
SECTION_KEYS = ("methods", "tools", "datasets")


def _apply_section_title_style(label: QW.QLabel) -> None:
    """Show a label as a small uppercase section title, like the welcome page."""
    font = label.font()
    font.setBold(True)
    font.setCapitalization(QG.QFont.AllUppercase)
    font.setLetterSpacing(QG.QFont.PercentageSpacing, 104)
    label.setFont(font)
    apply_subdued_color(label)


def _wrapped_label(text: str, rich: bool = False) -> QW.QLabel:
    """Create a label whose text wraps to the available width."""
    label = QW.QLabel(text)
    label.setWordWrap(True)
    if rich:
        label.setTextFormat(QC.Qt.RichText)
    return label


def _readiness_texts(
    recipe: RecipeDescriptor, readiness: RecipeReadiness | None, error: str | None
) -> tuple[str, str, list[str], list[str]]:
    """Return how to present the readiness of a recipe on the current selection.

    Args:
        recipe: recipe descriptor
        readiness: assessment of the current selection, or None on failure
        error: message explaining why the selection could not be assessed

    Returns:
        Color of the readiness dot, summary, all reasons, and the reasons worth
         showing besides the tooltip
    """
    if readiness is None:
        reasons = [error] if error else []
        return (
            READINESS_COLORS[RecipeReadinessStatus.NOT_READY],
            _("The current selection could not be assessed"),
            reasons,
            reasons,
        )
    summary, reasons = format_readiness(readiness, recipe)
    # Without input data, the reasons only repeat the expected inputs
    no_input = readiness.status is RecipeReadinessStatus.NO_INPUT
    return (
        READINESS_COLORS[readiness.status],
        summary,
        reasons,
        [] if no_input else (reasons),
    )


class CollapsibleSection(QW.QWidget):
    """Titled section whose content may be shown or hidden.

    Clicking the arrow or the title shows the content, or hides it.

    Args:
        title: section title
        content: section content
        expanded: True to show the content initially
        parent: parent widget
    """

    #: Emitted with True when the content is shown, False when it is hidden
    expanded_changed = QC.Signal(bool)

    def __init__(
        self,
        title: str,
        content: QW.QWidget,
        expanded: bool = True,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.content = content
        self.toggle_button = DisclosureButton()
        self.title_label = ClickableLabel(title)
        _apply_section_title_style(self.title_label)
        header = QW.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(self.toggle_button)
        header.addWidget(self.title_label, 1)
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(content)
        self.set_expanded(expanded)
        self.toggle_button.expanded_changed.connect(self.__show_content)
        self.title_label.clicked.connect(self.toggle_button.toggle_expanded)

    def __show_content(self, expanded: bool) -> None:
        """Show the content, or hide it, after a click on the section title."""
        self.content.setVisible(expanded)
        self.expanded_changed.emit(expanded)

    def is_expanded(self) -> bool:
        """Return whether the content is shown."""
        return self.toggle_button.is_expanded()

    def set_expanded(self, expanded: bool) -> None:
        """Show the content, or hide it.

        Args:
            expanded: True to show the content, False to hide it
        """
        self.toggle_button.blockSignals(True)
        self.toggle_button.set_expanded(expanded)
        self.toggle_button.blockSignals(False)
        self.content.setVisible(expanded)


class RecipeRow(QW.QFrame):
    """Selectable row presenting a recipe: readiness, title and run command.

    The row is selected by a click, or with the Up and Down keys from another
    row; the page then shows the details of its recipe. The outcome of the last
    run of the recipe is shown below the row.
    """

    #: Emitted with the recipe ID when running the recipe on the selection
    run_requested = QC.Signal(str)
    #: Emitted with the recipe ID when the row is clicked
    selection_requested = QC.Signal(str)
    #: Emitted with -1 or 1 to select the previous or the next row
    navigation_requested = QC.Signal(int)

    #: Size of the readiness dot icon, in pixels
    STATUS_ICON_SIZE = 16
    #: Width of the mark at the left of the selected row, in pixels
    SELECTION_MARK_WIDTH = 3

    def __init__(
        self, recipe: RecipeDescriptor, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.__selected = False
        self.status_color = READINESS_COLORS[RecipeReadinessStatus.NO_INPUT]
        self.status_summary = ""
        self.setFocusPolicy(QC.Qt.StrongFocus)
        self.status_icon_label = QW.QLabel()
        self.title_label = _wrapped_label(recipe.title)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.summary_label = _wrapped_label("")
        apply_subdued_color(self.summary_label)
        self.run_button = _command_button("analysis.svg", _("Run..."))
        self.run_button.setToolTip(_("Run this method on the current selection"))
        self.outcome_label = _outcome_label()

        layout = QW.QGridLayout(self)
        layout.setContentsMargins(self.SELECTION_MARK_WIDTH + 4, 4, 4, 4)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(1)
        layout.addWidget(self.status_icon_label, 0, 0)
        layout.addWidget(self.title_label, 0, 1)
        layout.addWidget(self.run_button, 0, 2, 2, 1, QC.Qt.AlignVCenter)
        layout.addWidget(self.summary_label, 1, 1)
        layout.addWidget(self.outcome_label, 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        self.run_button.clicked.connect(self.__run)
        self.set_status(self.status_color, "", "")

    def __run(self) -> None:
        """Select the row, then request to run its recipe."""
        self.selection_requested.emit(self.recipe.recipe_id)
        self.run_requested.emit(self.recipe.recipe_id)

    def mousePressEvent(self, event: QG.QMouseEvent) -> None:  # pylint: disable=invalid-name
        """Select the row on a left click."""
        if event.button() == QC.Qt.LeftButton:
            self.selection_requested.emit(self.recipe.recipe_id)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QG.QKeyEvent) -> None:  # pylint: disable=invalid-name
        """Select the previous or the next row with the Up and Down keys."""
        if event.key() in (QC.Qt.Key_Up, QC.Qt.Key_Down):
            self.navigation_requested.emit(-1 if event.key() == QC.Qt.Key_Up else 1)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QG.QPaintEvent) -> None:  # pylint: disable=invalid-name
        """Highlight the selected row with the color of the current theme."""
        if self.__selected:
            painter = QG.QPainter(self)
            color = QG.QColor(self.palette().color(QG.QPalette.Highlight))
            color.setAlpha(60)
            painter.fillRect(self.rect(), color)
            color.setAlpha(255)
            mark = QC.QRect(0, 0, self.SELECTION_MARK_WIDTH, self.height())
            painter.fillRect(mark, color)
            painter.end()
        super().paintEvent(event)

    def is_selected(self) -> bool:
        """Return whether the row is selected."""
        return self.__selected

    def set_selected(self, selected: bool) -> None:
        """Highlight the row, or remove its highlighting.

        Args:
            selected: True if the row is selected
        """
        self.__selected = selected
        self.update()

    def set_status(self, color: str, summary: str, tooltip: str) -> None:
        """Show the readiness of the recipe on the current selection.

        Args:
            color: color of the readiness dot
            summary: summary of the readiness
            tooltip: summary and reasons of the readiness
        """
        self.status_color = color
        self.status_summary = summary
        size = self.STATUS_ICON_SIZE
        self.status_icon_label.setPixmap(_status_icon(color).pixmap(size, size))
        self.summary_label.setText(summary)
        self.summary_label.setVisible(bool(summary))
        for widget in (self.status_icon_label, self.summary_label):
            widget.setToolTip(tooltip)

    def show_outcome(self, outcome: RecipeOutcome) -> None:
        """Show the number of created objects and the diagnostics of a run.

        Args:
            outcome: outcome of the last run of the recipe
        """
        self.outcome_label.setText(_format_outcome_html(outcome))
        self.outcome_label.show()


class RecipeDetails(QW.QWidget):
    """Details of a recipe: description, readiness, expected inputs and examples.

    Nothing is left out: the expected inputs give the description of each input
    and, for each metadata, its description and its full key.
    """

    #: Emitted with the example ID and the recipe ID when trying an example
    try_example_requested = QC.Signal(str, str)

    def __init__(
        self,
        recipe: RecipeDescriptor,
        examples: Iterable[PluginExample],
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.title_label = _wrapped_label(recipe.title)
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        self.title_label.setFont(font)
        self.readiness_label = _wrapped_label("", rich=True)
        self.inputs_widget = QW.QWidget()
        self.example_buttons: dict[str, QW.QPushButton] = {}

        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        version = QW.QLabel(_("Version: %s") % recipe.version)
        apply_subdued_color(version)
        layout.addWidget(version)
        if recipe.description:
            layout.addWidget(_wrapped_label(recipe.description))
        layout.addWidget(self.readiness_label)
        inputs_title = QW.QLabel(_("Expected inputs"))
        _apply_section_title_style(inputs_title)
        layout.addWidget(inputs_title)
        self.__setup_inputs()
        layout.addWidget(self.inputs_widget)
        self.__add_examples(layout, tuple(examples))
        layout.addStretch()

    def __setup_inputs(self) -> None:
        """List each input with its description and its metadata."""
        layout = QW.QVBoxLayout(self.inputs_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if not self.recipe.inputs:
            label = QW.QLabel(_("No input data"))
            apply_subdued_color(label)
            layout.addWidget(label)
        for index, slot in enumerate(self.recipe.inputs):
            if index:
                layout.addSpacing(6)
            layout.addWidget(
                _wrapped_label(
                    f"<b>{html.escape(slot.display_title)}</b> — "
                    f"{html.escape(format_slot_kind(slot))}",
                    rich=True,
                )
            )
            if slot.description:
                description = _wrapped_label(slot.description)
                apply_subdued_color(description)
                layout.addWidget(description)
            for requirement in slot.metadata:
                label = _("required metadata") if requirement.required else _("hint")
                text = f"<i>{html.escape(label)}</i>"
                if requirement.description:
                    text += f": {html.escape(requirement.description)}"
                requirement_label = _wrapped_label(text, rich=True)
                key_label = _wrapped_label(
                    f"<code>{html.escape(requirement.key)}</code>", rich=True
                )
                key_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
                apply_subdued_color(key_label)
                for widget in (requirement_label, key_label):
                    widget.setContentsMargins(14, 0, 0, 0)
                    layout.addWidget(widget)

    def __add_examples(
        self, layout: QW.QVBoxLayout, examples: tuple[PluginExample, ...]
    ) -> None:
        """List the examples designed for the recipe, each with its command."""
        if examples:
            title = QW.QLabel(_("Examples designed for this method"))
            _apply_section_title_style(title)
            layout.addSpacing(4)
            layout.addWidget(title)
        for example in examples:
            text = f"<b>{html.escape(example.title)}</b>"
            if example.description:
                text += f"<br>{html.escape(example.description)}"
            layout.addWidget(_wrapped_label(text, rich=True))
            button = _command_button("play_demo.svg", _("Try with this example"))
            button.clicked.connect(
                lambda _checked=False, example_id=example.id: (
                    self.try_example_requested.emit(example_id, self.recipe.recipe_id)
                )
            )
            self.example_buttons[example.id] = button
            button_layout = QW.QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(button)
            layout.addLayout(button_layout)

    def set_status(self, color: str, summary: str, reasons: list[str]) -> None:
        """Show the readiness of the recipe on the current selection.

        Args:
            color: color of the readiness dot
            summary: summary of the readiness
            reasons: reasons of the readiness
        """
        text = f'<span style="color:{color}">●</span> <b>{html.escape(summary)}</b>'
        if reasons:
            text += "<br>" + "<br>".join(
                f"• {html.escape(reason)}" for reason in reasons
            )
        self.readiness_label.setText(text)


def _scroll_area(widget: QW.QWidget) -> QW.QScrollArea:
    """Return a frameless area scrolling a widget vertically."""
    area = QW.QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QW.QFrame.NoFrame)
    return area


class ApplicationPage(QW.QWidget):
    """Present the methods, tools and datasets of one application plugin.

    The methods come first, then the tools and the datasets of the application,
    if any, each group in a section that may be collapsed. Below, separated by a
    splitter, the page gives the details of the selected method. Both parts
    scroll vertically, so that the page fits a narrow panel.
    """

    #: Emitted with the plugin and a recipe ID to run it on the selection
    start_requested = QC.Signal(object, str)
    #: Emitted with the plugin, an example ID and a recipe ID to try an example
    try_example_requested = QC.Signal(object, str, str)
    #: Emitted with the plugin and the ID of an example without recipe
    open_example_requested = QC.Signal(object, str)
    #: Emitted with the plugin and a tool ID
    tool_requested = QC.Signal(object, str)
    #: Emitted with the plugin ID and the recipe ID when a method is selected
    recipe_selected = QC.Signal(str, str)
    #: Emitted with the section key and True if it is expanded, False otherwise
    section_toggled = QC.Signal(str, bool)
    #: Emitted with the heights of the list and of the details when the user
    #: moves the splitter
    splitter_moved = QC.Signal(list)

    def __init__(self, plugin: PluginBase, parent: QW.QWidget | None = None):
        super().__init__(parent)
        self.plugin = plugin
        self.recipe_rows: dict[str, RecipeRow] = {}
        self.recipe_details: dict[str, RecipeDetails] = {}
        self.tool_buttons: dict[str, QW.QPushButton] = {}
        self.dataset_buttons: dict[str, QW.QPushButton] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        #: Outcome of the commands which are not attached to a method
        self.outcome_label = _outcome_label()
        self.splitter = QW.QSplitter(QC.Qt.Vertical)
        self.details_stack = QW.QStackedWidget()
        #: Title separating the details of the selected method from the list
        self.details_title_label = QW.QLabel(_("Details"))
        self.__details_areas: dict[str, QW.QScrollArea] = {}
        self.__current_recipe_id: str | None = None

        content = QW.QWidget()
        layout = QW.QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)
        info = plugin.info
        description = _wrapped_label(info.description)
        apply_subdued_color(description)
        description.setToolTip(
            _("Plugin ID: %s") % plugin.plugin_id
            + "\n"
            + _("Version: %s") % info.version
        )
        # Parented first: a parentless widget made visible flashes as a window
        layout.addWidget(description)
        description.setVisible(bool(info.description))
        collapsed = Conf.applications_collapsed_sections.get()
        self._add_methods(layout, collapsed)
        self._add_tools(layout, collapsed)
        self._add_datasets(layout, collapsed)
        layout.addWidget(self.outcome_label)
        layout.addStretch()
        self.list_area = _scroll_area(content)

        self.splitter.addWidget(self.list_area)
        self.splitter.setChildrenCollapsible(False)
        if self.recipe_rows:
            self.splitter.addWidget(self.__create_details_area())
            self.splitter.setStretchFactor(0, 2)
            self.splitter.setStretchFactor(1, 3)
            selected = Conf.applications_selected_methods.get().get(plugin.plugin_id)
            self.select_recipe(
                selected
                if selected in self.recipe_rows
                else next(iter(self.recipe_rows))
            )
        self.splitter.splitterMoved.connect(
            lambda *_args: self.splitter_moved.emit(self.splitter.sizes())
        )
        page_layout = QW.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(self.splitter)

    def __create_details_area(self) -> QW.QWidget:
        """Create the area showing the details of the selected method.

        A line and a title separate it from the list above.
        """
        area = QW.QWidget()
        layout = QW.QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_separator())
        _apply_section_title_style(self.details_title_label)
        self.details_title_label.setContentsMargins(8, 6, 8, 0)
        layout.addWidget(self.details_title_label)
        layout.addWidget(self.details_stack, 1)
        return area

    def _add_section(
        self,
        layout: QW.QVBoxLayout,
        key: str,
        title: str,
        content: QW.QWidget,
        collapsed: list[str],
    ) -> None:
        """Add a collapsible section to the page, expanded unless collapsed."""
        section = CollapsibleSection(title, content, key not in collapsed)
        section.expanded_changed.connect(
            lambda expanded: self.section_toggled.emit(key, expanded)
        )
        self.sections[key] = section
        layout.addWidget(section)

    def _add_methods(self, layout: QW.QVBoxLayout, collapsed: list[str]) -> None:
        """Add one row per recipe, and the details of each recipe."""
        recipes = self.plugin.get_recipes()
        if not recipes:
            label = QW.QLabel(_("No methods declared"))
            apply_subdued_color(label)
            layout.addWidget(label)
            return
        examples = self.plugin.get_examples()
        methods = QW.QWidget()
        methods_layout = QW.QVBoxLayout(methods)
        methods_layout.setContentsMargins(0, 0, 0, 0)
        methods_layout.setSpacing(0)
        for index, recipe in enumerate(recipes):
            if index:
                methods_layout.addWidget(_separator())
            row = RecipeRow(recipe)
            row.run_requested.connect(
                lambda recipe_id: self.start_requested.emit(self.plugin, recipe_id)
            )
            row.selection_requested.connect(self.select_recipe)
            row.navigation_requested.connect(self.__select_next)
            self.recipe_rows[recipe.recipe_id] = row
            methods_layout.addWidget(row)
            details = RecipeDetails(
                recipe,
                (
                    example
                    for example in examples
                    if recipe.recipe_id in example.recipe_ids
                ),
            )
            details.try_example_requested.connect(
                lambda example_id, recipe_id: self.try_example_requested.emit(
                    self.plugin, example_id, recipe_id
                )
            )
            self.recipe_details[recipe.recipe_id] = details
            area = _scroll_area(details)
            self.__details_areas[recipe.recipe_id] = area
            self.details_stack.addWidget(area)
        self._add_section(
            layout, "methods", _("Methods (%d)") % len(recipes), methods, collapsed
        )

    def _add_entries_section(
        self,
        layout: QW.QVBoxLayout,
        key: str,
        title: str,
        entry_layouts: list[QW.QHBoxLayout],
        collapsed: list[str],
    ) -> None:
        """Add a section listing entries, if any."""
        if not entry_layouts:
            return
        entries = QW.QWidget()
        entries_layout = QW.QVBoxLayout(entries)
        entries_layout.setContentsMargins(0, 0, 0, 0)
        for entry_layout in entry_layouts:
            entries_layout.addLayout(entry_layout)
        self._add_section(layout, key, title, entries, collapsed)

    def _add_tools(self, layout: QW.QVBoxLayout, collapsed: list[str]) -> None:
        """Add the plugin-owned tools, if any."""
        tools = self.plugin.get_tools()
        entry_layouts = []
        for tool in tools:
            button = _command_button("libre-toolbox.svg", _("Open tool"))
            button.clicked.connect(
                lambda _checked=False, tool_id=tool.id: self.tool_requested.emit(
                    self.plugin, tool_id
                )
            )
            self.tool_buttons[tool.id] = button
            entry_layouts.append(
                _entry_layout(
                    tool.title, tool.description, button, get_plugin_icon(tool.icon)
                )
            )
        self._add_entries_section(
            layout, "tools", _("Tools (%d)") % len(tools), entry_layouts, collapsed
        )

    def _add_datasets(self, layout: QW.QVBoxLayout, collapsed: list[str]) -> None:
        """Add the examples that are not designed for a specific recipe."""
        datasets = [
            example for example in self.plugin.get_examples() if not example.recipe_ids
        ]
        entry_layouts = []
        for example in datasets:
            button = _command_button("io/fileopen_h5.svg", _("Open dataset"))
            button.clicked.connect(
                lambda _checked=False, example_id=example.id: (
                    self.open_example_requested.emit(self.plugin, example_id)
                )
            )
            self.dataset_buttons[example.id] = button
            entry_layouts.append(
                _entry_layout(example.title, example.description, button)
            )
        self._add_entries_section(
            layout,
            "datasets",
            _("Datasets (%d)") % len(datasets),
            entry_layouts,
            collapsed,
        )

    def current_recipe_id(self) -> str | None:
        """Return the ID of the selected recipe, if any."""
        return self.__current_recipe_id

    def select_recipe(self, recipe_id: str) -> None:
        """Select a recipe: highlight its row and show its details.

        Args:
            recipe_id: recipe ID

        Raises:
            KeyError: if the application has no recipe with this ID
        """
        row = self.recipe_rows[recipe_id]
        for other_id, other_row in self.recipe_rows.items():
            other_row.set_selected(other_id == recipe_id)
        self.details_stack.setCurrentWidget(self.__details_areas[recipe_id])
        self.list_area.ensureWidgetVisible(row)
        if recipe_id != self.__current_recipe_id:
            self.__current_recipe_id = recipe_id
            self.recipe_selected.emit(self.plugin.plugin_id, recipe_id)

    def __select_next(self, step: int) -> None:
        """Select the previous (step -1) or the next (step 1) recipe."""
        recipe_ids = list(self.recipe_rows)
        index = recipe_ids.index(self.__current_recipe_id) + step
        if 0 <= index < len(recipe_ids):
            self.select_recipe(recipe_ids[index])
            self.recipe_rows[recipe_ids[index]].setFocus()

    def apply_section_states(self, collapsed: list[str]) -> None:
        """Collapse the sections given by their keys, and expand the others.

        Args:
            collapsed: keys of the collapsed sections
        """
        for key, section in self.sections.items():
            section.set_expanded(key not in collapsed)

    def set_splitter_sizes(self, sizes: tuple[int, ...] | None) -> None:
        """Set the heights of the list and of the details, if given.

        Args:
            sizes: heights of the list and of the details, or None
        """
        if sizes is not None and len(sizes) == self.splitter.count():
            self.splitter.setSizes(list(sizes))

    def refresh_readiness(self) -> None:
        """Assess every recipe and tool on the current selection."""
        for recipe_id, row in self.recipe_rows.items():
            error = None
            try:
                readiness = self.plugin.assess_recipe(recipe_id)
            except Exception as exc:  # pylint: disable=broad-except
                # Binding suggestions are third-party code: keep the panel usable
                readiness, error = None, str(exc)
            color, summary, reasons, shown_reasons = _readiness_texts(
                row.recipe, readiness, error
            )
            tooltip = "\n".join([summary, *(f"• {reason}" for reason in reasons)])
            row.set_status(color, summary, tooltip)
            self.recipe_details[recipe_id].set_status(color, summary, shown_reasons)
        for tool_id, button in self.tool_buttons.items():
            issue = self.plugin.assess_tool(tool_id)
            button.setEnabled(issue is None)
            button.setToolTip(issue or "")

    def show_outcome(
        self, outcome: RecipeOutcome, recipe_id: str | None = None
    ) -> None:
        """Show the number of created objects and the diagnostics of a run.

        Args:
            outcome: outcome of the run
            recipe_id: ID of the recipe which was run, if any: its row shows the
             outcome; otherwise, the outcome is shown at the bottom of the list
        """
        row = self.recipe_rows.get(recipe_id) if recipe_id is not None else None
        if row is not None:
            row.show_outcome(outcome)
        else:
            self.outcome_label.setText(_format_outcome_html(outcome))
            self.outcome_label.show()


class FlashOverlay(QW.QWidget):
    """Frame drawn over a widget, in the highlight color of the current theme.

    The frame lets mouse events through to the widget below.
    """

    #: Width of the frame, in pixels
    FRAME_WIDTH = 3

    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QC.Qt.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, event: QG.QPaintEvent) -> None:  # pylint: disable=invalid-name
        """Draw a light highlight over the widget, framed by a thicker border."""
        painter = QG.QPainter(self)
        color = QG.QColor(self.palette().color(QG.QPalette.Highlight))
        color.setAlpha(40)
        painter.fillRect(self.rect(), color)
        color.setAlpha(255)
        width = self.FRAME_WIDTH
        pen = QG.QPen(color, width)
        pen.setJoinStyle(QC.Qt.MiterJoin)
        painter.setPen(pen)
        half = width // 2
        painter.drawRect(self.rect().adjusted(half, half, -half - 1, -half - 1))
        painter.end()
        super().paintEvent(event)


class ApplicationsPanel(QW.QWidget, DockableWidgetMixin):
    """Dockable panel presenting the methods of the active application plugins.

    With several applications, a selector headed by their number chooses the
    application; with only one, its name is shown instead. The page of the
    application lists its methods with their readiness on the current selection,
    then its tools and datasets, above the details of the selected method. The
    other commands of the application (documentation, welcome page options) are
    in the menu next to the selector.

    The panel floats beside the main window the first time it is shown (see
    :meth:`place_beside`); it may then be docked, tabified or left floating, as
    any other panel.

    Args:
        parent: main window
    """

    LOCATION = QC.Qt.LeftDockWidgetArea

    #: Emitted when an application is shown, hidden, pinned or used
    SIG_WELCOME_PREFERENCES_CHANGED = QC.Signal()

    #: Minimum width of the panel
    MINIMUM_WIDTH = 280
    #: Preferred width of the panel, docked or floating beside the main window
    DEFAULT_WIDTH = 440
    #: Size of the application icon shown when there is only one application
    TITLE_ICON_SIZE = 24
    #: Number of flashes drawing attention to the panel, and their half period (ms)
    FLASH_COUNT = 3
    FLASH_INTERVAL = 150

    def __init__(self, parent: QW.QWidget | None = None):
        QW.QWidget.__init__(self, parent)
        DockableWidgetMixin.__init__(self)
        self.count_label = QW.QLabel()
        self.application_combo = QW.QComboBox()
        self.title_widget = QW.QWidget()
        self.title_icon_label = QW.QLabel()
        self.title_label = QW.QLabel()
        self.menu_button = QW.QToolButton()
        self.menu = QW.QMenu(self.menu_button)
        self.documentation_action = self.menu.addAction(
            get_icon("libre-gui-help.svg"), _("Documentation")
        )
        self.menu.addSeparator()
        self.show_on_welcome_action = self.menu.addAction(_("Show on welcome page"))
        self.pin_on_welcome_action = self.menu.addAction(_("Pin to the welcome page"))
        self.application_stack = QW.QStackedWidget()
        self.application_pages: list[ApplicationPage] = []
        self.empty_label = QW.QLabel(_("No application plugins are currently loaded."))
        self.__plugins: tuple[PluginBase, ...] | None = None
        self.__readiness_timer = QC.QTimer(self)
        self.__readiness_timer.setSingleShot(True)
        self.__readiness_timer.setInterval(150)
        self.__readiness_timer.timeout.connect(self.refresh_readiness)
        #: Frame drawn over the panel to draw attention to it (see :meth:`flash`)
        self.flash_overlay = FlashOverlay(self)
        self.__flash_timer = QC.QTimer(self)
        self.__flash_timer.setInterval(self.FLASH_INTERVAL)
        self.__flash_timer.timeout.connect(self.__toggle_flash)
        self.__flash_steps = 0
        self.setMinimumWidth(self.MINIMUM_WIDTH)
        self.__setup_ui()
        self.refresh()

    def __setup_ui(self) -> None:
        """Lay out the selector, its menu and the application pages."""
        _apply_section_title_style(self.count_label)
        combo = self.application_combo
        _configure_application_selector(combo)
        font = combo.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 1)
        combo.setFont(font)
        # The list keeps the regular font, to show the descriptions
        combo.view().setFont(self.font())
        combo.setToolTip(_("Click to show all available application plugins"))
        combo.setAccessibleName(_("Application"))
        combo.currentIndexChanged.connect(self.__show_application)

        title_layout = QW.QHBoxLayout(self.title_widget)
        title_layout.setContentsMargins(4, 0, 0, 0)
        title_layout.setSpacing(8)
        title_layout.addWidget(self.title_icon_label)
        title_layout.addWidget(self.title_label, 1)
        self.title_label.setFont(font)
        self.title_label.setWordWrap(True)

        button = self.menu_button
        button.setText("\u2026")
        font = button.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.3)
        button.setFont(font)
        button.setToolTip(_("More actions"))
        button.setAccessibleName(_("More actions"))
        button.setAutoRaise(True)
        button.setPopupMode(QW.QToolButton.InstantPopup)
        button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        button.setMenu(self.menu)
        self.menu.setToolTipsVisible(True)
        self.menu.aboutToShow.connect(self.sync_menu)
        self.documentation_action.triggered.connect(self.__open_current_documentation)
        for action in (self.show_on_welcome_action, self.pin_on_welcome_action):
            action.setCheckable(True)
        self.show_on_welcome_action.toggled.connect(self.__set_shown_on_welcome)
        self.pin_on_welcome_action.toggled.connect(self.__set_pinned_on_welcome)

        self.empty_label.setAlignment(QC.Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        apply_subdued_color(self.empty_label)

        selector_layout = QW.QHBoxLayout()
        selector_layout.setSpacing(2)
        selector_layout.addWidget(combo, 1)
        selector_layout.addWidget(self.title_widget, 1)
        selector_layout.addWidget(button)
        header = QW.QVBoxLayout()
        header.setContentsMargins(6, 6, 6, 0)
        header.setSpacing(2)
        header.addWidget(self.count_label)
        header.addLayout(selector_layout)
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.application_stack, 1)
        layout.addWidget(self.empty_label, 1)

    def sizeHint(self) -> QC.QSize:  # pylint: disable=invalid-name
        """Return the preferred size: wide enough for the application pages."""
        return QC.QSize(self.DEFAULT_WIDTH, super().sizeHint().height())

    def create_dockwidget(
        self, title: str
    ) -> tuple[QW.QDockWidget, QC.Qt.DockWidgetArea]:
        """Create the dock widget, with a title bar following the color theme."""
        dock, location = super().create_dockwidget(title)
        dock.topLevelChanged.connect(self.__fix_title_bar)
        return dock, location

    def __fix_title_bar(self, floating: bool) -> None:
        """Fix the title bar background of the floating panel on Windows."""
        if floating:
            win32_fix_title_bar_background(self.dockwidget)

    def visibility_changed(self, enable: bool) -> None:
        """Float beside the main window the first time, and keep floating after.

        Other panels are docked again when they are shown after being closed
        while floating; the Applications panel stays floating, so that it is
        shown again where the user left it, within the available screen area.
        Docked, it gets back the size it had when it was closed: alone in its
        dock area, it would otherwise get its minimum size.

        Args:
            enable: True if the panel is visible
        """
        was_visible = self.is_visible()
        super().visibility_changed(enable)
        self._was_closed_while_floating = False
        dock = self.dockwidget
        if enable:
            window = dock.parentWidget()
            if not Conf.applications_panel_shown.get():
                Conf.applications_panel_shown.set(True)
                # Placed again once shown: the geometry of a hidden floating dock
                # widget is not always taken into account
                if dock.isFloating():
                    self.place_beside(window)
            self.ensure_on_screen(window)
            if not dock.isFloating() and Conf.applications_panel_docked_size.get():
                # Once the main window has laid out the dock area again
                QC.QTimer.singleShot(0, self.restore_docked_size)
        elif was_visible and dock.isHidden() and not dock.isFloating():
            # Closed while docked (not only hidden behind another tab)
            Conf.applications_panel_docked_size.set((dock.width(), dock.height()))

    def restore_docked_size(self) -> None:
        """Give back to the docked panel the size it had when it was closed.

        The size is given back once: afterwards, the panel keeps the size given
        by the user or by the persisted window layout. It is limited to the room
        the main window can give without growing.
        """
        dock = self.dockwidget
        size = Conf.applications_panel_docked_size.get()
        if size is None or dock.isFloating() or dock.isHidden():
            return
        Conf.applications_panel_docked_size.set(None)
        window = dock.parentWidget()
        horizontal = window.dockWidgetArea(dock) in (
            QC.Qt.LeftDockWidgetArea,
            QC.Qt.RightDockWidgetArea,
        )
        width, height = size
        slack = window.size() - window.minimumSizeHint()
        if horizontal:
            target = min(width, dock.width() + max(0, slack.width()))
            if target > dock.width():
                # Unlike resizeDocks, which only takes room from the central
                # widget, a minimum size also takes room from other dock areas
                dock.setMinimumWidth(target)
        else:
            target = min(height, dock.height() + max(0, slack.height()))
            if target > dock.height():
                dock.setMinimumHeight(target)
        QC.QTimer.singleShot(0, self.__release_docked_size)

    def __release_docked_size(self) -> None:
        """Let the user resize the docked panel again."""
        self.dockwidget.setMinimumSize(0, 0)

    def place_beside(self, window: QW.QWidget) -> None:
        """Float the panel beside a window, on the screen of this window.

        The panel goes to the right of the window when the screen has room for
        it there, otherwise along the right edge of the window. It is as high as
        the window, and it stays within the available area of the screen.

        Args:
            window: window next to which the panel floats
        """
        dock = self.dockwidget
        dock.setFloating(True)
        available = window.screen().availableGeometry()
        frame = window.frameGeometry()
        # Leave room for a title bar as high as the one of the window
        decoration = max(0, frame.height() - window.height())
        width = min(
            max(self.DEFAULT_WIDTH, self.minimumSizeHint().width()), available.width()
        )
        height = max(1, min(window.height(), available.height() - decoration))
        if available.right() - frame.right() >= width:
            left = frame.right() + 1
        else:
            left = min(frame.right(), available.right()) + 1 - width
        left = max(available.left(), left)
        top = max(available.top(), min(frame.top(), available.bottom() - height))
        dock.resize(width, height)
        dock.move(left, top)

    def ensure_on_screen(self, window: QW.QWidget) -> None:
        """Keep the floating panel within the available area of a screen.

        A panel left on a screen which has been disconnected since floats
        beside the window again. Called once the panel is shown, when the size
        of its frame is known.

        Args:
            window: window next to which the panel floats
        """
        dock = self.dockwidget
        if not dock.isFloating():
            return
        screen = QG.QGuiApplication.screenAt(dock.frameGeometry().center())
        if screen is None:
            self.place_beside(window)
            screen = window.screen()
        available = screen.availableGeometry()
        frame = dock.frameGeometry()
        left = max(
            available.left(), min(frame.left(), available.right() + 1 - frame.width())
        )
        top = max(
            available.top(), min(frame.top(), available.bottom() + 1 - frame.height())
        )
        if (left, top) != (frame.left(), frame.top()):
            dock.move(left, top)

    def current_plugin_id(self) -> str | None:
        """Return the ID of the shown application plugin, if any."""
        return self.application_combo.currentData()

    def current_page(self) -> ApplicationPage | None:
        """Return the page of the shown application, if any."""
        page = self.application_stack.currentWidget()
        return page if isinstance(page, ApplicationPage) else None

    def refresh(self) -> None:
        """Rebuild the pages if the application plugins have changed.

        The shown application remains shown if it is still active; otherwise,
        the most recently used application is shown, or the first one.
        """
        plugins = get_application_plugins()
        if self.__plugins is not None and len(plugins) == len(self.__plugins):
            if all(new is old for new, old in zip(plugins, self.__plugins)):
                return
        self.__plugins = plugins
        current_id = self.current_plugin_id()
        combo = self.application_combo
        combo.blockSignals(True)
        combo.clear()
        self.application_pages.clear()
        while self.application_stack.count():
            widget = self.application_stack.widget(0)
            self.application_stack.removeWidget(widget)
            widget.deleteLater()
        for plugin in plugins:
            page = ApplicationPage(plugin)
            page.start_requested.connect(self._start_recipe)
            page.try_example_requested.connect(self._try_example)
            page.open_example_requested.connect(self._open_example)
            page.tool_requested.connect(self._launch_tool)
            page.recipe_selected.connect(self.__remember_recipe)
            page.section_toggled.connect(self.__remember_section)
            page.splitter_moved.connect(self.__remember_splitter_sizes)
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
            _add_application_item(combo, plugin)
        combo.blockSignals(False)

        several = len(plugins) > 1
        self.count_label.setText(_("Applications (%d)") % len(plugins))
        for widget, visible in (
            (self.count_label, several),
            (combo, several),
            (self.title_widget, len(plugins) == 1),
            (self.menu_button, bool(plugins)),
            (self.application_stack, bool(plugins)),
            (self.empty_label, not plugins),
        ):
            widget.setVisible(visible)
        if plugins:
            ids = [plugin.plugin_id for plugin in plugins]
            candidates = (current_id, *Conf.welcome_recent_applications.get())
            index = next(
                (ids.index(candidate) for candidate in candidates if candidate in ids),
                0,
            )
            combo.setCurrentIndex(index)
            self.__show_application(index)

    def __show_application(self, index: int) -> None:
        """Show the page of the application selected in the selector."""
        self.application_stack.setCurrentIndex(index)
        page = self.current_page()
        if page is not None:
            plugin = page.plugin
            size = self.TITLE_ICON_SIZE
            self.title_icon_label.setPixmap(
                get_plugin_icon(plugin.info.icon).pixmap(size, size)
            )
            self.title_label.setText(plugin.info.name)
            self.title_widget.setToolTip(
                self.application_combo.itemData(index, QC.Qt.ToolTipRole) or ""
            )
            # Sections and splitter follow the last changes made on other pages
            page.apply_section_states(Conf.applications_collapsed_sections.get())
            page.set_splitter_sizes(Conf.applications_details_sizes.get())
        self.sync_menu()
        self.schedule_readiness_update()

    @staticmethod
    def __remember_recipe(plugin_id: str, recipe_id: str) -> None:
        """Remember the method selected for an application."""
        selected = dict(Conf.applications_selected_methods.get())
        selected[plugin_id] = recipe_id
        Conf.applications_selected_methods.set(selected)

    @staticmethod
    def __remember_section(key: str, expanded: bool) -> None:
        """Remember whether a section is collapsed, for all applications."""
        collapsed = [
            other
            for other in Conf.applications_collapsed_sections.get()
            if other != key
        ]
        if not expanded:
            collapsed.append(key)
        Conf.applications_collapsed_sections.set(collapsed)

    @staticmethod
    def __remember_splitter_sizes(sizes: list[int]) -> None:
        """Remember the heights of the method list and of the method details."""
        Conf.applications_details_sizes.set(tuple(sizes))

    def select_plugin(self, plugin_id: str) -> None:
        """Show the page of an application plugin

        Args:
            plugin_id: ID of the application plugin

        Raises:
            KeyError: if no active application plugin has this ID
        """
        index = self.application_combo.findData(plugin_id)
        if index < 0:
            raise KeyError(f"Application plugin {plugin_id!r} not found")
        self.application_combo.setCurrentIndex(index)

    def sync_menu(self) -> None:
        """Update the menu from the shown application and the user preferences.

        The welcome page options may have been changed from the welcome page.
        """
        page = self.current_page()
        plugin_id = None if page is None else page.plugin.plugin_id
        url = None if page is None else page.plugin.info.documentation_url
        self.documentation_action.setEnabled(url is not None)
        self.documentation_action.setToolTip(url or "")
        for action, checked in (
            (
                self.show_on_welcome_action,
                plugin_id not in Conf.welcome_hidden_applications.get(),
            ),
            (
                self.pin_on_welcome_action,
                plugin_id in Conf.welcome_pinned_applications.get(),
            ),
        ):
            action.setEnabled(page is not None)
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def __set_shown_on_welcome(self, checked: bool) -> None:
        """Show the shown application on the welcome page, or hide it."""
        set_application_hidden(self.current_plugin_id(), not checked)
        self.sync_menu()
        self.SIG_WELCOME_PREFERENCES_CHANGED.emit()

    def __set_pinned_on_welcome(self, checked: bool) -> None:
        """Pin the shown application to the welcome page, or unpin it."""
        set_application_pinned(self.current_plugin_id(), checked)
        self.sync_menu()
        self.SIG_WELCOME_PREFERENCES_CHANGED.emit()

    def __open_current_documentation(self) -> None:
        """Open the documentation of the shown application."""
        page = self.current_page()
        if page is not None:
            self._open_documentation(page.plugin)

    def schedule_readiness_update(self, *_args: object) -> None:
        """Assess the methods of the shown application after a short delay.

        Connected to selection changes, which may come in bursts.
        """
        if self.isVisible():
            self.__readiness_timer.start()

    def refresh_readiness(self) -> None:
        """Assess the methods of the shown application on the selection."""
        page = self.current_page()
        if page is not None:
            page.refresh_readiness()

    def showEvent(self, event: QG.QShowEvent) -> None:  # pylint: disable=invalid-name
        """Assess the methods on the current selection when shown."""
        super().showEvent(event)
        self.refresh_readiness()

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Keep the flashing frame over the whole panel."""
        super().resizeEvent(event)
        self.flash_overlay.setGeometry(self.rect())

    def flash(self) -> None:
        """Flash a frame over the panel a few times, to draw attention to it.

        Used when the panel is requested while it is already shown: otherwise,
        nothing would seem to happen.
        """
        self.__flash_steps = 2 * self.FLASH_COUNT
        self.flash_overlay.setGeometry(self.rect())
        self.flash_overlay.raise_()
        self.flash_overlay.show()
        self.__flash_timer.start()

    def is_flashing(self) -> bool:
        """Return whether the panel is flashing."""
        return self.__flash_timer.isActive()

    def __toggle_flash(self) -> None:
        """Show the flashing frame, or hide it, until the flashing is over."""
        self.__flash_steps -= 1
        if self.__flash_steps <= 0:
            self.__flash_timer.stop()
            self.flash_overlay.hide()
        else:
            self.flash_overlay.setVisible(not self.flash_overlay.isVisible())

    def _page_of(self, plugin: PluginBase) -> ApplicationPage | None:
        """Return the page of an application plugin."""
        return next(
            (page for page in self.application_pages if page.plugin is plugin), None
        )

    def _record_use(self, plugin: PluginBase) -> None:
        """Remember an application use to order the welcome page."""
        record_application_use(plugin.plugin_id)
        self.SIG_WELCOME_PREFERENCES_CHANGED.emit()

    def _run_command(
        self,
        plugin: PluginBase,
        command: Callable[[], object],
        context: str,
        recipe_id: str | None = None,
    ) -> None:
        """Run a plugin command, report its failure, and show its outcome."""
        self._record_use(plugin)
        try:
            result = command()
        except Exception as exc:  # pylint: disable=broad-except
            # Plugin-owned commands are third-party code: never crash the app
            qt_handle_error_message(self.window(), exc, context)
            return
        page = self._page_of(plugin)
        if page is not None and isinstance(result, RecipeOutcome):
            page.show_outcome(result, recipe_id)
        self.schedule_readiness_update()

    def _start_recipe(self, plugin: PluginBase, recipe_id: str) -> None:
        """Run a recipe on the current selection."""
        self._run_command(
            plugin,
            lambda: plugin.launch_recipe(recipe_id),
            _("Starting analysis '%s'") % recipe_id,
            recipe_id,
        )

    def _try_example(self, plugin: PluginBase, example_id: str, recipe_id: str) -> None:
        """Open an example, then run one of the recipes designed for it."""
        self._run_command(
            plugin,
            lambda: plugin.try_example(example_id, recipe_id),
            _("Trying example '%s'") % example_id,
            recipe_id,
        )

    def _open_example(self, plugin: PluginBase, example_id: str) -> None:
        """Open an example that is not designed for a specific recipe."""
        self._run_command(
            plugin,
            lambda: plugin.launch_example(example_id),
            _("Opening example '%s'") % example_id,
        )

    def _launch_tool(self, plugin: PluginBase, tool_id: str) -> None:
        """Open a plugin-owned tool."""
        self._run_command(
            plugin,
            lambda: plugin.launch_tool(tool_id),
            _("Opening tool '%s'") % tool_id,
        )

    @staticmethod
    def _open_documentation(plugin: PluginBase) -> None:
        """Open the plugin's declared documentation URL."""
        documentation_url = plugin.info.documentation_url
        if documentation_url is not None:
            webbrowser.open(documentation_url)
