# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Catalog of application plugins and their declared workflows."""

from __future__ import annotations

import html
import webbrowser
from collections.abc import Callable, Iterable
from math import ceil
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.gui.plugins.recipe_inputs import (
    format_diagnostic,
    format_inputs_html,
    format_readiness,
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
    "ApplicationPage",
    "ApplicationsDialog",
    "RecipeCard",
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
        icon: ``package:path`` resource (SVG or bitmap), DataLab icon file name,
         or None

    Returns:
        Plugin icon, or the generic plugin icon if ``icon`` is None or cannot be
         loaded
    """
    if icon is not None:
        with try_or_log_error(f"Loading plugin icon {icon!r}"):
            if ":" not in icon:
                return get_icon(icon)
            resource = resolve_package_resource(icon, "Plugin icon")
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


def _configure_catalog_list(widget: QW.QListWidget) -> None:
    """Configure a descriptor list as a read-only, multiline catalog."""
    widget.setSelectionMode(QW.QAbstractItemView.SingleSelection)
    widget.setAlternatingRowColors(True)
    widget.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
    widget.setWordWrap(True)
    widget.setResizeMode(QW.QListView.Adjust)
    widget.setSpacing(2)
    widget.setItemDelegate(CatalogItemDelegate(widget))


def _set_catalog_data(
    item: QW.QListWidgetItem,
    title: str,
    description: str,
    version: str = "",
) -> None:
    """Store structured display metadata on a catalog item."""
    item.setData(CATALOG_TITLE_ROLE, title)
    item.setData(CATALOG_DESCRIPTION_ROLE, description)
    item.setData(CATALOG_VERSION_ROLE, version)


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


class RecipeCard(QW.QWidget):
    """Present one recipe: expected inputs, readiness, and dedicated examples."""

    #: Emitted with the recipe ID when running the recipe on the selection
    run_requested = QC.Signal(str)
    #: Emitted with the example ID and the recipe ID when trying an example
    try_example_requested = QC.Signal(str, str)
    #: Emitted when the shown readiness changes
    readiness_changed = QC.Signal()

    READINESS_COLORS = {
        RecipeReadinessStatus.READY: "#2e7d32",
        RecipeReadinessStatus.WARNINGS: "#b26a00",
        RecipeReadinessStatus.NEEDS_ASSIGNMENT: "#1565c0",
        RecipeReadinessStatus.NOT_READY: "#c62828",
        RecipeReadinessStatus.NO_INPUT: "#808080",
    }

    def __init__(
        self,
        recipe: RecipeDescriptor,
        examples: Iterable[PluginExample],
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.readiness: RecipeReadiness | None = None
        self.status_color = self.READINESS_COLORS[RecipeReadinessStatus.NO_INPUT]
        self.status_summary = ""
        layout = QW.QVBoxLayout(self)

        version = QW.QLabel(_("Version: %s") % recipe.version)
        apply_subdued_color(version)
        layout.addWidget(version)
        if recipe.description:
            description = QW.QLabel(recipe.description)
            description.setWordWrap(True)
            layout.addWidget(description)

        inputs_title = QW.QLabel(_("Expected inputs"))
        apply_subdued_color(inputs_title)
        layout.addWidget(inputs_title)
        self.inputs_label = QW.QLabel(format_inputs_html(recipe))
        self.inputs_label.setTextFormat(QC.Qt.RichText)
        self.inputs_label.setWordWrap(True)
        self.inputs_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self.inputs_label.setContentsMargins(12, 0, 0, 0)
        layout.addWidget(self.inputs_label)

        self.readiness_label = QW.QLabel()
        self.readiness_label.setTextFormat(QC.Qt.RichText)
        self.readiness_label.setWordWrap(True)
        self.run_button = _command_button("analysis.svg", _("Run on selection..."))
        self.run_button.clicked.connect(
            lambda _checked=False: self.run_requested.emit(recipe.recipe_id)
        )
        run_layout = QW.QHBoxLayout()
        run_layout.addWidget(self.readiness_label, 1)
        run_layout.addWidget(self.run_button, 0, QC.Qt.AlignTop)
        layout.addLayout(run_layout)

        self.example_buttons: dict[str, QW.QPushButton] = {}
        examples = tuple(examples)
        if examples:
            examples_title = QW.QLabel(_("Examples designed for this method"))
            apply_subdued_color(examples_title)
            layout.addWidget(examples_title)
        for example in examples:
            button = _command_button("play_demo.svg", _("Try with this example"))
            button.clicked.connect(
                lambda _checked=False, example_id=example.id: (
                    self.try_example_requested.emit(example_id, recipe.recipe_id)
                )
            )
            self.example_buttons[example.id] = button
            layout.addLayout(_entry_layout(example.title, example.description, button))
        layout.addStretch()

    def set_readiness(
        self, readiness: RecipeReadiness | None, error: str | None = None
    ) -> None:
        """Show whether the recipe can run on the current selection.

        Args:
            readiness: assessment of the current selection, or None on failure
            error: message explaining why the selection could not be assessed
        """
        self.readiness = readiness
        if readiness is None:
            color = self.READINESS_COLORS[RecipeReadinessStatus.NOT_READY]
            summary = _("The current selection could not be assessed")
            reasons = [error] if error else []
        else:
            color = self.READINESS_COLORS[readiness.status]
            summary, reasons = format_readiness(readiness, self.recipe)
        text = f'<span style="color:{color}">●</span> <b>{html.escape(summary)}</b>'
        if reasons:
            text += "<br>" + "<br>".join(
                f"&nbsp;&nbsp;• {html.escape(reason)}" for reason in reasons
            )
        self.readiness_label.setText(text)
        self.status_color = color
        self.status_summary = summary
        self.readiness_changed.emit()


class ApplicationPage(QW.QWidget):
    """Display one application plugin's methods, tools, and datasets."""

    #: Emitted with the plugin and a recipe ID to run it on the selection
    start_requested = QC.Signal(object, str)
    #: Emitted with the plugin, an example ID and a recipe ID to try an example
    try_example_requested = QC.Signal(object, str, str)
    #: Emitted with the plugin and the ID of an example without recipe
    open_example_requested = QC.Signal(object, str)
    #: Emitted with the plugin and a tool ID
    tool_requested = QC.Signal(object, str)
    documentation_requested = QC.Signal(object)
    welcome_preferences_changed = QC.Signal()

    HEADER_ICON_SIZE = 48

    def __init__(self, plugin: PluginBase, parent: QW.QWidget | None = None):
        super().__init__(parent)
        self.plugin = plugin
        self.icon = get_plugin_icon(plugin.info.icon)
        self.icon_label = QW.QLabel()
        self.show_on_welcome_checkbox = QW.QCheckBox(_("Show on welcome page"))
        self.pin_on_welcome_checkbox = QW.QCheckBox(_("Pin to the welcome page"))
        self.recipe_cards: dict[str, RecipeCard] = {}
        self.tool_buttons: dict[str, QW.QPushButton] = {}
        self.dataset_buttons: dict[str, QW.QPushButton] = {}
        self.toolbox = QW.QToolBox()
        self.documentation_button = _command_button(
            "libre-gui-help.svg", _("Documentation")
        )
        self.status_label = QW.QLabel()
        self.status_label.setTextFormat(QC.Qt.RichText)
        self.status_label.setWordWrap(True)
        self.status_label.hide()

        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.addLayout(self._create_header())

        metadata = QW.QLabel(
            _("Plugin ID: %s") % plugin.plugin_id
            + "\n"
            + _("Version: %s") % plugin.info.version
        )
        metadata.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        apply_subdued_color(metadata)
        layout.addWidget(metadata)
        layout.addLayout(self._create_welcome_layout())

        self._add_methods(layout)
        self._add_tools()
        self._add_datasets()
        layout.addWidget(self.toolbox, 1)
        if not self.toolbox.count():
            self.toolbox.hide()
            layout.addStretch()
        layout.addWidget(self.status_label)

        bottom_layout = QW.QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.documentation_button)
        layout.addLayout(bottom_layout)
        documentation_url = plugin.info.documentation_url
        self.documentation_button.setEnabled(documentation_url is not None)
        self.documentation_button.setToolTip(documentation_url or "")
        self.documentation_button.clicked.connect(
            lambda _checked=False: self.documentation_requested.emit(self.plugin)
        )

    def _create_header(self) -> QW.QHBoxLayout:
        """Create the application title row."""
        layout = QW.QHBoxLayout()
        size = self.HEADER_ICON_SIZE
        self.icon_label.setPixmap(self.icon.pixmap(size, size))
        layout.addWidget(self.icon_label)
        title = QW.QLabel(self.plugin.info.name)
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 3)
        title.setFont(font)
        layout.addWidget(title)
        layout.addStretch()
        return layout

    def _create_welcome_layout(self) -> QW.QHBoxLayout:
        """Create the welcome page preference row."""
        layout = QW.QHBoxLayout()
        layout.addWidget(self.show_on_welcome_checkbox)
        layout.addWidget(self.pin_on_welcome_checkbox)
        layout.addStretch()
        self.sync_welcome_preferences()
        self.show_on_welcome_checkbox.toggled.connect(self._set_shown_on_welcome)
        self.pin_on_welcome_checkbox.toggled.connect(self._set_pinned_on_welcome)
        return layout

    def sync_welcome_preferences(self) -> None:
        """Update the welcome page check boxes from the user preferences."""
        plugin_id = self.plugin.plugin_id
        for checkbox, checked in (
            (
                self.show_on_welcome_checkbox,
                plugin_id not in Conf.welcome_hidden_applications.get(),
            ),
            (
                self.pin_on_welcome_checkbox,
                plugin_id in Conf.welcome_pinned_applications.get(),
            ),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _set_shown_on_welcome(self, checked: bool) -> None:
        """Show the application on the welcome page, or hide it."""
        set_application_hidden(self.plugin.plugin_id, not checked)
        self.sync_welcome_preferences()
        self.welcome_preferences_changed.emit()

    def _set_pinned_on_welcome(self, checked: bool) -> None:
        """Pin the application to the welcome page, or unpin it."""
        set_application_pinned(self.plugin.plugin_id, checked)
        self.sync_welcome_preferences()
        self.welcome_preferences_changed.emit()

    def showEvent(self, event: QG.QShowEvent) -> None:  # pylint: disable=invalid-name
        """Reflect preferences changed from the welcome page."""
        self.sync_welcome_preferences()
        super().showEvent(event)

    def _add_methods(self, layout: QW.QVBoxLayout) -> None:
        """Add one item per recipe, with the examples designed for it."""
        recipes = self.plugin.get_recipes()
        examples = self.plugin.get_examples()
        for recipe in recipes:
            card = RecipeCard(
                recipe,
                (
                    example
                    for example in examples
                    if recipe.recipe_id in example.recipe_ids
                ),
            )
            card.run_requested.connect(
                lambda recipe_id: self.start_requested.emit(self.plugin, recipe_id)
            )
            card.try_example_requested.connect(
                lambda example_id, recipe_id: self.try_example_requested.emit(
                    self.plugin, example_id, recipe_id
                )
            )
            card.readiness_changed.connect(
                lambda card=card: self._show_readiness_of(card)
            )
            self.recipe_cards[recipe.recipe_id] = card
            self.toolbox.addItem(card, _status_icon(card.status_color), recipe.title)
        if not recipes:
            label = QW.QLabel(_("No methods declared"))
            apply_subdued_color(label)
            layout.addWidget(label)

    def _show_readiness_of(self, card: RecipeCard) -> None:
        """Reflect the readiness of a recipe on its toolbox item."""
        index = self.toolbox.indexOf(card)
        self.toolbox.setItemIcon(index, _status_icon(card.status_color))
        self.toolbox.setItemToolTip(index, card.status_summary)

    def _add_entries_item(
        self, title: str, icon_name: str, layouts: list[QW.QHBoxLayout]
    ) -> None:
        """Add a toolbox item listing entries, if any."""
        if not layouts:
            return
        page = QW.QWidget()
        page_layout = QW.QVBoxLayout(page)
        for entry_layout in layouts:
            page_layout.addLayout(entry_layout)
        page_layout.addStretch()
        self.toolbox.addItem(page, get_icon(icon_name), title)

    def _add_tools(self) -> None:
        """Add the plugin-owned tools, if any."""
        tools = self.plugin.get_tools()
        layouts = []
        for tool in tools:
            button = _command_button("libre-toolbox.svg", _("Open tool"))
            button.clicked.connect(
                lambda _checked=False, tool_id=tool.id: self.tool_requested.emit(
                    self.plugin, tool_id
                )
            )
            self.tool_buttons[tool.id] = button
            layouts.append(
                _entry_layout(
                    tool.title, tool.description, button, get_plugin_icon(tool.icon)
                )
            )
        self._add_entries_item(
            _("Tools (%d)") % len(tools), "libre-toolbox.svg", layouts
        )

    def _add_datasets(self) -> None:
        """Add the examples that are not designed for a specific recipe."""
        datasets = [
            example for example in self.plugin.get_examples() if not example.recipe_ids
        ]
        layouts = []
        for example in datasets:
            button = _command_button("io/fileopen_h5.svg", _("Open dataset"))
            button.clicked.connect(
                lambda _checked=False, example_id=example.id: (
                    self.open_example_requested.emit(self.plugin, example_id)
                )
            )
            self.dataset_buttons[example.id] = button
            layouts.append(_entry_layout(example.title, example.description, button))
        self._add_entries_item(
            _("Datasets (%d)") % len(datasets), "io/fileopen_h5.svg", layouts
        )

    def refresh_readiness(self) -> None:
        """Assess every recipe and tool on the current selection."""
        for recipe_id, card in self.recipe_cards.items():
            try:
                card.set_readiness(self.plugin.assess_recipe(recipe_id))
            except Exception as exc:  # pylint: disable=broad-except
                # Binding suggestions are third-party code: keep the catalog usable
                card.set_readiness(None, str(exc))
        for tool_id, button in self.tool_buttons.items():
            issue = self.plugin.assess_tool(tool_id)
            button.setEnabled(issue is None)
            button.setToolTip(issue or "")

    def show_outcome(self, outcome: RecipeOutcome) -> None:
        """Show the number of created objects and the diagnostics of a run."""
        lines = [
            html.escape(_("Created %d objects") % len(outcome.objects)),
            *(
                f"• {html.escape(format_diagnostic(diagnostic))}"
                for diagnostic in outcome.diagnostics
            ),
        ]
        self.status_label.setText("<br>".join(lines))
        self.status_label.show()


class ApplicationsDialog(QW.QDialog):
    """Browse active plugins that expose the application capability."""

    #: Emitted when an application is shown, hidden, pinned or used
    SIG_WELCOME_PREFERENCES_CHANGED = QC.Signal()

    #: Minimum window width when the application list is shown
    MINIMUM_WIDTH = 860
    CATALOG_MINIMUM_WIDTH = 260
    CATALOG_DEFAULT_WIDTH = 280

    def __init__(self, parent: QW.QWidget | None = None):
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowModality(QC.Qt.NonModal)
        self.setModal(False)
        self.search_edit = QW.QLineEdit()
        self.application_list = QW.QListWidget()
        self.application_stack = QW.QStackedWidget()
        self.application_pages: list[ApplicationPage] = []
        self.catalog_widget = QW.QWidget()
        self.list_toggle_button = QW.QToolButton()
        self.__splitter = QW.QSplitter(QC.Qt.Horizontal)
        self.__catalog_width = self.CATALOG_DEFAULT_WIDTH

        self.setWindowTitle(_("Applications"))
        self.setWindowIcon(get_icon("libre-gui-plugin.svg"))
        self.setMinimumSize(self.MINIMUM_WIDTH, 600)
        self.__readiness_timer = QC.QTimer(self)
        self.__readiness_timer.setSingleShot(True)
        self.__readiness_timer.setInterval(150)
        self.__readiness_timer.timeout.connect(self.refresh_readiness)
        self.search_edit.setPlaceholderText(_("Search applications..."))
        self.search_edit.setClearButtonEnabled(True)
        _configure_catalog_list(self.application_list)

        catalog = self.catalog_widget
        catalog.setMinimumWidth(self.CATALOG_MINIMUM_WIDTH)
        catalog.setMaximumWidth(360)
        catalog_layout = QW.QVBoxLayout(catalog)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.addWidget(self.search_edit)
        catalog_layout.addWidget(self.application_list)

        toggle = self.list_toggle_button
        toggle.setFixedWidth(16)
        toggle.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Expanding)
        toggle.setStyleSheet(
            "QToolButton { border: none; border-left: 1px solid palette(midlight);"
            " border-right: 1px solid palette(midlight);"
            " background: palette(alternate-base); }"
            "QToolButton:hover { background: palette(midlight); }"
        )
        font = toggle.font()
        font.setPointSize(font.pointSize() + 8)
        toggle.setFont(font)
        toggle.clicked.connect(self.toggle_application_list)
        pages = QW.QWidget()
        pages_layout = QW.QHBoxLayout(pages)
        pages_layout.setContentsMargins(0, 0, 0, 0)
        pages_layout.setSpacing(0)
        pages_layout.addWidget(toggle)
        pages_layout.addWidget(self.application_stack, 1)

        layout = QW.QVBoxLayout(self)
        splitter = self.__splitter
        splitter.addWidget(catalog)
        splitter.addWidget(pages)
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self.CATALOG_DEFAULT_WIDTH, 580])
        layout.addWidget(splitter, 1)

        button_box = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.application_list.currentRowChanged.connect(
            self.application_stack.setCurrentIndex
        )
        self.application_list.currentRowChanged.connect(self.schedule_readiness_update)
        self.search_edit.textChanged.connect(self.filter_applications)
        self.set_application_list_visible(not Conf.applications_list_collapsed.get())
        self.refresh()

    def is_application_list_visible(self) -> bool:
        """Return whether the application list is shown."""
        return not self.catalog_widget.isHidden()

    def toggle_application_list(self) -> None:
        """Hide the application list if it is shown, show it otherwise."""
        self.set_application_list_visible(not self.is_application_list_visible())

    def set_application_list_visible(self, visible: bool) -> None:
        """Show or hide the application list, and remember this choice.

        The application page keeps its width: the window shrinks when the list
        is hidden and grows back when the list is shown again.

        Args:
            visible: True to show the application list, False to hide it
        """
        if visible != self.is_application_list_visible():
            handle_width = self.__splitter.handleWidth()
            width = self.width()
            resize = self.isVisible() and not (
                self.isMaximized() or self.isFullScreen()
            )
            if visible:
                self.catalog_widget.show()
                self.setMinimumWidth(self.MINIMUM_WIDTH)
                if resize:
                    self.resize(
                        width + self.__catalog_width + handle_width, self.height()
                    )
                pages_width = self.__splitter.widget(1).width()
                self.__splitter.setSizes([self.__catalog_width, pages_width])
            else:
                if self.isVisible():
                    self.__catalog_width = self.catalog_widget.width()
                self.catalog_widget.hide()
                self.setMinimumWidth(self.MINIMUM_WIDTH - self.CATALOG_MINIMUM_WIDTH)
                if resize:
                    self.resize(
                        width - self.__catalog_width - handle_width, self.height()
                    )
        Conf.applications_list_collapsed.set(not visible)
        if visible:
            text, tooltip = "\u2039", _("Hide the application list")
        else:
            text, tooltip = "\u203a", _("Show the application list")
        self.list_toggle_button.setText(text)
        self.list_toggle_button.setToolTip(tooltip)
        self.list_toggle_button.setAccessibleName(tooltip)

    def refresh(self) -> None:
        """Rebuild the catalog from the currently registered plugins."""
        self.application_list.clear()
        self.application_pages.clear()
        while self.application_stack.count():
            widget = self.application_stack.widget(0)
            self.application_stack.removeWidget(widget)
            widget.deleteLater()

        plugins = get_application_plugins()
        if not plugins:
            label = QW.QLabel(_("No application plugins are currently loaded."))
            label.setAlignment(QC.Qt.AlignCenter)
            apply_subdued_color(label)
            self.application_stack.addWidget(label)
            return

        for plugin in plugins:
            page = ApplicationPage(plugin)
            item = QW.QListWidgetItem(page.icon, plugin.info.name)
            item.setData(QC.Qt.UserRole, plugin.plugin_id)
            _set_catalog_data(item, plugin.info.name, plugin.info.description)
            self.application_list.addItem(item)
            page.start_requested.connect(self._start_recipe)
            page.try_example_requested.connect(self._try_example)
            page.open_example_requested.connect(self._open_example)
            page.tool_requested.connect(self._launch_tool)
            page.documentation_requested.connect(self._open_documentation)
            page.welcome_preferences_changed.connect(
                self.SIG_WELCOME_PREFERENCES_CHANGED.emit
            )
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
        self.application_list.setCurrentRow(0)
        self.filter_applications(self.search_edit.text())
        self.schedule_readiness_update()

    def filter_applications(self, text: str) -> None:
        """Show only the applications matching a search text

        Args:
            text: Words to find in the application names and descriptions
        """
        words = text.casefold().split()
        first_visible = None
        for row in range(self.application_list.count()):
            item = self.application_list.item(row)
            haystack = (
                f"{item.data(CATALOG_TITLE_ROLE)} {item.data(CATALOG_DESCRIPTION_ROLE)}"
            ).casefold()
            item.setHidden(not all(word in haystack for word in words))
            if first_visible is None and not item.isHidden():
                first_visible = item
        current = self.application_list.currentItem()
        if first_visible is not None and (current is None or current.isHidden()):
            self.application_list.setCurrentItem(first_visible)

    def select_plugin(self, plugin_id: str) -> None:
        """Show the page of an application plugin

        Args:
            plugin_id: ID of the application plugin

        Raises:
            KeyError: if no active application plugin has this ID
        """
        for row in range(self.application_list.count()):
            item = self.application_list.item(row)
            if item.data(QC.Qt.UserRole) == plugin_id:
                if item.isHidden():
                    self.search_edit.clear()
                self.application_list.setCurrentRow(row)
                return
        raise KeyError(f"Application plugin {plugin_id!r} not found")

    def schedule_readiness_update(self, *_args: object) -> None:
        """Assess the methods of the shown application after a short delay.

        Connected to selection changes, which may come in bursts.
        """
        if self.isVisible():
            self.__readiness_timer.start()

    def refresh_readiness(self) -> None:
        """Assess the methods of the shown application on the selection."""
        page = self.application_stack.currentWidget()
        if isinstance(page, ApplicationPage):
            page.refresh_readiness()

    def showEvent(self, event: QG.QShowEvent) -> None:  # pylint: disable=invalid-name
        """Assess the methods on the current selection when shown."""
        super().showEvent(event)
        self.refresh_readiness()

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
        self, plugin: PluginBase, command: Callable[[], object], context: str
    ) -> None:
        """Run a plugin command, report its failure, and show its outcome."""
        self._record_use(plugin)
        try:
            result = command()
        except Exception as exc:  # pylint: disable=broad-except
            # Plugin-owned commands are third-party code: never crash the app
            qt_handle_error_message(self.parent() or self, exc, context)
            return
        page = self._page_of(plugin)
        if page is not None and isinstance(result, RecipeOutcome):
            page.show_outcome(result)
        self.schedule_readiness_update()

    def _start_recipe(self, plugin: PluginBase, recipe_id: str) -> None:
        """Run a recipe on the current selection."""
        self._run_command(
            plugin,
            lambda: plugin.launch_recipe(recipe_id),
            _("Starting analysis '%s'") % recipe_id,
        )

    def _try_example(self, plugin: PluginBase, example_id: str, recipe_id: str) -> None:
        """Open an example, then run one of the recipes designed for it."""
        self._run_command(
            plugin,
            lambda: plugin.try_example(example_id, recipe_id),
            _("Trying example '%s'") % example_id,
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
