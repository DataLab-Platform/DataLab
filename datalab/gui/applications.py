# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Catalog of application plugins and their declared workflows."""

from __future__ import annotations

import webbrowser
from math import ceil
from typing import TYPE_CHECKING, Iterable

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.plugin_resources import resolve_package_resource
from datalab.plugins import PluginCapability, PluginRegistry
from datalab.utils.qthelpers import qt_handle_error_message, try_or_log_error
from datalab.widgets.expandabletext import apply_subdued_color

if TYPE_CHECKING:
    from datalab.plugins import PluginBase


__all__ = [
    "ApplicationPage",
    "ApplicationsDialog",
    "get_application_plugins",
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


class ApplicationPage(QW.QWidget):
    """Display one application plugin's recipes and packaged examples."""

    start_requested = QC.Signal(object, str)
    open_example_requested = QC.Signal(object, str)
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
        self.recipe_list = QW.QListWidget()
        self.example_list = QW.QListWidget()
        self.start_button = QW.QPushButton(
            get_icon("analysis.svg"), _("Start analysis")
        )
        self.open_example_button = QW.QPushButton(
            get_icon("io/fileopen_h5.svg"), _("Open example")
        )
        self.documentation_button = QW.QPushButton(
            get_icon("libre-gui-help.svg"), _("Documentation")
        )
        for button in (
            self.start_button,
            self.open_example_button,
            self.documentation_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

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

        layout.addWidget(self._create_recipes_group(), 1)
        layout.addWidget(self._create_examples_group(), 1)
        layout.addLayout(self._create_actions_layout())

        self.recipe_list.currentItemChanged.connect(self._update_action_states)
        self.example_list.currentItemChanged.connect(self._update_action_states)
        self.start_button.clicked.connect(self._request_start)
        self.open_example_button.clicked.connect(self._request_open_example)
        self.documentation_button.clicked.connect(
            lambda: self.documentation_requested.emit(self.plugin)
        )
        if self.recipe_list.count():
            self.recipe_list.setCurrentRow(0)
        if self.example_list.count():
            self.example_list.setCurrentRow(0)
        self._update_action_states()

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

    def _create_recipes_group(self) -> QW.QGroupBox:
        """Create the recipe descriptor section."""
        group = QW.QGroupBox(_("Recipes"))
        layout = QW.QVBoxLayout(group)
        _configure_catalog_list(self.recipe_list)
        recipes = self.plugin.get_recipes()
        for recipe in recipes:
            item = QW.QListWidgetItem(f"{recipe.title}  v{recipe.version}")
            item.setData(QC.Qt.UserRole, recipe.recipe_id)
            _set_catalog_data(
                item,
                recipe.title,
                recipe.description,
                recipe.version,
            )
            self.recipe_list.addItem(item)
        if recipes:
            layout.addWidget(self.recipe_list)
        else:
            label = QW.QLabel(_("No recipes declared"))
            apply_subdued_color(label)
            layout.addWidget(label)
        return group

    def _create_examples_group(self) -> QW.QGroupBox:
        """Create the packaged example section."""
        group = QW.QGroupBox(_("Examples"))
        layout = QW.QVBoxLayout(group)
        _configure_catalog_list(self.example_list)
        examples = self.plugin.get_examples()
        for example in examples:
            item = QW.QListWidgetItem(example.title)
            item.setData(QC.Qt.UserRole, example.id)
            _set_catalog_data(item, example.title, example.description)
            self.example_list.addItem(item)
        if examples:
            layout.addWidget(self.example_list)
        else:
            label = QW.QLabel(_("No examples declared"))
            apply_subdued_color(label)
            layout.addWidget(label)
        return group

    def _create_actions_layout(self) -> QW.QHBoxLayout:
        """Create the application workflow command row."""
        layout = QW.QHBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.open_example_button)
        layout.addStretch()
        layout.addWidget(self.documentation_button)
        return layout

    @staticmethod
    def _current_id(widget: QW.QListWidget) -> str | None:
        """Return the stable identifier stored on the current list item."""
        item = widget.currentItem()
        return None if item is None else item.data(QC.Qt.UserRole)

    def _update_action_states(self, *_args: object) -> None:
        """Enable commands only when their declared target is available."""
        recipe_id = self._current_id(self.recipe_list)
        self.start_button.setEnabled(
            recipe_id is not None and recipe_id in self.plugin.get_recipe_launchers()
        )
        self.open_example_button.setEnabled(
            self._current_id(self.example_list) is not None
        )
        documentation_url = self.plugin.info.documentation_url
        self.documentation_button.setEnabled(documentation_url is not None)
        self.documentation_button.setToolTip(documentation_url or "")

    def _request_start(self) -> None:
        """Request execution of the selected recipe."""
        recipe_id = self._current_id(self.recipe_list)
        if recipe_id is not None:
            self.start_requested.emit(self.plugin, recipe_id)

    def _request_open_example(self) -> None:
        """Request opening of the selected packaged example."""
        example_id = self._current_id(self.example_list)
        if example_id is not None:
            self.open_example_requested.emit(self.plugin, example_id)


class ApplicationsDialog(QW.QDialog):
    """Browse active plugins that expose the application capability."""

    #: Emitted when an application is shown, hidden, pinned or used
    SIG_WELCOME_PREFERENCES_CHANGED = QC.Signal()

    def __init__(self, parent: QW.QWidget | None = None):
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowModality(QC.Qt.NonModal)
        self.setModal(False)
        self.search_edit = QW.QLineEdit()
        self.application_list = QW.QListWidget()
        self.application_stack = QW.QStackedWidget()
        self.application_pages: list[ApplicationPage] = []

        self.setWindowTitle(_("Applications"))
        self.setWindowIcon(get_icon("libre-gui-plugin.svg"))
        self.setMinimumSize(800, 540)
        self.search_edit.setPlaceholderText(_("Search applications..."))
        self.search_edit.setClearButtonEnabled(True)
        _configure_catalog_list(self.application_list)

        catalog = QW.QWidget()
        catalog.setMinimumWidth(260)
        catalog.setMaximumWidth(360)
        catalog_layout = QW.QVBoxLayout(catalog)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.addWidget(self.search_edit)
        catalog_layout.addWidget(self.application_list)

        layout = QW.QVBoxLayout(self)
        splitter = QW.QSplitter(QC.Qt.Horizontal)
        splitter.addWidget(catalog)
        splitter.addWidget(self.application_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 520])
        layout.addWidget(splitter, 1)

        button_box = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.application_list.currentRowChanged.connect(
            self.application_stack.setCurrentIndex
        )
        self.search_edit.textChanged.connect(self.filter_applications)
        self.refresh()

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
            page.open_example_requested.connect(self._open_example)
            page.documentation_requested.connect(self._open_documentation)
            page.welcome_preferences_changed.connect(
                self.SIG_WELCOME_PREFERENCES_CHANGED.emit
            )
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
        self.application_list.setCurrentRow(0)
        self.filter_applications(self.search_edit.text())

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

    def _record_use(self, plugin: PluginBase) -> None:
        """Remember an application use to order the welcome page."""
        record_application_use(plugin.plugin_id)
        self.SIG_WELCOME_PREFERENCES_CHANGED.emit()

    def _start_recipe(self, plugin: PluginBase, recipe_id: str) -> None:
        """Delegate to a plugin-owned recipe launcher."""
        self._record_use(plugin)
        try:
            plugin.launch_recipe(recipe_id)
        except Exception as exc:  # pylint: disable=broad-except
            # Plugin-owned launchers are third-party code: never crash the app
            qt_handle_error_message(
                self.parent() or self, exc, _("Starting analysis '%s'") % recipe_id
            )

    def _open_example(self, plugin: PluginBase, example_id: str) -> None:
        """Delegate packaged-example opening."""
        self._record_use(plugin)
        try:
            plugin.launch_example(example_id)
        except Exception as exc:  # pylint: disable=broad-except
            qt_handle_error_message(
                self.parent() or self, exc, _("Opening example '%s'") % example_id
            )

    @staticmethod
    def _open_documentation(plugin: PluginBase) -> None:
        """Open the plugin's declared documentation URL."""
        documentation_url = plugin.info.documentation_url
        if documentation_url is not None:
            webbrowser.open(documentation_url)
