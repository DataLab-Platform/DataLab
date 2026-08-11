# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Catalog of application plugins and their declared workflows."""

from __future__ import annotations

import webbrowser
from math import ceil
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.plugins import PluginCapability, PluginRegistry
from datalab.utils.qthelpers import qt_handle_error_message
from datalab.widgets.expandabletext import apply_subdued_color

if TYPE_CHECKING:
    from datalab.plugins import PluginBase


__all__ = ["ApplicationPage", "ApplicationsDialog", "get_application_plugins"]


CATALOG_TITLE_ROLE = QC.Qt.UserRole + 1
CATALOG_DESCRIPTION_ROLE = QC.Qt.UserRole + 2
CATALOG_VERSION_ROLE = QC.Qt.UserRole + 3


class CatalogItemDelegate(QW.QStyledItemDelegate):
    """Render a catalog entry with a title and a subdued description."""

    HORIZONTAL_MARGIN = 8
    VERTICAL_MARGIN = 6
    DESCRIPTION_SPACING = 2

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

    def paint(
        self,
        painter: QG.QPainter,
        option: QW.QStyleOptionViewItem,
        index: QC.QModelIndex,
    ) -> None:
        """Paint the native item background and the formatted catalog text."""
        styled_option = QW.QStyleOptionViewItem(option)
        self.initStyleOption(styled_option, index)
        styled_option.text = ""
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
        """Return the height required by the wrapped catalog text."""
        width = self._content_width(option)
        document = self._document(option, index, width)
        height = ceil(document.size().height()) + 2 * self.VERTICAL_MARGIN
        return QC.QSize(width + 2 * self.HORIZONTAL_MARGIN, height)


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


class ApplicationPage(QW.QWidget):
    """Display one application plugin's recipes and packaged examples."""

    start_requested = QC.Signal(object, str)
    open_example_requested = QC.Signal(object, str)
    documentation_requested = QC.Signal(object)

    def __init__(self, plugin: PluginBase, parent: QW.QWidget | None = None):
        super().__init__(parent)
        self.plugin = plugin
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
        title = QW.QLabel(self.plugin.info.name)
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 3)
        title.setFont(font)
        layout.addWidget(title)
        layout.addStretch()
        return layout

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

    def __init__(self, parent: QW.QWidget | None = None):
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowModality(QC.Qt.NonModal)
        self.setModal(False)
        self.application_list = QW.QListWidget()
        self.application_stack = QW.QStackedWidget()
        self.application_pages: list[ApplicationPage] = []

        self.setWindowTitle(_("Applications"))
        self.setWindowIcon(get_icon("libre-gui-plugin.svg"))
        self.setMinimumSize(800, 540)
        self.application_list.setMinimumWidth(260)
        self.application_list.setMaximumWidth(360)
        _configure_catalog_list(self.application_list)

        layout = QW.QVBoxLayout(self)
        splitter = QW.QSplitter(QC.Qt.Horizontal)
        splitter.addWidget(self.application_list)
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
            item = QW.QListWidgetItem(plugin.info.name)
            item.setData(QC.Qt.UserRole, plugin.plugin_id)
            _set_catalog_data(item, plugin.info.name, plugin.info.description)
            self.application_list.addItem(item)
            page = ApplicationPage(plugin)
            page.start_requested.connect(self._start_recipe)
            page.open_example_requested.connect(self._open_example)
            page.documentation_requested.connect(self._open_documentation)
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
        self.application_list.setCurrentRow(0)

    def _start_recipe(self, plugin: PluginBase, recipe_id: str) -> None:
        """Delegate to a plugin-owned recipe launcher."""
        try:
            plugin.launch_recipe(recipe_id)
        except Exception as exc:  # pylint: disable=broad-except
            # Plugin-owned launchers are third-party code: never crash the app
            qt_handle_error_message(
                self.parent() or self, exc, _("Starting analysis '%s'") % recipe_id
            )

    def _open_example(self, plugin: PluginBase, example_id: str) -> None:
        """Delegate packaged-example opening."""
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
