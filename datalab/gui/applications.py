# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Catalog of application plugins and their declared workflows."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.plugins import PluginCapability, PluginRegistry
from datalab.widgets.expandabletext import apply_subdued_color

if TYPE_CHECKING:
    from datalab.plugins import PluginBase


__all__ = ["ApplicationPage", "ApplicationsDialog", "get_application_plugins"]


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

        if plugin.info.description:
            description = QW.QLabel(plugin.info.description)
            description.setWordWrap(True)
            layout.addWidget(description)

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

    @staticmethod
    def _configure_list(widget: QW.QListWidget) -> None:
        """Configure a descriptor list as a read-only catalog."""
        widget.setSelectionMode(QW.QAbstractItemView.SingleSelection)
        widget.setAlternatingRowColors(True)

    def _create_recipes_group(self) -> QW.QGroupBox:
        """Create the recipe descriptor section."""
        group = QW.QGroupBox(_("Recipes"))
        layout = QW.QVBoxLayout(group)
        self._configure_list(self.recipe_list)
        recipes = self.plugin.get_recipes()
        for recipe in recipes:
            item = QW.QListWidgetItem(f"{recipe.title}  v{recipe.version}")
            item.setData(QC.Qt.UserRole, recipe.recipe_id)
            item.setToolTip(recipe.description)
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
        self._configure_list(self.example_list)
        examples = self.plugin.get_examples()
        for example in examples:
            item = QW.QListWidgetItem(example.title)
            item.setData(QC.Qt.UserRole, example.id)
            item.setToolTip(example.description)
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
        self.application_list = QW.QListWidget()
        self.application_stack = QW.QStackedWidget()
        self.application_pages: list[ApplicationPage] = []

        self.setWindowTitle(_("Applications"))
        self.setWindowIcon(get_icon("libre-gui-plugin.svg"))
        self.setMinimumSize(760, 520)
        self.application_list.setMinimumWidth(220)
        self.application_list.setMaximumWidth(320)

        layout = QW.QVBoxLayout(self)
        splitter = QW.QSplitter(QC.Qt.Horizontal)
        splitter.addWidget(self.application_list)
        splitter.addWidget(self.application_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 520])
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
            if plugin.info.description:
                item.setToolTip(plugin.info.description)
            self.application_list.addItem(item)
            page = ApplicationPage(plugin)
            page.start_requested.connect(self._start_recipe)
            page.open_example_requested.connect(self._open_example)
            page.documentation_requested.connect(self._open_documentation)
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
        self.application_list.setCurrentRow(0)

    def _start_recipe(self, plugin: PluginBase, recipe_id: str) -> None:
        """Close the catalog and delegate to a plugin-owned recipe launcher."""
        self.accept()
        plugin.launch_recipe(recipe_id)

    def _open_example(self, plugin: PluginBase, example_id: str) -> None:
        """Close the catalog and delegate packaged-example opening."""
        self.accept()
        plugin.launch_example(example_id)

    @staticmethod
    def _open_documentation(plugin: PluginBase) -> None:
        """Open the plugin's declared documentation URL."""
        documentation_url = plugin.info.documentation_url
        if documentation_url is not None:
            webbrowser.open(documentation_url)
