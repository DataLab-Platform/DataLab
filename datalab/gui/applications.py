# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Read-only catalog of application plugins and their declared workflows."""

from __future__ import annotations

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

    def __init__(self, plugin: PluginBase, parent: QW.QWidget | None = None):
        super().__init__(parent)
        self.plugin = plugin
        self.recipe_list = QW.QListWidget()
        self.example_list = QW.QListWidget()

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
        widget.setSelectionMode(QW.QAbstractItemView.NoSelection)
        widget.setFocusPolicy(QC.Qt.NoFocus)
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
            self.application_pages.append(page)
            self.application_stack.addWidget(page)
        self.application_list.setCurrentRow(0)
