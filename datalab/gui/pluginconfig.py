# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin Configuration Dialog
----------------------------

Dialog for managing plugin enable/disable state and viewing plugin information.
"""

from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING

from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.plugins import PluginRegistry

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.plugins import FailedPluginInfo, PluginBase


def _apply_palette_color(widget: QW.QWidget, color: QG.QColor) -> None:
    """Apply a foreground color to a widget via its palette (theme-safe).

    Args:
        widget: Target widget
        color: Foreground color to apply
    """
    palette = widget.palette()
    palette.setColor(QG.QPalette.WindowText, color)
    widget.setPalette(palette)


def _apply_subdued_color(widget: QW.QWidget) -> None:
    """Apply a subdued/secondary text color that works in both light and dark themes.

    Args:
        widget: Target widget
    """
    app_palette = QW.QApplication.instance().palette()
    text_color = app_palette.color(QG.QPalette.Text)
    # Blend the text color towards mid-tone for a subdued effect
    subdued = QG.QColor(text_color)
    subdued.setAlpha(150)
    palette = widget.palette()
    palette.setColor(QG.QPalette.WindowText, subdued)
    widget.setPalette(palette)


class PluginState:
    """Plugin state enumeration"""

    ENABLED = "enabled"  # Plugin is active (green)
    DISABLED = "disabled"  # Plugin is disabled (red)
    ERROR = "error"  # Plugin failed to load (gray)


class PluginInfoWidget(QW.QWidget):
    """Widget displaying information for a single plugin"""

    def __init__(
        self,
        plugin_class: type[PluginBase],
        enabled: bool,
        state: str,
        parent: QW.QWidget = None,
    ):
        """Initialize plugin info widget

        Args:
            plugin_class: Plugin class (not instance)
            enabled: Whether plugin is enabled in config
            state: Current state (enabled/disabled/error)
            parent: Parent widget
        """
        super().__init__(parent)
        self.plugin_class = plugin_class
        self.initial_enabled = enabled
        self.state = state

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        # Top row: checkbox, name, version, status indicator
        top_layout = QW.QHBoxLayout()

        # Checkbox
        self.checkbox = QW.QCheckBox()
        self.checkbox.setChecked(enabled)
        top_layout.addWidget(self.checkbox)

        # Plugin name
        name_label = QW.QLabel(self.plugin_class.PLUGIN_INFO.name)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        top_layout.addWidget(name_label)

        # Version
        version_label = QW.QLabel(f"v{self.plugin_class.PLUGIN_INFO.version}")
        _apply_subdued_color(version_label)
        top_layout.addWidget(version_label)

        # Status indicator
        status_label = QW.QLabel()
        status_font = status_label.font()
        status_font.setBold(True)
        status_label.setFont(status_font)
        if state == PluginState.ENABLED:
            status_label.setText("\u2713 " + _("Active"))
            _apply_palette_color(status_label, QG.QColor("#2ecc71"))
        elif state == PluginState.DISABLED:
            status_label.setText("\u2717 " + _("Disabled"))
            _apply_palette_color(status_label, QG.QColor("#e74c3c"))
        else:  # ERROR
            status_label.setText("\u2717 " + _("Error"))
            _apply_subdued_color(status_label)
        top_layout.addWidget(status_label)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Description area
        description = self.plugin_class.PLUGIN_INFO.description or _(
            "No description available"
        )
        is_long = len(description) > 100

        desc_container = QW.QWidget()
        desc_container_layout = QW.QVBoxLayout()
        desc_container_layout.setContentsMargins(20, 0, 0, 0)
        desc_container.setLayout(desc_container_layout)

        # Full description label (always present)
        self.desc_label = QW.QLabel(description)
        self.desc_label.setWordWrap(True)
        _apply_subdued_color(self.desc_label)

        if is_long:
            # Expandable scroll area for long descriptions
            self._desc_scroll = QW.QScrollArea()
            self._desc_scroll.setWidgetResizable(True)
            self._desc_scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            self._desc_scroll.setWidget(self.desc_label)
            self._desc_scroll.setMaximumHeight(60)
            self._desc_scroll.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }"
            )
            desc_container_layout.addWidget(self._desc_scroll)

            # Toggle expand/collapse button
            self._expanded = False
            self._toggle_btn = QW.QPushButton("\u25bc " + _("Show more"))
            self._toggle_btn.setFlat(True)
            self._toggle_btn.setCursor(QC.Qt.PointingHandCursor)
            link_color = QW.QApplication.instance().palette().color(QG.QPalette.Link)
            self._toggle_btn.setStyleSheet(
                f"QPushButton {{ color: {link_color.name()}; border: none; "
                "text-align: left; padding: 0; } "
                "QPushButton:hover { text-decoration: underline; }"
            )
            toggle_font = self._toggle_btn.font()
            toggle_font.setPointSize(toggle_font.pointSize() - 1)
            self._toggle_btn.setFont(toggle_font)
            self._toggle_btn.clicked.connect(self._toggle_description)
            desc_container_layout.addWidget(self._toggle_btn)
        else:
            desc_container_layout.addWidget(self.desc_label)

        layout.addWidget(desc_container)

        # Separator line
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        layout.addWidget(line)

    def _toggle_description(self):
        """Toggle between collapsed and expanded description"""
        self._expanded = not self._expanded
        if self._expanded:
            self._desc_scroll.setMaximumHeight(150)
            self._toggle_btn.setText("\u25b2 " + _("Show less"))
        else:
            self._desc_scroll.setMaximumHeight(60)
            self._toggle_btn.setText("\u25bc " + _("Show more"))

    def is_enabled(self) -> bool:
        """Check if plugin is enabled via checkbox

        Returns:
            True if checkbox is checked
        """
        return self.checkbox.isChecked()

    def has_changed(self) -> bool:
        """Check if enabled state has changed

        Returns:
            True if current state differs from initial state
        """
        return self.is_enabled() != self.initial_enabled


class FailedPluginInfoWidget(QW.QWidget):
    """Widget displaying information for a plugin that failed to load"""

    def __init__(
        self,
        failed_info: FailedPluginInfo,
        parent: QW.QWidget = None,
    ):
        """Initialize failed plugin info widget

        Args:
            failed_info: Structured information about the failed plugin
            parent: Parent widget
        """
        super().__init__(parent)

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        # Top row: disabled checkbox, file name, error status
        top_layout = QW.QHBoxLayout()

        # Disabled checkbox (always unchecked, non-interactive)
        checkbox = QW.QCheckBox()
        checkbox.setChecked(False)
        checkbox.setEnabled(False)
        top_layout.addWidget(checkbox)

        # Plugin file name (bold, grayed out)
        file_name = osp.basename(failed_info.name)
        name_label = QW.QLabel(file_name)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        _apply_subdued_color(name_label)
        top_layout.addWidget(name_label)

        # Status indicator (gray "Import error")
        status_label = QW.QLabel()
        status_font = status_label.font()
        status_font.setBold(True)
        status_label.setFont(status_font)
        status_label.setText("\u26a0 " + _("Import error"))
        _apply_subdued_color(status_label)
        top_layout.addWidget(status_label)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Description area: file path + traceback
        desc_container = QW.QWidget()
        desc_container_layout = QW.QVBoxLayout()
        desc_container_layout.setContentsMargins(20, 0, 0, 0)
        desc_container.setLayout(desc_container_layout)

        description = failed_info.filepath
        if failed_info.traceback:
            description += "\n\n" + failed_info.traceback.strip()

        # Full description label (always in expandable scroll area)
        desc_label = QW.QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        _apply_subdued_color(desc_label)

        # Use monospace font for the traceback
        mono_font = QG.QFont("Consolas", desc_label.font().pointSize() - 1)
        desc_label.setFont(mono_font)

        desc_scroll = QW.QScrollArea()
        desc_scroll.setWidgetResizable(True)
        desc_scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAsNeeded)
        desc_scroll.setWidget(desc_label)
        desc_scroll.setMaximumHeight(60)
        desc_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        desc_container_layout.addWidget(desc_scroll)

        # Toggle expand/collapse button
        self._expanded = False
        self._desc_scroll = desc_scroll
        self._toggle_btn = QW.QPushButton("\u25bc " + _("Show more"))
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCursor(QC.Qt.PointingHandCursor)
        link_color = QW.QApplication.instance().palette().color(QG.QPalette.Link)
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{ color: {link_color.name()}; border: none; "
            "text-align: left; padding: 0; } "
            "QPushButton:hover { text-decoration: underline; }"
        )
        toggle_font = self._toggle_btn.font()
        toggle_font.setPointSize(toggle_font.pointSize() - 1)
        self._toggle_btn.setFont(toggle_font)
        self._toggle_btn.clicked.connect(self._toggle_description)
        desc_container_layout.addWidget(self._toggle_btn)

        layout.addWidget(desc_container)

        # Separator line
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        layout.addWidget(line)

    def _toggle_description(self):
        """Toggle between collapsed and expanded description"""
        self._expanded = not self._expanded
        if self._expanded:
            self._desc_scroll.setMaximumHeight(200)
            self._toggle_btn.setText("\u25b2 " + _("Show less"))
        else:
            self._desc_scroll.setMaximumHeight(60)
            self._toggle_btn.setText("\u25bc " + _("Show more"))


class PluginConfigDialog(QW.QDialog):
    """Dialog for configuring plugins"""

    def __init__(self, parent: DLMainWindow):
        """Initialize plugin configuration dialog

        Args:
            parent: Main window instance
        """
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.main = parent
        self.plugin_widgets: list[PluginInfoWidget] = []

        self.setWindowTitle(_("Plugin Configuration"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # Main layout
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        # Title
        title_label = QW.QLabel(_("Manage Plugins"))
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title_label.setFont(title_font)
        # Ensure title color follows the system palette (fixes dark mode)
        _apply_palette_color(
            title_label,
            QW.QApplication.instance().palette().color(QG.QPalette.WindowText),
        )
        layout.addWidget(title_label)

        # Info text
        info_label = QW.QLabel(
            _(
                "Enable or disable plugins. Changes will be applied "
                "after clicking OK and reloading plugins."
            )
        )
        info_label.setWordWrap(True)
        _apply_subdued_color(info_label)
        layout.addWidget(info_label)

        # Select All / Deselect All buttons
        select_layout = QW.QHBoxLayout()
        self.select_all_btn = QW.QPushButton(_("Select All"))
        self.deselect_all_btn = QW.QPushButton(_("Deselect All"))
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.deselect_all_btn)
        select_layout.addStretch()
        layout.addLayout(select_layout)

        # Scroll area for plugins
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)

        # Container widget for plugins
        container = QW.QWidget()
        self.plugins_layout = QW.QVBoxLayout()
        self.plugins_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(self.plugins_layout)
        scroll.setWidget(container)

        layout.addWidget(scroll, 1)

        # Populate plugins
        self.populate_plugins()

        # Button box
        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_all(self):
        """Check all plugin checkboxes"""
        for widget in self.plugin_widgets:
            widget.checkbox.setChecked(True)

    def _deselect_all(self):
        """Uncheck all plugin checkboxes"""
        for widget in self.plugin_widgets:
            widget.checkbox.setChecked(False)

    def populate_plugins(self):
        """Populate the dialog with all discovered plugins"""
        # Get all discovered plugin classes (not just registered instances)
        plugin_classes = PluginRegistry.get_plugin_classes()

        # Get currently registered plugin names (active plugins)
        registered_names = {p.info.name for p in PluginRegistry.get_plugins()}

        # Get enabled plugins from config
        # None = all enabled (default), [] = none enabled, list = specific plugins
        enabled_plugins = Conf.main.plugins_enabled_list.get(None)

        for plugin_class in plugin_classes:
            plugin_name = plugin_class.PLUGIN_INFO.name

            # Determine if plugin is enabled in config
            # If None, all plugins are enabled by default
            if enabled_plugins is None:
                enabled = True
            else:
                enabled = plugin_name in enabled_plugins

            # Determine current state (is it actually loaded/active?)
            state = (
                PluginState.ENABLED
                if plugin_name in registered_names
                else PluginState.DISABLED
            )

            # Create widget for this plugin
            widget = PluginInfoWidget(plugin_class, enabled, state)
            self.plugin_widgets.append(widget)
            self.plugins_layout.addWidget(widget)

        # Add failed plugins at the end (grayed out, non-interactive)
        failed_plugins = PluginRegistry.get_failed_plugins()
        if failed_plugins:
            for failed_info in failed_plugins:
                failed_widget = FailedPluginInfoWidget(failed_info)
                self.plugins_layout.addWidget(failed_widget)

        # Add stretch at the end
        self.plugins_layout.addStretch()

    def accept(self):
        """Apply changes and close dialog"""
        # Check if any changes were made
        changes_made = any(widget.has_changed() for widget in self.plugin_widgets)

        if not changes_made:
            super().accept()
            return

        # Collect enabled plugin names
        enabled_plugins = [
            widget.plugin_class.PLUGIN_INFO.name
            for widget in self.plugin_widgets
            if widget.is_enabled()
        ]

        # Save to configuration
        Conf.main.plugins_enabled_list.set(enabled_plugins)

        # Inform user that reload is needed
        reply = QW.QMessageBox.question(
            self,
            _("Reload Plugins"),
            _(
                "Plugin configuration has been saved. "
                "Do you want to reload plugins now to apply changes?"
            ),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.Yes,
        )

        if reply == QW.QMessageBox.Yes:
            # Reload plugins
            self.main.reload_plugins()

        super().accept()
