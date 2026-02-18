# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin Configuration Dialog
----------------------------

Dialog for managing plugin enable/disable state and viewing plugin information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.plugins import PluginRegistry

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.plugins import PluginBase


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
        name_label = QW.QLabel(f"<b>{self.plugin_class.PLUGIN_INFO.name}</b>")
        top_layout.addWidget(name_label)

        # Version
        version_label = QW.QLabel(f"v{self.plugin_class.PLUGIN_INFO.version}")
        version_label.setStyleSheet("color: gray;")
        top_layout.addWidget(version_label)

        # Status indicator
        status_label = QW.QLabel()
        if state == PluginState.ENABLED:
            status_label.setText("● " + _("Active"))
            status_label.setStyleSheet("color: green; font-weight: bold;")
        elif state == PluginState.DISABLED:
            status_label.setText("● " + _("Disabled"))
            status_label.setStyleSheet("color: red; font-weight: bold;")
        else:  # ERROR
            status_label.setText("● " + _("Error"))
            status_label.setStyleSheet("color: gray; font-weight: bold;")
        top_layout.addWidget(status_label)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Description preview area
        desc_layout = QW.QHBoxLayout()
        desc_layout.setContentsMargins(20, 0, 0, 0)

        # Description text (truncated)
        description = self.plugin_class.PLUGIN_INFO.description or _(
            "No description available"
        )
        preview_text = (
            description[:100] + "..." if len(description) > 100 else description
        )
        self.desc_label = QW.QLabel(preview_text)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #555;")
        desc_layout.addWidget(self.desc_label, 1)

        # "Show more" button if description is long
        if len(description) > 100:
            self.show_more_btn = QW.QPushButton(_("Show more..."))
            self.show_more_btn.setMaximumWidth(100)
            self.show_more_btn.clicked.connect(self.show_full_description)
            desc_layout.addWidget(self.show_more_btn)

        layout.addLayout(desc_layout)

        # Separator line
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        layout.addWidget(line)

    def show_full_description(self):
        """Show full plugin description in a dialog"""
        description = self.plugin.PLUGIN_INFO.description or _(
            "No description available"
        )
        QW.QMessageBox.information(
            self,
            _("Plugin Description"),
            f"<b>{self.plugin.PLUGIN_INFO.name}</b><br><br>{description}",
        )

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


class PluginConfigDialog(QW.QDialog):
    """Dialog for configuring plugins"""

    def __init__(self, parent: DLMainWindow):
        """Initialize plugin configuration dialog

        Args:
            parent: Main window instance
        """
        super().__init__(parent)
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
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Info text
        info_label = QW.QLabel(
            _(
                "Enable or disable plugins. Changes will be applied "
                "after clicking OK and reloading plugins."
            )
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

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
