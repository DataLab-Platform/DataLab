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


STATUS_ENABLED_COLOR = QG.QColor("#2ecc71")
STATUS_DISABLED_COLOR = QG.QColor("#e74c3c")

FILTER_ALL = "all"
FILTER_ENABLED = "enabled"
FILTER_DISABLED = "disabled"
FILTER_ERRORS = "errors"


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


def _create_description_scroll_area(widget: QW.QWidget) -> QW.QScrollArea:
    """Create a description scroll area with consistent spacing behavior."""
    container = QW.QWidget()
    container_layout = QW.QVBoxLayout()
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(0)
    container_layout.addWidget(widget)
    container_layout.addStretch()
    container.setLayout(container_layout)

    scroll_area = QW.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
    scroll_area.setFrameShape(QW.QFrame.NoFrame)
    scroll_area.setWidget(container)
    scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
    return scroll_area


def _create_expand_toggle_button(callback) -> QW.QPushButton:
    """Create a theme-aware expand/collapse button."""
    button = QW.QPushButton("\u25bc " + _("Show more"))
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


def _create_status_label(text: str, color: QG.QColor | None = None) -> QW.QLabel:
    """Create a bold status label with optional palette color."""
    status_label = QW.QLabel(text)
    status_font = status_label.font()
    status_font.setBold(True)
    status_label.setFont(status_font)
    if color is None:
        _apply_subdued_color(status_label)
    else:
        _apply_palette_color(status_label, color)
    return status_label


def _create_description_container(
    description_label: QW.QLabel,
) -> tuple[QW.QWidget, QW.QScrollArea]:
    """Create the indented description container used by plugin widgets."""
    desc_container = QW.QWidget()
    desc_container_layout = QW.QVBoxLayout()
    desc_container_layout.setContentsMargins(20, 0, 0, 0)
    desc_container_layout.setSpacing(0)
    desc_container.setLayout(desc_container_layout)

    desc_scroll = _create_description_scroll_area(description_label)
    desc_scroll.setMaximumHeight(60)
    desc_container_layout.addWidget(desc_scroll)
    return desc_container, desc_scroll


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
        self._expanded = False

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        layout.addLayout(self._create_top_row(enabled, state))
        layout.addWidget(self._create_description_widget())
        layout.addWidget(self._create_separator())

    def _create_top_row(self, enabled: bool, state: str) -> QW.QHBoxLayout:
        """Create the top row with checkbox, title and metadata."""
        top_layout = QW.QHBoxLayout()

        self.checkbox = QW.QCheckBox()
        self.checkbox.setChecked(enabled)
        top_layout.addWidget(self.checkbox)

        name_label = QW.QLabel(self.plugin_class.PLUGIN_INFO.name)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        top_layout.addWidget(name_label)

        top_layout.addStretch()

        meta_layout = QW.QHBoxLayout()
        meta_layout.setSpacing(12)
        version_label = QW.QLabel(f"v{self.plugin_class.PLUGIN_INFO.version}")
        _apply_subdued_color(version_label)
        meta_layout.addWidget(version_label)
        meta_layout.addWidget(self._create_state_label(state))
        top_layout.addLayout(meta_layout)
        return top_layout

    @staticmethod
    def _create_state_label(state: str) -> QW.QLabel:
        """Create the state label."""
        if state == PluginState.ENABLED:
            return _create_status_label("\u2713 " + _("Active"), STATUS_ENABLED_COLOR)
        if state == PluginState.DISABLED:
            return _create_status_label(
                "\u2717 " + _("Disabled"), STATUS_DISABLED_COLOR
            )
        return _create_status_label("\u2717 " + _("Error"))

    def _create_description_widget(self) -> QW.QWidget:
        """Create the description area for the plugin."""
        description = self.plugin_class.PLUGIN_INFO.description or _(
            "No description available"
        )
        self.desc_label = QW.QLabel(description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignTop)
        _apply_subdued_color(self.desc_label)

        if len(description) <= 100:
            desc_container = QW.QWidget()
            desc_container_layout = QW.QVBoxLayout()
            desc_container_layout.setContentsMargins(20, 0, 0, 0)
            desc_container_layout.setSpacing(0)
            desc_container_layout.addWidget(self.desc_label)
            desc_container.setLayout(desc_container_layout)
            return desc_container

        desc_container, self._desc_scroll = _create_description_container(
            self.desc_label
        )
        self._toggle_btn = _create_expand_toggle_button(self._toggle_description)
        desc_container.layout().addWidget(self._toggle_btn)
        return desc_container

    @staticmethod
    def _create_separator() -> QW.QFrame:
        """Create the row separator."""
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        return line

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

    def matches_filter(self, filter_mode: str) -> bool:
        """Return whether widget should be visible for current filter."""
        if filter_mode == FILTER_ENABLED:
            return self.is_enabled()
        if filter_mode == FILTER_DISABLED:
            return not self.is_enabled()
        if filter_mode == FILTER_ERRORS:
            return False
        return True


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
        self._expanded = False

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        layout.addLayout(self._create_top_row(failed_info))
        layout.addWidget(self._create_description_widget(failed_info))
        layout.addWidget(PluginInfoWidget._create_separator())

    def _create_top_row(self, failed_info: FailedPluginInfo) -> QW.QHBoxLayout:
        """Create the top row for a failed plugin."""
        top_layout = QW.QHBoxLayout()
        checkbox = QW.QCheckBox()
        checkbox.setChecked(False)
        checkbox.setEnabled(False)
        top_layout.addWidget(checkbox)

        file_name = osp.basename(failed_info.name)
        name_label = QW.QLabel(file_name)
        name_font = name_label.font()
        name_font.setBold(True)
        name_label.setFont(name_font)
        _apply_subdued_color(name_label)
        top_layout.addWidget(name_label)

        top_layout.addStretch()
        meta_layout = QW.QHBoxLayout()
        meta_layout.setSpacing(12)
        meta_layout.addWidget(_create_status_label("\u26a0 " + _("Import error")))
        top_layout.addLayout(meta_layout)
        return top_layout

    def _create_description_widget(self, failed_info: FailedPluginInfo) -> QW.QWidget:
        """Create the expandable traceback/details area."""
        description = failed_info.filepath
        if failed_info.traceback:
            description += "\n\n" + failed_info.traceback.strip()

        desc_label = QW.QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignTop)
        desc_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        _apply_subdued_color(desc_label)

        mono_font = QG.QFont("Consolas", desc_label.font().pointSize() - 1)
        desc_label.setFont(mono_font)

        desc_container, self._desc_scroll = _create_description_container(desc_label)
        self._desc_scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAsNeeded)
        self._toggle_btn = _create_expand_toggle_button(self._toggle_description)
        desc_container.layout().addWidget(self._toggle_btn)
        return desc_container

    def _toggle_description(self):
        """Toggle between collapsed and expanded description"""
        self._expanded = not self._expanded
        if self._expanded:
            self._desc_scroll.setMaximumHeight(200)
            self._toggle_btn.setText("\u25b2 " + _("Show less"))
        else:
            self._desc_scroll.setMaximumHeight(60)
            self._toggle_btn.setText("\u25bc " + _("Show more"))

    @staticmethod
    def matches_filter(filter_mode: str) -> bool:
        """Return whether failed plugin widget should be visible."""
        return filter_mode in (FILTER_ALL, FILTER_ERRORS)


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
        self.failed_plugin_widgets: list[FailedPluginInfoWidget] = []

        self.setWindowTitle(_("Plugin Configuration"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # Main layout
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._create_title_label())
        layout.addWidget(self._create_info_label())
        layout.addLayout(self._create_controls_layout())
        layout.addWidget(self._create_scroll_area(), 1)

        # Populate plugins
        self.populate_plugins()

        # Button box
        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _create_title_label() -> QW.QLabel:
        """Create the dialog title label."""
        title_label = QW.QLabel(_("Manage Plugins"))
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title_label.setFont(title_font)
        _apply_palette_color(
            title_label,
            QW.QApplication.instance().palette().color(QG.QPalette.WindowText),
        )
        return title_label

    @staticmethod
    def _create_info_label() -> QW.QLabel:
        """Create the dialog information label."""
        info_label = QW.QLabel(
            _(
                "Enable or disable plugins. Changes will be applied "
                "after clicking OK and reloading plugins."
            )
        )
        info_label.setWordWrap(True)
        _apply_subdued_color(info_label)
        return info_label

    def _create_controls_layout(self) -> QW.QHBoxLayout:
        """Create the row with master controls."""
        controls_layout = QW.QHBoxLayout()
        self.toggle_all_checkbox = QW.QCheckBox(_("Enable all plugins"))
        self.toggle_all_checkbox.setTristate(True)
        self.toggle_all_checkbox.toggled.connect(self._set_all_enabled)
        controls_layout.addWidget(self.toggle_all_checkbox)

        controls_layout.addStretch()
        controls_layout.addWidget(QW.QLabel(_("Filter:")))
        self.filter_combo = QW.QComboBox()
        for label, data in (
            (_("All plugins"), FILTER_ALL),
            (_("Enabled plugins"), FILTER_ENABLED),
            (_("Disabled plugins"), FILTER_DISABLED),
            (_("Plugins with errors"), FILTER_ERRORS),
        ):
            self.filter_combo.addItem(label, data)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        controls_layout.addWidget(self.filter_combo)
        return controls_layout

    def _create_scroll_area(self) -> QW.QScrollArea:
        """Create the plugin scroll area."""
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        container = QW.QWidget()
        self.plugins_layout = QW.QVBoxLayout()
        self.plugins_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(self.plugins_layout)
        scroll.setWidget(container)
        return scroll

    def _set_all_enabled(self, enabled: bool):
        """Set all plugin checkboxes to the same state."""
        for widget in self.plugin_widgets:
            widget.checkbox.blockSignals(True)
            widget.checkbox.setChecked(enabled)
            widget.checkbox.blockSignals(False)
        self._apply_filter()
        self._sync_toggle_all_checkbox()

    def _sync_toggle_all_checkbox(self) -> None:
        """Reflect individual plugin states in the master checkbox."""
        if not self.plugin_widgets:
            return

        enabled_count = sum(widget.is_enabled() for widget in self.plugin_widgets)
        all_enabled = enabled_count == len(self.plugin_widgets)
        none_enabled = enabled_count == 0

        self.toggle_all_checkbox.blockSignals(True)
        if all_enabled:
            self.toggle_all_checkbox.setCheckState(QC.Qt.Checked)
        elif none_enabled:
            self.toggle_all_checkbox.setCheckState(QC.Qt.Unchecked)
        else:
            self.toggle_all_checkbox.setCheckState(QC.Qt.PartiallyChecked)
        self.toggle_all_checkbox.blockSignals(False)

    def _plugin_toggled(self) -> None:
        """Update dialog controls after a plugin checkbox changed."""
        self._apply_filter()
        self._sync_toggle_all_checkbox()

    def _apply_filter(self) -> None:
        """Update plugin visibility according to current filter."""
        filter_mode = self.filter_combo.currentData()
        for widget in self.plugin_widgets:
            widget.setVisible(widget.matches_filter(filter_mode))
        for widget in self.failed_plugin_widgets:
            widget.setVisible(widget.matches_filter(filter_mode))

    def populate_plugins(self):
        """Populate the dialog with all discovered plugins"""
        registered_names = {p.info.name for p in PluginRegistry.get_plugins()}
        enabled_plugins = Conf.main.plugins_enabled_list.get(None)

        self._add_failed_plugins()
        self._add_plugin_widgets(registered_names, enabled_plugins)

        # Add stretch at the end
        self.plugins_layout.addStretch()
        self._sync_toggle_all_checkbox()
        self._apply_filter()

    def _add_failed_plugins(self) -> None:
        """Add failed plugin widgets before regular plugins."""
        for failed_info in PluginRegistry.get_failed_plugins():
            failed_widget = FailedPluginInfoWidget(failed_info)
            self.failed_plugin_widgets.append(failed_widget)
            self.plugins_layout.addWidget(failed_widget)

    def _add_plugin_widgets(
        self, registered_names: set[str], enabled_plugins: list[str] | None
    ) -> None:
        """Add widgets for discovered plugins."""
        for plugin_class in PluginRegistry.get_plugin_classes():
            plugin_name = plugin_class.PLUGIN_INFO.name
            enabled = enabled_plugins is None or plugin_name in enabled_plugins
            state = (
                PluginState.ENABLED
                if plugin_name in registered_names
                else PluginState.DISABLED
            )
            widget = PluginInfoWidget(plugin_class, enabled, state)
            widget.checkbox.toggled.connect(self._plugin_toggled)
            self.plugin_widgets.append(widget)
            self.plugins_layout.addWidget(widget)

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
