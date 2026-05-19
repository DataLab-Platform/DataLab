# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin Configuration Dialog
----------------------------

Dialog for managing plugin enable/disable state and viewing plugin information.
"""

from __future__ import annotations

import inspect
import os
import os.path as osp
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import (
    DATALAB_PLUGINS_ENV_PATHS,
    DATALAB_PLUGINS_ENV_VAR,
    OTHER_PLUGINS_PATHLIST,
    PLUGIN_ERROR_COLOR,
    PLUGIN_OK_COLOR,
    Conf,
    _,
)
from datalab.plugins import PLUGINS_DEFAULT_PATH, PluginRegistry
from datalab.widgets.expandabletext import (
    ExpandableTextWidget,
    apply_palette_color,
    apply_subdued_color,
)

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.plugins import FailedPluginInfo, PluginBase


# --- Constants -------------------------------------------------------------------

#: Color for the "Active" status label
STATUS_ENABLED_COLOR = QG.QColor(PLUGIN_OK_COLOR)

#: Color for the "Disabled" / error status label
STATUS_DISABLED_COLOR = QG.QColor(PLUGIN_ERROR_COLOR)

#: Filter identifiers for the plugin list
FILTER_ALL = "all"
FILTER_ENABLED = "enabled"
FILTER_DISABLED = "disabled"
FILTER_ERRORS = "errors"

#: Content margins for individual plugin rows (left, top, right, bottom)
PLUGIN_ROW_MARGINS: tuple[int, int, int, int] = (5, 5, 5, 5)

#: Spacing between metadata items (version, state) in the top row
META_SPACING: int = 12

#: Dialog minimum dimensions
DIALOG_MIN_WIDTH: int = 600
DIALOG_MIN_HEIGHT: int = 400

#: Title font point-size increment relative to default
TITLE_FONT_SIZE_DELTA: int = 4

#: Font point-size decrement for monospace traceback labels
MONO_FONT_SIZE_DELTA: int = 1


# --- Helpers ---------------------------------------------------------------------


def _create_status_label(text: str, color: QG.QColor | None = None) -> QW.QLabel:
    """Create a bold status label with optional palette color."""
    status_label = QW.QLabel(text)
    status_font = status_label.font()
    status_font.setBold(True)
    status_label.setFont(status_font)
    if color is None:
        apply_subdued_color(status_label)
    else:
        apply_palette_color(status_label, color)
    return status_label


def _open_local_path(path: str) -> bool:
    """Open a local path with the desktop handler."""
    return QG.QDesktopServices.openUrl(QC.QUrl.fromLocalFile(path))


def _show_in_folder(path: str) -> bool:
    """Show a file in its containing folder, selecting it when supported."""
    filepath = osp.abspath(path)
    directory = osp.dirname(filepath)

    if sys.platform.startswith("win"):
        commands = [["explorer", f"/select,{osp.normpath(filepath)}"]]
    elif sys.platform == "darwin":
        commands = [["open", "-R", filepath]]
    else:
        commands = []
        if shutil.which("nautilus"):
            commands.append(["nautilus", "--select", filepath])
        if shutil.which("dolphin"):
            commands.append(["dolphin", "--select", filepath])
        if shutil.which("nemo"):
            commands.append(["nemo", filepath])
        if shutil.which("caja"):
            commands.append(["caja", "--select", filepath])

    for command in commands:
        try:
            subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            continue
    return _open_local_path(directory)


def _get_latest_plugin_load_at(main: DLMainWindow) -> datetime:
    """Return the most recent relevant plugin load timestamp."""
    timestamps = [
        value
        for value in (
            getattr(main, "started_at", None),
            getattr(main, "plugins_last_load_at", None),
        )
        if isinstance(value, datetime)
    ]
    if timestamps:
        return max(timestamps)
    return datetime.now().astimezone()


def _format_last_load_text(
    timestamp: datetime,
    now: datetime | None = None,
    locale: QC.QLocale | None = None,
) -> str:
    """Format the last load text with today/yesterday/date semantics."""
    if now is None:
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
    if locale is None:
        locale = QC.QLocale.system()

    if timestamp.date() == now.date():
        day_text = _("today")
    elif timestamp.date() == (now - timedelta(days=1)).date():
        day_text = _("yesterday")
    else:
        day_text = locale.toString(
            QC.QDate(timestamp.year, timestamp.month, timestamp.day),
            QC.QLocale.ShortFormat,
        )

    time_text = locale.toString(
        QC.QTime(timestamp.hour, timestamp.minute),
        "HH:mm",
    )
    return _("Last loaded: %s at %s") % (day_text, time_text)


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
        self.plugin_filepath = self._get_plugin_filepath()
        self.open_file_button: QW.QPushButton | None = None
        self.show_in_folder_button: QW.QPushButton | None = None
        self.setSizePolicy(QW.QSizePolicy.Preferred, QW.QSizePolicy.Maximum)

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(*PLUGIN_ROW_MARGINS)
        self.setLayout(layout)

        layout.addLayout(self._create_top_row(enabled, state))
        layout.addWidget(self._create_description_widget())
        actions_layout = self._create_actions_layout()
        if actions_layout is not None:
            layout.addLayout(actions_layout)
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
        meta_layout.setSpacing(META_SPACING)
        version_label = QW.QLabel(f"v{self.plugin_class.PLUGIN_INFO.version}")
        apply_subdued_color(version_label)
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
        self.description_widget = ExpandableTextWidget(description)
        self.description_widget.toggled.connect(self._sync_description_expanded_state)
        self.desc_label = self.description_widget.label
        self.toggle_button = self.description_widget.toggle_button
        self._toggle_btn = self.toggle_button
        return self.description_widget

    def _sync_description_expanded_state(self, expanded: bool) -> None:
        """Keep the legacy expanded state in sync with the description widget."""
        self._expanded = expanded

    def _get_plugin_filepath(self) -> str | None:
        """Return the Python file defining the plugin class, when available."""
        filepath = getattr(self.plugin_class, "__plugin_filepath__", None)
        if filepath:
            return osp.abspath(filepath)
        try:
            filepath = inspect.getsourcefile(self.plugin_class) or inspect.getfile(
                self.plugin_class
            )
        except (OSError, TypeError):
            return None
        return osp.abspath(filepath) if filepath else None

    def _create_actions_layout(self) -> QW.QHBoxLayout | None:
        """Create file/folder actions for the plugin source file."""
        if not self.plugin_filepath:
            return None

        actions_layout = QW.QHBoxLayout()
        actions_layout.addStretch()

        self.open_file_button = QW.QPushButton(
            get_icon("open_file_source.svg"), _("Open file")
        )
        self.open_file_button.clicked.connect(self._open_plugin_file)
        actions_layout.addWidget(self.open_file_button)

        self.show_in_folder_button = QW.QPushButton(
            get_icon("show_in_folder.svg"), _("Show in folder")
        )
        self.show_in_folder_button.clicked.connect(self._show_plugin_in_folder)
        actions_layout.addWidget(self.show_in_folder_button)
        return actions_layout

    def _open_plugin_file(self) -> None:
        """Open the plugin file with the desktop handler."""
        if self.plugin_filepath:
            _open_local_path(self.plugin_filepath)

    def _show_plugin_in_folder(self) -> None:
        """Show the plugin file in its containing folder."""
        if self.plugin_filepath:
            _show_in_folder(self.plugin_filepath)

    @staticmethod
    def _create_separator() -> QW.QFrame:
        """Create the row separator."""
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        return line

    def _toggle_description(self):
        """Toggle between collapsed and expanded description"""
        self.description_widget.set_expanded(not self.description_widget.is_expanded())
        self._expanded = self.description_widget.is_expanded()

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
            return self.state == PluginState.ENABLED
        if filter_mode == FILTER_DISABLED:
            return self.state == PluginState.DISABLED
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
        self.plugin_filepath = (
            osp.abspath(failed_info.filepath) if failed_info.filepath else None
        )
        self.open_file_button: QW.QPushButton | None = None
        self.show_in_folder_button: QW.QPushButton | None = None
        self.setSizePolicy(QW.QSizePolicy.Preferred, QW.QSizePolicy.Maximum)

        # Main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(*PLUGIN_ROW_MARGINS)
        self.setLayout(layout)

        layout.addLayout(self._create_top_row(failed_info))
        layout.addWidget(self._create_description_widget(failed_info))
        actions_layout = self._create_actions_layout()
        if actions_layout is not None:
            layout.addLayout(actions_layout)
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
        apply_subdued_color(name_label)
        top_layout.addWidget(name_label)

        top_layout.addStretch()
        meta_layout = QW.QHBoxLayout()
        meta_layout.setSpacing(META_SPACING)
        meta_layout.addWidget(_create_status_label("\u26a0 " + _("Import error")))
        top_layout.addLayout(meta_layout)
        return top_layout

    def _create_description_widget(self, failed_info: FailedPluginInfo) -> QW.QWidget:
        """Create the expandable traceback/details area."""
        description = failed_info.filepath
        if failed_info.traceback:
            description += "\n\n" + failed_info.traceback.strip()

        mono_font = QG.QFont("Consolas", self.font().pointSize() - MONO_FONT_SIZE_DELTA)
        self.description_widget = ExpandableTextWidget(
            description,
            text_interaction_flags=QC.Qt.TextSelectableByMouse,
            label_font=mono_font,
        )
        self.desc_label = self.description_widget.label
        self._toggle_btn = self.description_widget.toggle_button
        return self.description_widget

    def _create_actions_layout(self) -> QW.QHBoxLayout | None:
        """Create file/folder actions for the failed plugin path."""
        if not self.plugin_filepath:
            return None

        actions_layout = QW.QHBoxLayout()
        actions_layout.addStretch()

        self.open_file_button = QW.QPushButton(
            get_icon("open_file_source.svg"), _("Open file")
        )
        self.open_file_button.clicked.connect(self._open_plugin_file)
        actions_layout.addWidget(self.open_file_button)

        self.show_in_folder_button = QW.QPushButton(
            get_icon("show_in_folder.svg"), _("Show in folder")
        )
        self.show_in_folder_button.clicked.connect(self._show_plugin_in_folder)
        actions_layout.addWidget(self.show_in_folder_button)
        return actions_layout

    def _open_plugin_file(self) -> None:
        """Open the failed plugin file with the desktop handler."""
        if self.plugin_filepath:
            _open_local_path(self.plugin_filepath)

    def _show_plugin_in_folder(self) -> None:
        """Show the failed plugin file in its containing folder."""
        if self.plugin_filepath:
            _show_in_folder(self.plugin_filepath)

    def _toggle_description(self):
        """Toggle between collapsed and expanded description"""
        self.description_widget.set_expanded(not self.description_widget.is_expanded())
        self._expanded = self.description_widget.is_expanded()

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
        self.toggle_all_checkbox: QW.QCheckBox | None = None
        self.filter_combo: QW.QComboBox | None = None
        self.load_info_label: QW.QLabel | None = None
        self.plugins_layout: QW.QVBoxLayout | None = None

        self.setWindowTitle(_("Plugin Configuration"))
        self.setMinimumWidth(DIALOG_MIN_WIDTH)
        self.setMinimumHeight(DIALOG_MIN_HEIGHT)

        # Main layout
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._create_title_label())

        tabs = QW.QTabWidget()
        tabs.addTab(self._create_plugins_tab(), _("Enable/disable plugins"))
        tabs.addTab(self._create_search_paths_tab(), _("Plugin search paths"))
        layout.addWidget(tabs, 1)

        # Populate plugins
        self.populate_plugins()

        # Button box
        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addLayout(self._create_footer_layout(button_box))
        self._update_load_info_label()

    @staticmethod
    def _create_title_label() -> QW.QLabel:
        """Create the dialog title label."""
        title_label = QW.QLabel(_("Manage Plugins"))
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + TITLE_FONT_SIZE_DELTA)
        title_font.setBold(True)
        title_label.setFont(title_font)
        apply_palette_color(
            title_label,
            QW.QApplication.instance().palette().color(QG.QPalette.WindowText),
        )
        return title_label

    def _create_plugins_tab(self) -> QW.QWidget:
        """Create the 'Enable/disable plugins' tab content."""
        tab = QW.QWidget()
        tab_layout = QW.QVBoxLayout()
        tab.setLayout(tab_layout)
        tab_layout.addWidget(self._create_info_label())
        tab_layout.addLayout(self._create_controls_layout())
        tab_layout.addWidget(self._create_scroll_area(), 1)
        return tab

    def _create_search_paths_tab(self) -> QW.QWidget:
        """Create the 'Plugin search paths' tab content (scrollable)."""
        tab = QW.QWidget()
        tab_layout = QW.QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab.setLayout(tab_layout)

        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        container = QW.QWidget()
        container_layout = QW.QVBoxLayout()
        container.setLayout(container_layout)
        container_layout.addLayout(self._create_search_paths_layout())
        container_layout.addStretch()
        scroll.setWidget(container)
        tab_layout.addWidget(scroll)
        return tab

    @staticmethod
    def _create_info_label() -> QW.QLabel:
        """Create the dialog information label."""
        info_label = QW.QLabel(
            _("Changes will be applied after clicking OK and reloading plugins.")
        )
        info_label.setWordWrap(True)
        apply_subdued_color(info_label)
        return info_label

    @staticmethod
    def _collect_search_paths() -> list[tuple[str, bool]]:
        """Return active plugin search paths with their env-var origin flag.

        Returns:
            List of ``(path, from_env_var)`` tuples in discovery order.
        """
        seen: set[str] = set()
        entries: list[tuple[str, bool]] = []
        env_paths_norm = {osp.normpath(p) for p in DATALAB_PLUGINS_ENV_PATHS}
        candidates: list[str] = []
        custom = Conf.main.plugins_path.get()
        if custom:
            candidates.append(custom)
        candidates.append(PLUGINS_DEFAULT_PATH)
        candidates.extend(OTHER_PLUGINS_PATHLIST)
        for raw in candidates:
            if not raw:
                continue
            path = osp.normpath(raw)
            if path in seen:
                continue
            seen.add(path)
            entries.append((path, path in env_paths_norm))
        return entries

    def _create_search_paths_layout(self) -> QW.QVBoxLayout:
        """Create the layout listing active plugin search paths."""
        paths_layout = QW.QVBoxLayout()

        intro = QW.QLabel(
            _("The following directories are scanned at startup for plugins:")
        )
        intro.setWordWrap(True)
        paths_layout.addWidget(intro)

        items: list[str] = []
        for path, from_env in self._collect_search_paths():
            url = QC.QUrl.fromLocalFile(path).toString()
            link = f'<a href="{url}">{path}</a>'
            if from_env:
                link += f" <i>({_('from')} <code>{DATALAB_PLUGINS_ENV_VAR}</code>)</i>"
            items.append(f"<li>{link}</li>")
        if items:
            html = (
                "<ul style='margin:0; padding-left:18px;'>" + "".join(items) + "</ul>"
            )
        else:
            html = "<i>" + _("No plugin search path is currently active.") + "</i>"

        paths_label = QW.QLabel(html)
        paths_label.setTextInteractionFlags(QC.Qt.TextBrowserInteraction)
        paths_label.setOpenExternalLinks(True)
        paths_label.setWordWrap(True)
        paths_layout.addWidget(paths_label)

        hint = QW.QLabel(
            _(
                "Additional directories can be added via the "
                "<code>%s</code> environment variable "
                "(multiple paths separated by '<code>%s</code>'). "
                "Changes take effect at DataLab startup."
            )
            % (DATALAB_PLUGINS_ENV_VAR, os.pathsep)
        )
        hint.setWordWrap(True)
        apply_subdued_color(hint)
        paths_layout.addWidget(hint)
        return paths_layout

    def _create_controls_layout(self) -> QW.QHBoxLayout:
        """Create the row with master controls."""
        controls_layout = QW.QHBoxLayout()
        self.toggle_all_checkbox = QW.QCheckBox(_("Enable all plugins"))
        self.toggle_all_checkbox.setTristate(True)
        self.toggle_all_checkbox.toggled.connect(self.set_all_enabled)
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

    def _create_footer_layout(self, button_box: QW.QDialogButtonBox) -> QW.QHBoxLayout:
        """Create the footer with the last-load label and dialog buttons."""
        footer_layout = QW.QHBoxLayout()
        self.load_info_label = QW.QLabel()
        apply_subdued_color(self.load_info_label)
        footer_layout.addWidget(self.load_info_label)
        footer_layout.addStretch()
        footer_layout.addWidget(button_box)
        return footer_layout

    def _update_load_info_label(self) -> None:
        """Update the text describing the latest plugin load time."""
        if self.load_info_label is None:
            return
        latest_load = _get_latest_plugin_load_at(self.main)
        self.load_info_label.setText(_format_last_load_text(latest_load))

    def set_all_enabled(self, enabled: bool) -> None:
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
        self._update_load_info_label()
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
