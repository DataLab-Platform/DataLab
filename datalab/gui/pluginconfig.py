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
from html import escape
from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getexistingdirectory

from datalab.config import (
    DATALAB_PLUGINS_ENV_PATHS,
    DATALAB_PLUGINS_ENV_VAR,
    OTHER_PLUGINS_PATHLIST,
    PLUGIN_ERROR_COLOR,
    PLUGIN_OK_COLOR,
    Conf,
    _,
    get_user_plugin_paths,
    normalize_plugin_paths,
    set_user_plugin_paths,
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


def format_last_load_text(
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


class SearchPathItemWidget(QW.QWidget):
    """Widget representing one plugin search path."""

    def __init__(
        self,
        path: str,
        *,
        editable: bool,
        from_env: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.path = path
        self.from_env = from_env
        self.path_label: QW.QLabel | None = None
        self.edit_button: QW.QToolButton | None = None
        self.delete_button: QW.QToolButton | None = None

        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        path_label = QW.QLabel()
        path_label.setWordWrap(True)
        self.path_label = path_label
        self.set_links_enabled(True)
        layout.addWidget(path_label, 1)

        if editable:
            self.edit_button = QW.QToolButton()
            self.edit_button.setIcon(get_icon("annotations_edit.svg"))
            self.edit_button.setToolTip(_("Edit directory"))
            self.edit_button.setAutoRaise(True)
            layout.addWidget(self.edit_button)

            self.delete_button = QW.QToolButton()
            self.delete_button.setIcon(get_icon("annotations_delete.svg"))
            self.delete_button.setToolTip(_("Remove directory"))
            self.delete_button.setAutoRaise(True)
            layout.addWidget(self.delete_button)

    def set_links_enabled(self, enabled: bool) -> None:
        """Update link interactivity and appearance for enabled/disabled states."""
        if self.path_label is None:
            return

        self.path_label.setText(self._build_path_label_text(enabled))
        if enabled:
            self.path_label.setTextInteractionFlags(QC.Qt.TextBrowserInteraction)
            self.path_label.setOpenExternalLinks(True)
            self.path_label.setStyleSheet("")
            return

        disabled_color = QW.QApplication.palette().color(
            QG.QPalette.Disabled, QG.QPalette.WindowText
        )
        disabled_color_name = disabled_color.name()
        self.path_label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self.path_label.setOpenExternalLinks(False)
        self.path_label.setStyleSheet(f"QLabel {{ color: {disabled_color_name}; }}")

    def _build_path_label_text(self, enabled: bool) -> str:
        """Return rich text for the path label according to enabled state."""
        url = escape(QC.QUrl.fromLocalFile(self.path).toString(), quote=True)
        path = escape(self.path)
        if enabled:
            link_text = f'<a href="{url}">{path}</a>'
        else:
            disabled_color = (
                QW.QApplication.palette()
                .color(QG.QPalette.Disabled, QG.QPalette.WindowText)
                .name()
            )
            link_text = f'<span style="color: {disabled_color};">{path}</span>'
        if self.from_env:
            env_var = escape(DATALAB_PLUGINS_ENV_VAR)
            link_text += f" <i>({_('from')} <code>{env_var}</code>)</i>"
        return link_text


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
        self.fixed_path_widgets: list[SearchPathItemWidget] = []
        self.extra_path_widgets: list[SearchPathItemWidget] = []
        self.original_plugins_enabled = Conf.main.plugins_enabled.get(True)
        self.plugins_enabled = self.original_plugins_enabled
        self.original_v020_plugins_warning_ignore = (
            Conf.main.v020_plugins_warning_ignore.get(False)
        )
        self.v020_plugins_warning_ignore = self.original_v020_plugins_warning_ignore
        self.original_extra_plugin_paths = get_user_plugin_paths()
        self.extra_plugin_paths = list(self.original_extra_plugin_paths)
        self.tabs: QW.QTabWidget | None = None
        self.toggle_all_checkbox: QW.QCheckBox | None = None
        self.filter_combo: QW.QComboBox | None = None
        self.load_info_label: QW.QLabel | None = None
        self.plugins_layout: QW.QVBoxLayout | None = None
        self.plugins_content: QW.QWidget | None = None
        self.plugins_disabled_label: QW.QLabel | None = None
        self.extra_paths_layout: QW.QVBoxLayout | None = None
        self.extra_paths_placeholder: QW.QLabel | None = None
        self.settings_scroll: QW.QScrollArea | None = None
        self.settings_disabled_label: QW.QLabel | None = None
        self.add_path_button: QW.QPushButton | None = None
        self.reload_button: QW.QPushButton | None = None
        self.global_toggle_button: QW.QPushButton | None = None
        self.v020_warning_checkbox: QW.QCheckBox | None = None

        self.setWindowTitle(_("Plugin Configuration"))
        self.setMinimumWidth(DIALOG_MIN_WIDTH)
        self.setMinimumHeight(DIALOG_MIN_HEIGHT)

        # Main layout
        layout = QW.QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QW.QTabWidget()
        self.tabs.addTab(self._create_plugins_tab(), _("Enable/disable plugins"))
        self.tabs.addTab(self._create_search_paths_tab(), _("Plugin settings"))
        self.tabs.setCornerWidget(self._create_corner_widget(), QC.Qt.TopRightCorner)
        layout.addWidget(self.tabs, 1)

        # Populate plugins
        self.populate_plugins()

        # Button box
        button_box = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        self.reload_button = QW.QPushButton(
            get_icon("refresh-auto.svg"), _("Apply and reload plugins")
        )
        button_box.addButton(self.reload_button, QW.QDialogButtonBox.ActionRole)
        self.reload_button.clicked.connect(self._apply_and_reload_plugins)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addLayout(self._create_footer_layout(button_box))
        self._update_load_info_label()
        self._update_global_plugins_ui()

    def _create_corner_widget(self) -> QW.QWidget:
        """Create the low-impact global plugin toggle shown near the tab titles."""
        container = QW.QWidget()
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(6, 2, 0, 2)
        container.setLayout(layout)

        self.global_toggle_button = QW.QPushButton()
        self.global_toggle_button.setAutoDefault(False)
        self.global_toggle_button.setDefault(False)
        self.global_toggle_button.setCursor(QG.QCursor(QC.Qt.PointingHandCursor))
        self.global_toggle_button.setSizePolicy(
            QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed
        )
        self.global_toggle_button.setIconSize(QC.QSize(14, 14))
        self.global_toggle_button.setMinimumHeight(24)
        self.global_toggle_button.setStyleSheet("QPushButton { padding: 2px 8px; }")
        self.global_toggle_button.clicked.connect(self._toggle_global_plugins_enabled)
        layout.addWidget(self.global_toggle_button)
        return container

    def _create_plugins_tab(self) -> QW.QWidget:
        """Create the 'Enable/disable plugins' tab content."""
        tab = QW.QWidget()
        tab_layout = QW.QVBoxLayout()
        tab.setLayout(tab_layout)
        tab_layout.addWidget(self._create_info_label())

        self.plugins_disabled_label = QW.QLabel(
            _("Third-party plugins are globally disabled.")
        )
        self.plugins_disabled_label.setWordWrap(True)
        apply_subdued_color(self.plugins_disabled_label)
        self.plugins_disabled_label.hide()
        tab_layout.addWidget(self.plugins_disabled_label)

        self.plugins_content = QW.QWidget()
        content_layout = QW.QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.plugins_content.setLayout(content_layout)
        content_layout.addLayout(self._create_controls_layout())
        content_layout.addWidget(self._create_scroll_area(), 1)
        tab_layout.addWidget(self.plugins_content, 1)
        return tab

    def _create_search_paths_tab(self) -> QW.QWidget:
        """Create the 'Plugin settings' tab content (scrollable)."""
        tab = QW.QWidget()
        tab_layout = QW.QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab.setLayout(tab_layout)

        self.settings_disabled_label = QW.QLabel(
            _("Third-party plugins are globally disabled.")
        )
        self.settings_disabled_label.setWordWrap(True)
        apply_subdued_color(self.settings_disabled_label)
        self.settings_disabled_label.hide()
        tab_layout.addWidget(self.settings_disabled_label)

        self.settings_scroll = QW.QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        container = QW.QWidget()
        container_layout = QW.QVBoxLayout()
        container.setLayout(container_layout)
        container_layout.addLayout(self._create_search_paths_layout())
        container_layout.addSpacing(18)
        container_layout.addLayout(self._create_warning_settings_layout())
        container_layout.addStretch()
        self.settings_scroll.setWidget(container)
        tab_layout.addWidget(self.settings_scroll)
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
    def _collect_fixed_search_paths() -> list[tuple[str, bool]]:
        """Return fixed plugin search paths with their env-var origin flag.

        Returns:
            List of ``(path, from_env_var)`` tuples in discovery order.
        """
        seen: set[str] = set()
        entries: list[tuple[str, bool]] = []
        env_paths_norm = {osp.normpath(p) for p in DATALAB_PLUGINS_ENV_PATHS}
        candidates = [PLUGINS_DEFAULT_PATH]
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
        """Create the layout listing plugin search paths."""
        paths_layout = QW.QVBoxLayout()

        intro = QW.QLabel(
            _("The following directories are scanned at startup for plugins:")
        )
        intro.setWordWrap(True)
        paths_layout.addWidget(intro)

        fixed_title = QW.QLabel(_("Default plugin directories"))
        fixed_font = fixed_title.font()
        fixed_font.setBold(True)
        fixed_title.setFont(fixed_font)
        paths_layout.addWidget(fixed_title)

        fixed_layout = QW.QVBoxLayout()
        fixed_layout.setContentsMargins(18, 0, 0, 0)
        fixed_layout.setSpacing(4)
        for path, from_env in self._collect_fixed_search_paths():
            widget = SearchPathItemWidget(path, editable=False, from_env=from_env)
            self.fixed_path_widgets.append(widget)
            fixed_layout.addWidget(widget)
        paths_layout.addLayout(fixed_layout)

        paths_layout.addSpacing(12)

        header_layout = QW.QHBoxLayout()
        extra_title = QW.QLabel(_("Additional plugin directories"))
        extra_font = extra_title.font()
        extra_font.setBold(True)
        extra_title.setFont(extra_font)
        header_layout.addWidget(extra_title)
        header_layout.addStretch()
        self.add_path_button = QW.QPushButton(get_icon("metadata_add.svg"), _("Add"))
        self.add_path_button.clicked.connect(self._add_search_path)
        header_layout.addWidget(self.add_path_button)
        paths_layout.addLayout(header_layout)

        extra_intro = QW.QLabel(
            _("Additional directories are saved in DataLab configuration.")
        )
        extra_intro.setWordWrap(True)
        apply_subdued_color(extra_intro)
        paths_layout.addWidget(extra_intro)

        self.extra_paths_layout = QW.QVBoxLayout()
        self.extra_paths_layout.setContentsMargins(18, 0, 0, 0)
        self.extra_paths_layout.setSpacing(4)
        paths_layout.addLayout(self.extra_paths_layout)

        self.extra_paths_placeholder = QW.QLabel(
            _("No additional plugin directory is configured.")
        )
        self.extra_paths_placeholder.setWordWrap(True)
        apply_subdued_color(self.extra_paths_placeholder)
        self.extra_paths_layout.addWidget(self.extra_paths_placeholder)
        self._refresh_extra_path_widgets()

        hint = QW.QLabel(
            _(
                "Directories provided via the "
                "<code>%s</code> environment variable "
                "(multiple paths separated by '<code>%s</code>') "
                "also appear above as read-only entries. "
                "Changes take effect at DataLab startup."
            )
            % (DATALAB_PLUGINS_ENV_VAR, os.pathsep)
        )
        hint.setWordWrap(True)
        apply_subdued_color(hint)
        paths_layout.addWidget(hint)
        return paths_layout

    def _create_warning_settings_layout(self) -> QW.QVBoxLayout:
        """Create the layout for plugin compatibility warning options."""
        warnings_layout = QW.QVBoxLayout()

        title = QW.QLabel(_("Compatibility warnings"))
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        warnings_layout.addWidget(title)

        self.v020_warning_checkbox = QW.QCheckBox(
            _("Hide warnings for incompatible DataLab v0.20 plugins")
        )
        self.v020_warning_checkbox.setChecked(self.v020_plugins_warning_ignore)
        self.v020_warning_checkbox.toggled.connect(
            self._set_v020_plugins_warning_ignore
        )
        warnings_layout.addWidget(self.v020_warning_checkbox)

        hint = QW.QLabel(
            _(
                "If enabled, DataLab will not warn you about v0.20 plugins "
                "that are no longer compatible with v1.0."
            )
        )
        hint.setWordWrap(True)
        apply_subdued_color(hint)
        warnings_layout.addWidget(hint)
        return warnings_layout

    def _refresh_extra_path_widgets(self) -> None:
        """Rebuild the editable extra-path widgets."""
        if self.extra_paths_layout is None:
            return

        for widget in self.extra_path_widgets:
            self.extra_paths_layout.removeWidget(widget)
            widget.deleteLater()
        self.extra_path_widgets.clear()

        if self.extra_paths_placeholder is not None:
            self.extra_paths_placeholder.setVisible(not self.extra_plugin_paths)

        for path in self.extra_plugin_paths:
            widget = SearchPathItemWidget(path, editable=True)
            assert widget.edit_button is not None
            assert widget.delete_button is not None
            widget.edit_button.clicked.connect(
                lambda _checked=False, item=widget: self._edit_search_path(item)
            )
            widget.delete_button.clicked.connect(
                lambda _checked=False, item=widget: self._remove_search_path(item)
            )
            self.extra_path_widgets.append(widget)
            self.extra_paths_layout.addWidget(widget)

    def _browse_plugin_directory(self, initial_path: str | None = None) -> str | None:
        """Open a directory chooser for a plugin search path."""
        basedir = initial_path or Conf.main.base_dir.get(osp.expanduser("~"))
        directory = getexistingdirectory(self, _("Select plugin directory"), basedir)
        normalized = normalize_plugin_paths([directory])
        return normalized[0] if normalized else None

    def _get_fixed_search_path_set(self) -> set[str]:
        """Return the normalized set of fixed plugin search paths."""
        return {path for path, _from_env in self._collect_fixed_search_paths()}

    def _add_search_path(self) -> None:
        """Append a new extra plugin search path."""
        path = self._browse_plugin_directory()
        if not path:
            return
        if path in self._get_fixed_search_path_set() or path in self.extra_plugin_paths:
            return
        self.extra_plugin_paths.append(path)
        self._refresh_extra_path_widgets()

    def _edit_search_path(self, item: SearchPathItemWidget) -> None:
        """Edit an existing extra plugin search path."""
        path = self._browse_plugin_directory(item.path)
        if not path:
            return
        other_paths = [entry for entry in self.extra_plugin_paths if entry != item.path]
        if path in self._get_fixed_search_path_set() or path in other_paths:
            return
        index = self.extra_plugin_paths.index(item.path)
        self.extra_plugin_paths[index] = path
        self._refresh_extra_path_widgets()

    def _remove_search_path(self, item: SearchPathItemWidget) -> None:
        """Remove an extra plugin search path."""
        self.extra_plugin_paths = [
            path for path in self.extra_plugin_paths if path != item.path
        ]
        self._refresh_extra_path_widgets()

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
        self.load_info_label.setText(format_last_load_text(latest_load))

    def _update_global_plugins_ui(self) -> None:
        """Refresh button texts and enabled state for global plugin controls."""
        enabled = self.plugins_enabled
        if self.global_toggle_button is not None:
            self.global_toggle_button.setText(
                _("Disable plugins globally")
                if enabled
                else _("Enable plugins globally")
            )
            self.global_toggle_button.setIcon(
                get_icon("uncheck_all.svg") if enabled else get_icon("check_all.svg")
            )
        if self.reload_button is not None:
            self.reload_button.setEnabled(enabled)
        if self.plugins_content is not None:
            self.plugins_content.setEnabled(enabled)
        if self.plugins_disabled_label is not None:
            self.plugins_disabled_label.setVisible(not enabled)
        if self.settings_scroll is not None:
            self.settings_scroll.setEnabled(enabled)
        if self.settings_disabled_label is not None:
            self.settings_disabled_label.setVisible(not enabled)
        for widget in self.fixed_path_widgets + self.extra_path_widgets:
            widget.set_links_enabled(enabled)

    def _toggle_global_plugins_enabled(self) -> None:
        """Toggle the local global third-party plugin enabled state."""
        self.plugins_enabled = not self.plugins_enabled
        self._update_global_plugins_ui()

    def _set_v020_plugins_warning_ignore(self, state: bool) -> None:
        """Store the local compatibility-warning preference."""
        self.v020_plugins_warning_ignore = state

    def _has_changes(self) -> bool:
        """Return whether plugin enablement or search paths changed."""
        plugin_changes_made = any(
            widget.has_changed() for widget in self.plugin_widgets
        )
        path_changes_made = self.extra_plugin_paths != self.original_extra_plugin_paths
        plugins_enabled_changed = self.plugins_enabled != self.original_plugins_enabled
        warning_changed = (
            self.v020_plugins_warning_ignore
            != self.original_v020_plugins_warning_ignore
        )
        return (
            plugin_changes_made
            or path_changes_made
            or plugins_enabled_changed
            or warning_changed
        )

    def _has_reloadable_changes(self) -> bool:
        """Return whether changes require plugin reload while plugins are enabled."""
        plugin_changes_made = any(
            widget.has_changed() for widget in self.plugin_widgets
        )
        path_changes_made = self.extra_plugin_paths != self.original_extra_plugin_paths
        return plugin_changes_made or path_changes_made

    def _has_global_plugins_enabled_change(self) -> bool:
        """Return whether the global third-party plugin state changed."""
        return self.plugins_enabled != self.original_plugins_enabled

    def _save_configuration(self) -> None:
        """Persist current plugin enablement and search path settings."""
        Conf.main.plugins_enabled.set(self.plugins_enabled)
        Conf.main.v020_plugins_warning_ignore.set(self.v020_plugins_warning_ignore)
        enabled_plugins = [
            widget.plugin_class.PLUGIN_INFO.name
            for widget in self.plugin_widgets
            if widget.is_enabled()
        ]
        Conf.main.plugins_enabled_list.set(enabled_plugins)
        set_user_plugin_paths(self.extra_plugin_paths)

    def _mark_current_state_as_saved(self) -> None:
        """Synchronize original values with the current dialog state."""
        self.original_plugins_enabled = self.plugins_enabled
        self.original_v020_plugins_warning_ignore = self.v020_plugins_warning_ignore
        self.original_extra_plugin_paths = list(self.extra_plugin_paths)

    def _refresh_plugin_list(self) -> None:
        """Rebuild the plugin list from the current registry state."""
        if self.plugins_layout is None:
            return

        while self.plugins_layout.count():
            item = self.plugins_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.plugin_widgets.clear()
        self.failed_plugin_widgets.clear()
        self.populate_plugins()

    def _apply_and_reload_plugins(self) -> None:
        """Save configuration, reload plugins, and keep dialog open."""
        if not self.plugins_enabled:
            return
        global_plugins_enabled_changed = self._has_global_plugins_enabled_change()
        if self._has_changes():
            self._save_configuration()
        if global_plugins_enabled_changed:
            self.main.set_plugins_enabled(self.plugins_enabled)
        else:
            self.main.reload_plugins()
        self._mark_current_state_as_saved()
        self._refresh_plugin_list()
        self._update_load_info_label()
        self._update_global_plugins_ui()

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
        if not self._has_changes():
            super().accept()
            return

        global_plugins_enabled_changed = self._has_global_plugins_enabled_change()
        reloadable_changes = self._has_reloadable_changes()
        self._save_configuration()

        if global_plugins_enabled_changed:
            self.main.set_plugins_enabled(self.plugins_enabled)
            super().accept()
            return

        if not reloadable_changes:
            super().accept()
            return

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
