# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Welcome page
============

The :mod:`datalab.gui.welcome` module provides the welcome page: a dockable
panel gathering the main actions to get started with DataLab (equivalent of
the DataLab-Web welcome page).

.. autoclass:: WelcomeEntry

.. autoclass:: TileGrid

.. autoclass:: WelcomePanel
"""

from __future__ import annotations

import functools
import re
import urllib.parse
import webbrowser
from typing import TYPE_CHECKING, Callable

from guidata.configtools import get_icon, get_image_file_path
from guidata.qthelpers import add_actions
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

import datalab
from datalab.config import Conf, _
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.plugins.applications import (
    get_application_plugins,
    get_plugin_icon,
    record_application_use,
    set_application_hidden,
    set_application_pinned,
    sort_welcome_applications,
)
from datalab.utils.qthelpers import qt_handle_error_message, try_or_log_error
from datalab.widgets.expandabletext import apply_subdued_color

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.plugins import PluginBase
    from datalab.plugins.tiles import WelcomeTile

#: Width below which the two columns of the welcome page are stacked
SINGLE_COLUMN_WIDTH = 720
#: Width of an application tile
TILE_WIDTH = 200
#: Spacing between application tiles
TILE_SPACING = 12
#: Size of an application tile icon
TILE_ICON_SIZE = 48
#: Size of an action entry icon
ENTRY_ICON_SIZE = 24

ENTRY_STYLESHEET = """
QFrame#welcome_entry {
    border: 1px solid transparent;
    border-radius: 6px;
}
QFrame#welcome_tile, QFrame#welcome_card {
    border: 1px solid palette(mid);
    border-radius: 6px;
}
QFrame#welcome_card {
    background-color: palette(alternate-base);
}
QFrame#welcome_entry:hover, QFrame#welcome_entry:focus,
QFrame#welcome_tile:hover, QFrame#welcome_tile:focus {
    border-color: palette(highlight);
    background-color: palette(alternate-base);
}
QFrame#welcome_card:hover, QFrame#welcome_card:focus {
    border-color: palette(highlight);
}
"""


def get_release_notes_url() -> str:
    """Return the online release notes URL of the running DataLab version

    Returns:
        Release notes URL
    """
    major, minor = re.match(r"(\d+)\.(\d+)", datalab.__version__).groups()
    page = f"en/release_notes/release_{major}.{int(minor):02d}.html"
    return urllib.parse.urljoin(datalab.__docurl__, page)


class WelcomeEntry(QW.QFrame):
    """Clickable welcome page entry, showing an icon, a title and a description

    Args:
        icon: icon or icon file name
        title: entry title
        description: entry description
        parent: parent widget
        tile: if True, show the entry as a fixed-width application tile, with a
         larger icon above its title
        menu: if True (tile only), add a button requesting a menu of actions,
         also available from the context menu
        card: if True, show the entry as a framed card
        caret: if True, show a caret telling that the entry opens a menu
    """

    SIG_CLICKED = QC.Signal()
    #: Emitted with the global position where to show the menu of actions
    SIG_MENU_REQUESTED = QC.Signal(QC.QPoint)

    def __init__(
        self,
        icon: QG.QIcon | str,
        title: str,
        description: str,
        parent: QW.QWidget | None = None,
        tile: bool = False,
        menu: bool = False,
        card: bool = False,
        caret: bool = False,
    ) -> None:
        super().__init__(parent)
        if tile:
            self.setObjectName("welcome_tile")
        else:
            self.setObjectName("welcome_card" if card else "welcome_entry")
        self.setStyleSheet(ENTRY_STYLESHEET)
        self.setAttribute(QC.Qt.WA_Hover)
        self.setFocusPolicy(QC.Qt.StrongFocus)
        self.setCursor(QC.Qt.PointingHandCursor)
        self.setAccessibleName(title)
        self.setAccessibleDescription(description)
        self.menu_button: QW.QToolButton | None = None

        if isinstance(icon, str):
            icon = get_icon(icon)
        icon_size = TILE_ICON_SIZE if tile else ENTRY_ICON_SIZE
        self.icon_label = QW.QLabel()
        self.icon_label.setPixmap(icon.pixmap(icon_size, icon_size))
        self.title_label = QW.QLabel(title)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        description_label = QW.QLabel(description)
        description_label.setWordWrap(True)

        text_layout = QW.QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(description_label)
        if tile:
            self.title_label.setWordWrap(True)
            self.setFixedWidth(TILE_WIDTH)
            top_layout = QW.QHBoxLayout()
            top_layout.addWidget(self.icon_label)
            top_layout.addStretch(1)
            if menu:
                self.menu_button = QW.QToolButton()
                self.menu_button.setText("\u2026")
                button_font = self.menu_button.font()
                button_font.setBold(True)
                button_font.setPointSizeF(button_font.pointSizeF() * 1.3)
                self.menu_button.setFont(button_font)
                self.menu_button.setToolTip(_("More actions"))
                self.menu_button.setAutoRaise(True)
                # The tile context menu key already gives keyboard access
                self.menu_button.setFocusPolicy(QC.Qt.NoFocus)
                self.menu_button.clicked.connect(self.__request_button_menu)
                top_layout.addWidget(self.menu_button, 0, QC.Qt.AlignTop)
            layout = QW.QVBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(8)
            layout.addLayout(top_layout)
            layout.addLayout(text_layout)
            layout.addStretch(1)
        else:
            apply_subdued_color(description_label)
            layout = QW.QHBoxLayout(self)
            if card:
                layout.setContentsMargins(14, 12, 14, 12)
            else:
                layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(12)
            layout.addWidget(self.icon_label, 0, QC.Qt.AlignTop)
            layout.addLayout(text_layout, 1)
            if caret:
                caret_label = QW.QLabel("\u25be")
                caret_font = caret_label.font()
                caret_font.setPointSizeF(caret_font.pointSizeF() * 1.4)
                caret_label.setFont(caret_font)
                apply_subdued_color(caret_label)
                layout.addWidget(caret_label, 0, QC.Qt.AlignVCenter)

    def set_title(self, title: str) -> None:
        """Change the entry title

        Args:
            title: new title
        """
        self.title_label.setText(title)
        self.setAccessibleName(title)

    def __request_button_menu(self) -> None:
        """Request the menu of actions below the menu button"""
        button = self.menu_button
        self.SIG_MENU_REQUESTED.emit(button.mapToGlobal(QC.QPoint(0, button.height())))

    def contextMenuEvent(self, event: QG.QContextMenuEvent) -> None:  # pylint: disable=invalid-name
        """Request the menu of actions, if any, at the context menu position"""
        if self.menu_button is None:
            super().contextMenuEvent(event)
        else:
            self.SIG_MENU_REQUESTED.emit(event.globalPos())

    def mouseReleaseEvent(self, event: QG.QMouseEvent) -> None:  # pylint: disable=invalid-name
        """Emit the clicked signal on left button release"""
        if event.button() == QC.Qt.LeftButton:
            self.SIG_CLICKED.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QG.QKeyEvent) -> None:  # pylint: disable=invalid-name
        """Emit the clicked signal when activated from the keyboard"""
        if event.key() in (QC.Qt.Key_Return, QC.Qt.Key_Enter, QC.Qt.Key_Space):
            self.SIG_CLICKED.emit()
        else:
            super().keyPressEvent(event)


class TileGrid(QW.QWidget):
    """Grid of fixed-width tiles, wrapped on a limited number of rows

    Tiles come in groups: a main tile followed by secondary tiles. The number of
    columns follows the available width. When all groups fit, the free places
    show the secondary tiles of as many groups as possible, in display order,
    each group being shown whole or folded to its main tile. When the groups do
    not fit, all of them are folded and the last place of the grid shows an
    overflow tile, giving the number of groups left out.

    Args:
        overflow_tile: tile shown in place of the groups that do not fit
        parent: parent widget
    """

    def __init__(
        self, overflow_tile: WelcomeEntry, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        # Follow the available width instead of imposing the current grid width
        self.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Preferred)
        self.tiles: list[QW.QWidget] = []
        self.groups: list[list[QW.QWidget]] = []
        self.placed: list[QW.QWidget] = []
        self.expanded: set[QW.QWidget] = set()
        self.max_rows = 1
        self.columns = 0
        self.overflow_tile = overflow_tile
        overflow_tile.setParent(self)
        overflow_tile.hide()
        layout = QW.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TILE_SPACING)
        layout.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignTop)

    def __clear(self) -> None:
        """Remove the placed tiles from the grid"""
        for tile in self.placed:
            self.layout().removeWidget(tile)
            tile.hide()
        self.placed = []
        self.expanded = set()

    def set_tiles(self, tiles: list[QW.QWidget]) -> None:
        """Replace the tiles of the grid

        The new tiles are shown once :meth:`set_order` has placed them.

        Args:
            tiles: new tiles
        """
        self.__clear()
        for tile in self.tiles:
            tile.deleteLater()
        self.tiles = list(tiles)
        for tile in self.tiles:
            tile.setParent(self)
            tile.hide()
        self.set_order([], self.max_rows)

    def set_order(self, groups: list[list[QW.QWidget]], max_rows: int) -> None:
        """Choose the tiles to show and their order

        Args:
            groups: groups of tiles to show (among the grid tiles), in display
             order, each one being a main tile followed by its secondary tiles
            max_rows: maximum number of rows
        """
        self.groups = [list(group) for group in groups]
        self.max_rows = max(1, max_rows)
        self.columns = 0
        self.arrange(self.width())

    def is_expanded(self, tile: QW.QWidget) -> bool:
        """Return True if the secondary tiles of a main tile are shown

        Args:
            tile: main tile of a group

        Returns:
            True if the group is shown whole
        """
        return tile in self.expanded

    def arrange(self, width: int) -> None:
        """Place the tiles on as many columns as the width allows

        Args:
            width: available width
        """
        columns = max(1, (width + TILE_SPACING) // (TILE_WIDTH + TILE_SPACING))
        if columns == self.columns:
            return
        self.columns = columns
        self.__clear()
        capacity = columns * self.max_rows
        placed: list[QW.QWidget] = []
        expanded: set[QW.QWidget] = set()
        if len(self.groups) > capacity:
            left_out = len(self.groups) - capacity + 1
            self.overflow_tile.set_title(_("More applications (%d)") % left_out)
            placed = [group[0] for group in self.groups[: capacity - 1]]
            placed.append(self.overflow_tile)
        else:
            free = capacity - len(self.groups)
            for group in self.groups:
                if 1 < len(group) <= free + 1:
                    free -= len(group) - 1
                    expanded.add(group[0])
                    placed.extend(group)
                else:
                    placed.append(group[0])
        layout = self.layout()
        for index, tile in enumerate(placed):
            layout.addWidget(tile, index // columns, index % columns)
        self.placed = placed
        self.expanded = expanded
        for tile in placed:
            if self.placed is not placed:
                # Showing a tile resized the grid, which placed the tiles again
                break
            tile.show()

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Rearrange the tiles for the new width"""
        super().resizeEvent(event)
        self.arrange(event.size().width())


class WelcomePanel(QW.QWidget, DockableWidgetMixin):
    """Welcome page dock, gathering the main actions to get started

    Args:
        mainwindow: DataLab main window
        parent: parent widget
    """

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("Welcome")

    def __init__(
        self, mainwindow: DLMainWindow, parent: QW.QWidget | None = None
    ) -> None:
        QW.QWidget.__init__(self, parent)
        DockableWidgetMixin.__init__(self)
        self.mainwindow = mainwindow
        self.entries: dict[str, WelcomeEntry] = {}
        self.application_plugins: dict[str, PluginBase] = {}
        self.application_tiles: dict[str, WelcomeEntry] = {}
        self.application_groups: dict[str, list[WelcomeEntry]] = {}
        self.applications_section = QW.QWidget()
        overflow_tile = WelcomeEntry(
            "libre-gui-plugin.svg",
            "",
            _("Browse all applications in the catalog."),
            tile=True,
        )
        overflow_tile.SIG_CLICKED.connect(self.__show_applications)
        self.tile_grid = TileGrid(overflow_tile)
        self.browse_applications_button = QW.QPushButton(
            get_icon("libre-gui-plugin.svg"), _("Browse all applications...")
        )
        self.start_title = self.__create_section_title("")
        self.header_layout = QW.QBoxLayout(QW.QBoxLayout.LeftToRight)
        self.columns_layout = QW.QBoxLayout(QW.QBoxLayout.LeftToRight)
        self.startup_checkbox = QW.QCheckBox(
            _("Show welcome page when the current panel is empty")
        )
        self.startup_checkbox.setChecked(Conf.welcome_on_startup.get())
        self.startup_checkbox.toggled.connect(Conf.welcome_on_startup.set)
        self.__setup_ui()

    @staticmethod
    def __create_section_title(text: str) -> QW.QLabel:
        """Create a small uppercase section title label"""
        label = QW.QLabel(text)
        font = label.font()
        font.setBold(True)
        font.setCapitalization(QG.QFont.AllUppercase)
        font.setLetterSpacing(QG.QFont.PercentageSpacing, 104)
        label.setFont(font)
        apply_subdued_color(label)
        return label

    def __add_entry(
        self,
        layout: QW.QVBoxLayout,
        key: str,
        icon_name: str,
        title: str,
        description: str,
        callback: Callable[[], None],
        **kwargs: bool,
    ) -> None:
        """Create an entry, register it under ``key`` and add it to ``layout``

        Keyword arguments (``card``, ``caret``) are passed to the entry.
        """
        entry = WelcomeEntry(icon_name, title, description, self, **kwargs)
        entry.SIG_CLICKED.connect(callback)
        self.entries[key] = entry
        layout.addWidget(entry)

    def __setup_applications_section(self) -> None:
        """Setup the section gathering the application plugin tiles"""
        button = self.browse_applications_button
        button.setFlat(True)
        button.setCursor(QC.Qt.PointingHandCursor)
        button.clicked.connect(self.__show_applications)
        header_layout = QW.QHBoxLayout()
        header_layout.addWidget(self.__create_section_title(_("Applications")))
        header_layout.addStretch(1)
        header_layout.addWidget(button)
        layout = QW.QVBoxLayout(self.applications_section)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addLayout(header_layout)
        layout.addWidget(self.tile_grid)
        self.applications_section.hide()

    def __show_applications(self) -> None:
        """Open the application catalog"""
        self.mainwindow.show_applications()

    def refresh_application_tiles(self) -> None:
        """Rebuild the application tiles from the active application plugins

        Each application gets a main tile, followed by its other welcome tiles
        when the section has room for them; otherwise, they are actions of the
        main tile menu.
        """
        self.application_plugins.clear()
        self.application_tiles.clear()
        self.application_groups.clear()
        for plugin in get_application_plugins():
            with try_or_log_error(f"Creating welcome tiles for {plugin.info.name}"):
                main_tile, *other_tiles = plugin.get_welcome_tiles()
                group = []
                for tile in (main_tile, *other_tiles):
                    entry = WelcomeEntry(
                        get_plugin_icon(tile.icon),
                        tile.title,
                        tile.description,
                        tile=True,
                        menu=tile is main_tile,
                    )
                    entry.SIG_CLICKED.connect(
                        functools.partial(self.__launch_tile, plugin, tile)
                    )
                    group.append(entry)
                group[0].SIG_MENU_REQUESTED.connect(
                    functools.partial(self.__popup_tile_menu, plugin, other_tiles)
                )
                self.application_plugins[plugin.plugin_id] = plugin
                self.application_tiles[plugin.plugin_id] = group[0]
                self.application_groups[plugin.plugin_id] = group
        self.tile_grid.set_tiles(
            [entry for group in self.application_groups.values() for entry in group]
        )
        self.update_application_tiles()

    def update_application_tiles(self) -> None:
        """Order and place the application tiles following user preferences

        Hidden applications are left out; pinned applications come first, then
        recently used ones, then the others by name. Secondary tiles are shown
        when there is room for them, and applications that do not fit in the
        configured number of rows are replaced by an overflow tile.
        """
        plugins = sort_welcome_applications(self.application_plugins.values())
        self.tile_grid.set_order(
            [self.application_groups[plugin.plugin_id] for plugin in plugins],
            Conf.welcome_application_rows.get(),
        )
        # Keep the section (and its catalog button) when all tiles are hidden
        self.applications_section.setVisible(bool(self.application_tiles))

    def __create_tile_menu(
        self, plugin: PluginBase, tiles: list[WelcomeTile]
    ) -> QW.QMenu:
        """Create the menu of actions of an application tile

        Args:
            plugin: application plugin
            tiles: secondary welcome tiles of this plugin, listed unless they
             are shown in the grid

        Returns:
            Menu of actions
        """
        menu = QW.QMenu(self)
        if self.tile_grid.is_expanded(self.application_tiles[plugin.plugin_id]):
            tiles = []
        for tile in tiles:
            action = menu.addAction(get_plugin_icon(tile.icon), tile.title)
            action.setToolTip(tile.description)
            action.triggered.connect(
                lambda _checked=False, tile=tile: self.__launch_tile(plugin, tile)
            )
        if tiles:
            menu.addSeparator()
        plugin_id = plugin.plugin_id
        pinned = plugin_id in Conf.welcome_pinned_applications.get()
        pin_action = menu.addAction(_("Unpin") if pinned else _("Pin to the top"))
        pin_action.triggered.connect(
            lambda _checked=False: self.__set_pinned(plugin_id, not pinned)
        )
        hide_action = menu.addAction(_("Hide from welcome page"))
        hide_action.triggered.connect(lambda _checked=False: self.__hide(plugin_id))
        return menu

    def __popup_tile_menu(
        self, plugin: PluginBase, tiles: list[WelcomeTile], position: QC.QPoint
    ) -> None:
        """Show the menu of actions of an application tile

        Args:
            plugin: application plugin
            tiles: secondary welcome tiles of this plugin
            position: global position of the menu
        """
        menu = self.__create_tile_menu(plugin, tiles)
        menu.exec(position)
        menu.deleteLater()

    def __set_pinned(self, plugin_id: str, pinned: bool) -> None:
        """Pin an application to the top of the page, or unpin it"""
        set_application_pinned(plugin_id, pinned)
        self.update_application_tiles()

    def __hide(self, plugin_id: str) -> None:
        """Hide an application from the welcome page"""
        set_application_hidden(plugin_id, True)
        self.update_application_tiles()

    def __launch_tile(self, plugin: PluginBase, tile: WelcomeTile) -> None:
        """Launch an application tile, reporting errors raised by the plugin

        The use is remembered to order the tiles the next time the page is shown.

        Args:
            plugin: application plugin
            tile: welcome page tile of this plugin
        """
        record_application_use(plugin.plugin_id)
        try:
            plugin.launch_welcome_tile(tile.id)
        except Exception as exc:  # pylint: disable=broad-except
            # Plugin-owned launchers are third-party code: never crash the app
            qt_handle_error_message(
                self.mainwindow, exc, _("Launching '%s'") % tile.title
            )

    def __setup_header(self) -> None:
        """Setup the header: logo, slogan with version, and edition"""
        logo = QW.QLabel()
        logo.setPixmap(QG.QPixmap(get_image_file_path("DataLab-Banner-200.png")))
        slogan = QW.QLabel(
            _("Scientific signal and image processing \u2014 version %s")
            % datalab.__version__
        )
        slogan.setWordWrap(True)
        font = slogan.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        slogan.setFont(font)
        apply_subdued_color(slogan)
        edition = QW.QLabel(_("Desktop edition of DataLab"))
        font = edition.font()
        font.setItalic(True)
        font.setPointSizeF(font.pointSizeF() * 0.9)
        edition.setFont(font)
        apply_subdued_color(edition)
        title_layout = QW.QVBoxLayout()
        title_layout.setSpacing(2)
        title_layout.addStretch(1)
        title_layout.addWidget(slogan)
        title_layout.addWidget(edition)
        title_layout.addStretch(1)
        self.header_layout.setSpacing(18)
        self.header_layout.addWidget(logo, 0, QC.Qt.AlignLeft | QC.Qt.AlignVCenter)
        self.header_layout.addLayout(title_layout, 1)

    def __setup_ui(self) -> None:
        """Setup welcome page widgets"""
        self.__setup_header()

        mw = self.mainwindow
        start_layout = QW.QVBoxLayout()
        start_layout.setSpacing(6)
        start_layout.addWidget(self.start_title)
        start_layout.addSpacing(4)
        for args, caret in (
            (
                (
                    "create",
                    "new_sig.svg",
                    _("Create..."),
                    _("Generate a 1D signal or 2D image from a template."),
                    self.__popup_create_menu,
                ),
                True,
            ),
            (
                (
                    "open",
                    "fileopen_sig.svg",
                    _("Open file..."),
                    _("Load a signal or image from your computer."),
                    self.__open_files,
                ),
                True,
            ),
            (
                (
                    "browse_h5",
                    "h5browser.svg",
                    _("Browse HDF5 file..."),
                    _(
                        "Inspect any HDF5 file and import selected datasets "
                        "as signals or images."
                    ),
                    mw.browseh5_action.trigger,
                ),
                False,
            ),
            (
                (
                    "open_h5",
                    "fileopen_h5.svg",
                    _("Open HDF5 workspace..."),
                    _("Resume a previously saved DataLab workspace."),
                    mw.openh5_action.trigger,
                ),
                False,
            ),
            (
                (
                    "import_text",
                    "import_text.svg",
                    _("Import text data..."),
                    _("Bring in CSV / TSV / column data with the wizard."),
                    self.__import_text,
                ),
                True,
            ),
        ):
            self.__add_entry(start_layout, *args, caret=caret)
        start_layout.addStretch(1)

        learn_layout = QW.QVBoxLayout()
        learn_layout.setSpacing(6)
        learn_layout.addWidget(self.__create_section_title(_("Walkthroughs")))
        learn_layout.addSpacing(4)
        for args in (
            (
                "ai_assistant",
                "ai-assistant.svg",
                _("Ask the AI assistant"),
                _(
                    "Chat with the built-in assistant to inspect, create "
                    "and process your data."
                ),
                self.__show_ai_assistant,
            ),
            (
                "tour",
                "tour.svg",
                _("Take the guided tour"),
                _("A short interactive walk-through of the DataLab workspace."),
                mw.show_tour,
            ),
            (
                "demo",
                "play_demo.svg",
                _("Run the demo"),
                _("Watch DataLab create, process and analyze signals and images."),
                mw.play_demo,
            ),
            (
                "documentation",
                "libre-gui-help.svg",
                _("Read the documentation"),
                _("Open the online user guide, tutorials and reference."),
                lambda: webbrowser.open(Conf.app_docurl.get()),
            ),
            (
                "release_notes",
                "libre-gui-about.svg",
                _("What's new in version %s") % datalab.__version__,
                _("Browse the release notes of this version."),
                lambda: webbrowser.open(get_release_notes_url()),
            ),
        ):
            self.__add_entry(learn_layout, *args, card=True)
        learn_layout.addStretch(1)

        self.columns_layout.setSpacing(28)
        self.columns_layout.addLayout(start_layout, 1)
        self.columns_layout.addLayout(learn_layout, 1)

        separator = QW.QFrame()
        separator.setObjectName("welcome_separator")
        separator.setFixedHeight(1)
        separator.setStyleSheet(
            "QFrame#welcome_separator { background-color: palette(mid); }"
        )

        content = QW.QWidget()
        content.setMaximumWidth(880)
        content_layout = QW.QVBoxLayout(content)
        content_layout.setContentsMargins(16, 24, 16, 16)
        content_layout.setSpacing(12)
        content_layout.addLayout(self.header_layout)
        content_layout.addSpacing(16)
        self.__setup_applications_section()
        content_layout.addWidget(self.applications_section)
        content_layout.addLayout(self.columns_layout)
        content_layout.addSpacing(16)
        content_layout.addWidget(separator)
        content_layout.addWidget(self.startup_checkbox)
        content_layout.addStretch(1)

        container = QW.QWidget()
        container_layout = QW.QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(content, 100)
        container_layout.addStretch(1)

        scroll = QW.QScrollArea()
        scroll.setFrameShape(QW.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def __data_panels(self) -> tuple[tuple[BaseDataPanel, str], ...]:
        """Return the signal and image panels with their translated labels"""
        mw = self.mainwindow
        return ((mw.signalpanel, _("Signal")), (mw.imagepanel, _("Image")))

    def __popup_position(self, key: str) -> QC.QPoint:
        """Return the global position of a popup menu below the ``key`` entry"""
        entry = self.entries[key]
        return entry.mapToGlobal(QC.QPoint(0, entry.height()))

    def __choose_panel(self, key: str) -> BaseDataPanel | None:
        """Let the user choose between the signal and image panels

        Args:
            key: key of the entry below which the choice menu is shown

        Returns:
            Chosen panel, or None if the menu was dismissed
        """
        menu = QW.QMenu(self)
        choices = {}
        for panel, label in self.__data_panels():
            action = menu.addAction(get_icon(f"{panel.PANEL_STR_ID}.svg"), label)
            choices[action] = panel
        chosen = menu.exec(self.__popup_position(key))
        menu.deleteLater()
        return choices.get(chosen)

    def __popup_create_menu(self) -> None:
        """Show the signal and image creation actions"""
        menu = QW.QMenu(self)
        for panel, label in self.__data_panels():
            submenu = menu.addMenu(get_icon(f"{panel.PANEL_STR_ID}.svg"), label)
            add_actions(submenu, panel.get_category_actions(ActionCategory.CREATE))
        menu.exec(self.__popup_position("create"))
        menu.deleteLater()

    def __open_files(self) -> None:
        """Open signal or image files"""
        panel = self.__choose_panel("open")
        if panel is not None:
            self.mainwindow.set_current_panel(panel)
            panel.load_from_files()

    def __import_text(self) -> None:
        """Import text data as signals or images"""
        panel = self.__choose_panel("import_text")
        if panel is not None:
            self.mainwindow.set_current_panel(panel)
            panel.exec_import_wizard()

    def __show_ai_assistant(self) -> None:
        """Show and raise the AI assistant dock"""
        dock = self.mainwindow.docks[self.mainwindow.aiassistantpanel]
        dock.show()
        dock.raise_()

    def visibility_changed(self, enable: bool) -> None:
        """Refresh the page contents when its dock is shown or its tab selected

        Args:
            enable: dock widget visibility state
        """
        super().visibility_changed(enable)
        if enable:
            mw = self.mainwindow
            empty = not any(len(panel) for panel in (mw.signalpanel, mw.imagepanel))
            title = _("Get started") if empty else _("Quick actions")
            self.start_title.setText(title)
            self.startup_checkbox.setChecked(Conf.welcome_on_startup.get())
            # Bring recently used applications forward, out of the user's sight
            self.update_application_tiles()

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Stack the columns when the page is too narrow"""
        if event.size().width() < SINGLE_COLUMN_WIDTH:
            direction = QW.QBoxLayout.TopToBottom
        else:
            direction = QW.QBoxLayout.LeftToRight
        for layout in (self.header_layout, self.columns_layout):
            if layout.direction() != direction:
                layout.setDirection(direction)
        super().resizeEvent(event)
