# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Welcome page
============

The :mod:`datalab.gui.welcome` module provides the welcome page: a dockable
panel gathering the main actions to get started with DataLab (equivalent of
the DataLab-Web welcome page).

.. autoclass:: WelcomeEntry

.. autoclass:: WelcomePanel
"""

from __future__ import annotations

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
from datalab.config import APP_DESC, Conf, _
from datalab.gui.actionhandler import ActionCategory

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel

#: Width below which the two columns of the welcome page are stacked
SINGLE_COLUMN_WIDTH = 720

ENTRY_STYLESHEET = """
QFrame#welcome_entry {
    border: 1px solid transparent;
    border-radius: 4px;
}
QFrame#welcome_entry:hover, QFrame#welcome_entry:focus {
    border-color: palette(highlight);
    background-color: palette(alternate-base);
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
        icon_name: icon file name
        title: entry title
        description: entry description
        parent: parent widget
    """

    SIG_CLICKED = QC.Signal()

    def __init__(
        self,
        icon_name: str,
        title: str,
        description: str,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("welcome_entry")
        self.setStyleSheet(ENTRY_STYLESHEET)
        self.setAttribute(QC.Qt.WA_Hover)
        self.setFocusPolicy(QC.Qt.StrongFocus)
        self.setCursor(QC.Qt.PointingHandCursor)
        self.setAccessibleName(title)
        self.setAccessibleDescription(description)

        icon_label = QW.QLabel()
        icon_label.setPixmap(get_icon(icon_name).pixmap(32, 32))
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
        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)
        layout.addWidget(icon_label, 0, QC.Qt.AlignTop)
        layout.addLayout(text_layout, 1)

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
        self.start_title = self.__create_section_title("")
        self.columns_layout = QW.QBoxLayout(QW.QBoxLayout.LeftToRight)
        self.startup_checkbox = QW.QCheckBox(_("Show welcome page on startup"))
        self.startup_checkbox.setChecked(Conf.welcome_on_startup.get())
        self.startup_checkbox.toggled.connect(Conf.welcome_on_startup.set)
        self.__setup_ui()

    @staticmethod
    def __create_section_title(text: str) -> QW.QLabel:
        """Create a section title label"""
        label = QW.QLabel(text)
        font = label.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        font.setBold(True)
        label.setFont(font)
        return label

    def __add_entry(
        self,
        layout: QW.QVBoxLayout,
        key: str,
        icon_name: str,
        title: str,
        description: str,
        callback: Callable[[], None],
    ) -> None:
        """Create an entry, register it under ``key`` and add it to ``layout``"""
        entry = WelcomeEntry(icon_name, title, description, self)
        entry.SIG_CLICKED.connect(callback)
        self.entries[key] = entry
        layout.addWidget(entry)

    def __setup_ui(self) -> None:
        """Setup welcome page widgets"""
        logo = QW.QLabel()
        logo.setPixmap(QG.QPixmap(get_image_file_path("DataLab-Banner-200.png")))
        description = QW.QLabel(APP_DESC)
        description.setWordWrap(True)
        version = QW.QLabel(_("Version %s") % datalab.__version__)

        mw = self.mainwindow
        start_layout = QW.QVBoxLayout()
        start_layout.addWidget(self.start_title)
        for args in (
            (
                "create",
                "new_sig.svg",
                _("Create..."),
                _("Generate a 1D signal or 2D image from a template."),
                self.__popup_create_menu,
            ),
            (
                "open",
                "fileopen_sig.svg",
                _("Open file..."),
                _("Load a signal or image from your computer."),
                self.__open_files,
            ),
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
            (
                "open_h5",
                "fileopen_h5.svg",
                _("Open HDF5 workspace..."),
                _("Resume a previously saved DataLab workspace."),
                mw.openh5_action.trigger,
            ),
            (
                "import_text",
                "import_text.svg",
                _("Import text data..."),
                _("Bring in CSV / TSV / column data with the wizard."),
                self.__import_text,
            ),
        ):
            self.__add_entry(start_layout, *args)
        start_layout.addStretch(1)

        learn_layout = QW.QVBoxLayout()
        learn_layout.addWidget(self.__create_section_title(_("Walkthroughs")))
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
            self.__add_entry(learn_layout, *args)
        learn_layout.addStretch(1)

        self.columns_layout.setSpacing(24)
        self.columns_layout.addLayout(start_layout, 1)
        self.columns_layout.addLayout(learn_layout, 1)

        content = QW.QWidget()
        content.setMaximumWidth(880)
        content_layout = QW.QVBoxLayout(content)
        content_layout.setContentsMargins(16, 24, 16, 16)
        content_layout.setSpacing(12)
        content_layout.addWidget(logo)
        content_layout.addWidget(description)
        content_layout.addWidget(version)
        content_layout.addSpacing(12)
        content_layout.addLayout(self.columns_layout)
        content_layout.addSpacing(12)
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

    def resizeEvent(self, event: QG.QResizeEvent) -> None:  # pylint: disable=invalid-name
        """Stack the columns when the page is too narrow"""
        if event.size().width() < SINGLE_COLUMN_WIDTH:
            direction = QW.QBoxLayout.TopToBottom
        else:
            direction = QW.QBoxLayout.LeftToRight
        if self.columns_layout.direction() != direction:
            self.columns_layout.setDirection(direction)
        super().resizeEvent(event)
