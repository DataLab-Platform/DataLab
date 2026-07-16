"""
Command palette
---------------

VSCode-style command palette for DataLab: a searchable, keyboard-driven
dialog listing every menu command by its localised menu path (e.g.
"Processing › Fourier analysis › FFT"). Selecting an entry triggers the
underlying :class:`QAction`.

This mirrors the DataLab-Web command palette (``src/components/CommandPalette.tsx``
and ``src/actions/commandSearch.ts``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.gui.actionhandler import ActionCategory

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel

#: Separator between menu-path segments in the command palette
#: (e.g. "Processing › Fourier analysis › FFT").
COMMAND_PATH_SEPARATOR = " \u203a "

#: Characters that start a new "word" in a command path. A query character
#: matching right after one of these earns a word-boundary bonus.
_BOUNDARY_CHARS = set(" \u203a/-_.([:")


def fuzzy_score(query: str, text: str) -> int | None:
    """Fuzzy-match ``query`` against ``text`` (subsequence with scoring).

    Every character of ``query`` must appear in ``text`` in order;
    consecutive matches and matches at word boundaries are rewarded, while
    gaps between matched characters are penalised.

    Args:
        query: Lowercased, trimmed search query.
        text: Lowercased haystack to match against.

    Returns:
        A score (higher is better) when ``query`` is a subsequence of
        ``text``, otherwise ``None``.
    """
    if not query:
        return 0
    if len(query) > len(text):
        return None
    score = 0
    text_index = 0
    prev = -2
    for char in query:
        found = text.find(char, text_index)
        if found == -1:
            return None
        score += 1
        if found == prev + 1:
            score += 5
        if found == 0 or text[found - 1] in _BOUNDARY_CHARS:
            score += 3
        if prev >= 0:
            score -= min(found - prev - 1, 3)
        prev = found
        text_index = found + 1
    return score


def _iter_command_leaves(
    items: list[QW.QAction | QW.QMenu | None],
    prefix: str,
    out: list[tuple[QW.QAction, str]],
    seen: set[int],
) -> None:
    """Recursively collect leaf actions with their localised menu path.

    Args:
        items: Menu entries (actions, submenus or ``None`` separators).
        prefix: Localised path of the parent menu (already separator-joined).
        out: Accumulator of ``(action, path)`` pairs.
        seen: Ids of already-collected actions (de-duplication).
    """
    for item in items:
        if item is None:
            continue
        if isinstance(item, QW.QMenu):
            submenu, title = item, item.title()
        elif isinstance(item, QW.QAction):
            if item.isSeparator():
                continue
            child_menu = item.menu()
            if child_menu is not None:
                submenu, title = child_menu, item.text()
            else:
                text = item.text().replace("&", "").strip()
                if text and id(item) not in seen:
                    seen.add(id(item))
                    out.append((item, prefix + text))
                continue
        else:
            continue
        title = title.replace("&", "").strip()
        _iter_command_leaves(
            submenu.actions(), prefix + title + COMMAND_PATH_SEPARATOR, out, seen
        )


def collect_commands(
    window: DLMainWindow, panel: BaseDataPanel
) -> list[tuple[QW.QAction, str]]:
    """Collect every menu command available for ``panel``.

    Walks the action categories of the current data panel (the same source
    that populates the menu bar) and builds, for each leaf action, its
    localised menu path (e.g. "Processing › Fourier analysis › FFT").

    Args:
        window: Main window (provides the localised category titles).
        panel: Current data panel (signal or image).

    Returns:
        List of ``(action, menu_path)`` pairs.
    """
    category_menus = (
        (ActionCategory.FILE, window.file_menu),
        (ActionCategory.CREATE, window.create_menu),
        (ActionCategory.EDIT, window.edit_menu),
        (ActionCategory.ROI, window.roi_menu),
        (ActionCategory.OPERATION, window.operation_menu),
        (ActionCategory.PROCESSING, window.processing_menu),
        (ActionCategory.ANALYSIS, window.analysis_menu),
        (ActionCategory.VIEW, window.view_menu),
        (ActionCategory.PLUGINS, window.plugins_menu),
    )
    commands: list[tuple[QW.QAction, str]] = []
    seen: set[int] = set()
    for category, menu in category_menus:
        title = menu.title().replace("&", "").strip()
        actions = panel.get_category_actions(category)
        _iter_command_leaves(actions, title + COMMAND_PATH_SEPARATOR, commands, seen)
    return commands


class CommandPaletteDialog(QW.QDialog):
    """VSCode-style command palette.

    A searchable, keyboard-driven list of every menu command, each
    identified by its localised menu path. Selecting an entry triggers the
    underlying :class:`QAction`. Disabled commands are shown greyed and are
    not selectable.

    Args:
        parent: Parent widget (main window).
        commands: ``(action, menu_path)`` pairs to list.
    """

    def __init__(
        self, parent: QW.QWidget, commands: list[tuple[QW.QAction, str]]
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Command palette"))
        self.setWindowIcon(get_icon("command_palette.svg"))
        self.setModal(True)
        self._commands = commands
        self._selected_action: QW.QAction | None = None

        layout = QW.QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.search = QW.QLineEdit()
        self.search.setPlaceholderText(_("Type to search for a command…"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._update_list)
        self.search.installEventFilter(self)
        layout.addWidget(self.search)

        self.listwidget = QW.QListWidget()
        self.listwidget.setAlternatingRowColors(True)
        self.listwidget.itemClicked.connect(self._activate_item)
        layout.addWidget(self.listwidget)

        self.setLayout(layout)
        self.resize(640, 420)
        self._update_list("")
        self.search.setFocus()

    def get_selected_action(self) -> QW.QAction | None:
        """Return the action chosen by the user (or ``None``)."""
        return self._selected_action

    def _update_list(self, text: str) -> None:
        """Rebuild the result list from the current search ``text``."""
        self.listwidget.clear()
        query = text.strip().lower()
        scored: list[tuple[int, str, QW.QAction]] = []
        for action, path in self._commands:
            score = fuzzy_score(query, path.lower())
            if score is not None:
                scored.append((score, path, action))
        if query:
            scored.sort(key=lambda entry: (-entry[0], entry[1]))
        else:
            scored.sort(key=lambda entry: entry[1])
        first_enabled: int | None = None
        for _score, path, action in scored:
            item = QW.QListWidgetItem(action.icon(), path)
            item.setData(QC.Qt.UserRole, action)
            if action.isEnabled():
                if first_enabled is None:
                    first_enabled = self.listwidget.count()
            else:
                item.setFlags(item.flags() & ~QC.Qt.ItemIsEnabled)
            self.listwidget.addItem(item)
        if first_enabled is not None:
            self.listwidget.setCurrentRow(first_enabled)

    def _activate_item(self, item: QW.QListWidgetItem | None) -> None:
        """Accept the dialog with ``item``'s action if it is enabled."""
        if item is None:
            return
        action = item.data(QC.Qt.UserRole)
        if action is not None and action.isEnabled():
            self._selected_action = action
            self.accept()

    def _step_row(self, direction: int) -> None:
        """Move the selection by ``direction`` (±1), skipping disabled rows."""
        count = self.listwidget.count()
        if count == 0:
            return
        row = self.listwidget.currentRow()
        row = (0 if direction > 0 else count - 1) if row < 0 else row + direction
        for _i in range(count):
            if row < 0:
                row = count - 1
            elif row >= count:
                row = 0
            if self.listwidget.item(row).flags() & QC.Qt.ItemIsEnabled:
                self.listwidget.setCurrentRow(row)
                return
            row += direction

    def eventFilter(  # pylint: disable=invalid-name
        self, widget: QC.QObject, event: QC.QEvent
    ) -> bool:
        """Route Up/Down/Enter from the search field to the result list."""
        if widget is self.search and event.type() == QC.QEvent.KeyPress:
            key = event.key()
            if key == QC.Qt.Key_Down:
                self._step_row(1)
                return True
            if key == QC.Qt.Key_Up:
                self._step_row(-1)
                return True
            if key in (QC.Qt.Key_Return, QC.Qt.Key_Enter):
                self._activate_item(self.listwidget.currentItem())
                return True
        return super().eventFilter(widget, event)


class CommandSearchField(QW.QFrame):
    """Search-box-styled launcher for the command palette.

    A clickable, search-box-looking widget shown in the menu-bar corner so
    the command palette is discoverable at a glance (mirrors the search
    affordance of the DataLab-Web button). It is composed of an icon, a
    placeholder label and a shortcut label laid out horizontally, so it
    always reports a size large enough to display its full content — unlike
    a plain ``QLineEdit`` placeholder, which the menu bar clips (its size
    hint ignores the placeholder, and theme padding/fonts make it worse in
    dark mode). Clicking the field — or pressing Enter/Space while it has
    focus — opens the palette.

    Args:
        parent: Parent widget.
        on_open: Callback opening the command palette.
        shortcut_text: Human-readable shortcut shown on the right
         (e.g. "Ctrl+Maj+P").
    """

    def __init__(
        self,
        parent: QW.QWidget,
        on_open: Callable[[], None],
        shortcut_text: str = "",
    ) -> None:
        super().__init__(parent)
        self._on_open = on_open
        self.setObjectName("commandSearchField")
        self.setCursor(QC.Qt.PointingHandCursor)
        self.setFocusPolicy(QC.Qt.StrongFocus)
        tooltip = _("Command palette")
        if shortcut_text:
            tooltip = f"{tooltip} ({shortcut_text})"
        self.setToolTip(tooltip)
        # Subtle search-box look that adapts to both light and dark themes
        # (semi-transparent grey works on either background).
        self.setStyleSheet(
            "#commandSearchField {"
            " border: 1px solid rgba(127, 127, 127, 0.5);"
            " border-radius: 4px;"
            " background: rgba(127, 127, 127, 0.08); }"
            "#commandSearchField:hover {"
            " border-color: rgba(127, 127, 127, 0.9); }"
        )

        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        icon_label = QW.QLabel(self)
        icon_label.setPixmap(get_icon("command_palette.svg").pixmap(14, 14))
        layout.addWidget(icon_label)

        self.placeholder_label = QW.QLabel(_("Search a command…"), self)
        self.placeholder_label.setStyleSheet("color: gray; border: none;")
        layout.addWidget(self.placeholder_label)

        if shortcut_text:
            layout.addSpacing(16)
            self.shortcut_label = QW.QLabel(shortcut_text, self)
            self.shortcut_label.setStyleSheet("color: gray; border: none;")
            layout.addWidget(self.shortcut_label)
        else:
            self.shortcut_label = None

    def mousePressEvent(  # pylint: disable=invalid-name,unused-argument
        self, event: QG.QMouseEvent
    ) -> None:
        """Open the command palette on click."""
        self._on_open()

    def keyPressEvent(  # pylint: disable=invalid-name
        self, event: QG.QKeyEvent
    ) -> None:
        """Open the command palette on Enter/Space, ignore other keys."""
        if event.key() in (QC.Qt.Key_Return, QC.Qt.Key_Enter, QC.Qt.Key_Space):
            self._on_open()
        else:
            super().keyPressEvent(event)
