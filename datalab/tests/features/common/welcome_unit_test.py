# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Welcome page unit tests
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import create_signal
from sigimax.utils import qthelpers as sgmx_qth

import datalab
from datalab.config import Conf, _
from datalab.gui import welcome
from datalab.gui.applications import ApplicationsDialog
from datalab.gui.main import DLMainWindow
from datalab.plugin_tiles import WelcomeTile
from datalab.plugins import PluginBase, PluginCapability, PluginInfo, PluginRegistry
from datalab.tests import datalab_test_app_context

PLUGIN_ICON = "datalab:data/icons/libre-gui-plugin.svg"


def test_release_notes_url(monkeypatch) -> None:
    """Release notes URL follows the documentation page naming"""
    monkeypatch.setattr(datalab, "__version__", "1.4.0rc1")
    url = welcome.get_release_notes_url()
    assert url == "https://datalab-platform.com/en/release_notes/release_1.04.html"


def is_in_front(dock: QW.QDockWidget) -> bool:
    """Return True if the dock is the selected tab of its group"""
    # Qt keeps background dock tabs visible: only their visible region is empty
    return not dock.visibleRegion().isEmpty()


def test_welcome_page(monkeypatch) -> None:
    """Welcome page dock, entries and startup preference"""
    calls: list[object] = []
    monkeypatch.setattr(DLMainWindow, "show_tour", lambda self: calls.append("tour"))
    monkeypatch.setattr(DLMainWindow, "play_demo", lambda self: calls.append("demo"))
    monkeypatch.setattr(welcome.webbrowser, "open", calls.append)

    def choose_image(menu: QW.QMenu, *_args) -> QW.QAction:
        return next(act for act in menu.actions() if act.text() == _("Image"))

    def create_gaussian(menu: QW.QMenu, *_args) -> None:
        sig_menu = next(
            act.menu() for act in menu.actions() if act.text() == _("Signal")
        )
        next(act for act in sig_menu.actions() if act.text() == _("Gaussian")).trigger()

    with datalab_test_app_context(console=False) as win:
        page = win.welcomepanel
        dock = win.docks[page]
        sigdock = win.docks[win.signalpanel]
        assert dock in win.tabifiedDockWidgets(sigdock)
        assert page not in win.panels

        win.show_welcome_page()
        QW.QApplication.processEvents()
        assert is_in_front(dock) and not is_in_front(sigdock)
        assert page.start_title.text() == _("Get started")

        # Panel-specific entries ask for the destination panel first
        monkeypatch.setattr(QW.QMenu, "exec", choose_image)
        for key, method in (
            ("open", "load_from_files"),
            ("import_text", "exec_import_wizard"),
        ):
            monkeypatch.setattr(
                win.imagepanel, method, lambda key=key: calls.append(key)
            )
            page.entries[key].SIG_CLICKED.emit()
            assert win.tabwidget.currentWidget() is win.imagepanel
        monkeypatch.setattr(
            win, "open_h5_files", lambda import_all=None: calls.append(import_all)
        )
        for key in ("browse_h5", "open_h5", "tour", "demo"):
            page.entries[key].SIG_CLICKED.emit()
        for key in ("documentation", "release_notes"):
            page.entries[key].SIG_CLICKED.emit()
        assert calls == [
            "open",
            "import_text",
            None,
            True,
            "tour",
            "demo",
            Conf.app_docurl.get(),
            welcome.get_release_notes_url(),
        ]

        page.entries["ai_assistant"].SIG_CLICKED.emit()
        QW.QApplication.processEvents()
        assert is_in_front(win.docks[win.aiassistantpanel])

        # Creating an object brings the signal view back to the front
        monkeypatch.setattr(QW.QMenu, "exec", create_gaussian)
        win.show_welcome_page()
        page.entries["create"].SIG_CLICKED.emit()
        QW.QApplication.processEvents()
        assert len(win.signalpanel) == 1
        assert is_in_front(sigdock) and not is_in_front(dock)
        win.show_welcome_page()
        QW.QApplication.processEvents()
        assert page.start_title.text() == _("Quick actions")

        initial = Conf.welcome_on_startup.get()
        try:
            page.startup_checkbox.setChecked(not initial)
            assert Conf.welcome_on_startup.get() is (not initial)
        finally:
            Conf.welcome_on_startup.set(initial)


def _application_plugin(
    plugin_id: str,
    tiles: tuple[WelcomeTile, ...] | Exception,
    launched: list[tuple[str, str]],
    capability: PluginCapability = PluginCapability.APPLICATION,
) -> object:
    """Build the minimal active-plugin surface consumed by the welcome page"""

    def get_welcome_tiles() -> tuple[WelcomeTile, ...]:
        if isinstance(tiles, Exception):
            raise tiles
        return tiles

    def launch_welcome_tile(tile_id: str) -> None:
        launched.append((plugin_id, tile_id))
        if tile_id == "broken":
            raise RuntimeError("Broken launcher")

    return SimpleNamespace(
        plugin_id=plugin_id,
        info=PluginInfo(
            id=plugin_id,
            name=plugin_id.rsplit(".", maxsplit=1)[-1].title(),
            capabilities=(capability,),
        ),
        get_welcome_tiles=get_welcome_tiles,
        launch_welcome_tile=launch_welcome_tile,
    )


def _choose_menu_action(text: str | None, menus: list[list[str]] | None = None):
    """Return a ``QMenu.exec`` replacement triggering the action with ``text``"""

    def exec_menu(menu: QW.QMenu, *_args) -> None:
        if menus is not None:
            menus.append([action.text() for action in menu.actions()])
        if text is not None:
            next(act for act in menu.actions() if act.text() == text).trigger()

    return exec_menu


def test_welcome_page_application_tiles(monkeypatch) -> None:
    """Application plugins add tiles, folded into the tile menu without room"""
    launched: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    shown: list[str | None] = []
    monkeypatch.setattr(
        welcome,
        "qt_handle_error_message",
        lambda _widget, exc, context: errors.append((str(exc), context)),
    )
    camera = _application_plugin(
        "org.example.camera",
        (WelcomeTile(id="application", title="Camera", icon=PLUGIN_ICON),),
        launched,
    )
    pulse = _application_plugin(
        "org.example.pulse",
        (
            WelcomeTile(id="application", title="Pulse"),
            WelcomeTile(
                id="demo", title="Demo", icon="play_demo.svg", launcher="open_demo"
            ),
            WelcomeTile(id="broken", title="Broken"),
        ),
        launched,
    )
    processing = _application_plugin(
        "org.example.processing", (), launched, PluginCapability.PROCESSING
    )
    registry = PluginRegistry.get_plugins()
    with datalab_test_app_context(console=False) as win:
        monkeypatch.setattr(
            win, "show_applications", lambda plugin_id=None: shown.append(plugin_id)
        )
        page = win.welcomepanel
        previous_plugins = list(registry)
        try:
            registry[:] = [pulse, processing, camera]
            # Leave room for all tiles, whatever the width of the page
            Conf.welcome_application_rows.set(10)
            page.refresh_application_tiles()
            win.show_welcome_page()
            QW.QApplication.processEvents()

            keys = ["org.example.camera", "org.example.pulse"]
            assert not page.applications_section.isHidden()
            assert list(page.application_tiles) == keys
            camera_group, pulse_group = (page.application_groups[key] for key in keys)
            assert len(camera_group) == 1 and len(pulse_group) == 3
            pulse_entry = pulse_group[0]
            assert page.tile_grid.placed == [*camera_group, *pulse_group]
            assert page.tile_grid.is_expanded(pulse_entry)
            assert pulse_group[2].menu_button is None
            for entry in (*camera_group, *pulse_group):
                assert not entry.icon_label.pixmap().isNull()
                entry.SIG_CLICKED.emit()
            assert launched == [
                ("org.example.camera", "application"),
                ("org.example.pulse", "application"),
                ("org.example.pulse", "demo"),
                ("org.example.pulse", "broken"),
            ]
            assert errors == [("Broken launcher", _("Launching '%s'") % "Broken")]

            # Secondary tiles are only listed in the menu when they are folded
            menus: list[list[str]] = []
            monkeypatch.setattr(QW.QMenu, "exec", _choose_menu_action(None, menus))
            pulse_entry.menu_button.click()
            monkeypatch.setattr(page.tile_grid, "is_expanded", lambda _tile: False)
            monkeypatch.setattr(QW.QMenu, "exec", _choose_menu_action("Demo", menus))
            pulse_entry.menu_button.click()
            pin_hide = [_("Pin to the top"), _("Hide from welcome page")]
            assert menus == [pin_hide, ["Demo", "Broken", "", *pin_hide]]
            assert launched[-1] == ("org.example.pulse", "demo")
            page.browse_applications_button.click()
            assert shown == [None]

            registry[:] = [processing]
            page.refresh_application_tiles()
            assert page.application_tiles == {}
            assert page.applications_section.isHidden()
        finally:
            registry[:] = previous_plugins


def test_tile_grid_wraps_and_overflows() -> None:
    """Tiles wrap to the width, an overflow tile replaces those left out"""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    overflow = welcome.WelcomeEntry("libre-gui-plugin.svg", "", "", tile=True)
    grid = welcome.TileGrid(overflow)
    layout = grid.layout()
    tiles = [QW.QWidget() for _index in range(8)]

    def position(tile: QW.QWidget) -> tuple[int, int]:
        return layout.getItemPosition(layout.indexOf(tile))[:2]

    def overflow_title(count: int) -> str:
        return _("More applications (%d)") % count

    try:
        grid.set_tiles(tiles)
        assert grid.placed == [] and all(tile.isHidden() for tile in tiles)

        # The grid is never shown: its width is only changed by the test
        grid.resize(welcome.TILE_WIDTH, 100)
        grid.set_order([[tile] for tile in tiles[:3]], max_rows=2)
        assert grid.columns == 1
        assert grid.placed == [tiles[0], overflow]
        assert position(overflow) == (1, 0)
        assert overflow.title_label.text() == overflow_title(2)
        grid.arrange(3 * welcome.TILE_WIDTH + 2 * welcome.TILE_SPACING)
        assert grid.columns == 3
        assert grid.placed == tiles[:3]
        assert position(tiles[2]) == (0, 2)
        assert overflow.isHidden() and not tiles[2].isHidden()

        grid.resize(3 * welcome.TILE_WIDTH + 2 * welcome.TILE_SPACING, 100)
        grid.set_order([[tile] for tile in tiles], max_rows=2)
        assert grid.placed == [*tiles[:5], overflow]
        assert position(overflow) == (1, 2)
        assert overflow.title_label.text() == overflow_title(3)
        assert all(tile.isHidden() for tile in tiles[5:])
        grid.set_order([[tile] for tile in reversed(tiles)], max_rows=1)
        assert grid.placed == [tiles[7], tiles[6], overflow]
        assert overflow.title_label.text() == overflow_title(6)
        grid.arrange(welcome.TILE_WIDTH)
        assert grid.placed == [overflow]
        assert overflow.title_label.text() == overflow_title(8)

        grid.set_tiles([])
        assert grid.placed == [] and overflow.isHidden()
        assert qt_app is not None
    finally:
        grid.deleteLater()


def test_tile_grid_unfolds_secondary_tiles() -> None:
    """Free places show whole groups of secondary tiles, in display order"""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    overflow = welcome.WelcomeEntry("libre-gui-plugin.svg", "", "", tile=True)
    grid = welcome.TileGrid(overflow)
    layout = grid.layout()
    tiles = [QW.QWidget() for _index in range(11)]
    a, a1, a2, a3, a4, b, b1, c, c1, c2, d = tiles
    try:
        grid.set_tiles(tiles)
        grid.resize(3 * welcome.TILE_WIDTH + 2 * welcome.TILE_SPACING, 100)

        # Three columns and two rows leave three free places for three groups
        grid.set_order([[a, a1, a2], [b], [c, c1]], max_rows=2)
        assert grid.placed == [a, a1, a2, b, c, c1]
        assert layout.getItemPosition(layout.indexOf(c1))[:2] == (1, 2)
        assert grid.is_expanded(a) and grid.is_expanded(c)
        assert not grid.is_expanded(b)

        # A group that does not fit is folded, the next ones may still unfold
        grid.set_order([[a, a1, a2, a3, a4], [b, b1], [c, c1]], max_rows=2)
        assert grid.placed == [a, b, b1, c, c1]
        assert not grid.is_expanded(a) and grid.is_expanded(b)
        assert all(tile.isHidden() for tile in (a1, a2, a3, a4))
        grid.arrange(4 * welcome.TILE_WIDTH + 3 * welcome.TILE_SPACING)
        assert grid.placed == [a, a1, a2, a3, a4, b, b1, c]
        assert not grid.is_expanded(c)

        # Without free places, or with too many groups, all groups are folded
        grid.set_order([[a, a1], [b, b1], [c, c1]], max_rows=1)
        assert grid.placed == [a, b, c] and not grid.expanded
        grid.set_order([[a, a1], [b, b1], [c, c1, c2], [d]], max_rows=1)
        assert grid.placed == [a, b, overflow] and not grid.expanded
        assert qt_app is not None
    finally:
        grid.deleteLater()


def test_welcome_page_orders_and_limits_application_tiles(monkeypatch) -> None:
    """Application tiles follow user preferences within a limited number of rows"""
    launched: list[tuple[str, str]] = []
    shown: list[str | None] = []
    plugins = [
        _application_plugin(
            f"org.example.app{index:02d}",
            (WelcomeTile(id="application", title=f"App {index:02d}"),),
            launched,
        )
        for index in range(1, 13)
    ]
    ids = [plugin.plugin_id for plugin in plugins]
    registry = PluginRegistry.get_plugins()
    with datalab_test_app_context(console=False) as win:
        monkeypatch.setattr(
            win, "show_applications", lambda plugin_id=None: shown.append(plugin_id)
        )
        page = win.welcomepanel
        grid = page.tile_grid

        def ordered() -> list[str]:
            """Return the IDs of the applications shown, in display order"""
            tile_ids = {tile: key for key, tile in page.application_tiles.items()}
            return [tile_ids[group[0]] for group in grid.groups]

        previous_plugins = list(registry)
        try:
            registry[:] = plugins
            page.refresh_application_tiles()
            assert ordered() == ids
            assert grid.max_rows == 2
            grid.overflow_tile.SIG_CLICKED.emit()
            assert shown == [None]

            # The number of rows is a user preference
            Conf.welcome_application_rows.set(1)
            page.update_application_tiles()
            assert grid.max_rows == 1

            # Pinned applications come first, hidden ones are left out
            monkeypatch.setattr(
                QW.QMenu, "exec", _choose_menu_action(_("Pin to the top"))
            )
            page.application_tiles[ids[11]].menu_button.click()
            monkeypatch.setattr(
                QW.QMenu, "exec", _choose_menu_action(_("Hide from welcome page"))
            )
            event = QG.QContextMenuEvent(
                QG.QContextMenuEvent.Keyboard, QC.QPoint(), QC.QPoint()
            )
            QW.QApplication.sendEvent(page.application_tiles[ids[0]], event)
            assert ordered() == [ids[11], *ids[1:11]]
            assert Conf.welcome_pinned_applications.get() == [ids[11]]
            assert Conf.welcome_hidden_applications.get() == [ids[0]]

            # Recently used applications come next, once the page is shown again
            page.application_tiles[ids[8]].SIG_CLICKED.emit()
            assert launched == [(ids[8], "application")]
            assert ordered() == [ids[11], *ids[1:11]]
            page.visibility_changed(True)
            assert ordered() == [ids[11], ids[8], *ids[1:8], *ids[9:11]]

            monkeypatch.setattr(QW.QMenu, "exec", _choose_menu_action(_("Unpin")))
            page.application_tiles[ids[11]].menu_button.click()
            assert ordered() == [ids[8], *ids[1:8], *ids[9:12]]
        finally:
            registry[:] = previous_plugins


def test_welcome_page_isolates_invalid_application_tiles(monkeypatch) -> None:
    """A failing plugin or a missing icon does not hide the other tiles"""
    broken = _application_plugin("org.example.broken", RuntimeError("Invalid"), [])
    missing_icon = _application_plugin(
        "org.example.icon",
        (
            WelcomeTile(
                id="application", title="Icon", icon="datalab:data/icons/missing.svg"
            ),
        ),
        [],
    )
    registry = PluginRegistry.get_plugins()
    with datalab_test_app_context(console=False) as win:
        page = win.welcomepanel
        previous_plugins = list(registry)
        try:
            registry[:] = [broken, missing_icon]
            # Outside tests, plugin errors are logged instead of raised
            monkeypatch.setattr(sgmx_qth, "is_running_tests", lambda: False)
            page.refresh_application_tiles()
            assert list(page.application_tiles) == ["org.example.icon"]
            assert not page.applications_section.isHidden()
        finally:
            monkeypatch.undo()
            registry[:] = previous_plugins


def test_welcome_page_application_plugin_lifecycle() -> None:
    """A registered plugin tile opens its catalog page until plugins are disabled"""
    plugin_id = "org.example.welcome-tiles"
    with datalab_test_app_context(console=False) as win:
        # Defined after startup, which clears the registered plugin classes
        class TileApplicationPlugin(PluginBase):
            """Application plugin relying on the default welcome page tile"""

            PLUGIN_INFO = PluginInfo(
                id=plugin_id,
                name="Welcome tiles application",
                description="Application exposed on the welcome page",
                icon=PLUGIN_ICON,
                capabilities=(PluginCapability.APPLICATION,),
            )

            def create_actions(self) -> None:
                """Create no actions for this welcome page test"""

        try:
            TileApplicationPlugin().register(win)
            page = win.welcomepanel
            page.refresh_application_tiles()
            entry = page.application_tiles[plugin_id]
            assert entry.title_label.text() == "Welcome tiles application"

            entry.SIG_CLICKED.emit()
            QW.QApplication.processEvents()
            (dialog,) = win.findChildren(ApplicationsDialog)
            assert dialog.isVisible()
            assert dialog.application_list.currentItem().data(QC.Qt.UserRole) == (
                plugin_id
            )
            dialog.close()

            win.set_plugins_enabled(False)
            assert page.application_tiles == {}
            assert page.applications_section.isHidden()
        finally:
            PluginRegistry.get_plugin_classes().remove(TileApplicationPlugin)


def test_welcome_page_follows_empty_current_panel() -> None:
    """The welcome page replaces the view of an empty current panel"""
    x = np.linspace(0.0, 1.0, 10)
    with datalab_test_app_context(console=False) as win:
        docks = [win.docks[win.welcomepanel]]
        docks += [win.docks[panel] for panel in (win.signalpanel, win.imagepanel)]
        welcome_dock, signal_dock, image_dock = docks

        def front_dock() -> QW.QDockWidget:
            QW.QApplication.processEvents()
            (dock,) = [dock for dock in docks if is_in_front(dock)]
            return dock

        assert front_dock() is welcome_dock
        win.signalpanel.add_object(create_signal("Signal 1", x, x))
        assert front_dock() is signal_dock
        win.set_current_panel("image")
        assert front_dock() is welcome_dock
        win.set_current_panel("signal")
        assert front_dock() is signal_dock
        win.signalpanel.remove_all_objects()
        assert front_dock() is welcome_dock

        # A closed welcome page is opened again when the current panel is empty
        win.signalpanel.add_object(create_signal("Signal 2", x, x))
        welcome_dock.close()
        win.set_current_panel("image")
        assert welcome_dock.isVisible() and front_dock() is welcome_dock

        Conf.welcome_on_startup.set(False)
        win.set_current_panel("signal")
        win.set_current_panel("image")
        assert front_dock() is image_dock
