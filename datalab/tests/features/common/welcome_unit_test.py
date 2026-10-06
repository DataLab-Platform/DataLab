# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Welcome page unit tests
"""

from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
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


def test_welcome_page_application_tiles(monkeypatch) -> None:
    """Application plugins add tiles launching their declared entry points"""
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
            page.refresh_application_tiles()
            win.show_welcome_page()
            QW.QApplication.processEvents()

            keys = [
                ("org.example.camera", "application"),
                ("org.example.pulse", "application"),
                ("org.example.pulse", "demo"),
                ("org.example.pulse", "broken"),
            ]
            assert not page.applications_section.isHidden()
            assert list(page.application_tiles) == keys
            for entry in page.application_tiles.values():
                assert not entry.icon_label.pixmap().isNull()
                entry.SIG_CLICKED.emit()
            assert launched == keys
            assert errors == [("Broken launcher", _("Launching '%s'") % "Broken")]
            page.browse_applications_button.click()
            assert shown == [None]

            grid = page.tile_grid
            grid.arrange(welcome.TILE_WIDTH)
            assert grid.columns == 1
            assert grid.layout().getItemPosition(3)[:2] == (3, 0)
            grid.arrange(3 * welcome.TILE_WIDTH + 2 * welcome.TILE_SPACING)
            assert grid.columns == 3
            assert grid.layout().getItemPosition(3)[:2] == (1, 0)

            registry[:] = [processing]
            page.refresh_application_tiles()
            assert page.application_tiles == {}
            assert page.applications_section.isHidden()
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
            assert list(page.application_tiles) == [("org.example.icon", "application")]
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
            entry = page.application_tiles[(plugin_id, "application")]
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
