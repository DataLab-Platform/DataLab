# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for the Applications panel."""

from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace

import numpy as np
import pytest
import sigima.objects
from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtTest as QT
from qtpy import QtWidgets as QW
from sigimax.utils import qthelpers as sgmx_qth

from datalab.config import Conf
from datalab.gui import main
from datalab.gui.plugins import applications as applications_module
from datalab.gui.plugins.applications import (
    READINESS_COLORS,
    ApplicationPage,
    ApplicationsPanel,
    RecipeDetails,
    get_application_plugins,
    get_declared_metadata_keys,
    record_application_use,
    set_application_hidden,
    set_application_pinned,
    sort_welcome_applications,
)
from datalab.plugins import PluginCapability, PluginInfo, PluginRegistry
from datalab.plugins.examples import PluginExample
from datalab.plugins.recipe_binding import (
    RecipeInputIssue,
    RecipeReadiness,
    RecipeReadinessStatus,
)
from datalab.plugins.recipes import (
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeInputSlot,
    RecipeMetadataRequirement,
    RecipeOutcome,
)
from datalab.plugins.tools import PluginTool
from datalab.tests import datalab_test_app_context

RECIPE = RecipeDescriptor(
    recipe_id="org.example.camera:quick-check",
    plugin_version="1.2.3",
    title="Quick camera check",
    version="2.0.0",
    description="Run the quick workflow",
    run=lambda *_args: RecipeOutcome(),
    inputs=(
        RecipeInputSlot(
            "flat_frames",
            "image",
            "many",
            description="Uniformly illuminated frames",
            min_count=2,
            metadata=(RecipeMetadataRequirement("exposure_s", "Exposure time"),),
        ),
    ),
)
OTHER_RECIPE = RecipeDescriptor(
    recipe_id="org.example.camera:dark-current",
    plugin_version="1.2.3",
    title="Dark current",
    version="1.0.0",
    run=lambda *_args: RecipeOutcome(),
    inputs=(RecipeInputSlot("dark_frames", "image", "many"),),
)
EXAMPLE = PluginExample(
    id="quickstart",
    title="Scientific camera quickstart",
    description="Open the packaged workspace",
    resource="datalab:data/tests/reordering_test.h5",
    recipe_ids=(RECIPE.recipe_id,),
)
DATASET = PluginExample(
    id="raw-frames",
    title="Raw frames",
    description="Frames without a dedicated method",
    resource="datalab:data/tests/reordering_test.h5",
)
TOOL = PluginTool(
    id="annotate",
    title="Annotate frames",
    launcher="annotate_frames",
    description="Set frame roles",
)
APPLICATION = frozenset({PluginCapability.APPLICATION})


def _plugin(
    plugin_id: str,
    name: str,
    capabilities: frozenset[PluginCapability] = APPLICATION,
    *,
    recipes: tuple[RecipeDescriptor, ...] = (),
    examples: tuple[PluginExample, ...] = (),
    tools: tuple[PluginTool, ...] = (),
    documentation_url: str | None = None,
    icon: str | None = None,
    calls: list[tuple] | None = None,
    readiness: RecipeReadiness | None = None,
) -> object:
    """Build the minimal active-plugin surface consumed by the panel."""
    calls = [] if calls is None else calls
    if readiness is None:
        readiness = RecipeReadiness(RecipeReadinessStatus.NO_INPUT, {})
    plugin = SimpleNamespace(
        plugin_id=plugin_id,
        info=PluginInfo(
            id=plugin_id,
            name=name,
            version="1.2.3",
            description=f"{name} description",
            icon=icon,
            capabilities=capabilities,
            documentation_url=documentation_url,
        ),
        get_recipes=lambda: recipes,
        get_examples=lambda: examples,
        get_tools=lambda: tools,
        assess_recipe=lambda recipe_id: readiness,
        assess_tool=lambda tool_id: None,
        launch_recipe=lambda recipe_id: calls.append(("recipe", recipe_id)),
        try_example=lambda example_id, recipe_id: calls.append(
            ("try", example_id, recipe_id)
        ),
        launch_example=lambda example_id: calls.append(("example", example_id)),
        launch_tool=lambda tool_id: calls.append(("tool", tool_id)),
    )
    # Registered plugins are unregistered after each test (see conftest.py)
    plugin.unregister = lambda: PluginRegistry.unregister_plugin(plugin)
    return plugin


@pytest.fixture(name="registry")
def fixture_registry() -> Generator[list, None, None]:
    """Give the list of registered plugins, restored after the test."""
    plugins = PluginRegistry.get_plugins()
    previous_plugins = list(plugins)
    try:
        yield plugins
    finally:
        plugins[:] = previous_plugins


@pytest.fixture(name="make_panel")
def fixture_make_panel(
    registry: list,
) -> Generator[Callable[..., ApplicationsPanel], None, None]:
    """Give a factory of panels presenting plugins, closed after the test."""
    # Keep a reference: a garbage-collected application makes Qt crash
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    panels: list[ApplicationsPanel] = []

    def make_panel(*plugins: object, show: bool = True) -> ApplicationsPanel:
        registry[:] = list(plugins)
        panel = ApplicationsPanel()
        panels.append(panel)
        if show:
            panel.resize(ApplicationsPanel.DEFAULT_WIDTH, 700)
            panel.show()
            QW.QApplication.processEvents()
        return panel

    try:
        yield make_panel
    finally:
        for panel in panels:
            panel.close()
            panel.deleteLater()
        qt_app.processEvents()


def _center_color(pixmap: QG.QPixmap) -> str:
    """Return the color at the center of a pixmap."""
    image = pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2).name()


def _labels_text(widget: QW.QWidget) -> str:
    """Return the texts of the labels of a widget."""
    return " ".join(label.text() for label in widget.findChildren(QW.QLabel))


def test_applications_panel_renders_declared_contracts(make_panel) -> None:
    """Methods show their inputs and examples; tools and datasets are listed."""
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION, PluginCapability.PROCESSING}),
        recipes=(RECIPE,),
        examples=(EXAMPLE, DATASET),
        tools=(TOOL,),
        documentation_url="https://example.org/camera/docs",
    )
    processing = _plugin(
        "org.example.processing",
        "Processing Only",
        frozenset({PluginCapability.PROCESSING}),
    )
    panel = make_panel(processing, application)
    assert get_application_plugins() == (application,)

    # With only one application, its name replaces the selector
    combo = panel.application_combo
    assert combo.isHidden() and panel.count_label.isHidden()
    assert not panel.title_widget.isHidden()
    assert panel.title_label.text() == "Camera Characterization"
    assert "org.example.camera" in panel.title_widget.toolTip()
    assert combo.count() == 1
    assert combo.itemData(0) == "org.example.camera"
    assert combo.itemText(0) == "Camera Characterization"
    assert combo.itemData(0, applications_module.CATALOG_DESCRIPTION_ROLE) == (
        "Camera Characterization description"
    )
    assert combo.itemData(0, applications_module.CATALOG_VERSION_ROLE) == "1.2.3"

    page = panel.current_page()
    assert page is panel.application_pages[0]
    # Methods, tools and datasets are sections, all expanded by default
    assert list(page.sections) == ["methods", "tools", "datasets"]
    assert all(section.is_expanded() for section in page.sections.values())
    assert page.sections["methods"].title_label.text() == "Methods (1)"
    assert page.sections["tools"].title_label.text() == "Tools (1)"
    assert page.sections["datasets"].title_label.text() == "Datasets (1)"
    assert list(page.recipe_rows) == [RECIPE.recipe_id]
    row = page.recipe_rows[RECIPE.recipe_id]
    details = page.recipe_details[RECIPE.recipe_id]
    # The method is selected: the page shows its details below the list
    assert page.current_recipe_id() == RECIPE.recipe_id and row.is_selected()
    assert page.splitter.count() == 2
    assert page.details_stack.currentWidget().widget() is details
    assert row.title_label.text() == details.title_label.text() == RECIPE.title
    # Nothing is left out: input descriptions, metadata descriptions and keys
    inputs_text = _labels_text(details.inputs_widget)
    for text in (
        "Flat frames",
        "At least 2 images",
        "Uniformly illuminated frames",
        "exposure_s",
        "Exposure time",
    ):
        assert text in inputs_text
    assert RECIPE.description in _labels_text(details)
    # Examples are listed in the details of the methods they were designed for
    assert list(details.example_buttons) == [EXAMPLE.id]
    assert list(page.dataset_buttons) == [DATASET.id]
    assert list(page.tool_buttons) == [TOOL.id]
    assert "Select the input data" in row.summary_label.text()
    assert "Select the input data" in details.readiness_label.text()
    assert row.run_button.isEnabled()
    assert panel.documentation_action.isEnabled()
    assert panel.documentation_action.toolTip() == "https://example.org/camera/docs"


def test_methods_are_selected_to_show_their_details(make_panel) -> None:
    """A click, the arrow keys or Run select a method, which is remembered."""
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        recipes=(RECIPE, OTHER_RECIPE),
    )
    page = make_panel(application).current_page()
    first, second = page.recipe_rows.values()

    def shown_details() -> RecipeDetails:
        return page.details_stack.currentWidget().widget()

    assert page.current_recipe_id() == RECIPE.recipe_id
    assert first.is_selected() and not second.is_selected()
    QT.QTest.mouseClick(second, QC.Qt.LeftButton)
    assert page.current_recipe_id() == OTHER_RECIPE.recipe_id
    assert second.is_selected() and not first.is_selected()
    assert shown_details() is page.recipe_details[OTHER_RECIPE.recipe_id]
    assert Conf.applications_selected_methods.get() == {
        "org.example.camera": OTHER_RECIPE.recipe_id
    }
    QT.QTest.keyClick(second, QC.Qt.Key_Up)
    assert page.current_recipe_id() == RECIPE.recipe_id
    QT.QTest.keyClick(first, QC.Qt.Key_Up)
    assert page.current_recipe_id() == RECIPE.recipe_id
    QT.QTest.keyClick(first, QC.Qt.Key_Down)
    assert shown_details() is page.recipe_details[OTHER_RECIPE.recipe_id]
    first.run_button.click()
    assert page.current_recipe_id() == RECIPE.recipe_id
    with pytest.raises(KeyError):
        page.select_recipe("org.example.camera:unknown")

    # The selected method of each application is remembered
    second.run_button.click()
    other_page = make_panel(application).current_page()
    assert other_page.current_recipe_id() == OTHER_RECIPE.recipe_id


def test_sections_and_splitter_are_remembered(make_panel) -> None:
    """Collapsed sections and the details height apply to every application."""
    camera = _plugin("org.example.camera", "Camera Characterization", recipes=(RECIPE,))
    pulse = _plugin(
        "org.example.pulse",
        "Pulse Characterization",
        recipes=(OTHER_RECIPE,),
        tools=(TOOL,),
    )
    panel = make_panel(camera, pulse)
    camera_page, pulse_page = panel.application_pages
    # Several applications: a selector headed by their number, which invites
    # to show them all
    assert not panel.count_label.isHidden()
    assert panel.count_label.text() == "Applications (2)"
    assert not panel.application_combo.isHidden()
    assert "available application plugins" in panel.application_combo.toolTip()
    assert panel.title_widget.isHidden()

    methods = camera_page.sections["methods"]
    methods.title_label.clicked.emit()
    assert not methods.is_expanded() and methods.content.isHidden()
    assert Conf.applications_collapsed_sections.get() == ["methods"]
    panel.select_plugin("org.example.pulse")
    assert not pulse_page.sections["methods"].is_expanded()
    assert pulse_page.sections["tools"].is_expanded()
    pulse_page.sections["methods"].toggle_button.click()
    assert Conf.applications_collapsed_sections.get() == []

    pulse_page.splitter.setSizes([100, 300])
    pulse_page.splitter.splitterMoved.emit(100, 1)
    sizes = Conf.applications_details_sizes.get()
    assert sizes is not None and sizes[0] < sizes[1]
    panel.select_plugin("org.example.camera")
    assert camera_page.sections["methods"].is_expanded()
    list_height, details_height = camera_page.splitter.sizes()
    assert list_height < details_height


def test_applications_panel_is_compact(make_panel) -> None:
    """Long texts wrap instead of widening the panel, which fits a narrow dock."""
    long_recipe = RecipeDescriptor(
        recipe_id="org.example.wordy:quick-check",
        plugin_version="1.2.3",
        title=" ".join([RECIPE.title] * 8),
        version="2.0.0",
        description=" ".join([RECIPE.description] * 8),
        run=lambda *_args: RecipeOutcome(),
        inputs=RECIPE.inputs,
    )
    concise = _plugin("org.example.camera", "Camera", recipes=(RECIPE,))
    wordy = _plugin(
        "org.example.wordy", " ".join(["Wordy"] * 20), recipes=(long_recipe,)
    )
    panel = make_panel(concise, wordy)
    assert panel.minimumWidth() == ApplicationsPanel.MINIMUM_WIDTH
    assert panel.minimumSizeHint().width() <= ApplicationsPanel.DEFAULT_WIDTH
    assert panel.sizeHint().width() == ApplicationsPanel.DEFAULT_WIDTH
    concise_page, wordy_page = panel.application_pages

    def minimum_widths(page: ApplicationPage) -> tuple[int, int]:
        # Measured once shown, with the fonts of the current style
        panel.select_plugin(page.plugin.plugin_id)
        page.refresh_readiness()
        QW.QApplication.processEvents()
        details_area = page.details_stack.widget(0)
        return (
            page.list_area.widget().minimumSizeHint().width(),
            details_area.widget().minimumSizeHint().width(),
        )

    assert minimum_widths(wordy_page) == minimum_widths(concise_page)


def test_tool_buttons_follow_the_tool_selection(make_panel) -> None:
    """A tool that cannot be opened on the selection has a disabled button."""
    issues = {"annotate": "Select one image"}
    application = _plugin(
        "org.example.camera", "Camera Characterization", tools=(TOOL,)
    )
    application.assess_tool = issues.get
    panel = make_panel(application)
    button = panel.current_page().tool_buttons[TOOL.id]
    assert not button.isEnabled()
    assert button.toolTip() == "Select one image"
    issues.clear()
    panel.refresh_readiness()
    assert button.isEnabled()
    assert button.toolTip() == ""


def test_applications_panel_has_an_explicit_empty_state(make_panel) -> None:
    """Without application plugins, the panel says so instead of a selector."""
    panel = make_panel()
    assert panel.application_pages == []
    assert panel.current_page() is None
    assert panel.current_plugin_id() is None
    assert not panel.empty_label.isHidden()
    for widget in (
        panel.count_label,
        panel.application_combo,
        panel.title_widget,
        panel.menu_button,
        panel.application_stack,
    ):
        assert widget.isHidden()
    panel.refresh_readiness()
    panel.sync_menu()
    assert not panel.documentation_action.isEnabled()


def _icon_image(icon: QG.QIcon) -> QG.QImage:
    """Render an icon to compare it with another one."""
    return icon.pixmap(32, 32).toImage()


def test_applications_panel_shows_plugin_icons(make_panel, monkeypatch) -> None:
    """The selector shows the declared plugin icon, or the generic one."""
    declared = _plugin(
        "org.example.camera",
        "Camera Characterization",
        icon="datalab:data/icons/analysis.svg",
    )
    missing = _plugin(
        "org.example.pulse",
        "Pulse Characterization",
        icon="datalab:data/icons/missing.svg",
    )
    # Outside tests, icon loading errors are logged instead of raised
    monkeypatch.setattr(sgmx_qth, "is_running_tests", lambda: False)
    generic_image = _icon_image(get_icon("libre-gui-plugin.svg"))
    combo = make_panel(declared, missing).application_combo
    assert _icon_image(combo.itemIcon(0)) != generic_image
    assert _icon_image(combo.itemIcon(1)) == generic_image
    # The list of the selector shows the icon, the name and the description
    delegate = combo.itemDelegate()
    assert isinstance(delegate, applications_module.CatalogItemDelegate)
    assert combo.view().sizeHintForRow(0) >= (
        delegate.ICON_SIZE + 2 * delegate.VERTICAL_MARGIN
    )


def test_plugin_icon_registered_image(monkeypatch) -> None:
    """Plugin icons also accept images registered with guidata ("module:filename"),
    consistently with guidata's image names."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    # Outside tests, icon loading errors are logged instead of raised
    monkeypatch.setattr(sgmx_qth, "is_running_tests", lambda: False)
    generic_image = _icon_image(get_icon("libre-gui-plugin.svg"))
    delete_image = _icon_image(get_icon("datalab:delete.svg"))
    assert delete_image != generic_image
    icon = applications_module.get_plugin_icon("datalab:delete.svg")
    assert _icon_image(icon) == delete_image
    icon = applications_module.get_plugin_icon("datalab:edit/delete.svg")
    assert _icon_image(icon) == delete_image
    for missing in ("datalab:missing.svg", "unknown_package:icons/missing.svg"):
        icon = applications_module.get_plugin_icon(missing)
        assert _icon_image(icon) == generic_image
    assert qt_app is not None


def test_application_page_has_explicit_empty_states(make_panel) -> None:
    """A page without methods says so; commands need a declared target."""
    panel = make_panel(_plugin("org.example.catalog-only", "Catalog only"))
    page = panel.current_page()
    assert page.recipe_rows == {}
    assert page.current_recipe_id() is None
    # Without methods, there are no details to show
    assert page.splitter.count() == 1
    assert page.tool_buttons == {}
    assert page.dataset_buttons == {}
    assert page.sections == {}
    assert not panel.documentation_action.isEnabled()


def test_application_page_shows_readiness_of_each_method(make_panel) -> None:
    """Method rows summarize whether the current selection can be analyzed."""
    readiness = RecipeReadiness(
        RecipeReadinessStatus.NOT_READY,
        {},
        (
            RecipeInputIssue(
                "missing_metadata",
                "flat_frames",
                {"key": "exposure_s", "count": 2, "titles": ("A", "B")},
            ),
        ),
    )
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        recipes=(RECIPE,),
        readiness=readiness,
    )
    panel = make_panel(application)
    page = panel.current_page()
    row = page.recipe_rows[RECIPE.recipe_id]
    details = page.recipe_details[RECIPE.recipe_id]
    # The row summarizes the readiness, its tooltip and the details give why
    assert row.summary_label.text() == "The current selection cannot be analyzed"
    for widget in (row.status_icon_label, row.summary_label):
        assert "Add metadata" in widget.toolTip()
    text = details.readiness_label.text()
    assert "cannot be analyzed" in text
    assert "Flat frames: metadata &#x27;exposure_s&#x27; missing on 2" in text
    assert "Edit &gt; Metadata &gt; Add metadata..." in text
    assert (
        _center_color(row.status_icon_label.pixmap())
        == (READINESS_COLORS[RecipeReadinessStatus.NOT_READY])
    )
    assert row.status_color == READINESS_COLORS[RecipeReadinessStatus.NOT_READY]

    # Without input data, the reasons only repeat the inputs: they are in the
    # tooltip only
    application.assess_recipe = lambda _recipe_id: RecipeReadiness(
        RecipeReadinessStatus.NO_INPUT,
        {},
        (RecipeInputIssue("missing", "flat_frames", {}),),
    )
    panel.refresh_readiness()
    assert row.summary_label.text() == "Select the input data in the workspace"
    assert "Flat frames: no object assigned" in row.summary_label.toolTip()
    assert "no object assigned" not in details.readiness_label.text()
    assert (
        _center_color(row.status_icon_label.pixmap())
        == (READINESS_COLORS[RecipeReadinessStatus.NO_INPUT])
    )

    def failing_assessment(_recipe_id: str) -> RecipeReadiness:
        raise RuntimeError("suggestion exploded")

    application.assess_recipe = failing_assessment
    panel.refresh_readiness()
    assert "could not be assessed" in row.summary_label.text()
    assert "suggestion exploded" in details.readiness_label.text()

    application.assess_recipe = lambda _recipe_id: RecipeReadiness(
        RecipeReadinessStatus.READY, {}
    )
    panel.schedule_readiness_update()
    qt_app = QW.QApplication.instance()
    QC.QTimer.singleShot(300, qt_app.quit)
    qt_app.exec()
    assert row.summary_label.text() == "Ready to run on the current selection"
    assert row.status_summary == "Ready to run on the current selection"


def test_welcome_application_preferences() -> None:
    """Welcome page order: pinned, then recently used, then by name"""
    plugins = [
        SimpleNamespace(
            plugin_id=f"org.example.{name}", info=SimpleNamespace(name=name)
        )
        for name in ("delta", "Alpha", "charlie", "Bravo", "echo")
    ]
    ids = {plugin.info.name: plugin.plugin_id for plugin in plugins}

    def order() -> list[str]:
        return [plugin.info.name for plugin in sort_welcome_applications(plugins)]

    assert order() == ["Alpha", "Bravo", "charlie", "delta", "echo"]
    record_application_use(ids["delta"])
    record_application_use(ids["echo"])
    record_application_use(ids["delta"])
    assert Conf.welcome_recent_applications.get() == [ids["delta"], ids["echo"]]
    set_application_pinned(ids["charlie"], True)
    set_application_pinned(ids["Bravo"], True)
    assert order() == ["charlie", "Bravo", "delta", "echo", "Alpha"]

    # Hiding an application unpins it, pinning it shows it again
    set_application_hidden(ids["charlie"], True)
    assert order() == ["Bravo", "delta", "echo", "Alpha"]
    assert Conf.welcome_pinned_applications.get() == [ids["Bravo"]]
    set_application_pinned(ids["charlie"], True)
    assert order() == ["Bravo", "charlie", "delta", "echo", "Alpha"]
    assert Conf.welcome_hidden_applications.get() == []
    set_application_pinned(ids["Bravo"], False)
    assert order() == ["charlie", "delta", "echo", "Alpha", "Bravo"]

    for index in range(applications_module.MAX_RECENT_APPLICATIONS + 5):
        record_application_use(f"org.example.other{index}")
    recent = Conf.welcome_recent_applications.get()
    assert len(recent) == applications_module.MAX_RECENT_APPLICATIONS
    assert (
        recent[0]
        == f"org.example.other{applications_module.MAX_RECENT_APPLICATIONS + 4}"
    )


def test_applications_panel_selects_and_keeps_the_application(
    make_panel, registry
) -> None:
    """The most recently used application comes first, and stays shown."""
    camera = _plugin("org.example.camera", "Camera Characterization")
    pulse = _plugin("org.example.pulse", "Pulse Characterization")
    # Without recent use, the first application (by name) is shown
    panel = make_panel(camera, pulse)
    assert panel.current_plugin_id() == "org.example.camera"

    record_application_use("org.example.pulse")
    panel = make_panel(camera, pulse)
    assert panel.current_plugin_id() == "org.example.pulse"
    assert panel.current_page().plugin is pulse

    panel.select_plugin("org.example.camera")
    assert panel.current_page().plugin is camera
    assert panel.application_combo.currentText() == "Camera Characterization"
    with pytest.raises(KeyError):
        panel.select_plugin("org.example.unknown")

    # Refreshing with the same plugins keeps the pages (and their state)
    pages = list(panel.application_pages)
    panel.refresh()
    assert panel.application_pages == pages

    # New plugin instances (e.g. reloaded plugins) rebuild the pages, and the
    # shown application stays shown
    new_camera = _plugin("org.example.camera", "Camera Characterization")
    other = _plugin("org.example.other", "Another application")
    registry[:] = [other, pulse, new_camera]
    panel.refresh()
    assert panel.application_combo.count() == 3
    assert panel.current_page().plugin is new_camera
    assert all(page not in pages for page in panel.application_pages[::2])


def test_applications_panel_menu_and_welcome_preferences(
    make_panel, monkeypatch
) -> None:
    """The menu opens the documentation and edits the welcome page options."""
    opened_urls: list[str] = []
    monkeypatch.setattr(
        applications_module.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    camera = _plugin(
        "org.example.camera",
        "Camera Characterization",
        documentation_url="https://example.org/camera/docs",
    )
    pulse = _plugin("org.example.pulse", "Pulse Characterization")
    panel = make_panel(camera, pulse)
    changes: list[None] = []
    panel.SIG_WELCOME_PREFERENCES_CHANGED.connect(lambda: changes.append(None))
    show_action = panel.show_on_welcome_action
    pin_action = panel.pin_on_welcome_action
    assert panel.menu_button.menu() is panel.menu
    assert panel.menu_button.toolTip() == "More actions"

    panel.documentation_action.trigger()
    assert opened_urls == ["https://example.org/camera/docs"]
    assert show_action.isChecked() and not pin_action.isChecked()
    pin_action.trigger()
    assert Conf.welcome_pinned_applications.get() == ["org.example.camera"]
    show_action.trigger()
    assert Conf.welcome_hidden_applications.get() == ["org.example.camera"]
    assert Conf.welcome_pinned_applications.get() == []
    assert not pin_action.isChecked()
    pin_action.trigger()
    assert show_action.isChecked()
    assert Conf.welcome_hidden_applications.get() == []
    assert len(changes) == 3

    # The menu follows the shown application
    panel.select_plugin("org.example.pulse")
    assert not panel.documentation_action.isEnabled()
    assert not pin_action.isChecked()

    # Preferences changed elsewhere (e.g. the welcome page) are shown by the menu
    set_application_hidden("org.example.pulse", True)
    panel.menu.aboutToShow.emit()
    assert not show_action.isChecked()


def test_main_window_shows_applications_panel(monkeypatch) -> None:
    """The Applications action shows one panel, floating beside the main window
    the first time only, without blocking DataLab."""
    updates: list[tuple] = []
    monkeypatch.setattr(
        main.ApplicationsPanel,
        "schedule_readiness_update",
        lambda _panel, *args: updates.append(args),
    )

    with datalab_test_app_context(console=False, exec_loop=False) as window:
        menu_actions = window.menuBar().actions()
        assert window.applications_action in menu_actions
        assert window.applications_action.text() == ""
        panel = window.applicationspanel
        dock = window.docks[panel]
        # The panel exists from startup, hidden, for its layout to be restored
        assert dock.objectName() == "applications_panel"
        assert window.dockWidgetArea(dock) == QC.Qt.LeftDockWidgetArea
        assert not dock.isVisible() and dock.isFloating()
        assert dock.toggleViewAction() in window.createPopupMenu().actions()
        assert not Conf.applications_panel_shown.get()

        # The panel is placed beside the window as it is when first shown
        window.resize(window.width(), window.height() - 100)
        QW.QApplication.processEvents()
        window.applications_action.trigger()
        QW.QApplication.processEvents()
        assert window.findChildren(ApplicationsPanel) == [panel]
        assert dock.isVisible() and dock.isFloating()
        assert Conf.applications_panel_shown.get()
        available = window.screen().availableGeometry()
        assert available.contains(dock.frameGeometry())
        assert dock.width() == ApplicationsPanel.DEFAULT_WIDTH
        assert dock.height() == window.height()
        assert window.isEnabled()

        window.set_current_panel("image")
        assert window.get_current_panel() == "image"
        assert dock.isVisible()

        # Afterwards, the panel is shown where the user left it
        position = available.topLeft() + QC.QPoint(10, 10)
        dock.move(position)
        dock.close()
        window.applications_action.trigger()
        QW.QApplication.processEvents()
        assert dock.isVisible() and dock.isFloating()
        assert dock.pos() == position
        assert window.findChildren(ApplicationsPanel) == [panel]

        # ...unless it is out of all screens (e.g. on a disconnected screen)
        dock.move(available.right() + 5000, available.bottom() + 5000)
        window.applications_action.trigger()
        QW.QApplication.processEvents()
        assert available.contains(dock.frameGeometry())

        # Selection changes and object edits re-assess the methods
        updates.clear()
        window.imagepanel.SIG_OBJECT_MODIFIED.emit()
        window.signalpanel.objview.SIG_SELECTION_CHANGED.emit()
        assert len(updates) == 2


def test_applications_panel_floats_when_shown_from_the_view_menu() -> None:
    """Shown for the first time from the View menu, the panel floats too; once
    docked and closed, it is shown docked again."""
    with datalab_test_app_context(console=False, exec_loop=False) as window:
        dock = window.docks[window.applicationspanel]
        dock.toggleViewAction().trigger()
        QW.QApplication.processEvents()
        assert dock.isVisible() and dock.isFloating()
        assert Conf.applications_panel_shown.get()
        available = window.screen().availableGeometry()
        assert available.contains(dock.frameGeometry())

        dock.setFloating(False)
        dock.close()
        window.show_applications()
        QW.QApplication.processEvents()
        assert dock.isVisible() and not dock.isFloating()


def test_docked_applications_panel_keeps_its_size() -> None:
    """Closed while docked, the panel is shown again with the same size, which
    the user may then change again."""
    with datalab_test_app_context(console=False, exec_loop=False) as window:
        qt_app = QW.QApplication.instance()
        dock = window.docks[window.applicationspanel]
        window.show_applications()
        dock.setFloating(False)
        qt_app.processEvents()
        window.resizeDocks([dock], [ApplicationsPanel.DEFAULT_WIDTH], QC.Qt.Horizontal)
        qt_app.processEvents()
        width = dock.width()
        assert width > ApplicationsPanel.MINIMUM_WIDTH

        dock.close()
        qt_app.processEvents()
        assert Conf.applications_panel_docked_size.get() == (width, dock.height())
        window.show_applications()
        for _index in range(3):
            qt_app.processEvents()
        assert not dock.isFloating()
        assert dock.width() == width
        assert dock.minimumWidth() == 0
        assert Conf.applications_panel_docked_size.get() is None


def test_applications_panel_floats_beside_a_window() -> None:
    """With room on the screen, the panel floats to the right of the window,
    as high as the window."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    window = QW.QMainWindow()
    try:
        window.setCentralWidget(QW.QLabel("Central widget"))
        panel = ApplicationsPanel(window)
        dock, area = panel.create_dockwidget("Applications")
        window.addDockWidget(area, dock)
        dock.hide()
        available = window.screen().availableGeometry()
        window.setGeometry(available.left(), available.top(), 200, 300)
        window.show()
        qt_app.processEvents()
        assert available.right() - window.frameGeometry().right() >= (
            ApplicationsPanel.DEFAULT_WIDTH
        )

        panel.place_beside(window)
        dock.show()
        panel.ensure_on_screen(window)
        qt_app.processEvents()
        assert dock.isFloating()
        assert dock.frameGeometry().left() == window.frameGeometry().right() + 1
        assert dock.frameGeometry().top() == window.frameGeometry().top()
        assert dock.width() == ApplicationsPanel.DEFAULT_WIDTH
        assert dock.height() == window.height()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


def test_add_metadata_suggests_keys_declared_by_applications() -> None:
    """Add metadata offers the keys expected by application methods"""
    application = _plugin(
        "org.example.camera", "Camera Characterization", recipes=(RECIPE,)
    )
    with datalab_test_app_context(console=False, exec_loop=False) as window:
        # DataLab unregisters the plugins on close: restore the registry first
        registry = PluginRegistry.get_plugins()
        previous_plugins = list(registry)
        registry[:] = [application]
        try:
            assert get_declared_metadata_keys("image") == [
                ("exposure_s", "Exposure time")
            ]
            assert get_declared_metadata_keys("signal") == []
            image = sigima.objects.create_image("Flat", np.zeros((2, 2)))
            image.metadata["gain"] = 2
            assert window.imagepanel.get_known_metadata_keys([image]) == [
                ("gain", "e.g. 2"),
                ("exposure_s", "Exposure time"),
            ]
            assert window.signalpanel.get_known_metadata_keys([]) == []
        finally:
            registry[:] = previous_plugins


def test_application_commands_delegate_to_plugin_contracts(make_panel) -> None:
    """Each panel command calls its plugin entry point; outcomes are shown."""
    calls: list[tuple] = []
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        recipes=(RECIPE,),
        examples=(EXAMPLE, DATASET),
        tools=(TOOL,),
        calls=calls,
    )
    panel = make_panel(application)
    page = panel.current_page()
    row = page.recipe_rows[RECIPE.recipe_id]
    details = page.recipe_details[RECIPE.recipe_id]
    row.run_button.click()
    details.example_buttons[EXAMPLE.id].click()
    page.dataset_buttons[DATASET.id].click()
    page.tool_buttons[TOOL.id].click()
    assert calls == [
        ("recipe", RECIPE.recipe_id),
        ("try", EXAMPLE.id, RECIPE.recipe_id),
        ("example", DATASET.id),
        ("tool", TOOL.id),
    ]
    assert panel.isVisible()
    assert Conf.welcome_recent_applications.get() == ["org.example.camera"]
    assert row.outcome_label.isHidden()
    assert page.outcome_label.isHidden()

    # The outcome of a method is shown below its row
    application.launch_recipe = lambda _recipe_id: RecipeOutcome(
        diagnostics=(RecipeDiagnostic("warning", "excluded-shots", "2 shots excluded"),)
    )
    row.run_button.click()
    assert not row.outcome_label.isHidden()
    assert "Created 0 objects" in row.outcome_label.text()
    assert "Warning: 2 shots excluded" in row.outcome_label.text()
    assert page.outcome_label.isHidden()

    # Outcomes of other commands are shown at the bottom of the page
    application.launch_tool = lambda _tool_id: RecipeOutcome()
    page.tool_buttons[TOOL.id].click()
    assert not page.outcome_label.isHidden()
    assert "Created 0 objects" in page.outcome_label.text()


def test_application_command_failures_are_reported_not_raised(
    make_panel, monkeypatch
) -> None:
    """A failing plugin entry point shows an error dialog instead of crashing."""
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        recipes=(RECIPE,),
        examples=(EXAMPLE, DATASET),
        tools=(TOOL,),
    )

    def fail(message: str):
        def failing(*_args) -> None:
            raise RuntimeError(message)

        return failing

    application.launch_recipe = fail("launcher exploded")
    application.try_example = fail("try exploded")
    application.launch_example = fail("example exploded")
    application.launch_tool = fail("tool exploded")
    reported: list[tuple[str, str]] = []
    monkeypatch.setattr(
        applications_module,
        "qt_handle_error_message",
        lambda _widget, message, context=None: reported.append((str(message), context)),
    )
    page = make_panel(application, show=False).current_page()
    row = page.recipe_rows[RECIPE.recipe_id]
    details = page.recipe_details[RECIPE.recipe_id]
    row.run_button.click()
    details.example_buttons[EXAMPLE.id].click()
    page.dataset_buttons[DATASET.id].click()
    page.tool_buttons[TOOL.id].click()
    assert [message for message, _context in reported] == [
        "launcher exploded",
        "try exploded",
        "example exploded",
        "tool exploded",
    ]
    assert all(_context for _message, _context in reported)
