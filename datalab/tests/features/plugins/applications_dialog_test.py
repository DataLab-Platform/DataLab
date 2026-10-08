# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for the application plugin catalog."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import sigima.objects
from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigimax.utils import qthelpers as sgmx_qth

from datalab.config import Conf
from datalab.gui import main
from datalab.gui.plugins import applications as applications_module
from datalab.gui.plugins.applications import (
    ApplicationsDialog,
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


def _plugin(
    plugin_id: str,
    name: str,
    capabilities: frozenset[PluginCapability],
    *,
    recipes: tuple[RecipeDescriptor, ...] = (),
    examples: tuple[PluginExample, ...] = (),
    tools: tuple[PluginTool, ...] = (),
    documentation_url: str | None = None,
    icon: str | None = None,
    calls: list[tuple] | None = None,
    readiness: RecipeReadiness | None = None,
) -> object:
    """Build the minimal active-plugin surface consumed by the catalog."""
    calls = [] if calls is None else calls
    if readiness is None:
        readiness = RecipeReadiness(RecipeReadinessStatus.NO_INPUT, {})
    return SimpleNamespace(
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


def test_applications_dialog_filters_and_renders_declared_contracts() -> None:
    """Methods show their inputs and examples; tools and datasets are listed."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
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
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [processing, application]
    try:
        assert get_application_plugins() == (application,)
        dialog = ApplicationsDialog()
        dialog.show()
        QW.QApplication.processEvents()

        assert dialog.application_list.count() == 1
        application_item = dialog.application_list.item(0)
        assert application_item.data(QC.Qt.UserRole) == "org.example.camera"
        assert application_item.data(applications_module.CATALOG_TITLE_ROLE) == (
            "Camera Characterization"
        )
        assert (
            application_item.data(applications_module.CATALOG_DESCRIPTION_ROLE)
            == "Camera Characterization description"
        )

        page = dialog.application_pages[0]
        assert list(page.recipe_cards) == [RECIPE.recipe_id]
        card = page.recipe_cards[RECIPE.recipe_id]
        # Methods, tools and datasets are the items of a single accordion
        toolbox = page.toolbox
        assert [toolbox.itemText(index) for index in range(toolbox.count())] == [
            RECIPE.title,
            "Tools (1)",
            "Datasets (1)",
        ]
        assert toolbox.currentWidget() is card
        inputs_text = card.inputs_label.text()
        for text in (
            "Flat frames",
            "Uniformly illuminated frames",
            "exposure_s",
            "Exposure time",
        ):
            assert text in inputs_text
        # Examples are listed under the methods they were designed for
        assert list(card.example_buttons) == [EXAMPLE.id]
        assert list(page.dataset_buttons) == [DATASET.id]
        assert list(page.tool_buttons) == [TOOL.id]
        assert "Select the input data" in card.readiness_label.text()
        assert card.run_button.isEnabled()
        assert page.documentation_button.isEnabled()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.close()
            dialog.deleteLater()
            QW.QApplication.processEvents()


def test_tool_buttons_follow_the_tool_selection() -> None:
    """A tool that cannot be opened on the selection has a disabled button."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    issues = {"annotate": "Select one image"}
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
        tools=(TOOL,),
    )
    application.assess_tool = issues.get
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [application]
    try:
        dialog = ApplicationsDialog()
        dialog.show()
        QW.QApplication.processEvents()
        button = dialog.application_pages[0].tool_buttons[TOOL.id]
        assert not button.isEnabled()
        assert button.toolTip() == "Select one image"
        issues.clear()
        dialog.refresh_readiness()
        assert button.isEnabled()
        assert button.toolTip() == ""
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.close()
            dialog.deleteLater()
            QW.QApplication.processEvents()


def test_applications_dialog_has_an_explicit_empty_state() -> None:
    """An empty registry produces a stable catalog placeholder."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry.clear()
    try:
        dialog = ApplicationsDialog()
        assert dialog.application_list.count() == 0
        assert dialog.application_pages == []
        assert dialog.application_stack.count() == 1
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()


def _icon_image(icon: QG.QIcon) -> QG.QImage:
    """Render an icon to compare it with another one."""
    return icon.pixmap(32, 32).toImage()


def test_applications_dialog_shows_plugin_icons(monkeypatch) -> None:
    """Entries and pages show the declared plugin icon, or the generic one."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    capabilities = frozenset({PluginCapability.APPLICATION})
    declared = _plugin(
        "org.example.camera",
        "Camera Characterization",
        capabilities,
        icon="datalab:data/icons/analysis.svg",
    )
    missing = _plugin(
        "org.example.pulse",
        "Pulse Characterization",
        capabilities,
        icon="datalab:data/icons/missing.svg",
    )
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [declared, missing]
    # Outside tests, icon loading errors are logged instead of raised
    monkeypatch.setattr(sgmx_qth, "is_running_tests", lambda: False)
    generic_image = _icon_image(get_icon("libre-gui-plugin.svg"))
    try:
        dialog = ApplicationsDialog()
        declared_item = dialog.application_list.item(0)
        missing_item = dialog.application_list.item(1)
        assert _icon_image(declared_item.icon()) != generic_image
        assert _icon_image(missing_item.icon()) == generic_image
        assert all(
            not page.icon_label.pixmap().isNull() for page in dialog.application_pages
        )
        delegate = dialog.application_list.itemDelegate()
        assert dialog.application_list.sizeHintForRow(0) >= (
            delegate.ICON_SIZE + 2 * delegate.VERTICAL_MARGIN
        )
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()


def test_application_page_has_explicit_empty_states() -> None:
    """A page without methods says so; commands need a declared target."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    application = _plugin(
        "org.example.catalog-only",
        "Catalog only",
        frozenset({PluginCapability.APPLICATION}),
    )
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [application]
    try:
        dialog = ApplicationsDialog()
        page = dialog.application_pages[0]
        assert page.recipe_cards == {}
        assert page.tool_buttons == {}
        assert page.dataset_buttons == {}
        assert not page.documentation_button.isEnabled()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()


def test_application_page_shows_readiness_of_each_method() -> None:
    """Method cards summarize whether the current selection can be analyzed."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
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
        frozenset({PluginCapability.APPLICATION}),
        recipes=(RECIPE,),
        readiness=readiness,
    )
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [application]
    try:
        dialog = ApplicationsDialog()
        dialog.show()
        QW.QApplication.processEvents()
        card = dialog.application_pages[0].recipe_cards[RECIPE.recipe_id]
        text = card.readiness_label.text()
        assert "cannot be analyzed" in text
        assert "Flat frames: metadata &#x27;exposure_s&#x27; missing on 2" in text
        assert "Edit &gt; Metadata &gt; Add metadata..." in text
        # The method item shows the readiness without being opened
        toolbox = dialog.application_pages[0].toolbox
        assert "cannot be analyzed" in toolbox.itemToolTip(0)
        center = toolbox.itemIcon(0).pixmap(32, 32).toImage().pixelColor(16, 16)
        assert (
            center.name()
            == applications_module.RecipeCard.READINESS_COLORS[
                RecipeReadinessStatus.NOT_READY
            ]
        )

        def failing_assessment(_recipe_id: str) -> RecipeReadiness:
            raise RuntimeError("suggestion exploded")

        application.assess_recipe = failing_assessment
        dialog.refresh_readiness()
        text = card.readiness_label.text()
        assert "could not be assessed" in text
        assert "suggestion exploded" in text

        application.assess_recipe = lambda _recipe_id: RecipeReadiness(
            RecipeReadinessStatus.READY, {}
        )
        dialog.schedule_readiness_update()
        QC.QTimer.singleShot(300, qt_app.quit)
        qt_app.exec()
        assert "Ready to run on the current selection" in card.readiness_label.text()
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.close()
            dialog.deleteLater()
            QW.QApplication.processEvents()


def test_applications_dialog_collapses_the_application_list() -> None:
    """Hiding the list shrinks the window, keeps the page width and persists"""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    camera = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
    )
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [camera]
    previous_collapsed = Conf.applications_list_collapsed.get()
    dialogs: list[ApplicationsDialog] = []
    try:
        Conf.applications_list_collapsed.set(False)
        dialog = ApplicationsDialog()
        dialogs.append(dialog)
        dialog.resize(1000, 650)
        dialog.show()
        QW.QApplication.processEvents()
        width = dialog.width()
        list_width = dialog.catalog_widget.width()
        page_width = dialog.application_stack.width()
        toggle = dialog.list_toggle_button
        assert toggle.toolTip() == "Hide the application list"

        toggle.click()
        QW.QApplication.processEvents()
        assert not dialog.is_application_list_visible()
        assert Conf.applications_list_collapsed.get()
        assert toggle.toolTip() == "Show the application list"
        assert dialog.width() < width - list_width
        assert dialog.application_stack.width() == page_width

        toggle.click()
        QW.QApplication.processEvents()
        assert dialog.is_application_list_visible()
        assert not Conf.applications_list_collapsed.get()
        assert dialog.width() == width
        assert dialog.catalog_widget.width() == list_width
        assert dialog.application_stack.width() == page_width

        # The choice is restored by the next window
        dialog.set_application_list_visible(False)
        dialogs.append(ApplicationsDialog())
        assert not dialogs[-1].is_application_list_visible()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        Conf.applications_list_collapsed.set(previous_collapsed)
        for dialog in dialogs:
            dialog.close()
            dialog.deleteLater()
        QW.QApplication.processEvents()


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


def test_applications_dialog_search_and_welcome_preferences() -> None:
    """The catalog filters applications and edits their welcome page options"""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    capabilities = frozenset({PluginCapability.APPLICATION})
    camera = _plugin("org.example.camera", "Camera Characterization", capabilities)
    pulse = _plugin("org.example.pulse", "Pulse Characterization", capabilities)
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [camera, pulse]
    changes: list[None] = []
    try:
        dialog = ApplicationsDialog()
        dialog.SIG_WELCOME_PREFERENCES_CHANGED.connect(lambda: changes.append(None))
        items = [dialog.application_list.item(row) for row in range(2)]

        def selected() -> str:
            return dialog.application_list.currentItem().data(QC.Qt.UserRole)

        # Search words match names and descriptions, whatever their case
        dialog.search_edit.setText("PULSE description")
        assert [item.isHidden() for item in items] == [True, False]
        assert selected() == "org.example.pulse"
        assert dialog.application_stack.currentWidget().plugin is pulse
        dialog.search_edit.setText("characterization")
        assert [item.isHidden() for item in items] == [False, False]
        assert selected() == "org.example.pulse"
        dialog.search_edit.setText("pulse")
        dialog.select_plugin("org.example.camera")
        assert dialog.search_edit.text() == ""
        assert selected() == "org.example.camera"

        page = dialog.application_pages[0]
        assert page.show_on_welcome_checkbox.isChecked()
        assert not page.pin_on_welcome_checkbox.isChecked()
        page.pin_on_welcome_checkbox.setChecked(True)
        assert Conf.welcome_pinned_applications.get() == ["org.example.camera"]
        page.show_on_welcome_checkbox.setChecked(False)
        assert Conf.welcome_hidden_applications.get() == ["org.example.camera"]
        assert Conf.welcome_pinned_applications.get() == []
        assert not page.pin_on_welcome_checkbox.isChecked()
        page.pin_on_welcome_checkbox.setChecked(True)
        assert page.show_on_welcome_checkbox.isChecked()
        assert Conf.welcome_hidden_applications.get() == []
        assert len(changes) == 3

        # Preferences changed elsewhere are shown when the page is shown again
        set_application_hidden("org.example.camera", True)
        dialog.show()
        QW.QApplication.processEvents()
        assert not page.show_on_welcome_checkbox.isChecked()
        assert not page.pin_on_welcome_checkbox.isChecked()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.close()
            dialog.deleteLater()
            QW.QApplication.processEvents()


def test_main_window_exposes_applications_catalog(monkeypatch) -> None:
    """The catalog action reuses one modeless window without blocking DataLab."""
    executed: list[ApplicationsDialog] = []
    monkeypatch.setattr(
        main.ApplicationsDialog,
        "exec",
        lambda dialog: executed.append(dialog),
    )
    updates: list[tuple] = []
    monkeypatch.setattr(
        main.ApplicationsDialog,
        "schedule_readiness_update",
        lambda _dialog, *args: updates.append(args),
    )

    with datalab_test_app_context(console=False, exec_loop=False) as window:
        menu_actions = window.menuBar().actions()
        assert window.applications_action in menu_actions
        assert window.applications_action.text() == ""

        window.applications_action.trigger()
        QW.QApplication.processEvents()
        dialogs = window.findChildren(ApplicationsDialog)

        assert executed == []
        assert len(dialogs) == 1
        dialog = dialogs[0]
        assert dialog.isVisible()
        assert dialog.windowModality() == QC.Qt.NonModal
        assert window.isEnabled()

        window.set_current_panel("image")
        assert window.get_current_panel() == "image"
        assert dialog.isVisible()

        window.applications_action.trigger()
        QW.QApplication.processEvents()

        assert window.findChildren(ApplicationsDialog) == [dialog]

        # Editing objects (e.g. their metadata) re-assesses the methods
        updates.clear()
        window.imagepanel.SIG_OBJECT_MODIFIED.emit()
        assert updates


def test_add_metadata_suggests_keys_declared_by_applications() -> None:
    """Add metadata offers the keys expected by application methods"""
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
        recipes=(RECIPE,),
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


def test_application_commands_delegate_to_plugin_contracts(monkeypatch) -> None:
    """Each catalog command calls its plugin entry point and keeps the dialog."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    calls: list[tuple] = []
    opened_urls: list[str] = []
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
        recipes=(RECIPE,),
        examples=(EXAMPLE, DATASET),
        tools=(TOOL,),
        documentation_url="https://example.org/camera/docs",
        calls=calls,
    )
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [application]
    monkeypatch.setattr(
        applications_module.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    try:
        dialog = ApplicationsDialog()
        dialog.show()
        QW.QApplication.processEvents()
        page = dialog.application_pages[0]
        card = page.recipe_cards[RECIPE.recipe_id]
        page.documentation_button.click()
        card.run_button.click()
        card.example_buttons[EXAMPLE.id].click()
        page.dataset_buttons[DATASET.id].click()
        page.tool_buttons[TOOL.id].click()
        assert calls == [
            ("recipe", RECIPE.recipe_id),
            ("try", EXAMPLE.id, RECIPE.recipe_id),
            ("example", DATASET.id),
            ("tool", TOOL.id),
        ]
        assert opened_urls == ["https://example.org/camera/docs"]
        assert dialog.isVisible()
        assert Conf.welcome_recent_applications.get() == ["org.example.camera"]
        assert page.status_label.isHidden()

        application.launch_recipe = lambda _recipe_id: RecipeOutcome(
            diagnostics=(
                RecipeDiagnostic("warning", "excluded-shots", "2 shots excluded"),
            )
        )
        card.run_button.click()
        assert not page.status_label.isHidden()
        assert "Created 0 objects" in page.status_label.text()
        assert "Warning: 2 shots excluded" in page.status_label.text()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.close()
            dialog.deleteLater()
            QW.QApplication.processEvents()


def test_application_command_failures_are_reported_not_raised(monkeypatch) -> None:
    """A failing plugin entry point shows an error dialog instead of crashing."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
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
    registry = PluginRegistry.get_plugins()
    previous_plugins = list(registry)
    registry[:] = [application]
    try:
        dialog = ApplicationsDialog()
        page = dialog.application_pages[0]
        card = page.recipe_cards[RECIPE.recipe_id]
        card.run_button.click()
        card.example_buttons[EXAMPLE.id].click()
        page.dataset_buttons[DATASET.id].click()
        page.tool_buttons[TOOL.id].click()
        assert [message for message, _context in reported] == [
            "launcher exploded",
            "try exploded",
            "example exploded",
            "tool exploded",
        ]
        assert all(_context for _message, _context in reported)
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()
