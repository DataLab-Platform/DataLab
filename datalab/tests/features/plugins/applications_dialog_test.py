# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for the application plugin catalog."""

from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.gui import applications as applications_module
from datalab.gui import main
from datalab.gui.applications import ApplicationsDialog, get_application_plugins
from datalab.plugins import PluginCapability, PluginInfo, PluginRegistry
from datalab.tests import datalab_test_app_context


def _plugin(
    plugin_id: str,
    name: str,
    capabilities: frozenset[PluginCapability],
    *,
    recipes: tuple[object, ...] = (),
    examples: tuple[object, ...] = (),
    documentation_url: str | None = None,
    launched_recipes: list[str] | None = None,
    opened_examples: list[str] | None = None,
) -> object:
    """Build the minimal active-plugin surface consumed by the catalog."""
    return SimpleNamespace(
        plugin_id=plugin_id,
        info=PluginInfo(
            id=plugin_id,
            name=name,
            version="1.2.3",
            description=f"{name} description",
            capabilities=capabilities,
            documentation_url=documentation_url,
        ),
        get_recipes=lambda: recipes,
        get_examples=lambda: examples,
        get_recipe_launchers=lambda: {recipe.recipe_id: "launch" for recipe in recipes},
        launch_recipe=lambda recipe_id: (
            None if launched_recipes is None else launched_recipes.append(recipe_id)
        ),
        launch_example=lambda example_id: (
            None if opened_examples is None else opened_examples.append(example_id)
        ),
    )


def test_applications_dialog_filters_and_renders_declared_contracts() -> None:
    """Only application plugins expose their recipes and examples."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    recipe = SimpleNamespace(
        recipe_id="org.example.camera:quick-check",
        title="Quick camera check",
        version="2.0.0",
        description="Run the quick workflow",
    )
    example = SimpleNamespace(
        id="quickstart",
        title="Scientific camera quickstart",
        description="Open the packaged workspace",
    )
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION, PluginCapability.PROCESSING}),
        recipes=(recipe,),
        examples=(example,),
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
        assert page.recipe_list.count() == 1
        recipe_item = page.recipe_list.item(0)
        assert recipe_item.data(QC.Qt.UserRole) == recipe.recipe_id
        assert recipe_item.data(applications_module.CATALOG_TITLE_ROLE) == recipe.title
        assert recipe_item.data(applications_module.CATALOG_DESCRIPTION_ROLE) == (
            recipe.description
        )
        assert recipe_item.data(applications_module.CATALOG_VERSION_ROLE) == (
            recipe.version
        )
        assert page.example_list.count() == 1
        example_item = page.example_list.item(0)
        assert example_item.data(QC.Qt.UserRole) == example.id
        assert (
            example_item.data(applications_module.CATALOG_TITLE_ROLE) == example.title
        )
        assert example_item.data(applications_module.CATALOG_DESCRIPTION_ROLE) == (
            example.description
        )
        assert page.start_button.isEnabled()
        assert page.open_example_button.isEnabled()
        assert page.documentation_button.isEnabled()
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


def test_application_commands_require_declared_targets() -> None:
    """Commands stay disabled without a launcher, example, or documentation."""
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
        assert not page.start_button.isEnabled()
        assert not page.open_example_button.isEnabled()
        assert not page.documentation_button.isEnabled()
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()


def test_main_window_exposes_applications_catalog(monkeypatch) -> None:
    """The catalog has a visible main-menu entry independent of plugin actions."""
    opened: list[ApplicationsDialog] = []
    monkeypatch.setattr(
        main.ApplicationsDialog,
        "exec",
        lambda dialog: opened.append(dialog),
    )

    with datalab_test_app_context(console=False) as window:
        menu_actions = window.menuBar().actions()
        assert window.applications_action in menu_actions
        assert window.applications_action.text() == ""

        window.applications_action.trigger()
        assert len(opened) == 1
        assert opened[0].parent() is window


def test_application_commands_delegate_to_plugin_contracts(monkeypatch) -> None:
    """Catalog commands preserve plugin-owned interactions and documentation."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    recipe = SimpleNamespace(
        recipe_id="org.example.camera:quick-check",
        title="Quick camera check",
        version="2.0.0",
        description="Run the quick workflow",
    )
    example = SimpleNamespace(
        id="quickstart",
        title="Scientific camera quickstart",
        description="Open the packaged workspace",
    )
    launched_recipes: list[str] = []
    opened_examples: list[str] = []
    opened_urls: list[str] = []
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
        recipes=(recipe,),
        examples=(example,),
        documentation_url="https://example.org/camera/docs",
        launched_recipes=launched_recipes,
        opened_examples=opened_examples,
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
        page = dialog.application_pages[0]
        page.documentation_button.click()
        page.start_button.click()
        assert launched_recipes == [recipe.recipe_id]
        assert opened_urls == ["https://example.org/camera/docs"]

        dialog = ApplicationsDialog()
        dialog.application_pages[0].open_example_button.click()
        assert opened_examples == [example.id]
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()


def test_application_command_failures_are_reported_not_raised(monkeypatch) -> None:
    """A failing plugin launcher shows an error dialog instead of crashing."""
    qt_app = QW.QApplication.instance() or QW.QApplication([])
    recipe = SimpleNamespace(
        recipe_id="org.example.camera:quick-check",
        title="Quick camera check",
        version="2.0.0",
        description="Run the quick workflow",
    )
    example = SimpleNamespace(
        id="quickstart",
        title="Scientific camera quickstart",
        description="Open the packaged workspace",
    )
    application = _plugin(
        "org.example.camera",
        "Camera Characterization",
        frozenset({PluginCapability.APPLICATION}),
        recipes=(recipe,),
        examples=(example,),
    )

    def failing_launch_recipe(recipe_id: str) -> None:
        raise RuntimeError("launcher exploded")

    def failing_launch_example(example_id: str) -> None:
        raise RuntimeError("example exploded")

    application.launch_recipe = failing_launch_recipe
    application.launch_example = failing_launch_example
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
        dialog.application_pages[0].start_button.click()
        dialog = ApplicationsDialog()
        dialog.application_pages[0].open_example_button.click()
        assert [message for message, _context in reported] == [
            "launcher exploded",
            "example exploded",
        ]
        assert all(_context for _message, _context in reported)
        assert qt_app is not None
    finally:
        registry[:] = previous_plugins
        if "dialog" in locals():
            dialog.deleteLater()
