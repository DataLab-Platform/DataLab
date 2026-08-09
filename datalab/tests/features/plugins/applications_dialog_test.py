# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for the read-only application plugin catalog."""

from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

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
        ),
        get_recipes=lambda: recipes,
        get_examples=lambda: examples,
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
        assert dialog.application_list.minimumWidth() == 220
        assert dialog.application_list.maximumWidth() == 320
        assert dialog.application_list.item(0).data(QC.Qt.UserRole) == (
            "org.example.camera"
        )
        page = dialog.application_pages[0]
        assert page.recipe_list.count() == 1
        assert page.recipe_list.item(0).data(QC.Qt.UserRole) == recipe.recipe_id
        assert page.example_list.count() == 1
        assert page.example_list.item(0).data(QC.Qt.UserRole) == example.id
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
        assert window.applications_action.text() == "Applications..."

        window.applications_action.trigger()
        assert len(opened) == 1
        assert opened[0].parent() is window
