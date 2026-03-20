# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plugin UI configuration and visual behavior tests."""

from __future__ import annotations

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.pluginconfig import PluginConfigDialog, PluginInfoWidget, PluginState
from datalab.plugins import PluginRegistry
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.test_plugins import (
    create_plugin_file,
    temporary_plugin_dir,
    temporary_template_plugin,
)


def _show_dialog(dialog: PluginConfigDialog) -> None:
    """Show a dialog so child widget visibility reflects the active filter."""
    dialog.show()
    QW.QApplication.processEvents()


def _close_dialog(dialog: PluginConfigDialog) -> None:
    """Close and delete a dialog to avoid leaking UI state between tests."""
    dialog.close()
    dialog.deleteLater()
    QW.QApplication.processEvents()


def test_plugin_enable_disable_config():
    """Test plugin enable/disable filtering and configuration dialog."""
    main_config = Conf.to_dict().get("main", {})
    had_config = "plugins_enabled_list" in main_config
    original_enabled_list = Conf.main.plugins_enabled_list.get(None)

    try:
        with temporary_plugin_dir() as plugin_dir:
            execenv.print(f"Using temporary plugin directory: {plugin_dir}")
            create_plugin_file(
                plugin_dir,
                "datalab_test_plugin_1.py",
                "TestPluginOne",
                "Test Plugin 1",
                "Action One",
                "action_1",
            )
            create_plugin_file(
                plugin_dir,
                "datalab_test_plugin_2.py",
                "TestPluginTwo",
                "Test Plugin 2",
                "Action Two",
                "action_2",
            )
            Conf.main.plugins_enabled_list.set(None)

            with datalab_test_app_context(console=False) as win:
                QW.QApplication.processEvents()
                dialog = PluginConfigDialog(win)
                _show_dialog(dialog)
                widget_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog.plugin_widgets
                ]
                assert "Test Plugin 1" in widget_names
                assert "Test Plugin 2" in widget_names
                assert dialog.toggle_all_checkbox.checkState() == QC.Qt.Checked

                dialog.filter_combo.setCurrentIndex(2)
                QW.QApplication.processEvents()
                assert [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog.plugin_widgets
                    if widget.isVisible()
                ] == []
                _close_dialog(dialog)

                Conf.main.plugins_enabled_list.set(["Test Plugin 1"])
                win.reload_plugins()
                QW.QApplication.processEvents()

                dialog2 = PluginConfigDialog(win)
                _show_dialog(dialog2)
                enabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.checkbox.isChecked()
                ]
                assert enabled_names == ["Test Plugin 1"]
                assert dialog2.toggle_all_checkbox.checkState() == (
                    QC.Qt.PartiallyChecked
                )

                dialog2.filter_combo.setCurrentIndex(1)
                QW.QApplication.processEvents()
                visible_enabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert "Test Plugin 1" in visible_enabled_names
                assert "Test Plugin 2" not in visible_enabled_names

                dialog2.filter_combo.setCurrentIndex(2)
                QW.QApplication.processEvents()
                visible_disabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert "Test Plugin 2" in visible_disabled_names
                assert "Test Plugin 1" not in visible_disabled_names

                dialog2._set_all_enabled(True)
                assert all(
                    widget.checkbox.isChecked() for widget in dialog2.plugin_widgets
                )
                _close_dialog(dialog2)
    finally:
        if had_config:
            Conf.main.plugins_enabled_list.set(original_enabled_list)
        else:
            Conf.main.plugins_enabled_list.remove()


def test_plugin_many_actions_visual():
    """Test plugin with many actions in dropdown menu."""
    with temporary_template_plugin(
        "datalab_test_plugin_many_actions.py",
        "plugin_many_actions.py.template",
        {
            "{class_name}": "TestPluginManyActions",
            "{plugin_name}": "Many Actions Test",
            "{menu_name}": "Test Menu with Many Actions",
            "{action_prefix}": "Test Action",
            "{test_code_1}": "self.main._test_action_1 = True",
            "{test_code_2}": "self.main._test_action_2 = True",
            "{test_code_3}": "self.main._test_action_3 = True",
            "{test_code_4}": "self.main._test_action_4 = True",
            "{test_code_5}": "self.main._test_action_5 = True",
        },
    ):
        with datalab_test_app_context(console=False) as win:
            QW.QApplication.processEvents()
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()
            win.plugins_menu.aboutToShow.emit()
            plugin_actions = win.signalpanel.get_category_actions(
                ActionCategory.PLUGINS
            )

            test_menu = next(
                item
                for item in plugin_actions
                if isinstance(item, QW.QMenu)
                and item.title() == "Test Menu with Many Actions"
            )
            test_menu.aboutToShow.emit()

            action_texts = [
                action.text()
                for action in test_menu.actions()
                if not action.isSeparator()
            ]
            assert len(action_texts) == 5
            assert "Test Action 3" in action_texts
            action_3 = next(
                action
                for action in test_menu.actions()
                if action.text() == "Test Action 3"
            )
            assert action_3.isEnabled()


def test_plugin_long_description():
    """Test plugin with very long description and expand/collapse button."""
    long_description = (
        "This is an extremely long description that is designed to test "
        "how the plugin system handles descriptions that span multiple lines "
        "and contain a large amount of text. The description should not break "
        "the UI layout or cause any rendering issues. It should be properly "
        "truncated or wrapped in any display contexts such as tooltips, status "
        "bars, or configuration dialogs. This text continues to be very long "
        "to ensure we adequately test the edge case of exceptionally verbose "
        "plugin descriptions that might be provided by third-party developers "
        "who want to thoroughly explain what their plugin does and how to use it."
    )

    with temporary_template_plugin(
        "datalab_test_plugin_long_desc.py",
        "plugin_long_description.py.template",
        {
            "{class_name}": "TestPluginLongDescription",
            "{plugin_name}": "Long Description Test",
            "{long_description}": long_description,
            "{action_name}": "Test Action",
            "{test_code}": "self.main._test_long_desc = True",
        },
    ):
        with datalab_test_app_context(console=False) as win:
            QW.QApplication.processEvents()
            plugin = next(
                plugin
                for plugin in PluginRegistry.get_plugins()
                if plugin.info.name == "Long Description Test"
            )
            plugin_class = type(plugin)

            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()
            win.plugins_menu.aboutToShow.emit()
            for action in win.plugins_menu.actions():
                if action.text() == "Test Action":
                    action.trigger()
                    break
            assert getattr(win, "_test_long_desc", None) is True

            widget = PluginInfoWidget(
                plugin_class, enabled=True, state=PluginState.ENABLED
            )
            toggle_description = getattr(widget, "_toggle_description")
            desc_scroll = getattr(widget, "_desc_scroll")
            toggle_description()
            assert getattr(widget, "_expanded") is True
            assert desc_scroll.maximumHeight() == 150
            toggle_description()
            assert getattr(widget, "_expanded") is False
            assert desc_scroll.maximumHeight() == 60
