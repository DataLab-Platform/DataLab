# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plugin UI configuration and interactive behavior tests."""

from __future__ import annotations

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.pluginconfig import (
    ExpandableTextWidget,
    FailedPluginInfoWidget,
    PluginConfigDialog,
    PluginInfoWidget,
    PluginState,
)
from datalab.plugins import FailedPluginInfo
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.plugin_test_dataset import (
    create_plugin_file,
    temporary_plugin_dir,
    temporary_template_plugin,
)


def _make_dummy_plugin_class(name: str, description: str):
    """Create a minimal plugin class for UI-only widget tests."""

    class DummyPlugin:
        """Minimal plugin-like class exposing PLUGIN_INFO."""

    DummyPlugin.PLUGIN_INFO = type(
        "PluginInfoHolder",
        (),
        {
            "name": name,
            "version": "1.0.0",
            "description": description,
            "icon": None,
        },
    )()

    return DummyPlugin


def _show_dialog(dialog: PluginConfigDialog) -> None:
    """Show a dialog so child widget visibility reflects the active filter."""
    dialog.show()
    QW.QApplication.processEvents()


def _close_dialog(dialog: PluginConfigDialog) -> None:
    """Close and delete a dialog to avoid leaking UI state between tests."""
    dialog.close()
    dialog.deleteLater()
    QW.QApplication.processEvents()


def _create_plugin_description_widget(
    name: str, description: str, *, width: int
) -> PluginInfoWidget:
    """Create a plugin info widget with a controlled rendered width."""
    widget = PluginInfoWidget(
        _make_dummy_plugin_class(name, description),
        enabled=True,
        state=PluginState.ENABLED,
    )
    widget.setFixedWidth(width)
    widget.show()
    widget.description_widget.refresh_description()
    QW.QApplication.processEvents()
    return widget


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

                plugin_2_widget = next(
                    widget
                    for widget in dialog2.plugin_widgets
                    if widget.plugin_class.PLUGIN_INFO.name == "Test Plugin 2"
                )
                plugin_2_widget.checkbox.setChecked(True)
                QW.QApplication.processEvents()
                visible_disabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert "Test Plugin 2" in visible_disabled_names

                dialog2.filter_combo.setCurrentIndex(1)
                QW.QApplication.processEvents()
                visible_enabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert "Test Plugin 2" not in visible_enabled_names

                dialog2.set_all_enabled(True)
                QW.QApplication.processEvents()
                assert all(
                    widget.checkbox.isChecked() for widget in dialog2.plugin_widgets
                )
                _close_dialog(dialog2)
    finally:
        if had_config:
            Conf.main.plugins_enabled_list.set(original_enabled_list)
        else:
            Conf.main.plugins_enabled_list.remove()


def test_plugin_many_actions_menu_behavior():
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
            assert "menu-scrollable" in win.plugins_menu.styleSheet()
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
            assert "menu-scrollable" in test_menu.styleSheet()

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
    with datalab_test_app_context(console=False):
        widget = _create_plugin_description_widget(
            "Long Description Test", long_description, width=260
        )

        toggle_description = getattr(widget, "_toggle_description")
        toggle_button = widget.toggle_button
        desc_label = getattr(widget, "desc_label")
        scroll_area = widget.description_widget.scroll_area

        assert toggle_button.isVisible()
        assert not widget.description_widget.is_expanded()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAlwaysOff

        toggle_description()
        assert getattr(widget, "_expanded") is True
        assert widget.description_widget.is_expanded()
        expanded_height = desc_label.height()

        toggle_description()
        assert getattr(widget, "_expanded") is False
        assert not widget.description_widget.is_expanded()
        assert desc_label.height() < expanded_height
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAlwaysOff
        widget.close()
        widget.deleteLater()
        QW.QApplication.processEvents()


def test_plugin_description_toggle_depends_on_dialog_width():
    """Show more should depend on rendered height, not raw character count."""
    description = (
        "This description is intentionally sized so that it needs the 'Show more' "
        "button in the default plugin configuration width, yet it should become "
        "fully visible once the configuration dialog is made significantly wider. "
        "That verifies the behavior depends on rendered space instead of raw "
        "character count alone."
    )
    app = QW.QApplication.instance() or QW.QApplication([])
    widget = ExpandableTextWidget(description)
    assert widget.needs_toggle_for_width(240)

    wide_width = next(
        width
        for width in (800, 1200, 1600, 2400)
        if not widget.needs_toggle_for_width(width)
    )

    widget.setFixedWidth(240)
    widget.refresh_description()
    assert not widget.toggle_button.isHidden()
    assert not widget.is_expanded()

    widget.setFixedWidth(wide_width)
    widget.set_expanded(True)
    assert widget.label.toPlainText() == description
    widget.close()
    widget.deleteLater()
    QW.QApplication.processEvents()
    assert app is not None


def test_plugin_very_long_description_scrolls_only_when_expanded():
    """Very long descriptions should use an internal scrollbar only when expanded."""
    description = " ".join(["Very long plugin description for scrollbar testing."] * 80)

    with datalab_test_app_context(console=False):
        widget = _create_plugin_description_widget(
            "Very Long Description Test", description, width=260
        )

        scroll_area = widget.description_widget.scroll_area
        collapsed_height = scroll_area.height()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAlwaysOff
        assert not widget.description_widget.is_expanded()

        widget.description_widget.set_expanded(True)
        QW.QApplication.processEvents()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAsNeeded
        assert scroll_area.height() > collapsed_height

        widget.description_widget.set_expanded(False)
        QW.QApplication.processEvents()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAlwaysOff
        assert scroll_area.height() == collapsed_height
        widget.close()
        widget.deleteLater()
        QW.QApplication.processEvents()


def test_failed_plugin_description_uses_same_expand_collapse_behavior():
    """Failed plugins should use the same truncated/expanded display behavior."""
    traceback_text = "\n".join(
        [
            f"Traceback line {index}: import failure in plugin loading."
            for index in range(80)
        ]
    )
    failed_info = FailedPluginInfo(
        name="bad_plugin.py",
        filepath="C:/plugins/bad_plugin.py",
        traceback=traceback_text,
    )

    with datalab_test_app_context(console=False):
        widget = FailedPluginInfoWidget(failed_info)
        widget.setFixedWidth(260)
        widget.show()
        widget.description_widget.refresh_description()
        QW.QApplication.processEvents()

        scroll_area = widget.description_widget.scroll_area
        collapsed_height = scroll_area.height()
        assert widget.description_widget.toggle_button.isVisible()
        assert not widget.description_widget.is_expanded()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAlwaysOff

        widget.description_widget.set_expanded(True)
        QW.QApplication.processEvents()
        assert widget.description_widget.is_expanded()
        assert scroll_area.verticalScrollBarPolicy() == QC.Qt.ScrollBarAsNeeded
        assert scroll_area.height() > collapsed_height

        widget.close()
        widget.deleteLater()
        QW.QApplication.processEvents()
