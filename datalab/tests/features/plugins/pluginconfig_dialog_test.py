# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plugin UI configuration and interactive behavior tests."""

from __future__ import annotations

from datetime import datetime

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import (
    OTHER_PLUGINS_PATHLIST,
    Conf,
    get_user_plugin_paths,
    set_user_plugin_paths,
)
from datalab.env import execenv
from datalab.gui import pluginconfig
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


def _make_dummy_plugin_class(name: str, description: str, filepath: str | None = None):
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
    if filepath is not None:
        DummyPlugin.__plugin_filepath__ = filepath

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
    plugin_1_name = "Test Plugin 1"
    plugin_2_name = "Test Plugin 2"
    main_config = Conf.to_dict().get("main", {})
    had_config = "plugins_enabled_list" in main_config
    original_enabled_list = Conf.main.plugins_enabled_list.get(None)
    plugin_1_path: str | None = None
    plugin_2_path: str | None = None

    try:
        with temporary_plugin_dir() as plugin_dir:
            execenv.print(f"Using temporary plugin directory: {plugin_dir}")
            plugin_1_path = create_plugin_file(
                plugin_dir,
                "datalab_test_plugin_1.py",
                "TestPluginOne",
                plugin_1_name,
                "Action One",
                "action_1",
            )
            plugin_2_path = create_plugin_file(
                plugin_dir,
                "datalab_test_plugin_2.py",
                "TestPluginTwo",
                plugin_2_name,
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
                assert plugin_1_name in widget_names
                assert plugin_2_name in widget_names
                plugin_1_widget = next(
                    widget
                    for widget in dialog.plugin_widgets
                    if widget.plugin_class.PLUGIN_INFO.name == plugin_1_name
                )
                plugin_2_widget = next(
                    widget
                    for widget in dialog.plugin_widgets
                    if widget.plugin_class.PLUGIN_INFO.name == plugin_2_name
                )
                assert plugin_1_widget.plugin_filepath == plugin_1_path
                assert plugin_2_widget.plugin_filepath == plugin_2_path
                assert plugin_1_widget.open_file_button is not None
                assert plugin_1_widget.show_in_folder_button is not None
                assert plugin_2_widget.open_file_button is not None
                assert plugin_2_widget.show_in_folder_button is not None
                assert dialog.toggle_all_checkbox.checkState() == QC.Qt.Checked

                dialog.filter_combo.setCurrentIndex(2)
                QW.QApplication.processEvents()
                assert [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog.plugin_widgets
                    if widget.isVisible()
                ] == []
                _close_dialog(dialog)

                Conf.main.plugins_enabled_list.set([plugin_1_name])
                win.reload_plugins()
                QW.QApplication.processEvents()

                dialog2 = PluginConfigDialog(win)
                _show_dialog(dialog2)
                enabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.checkbox.isChecked()
                ]
                assert enabled_names == [plugin_1_name]
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
                assert plugin_1_name in visible_enabled_names
                assert plugin_2_name not in visible_enabled_names

                dialog2.filter_combo.setCurrentIndex(2)
                QW.QApplication.processEvents()
                visible_disabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert plugin_2_name in visible_disabled_names
                assert plugin_1_name not in visible_disabled_names

                plugin_2_widget = next(
                    widget
                    for widget in dialog2.plugin_widgets
                    if widget.plugin_class.PLUGIN_INFO.name == plugin_2_name
                )
                plugin_2_widget.checkbox.setChecked(True)
                QW.QApplication.processEvents()
                visible_disabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert plugin_2_name in visible_disabled_names

                dialog2.filter_combo.setCurrentIndex(1)
                QW.QApplication.processEvents()
                visible_enabled_names = [
                    widget.plugin_class.PLUGIN_INFO.name
                    for widget in dialog2.plugin_widgets
                    if widget.isVisible()
                ]
                assert plugin_2_name not in visible_enabled_names

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


def test_last_load_text_uses_today_yesterday_or_date():
    """Last load label should use relative day words when applicable."""

    def locale_to_string(value, _format):
        """Return deterministic date/time strings for label tests."""
        return (
            f"{value.year():04d}-{value.month():02d}-{value.day():02d}"
            if isinstance(value, QC.QDate)
            else f"{value.hour():02d}:{value.minute():02d}"
        )

    locale = type("DummyLocale", (), {"toString": staticmethod(locale_to_string)})()
    now = datetime(2026, 5, 19, 15, 30)

    today_text = pluginconfig.format_last_load_text(
        datetime(2026, 5, 19, 9, 45), now=now, locale=locale
    )
    yesterday_text = pluginconfig.format_last_load_text(
        datetime(2026, 5, 18, 23, 10), now=now, locale=locale
    )
    older_timestamp = datetime(2026, 5, 17, 8, 5)
    older_text = pluginconfig.format_last_load_text(
        older_timestamp, now=now, locale=locale
    )

    assert today_text == "Last loaded: today at 09:45"
    assert yesterday_text == "Last loaded: yesterday at 23:10"
    assert "today" not in older_text
    assert "yesterday" not in older_text
    assert older_text.endswith("08:05")
    assert "2026-05-17" in older_text


def test_plugin_dialog_shows_latest_load_text(monkeypatch):
    """Dialog should display the most recent timestamp between startup and load."""

    def _format_load_marker(timestamp, now=None, locale=None):
        del now, locale
        return f"Last loaded marker: {timestamp.hour:02d}:{timestamp.minute:02d}"

    monkeypatch.setattr(
        pluginconfig,
        "format_last_load_text",
        _format_load_marker,
    )

    with datalab_test_app_context(console=False) as win:
        win.started_at = datetime(2026, 5, 19, 9, 0)
        win.plugins_last_load_at = datetime(2026, 5, 19, 11, 30)

        dialog = PluginConfigDialog(win)
        _show_dialog(dialog)

        assert dialog.load_info_label is not None
        assert dialog.load_info_label.text() == "Last loaded marker: 11:30"

        _close_dialog(dialog)


def test_plugin_search_paths_can_be_added_edited_removed_and_persisted(
    monkeypatch, tmp_path
):
    """Search path tab should manage persistent extra plugin directories."""
    added_dir = tmp_path / "plugins_added"
    edited_dir = tmp_path / "plugins_edited"
    kept_dir = tmp_path / "plugins_kept"
    for directory in (added_dir, edited_dir, kept_dir):
        directory.mkdir()

    original_paths = get_user_plugin_paths()

    def answer_no(*args, **kwargs):
        """Decline plugin reload after saving configuration."""
        del args, kwargs
        return QW.QMessageBox.No

    try:
        set_user_plugin_paths([])
        with datalab_test_app_context(console=False) as win:
            dialog = PluginConfigDialog(win)
            _show_dialog(dialog)

            assert len(dialog.fixed_path_widgets) >= 2
            assert all(
                widget.edit_button is None and widget.delete_button is None
                for widget in dialog.fixed_path_widgets
            )
            assert dialog.add_path_button is not None
            assert not dialog.extra_path_widgets

            selected_paths = iter([str(added_dir), str(edited_dir), str(kept_dir)])

            def browse_directory(_initial_path=None):
                """Return deterministic directories for add/edit actions."""
                return next(selected_paths)

            monkeypatch.setattr(dialog, "_browse_plugin_directory", browse_directory)
            monkeypatch.setattr(QW.QMessageBox, "question", answer_no)

            dialog.add_path_button.click()
            QW.QApplication.processEvents()
            assert [widget.path for widget in dialog.extra_path_widgets] == [
                str(added_dir)
            ]

            editable_widget = dialog.extra_path_widgets[0]
            assert editable_widget.edit_button is not None
            assert editable_widget.delete_button is not None

            editable_widget.edit_button.click()
            QW.QApplication.processEvents()
            assert [widget.path for widget in dialog.extra_path_widgets] == [
                str(edited_dir)
            ]

            dialog.add_path_button.click()
            QW.QApplication.processEvents()
            assert [widget.path for widget in dialog.extra_path_widgets] == [
                str(edited_dir),
                str(kept_dir),
            ]

            dialog.extra_path_widgets[0].delete_button.click()
            QW.QApplication.processEvents()
            assert [widget.path for widget in dialog.extra_path_widgets] == [
                str(kept_dir)
            ]

            dialog.accept()
            QW.QApplication.processEvents()

        assert get_user_plugin_paths() == [str(kept_dir)]

        with datalab_test_app_context(console=False) as win:
            dialog = PluginConfigDialog(win)
            _show_dialog(dialog)
            assert [widget.path for widget in dialog.extra_path_widgets] == [
                str(kept_dir)
            ]
            _close_dialog(dialog)
    finally:
        set_user_plugin_paths(original_paths)


def test_env_var_plugin_paths_appear_as_fixed_read_only_entries(monkeypatch, tmp_path):
    """Environment-provided plugin directories should be visible but not editable."""
    env_dir = tmp_path / "env_plugins"
    env_dir.mkdir()
    env_path = str(env_dir)
    original_paths = get_user_plugin_paths()

    try:
        set_user_plugin_paths([])
        monkeypatch.setattr(pluginconfig, "DATALAB_PLUGINS_ENV_PATHS", [env_path])
        monkeypatch.setattr(
            pluginconfig,
            "OTHER_PLUGINS_PATHLIST",
            OTHER_PLUGINS_PATHLIST + [env_path],
        )

        with datalab_test_app_context(console=False) as win:
            dialog = PluginConfigDialog(win)
            _show_dialog(dialog)

            env_widget = next(
                widget
                for widget in dialog.fixed_path_widgets
                if widget.path == env_path
            )
            assert env_widget.edit_button is None
            assert env_widget.delete_button is None
            assert not any(
                widget.path == env_path for widget in dialog.extra_path_widgets
            )

            fixed_paths = [widget.path for widget in dialog.fixed_path_widgets]
            assert env_path in fixed_paths
            _close_dialog(dialog)
    finally:
        set_user_plugin_paths(original_paths)


def test_apply_and_reload_button_keeps_dialog_open_and_saves_changes(
    monkeypatch, tmp_path
):
    """Apply/reload button should save changes, reload plugins, and keep dialog open."""
    added_dir = tmp_path / "plugins_added"
    added_dir.mkdir()
    original_paths = get_user_plugin_paths()

    try:
        set_user_plugin_paths([])
        with datalab_test_app_context(console=False) as win:
            dialog = PluginConfigDialog(win)
            _show_dialog(dialog)

            assert dialog.reload_button is not None
            assert dialog.reload_button.text() == "Apply and reload plugins"

            monkeypatch.setattr(
                dialog,
                "_browse_plugin_directory",
                lambda _initial_path=None: str(added_dir),
            )

            reload_calls: list[bool] = []

            def fake_reload_plugins() -> None:
                """Record reload requests without closing the dialog."""
                reload_calls.append(True)
                win.plugins_last_load_at = datetime(2026, 5, 20, 14, 30)

            monkeypatch.setattr(win, "reload_plugins", fake_reload_plugins)

            dialog.add_path_button.click()
            QW.QApplication.processEvents()
            dialog.reload_button.click()
            QW.QApplication.processEvents()

            assert reload_calls == [True]
            assert dialog.isVisible()
            assert get_user_plugin_paths() == [str(added_dir)]

            _close_dialog(dialog)
    finally:
        set_user_plugin_paths(original_paths)


def test_plugin_many_actions_menu_behavior():
    """Test plugin with many actions in dropdown menu."""
    main_config = Conf.to_dict().get("main", {})
    had_config = "plugins_enabled_list" in main_config
    original_enabled_list = Conf.main.plugins_enabled_list.get(None)

    try:
        Conf.main.plugins_enabled_list.set(None)
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
    finally:
        if had_config:
            Conf.main.plugins_enabled_list.set(original_enabled_list)
        else:
            Conf.main.plugins_enabled_list.remove()


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


def test_plugin_widget_can_open_plugin_file_and_show_in_folder(monkeypatch):
    """Plugin widget should expose actions for opening file and showing it."""
    opened_paths: list[str] = []
    shown_paths: list[str] = []

    def _open_local_path(path: str) -> bool:
        opened_paths.append(path)
        return True

    def _show_in_folder(path: str) -> bool:
        shown_paths.append(path)
        return True

    monkeypatch.setattr(pluginconfig, "_open_local_path", _open_local_path)
    monkeypatch.setattr(pluginconfig, "_show_in_folder", _show_in_folder)

    with datalab_test_app_context(console=False):
        widget = PluginInfoWidget(
            _make_dummy_plugin_class(
                "Path Actions Test",
                "Plugin with location actions.",
                filepath=__file__,
            ),
            enabled=True,
            state=PluginState.ENABLED,
        )

        assert widget.open_file_button is not None
        assert widget.show_in_folder_button is not None
        assert widget.show_in_folder_button.text() == "Show in folder"

        widget.open_file_button.click()
        widget.show_in_folder_button.click()

        assert opened_paths == [__file__]
        assert shown_paths == [__file__]

        widget.close()
        widget.deleteLater()
        QW.QApplication.processEvents()


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


def test_failed_plugin_widget_can_open_plugin_file_and_show_in_folder(monkeypatch):
    """Failed plugin widget should expose actions for opening file and showing it."""
    opened_paths: list[str] = []
    shown_paths: list[str] = []

    def _open_local_path(path: str) -> bool:
        opened_paths.append(path)
        return True

    def _show_in_folder(path: str) -> bool:
        shown_paths.append(path)
        return True

    monkeypatch.setattr(pluginconfig, "_open_local_path", _open_local_path)
    monkeypatch.setattr(pluginconfig, "_show_in_folder", _show_in_folder)

    failed_info = FailedPluginInfo(
        name="bad_plugin.py",
        filepath=__file__,
        traceback="Traceback",
    )

    with datalab_test_app_context(console=False):
        widget = FailedPluginInfoWidget(failed_info)

        assert widget.open_file_button is not None
        assert widget.show_in_folder_button is not None
        assert widget.show_in_folder_button.text() == "Show in folder"

        widget.open_file_button.click()
        widget.show_in_folder_button.click()

        assert opened_paths == [__file__]
        assert shown_paths == [__file__]

        widget.close()
        widget.deleteLater()
        QW.QApplication.processEvents()
