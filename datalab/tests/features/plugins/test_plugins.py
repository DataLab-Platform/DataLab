# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin system tests
-------------------

Testing the plugin system (discovery, loading, reloading, cleanup) using
the main window.
"""

# guitest: show

import importlib
import importlib.util
import os
import os.path as osp
from unittest.mock import patch

from qtpy import QtWidgets as QW

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.main import DLMainWindow
from datalab.plugins import PluginRegistry
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.plugin_test_dataset import (
    create_plugin_file,
    create_plugin_from_template,
    temporary_plugin_dir,
)


# This end-to-end regression test intentionally keeps the whole plugin
# lifecycle in one linear scenario: discovery, menu wiring, reload and cleanup.
def test_plugin_system():  # pylint: disable=too-many-statements
    """Test the entire plugin lifecycle: discovery, reload, cleanup"""
    # Ensure plugins_enabled_list is None (all plugins enabled)
    Conf.main.plugins_enabled_list.set(None)

    # We need to monkeypatch the plugin path or add to sys.path
    # The temporary_plugin_dir context manager handles adding to sys.path

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # 1. Create a valid plugin
        plugin1_filename = "datalab_test_plugin_1.py"
        create_plugin_file(
            plugin_dir,
            plugin1_filename,
            "TestPlugin1",
            "Test Plugin 1",
            "Action Plugin 1",
            "action_plugin_1",
        )

        # Force discovery of new plugin by clearing sys.modules cache for it if needed
        # and re-running discovery
        # DataLab main window startup calls discover_plugins().
        # We need to make sure our plugin_dir is in sys.path BEFORE main window creation
        # The context manager adds it to sys.path, so it should be fine.

        # Start the application
        # We start with one plugin already present
        with datalab_test_app_context(console=False) as win:
            win: DLMainWindow

            # Allow time for initial load
            QW.QApplication.processEvents()

            # --- Check Initial State ---
            execenv.print("Verifying initial plugin load...")

            # Check if plugin is registered
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            execenv.print(f"Registered plugins: {plugin_names}")

            # We must be tolerant of other plugins being present
            assert "Test Plugin 1" in plugin_names, (
                f"Test Plugin 1 should be loaded. Loaded: {plugin_names}"
            )

            # --- Test 2: Add a second plugin and Reload ---
            execenv.print("Adding second plugin and reloading...")

            plugin2_filename = "datalab_test_plugin_2.py"
            create_plugin_file(
                plugin_dir,
                plugin2_filename,
                "TestPlugin2",
                "Test Plugin 2",
                "Action Plugin 2",
                "action_plugin_2",
            )

            # Trigger reload
            win.reload_plugins()
            QW.QApplication.processEvents()

            # Verify both plugins are present
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            execenv.print(f"Registered plugins after reload: {plugin_names}")
            assert "Test Plugin 1" in plugin_names
            assert "Test Plugin 2" in plugin_names

            # Verify menu updates
            # Verify Signal Panel actions in menu
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()

            win.plugins_menu.aboutToShow.emit()
            actions = win.plugins_menu.actions()
            action_names = [a.text() for a in actions]
            execenv.print(f"Menu actions after reload (Signal): {action_names}")

            for suffix in ["_common", "_signal"]:
                assert f"Action Plugin 1{suffix}" in action_names
                assert f"Action Plugin 2{suffix}" in action_names

            # Image actions should NOT be here
            assert "Action Plugin 1_image" not in action_names
            assert "Action Plugin 2_image" not in action_names

            # Verify Image Panel actions in menu
            win.tabwidget.setCurrentWidget(win.imagepanel)
            QW.QApplication.processEvents()

            win.plugins_menu.aboutToShow.emit()
            actions = win.plugins_menu.actions()
            action_names = [a.text() for a in actions]
            execenv.print(
                f"Menu actions after switching to Image Panel: {action_names}"
            )

            for suffix in ["_common", "_image"]:
                assert f"Action Plugin 1{suffix}" in action_names
                assert f"Action Plugin 2{suffix}" in action_names

            # Signal actions should NOT be here
            assert "Action Plugin 1_signal" not in action_names
            assert "Action Plugin 2_signal" not in action_names

            # Verify actions are registered in both panels (Signal and Image)
            sp_all_actions = []
            # We need to access the private attribute _BaseActionHandler__actions
            # The mangled name depends on the class defining the attribute
            sig_actions = getattr(
                win.signalpanel.acthandler, "_BaseActionHandler__actions"
            ).values()
            for act_list in sig_actions:
                # Handle both QAction and QMenu (QMenu uses .title() instead of .text())
                sp_all_actions.extend(
                    [
                        a.title() if isinstance(a, QW.QMenu) else a.text()
                        for a in act_list
                    ]
                )

            ip_all_actions = []
            img_actions = getattr(
                win.imagepanel.acthandler, "_BaseActionHandler__actions"
            ).values()
            for act_list in img_actions:
                # Handle both QAction and QMenu (QMenu uses .title() instead of .text())
                ip_all_actions.extend(
                    [
                        a.title() if isinstance(a, QW.QMenu) else a.text()
                        for a in act_list
                    ]
                )

            # Verify Signal Panel actions
            assert "Action Plugin 1_common" in sp_all_actions
            assert "Action Plugin 1_signal" in sp_all_actions
            assert "Action Plugin 1_image" not in sp_all_actions  # Independence check

            assert "Action Plugin 2_common" in sp_all_actions
            assert "Action Plugin 2_signal" in sp_all_actions
            assert "Action Plugin 2_image" not in sp_all_actions  # Independence check

            # Verify Image Panel actions
            assert "Action Plugin 1_common" in ip_all_actions
            assert "Action Plugin 1_image" in ip_all_actions
            assert "Action Plugin 1_signal" not in ip_all_actions  # Independence check

            assert "Action Plugin 2_common" in ip_all_actions
            assert "Action Plugin 2_image" in ip_all_actions
            assert "Action Plugin 2_signal" not in ip_all_actions  # Independence check

            # --- Test 3: Remove a plugin and Reload ---
            execenv.print("Removing first plugin and reloading...")

            os.remove(osp.join(plugin_dir, plugin1_filename))

            # Trigger reload
            win.reload_plugins()
            QW.QApplication.processEvents()

            # Verify cleanup
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            execenv.print(f"Registered plugins after removal: {plugin_names}")
            assert "Test Plugin 1" not in plugin_names
            assert "Test Plugin 2" in plugin_names

            # Verify menu cleanup
            win.plugins_menu.aboutToShow.emit()
            actions = win.plugins_menu.actions()
            action_names = [a.text() for a in actions]
            # Ensure no actions from Plugin 1 remain
            assert "Action Plugin 1_common" not in action_names
            assert "Action Plugin 1_signal" not in action_names
            assert "Action Plugin 1_image" not in action_names
            # Ensure Plugin 2 actions are still there
            assert "Action Plugin 2_common" in action_names

            # --- Test 4: Modify a plugin (Rename action + change logic) and Reload ---
            execenv.print("Modifying plugin and reloading...")

            # Overwrite plugin 2 with new action name and behavior.
            # We'll set a flag on the main window to verify the action code results.
            create_plugin_file(
                plugin_dir,
                plugin2_filename,
                "TestPlugin2",
                "Test Plugin 2",
                "Action Plugin 2 Updated",
                "action_plugin_2",
                test_code="self.main._test_plugin_flag = 'updated'",
            )

            # Trigger reload
            win.reload_plugins()
            QW.QApplication.processEvents()

            # Verify update

            # Switch to Signal Panel first for consistent checking
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()

            win.plugins_menu.aboutToShow.emit()
            actions = win.plugins_menu.actions()
            action_names = [a.text() for a in actions]

            # Old actions gone
            assert "Action Plugin 2_common" not in action_names
            # New actions present (Signal context)
            assert "Action Plugin 2 Updated_common" in action_names
            assert "Action Plugin 2 Updated_signal" in action_names
            assert "Action Plugin 2 Updated_image" not in action_names

            assert action_names.count("Action Plugin 2 Updated_common") == 1

            # Trigger the action to verify the new code is running
            # Find the action (using the common one as proxy)
            found_action = False
            for act in actions:
                if act.text() == "Action Plugin 2 Updated_common":
                    act.trigger()
                    found_action = True
                    break
            assert found_action

            # Check if the side effect occurred
            assert getattr(win, "_test_plugin_flag", None) == "updated"

            # --- Test 5: Broken Plugin Handling ---
            execenv.print("Testing broken plugin handling (ImportError)...")

            broken_filename = "datalab_test_plugin_broken.py"
            # Create a file with syntax error or runtime error
            # We use a string here because plugin_error.py is valid python code but
            # fails on __init__. Here we want to test the
            # ImportError/SyntaxError handling
            # in discover_plugins function
            with open(
                osp.join(plugin_dir, broken_filename), "w", encoding="utf-8"
            ) as f:
                f.write("import non_existent_module\nclass BrokenPlugin: pass")

            # --- Test 6: Plugin with Init Error ---
            execenv.print("Testing plugin with __init__ error...")
            init_error_filename = "datalab_test_plugin_init_error.py"
            with open(
                osp.join(plugin_dir, init_error_filename), "w", encoding="utf-8"
            ) as f:
                f.write(
                    "from datalab.plugins import PluginBase, PluginInfo\n\n\n"
                    "class BrokenPlugin(PluginBase):\n"
                    "    PLUGIN_INFO = PluginInfo(\n"
                    '        name="Broken Plugin",\n'
                    '        version="1.0.0",\n'
                    '        description="This plugin raises an error on init",\n'
                    "    )\n\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    '        raise RuntimeError("Planned failure")\n\n'
                    "    def create_actions(self):\n"
                    "        pass\n"
                )

            # Trigger reload
            # Expected behavior:
            # 1. broken_filename fails to import -> Caught by discover_plugins
            # 2. init_error_filename imports ok, but fails __init__ -> Caught by main.py
            #    BUT try_or_log_error re-raises in test mode, so we mock
            #    is_running_tests
            # 3. Valid plugins remain valid
            with patch("datalab.utils.qthelpers.is_running_tests") as mock_run_tests:
                mock_run_tests.return_value = False
                win.reload_plugins()
            QW.QApplication.processEvents()

            # Verify app is still alive and healthy plugin is still there
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            execenv.print(f"Plugins after broken tests: {plugin_names}")
            assert "Test Plugin 2" in plugin_names
            assert "Broken Plugin" not in plugin_names

            # Verify that the healthy plugin is still functional
            # Reset flag
            setattr(win, "_test_plugin_flag", None)

            win.plugins_menu.aboutToShow.emit()
            actions = win.plugins_menu.actions()
            found_action = False
            for act in actions:
                if act.text() == "Action Plugin 2 Updated_common":
                    act.trigger()
                    found_action = True
                    break
            assert found_action
            assert getattr(win, "_test_plugin_flag", None) == "updated"

            # Cleanup broken plugins for cleanliness
            os.remove(osp.join(plugin_dir, broken_filename))
            os.remove(osp.join(plugin_dir, init_error_filename))


def test_plugin_config_disabled():
    """Test that reload_plugins() shows info dialog when plugins are disabled"""
    with temporary_plugin_dir():
        with datalab_test_app_context(console=False) as win:
            win: DLMainWindow

            # Mock the config to disable plugins
            with patch("datalab.config.Conf.main.plugins_enabled.get") as mock_enabled:
                mock_enabled.return_value = False

                # Mock QMessageBox to avoid blocking dialog
                with patch("datalab.gui.main.QW.QMessageBox.information") as mock_info:
                    win.reload_plugins()

                    # Verify info dialog was shown
                    assert mock_info.called
                    args = mock_info.call_args
                    assert (
                        "disabled" in args[0][2].lower()
                        or "désactivés" in args[0][2].lower()
                    )


def test_plugin_error_handling():
    """Test that various malformed plugins are handled gracefully.

    Verifies that the application survives and valid plugins continue to work
    when encountering plugins with:
    - PLUGIN_INFO set to None
    - Missing create_actions method (abstract)
    - Syntax errors in the source file
    """
    Conf.main.plugins_enabled_list.set(None)

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # Create a valid plugin to verify it survives alongside bad ones
        create_plugin_file(
            plugin_dir,
            "datalab_test_plugin_good.py",
            "TestPluginGood",
            "Valid Plugin",
            "Action Valid",
            "action_valid",
        )

        # Plugin with PLUGIN_INFO = None
        with open(
            osp.join(plugin_dir, "datalab_test_plugin_invalid_info.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "from datalab.plugins import PluginBase\n\n"
                "class InvalidInfoPlugin(PluginBase):\n"
                "    PLUGIN_INFO = None\n"
                "    def create_actions(self):\n"
                "        pass\n"
            )

        # Plugin without create_actions method (abstract)
        with open(
            osp.join(plugin_dir, "datalab_test_plugin_no_actions.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "from datalab.plugins import PluginBase, PluginInfo\n\n\n"
                "class NoCreateActionsPlugin(PluginBase):\n"
                "    PLUGIN_INFO = PluginInfo(\n"
                '        name="No Create Actions Plugin",\n'
                '        version="1.0.0",\n'
                '        description="Plugin without create_actions method",\n'
                "    )\n\n"
                "    # Missing create_actions() method - should raise error\n"
            )

        # Plugin with syntax error
        with open(
            osp.join(plugin_dir, "datalab_test_plugin_syntax_error.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "from datalab.plugins import PluginBase, PluginInfo\n\n"
                "class SyntaxErrorPlugin(PluginBase):\n"
                "    PLUGIN_INFO = PluginInfo(\n"
                '        name="Syntax Error Plugin",\n'
                '        version="1.0.0"\n'
                '        description="Missing comma"\n'
                "    )\n\n"
                "    def create_actions(self):\n"
                "        pass\n"
            )

        with patch("datalab.utils.qthelpers.is_running_tests") as mock_run_tests:
            mock_run_tests.return_value = False
            with datalab_test_app_context(console=False):
                QW.QApplication.processEvents()

                plugins = PluginRegistry.get_plugins()
                plugin_names = [p.info.name for p in plugins]
                plugin_classes = [p.__class__.__name__ for p in plugins]
                execenv.print(f"Registered plugins: {plugin_names}")

                # Valid plugin must survive
                assert "Valid Plugin" in plugin_names

                # Bad plugins must NOT be loaded
                assert "InvalidInfoPlugin" not in plugin_classes
                assert "No Create Actions Plugin" not in plugin_names
                assert "Syntax Error Plugin" not in plugin_names


def test_plugin_duplicate_name():
    """Test that duplicate plugin names are detected and handled"""
    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # Create first plugin
        create_plugin_file(
            plugin_dir,
            "datalab_test_plugin_dup1.py",
            "TestPluginDup1",
            "Duplicate Name Plugin",
            "Action Dup 1",
            "action_dup_1",
        )

        # Create second plugin with SAME NAME
        create_plugin_file(
            plugin_dir,
            "datalab_test_plugin_dup2.py",
            "TestPluginDup2",
            "Duplicate Name Plugin",  # Same name
            "Action Dup 2",
            "action_dup_2",
        )

        # Start application - should handle duplicate gracefully
        with patch("datalab.utils.qthelpers.is_running_tests") as mock_run_tests:
            mock_run_tests.return_value = False
            with datalab_test_app_context(console=False):
                QW.QApplication.processEvents()

                # Verify app is still alive
                plugins = PluginRegistry.get_plugins()
                plugin_names = [p.info.name for p in plugins]
                execenv.print(f"Registered plugins: {plugin_names}")

                # Count how many times the duplicate name appears
                duplicate_count = plugin_names.count("Duplicate Name Plugin")
                execenv.print(f"Duplicate name count: {duplicate_count}")

                # Should be 1 or 0 (second should fail to register)
                assert duplicate_count <= 1, (
                    f"Duplicate plugin name should be rejected, found {duplicate_count}"
                )


def test_plugin_nested_menus():
    """Test plugin with nested submenus (3 levels deep)"""
    # Ensure plugins_enabled_list is None (all plugins enabled)
    Conf.main.plugins_enabled_list.set(None)

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # Create plugin with nested menus (using plugin_nested_menus.py template)
        create_plugin_from_template(
            plugin_dir,
            "datalab_test_plugin_nested.py",
            "plugin_nested_menus.py.template",
            {
                "{class_name}": "TestPluginNested",
                "{plugin_name}": "Nested Menus Plugin",
                "{menu_level_1}": "Level 1 Menu",
                "{action_level_1}": "Action Level 1",
                "{test_code_1}": "self.main._test_level_1 = True",
                "{menu_level_2}": "Level 2 Submenu",
                "{action_level_2}": "Action Level 2",
                "{test_code_2}": "self.main._test_level_2 = True",
                "{menu_level_3}": "Level 3 Submenu",
                "{action_level_3}": "Action Level 3",
                "{test_code_3}": "self.main._test_level_3 = True",
            },
        )

        with datalab_test_app_context(console=False) as win:
            QW.QApplication.processEvents()

            # Verify plugin loaded
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            assert "Nested Menus Plugin" in plugin_names

            # Switch to signal panel
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()

            # Get plugin category actions from signal panel
            plugin_actions = win.signalpanel.get_category_actions(
                ActionCategory.PLUGINS
            )

            # Find the Level 1 menu in plugin actions
            level_1_menu = None
            for item in plugin_actions:
                if isinstance(item, QW.QMenu) and item.title() == "Level 1 Menu":
                    level_1_menu = item
                    break

            assert level_1_menu is not None, "Level 1 menu not found in plugin actions"

            # Qt menus populate their actions lazily, trigger aboutToShow
            level_1_menu.aboutToShow.emit()

            # Find Level 2 submenu inside Level 1
            # In Qt, submenus are QActions with menu() != None
            level_2_menu = None
            for act in level_1_menu.actions():
                submenu = act.menu() if isinstance(act, QW.QAction) else None
                if submenu and submenu.title() == "Level 2 Submenu":
                    level_2_menu = submenu
                    break

            assert level_2_menu is not None, "Level 2 submenu not found"

            # Trigger aboutToShow for Level 2
            level_2_menu.aboutToShow.emit()

            # Find Level 3 submenu inside Level 2
            level_3_menu = None
            for act in level_2_menu.actions():
                submenu = act.menu() if isinstance(act, QW.QAction) else None
                if submenu and submenu.title() == "Level 3 Submenu":
                    level_3_menu = submenu
                    break

            assert level_3_menu is not None, "Level 3 submenu not found"

            # Trigger aboutToShow for Level 3
            level_3_menu.aboutToShow.emit()

            # Find and trigger action at level 3
            action_level_3 = None
            for act in level_3_menu.actions():
                if isinstance(act, QW.QAction) and act.text() == "Action Level 3":
                    action_level_3 = act
                    break

            assert action_level_3 is not None, "Level 3 action not found"

            # Trigger the action and verify it runs
            action_level_3.trigger()
            assert getattr(win, "_test_level_3", None) is True


def test_plugin_with_dialogs():
    """Test plugin using dialog methods (show_warning, show_info, etc.)"""
    # Ensure plugins_enabled_list is None (all plugins enabled)
    Conf.main.plugins_enabled_list.set(None)

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # Create plugin with dialog tests (using plugin_with_dialogs.py template)
        create_plugin_from_template(
            plugin_dir,
            "datalab_test_plugin_dialogs.py",
            "plugin_with_dialogs.py.template",
            {
                "{class_name}": "TestPluginDialogs",
                "{plugin_name}": "Dialogs Plugin",
                "{menu_name}": "Test Dialogs",
                "{test_code}": "self.main._test_dialog_flag = True",
            },
        )

        with datalab_test_app_context(console=False):
            QW.QApplication.processEvents()

            # Verify plugin loaded
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            assert "Dialogs Plugin" in plugin_names

            # Get the plugin instance
            plugin = PluginRegistry.get_plugin("Dialogs Plugin")
            assert plugin is not None

            # Mock dialog methods to avoid blocking
            with (
                patch("datalab.plugins.QW.QMessageBox.warning") as mock_warning,
                patch("datalab.plugins.QW.QMessageBox.critical") as mock_error,
                patch("datalab.plugins.QW.QMessageBox.information") as mock_info,
                patch("datalab.plugins.QW.QMessageBox.question") as mock_question,
            ):
                mock_question.return_value = QW.QMessageBox.Yes

                # Test show_warning
                plugin.show_warning("Test warning")
                assert mock_warning.called

                # Test show_error
                plugin.show_error("Test error")
                assert mock_error.called

                # Test show_info
                plugin.show_info("Test info")
                assert mock_info.called

                # Test ask_yesno
                result = plugin.ask_yesno("Test question?")
                assert mock_question.called
                assert result is True


if __name__ == "__main__":
    _launch_path = osp.join(osp.dirname(__file__), "launch_with_test_plugins.py")
    _spec = importlib.util.spec_from_file_location(
        "launch_with_test_plugins", _launch_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.main()
