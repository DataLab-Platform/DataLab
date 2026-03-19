# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plugin system tests
-------------------

Testing the plugin system (discovery, loading, reloading, cleanup) using
the main window.
"""

# guitest: show

# ruff: noqa: E402
# pylint: disable=import-outside-toplevel

import contextlib
import importlib
import importlib.util
import os
import os.path as osp
import shutil
import sys
from unittest.mock import patch

# Ensure project root is on path (standalone execution)
_project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.plugins import PluginRegistry
from datalab.tests import datalab_test_app_context

# Path to plugin templates
TEMPLATES_DIR = osp.abspath(osp.join(osp.dirname(__file__), "templates"))

# Fixed directory for temporary test plugin files (inside the project tree)
_TEST_PLUGINS_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), "..", "..", "..", "data", "tests", "plugins")
)


def get_plugin_code(filename):
    """Read a plugin template file from TEMPLATES_DIR."""
    with open(osp.join(TEMPLATES_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


@contextlib.contextmanager
def temporary_plugin_dir():
    """Create a fixed directory for test plugins and add it to sys.path.

    The directory ``datalab/data/tests/plugins/`` is used so that the path
    is deterministic and stays within the project tree (cross-platform).

    On exit, removes the directory from sys.path, purges any
    ``datalab_test_plugin_*`` modules from ``sys.modules`` so that the
    next test starts with a clean import state, and deletes the directory
    contents.  This prevents stale module objects (which hold references
    to destroyed Qt widgets) from surviving across tests – a common cause
    of ACCESS_VIOLATION crashes under coverage.
    """
    if osp.exists(_TEST_PLUGINS_DIR):
        shutil.rmtree(_TEST_PLUGINS_DIR)
    os.makedirs(_TEST_PLUGINS_DIR)

    # Add to sys.path and invalidate import caches so that Python's
    # path finders pick up the freshly (re)created directory.
    sys.path.insert(0, _TEST_PLUGINS_DIR)
    sys.path_importer_cache.pop(_TEST_PLUGINS_DIR, None)
    importlib.invalidate_caches()

    try:
        yield _TEST_PLUGINS_DIR
    finally:
        # Purge cached test-plugin modules so they cannot leak Qt references
        stale_modules = [
            name for name in sys.modules if name.startswith("datalab_test_plugin")
        ]
        for name in stale_modules:
            del sys.modules[name]

        # Remove from sys.path and clean up finder cache
        if _TEST_PLUGINS_DIR in sys.path:
            sys.path.remove(_TEST_PLUGINS_DIR)
        sys.path_importer_cache.pop(_TEST_PLUGINS_DIR, None)
        importlib.invalidate_caches()
        shutil.rmtree(_TEST_PLUGINS_DIR, ignore_errors=True)


def create_plugin_file(
    directory,
    filename,
    class_name,
    plugin_name,
    action_name,
    action_obj_name,
    test_code="pass",
):
    """Create a plugin file in the specified directory"""
    template = get_plugin_code("plugin_valid.py.template")
    # Replace placeholders manually since .format() might fail if the template contains
    # other braces
    content = template.replace("{class_name}", class_name)
    content = content.replace("{plugin_name}", plugin_name)
    content = content.replace("{action_name}", action_name)
    content = content.replace("{action_object_name}", action_obj_name)
    content = content.replace("{test_code}", test_code)

    with open(osp.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)


def create_plugin_from_template(directory, filename, template_name, replacements):
    """Create a plugin file from any template with placeholder replacements.

    Args:
        directory: Target directory
        filename: Output filename
        template_name: Template file in TEMPLATES_DIR
         (e.g. "plugin_nested_menus.py.template")
        replacements: Dict of {placeholder: value} to substitute
    """
    content = get_plugin_code(template_name)
    for key, value in replacements.items():
        content = content.replace(key, value)
    with open(osp.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)


def test_plugin_system():
    """Test the entire plugin lifecycle: discovery, reload, cleanup"""
    from datalab.config import Conf

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
            # We access private member for testing purposes
            # pylint: disable=protected-access
            sp_all_actions = []
            # We need to access the private attribute _BaseActionHandler__actions
            # The mangled name depends on the class defining the attribute
            sig_actions = (  # type: ignore
                win.signalpanel.acthandler._BaseActionHandler__actions.values()
            )
            for act_list in sig_actions:
                # Handle both QAction and QMenu (QMenu uses .title() instead of .text())
                sp_all_actions.extend(
                    [
                        a.title() if isinstance(a, QW.QMenu) else a.text()
                        for a in act_list
                    ]
                )

            ip_all_actions = []
            img_actions = (  # type: ignore
                win.imagepanel.acthandler._BaseActionHandler__actions.values()
            )
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
            win._test_plugin_flag = None  # pylint: disable=protected-access

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
    from datalab.config import Conf

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
    from datalab.config import Conf

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
            from datalab.gui.actionhandler import ActionCategory

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
    from datalab.config import Conf

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


def test_plugin_enable_disable_config():
    """Test plugin enable/disable filtering and configuration dialog.

    Verifies that:
    - plugins_enabled_list correctly filters which plugins are loaded
    - Disabled plugins remain visible in the configuration dialog
    - Plugin checkboxes reflect the enabled/disabled state
    - Re-enabling all plugins restores them
    """
    from datalab.config import Conf
    from datalab.gui.pluginconfig import PluginConfigDialog

    # Save original config
    try:
        original_enabled_list = Conf.main.plugins_enabled_list.get()
        had_config = True
    except Exception:  # noqa: BLE001
        original_enabled_list = None
        had_config = False

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

            with patch("datalab.utils.qthelpers.is_running_tests") as mock:
                mock.return_value = False
                with datalab_test_app_context(console=False) as win:
                    QW.QApplication.processEvents()

                    # --- All enabled ---
                    plugins = PluginRegistry.get_plugins()
                    plugin_names = [p.info.name for p in plugins]
                    execenv.print(f"Registered plugins (all enabled): {plugin_names}")
                    assert "Test Plugin 1" in plugin_names
                    assert "Test Plugin 2" in plugin_names

                    # Config dialog shows both
                    dialog = PluginConfigDialog(win)
                    widget_names = [
                        w.plugin_class.PLUGIN_INFO.name for w in dialog.plugin_widgets
                    ]
                    assert "Test Plugin 1" in widget_names
                    assert "Test Plugin 2" in widget_names
                    assert dialog.toggle_all_checkbox.checkState() == QC.Qt.Checked

                    dialog.filter_combo.setCurrentIndex(2)
                    QW.QApplication.processEvents()
                    visible_disabled = [
                        w.plugin_class.PLUGIN_INFO.name
                        for w in dialog.plugin_widgets
                        if w.isVisible()
                    ]
                    assert visible_disabled == []

                    dialog.filter_combo.setCurrentIndex(0)
                    QW.QApplication.processEvents()
                    dialog.close()
                    dialog.deleteLater()
                    QW.QApplication.processEvents()

                    # --- Disable Plugin 2 ---
                    Conf.main.plugins_enabled_list.set(["Test Plugin 1"])
                    win.reload_plugins()
                    QW.QApplication.processEvents()

                    plugins = PluginRegistry.get_plugins()
                    plugin_names = [p.info.name for p in plugins]
                    execenv.print(f"Registered plugins (only Plugin 1): {plugin_names}")
                    assert "Test Plugin 1" in plugin_names
                    assert "Test Plugin 2" not in plugin_names

                    # Config dialog still shows both, Plugin 2 unchecked
                    dialog2 = PluginConfigDialog(win)
                    widget_names = [
                        w.plugin_class.PLUGIN_INFO.name for w in dialog2.plugin_widgets
                    ]
                    assert "Test Plugin 1" in widget_names
                    assert "Test Plugin 2" in widget_names, (
                        "Disabled plugin should still be visible in config dialog"
                    )
                    for widget in dialog2.plugin_widgets:
                        name = widget.plugin_class.PLUGIN_INFO.name
                        if name == "Test Plugin 1":
                            assert widget.checkbox.isChecked(), (
                                "Plugin 1 should be checked (enabled)"
                            )
                        elif name == "Test Plugin 2":
                            assert not widget.checkbox.isChecked(), (
                                "Plugin 2 should be unchecked (disabled)"
                            )

                    assert dialog2.toggle_all_checkbox.checkState() == (
                        QC.Qt.PartiallyChecked
                    )

                    dialog2.filter_combo.setCurrentIndex(1)
                    QW.QApplication.processEvents()
                    visible_enabled = [
                        w.plugin_class.PLUGIN_INFO.name
                        for w in dialog2.plugin_widgets
                        if w.isVisible()
                    ]
                    assert visible_enabled == ["Test Plugin 1"]

                    dialog2.filter_combo.setCurrentIndex(2)
                    QW.QApplication.processEvents()
                    visible_disabled = [
                        w.plugin_class.PLUGIN_INFO.name
                        for w in dialog2.plugin_widgets
                        if w.isVisible()
                    ]
                    assert visible_disabled == ["Test Plugin 2"]

                    dialog2.toggle_all_checkbox.setChecked(True)
                    QW.QApplication.processEvents()
                    assert all(
                        widget.checkbox.isChecked() for widget in dialog2.plugin_widgets
                    )
                    assert dialog2.toggle_all_checkbox.checkState() == QC.Qt.Checked

                    dialog2.close()
                    dialog2.deleteLater()
                    QW.QApplication.processEvents()

                    # --- Re-enable all ---
                    Conf.main.plugins_enabled_list.set(None)
                    win.reload_plugins()
                    QW.QApplication.processEvents()

                    plugins = PluginRegistry.get_plugins()
                    plugin_names = [p.info.name for p in plugins]
                    execenv.print(
                        f"Registered plugins (all re-enabled): {plugin_names}"
                    )
                    assert "Test Plugin 1" in plugin_names
                    assert "Test Plugin 2" in plugin_names
    finally:
        if had_config:
            Conf.main.plugins_enabled_list.set(original_enabled_list)
        else:
            try:
                Conf.remove_option(  # pylint: disable=no-member
                    "main", "plugins_enabled_list"
                )
            except Exception:  # noqa: BLE001
                pass


def test_plugin_many_actions_visual():
    """Test plugin with many actions in dropdown menu - Visual test for user observation

    This test launches DataLab with a plugin containing multiple actions to verify
    the dropdown menu behavior. The test will pause to allow user observation.
    """
    # guitest: show
    from datalab.config import Conf

    # Ensure plugins_enabled_list is None (all plugins enabled)
    Conf.main.plugins_enabled_list.set(None)

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        # Create plugin with many actions (using plugin_many_actions.py template)
        create_plugin_from_template(
            plugin_dir,
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
        )

        with datalab_test_app_context(console=False) as win:
            QW.QApplication.processEvents()

            # Verify plugin loaded
            plugins = PluginRegistry.get_plugins()
            plugin_names = [p.info.name for p in plugins]
            execenv.print(f"Loaded plugins: {plugin_names}")
            assert "Many Actions Test" in plugin_names

            # Switch to signal panel
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()

            # Open plugins menu to show dropdown
            win.plugins_menu.aboutToShow.emit()

            # Find the submenu with multiple actions
            from datalab.gui.actionhandler import ActionCategory

            plugin_actions = win.signalpanel.get_category_actions(
                ActionCategory.PLUGINS
            )

            # Find the menu
            test_menu = None
            for item in plugin_actions:
                if (
                    isinstance(item, QW.QMenu)
                    and item.title() == "Test Menu with Many Actions"
                ):
                    test_menu = item
                    break

            assert test_menu is not None, "Test menu not found"

            # Trigger aboutToShow to populate the menu
            test_menu.aboutToShow.emit()

            # Verify all 5 actions are present
            menu_actions = test_menu.actions()
            action_texts = [a.text() for a in menu_actions if not a.isSeparator()]
            execenv.print(f"Menu actions: {action_texts}")

            assert len(action_texts) == 5
            for i in range(1, 6):
                assert f"Test Action {i}" in action_texts

            # Test triggering one action
            for act in menu_actions:
                if act.text() == "Test Action 3":
                    act.trigger()
                    break

            assert getattr(win, "_test_action_3", None) is True

            execenv.print(
                "Visual test passed - Multiple actions displayed correctly in dropdown"
            )


def test_plugin_long_description():
    """Test plugin with very long description and "Show more" button.

    Verifies that:
    - Plugins with long descriptions load correctly and actions work
    - The show_full_description widget method works without AttributeError
      (regression test for self.plugin → self.plugin_class fix)
    """
    from datalab.config import Conf
    from datalab.gui.pluginconfig import PluginInfoWidget, PluginState

    Conf.main.plugins_enabled_list.set(None)

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

    with temporary_plugin_dir() as plugin_dir:
        execenv.print(f"Using temporary plugin directory: {plugin_dir}")

        create_plugin_from_template(
            plugin_dir,
            "datalab_test_plugin_long_desc.py",
            "plugin_long_description.py.template",
            {
                "{class_name}": "TestPluginLongDescription",
                "{plugin_name}": "Long Description Test",
                "{long_description}": long_description,
                "{action_name}": "Test Action",
                "{test_code}": "self.main._test_long_desc = True",
            },
        )

        with datalab_test_app_context(console=False) as win:
            QW.QApplication.processEvents()

            # Verify plugin loaded with long description
            plugins = PluginRegistry.get_plugins()
            plugin = None
            plugin_class = None
            for p in plugins:
                if p.info.name == "Long Description Test":
                    plugin = p
                    plugin_class = type(p)
                    break

            assert plugin is not None, "Long Description Test plugin not loaded"
            assert len(plugin.info.description) > 500, "Description not long enough"
            execenv.print(
                f"Plugin description length: {len(plugin.info.description)} chars"
            )

            # Verify the action works
            win.tabwidget.setCurrentWidget(win.signalpanel)
            QW.QApplication.processEvents()

            win.plugins_menu.aboutToShow.emit()
            for act in win.plugins_menu.actions():
                if act.text() == "Test Action":
                    act.trigger()
                    break
            assert getattr(win, "_test_long_desc", None) is True

            # Test _toggle_description doesn't raise AttributeError
            # (regression test for expandable description widget)
            widget = PluginInfoWidget(
                plugin_class, enabled=True, state=PluginState.ENABLED
            )
            try:
                # Long description should have the toggle button
                assert hasattr(widget, "_toggle_btn"), (
                    "Long description should have a toggle button"
                )
                assert hasattr(widget, "_desc_scroll"), (
                    "Long description should have a scroll area"
                )
                # Toggle expand
                # pylint: disable=protected-access
                widget._toggle_description()
                assert widget._expanded is True
                assert widget._desc_scroll.maximumHeight() == 150
                # Toggle collapse
                widget._toggle_description()
                assert widget._expanded is False
                assert widget._desc_scroll.maximumHeight() == 60
                # pylint: enable=protected-access
            except AttributeError as e:
                raise AssertionError(
                    f"_toggle_description raised AttributeError: {e}"
                ) from e
            del widget

            execenv.print("Long description plugin handled correctly")


if __name__ == "__main__":
    _launch_path = osp.join(osp.dirname(__file__), "launch_with_test_plugins.py")
    _spec = importlib.util.spec_from_file_location(
        "launch_with_test_plugins", _launch_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.main()
