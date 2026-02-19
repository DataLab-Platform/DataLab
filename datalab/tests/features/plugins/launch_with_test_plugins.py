# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Launch DataLab with test plugins for visual inspection
------------------------------------------------------

This script creates temporary test plugins and launches DataLab so you can:

1. Verify that the plugin menu shows all plugins correctly
2. Test the scrollbar in the plugin configuration dialog (many plugins)
3. Test the "Show more" button for long descriptions
4. Test nested menus and multiple actions in dropdown menus
5. Verify disabled/enabled plugin appearance

Usage::

    python scripts/run_with_env.py python datalab/tests/features/plugins/launch_with_test_plugins.py

Close the DataLab window to end the script.
"""

# guitest: show

from __future__ import annotations

import os.path as osp
import shutil
import sys
import tempfile

# Ensure project root is on path
_project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Path to plugin templates
DATA_TESTS_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), "../../../data/tests/plugin")
)


def get_plugin_template(filename: str) -> str:
    """Read a plugin template file.

    Args:
        filename: Template filename in the plugin templates directory

    Returns:
        Template content as string
    """
    with open(osp.join(DATA_TESTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def create_plugin_from_template(
    plugin_dir: str,
    output_filename: str,
    template_name: str,
    replacements: dict[str, str],
) -> str:
    """Create a plugin file from a template with placeholder replacements.

    Args:
        plugin_dir: Directory to write the plugin file
        output_filename: Output filename (must start with ``datalab_``)
        template_name: Template filename in the plugin templates directory
        replacements: Dict of ``{placeholder: value}`` to substitute

    Returns:
        Full path to the created file
    """
    content = get_plugin_template(template_name)
    for key, value in replacements.items():
        content = content.replace(key, value)

    filepath = osp.join(plugin_dir, output_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def generate_bulk_plugins(plugin_dir: str, count: int = 20) -> list[str]:
    """Generate many simple plugins for scrollbar testing.

    All generated files follow the ``datalab_plugin_*`` naming convention
    so that :func:`discover_plugins` picks them up.

    Args:
        plugin_dir: Directory to write the plugin files
        count: Number of plugins to generate

    Returns:
        List of created file paths
    """
    created: list[str] = []
    for i in range(1, count + 1):
        long_desc = i % 4 == 0  # Every 4th plugin gets a long description

        description = f"Auto-generated test plugin #{i} for scrollbar testing."
        if long_desc:
            description = (
                f"Auto-generated test plugin #{i}. "
                + "This plugin has a very long description to test the "
                "'Show more' button behaviour. "
                * 6
                + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            )

        plugin_code = f'''\
"""Auto-generated test plugin {i}"""
from datalab.plugins import PluginBase, PluginInfo


class AutoTestPlugin{i}(PluginBase):
    """Auto-generated test plugin {i}."""

    PLUGIN_INFO = PluginInfo(
        name="Auto Test Plugin {i}",
        version="1.0.{i}",
        description="""{description}""",
    )

    def create_actions(self):
        acth = self.signalpanel.acthandler
        acth.new_action(
            "Auto Plugin {i} Action",
            triggered=lambda: print("Auto plugin {i} action triggered"),
            select_condition="always",
        )
'''
        filename = f"datalab_plugin_auto_{i:02d}.py"
        filepath = osp.join(plugin_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(plugin_code)
        created.append(filepath)

    return created


def main():
    """Create test plugins and launch DataLab for visual inspection."""
    from datalab.config import Conf

    tmpdir = tempfile.mkdtemp(prefix="datalab_visual_test_plugins_")
    print(f"Temporary plugin directory: {tmpdir}")
    sys.path.insert(0, tmpdir)

    try:
        # ------------------------------------------------------------------
        # 1) Plugin with many actions in dropdown menu
        # ------------------------------------------------------------------
        create_plugin_from_template(
            tmpdir,
            "datalab_plugin_many_actions_visual.py",
            "plugin_many_actions.py",
            {
                "{class_name}": "VisualTestManyActions",
                "{plugin_name}": "Visual Test: Many Actions",
                "{menu_name}": "📋 Test Menu with Many Actions",
                "{action_prefix}": "Test Action",
                "{test_code_1}": "print('Action 1 triggered')",
                "{test_code_2}": "print('Action 2 triggered')",
                "{test_code_3}": "print('Action 3 triggered')",
                "{test_code_4}": "print('Action 4 triggered')",
                "{test_code_5}": "print('Action 5 triggered')",
            },
        )

        # ------------------------------------------------------------------
        # 2) Plugin with long description (test "Show more" button)
        # ------------------------------------------------------------------
        long_desc = (
            "This is an EXTREMELY LONG DESCRIPTION to test how DataLab handles "
            "verbose plugin descriptions. "
            * 10
            + "The description should be displayed properly in tooltips, status "
            "bars, and configuration dialogs without breaking the UI layout. " * 5
        )
        create_plugin_from_template(
            tmpdir,
            "datalab_plugin_long_description_visual.py",
            "plugin_long_description.py",
            {
                "{class_name}": "VisualTestLongDescription",
                "{plugin_name}": "Visual Test: Long Description",
                "{long_description}": long_desc,
                "{action_name}": "📄 Test Long Description",
                "{test_code}": "print('Long description plugin action triggered')",
            },
        )

        # ------------------------------------------------------------------
        # 3) Plugin with nested menus
        # ------------------------------------------------------------------
        create_plugin_from_template(
            tmpdir,
            "datalab_plugin_nested_menus_visual.py",
            "plugin_nested_menus.py",
            {
                "{class_name}": "VisualTestNestedMenus",
                "{plugin_name}": "Visual Test: Nested Menus",
                "{menu_level_1}": "📁 Level 1 Menu",
                "{action_level_1}": "Action at Level 1",
                "{test_code_1}": "print('Level 1 action triggered')",
                "{menu_level_2}": "📁 Level 2 Submenu",
                "{action_level_2}": "Action at Level 2",
                "{test_code_2}": "print('Level 2 action triggered')",
                "{menu_level_3}": "📁 Level 3 Submenu",
                "{action_level_3}": "Action at Level 3",
                "{test_code_3}": "print('Level 3 action triggered')",
            },
        )

        # ------------------------------------------------------------------
        # 4) Many auto-generated plugins for scrollbar testing
        # ------------------------------------------------------------------
        bulk = generate_bulk_plugins(tmpdir, count=20)
        print(f"Generated {len(bulk)} bulk plugins for scrollbar testing")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("VISUAL TEST SETUP COMPLETE")
        print("=" * 70)
        print("\nCreated plugins:")
        print("  1. 📋 Many Actions       – dropdown menu with 5 actions")
        print("  2. 📄 Long Description   – test 'Show more' button")
        print("  3. 📁 Nested Menus       – 3-level nested submenus")
        print("  4. 🔢 20 Auto Plugins    – scrollbar in Settings > Plugins")
        print("\nManual verification checklist:")
        print("  ✓ Open Plugins menu → see all plugin actions")
        print("  ✓ Open Settings > Plugins → verify scrollbar with 23 plugins")
        print("  ✓ Click 'Show more' on long description plugin")
        print("  ✓ Toggle enable/disable checkboxes")
        print("\nLaunching DataLab...")
        print("=" * 70 + "\n")

        # Configure: enable all plugins, add temp dir to path
        Conf.main.plugins_enabled_list.set(None)
        original_path = Conf.main.plugins_path.get()
        if original_path:
            Conf.main.plugins_path.set(original_path + ";" + tmpdir)
        else:
            Conf.main.plugins_path.set(tmpdir)

        # Launch DataLab
        from datalab.app import run

        run()

    finally:
        # Restore config and cleanup
        print(f"\nCleaning up: {tmpdir}")
        if tmpdir in sys.path:
            sys.path.remove(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
