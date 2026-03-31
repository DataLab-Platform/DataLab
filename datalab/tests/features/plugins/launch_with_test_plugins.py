# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Launch DataLab with temporary plugin test dataset
------------------------------------------------

This script creates temporary test plugins and launches DataLab so you can:

1. Verify that the plugin menu shows all plugins correctly
2. Test the scrollbar in the plugin configuration dialog (many plugins)
3. Test the "Show more" button for long descriptions
4. Verify that description truncation reacts to dialog width changes
5. Test nested menus and multiple actions in dropdown menus
6. Verify disabled/enabled plugin appearance

Usage::

    python scripts/run_with_env.py
    python datalab/tests/features/plugins/launch_with_test_plugins.py

Close the DataLab window to end the script.
"""

# guitest: show

from __future__ import annotations

import os.path as osp
import sys
from importlib import import_module

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PLUGIN_TEST_DATASET = import_module(
    "datalab.tests.features.plugins.plugin_test_dataset"
)
MANUAL_PLUGIN_MODULE_PREFIXES = PLUGIN_TEST_DATASET.MANUAL_PLUGIN_MODULE_PREFIXES
MANUAL_TEST_PLUGINS_DIR = PLUGIN_TEST_DATASET.MANUAL_TEST_PLUGINS_DIR
clear_plugin_directory = PLUGIN_TEST_DATASET.clear_plugin_directory
create_manual_test_plugin_dataset = (
    PLUGIN_TEST_DATASET.create_manual_test_plugin_dataset
)
create_plugin_directory = PLUGIN_TEST_DATASET.create_plugin_directory
ensure_plugin_dir_on_syspath = PLUGIN_TEST_DATASET.ensure_plugin_dir_on_syspath


def _get_enabled_plugins_option(conf_class):
    """Return the optional enabled-plugins config entry when available."""
    return getattr(conf_class.main, "plugins_enabled_list", None)


def main():
    """Create manual test plugins, launch DataLab, then clean them up."""
    conf_class = import_module("datalab.config").Conf
    enabled_plugins_option = _get_enabled_plugins_option(conf_class)

    # Save the original plugins_path before any modification
    original_path = conf_class.main.plugins_path.get()
    original_enabled_list = None
    if enabled_plugins_option is not None:
        original_enabled_list = enabled_plugins_option.get(None)

    plugin_dir = create_plugin_directory(MANUAL_TEST_PLUGINS_DIR)
    ensure_plugin_dir_on_syspath(plugin_dir)
    print(f"Manual test plugin directory: {plugin_dir}")
    created = create_manual_test_plugin_dataset(plugin_dir, bulk_count=40)
    bulk_count = len(
        [
            path
            for path in created
            if osp.basename(path).startswith("datalab_plugin_auto_")
        ]
    )
    print(f"Generated {bulk_count} bulk plugins for scrollbar testing")

    try:
        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PLUGIN TEST SETUP COMPLETE")
        print("=" * 70)
        print("\nCreated plugins:")
        print("  1.  Many Actions       – dropdown menu with 5 actions")
        print("  2.  Long Description   – test 'Show more' and 'Show less'")
        print("  3.  Responsive Desc.   – resize Plugin Configuration dialog")
        print("  4.  Nested Menus       – 3-level nested submenus")
        print(f"  5.  {bulk_count} Auto Plugins    – scrollbar in Settings > Plugins")
        print("\nManual verification checklist:")
        print("  ✓ Open Plugins menu → see all plugin actions")
        print("  ✓ Open Settings > Plugins → verify scrollbar in plugin list")
        print("  ✓ Click 'Show more' on long description plugin")
        print("  ✓ Click 'Show less' → text is truncated again without scrollbar")
        print("  ✓ Resize Plugin Configuration → responsive description updates")
        print("  ✓ Toggle enable/disable checkboxes")
        print("\nLaunching DataLab...")
        print("=" * 70 + "\n")

        # Configure: enable all plugins, point plugins_path to the managed
        # dataset under datalab/data/tests. The original path (if any) is
        # restored afterwards.
        if enabled_plugins_option is not None:
            enabled_plugins_option.set(None)
        conf_class.main.plugins_path.set(plugin_dir)

        # Launch DataLab
        run = import_module("datalab.app").run

        run()

    finally:
        if enabled_plugins_option is not None:
            enabled_plugins_option.set(original_enabled_list)
        conf_class.main.plugins_path.set(original_path)
        clear_plugin_directory(
            plugin_dir,
            module_prefixes=MANUAL_PLUGIN_MODULE_PREFIXES,
        )
        print(f"\nCleaning up: {plugin_dir}")


if __name__ == "__main__":
    main()
