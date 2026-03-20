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

    python scripts/run_with_env.py
    python datalab/tests/features/plugins/launch_with_test_plugins.py

Close the DataLab window to end the script.
"""

# guitest: show

from __future__ import annotations

import os
import os.path as osp
import sys
from importlib import import_module

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datalab.tests.features.plugins.plugin_test_dataset import (
    VISUAL_PLUGIN_MODULE_PREFIXES,
    VISUAL_PLUGINS_DIR,
    create_visual_test_dataset,
    temporary_plugin_dir,
)


def _get_enabled_plugins_option(conf_class):
    """Return the optional enabled-plugins config entry when available."""
    return getattr(conf_class.main, "plugins_enabled_list", None)


def main():
    """Create test plugins and launch DataLab for visual inspection."""
    conf_class = import_module("datalab.config").Conf
    enabled_plugins_option = _get_enabled_plugins_option(conf_class)

    # Save the original plugins_path before any modification
    original_path = conf_class.main.plugins_path.get()
    original_enabled_list = None
    if enabled_plugins_option is not None:
        original_enabled_list = enabled_plugins_option.get(None)

    with temporary_plugin_dir(
        VISUAL_PLUGINS_DIR, module_prefixes=VISUAL_PLUGIN_MODULE_PREFIXES
    ) as tmpdir:
        print(f"Visual test plugin directory: {tmpdir}")
        created = create_visual_test_dataset(tmpdir, bulk_count=40)
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
            print("VISUAL TEST SETUP COMPLETE")
            print("=" * 70)
            print("\nCreated plugins:")
            print("  1. 📋 Many Actions       – dropdown menu with 5 actions")
            print("  2. 📄 Long Description   – test 'Show more' button")
            print("  3. 📁 Nested Menus       – 3-level nested submenus")
            print(
                f"  4. 🔢 {bulk_count} Auto Plugins    – scrollbar in Settings > Plugins"
            )
            print("\nManual verification checklist:")
            print("  ✓ Open Plugins menu → see all plugin actions")
            print("  ✓ Open Settings > Plugins → verify scrollbar in plugin list")
            print("  ✓ Click 'Show more' on long description plugin")
            print("  ✓ Toggle enable/disable checkboxes")
            print("\nLaunching DataLab...")
            print("=" * 70 + "\n")

            # Configure: enable all plugins, point plugins_path to temp dir.
            # plugins_path is a DirectoryItem (single directory), so we replace it
            # rather than concatenating.  The original path (if any) is already on
            # sys.path thanks to DataLab startup, so existing plugins stay visible.
            if enabled_plugins_option is not None:
                enabled_plugins_option.set(None)
            conf_class.main.plugins_path.set(tmpdir)

            # Launch DataLab
            run = import_module("datalab.app").run

            run()

        finally:
            if enabled_plugins_option is not None:
                enabled_plugins_option.set(original_enabled_list)
            conf_class.main.plugins_path.set(original_path)
            print(f"\nCleaning up: {tmpdir}")


if __name__ == "__main__":
    main()
