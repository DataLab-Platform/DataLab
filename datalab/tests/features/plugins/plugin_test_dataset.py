# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Standalone helpers to generate and clean plugin test datasets.

This module centralizes all file generation used by plugin-related tests and
manual inspection checks. It may be imported by tests, or executed directly to
create or clear a full dataset on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import os.path as osp
import shutil
import sys
import textwrap
from collections.abc import Iterable

TEMPLATES_DIR = osp.abspath(osp.join(osp.dirname(__file__), "templates"))
TEST_PLUGINS_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), "..", "..", "..", "data", "tests", "plugins")
)
MANUAL_TEST_PLUGINS_DIR = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        "..",
        "..",
        "..",
        "data",
        "tests",
        "manual_plugins",
    )
)

TEST_PLUGIN_MODULE_PREFIXES = ("datalab_test_plugin",)
MANUAL_PLUGIN_MODULE_PREFIXES = ("datalab_plugin",)

__all__ = [
    "MANUAL_TEST_PLUGINS_DIR",
    "TEST_PLUGINS_DIR",
    "clear_plugin_directory",
    "create_manual_test_plugin_dataset",
    "create_plugin_directory",
    "create_plugin_file",
    "create_plugin_from_template",
    "create_test_plugin_dataset",
    "ensure_plugin_dir_on_syspath",
    "generate_bulk_plugins",
    "get_plugin_template",
    "temporary_plugin_dir",
    "temporary_template_plugin",
]


def get_plugin_template(filename: str) -> str:
    """Read a plugin template file from the templates directory."""
    with open(osp.join(TEMPLATES_DIR, filename), "r", encoding="utf-8") as handle:
        return handle.read()


def _format_python_string_literal(
    text: str, *, inner_indent: int = 12, outer_indent: int = 8
) -> str:
    """Format a string as a wrapped Python string literal expression."""
    wrapped = textwrap.wrap(
        text,
        width=68,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) <= 1:
        return repr(text)

    inner_prefix = " " * inner_indent
    outer_prefix = " " * outer_indent
    joined = "\n".join(f"{inner_prefix}{part!r}" for part in wrapped)
    return f"(\n{joined}\n{outer_prefix})"


def create_plugin_directory(plugin_dir: str, *, clear_existing: bool = True) -> str:
    """Create the target plugin directory.

    Args:
        plugin_dir: Directory receiving generated plugin files.
        clear_existing: Remove the directory first when it already exists.

    Returns:
        Absolute plugin directory path.
    """
    plugin_dir = osp.abspath(plugin_dir)
    if clear_existing and osp.exists(plugin_dir):
        shutil.rmtree(plugin_dir)
    os.makedirs(plugin_dir, exist_ok=True)
    return plugin_dir


def _matches_module_prefix(name: str, module_prefixes: Iterable[str]) -> bool:
    """Return whether a module name belongs to one of the managed prefixes."""
    return any(name.startswith(prefix) for prefix in module_prefixes)


def purge_plugin_modules(module_prefixes: Iterable[str]) -> None:
    """Remove generated plugin modules from the import cache."""
    stale_modules = [
        name for name in sys.modules if _matches_module_prefix(name, module_prefixes)
    ]
    for name in stale_modules:
        del sys.modules[name]


def clear_plugin_directory(
    plugin_dir: str,
    *,
    module_prefixes: Iterable[str] = TEST_PLUGIN_MODULE_PREFIXES,
    remove_from_syspath: bool = True,
) -> None:
    """Clear a generated plugin directory and associated import state."""
    plugin_dir = osp.abspath(plugin_dir)
    purge_plugin_modules(module_prefixes)
    if remove_from_syspath and plugin_dir in sys.path:
        sys.path.remove(plugin_dir)
    sys.path_importer_cache.pop(plugin_dir, None)
    importlib.invalidate_caches()
    shutil.rmtree(plugin_dir, ignore_errors=True)


def ensure_plugin_dir_on_syspath(plugin_dir: str) -> str:
    """Ensure a generated plugin directory is importable."""
    plugin_dir = osp.abspath(plugin_dir)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    sys.path_importer_cache.pop(plugin_dir, None)
    importlib.invalidate_caches()
    return plugin_dir


@contextlib.contextmanager
def temporary_plugin_dir(
    plugin_dir: str = TEST_PLUGINS_DIR,
    *,
    module_prefixes: Iterable[str] = TEST_PLUGIN_MODULE_PREFIXES,
):
    """Create, expose and clean a deterministic plugin directory for tests."""
    plugin_dir = create_plugin_directory(plugin_dir)
    ensure_plugin_dir_on_syspath(plugin_dir)
    try:
        yield plugin_dir
    finally:
        clear_plugin_directory(plugin_dir, module_prefixes=module_prefixes)


def create_plugin_file(
    directory: str,
    filename: str,
    class_name: str,
    plugin_name: str,
    action_name: str,
    action_obj_name: str,
    test_code: str = "pass",
) -> str:
    """Create a plugin file from the standard valid plugin template."""
    template = get_plugin_template("plugin_valid.py.template")
    content = template.replace("{class_name}", class_name)
    content = content.replace("{plugin_name}", plugin_name)
    content = content.replace("{action_name}", action_name)
    content = content.replace("{action_object_name}", action_obj_name)
    content = content.replace("{test_code}", test_code)

    filepath = osp.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(content)
    return filepath


def create_plugin_from_template(
    directory: str,
    filename: str,
    template_name: str,
    replacements: dict[str, str],
) -> str:
    """Create a plugin file from any template with placeholder replacements."""
    content = get_plugin_template(template_name)
    for key, value in replacements.items():
        content = content.replace(key, value)

    filepath = osp.join(directory, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(content)
    return filepath


@contextlib.contextmanager
def temporary_template_plugin(
    filename: str,
    template_name: str,
    replacements: dict[str, str],
):
    """Create a single template-based plugin inside a temporary plugin directory."""
    with temporary_plugin_dir() as plugin_dir:
        create_plugin_from_template(plugin_dir, filename, template_name, replacements)
        yield plugin_dir


def create_test_plugin_dataset(plugin_dir: str = TEST_PLUGINS_DIR) -> list[str]:
    """Create the standard plugin dataset used by plugin-related tests."""
    plugin_dir = create_plugin_directory(plugin_dir)
    created = [
        create_plugin_file(
            plugin_dir,
            "datalab_test_plugin_1.py",
            "TestPluginOne",
            "Test Plugin 1",
            "Action One",
            "action_1",
        ),
        create_plugin_file(
            plugin_dir,
            "datalab_test_plugin_2.py",
            "TestPluginTwo",
            "Test Plugin 2",
            "Action Two",
            "action_2",
        ),
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
        ),
        create_plugin_from_template(
            plugin_dir,
            "datalab_test_plugin_long_desc.py",
            "plugin_long_description.py.template",
            {
                "{class_name}": "TestPluginLongDescription",
                "{plugin_name}": "Long Description Test",
                "{long_description}": _format_python_string_literal(
                    "This is an extremely long description that is designed to test "
                    "how the plugin system handles descriptions that span "
                    "multiple lines and contain a large amount of text. "
                    "The description should not break the UI layout or cause "
                    "any rendering issues. It should be properly truncated "
                    "or wrapped in any display contexts such as tooltips, "
                    "status bars, or configuration dialogs. This text "
                    "continues to be very long to ensure we adequately test "
                    "the edge case of exceptionally verbose plugin "
                    "descriptions that might be provided by third-party "
                    "developers who want to thoroughly explain what their "
                    "plugin does and how to use it."
                ),
                "{action_name}": "Test Action",
                "{test_code}": "self.main._test_long_desc = True",
            },
        ),
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
        ),
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
        ),
    ]
    return created


def generate_bulk_plugins(plugin_dir: str, count: int = 20) -> list[str]:
    """Generate many simple plugins for scrollbar and list overflow checks."""
    created: list[str] = []
    for index in range(1, count + 1):
        long_desc = index % 4 == 0

        description = f"Auto-generated test plugin #{index} for scrollbar testing."
        if long_desc:
            description = (
                f"Auto-generated test plugin #{index}. "
                + "This plugin has a very long description to test the "
                "'Show more' button behaviour. "
                * 6
                + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            )

        description_literal = _format_python_string_literal(description)

        plugin_code = f'''\
"""Auto-generated test plugin {index}"""
from datalab.plugins import PluginBase, PluginInfo


class AutoTestPlugin{index}(PluginBase):
    """Auto-generated test plugin {index}."""

    PLUGIN_INFO = PluginInfo(
        name="Auto Test Plugin {index}",
        version="1.0.{index}",
        description={description_literal},
    )

    def create_actions(self):
        acth = self.signalpanel.acthandler
        acth.new_action(
            "Auto Plugin {index} Action",
            triggered=lambda: print("Auto plugin {index} action triggered"),
            select_condition="always",
        )
'''
        filename = f"datalab_plugin_auto_{index:02d}.py"
        filepath = osp.join(plugin_dir, filename)
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(plugin_code)
        created.append(filepath)

    return created


def create_manual_test_plugin_dataset(
    plugin_dir: str = MANUAL_TEST_PLUGINS_DIR, *, bulk_count: int = 40
) -> list[str]:
    """Create the manual inspection plugin dataset."""
    plugin_dir = create_plugin_directory(plugin_dir)
    created = [
        create_plugin_from_template(
            plugin_dir,
            "datalab_plugin_many_actions_test.py",
            "plugin_many_actions.py.template",
            {
                "{class_name}": "PluginTestManyActions",
                "{plugin_name}": "Plugin Test: Many Actions",
                "{menu_name}": "Test Menu with Many Actions",
                "{action_prefix}": "Test Action",
                "{test_code_1}": "print('Action 1 triggered')",
                "{test_code_2}": "print('Action 2 triggered')",
                "{test_code_3}": "print('Action 3 triggered')",
                "{test_code_4}": "print('Action 4 triggered')",
                "{test_code_5}": "print('Action 5 triggered')",
            },
        ),
        create_plugin_from_template(
            plugin_dir,
            "datalab_plugin_long_description_test.py",
            "plugin_long_description.py.template",
            {
                "{class_name}": "PluginTestLongDescription",
                "{plugin_name}": "Plugin Test: Long Description",
                "{long_description}": _format_python_string_literal(
                    "This is an EXTREMELY LONG DESCRIPTION to test how DataLab handles "
                    "verbose plugin descriptions. "
                    * 10
                    + "The description should be displayed properly in tooltips, "
                    "status "
                    "bars, and configuration dialogs without breaking the UI layout. "
                    * 50
                ),
                "{action_name}": "Test Long Description",
                "{test_code}": "print('Long description plugin action triggered')",
            },
        ),
        create_plugin_from_template(
            plugin_dir,
            "datalab_plugin_responsive_description_test.py",
            "plugin_long_description.py.template",
            {
                "{class_name}": "PluginTestResponsiveDescription",
                "{plugin_name}": "Plugin Test: Responsive Description",
                "{long_description}": _format_python_string_literal(
                    "This medium-length description is tuned for manual verification "
                    "of the plugin configuration dialog. At the default window "
                    "width, it should be truncated with a Show more button. "
                    "When you widen the "
                    "configuration dialog, the same description should become fully "
                    "visible and the button should disappear because the "
                    "rendered space "
                    "is sufficient."
                ),
                "{action_name}": "↔ Test Responsive Description",
                "{test_code}": (
                    "print('Responsive description plugin action triggered')"
                ),
            },
        ),
        create_plugin_from_template(
            plugin_dir,
            "datalab_plugin_nested_menus_test.py",
            "plugin_nested_menus.py.template",
            {
                "{class_name}": "PluginTestNestedMenus",
                "{plugin_name}": "Plugin Test: Nested Menus",
                "{menu_level_1}": "Level 1 Menu",
                "{action_level_1}": "Action at Level 1",
                "{test_code_1}": "print('Level 1 action triggered')",
                "{menu_level_2}": "Level 2 Submenu",
                "{action_level_2}": "Action at Level 2",
                "{test_code_2}": "print('Level 2 action triggered')",
                "{menu_level_3}": "Level 3 Submenu",
                "{action_level_3}": "Action at Level 3",
                "{test_code_3}": "print('Level 3 action triggered')",
            },
        ),
    ]
    created.extend(generate_bulk_plugins(plugin_dir, count=bulk_count))
    return created


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset management."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "clear"))
    parser.add_argument(
        "--dataset",
        choices=("test", "manual"),
        default="test",
        help="Dataset recipe to manage.",
    )
    parser.add_argument(
        "--plugin-dir",
        help="Override the target directory used for generated plugins.",
    )
    parser.add_argument(
        "--bulk-count",
        type=int,
        default=40,
        help="Number of auto-generated manual-test plugins.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    args = _build_parser().parse_args()
    if args.dataset == "manual":
        plugin_dir = args.plugin_dir or MANUAL_TEST_PLUGINS_DIR
        module_prefixes = MANUAL_PLUGIN_MODULE_PREFIXES

        def create_dataset() -> list[str]:
            return create_manual_test_plugin_dataset(
                plugin_dir, bulk_count=args.bulk_count
            )
    else:
        plugin_dir = args.plugin_dir or TEST_PLUGINS_DIR
        module_prefixes = TEST_PLUGIN_MODULE_PREFIXES

        def create_dataset() -> list[str]:
            return create_test_plugin_dataset(plugin_dir)

    if args.action == "clear":
        clear_plugin_directory(plugin_dir, module_prefixes=module_prefixes)
        print(f"Cleared plugin dataset: {osp.abspath(plugin_dir)}")
        return

    created = create_dataset()
    print(f"Created {len(created)} plugin files in {osp.abspath(plugin_dir)}")
    for filepath in created:
        print(f" - {osp.basename(filepath)}")


if __name__ == "__main__":
    main()
