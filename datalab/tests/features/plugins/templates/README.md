# Plugin Test Templates

This directory contains template files for testing the DataLab plugin system.

## Template Files

Each template file contains placeholders (e.g., `{class_name}`, `{plugin_name}`) that are replaced by test code to generate complete plugin files.

### Available Templates

Template files use the `.py.template` extension so that linters (Ruff, Pylint)
skip them – they contain `{placeholders}` that are not valid Python.

- **plugin_valid.py.template**: Basic valid plugin template with actions for both signal and image panels
- **plugin_nested_menus.py.template**: Plugin with nested submenus (3 levels deep)
- **plugin_with_dialogs.py.template**: Plugin demonstrating dialog methods (show_warning, show_error, etc.)
- **plugin_many_actions.py.template**: Plugin with multiple actions in a dropdown menu
- **plugin_long_description.py.template**: Plugin with an extremely long description

Note: Simple error-case plugins (init error, missing `create_actions`, invalid
`PLUGIN_INFO`, syntax errors) are inlined directly in the test file.

## Usage in Tests

Tests use these templates via the `create_plugin_file()` and `create_plugin_from_template()`
helper functions in `plugin_test_dataset.py`:

```python
# For plugin_valid.py.template (most common):
create_plugin_file(plugin_dir, "datalab_my_plugin.py",
                   "MyPlugin", "My Test Plugin", "My Action", "my_action")

# For any template with custom placeholders:
create_plugin_from_template(plugin_dir, "datalab_my_plugin.py",
                            "plugin_nested_menus.py.template",
                            {"{class_name}": "MyPlugin", ...})
```

## Manual Inspection Script

Run `plugin_test_dataset.py` to create or clear a standalone plugin dataset:

```bash
python scripts/run_with_env.py python datalab/tests/features/plugins/plugin_test_dataset.py create --dataset test
python scripts/run_with_env.py python datalab/tests/features/plugins/plugin_test_dataset.py clear --dataset test
python scripts/run_with_env.py python datalab/tests/features/plugins/plugin_test_dataset.py create --dataset manual
python scripts/run_with_env.py python datalab/tests/features/plugins/plugin_test_dataset.py clear --dataset manual
```

Run `launch_with_test_plugins.py` to launch DataLab with the manual plugin test dataset:

```bash
python scripts/run_with_env.py python datalab/tests/features/plugins/launch_with_test_plugins.py
```

This allows manual inspection of:
- Dropdown menus with many actions
- Plugins with long descriptions
- Nested submenu behavior
- Plugin enable/disable UI

## Test Coverage

The plugin system tests (`datalab/tests/features/plugins/test_plugins.py`) cover:

1. **Plugin Lifecycle**: Discovery, loading, reloading, cleanup
2. **Error Handling**: ImportError, InitError, SyntaxError, invalid PLUGIN_INFO
3. **UI Integration**: Menu population, action registration, panel-specific actions
4. **Configuration**: Enable/disable plugins, persistent settings
5. **Edge Cases**: Duplicate names, missing methods, long descriptions
6. **Manual Inspection**: Observable behavior in a running DataLab instance

## Adding New Templates

When creating a new template:

1. Use `{placeholder}` syntax for values that will be replaced
2. Avoid f-strings with placeholders (use variables instead)
3. Document required placeholders in template comments
4. Add corresponding test in `test_plugins.py`
5. Update this README

## Directory Structure

```
datalab/tests/features/plugins/templates/
├── README.md                      # This file
├── plugin_valid.py.template                # Standard valid plugin
├── plugin_nested_menus.py.template         # Nested submenus
├── plugin_with_dialogs.py.template         # Dialog methods
├── plugin_many_actions.py.template         # Many dropdown actions
└── plugin_long_description.py.template     # Long description text
```
