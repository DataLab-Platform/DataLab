# Version 1.2 #

## DataLab Version 1.2.0 (2026-04-20) ##

### ✨ New Features ###

**Plugin configuration dialog:**

DataLab now provides a dedicated **plugin configuration dialog** (accessible via "Plugins > Configure plugins...") that gives full control over third-party plugin management:

* Enable or disable individual plugins using checkboxes, or toggle all plugins at once with a tri-state master checkbox
* Filter plugins by status: all, enabled, disabled, or plugins with import errors
* View plugin details including version, author, and expandable long descriptions directly in the dialog
* Plugins with import errors are displayed prominently at the top with their full traceback, making it easy to diagnose installation issues

**Plugin hot-reload:**

* Third-party plugins can now be reloaded at runtime without restarting DataLab, via "Plugins > Reload plugins"
* Enabling or disabling plugins in the Preferences dialog or plugin configuration dialog takes effect immediately - no restart required
* The Plugins menu, status widget, and plugin actions are automatically refreshed after configuration changes

**Multi-instance detection:**

DataLab now detects when another instance is already running and warns the user before opening a second instance:

* Uses a PID-based lock file mechanism that supports multiple concurrent instances (reference counting)
* Stale PIDs from crashed processes are automatically cleaned up
* Cross-platform support (Windows, macOS, Linux) using platform-specific process detection
* Closing one instance no longer incorrectly removes the lock file when other instances are still running

**Image ROI editor contrast synchronization:**

* The image ROI editor now shares contrast (LUT) settings with the source image panel
* Adjusting the contrast in the ROI editor is reflected back in the main panel and vice versa
* Contrast controls are fully re-enabled in the image ROI editor dialog

### 📖 Documentation ###

* Added API documentation for the `datalab.objectmodel` module
* Added screenshots for the "Paste metadata" dialog (signal and image panels)
* Updated plugin documentation to describe the new configuration dialog, hot-reload workflow, and plugin API helpers
* Updated third-party plugin development guide with new template references and test coverage information
* Updated French translations across all new and modified documentation pages

### 🔧 Improvements ###

**Compatibility:**

* Officially support pandas 3.0.x (updated dependency constraint from `< 3.0` to `< 3.1`)
* Updated minimum Sigima requirement from 1.1.0 to 1.1.2 to benefit from latest computation engine fixes and improvements
* Added legacy support for the `WINPYDIRBASE` environment variable for WinPython-based deployments

**Plugin system hardening:**

* Plugin discovery and registration is now resilient to third-party import failures while preserving error reporting in the console, logs, and configuration dialog
* Plugin submenus are now scrollable to prevent overflow when many plugin entries are registered

**Development tooling:**

* New `run_with_env.py` script for running tasks across multiple Python environment contexts (WinPython, venv, etc.)
* Simplified environment variable handling by removing the `DATALAB_ENV_LOADED` system
* Fixed Coverage full VS Code task to properly use the `run_with_env.py` wrapper

### 🛠️ Bug Fixes ###

**HDF5 workspace - Table serialization:**

* Fixed callable metadata not being stripped during HDF5 save/load of `TableResult` objects
* Fixed string-based enum values in table results not being restored correctly after HDF5 round-trip
* Fixed `column_formats` attribute not surviving HDF5 round-trip for both `TableResult` and `TableResultBuilder` outputs

**Plugin system:**

* Fixed `AttributeError` in plugin configuration dialog when clicking "Show full description" (incorrect attribute reference)
* Fixed plugin import errors being silently swallowed when they occurred before the internal console was initialized

**Remote control API:**

* Added new `set_object` method to the proxy API (XML-RPC and Web API) allowing users to push modified signal/image objects back to DataLab after retrieving them with `get_object` — previously, modifications to object properties (e.g. `dx`, `dy`, `x0`, `y0`, `title`) were lost because `get_object` returns a copy (fixes [Issue #305](https://github.com/DataLab-Platform/DataLab/issues/305))
