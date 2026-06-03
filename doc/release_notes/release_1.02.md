# Version 1.2 #

## DataLab Version 1.2.1 ##

### 🛠️ Bug Fixes ###

**Signal panel menus:**

* Restored the "Linear calibration" entry in the "Processing > Axis transformation" menu of the Signal panel — it had been inadvertently dropped during a menu reorganization and was no longer reachable from the menu bar, even though the underlying computation was still available (fixes [Issue #312](https://github.com/DataLab-Platform/DataLab/issues/312))

**Signal panel — Y range cursor:**

* Fixed the Y range cursor annotation displaying an incorrect inequality (e.g. `5 < y < 2`) and a negative ∆y when the top cursor was dragged below the bottom cursor — the annotation now always shows values in sorted order with a positive range width (fixes [Issue #306](https://github.com/DataLab-Platform/DataLab/issues/306))

**Image panel — Z-axis logarithmic scale:**

* Fixed the "Base-10 logarithmic Z axis" toolbar action being permanently greyed out in the image panel — the tool now works correctly for all image types (fixes [Issue #313](https://github.com/DataLab-Platform/DataLab/issues/313), fixed upstream in PlotPy ≥ 2.10.0)

**Image panel — contour detection:**

* Fixed ellipse and circle contour detection producing incorrect results (wrong positions and sizes) due to swapped X/Y coordinates and missing unit conversions in the scikit-image model fitting code (fixes [Issue #326](https://github.com/DataLab-Platform/DataLab/issues/326), fixed upstream in Sigima ≥ 1.1.3)

**ROI editor:**

* Fixed the ROI editor dialog ignoring non-linear axis scales (e.g. logarithmic) of the source plot — the dialog now preserves the same axis scale configuration as the main panel (fixes [Issue #315](https://github.com/DataLab-Platform/DataLab/issues/315))

**Theme and display:**

* Fixed plot marker and shape colors being reverted to PlotPy defaults (yellow) instead of DataLab's configured colors (red) when the color mode was explicitly set to "light" or "dark" in settings — the custom color overrides are now re-applied after each theme switch (fixes [Issue #297](https://github.com/DataLab-Platform/DataLab/issues/297))
* Fixed UI elements (text, icons, dialog layouts) not scaling properly with high-DPI display settings — DataSet HTML rendering, icon sizes, and viewport dimensions now adapt to the system scale factor (fixes [Issue #317](https://github.com/DataLab-Platform/DataLab/issues/317), fixed upstream in guidata ≥ 3.14.4)

**Image properties editor:**

* Fixed pixel size (`Δx` / `Δy`) input field corrupting typed values — most digits typed after the first were misinterpreted as decimal places due to a reactive update loop in the parameter editing widget (fixes [Issue #320](https://github.com/DataLab-Platform/DataLab/issues/320), fixed upstream in guidata ≥ 3.14.4)

**Debug environment variable renamed:**

* Renamed the debug environment variable from `DEBUG` to `DATALAB_DEBUG` — the generic `DEBUG` name collided with widely-used third-party conventions (Django, Flask, Node.js tooling, CI systems) and could silently reset the user configuration file at startup when set for unrelated reasons. Setting `DATALAB_DEBUG=1` now activates debug mode; the bare `DEBUG` variable is ignored (fixes [Issue #319](https://github.com/DataLab-Platform/DataLab/issues/319))

**Remote control API:**

* Exposed `get_current_object_uuid()` on the proxy API, making it available through XML-RPC and Web API

### 🔧 Improvements ###

**Compatibility:**

* Updated minimum guidata requirement from 3.14.3 to 3.14.4 (high DPI and screen scaling issue, dataset input field fix, add secure build cli cmd)
* Updated minimum PlotPy requirement from 2.8.2 to 2.10.0 (Z-axis log scale fix, toolbar overflow button visibility in dark theme, PythonQwt 0.16.0 optimization)
* Updated minimum Sigima requirement from 1.1.2 to 1.1.3 (ellipse/circle contour detection fix)

## DataLab Version 1.2.0 (2026-04-20) ##

### ✨ New Features ###

**Plugin configuration dialog:**

DataLab now provides a dedicated **plugin configuration dialog** (accessible via "Plugins > Configure plugins...") that gives full control over third-party plugin management:

* Enable or disable individual plugins using checkboxes, or toggle all plugins at once with a tri-state master checkbox
* Filter plugins by status: all, enabled, disabled, or plugins with import errors
* View plugin details including version, author, and expandable long descriptions directly in the dialog
* Plugins with import errors are displayed prominently at the top with their full traceback, making it easy to diagnose installation issues
* The expandable text widget used for long descriptions computes its preferred width from a fixed measurement context, ensuring stable layout and reliable "Show full description" toggling regardless of dialog resizing or offscreen rendering

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

**Remote control API — push modified objects back to DataLab:**

The proxy API (XML-RPC and Web API) now exposes a new `set_object` method that updates an existing signal or image in DataLab from a modified copy obtained via `get_object` (fixes [Issue #305](https://github.com/DataLab-Platform/DataLab/issues/305)):

* Previously, modifications to object properties (e.g. `dx`, `dy`, `x0`, `y0`, `title`) made on the result of `get_object` were lost because `get_object` returns a copy — `set_object` now provides a clean round-trip workflow
* Works for both signal and image objects: computed result items attached to the object are preserved during the update, so updating an `ImageObj` no longer triggers a type mismatch
* The properties panel is automatically refreshed after `set_object`, so updated object properties (title, units, axes, uncertainties, etc.) are immediately visible in the GUI

### 📖 Documentation ###

* Added API documentation for the `datalab.objectmodel` module
* Added screenshots for the "Paste metadata" dialog (signal and image panels)
* Updated plugin documentation to describe the new configuration dialog, hot-reload workflow, and plugin API helpers
* Updated third-party plugin development guide with new template references and test coverage information
* Expanded Web API reference: documented the binary data transfer options (`?compress=false` for faster uncompressed NPZ downloads, `?overwrite=true` for atomic replacement of existing objects), the in-place `PUT /objects/{name}` endpoint that updates an object while preserving its identity, group membership and position, and the new "Computation" section listing the `/select` and `/calc` endpoints used to drive DataLab computations remotely
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
