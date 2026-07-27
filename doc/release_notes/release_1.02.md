# Version 1.2 #

## DataLab Version 1.2.2 ##

### 🛠️ Bug Fixes since version 1.2.1 ###

**Signal peak creation and fitting:**

* Gaussian, Lorentzian and Voigt creation and fit dialogs now express amplitude as signed peak height above the baseline, in the signal's Y units. Objects carrying historical area-based peak parameters remain displayable and keep their original samples; editing or re-evaluating them now requires an explicit conversion confirmed by the user (fixes [Issue #350](https://github.com/DataLab-Platform/DataLab/issues/350))
* Interactive multi-Gaussian and multi-Lorentzian results now retain every detected peak position (`x0_i`) in canonical fit metadata, allowing complete theoretical models to be exported or re-evaluated on another X axis without adding center controls to the fit dialog (fixes [Issue #351](https://github.com/DataLab-Platform/DataLab/issues/351))
* The "Evaluate fit" feature is now available for every interactive fit, not just peak fits: linear, polynomial, exponential, sinusoidal, CDF, Planckian, two half-Gaussian and piecewise exponential fitting curves may now be re-evaluated on another X axis or exported as theoretical models (fixes [Issue #355](https://github.com/DataLab-Platform/DataLab/issues/355)). Fitting curves computed by earlier versions carry parameters that were labelled in the interface language and cannot be re-evaluated: DataLab now reports this explicitly and such curves have to be recomputed.
* Scripts and remote clients may migrate historical fit metadata explicitly by retrieving an object with `get_object()`, converting a copy with Sigima's `convert_legacy_peak_fit_params()`, and updating the existing object with `set_object()`; the conversion never regenerates the signal samples.
* Fixed interactive fit parameter sliders whose bounds excluded reachable solutions: the exponential `B` slider could not express a decay, the cumulative distribution function amplitude could not express a descending transition, and the piecewise exponential amplitude bounds were inverted for negative initial guesses (fixes [Issue #352](https://github.com/DataLab-Platform/DataLab/issues/352))
* Fixed the Planckian fit dialog labelling the `x0` parameter as the peak wavelength — the maximum of the model is located at approximately `1.007 * x0 / sigma`, so the two only coincide when `sigma` equals 1

**HDF5 import:**

* Fixed a crash when running a computation on an object imported from an HDF5 file containing object or region reference attributes (e.g. `DIMENSION_LIST` / `REFERENCE_LIST` arrays produced by `h4toh5convert`) — such attributes cannot be serialized to worker processes and are now skipped instead of being copied to the object metadata (fixes [Issue #346](https://github.com/DataLab-Platform/DataLab/issues/346))

**Text import wizard:**

* Fixed a sporadic application crash when closing the Text Import Wizard — the embedded plot preview is now released deterministically instead of being torn down later by the garbage collector (fixes [Issue #335](https://github.com/DataLab-Platform/DataLab/issues/335))

**Working directory handling:**

* DataLab no longer changes its process working directory when a file is opened or saved. The last used folder is still remembered for file dialogs, but the directory itself is no longer locked for the lifetime of the application on Windows (it can be moved, renamed or deleted while DataLab is running), and macros or scripts started from DataLab no longer inherit an unexpected working directory

**Packaging:**

* Fixed missing data files in the distributed package: all `.h5` test/demo files and `.template` files are now included in the wheel and source distribution

### 📖 Documentation since version 1.2.1 ###

* Added a new "DataLab Platform ecosystem" page positioning DataLab (desktop), DataLab-Web (browser edition), DataLab-Kernel (Jupyter kernel) and Sigima (computation engine) with respect to each other; the browser edition is now surfaced on the home and getting started pages, plugin web compatibility is documented, and the roadmap was refreshed (web frontend and Jupyter kernel delivered)
* Added NixOS installation instructions: DataLab is packaged for NixOS and distributed through the NGI Forge, thanks to a contribution from the Nix@NGI team as part of DataLab's funding through the NGI0 Commons Fund
* Added a "Talks & Events" section to the documentation, starting with the EuroSciPy 2026 presentation details
* Fixed the polynomial signal creation formula, which listed a `y0` term and an arbitrary number of coefficients whereas the parameters actually expose `a0` to `a5`, and documented the sigmoid fit, which was available but undocumented (fixes [Issue #353](https://github.com/DataLab-Platform/DataLab/issues/353))
* Aligned the documented signal model formulas with the new height-based parameterization of the Gaussian, Lorentzian and Voigt models
* Refreshed the signal and image "Edit" documentation examples, which still showed object titles produced by older versions (fixes [Issue #354](https://github.com/DataLab-Platform/DataLab/issues/354))
* Visiting the documentation site root now redirects to the French or English version depending on the browser language, instead of always serving the English pages
* Updated French translations across all new and modified documentation pages

### 🔧 Improvements since version 1.2.1 ###

**Compatibility:**

* Updated minimum Sigima requirement from 1.1.3 to 1.1.6 (computation fixes and versioned peak-height parameters introduced in 1.1.6)

## DataLab Version 1.2.1 (2026-06-05) ##

### 🛠️ Bug Fixes since version 1.2.0 ###

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

### 🔧 Improvements since version 1.2.0 ###

**Compatibility:**

* Updated minimum guidata requirement from 3.14.3 to 3.14.4 (high DPI and screen scaling issue, dataset input field fix, add secure build CLI cmd)
* Updated minimum PlotPy requirement from 2.8.2 to 2.10.0 (Z-axis log scale fix, toolbar overflow button visibility in dark theme, PythonQwt 0.16.0 optimization)
* Updated minimum Sigima requirement from 1.1.2 to 1.1.3 (ellipse and circle contour detection fix)

## DataLab Version 1.2.0 (2026-04-20) ##

### ✨ New Features since version 1.1 ###

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

### 📖 Documentation since version 1.1 ###

* Added API documentation for the `datalab.objectmodel` module
* Added screenshots for the "Paste metadata" dialog (signal and image panels)
* Updated plugin documentation to describe the new configuration dialog, hot-reload workflow, and plugin API helpers
* Updated third-party plugin development guide with new template references and test coverage information
* Expanded Web API reference: documented the binary data transfer options (`?compress=false` for faster uncompressed NPZ downloads, `?overwrite=true` for atomic replacement of existing objects), the in-place `PUT /objects/{name}` endpoint that updates an object while preserving its identity, group membership and position, and the new "Computation" section listing the `/select` and `/calc` endpoints used to drive DataLab computations remotely
* Updated French translations across all new and modified documentation pages

### 🔧 Improvements since version 1.1 ###

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

### 🛠️ Bug Fixes since version 1.1 ###

**HDF5 workspace - Table serialization:**

* Fixed callable metadata not being stripped during HDF5 save/load of `TableResult` objects
* Fixed string-based enum values in table results not being restored correctly after HDF5 round-trip
* Fixed `column_formats` attribute not surviving HDF5 round-trip for both `TableResult` and `TableResultBuilder` outputs

**Plugin system:**

* Fixed `AttributeError` in plugin configuration dialog when clicking "Show full description" (incorrect attribute reference)
* Fixed plugin import errors being silently swallowed when they occurred before the internal console was initialized
