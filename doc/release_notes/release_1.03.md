# Version 1.3 #

## DataLab Version 1.3.0 ##

### ✨ New Features ###

**History Panel:**

DataLab now records what you do and lets you replay it. A new **History** panel keeps track of every action applied to signals
and images, from object creation and file loading to processing, analysis and
region-of-interest edits (implements
[Issue #90](https://github.com/DataLab-Platform/DataLab/issues/90)).

* **Recording sessions**: actions are grouped into sessions, one active session per panel. Recording is disabled by default and is proposed when you create or load an object; configurable session policies control how plugin and macro objects are attached to sessions
* **Processing chains**: each session is presented as a linear processing chain, making the lineage of a result immediately readable
* **Replay**: replay a whole session, or step through it action by action, to reproduce a workflow on new data. Interactive curve fits and intensity profiles are recorded as deterministic actions and replay without reopening their dialogs
* **In-place editing**: processing, analysis and object-creation parameters can be edited directly from the history, and downstream steps are recomputed through the cascade
* **Duplication**: duplicate a chain to explore a variant without touching the original one
* **Persistence**: histories are stored in the HDF5 workspace and can also be exported and imported as standalone `.dlhist` files
* **Compatibility checks**: before replaying, DataLab validates the recorded actions against the current workspace state and reports what can or cannot be applied
* The Delete key removes the selected entries in the History panel (same behavior as in the Signal and Image panels), and only applies while the history tree has focus
* A resolution dialog is displayed when replaying a processing chain broken by deleted objects: choose "Repair and continue" to prune the broken steps and replay the remaining valid ones, or cancel without modifying anything
* A dedicated documentation page and a step-by-step tutorial are available

**Built-in AI Assistant:**

DataLab now ships an optional **AI Assistant** dock panel that can converse with
you, inspect the workspace, create and process signals and images, and write and
run Python macros (implements
[Issue #316](https://github.com/DataLab-Platform/DataLab/issues/316)).

* Works with any **OpenAI-compatible endpoint**: OpenAI, GitHub Models, Azure OpenAI, Ollama, LM Studio, etc. A **mock provider** is available to try the assistant offline, without any API key
* The assistant can call a set of **built-in tools**: list and inspect objects, introspect the available operations, create synthetic signals and images, load files, apply any registered processing, trigger plugin actions, and create and run macros
* **Every action that modifies the workspace requires an explicit confirmation**, with a parameter preview and a syntax-highlighted preview of the macro source when a macro is proposed. Read-only tools may be auto-approved
* Conversations are persisted, can be browsed and reloaded, and exported to Markdown; token usage and context size are displayed
* Settings (provider, model, API key, base URL, temperature, timeout, iteration cap, auto-approval) are grouped in a dedicated **AI Assistant** tab of the preferences dialog, with a connection test button
* The API key is stored in clear text in the DataLab configuration file, and the settings dialog states it explicitly

**Command palette:**

* A VSCode-style **command palette** lets you search and run any menu command by typing part of its name or of its menu path, e.g. "Processing ‣ Fourier analysis ‣ FFT" (implements [Issue #337](https://github.com/DataLab-Platform/DataLab/issues/337))
* Three entry points: a search box in the top-right corner of the menu bar, a "Command palette..." entry at the top of the Help menu, and the `Ctrl+Shift+P` keyboard shortcut
* Fuzzy matching ranks the results, so a few characters are enough to reach the right command, with full keyboard navigation

**Spectral analysis support:**

Following the spectral analysis work carried out in Sigima, DataLab gains the
matching interface (implements
[Issue #309](https://github.com/DataLab-Platform/DataLab/issues/309)).

* New **Analysis > Extract peak positions** feature, storing the detected peaks as an XY-markers table result rendered as cross markers on the plot
* New **Analysis > Peak detection...** interactive dialog, to adjust threshold and minimum distance visually before storing the result
* New **Operations > Create signal from markers table...** feature, building a sticks signal from an XY-markers table result — convenient to overlay reference lines on a spectrum
* Marker results can be labelled in result tables (`#1`, `#2`, ... for XY markers, `a`, `b`, `c`, ... for axis markers), so each row can be matched with the corresponding marker drawn on the plot. This is controlled by a new "Show marker labels in result tables" option in the settings
* Result labels now support a customizable position and offset

**Signal regions of interest rendered along the curve:**

* Signal ROIs are no longer drawn as a single full-height yellow rectangle: the translucent fill is now **clipped to the signal curve** with a baseline at `y = 0`, and each ROI uses a **different color** taken from a cyclic ten-color palette (implements [Issue #314](https://github.com/DataLab-Platform/DataLab/issues/314))
* Colors are derived from the ROI index, so a given ROI keeps the same color across deletions, reorderings and editor reopenings
* Non-finite samples are filtered out before drawing, so signals with gaps render cleanly; on a logarithmic Y axis, the baseline falls back to the canvas bottom
* ROI persistence is unchanged: only coordinates are stored, visual properties are recomputed at render time

**Regions of interest from detection (auto mask):**

* Contour, blob, peak and Hough circle detection can create regions of interest directly from the detected features, turning intensity contours into editable polygon ROIs usable by every ROI-aware processing feature (implements [Issue #294](https://github.com/DataLab-Platform/DataLab/issues/294))

**Third-party plugin discovery via environment variable:**

* Added support for the `DATALAB_PLUGINS` environment variable, allowing one or more directories to be specified as additional plugin search paths (implements [Issue #311](https://github.com/DataLab-Platform/DataLab/issues/311))
* Multiple directories can be listed using the OS path separator (`;` on Windows, `:` on Linux/macOS), following the same convention as `PYTHONPATH`
* Listed directories are appended to the existing plugin search paths at startup and are picked up automatically by the plugin discovery mechanism
* Non-existent directories are silently skipped (a warning is recorded in the log file), so a stale environment variable on another machine will not prevent DataLab from starting

**Plugin management user experience:**

The plugin configuration dialog has been reworked to make third-party plugins
easier to install, inspect and troubleshoot (implements
[Issue #302](https://github.com/DataLab-Platform/DataLab/issues/302)).

* **Persistent additional plugin directories**: several extra directories can be added, edited and removed from the dialog and are saved in the DataLab configuration, next to the read-only list of default directories
* **Open file** and **Show in folder** buttons per plugin, to inspect or edit a plugin script without hunting through the file system
* An **"Apply and reload plugins"** button and a **Plugins > Reload plugins** menu entry, to pick up plugin changes without restarting DataLab
* A **last load timestamp** displayed for the current session, to confirm that changes have been taken into account
* Plugin list **filtering** (all / enabled / disabled / with errors), a global enable/disable switch, and an option to hide compatibility warnings for legacy v0.20 plugins
* A **Show in folder** action is now available for the DataLab configuration file, from the installation and configuration viewer

**Macro editor:**

* **Crash recovery**: unsaved macros are auto-saved, and DataLab proposes to recover them at the next start
* **Recent macros** dialog, listing the macros used in previous sessions and allowing to reopen one in a single click
* **Macro templates**, to start from a working skeleton (simple macro, image processing, generic method call)
* **Code completion** and a **find & replace** widget in the macro editor

**Object titles:**

* Short identifiers embedded in object titles (e.g. `s001`, `i012`) are now rendered as **clickable links** in the signal and image trees, so the source objects of a computation can be selected in a single click

**Replace special values processing (signal and image):**

DataLab now provides a dedicated **"Replace special values"** processing
function that detects and replaces `NaN`, `+Inf` and `-Inf` values in both
signals and images. The feature is available under
**Processing → Level adjustment → Replace special values** in both the Signal
and Image panels (implements
[Issue #142](https://github.com/DataLab-Platform/DataLab/issues/142)).

* Each target (`NaN`, `+Inf`, `-Inf`) can be processed independently with its
  own strategy
* Signal strategies: do nothing, replace with zero / constant / minimum /
  maximum / mean / median, delete affected points, forward fill, backward
  fill, interpolation (linear, spline, quadratic, cubic, PCHIP), N-neighbor
  minimum / maximum / mean / median
* Image strategies: do nothing, replace with zero / constant / minimum /
  maximum / mean / median, N-neighbor minimum / maximum / mean / median
* The parameter dialog displays **colored badges** showing the number (and
  percentage) of `NaN`, `+Inf` and `-Inf` samples found in the source object,
  giving immediate visibility on what will be modified
* When a neighbor strategy is selected, a **live kernel preview** shows the
  shape of the neighborhood that will be used for the replacement
* Integer images are handled explicitly: because `NaN` and infinite values
  cannot exist in integer data, the dialog explains that the operation is not
  applicable and prevents accidental processing, while preserving the original
  image data type without unnecessary float conversion

### 🔄 Changes ###

**DataLab now builds on SigimaX:**

* All the generic, application-level parts of DataLab (main window skeleton, configuration system, dockable plot widgets, HDF5 workspace and browser, log viewer, splash screen, status bar, scientific dialogs and PlotPy adapters) have been extracted into a new reusable library, **SigimaX**, and DataLab now derives from it instead of maintaining its own copies (implements [Issue #182](https://github.com/DataLab-Platform/DataLab/issues/182))
* This is an internal refactoring: existing workflows, settings and files are unchanged, but it considerably reduces duplicated code and makes it possible to build other Qt scientific applications on the same foundation
* As a consequence, DataLab now requires SigimaX ≥ 1.0.1, and its minimum requirements are aligned with it: Sigima ≥ 1.2.0, guidata ≥ 3.15.0 and PlotPy ≥ 2.11.0

**Configuration system:**

* DataLab's configuration has been migrated to the typed option system provided by SigimaX: options now carry an explicit type, default value and description, and are validated when read or written
* Settings recorded by earlier versions are migrated automatically at first start; the configuration file layout is otherwise preserved

**`cdlclient` package archived:**

* The standalone `cdlclient` package is deprecated and archived. Its lightweight XML-RPC client is now provided by Sigima, as `sigima.client.SimpleRemoteProxy` (implements [Issue #183](https://github.com/DataLab-Platform/DataLab/issues/183))
* Its Qt companion widgets have been taken over by SigimaX: the connection progress dialog is now `sigimax.widgets.connection.ConnectionDialog`, usable with any remote proxy exposing a blocking `connect()` method
* The documentation no longer refers to `cdlclient`

**Detection tools now preserve existing regions of interest:**

* Regions of interest created by peak, blob, Hough circle, and contour
  detection are now added to existing ROIs instead of replacing them, without
  displaying a destructive confirmation dialog (implements
  [Issue #340](https://github.com/DataLab-Platform/DataLab/issues/340))

**Analysis results are now refreshed on demand instead of automatically:**

* Analysis results (statistics, FWHM, centroid, peak/contour/blob detection,
  etc.) are no longer recomputed automatically when you modify a region of
  interest, transform the data, or edit object properties (implements
  [Issue #341](https://github.com/DataLab-Platform/DataLab/issues/341))
* Existing analysis results are now left untouched after such edits, avoiding
  surprising side effects and results that could become misleading once the
  data no longer matches the stored analysis parameters
* Ordinary replay and ordinary mutations do not recompute analyses; however,
  in History edit mode, editing upstream parameters recomputes downstream
  analysis actions through the cascade
* The familiar **"Recompute"** action (Edit menu, `Ctrl+R`) now refreshes both
  processing *and* analysis results, giving you full control over when analyses
  are updated
* For developers, the former panel-level `recompute_analysis` entry point has
  been renamed to `recompute_selected`; `recompute_analysis` now refers to a
  different processor-level helper dedicated to explicit 1-to-0 analysis
  refresh

### 🛠️ Bug Fixes ###

**Regions of interest:**

* Fixed regions of interest created by a detection function (blob, contour, peak or Hough circle detection) reappearing immediately after being deleted or modified — detection ROIs are now ordinary, fully editable ROIs (fixes [Issue #329](https://github.com/DataLab-Platform/DataLab/issues/329))
* Fixed deleted regions of interest reappearing after a post-processing automatic recompute
* Fixed a crash when the region-of-interest index was out of bounds while building an ROI title

**Intensity profiles:**

* Fixed the "Reset selection" button of the intensity profile dialogs not clearing the previous selection: after resetting and drawing a new region, the extracted profile still corresponded to the first selection (fixes [Issue #322](https://github.com/DataLab-Platform/DataLab/issues/322))
* Fixed profile recompute using the parameters of another object when several objects were selected

**Command palette:**

* Fixed search results being irrelevant for short queries — typing "rota" no longer returns unrelated commands before the rotation entries (fixes [Issue #345](https://github.com/DataLab-Platform/DataLab/issues/345))

**Full width at half maximum:**

* A user-reported FWHM regression observed in DataLab v0.20.1 (an aberrant, far too small width) has been pinned down by a non-regression test built on the original signal, now part of Sigima's test data. The computation was already correct in v1.2.x and remains correct in this version (fixes [Issue #356](https://github.com/DataLab-Platform/DataLab/issues/356))

**Plugins:**

* Fixed plugin processing functions failing in spawned worker processes after a plugin reload

**Main window:**

* Fixed the visible plot dock not following the current panel tab in some situations, in particular when a persisted layout left the Macro, History or AI Assistant dock raised at startup

**File saving:**

* Fixed an error raised when saving a signal to a file in some situations

**History Panel:**

* Fixed replay of load actions doing nothing when the loaded objects had been deleted while recording was active — the objects are now reloaded from their files
* Fixed duplicated processing chains interfering with their source chain: replaying either session now only affects its own objects
* Fixed session replay aborting when the session contained a recorded object deletion — recorded deletions that can no longer be safely applied (captured state mismatch, targets re-created earlier in the same replay, or targets belonging to another session) are now skipped with a warning
* Removed the yellow highlight that could persist on history entries after a failed replay
* Fixed outdated steps having no visual indication in the history tree — actions left outdated by a failed or interrupted recompute, or by parameter edits pending propagation, are now highlighted with an amber background and a tooltip suggesting to replay them
* Fixed potential crashes or corrupted histories when clicking History panel commands (Replay, Delete, Duplicate, ...) while a long replay or recompute was still running — commands are now unavailable until the run completes
* Fixed a single corrupted history entry (e.g. a ROI saved by an incompatible version) preventing an entire `.dlhist` file or HDF5 workspace history from loading — the affected action now loads as incompatible while the rest of the history loads normally
* Fixed recording aborting the user's operation when a selected object had no data
* Fixed the active session remaining highlighted after loading an HDF5 file
* Fixed deleting an image history action switching to the Signal panel
* Fixed a macro generation error (`object has no attribute uuid`) for 2-to-1 processing pipelines
* Improved performance: opening many files or replaying long processing chains no longer freezes the interface while the History panel repeatedly rechecks action compatibility

### 📖 Documentation ###

* Added a complete **History Panel** feature page and a step-by-step tutorial covering recording, replay, parameter editing and session import/export
* Added **use case** pages illustrating DataLab in photonics, spectroscopy and non-destructive testing
* Documented the legacy HDF5 dataset selector syntax
* Reworked the documentation layout: more compact navigation bar, Open Graph metadata, and software citation metadata
* Added a plugin example showing iso-level contour lines on an image
* Updated French translations across all new and modified documentation pages
