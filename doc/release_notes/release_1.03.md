# Version 1.3 #

## DataLab Version 1.3.0 ##

### ✨ New Features ###

**Third-party plugin discovery via environment variable:**

* Added support for the `DATALAB_PLUGINS` environment variable, allowing one or more directories to be specified as additional plugin search paths
* Multiple directories can be listed using the OS path separator (`;` on Windows, `:` on Linux/macOS), following the same convention as `PYTHONPATH`
* Listed directories are appended to the existing plugin search paths at startup and are picked up automatically by the plugin discovery mechanism
* Non-existent directories are silently skipped (a warning is recorded in the log file), so a stale environment variable on another machine will not prevent DataLab from starting

**History Panel sessions:**

* Added serialized and replayable history sessions with workspace-state validation
* Added `.dlhist` import/export support and separated reset sessions from regular history sessions
* Improved replay compatibility reporting for clearer user feedback
* The Delete key now removes the selected entries in the History panel (same behavior as in the Signal and Image panels), and only applies while the history tree has focus
* Added a resolution dialog when replaying a processing chain broken by deleted objects: choose "Repair and continue" to prune the broken steps and replay the remaining valid ones, or cancel without modifying anything

**Replace special values processing (signal and image):**

DataLab now provides a dedicated **"Replace special values"** processing
function that detects and replaces `NaN`, `+Inf` and `-Inf` values in both
signals and images. The feature is available under
**Processing → Level adjustment → Replace special values** in both the Signal
and Image panels.

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

**Detection tools now preserve existing regions of interest:**

* Regions of interest created by peak, blob, Hough circle, and contour
  detection are now added to existing ROIs instead of replacing them, without
  displaying a destructive confirmation dialog

**Analysis results are now refreshed on demand instead of automatically:**

* Analysis results (statistics, FWHM, centroid, peak/contour/blob detection,
  etc.) are no longer recomputed automatically when you modify a region of
  interest, transform the data, or edit object properties
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

**History Panel:**

* Fixed replay of load actions doing nothing when the loaded objects had been deleted while recording was active — the objects are now reloaded from their files
* Fixed duplicated processing chains interfering with their source chain: replaying either session now only affects its own objects
* Fixed session replay aborting when the session contained a recorded object deletion — recorded deletions that can no longer be safely applied (captured state mismatch, targets re-created earlier in the same replay, or targets belonging to another session) are now skipped with a warning
* Removed the yellow highlight that could persist on history entries after a failed replay
* Fixed outdated steps having no visual indication in the history tree — actions left outdated by a failed or interrupted recompute, or by parameter edits pending propagation, are now highlighted with an amber background and a tooltip suggesting to replay them
* Fixed potential crashes or corrupted histories when clicking History panel commands (Replay, Delete, Duplicate, ...) while a long replay or recompute was still running — commands are now unavailable until the run completes
* Fixed a single corrupted history entry (e.g. a ROI saved by an incompatible version) preventing an entire `.dlhist` file or HDF5 workspace history from loading — the affected action now loads as incompatible while the rest of the history loads normally
* Fixed recording aborting the user's operation when a selected object had no data
* Improved performance: opening many files or replaying long processing chains no longer freezes the interface while the History panel repeatedly rechecks action compatibility
