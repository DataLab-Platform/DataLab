# Version 1.3 #

## DataLab Version 1.3.0 ##

### ✨ New Features ###

**Application plugins:**

* Added a top-level **Applications** entry that presents active scientific
  application plugins in a dedicated catalog
* Each application displays its description, identity, version, available
  recipes, and packaged examples; processing-only plugins remain outside this
  focused view
* The catalog is non-modal and remains open after starting a method or opening
  an example, so users may continue interacting with the DataLab workspace
* Completed application methods select their last generated object, consistently
  with standard DataLab processing

**Plugin project generator:**

* Hardened ``datalab-plugin create`` from the Camera pilot feedback: generated
  projects now separate host-independent ``core`` code, headless ``workflow``
  orchestration, and Desktop/Web ``adapters`` from the first commit
* Generated package roots expose stable identity without importing Qt; the
  installed entry point targets the Desktop adapter while Web support is
  explicitly marked unsupported
* Every generated project includes an executable architecture regression test,
  contribution and architecture documentation, and a changelog
* Long plugin names and descriptions are now formatted so newly generated
  projects pass their bundled Ruff checks without manual source edits

**Third-party plugin discovery via environment variable:**

* Added support for the `DATALAB_PLUGINS` environment variable, allowing one or more directories to be specified as additional plugin search paths
* Multiple directories can be listed using the OS path separator (`;` on Windows, `:` on Linux/macOS), following the same convention as `PYTHONPATH`
* Listed directories are appended to the existing plugin search paths at startup and are picked up automatically by the plugin discovery mechanism
* Non-existent directories are silently skipped (a warning is recorded in the log file), so a stale environment variable on another machine will not prevent DataLab from starting

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

**Large campaign responsiveness:**

* Opening, selecting, and displaying large generated campaigns no longer
  repeats a complete workspace and plot refresh for every object
* Native HDF5 workspaces and multi-object recipe results now load in groups,
  making large signal and image collections available substantially faster
  while preserving every selected and visible object

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
* The familiar **"Recompute"** action (Edit menu, `Ctrl+R`) now refreshes both
  processing *and* analysis results, giving you full control over when analyses
  are updated
* For developers, the former panel-level `recompute_analysis` entry point has
  been renamed to `recompute_selected`; `recompute_analysis` now refers to a
  different processor-level helper dedicated to explicit 1-to-0 analysis
  refresh
