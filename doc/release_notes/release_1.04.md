# Version 1.4 #

## DataLab Version 1.4.0 ##

### ✨ New Features ###

**Brightness and contrast:**

* A new **Processing > Exposure > Brightness and contrast** operation provides a histogram with synchronized minimum, maximum, brightness and contrast controls, deterministic Auto and Reset actions, ROI-aware remapping, and live preview. One input window is initialized from the first selected image and applied to the full selection, with each result preserving its source image and data type.
* Intensity windows may extend beyond the observed data range. Exact floating-point bounds survive editing and reapplication, while declared parameter constraints remain enforced and Cancel preserves the original parameters.

**Live processing preview:**

* Parameter dialogs for compatible 1-to-1 processing now offer an optional live preview for signals and images, including image-to-signal operations. Preview is off by default; Cancel leaves the workspace unchanged, while OK reuses a completed current preview for a single source and otherwise performs the normal processing on the original selection.
* Completed previews keep their dedicated process ready for later dialogs, avoiding repeated process startup. Cancelling a preview that is still running stops its process so speculative work cannot continue in the background.
* Bounded numeric parameters now offer sliders alongside precise text entry. The Processing tab retains Apply and automatic re-processing, with updates deferred until slider release and valid input.

**Application plugins:**

* Added a top-level **Applications** entry that presents active scientific application plugins in a dedicated catalog
* Each application displays its description, identity, version, available recipes, and packaged examples; processing-only plugins remain outside this focused view
* The catalog is non-modal and remains open after starting a method or opening an example, so users may continue interacting with the DataLab workspace
* Completed application methods select their last generated object, consistently with standard DataLab processing

**Plugin project generator:**

* Hardened ``datalab-plugin create`` from the Camera pilot feedback: generated projects now separate host-independent ``core`` code, headless ``workflow`` orchestration, and Desktop/Web ``adapters`` from the first commit
* Generated package roots expose stable identity without importing Qt; the installed entry point targets the Desktop adapter while Web support is explicitly marked unsupported
* Every generated project includes an executable architecture regression test, contribution and architecture documentation, and a changelog
* Long plugin names and descriptions are now formatted so newly generated projects pass their bundled Ruff checks without manual source edits

### 🔄 Changes ###

**Large campaign responsiveness:**

* Opening, selecting, and displaying large generated campaigns no longer repeats a complete workspace and plot refresh for every object
* Native HDF5 workspaces and multi-object recipe results now load in groups, making large signal and image collections available substantially faster while preserving every selected and visible object

**Portable plot annotations:**

* Plot annotations are now stored in a renderer-independent format shared with DataLab-Web, so annotations in workspaces and `.dlabann` files are no longer tied to PlotPy
* Existing PlotPy annotations remain readable and are converted only after an annotation edit is accepted; simply opening a workspace or cancelling the editor leaves its data unchanged
* Annotation identifiers, lock state, custom metadata and extension data are preserved across edits, while unknown third-party payloads are retained without modification
* DataLab now requires Sigima ≥ 1.3.0 and SigimaX ≥ 1.1.0

### 🛠️ Bug Fixes ###

**Plugins and History:**

* Replaying a History action or re-processing a result created by a plugin feature that shares its name with a built-in feature now runs the plugin feature instead of the built-in one
