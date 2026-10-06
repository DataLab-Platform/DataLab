# Version 1.4 #

## DataLab Version 1.4.0 ##

### ✨ New Features ###

**Welcome page:**

* A new **Welcome** page, similar to the one of DataLab-Web, is shown at startup and whenever the current signal or image panel is empty. It gathers the main actions to get started: create a signal or an image, open files, browse an HDF5 file, open an HDF5 workspace, import text data, ask the AI Assistant, take the guided tour, run the demo, read the documentation or browse the release notes
* The page is a dockable panel, tabbed with the signal and image views: it remains available after data is created or opened, and can be reopened at any time from the **?** menu (**Welcome page**) or from the **View** menu
* A **Show welcome page when the current panel is empty** option, available on the page itself and in the General settings, controls whether the page is shown automatically, including at startup

**Brightness and contrast:**

* A new **Processing > Exposure > Brightness and contrast** operation provides a histogram with synchronized minimum, maximum, brightness and contrast controls, deterministic Auto and Reset actions, ROI-aware remapping, and live preview. One input window is initialized from the first selected image and applied to the full selection, with each result preserving its source image and data type.
* Intensity windows may extend beyond the observed data range. Exact floating-point bounds survive editing and reapplication, while declared parameter constraints remain enforced and Cancel preserves the original parameters.

**Live processing preview:**

* Parameter dialogs for compatible 1-to-1 processing now offer an optional live preview for signals and images, including image-to-signal operations. Preview is off by default; Cancel leaves the workspace unchanged, while OK reuses a completed current preview for a single source and otherwise performs the normal processing on the original selection.
* Completed previews keep their dedicated process ready for later dialogs, avoiding repeated process startup. Cancelling a preview that is still running stops its process so speculative work cannot continue in the background.
* Bounded numeric parameters now offer sliders alongside precise text entry. The Processing tab retains Apply and automatic re-processing, with updates deferred until slider release and valid input.

### 🔄 Changes ###

**Guided tour:**

* The guided tour no longer starts automatically at first launch: it is now offered from the welcome page, and remains available from the **?** menu

**Portable plot annotations:**

* Plot annotations are now stored in a renderer-independent format shared with DataLab-Web, so annotations in workspaces and `.dlabann` files are no longer tied to PlotPy
* Existing PlotPy annotations remain readable and are converted only after an annotation edit is accepted; simply opening a workspace or cancelling the editor leaves its data unchanged
* Annotation identifiers, lock state, custom metadata and extension data are preserved across edits, while unknown third-party payloads are retained without modification
* DataLab now requires Sigima ≥ 1.3.0 and SigimaX ≥ 1.1.0

### 🛠️ Bug Fixes ###

**Plugins and History:**

* Replaying a History action or re-processing a result created by a plugin feature that shares its name with a built-in feature now runs the plugin feature instead of the built-in one
