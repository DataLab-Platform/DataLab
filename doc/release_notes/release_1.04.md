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

**Add metadata:**

* **Edit > Metadata > Add metadata...** can now extract the value from the formatted text with a regular expression, for example an exposure time or a shot number read from object titles. Objects without a match are left unchanged unless you ask for an error, and a scale factor converts numeric values (e.g. milliseconds to seconds)
* Metadata keys may now contain dots and hyphens, as used by plugin keys, and a **Known keys** list copies a key already present on the selected objects

**Object references in titles:**

DataLab now uses stable UUID-based object references across object trees, result titles, plot legends and HDF5 workspaces (implements [Issue #367](https://github.com/DataLab-Platform/DataLab/issues/367) and [Issue #149](https://github.com/DataLab-Platform/DataLab/issues/149)).

* Object items in the signal and image trees now show their current title followed by their own stable `#UUID8` reference; group headers keep the familiar `gsNNN` and `giNNN` identifiers
* Source references embedded in result titles no longer change when objects are reordered, removed or renumbered, and remain clickable to select the source object
* The new **References in result titles** setting lets you display these references either as 8-character UUIDs (default) or as the current source titles, which follow later renames
* When a source object is deleted, its reference falls back to its 8-character UUID instead of becoming an anonymous placeholder

### 🔄 Changes ###

**Guided tour:**

* The guided tour no longer starts automatically at first launch: it is now offered from the welcome page, and remains available from the **?** menu

**Portable plot annotations:**

* Plot annotations are now stored in a renderer-independent format shared with DataLab-Web, so annotations in workspaces and `.dlabann` files are no longer tied to PlotPy
* Existing PlotPy annotations remain readable and are converted only after an annotation edit is accepted; simply opening a workspace or cancelling the editor leaves its data unchanged
* Annotation identifiers, lock state, custom metadata and extension data are preserved across edits, while unknown third-party payloads are retained without modification
* DataLab now requires Sigima ≥ 1.3.0 and SigimaX ≥ 1.1.0

### 🛠️ Bug Fixes ###

**Window layout:**

* Fixed the width of the signal/image panels not being restored at startup: after narrowing the panels and restarting DataLab, they came back wider (up to their maximum width) instead of keeping the width chosen by the user

**Metadata:**

* Adding, pasting or deleting metadata now marks the workspace as modified, so DataLab asks to save these changes before closing

**Plugins and History:**

* Replaying a History action or re-processing a result created by a plugin feature that shares its name with a built-in feature now runs the plugin feature instead of the built-in one

**Object titles:**

* Fixed plot legends still showing the original generated title after renaming an object, e.g. a profile extracted from an image (fixes [Issue #323](https://github.com/DataLab-Platform/DataLab/issues/323))
* Fixed source references in titles being unreadable when the object is selected in the signal or image tree, notably on Linux (fixes [Issue #358](https://github.com/DataLab-Platform/DataLab/issues/358))
* Fixed result titles pointing to the wrong source objects after appending an HDF5 workspace to a non-empty session; appending the same workspace twice now also keeps each imported processing chain independent, so **Recompute** and **Select source objects** use the sources of the same import (fixes [Issue #357](https://github.com/DataLab-Platform/DataLab/issues/357))
