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

**Application plugins:**

* Added a top-level **Applications** entry that presents active scientific application plugins in a dedicated catalog
* Each application displays its description, identity, version, methods, tools, and examples; processing-only plugins remain outside this focused view
* Each method shows the inputs it expects (object type and count, required metadata) and a live status telling whether the current selection can be analyzed, and why not
* When required metadata are missing, the status points to **Edit > Metadata > Add metadata...**, whose **Known keys** list also offers the keys expected by application methods; the status is updated as soon as objects are modified
* **Run on selection...** starts a method directly when the selection is ready, and asks for the input assignment only when it is ambiguous or invalid; the assignment dialog checks the chosen objects before the run can continue
* Examples are listed under the methods they are designed for: **Try with this example** opens the example, prefills the method parameters, and runs the method once they are accepted. One example may serve several methods, and examples designed for no method are listed as datasets
* Application plugins may also list tools, such as wizards or editors, in the catalog
* Plugin tools also appear in the plugin's submenu of the **Plugins** menu, in the Signal or Image panel they work on. A tool that needs a selection is disabled until the right objects are selected, and its catalog button tells what to select
* Application plugins may provide instruments, such as simulated cameras or oscilloscopes, without writing any user interface: DataLab shows a window with a live view and the instrument settings, refreshed as settings change or continuously in **Live** mode, and **Acquire** adds the acquired objects to a new group, ready for the application's methods
* Plugin developers describe recipe inputs declaratively (titles, minimum counts, metadata requirements) and may add binding suggestions and fast input checks; every recipe then runs through DataLab's generic launcher, without plugin-specific input dialogs
* The catalog is non-modal and remains open after starting a method or opening an example, so users may continue interacting with the DataLab workspace
* The methods, tools and datasets of an application form an accordion: one section is open at a time, and a colored dot tells, without opening it, whether each method can run on the current selection
* The application list of the catalog may be hidden with the strip separating it from the application page: the window shrinks accordingly, and the choice is remembered
* Completed application methods select their last generated object, consistently with standard DataLab processing
* Application plugins add tiles to a new **Applications** section of the welcome page: by default, one tile per plugin opens its page in the catalog, and plugins may declare additional tiles, for example to open an example directly. Additional tiles are shown when there is room for them, and otherwise move to the menu of the main tile
* The welcome page stays tidy when many applications are installed: tiles are limited to a configurable number of rows (two by default), with a last tile opening the catalog for the others. Applications may be pinned to the top or hidden, and recently used ones come first; the catalog offers a search field and the same welcome page options
* Plugins may declare an icon, shown in the **Applications** catalog, on their welcome page tile and in the **Configure plugins...** dialog

**Plugin installation:**

* Plugins can now be installed from a file, without any Python tool, including in the standalone version of DataLab: the new **Install plugins** tab of **Plugins > Configure plugins...** accepts a pure-Python wheel (`.whl`) or a single `datalab_<name>.py` module
* Before installing, DataLab checks the file without running it and shows its name, version, plugin classes, dependencies and SHA-256 digest; a wheel requiring a package that DataLab does not provide is refused
* Installed plugins are listed in the same tab, where they can be uninstalled; a newly installed plugin is enabled and loaded after a plugin reload, and a new version of an already loaded plugin is used at the next start
* The new **Available plugins** tab lists the plugins of the [DataLab plugin catalog](https://datalab-platform.com/plugins/), official or from the community, with a search field: **Install** and **Update** download the wheel, check it against the catalog and install it after confirmation. The catalog address may be changed to use a private copy

**Plugin project generator:**

* Hardened ``datalab-plugin create`` from the Camera pilot feedback: generated projects now separate host-independent ``core`` code, headless ``workflow`` orchestration, and Desktop/Web ``adapters`` from the first commit
* Generated package roots expose stable identity without importing Qt; the installed entry point targets the Desktop adapter while Web support is explicitly marked unsupported
* Every generated project includes an executable architecture regression test, contribution and architecture documentation, and a changelog
* Long plugin names and descriptions are now formatted so newly generated projects pass their bundled Ruff checks without manual source edits
* Generated projects are ready to publish: a release workflow attaches the wheel to a GitHub release when a version tag is pushed and prints the entry to submit to the DataLab plugin catalog; `pyproject.toml` requires the current DataLab major version and declares the project URLs
* `datalab-plugin create` asks for the GitHub account hosting the project and derives the default plugin ID from it (`io.github.<account>.<name>`), `org.datalab.` IDs being reserved for DataLab-Platform plugins

**Add metadata:**

* **Edit > Metadata > Add metadata...** can now extract the value from the formatted text with a regular expression, for example an exposure time or a shot number read from object titles. Objects without a match are left unchanged unless you ask for an error, and a scale factor converts numeric values (e.g. milliseconds to seconds)
* Metadata keys may now contain dots and hyphens, as used by plugin keys, and a **Known keys** list copies a key already present on the selected objects

### 🔄 Changes ###

**Guided tour:**

* The guided tour no longer starts automatically at first launch: it is now offered from the welcome page, and remains available from the **?** menu

**Large campaign responsiveness:**

* Opening, selecting, and displaying large generated campaigns no longer repeats a complete workspace and plot refresh for every object
* Native HDF5 workspaces and multi-object recipe results now load in groups, making large signal and image collections available substantially faster while preserving every selected and visible object

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
