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
* Instruments are called outside the user interface thread, one call at a time: a slow instrument, such as a real device waiting for its data, no longer freezes DataLab, and repeated refresh requests made meanwhile are merged. Instruments may also declare `open` and `close` methods, called when their window opens and closes (including when plugins are reloaded or DataLab quits), for example to connect to a device and to release it
* Plugin developers describe recipe inputs declaratively (titles, minimum counts, metadata requirements) and may add binding suggestions and fast input checks; every recipe then runs through DataLab's generic launcher, without plugin-specific input dialogs
* The catalog is non-modal and remains open after starting a method or opening an example, so users may continue interacting with the DataLab workspace
* The methods, tools and datasets of an application form an accordion: one section is open at a time, and a colored dot tells, without opening it, whether each method can run on the current selection
* The application list of the catalog may be hidden with the strip separating it from the application page: the window shrinks accordingly, and the choice is remembered
* Completed application methods select their last generated object, consistently with standard DataLab processing
* Application plugins add tiles to a new **Applications** section of the welcome page: by default, one tile per plugin opens its page in the catalog, and plugins may declare additional tiles, for example to open an example directly. Additional tiles are shown when there is room for them, and otherwise move to the menu of the main tile
* The welcome page stays tidy when many applications are installed: tiles are limited to a configurable number of rows (two by default), with a last tile opening the catalog for the others. Applications may be pinned to the top or hidden, and recently used ones come first; the catalog offers a search field and the same welcome page options
* Plugins may declare an icon, shown in the **Applications** catalog, on their welcome page tile and in the **Configure plugins...** dialog

**Plugin installation:**

* Online and offline Desktop catalogs now accept single-file `datalab_*.py` plugins alongside wheels. Historical files do not need a packaging project or source changes to be listed; catalog inspection never imports or executes them (implements [Issue #419](https://github.com/DataLab-Platform/DataLab/issues/419))
* Known script versions can be updated; unversioned files are identified by SHA-256 and offer explicit replacement when their content changes. Scripts previously installed from a file are recognized, and existing development copies are not overwritten
* The local catalog commands handle mixed folders of scripts and wheels. Single-file plugins use libraries already available to DataLab and do not trigger dependency downloads. The documentation now explicitly distinguishes source organization, distribution formats and functional capabilities, including in the French offline guide
* Plugins can now be installed from a file, without any Python tool, including in the standalone version of DataLab: the new **Install plugins** tab of **Plugins > Configure plugins...** accepts a pure-Python wheel (`.whl`) or a single `datalab_<name>.py` module
* Before installing, DataLab checks the file without running it and shows its name, version, plugin classes, dependencies and SHA-256 digest
* Plugins may now include additional dependencies as separate wheels, including compatible native libraries, without changing packages already supplied by DataLab. A complete local catalog can be installed without Internet access; missing files and version conflicts are reported before any partial installation is activated
* The confirmation dialog lists the additional packages and their fingerprints. They are shared by plugins of the same Python runtime, activated after restarting DataLab, and kept when a plugin is uninstalled
* Installed plugins are listed in the same tab, where they can be uninstalled; a newly installed plugin is enabled after a plugin reload when no additional packages are needed, or after restarting when dependencies were added. A new version of an already loaded plugin is used at the next start
* The new **Available plugins** tab lists the plugins of the [DataLab plugin catalog](https://datalab-platform.com/plugins/), official or from the community, with a search field: **Install** and **Update** download the wheel, check it against the catalog and install it after confirmation. The catalog address may be changed to use a private copy
* Organizations can distribute their own plugins, or the versions of public plugins they have approved, from local plugin catalogs: folders, typically shared on the network, declared in the **Plugin settings** tab or with the `DATALAB_PLUGINS_CATALOG` environment variable. Their plugins are listed in the **Available plugins** tab before those of the DataLab plugin catalog, and hide them when both offer the same plugin
* A local catalog folder may hold plugin wheels and single-file modules, listed as they are copied to it, or a `catalog.json` index giving display names, deprecations and withdrawn releases; `datalab-plugin catalog build` writes this index while keeping curated metadata, and `datalab-plugin catalog check` reports missing, changed, refused or unlisted artifacts
* Authors can prepare an offline catalog with `datalab-plugin catalog prepare`: it downloads the additional dependency wheels needed by the reference host without installing them, then writes the catalog. The complete folder can be distributed on a network share or copied to computers without Internet access
* A catalog that cannot be read, such as a network share that is not connected or the online catalog on a computer without Internet access, is reported in the **Available plugins** tab without hiding the plugins of the other catalogs

**Plugin project generator:**

* `datalab-plugin create` now creates a single-file plugin by default, the simplest way to extend DataLab: it only asks for the plugin name and description, then writes a `datalab_<name>.py` file to the user plugin directory. Once the plugins are reloaded, the plugin adds its submenu to the **Plugins** menu with a sample processing of the selected signals (or images, with `--object-kind image`), whose short NumPy function is the only part to edit. The file never overwrites an existing one, and can be shared and installed with **Plugins > Configure plugins... > Install plugins**. The layered installable project, described below, is created with `--project` (implements [Issue #413](https://github.com/DataLab-Platform/DataLab/issues/413))
* Hardened ``datalab-plugin create`` from the Camera pilot feedback: generated projects now separate host-independent ``core`` code, headless ``workflow`` orchestration, and Desktop/Web ``adapters`` from the first commit
* Generated package roots expose stable identity without importing Qt; the installed entry point targets the Desktop adapter while Web support is explicitly marked unsupported
* Every generated project includes an executable architecture regression test, contribution and architecture documentation, and a changelog
* Long plugin names and descriptions are now formatted so newly generated projects pass their bundled Ruff checks without manual source edits
* Generated projects are ready to publish: a release workflow attaches the wheel to a GitHub release when a version tag is pushed and prints the entry to submit to the DataLab plugin catalog; `pyproject.toml` requires the current DataLab major version and declares the project URLs
* `datalab-plugin create --project` asks for the GitHub account hosting the project and derives the default plugin ID from it (`io.github.<account>.<name>`), `org.datalab.` IDs being reserved for DataLab-Platform plugins

**Add metadata:**

* **Edit > Metadata > Add metadata...** can now extract the value from the formatted text with a regular expression, for example an exposure time or a shot number read from object titles. Objects without a match are left unchanged unless you ask for an error, and a scale factor converts numeric values (e.g. milliseconds to seconds)
* Metadata keys may now contain dots and hyphens, as used by plugin keys, and a **Known keys** list copies a key already present on the selected objects

**Workspace provenance:**

* DataLab now records how each result was obtained: the processing applied, its parameters, the objects it used and their exact content, and the software versions. This record is saved with the HDF5 workspace and read back when the workspace is reopened, by DataLab or by DataLab-Web, so the history of a result survives saving, closing and sharing
* Signal and image processing, operations combining several objects and analyses (geometry and table results) are recorded, each with the identity of its Sigima operation; regions of interest, uncertainties, the X alignment applied to signals on different grids and the global options read by the computation (such as kernel normalization) are part of the record, and signals and images of every numeric type (float32, complex...) are covered. An analysis that finds nothing is recorded as such. A result whose data, calibration or regions of interest were changed after it was computed is reported as altered rather than trusted
* Any recorded processing or analysis of a qualified Sigima operation (all of them but the phase spectrum) can be recomputed and every result compared with the stored one: exactly, or within the tolerance declared for the operation when the numerical libraries differ (exact, within tolerance, different, or not verified with the reason). Recomputation is available from the **Provenance** tab of the Properties panel (**Verify computation**) and from scripts; it never changes the workspace. When only the regions of interest of an input changed since the computation (for example regions created on the analysed image by a detection), it is recomputed on a copy carrying the recorded regions of interest, so that such an analysis remains verifiable
* The **History** tab of the Properties panel is renamed **Provenance** (to avoid any confusion with the History panel) and now shows the provenance of the selected object: the recorded execution that produced its current content, its parameters and inputs, and, step by step, how each input was itself produced, as it was at the time (e.g. the sigma actually used, even if the input was filtered again since), with the software that recorded it (DataLab Desktop or DataLab Web). Each step is a card with its parameters, inputs and results, and links next to what they act on: show the recorded step of an input, **Go to object**, **Verify this step**, **History action** and details. **Check data** compares the object, its direct inputs and the earlier steps shown with their recorded states without computing; deleted inputs are reported at once, modified or unavailable ones after checking. Going to the History action of a step keeps the Provenance tab open
* The **Processing** tab restores the recorded call of the selected result, including operations combining several objects (difference of two signals, average of several images, ...): every input is kept in order with its role, parameters can be edited, and processing without parameters offers a **Reapply** button. When the processing cannot be applied again (e.g. an input was deleted), the tab says why. This also works for workspaces reopened from a file, saved by DataLab or by DataLab-Web
* History actions are linked to the provenance of their last successful execution: **Show execution provenance** in the History panel displays it, and the provenance of a result leads to the History action that produced it, analyses included. In the History panel edit mode, editing one result of a command producing several results changes only its parameters in the action
* When a workspace file holding provenance is added to the current workspace (without clearing it), DataLab now says before anything changes that its earlier provenance will not be imported: in the question about clearing the workspace (files are chosen first), or in a single warning when that question is not asked. Added objects never inherit the provenance of another object (they are neither shown nor used as the original input of a recorded step, including when verifying it), and their later processing is recorded
* DataLab Desktop and DataLab-Web record the same processing identically and replay each other's results
* Before interpolating signals whose X grids differ, DataLab still asks for confirmation; the alignment it applies is the one it has always applied, now named and recorded so that a recomputation applies it identically. Several selected pairs of a two-signal operation now each align the operand on the grid of their own source signal
* Workspaces saved by earlier versions keep their processing history as stored: DataLab shows what is known and what is missing, invents nothing, and never recomputes such entries
* A workspace whose provenance record is damaged is refused with an explicit message, and the current workspace is left unchanged
* Workspaces can be exchanged as capsules (a workspace plus a description of its provenance) from **File > Workspace capsule...**, the same dialog as in DataLab-Web, so that someone else can check and replay them with either edition:
  * **Export** the workspace, with a name, a description, light attachments (protocol, notes, small tables) and, for a workspace opened from a capsule (also after it was saved and opened again), the capsule it is a revision of; the workspace can also be referenced by its publication address instead of being included
  * **Inspect** a capsule without opening it: its activities, its attachments (saved only on request, never opened), its integrity (workspace and stored states checked, nothing recomputed) and the verification reports and reviews supplied with it, each checked against the capsule; write a review of it, or **Open as workspace** (recorded in the History panel)
  * **Local replay** recomputes a recorded activity of the open workspace and compares it with its stored result, without changing the workspace; the verification report can be saved to be shared with the capsule
  * **Revisions** lists the capsules exported, inspected or retrieved on this computer and compares any two of them (scientific, presentation, storage, environment and attachment changes)
  * **Registry** connects to an optional capsule registry to publish capsules (never their workspaces) and reviews, list the published revisions and retrieve them, everything returned being checked
  * Scripts and macros can export and open capsules too (`export_capsule` and `open_capsule` of the proxy), and the `workspace-capsule` command line of [DataLab-Capsule](https://github.com/DataLab-Platform/DataLab-Capsule) still validates, compares and verifies them
  * A capsule, or the workspace it references, that exceeds the size limit is refused before being read, and DataLab never writes a capsule it would refuse to open, nor writes the capsule or its referenced workspace over each other or over an attachment; the limits and the measured costs are given in the *Workspace capsules* page
  * The standalone version (installer) embeds what capsules need (schemas and RO-Crate context of DataLab-Capsule)

**Object references in titles:**

DataLab now uses stable UUID-based object references across object trees, result titles, plot legends and HDF5 workspaces (implements [Issue #367](https://github.com/DataLab-Platform/DataLab/issues/367) and [Issue #149](https://github.com/DataLab-Platform/DataLab/issues/149)).

* Signal, image and group items in the trees now show their current title followed by their own stable `#UUID8` reference (`#gUUID8` for groups), replacing the former `s001`, `i001` and `gs001` identifiers
* Source references embedded in result titles no longer change when objects are reordered, removed or renumbered, and remain clickable to select the source object
* The new **References in result titles** setting lets you display these references either as 8-character UUIDs (default) or as the current source titles, which follow later renames and are resolved through whole processing chains (e.g. `derivative(derivative(My signal))`)
* When a source object is deleted, its reference falls back to its 8-character UUID instead of becoming an anonymous placeholder
* Titles of workspaces saved by previous versions keep their former `s001`-style references as plain text

**Curve fitting formulas:**

* Interactive fitting dialogs now show the formula of the fitted model below the fit parameters, and each parameter label shows its symbol in the formula (e.g. "Scale factor (x₀)"): the parameters stored in the fitted signal metadata can thus be related to the model without reading the source code. The *Fitting* documentation also gives, for each model, the name under which each parameter is stored (fixes [Issue #400](https://github.com/DataLab-Platform/DataLab/issues/400))
* Polynomial and linear fit coefficients are labelled `a`, `b`, `c`... as in the formula and in the metadata, instead of `c2`, `c1`, `c0`
* The *Fitting* documentation explains how to derive the peak wavelength, the temperature and the emissivity from the parameters of a Planckian fit

**Remote control and plugins:**

* `add_group` now returns the UUID of the created group, so that a remote client, a macro or a plugin can add objects to it with `add_object(obj, group_id=...)` (or `add_signal`/`add_image`) instead of looking the group up by its title

**NeXus files:**

* NeXus `NXdata` groups are shown as single items in the HDF5 browser and imported as signals or images with their axes, units, labels and, for signals, uncertainties. Bin edges are converted to bin centers, and non-uniform image axes are kept. Invalid axes are reported in the browser instead of being guessed
* Imported signals and images record where they come from (file, group, signal and axes, applied conversions) in their `experimental_source` metadata, which is saved with the workspace: a saved workspace reopens without the NeXus file
* `.nxs` and `.nx5` files are proposed when opening HDF5 files
* Plugins can describe the signals and images of files that NeXus `NXdata` groups designate incompletely (no axes, several counters...) with declarative data mappings (JSON) registered with `sigimax.h5.mapping.register_mapping`: the HDF5 browser then shows each described object as an extra item, imported with its axes, labels and units

### 🔄 Changes ###

**Synthetic images in physical coordinates:**

* The 2D Gaussian, 2D ramp, sinusoidal grating, ring pattern, Siemens star and 2D sinc images were computed over the [x<sub>min</sub>, x<sub>max</sub>] × [y<sub>min</sub>, y<sub>max</sub>] domain of their creation parameters, but were displayed in pixel coordinates: their centers, widths and spatial frequencies did not match the axes (a sinusoidal grating of frequency 0.1 over [0, 100] showed 10 periods along an axis graduated from 0 to 1023). These images are now displayed over their domain. The new **Physical coordinates** option of their creation parameters restores pixel coordinates when disabled (fixes [Issue #400](https://github.com/DataLab-Platform/DataLab/issues/400))
* Images created with earlier versions keep their pixel coordinates when their creation parameters are edited or recomputed

**Guided tour:**

* The guided tour no longer starts automatically at first launch: it is now offered from the welcome page, and remains available from the **?** menu

**Large campaign responsiveness:**

* Opening, selecting, and displaying large generated campaigns no longer repeats a complete workspace and plot refresh for every object
* Native HDF5 workspaces and multi-object recipe results now load in groups, making large signal and image collections available substantially faster while preserving every selected and visible object

**Portable plot annotations:**

* Plot annotations are now stored in a renderer-independent format shared with DataLab-Web, so annotations in workspaces and `.dlabann` files are no longer tied to PlotPy
* Existing PlotPy annotations remain readable and are converted only after an annotation edit is accepted; simply opening a workspace or cancelling the editor leaves its data unchanged
* Annotation identifiers, lock state, custom metadata and extension data are preserved across edits, while unknown third-party payloads are retained without modification
* DataLab now requires Sigima ≥ 1.4.0, SigimaX ≥ 1.1.0 and DataLab-Capsule 0.1.x; HDF5 imports use Sigima's experimental-source API

**HDF5 browser:**

* Browsing an HDF5 file no longer reads its datasets entirely: they are listed and described from their shape and data type, and the value previews read only their first elements. Large files open faster and use less memory; data is read when a signal or an image is previewed or imported
* The HDF5 browser stays responsive while a preview is read. Ordinary 1D arrays and 2D images above 2²⁴ values (about 16.8 million) are previewed with a step along each axis; NeXus and mapped objects, compound datasets and signals stored in 2D tables still use complete objects for graphical previews
* Datasets with three dimensions or more (image stacks, spectral cubes...) are now listed in the HDF5 browser, but are not directly importable and have no graphical slice selector
* Dataset attributes with more than 10 000 values are no longer copied to the metadata of imported objects

See the [HDF5 browser guide](../features/common/h5browser.rst) for scientific object import, experimental origin and preview limits.

### 🛠️ Bug Fixes ###

**Performance with many objects:**

* Fixed processing becoming progressively slower as signals or images accumulate in a panel: each new result rewrote the titles of every existing object, rebuilt the whole object tree and measured every title again, so creating 1000 signals one after the other took about 39 minutes; it now takes about 3 minutes (a processing step with 1000 objects in the panel dropped from 2.3 s to about 0.3 s)
* Image panels no longer slow down because of cross-section curves kept for hidden images (requires PlotPy ≥ 2.11.2)
* Fixed titles referring to objects numbered 1000 or above (e.g., `s1000`) being corrupted when objects were added or reordered

**Window layout:**

* Fixed the width of the signal/image panels not being restored at startup: after narrowing the panels and restarting DataLab, they came back wider (up to their maximum width) instead of keeping the width chosen by the user

**HDF5 import:**

* Fixed images with unsigned 32-bit values above 2 147 483 647 being clipped when imported from an HDF5 file ("Clipping uint32 data to int32" warning): such images are now imported as 64-bit floating-point images keeping every value exactly. Other integer images that no supported integer type can hold also use a sufficiently precise floating-point type when possible, instead of rounding to 32-bit floating-point values beyond 16 777 216; a warning is emitted when no supported type guarantees exact conversion (in particular beyond ±2⁵³ for float64)

**Metadata:**

* Adding, pasting or deleting metadata now marks the workspace as modified, so DataLab asks to save these changes before closing
* Importing metadata from a file into an object (**Edit > Metadata > Import metadata**) no longer gives it the identity of the object the metadata were exported from: two objects sharing an identity confused source references in titles, **Select source objects** and the saving of the workspace

**Processing:**

* Results of the extraction of several regions of interest into a single object can be processed again: the **Processing** tab showed "Unknown computing feature: extract_rois" with **Reapply** disabled, and replaying the History skipped the action
* **Edit > Recompute** now also recomputes results of operations combining several objects (e.g., difference of two signals, average of several images), which were skipped

**Plugins:**

* Plugins organized as a folder, such as those created by `datalab-plugin create --project`, are now loaded when their folder is added to the plugin directories, without being installed first: they were silently ignored, unless an earlier installation had left package information in the project. The project folder itself can now be added, instead of its `src` subfolder (fixes [Issue #411](https://github.com/DataLab-Platform/DataLab/issues/411))

**Plugins and History:**

* Replaying a History action or re-processing a result created by a plugin feature that shares its name with a built-in feature now runs the plugin feature instead of the built-in one

**Object titles:**

* Fixed plot legends still showing the original generated title after renaming an object, e.g. a profile extracted from an image (fixes [Issue #323](https://github.com/DataLab-Platform/DataLab/issues/323))
* Fixed source references in titles being unreadable when the object is selected in the signal or image tree, notably on Linux (fixes [Issue #358](https://github.com/DataLab-Platform/DataLab/issues/358))
* Fixed result titles pointing to the wrong source objects after appending an HDF5 workspace to a non-empty session; appending the same workspace twice now also keeps each imported processing chain independent, so **Recompute** and **Select source objects** use the sources of the same import (fixes [Issue #357](https://github.com/DataLab-Platform/DataLab/issues/357))

**Curve fitting:**

* Fixed the automatic Planckian fit (**Processing > Fitting > Planckian fit**) returning a curve about 150 times lower than the signal on data in physical units, such as a blackbody signal with wavelengths in meters; the interactive Planckian fit dialog also starts from a curve matching the signal peak (fixes [Issue #400](https://github.com/DataLab-Platform/DataLab/issues/400))
* Fixed exponential fitting failing on short time constants depending on the X unit, e.g. a 1 ms time constant with X in seconds (automatic and interactive fits)

**Signal creation:**

* The blackbody signal (**Create > Blackbody (Planck's law)**) now has axis titles and units (wavelength in m, spectral radiance in W·sr⁻¹·m⁻³), which were missing (fixes [Issue #400](https://github.com/DataLab-Platform/DataLab/issues/400))

**Remote control:**

* Fixed files loaded remotely with `load_from_files` or `load_from_directory` being sometimes silently ignored, or still loading after the call had returned, when requested right after DataLab started: requests received while DataLab is starting are now processed as soon as its main window is ready (fixes [Issue #398](https://github.com/DataLab-Platform/DataLab/issues/398))
* Errors raised by remote commands, such as `calc` with an unknown computation function or `set_object` with an unknown object, are now reported to the client instead of being silently ignored
* Fixed remote commands sometimes returning the result of the previous command instead of their own: for example, `get_current_object_uuid` could return a panel name instead of `None` when there was no current object
* Fixed DataLab freezing when closed while a remote command was still running (e.g. when closing the main window with `call_method("close")`)

**Security:**

* Opening a workspace received from someone else can no longer make DataLab load or run other code installed on the computer. The parameters stored with objects, analysis results and History actions name their class, and DataLab imported whatever module they named and called the object named there, which a modified file could use, for example, to make DataLab quit. Only parameter classes of DataLab, Sigima, guidata and installed plugins are now used; other stored parameters are reported as unreadable (fixes [Issue #407](https://github.com/DataLab-Platform/DataLab/issues/407))
