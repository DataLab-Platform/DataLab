# Version 1.4 #

## DataLab Version 1.4.0 ##

### Requirements ###

* The Python package and newly generated plugin projects now require Python 3.10 or later. Python 3.9 users must keep an earlier compatible release. The standalone application continues to include its own Python interpreter.
* The Windows test runner excludes Python 3.9 from current matrices, while retaining explicit tests of pinned historical releases.

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
* The **New group** option of the instrument window may be turned off to add the next acquisitions to the current group of the panel instead of creating a group for each of them. The option and the current group are taken when **Acquire** is clicked, so that selecting another group during an acquisition does not change its destination. Each instrument sets the default state of the option, and the choice of the user is remembered for each instrument
* Instruments are called outside the user interface thread, one call at a time: a slow instrument, such as a real device waiting for its data, no longer freezes DataLab, and repeated refresh requests made meanwhile are merged. Instruments may also declare `open` and `close` methods, called when their window opens and closes (including when plugins are reloaded or DataLab quits), for example to connect to a device and to release it. Calls stay one at a time when a window is closed and opened again while a call is running. When `open` fails, its message stays in the window and the instrument is opened again before the next refresh or acquisition, for example after a settings change
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

**Signal and image lists:**

Signals, images and groups can now be deleted directly from the lists, as in DataLab-Web (implements [Issue #449](https://github.com/DataLab-Platform/DataLab/issues/449)).

* The item under the mouse cursor is subtly highlighted, in light and dark modes alike (the hovered item was not highlighted at all in dark mode)
* A delete button appears at the right of the hovered signal, image or group: it turns red when the mouse cursor is over it, and its tooltip tells what it deletes
* Clicking it deletes the item, or the whole selection when the item is part of it, after confirmation. Deleting an item which is not selected keeps the current selection and the displayed objects unchanged
* As with the **Remove** action, the deletion is recorded in the History panel
* The button may be hidden with the new **Show delete button when hovering list items** setting (Visualization > Common)

**Curve fitting formulas:**

* Interactive fitting dialogs now show the formula of the fitted model below the fit parameters, and each parameter label shows its symbol in the formula (e.g. "Scale factor (x₀)"): the parameters stored in the fitted signal metadata can thus be related to the model without reading the source code. The *Fitting* documentation also gives, for each model, the name under which each parameter is stored (fixes [Issue #400](https://github.com/DataLab-Platform/DataLab/issues/400))
* Polynomial and linear fit coefficients are labelled `a`, `b`, `c`... as in the formula and in the metadata, instead of `c2`, `c1`, `c0`
* The *Fitting* documentation explains how to derive the peak wavelength, the temperature and the emissivity from the parameters of a Planckian fit

**Remote control and plugins:**

* `add_group` now returns the UUID of the created group, so that a remote client, a macro or a plugin can add objects to it with `add_object(obj, group_id=...)` (or `add_signal`/`add_image`) instead of looking the group up by its title

**NeXus files:**

* NeXus `NXdata` groups are shown as single items in the HDF5 browser and imported as signals or images with their axes, units, labels and, for signals, uncertainties. Bin edges are converted to bin centers, non-uniform image axes are kept, and decreasing image axes are reversed with the image. Invalid axes are reported in the browser instead of being guessed
* Files written with the legacy NeXus conventions, such as the scans of synchrotron SOLEIL, are imported with their axes (`axis` and `primary` attributes of the axis datasets), and every counter and detector of a scan is shown as an item of its own: counters as signals versus the scanned motor, detectors as images (a spectrum per point) or image stacks (an image per point) with the motor along their first axis
* The HDF5 browser selects and previews the default measurement of a NeXus file (`default` attributes) when it is opened
* Compressed detector data, such as the bitshuffle/LZ4 data of Eiger detectors, is now read (with hdf5plugin, also in the standalone application); a dataset compressed with an unavailable filter is reported in the browser instead of failing
* Imported signals and images record where they come from (file, group, signal and axes, applied conversions, title and times of the NeXus entry) in their `experimental_source` metadata, which is saved with the workspace: a saved workspace reopens without the NeXus file
* `.nxs` and `.nx5` files are proposed when opening HDF5 files
* Plugins can describe the signals and images of files that NeXus `NXdata` groups designate incompletely (no axes, several counters...) with declarative data mappings (JSON) registered with `sigimax.h5.mapping.register_mapping`: the HDF5 browser then shows each described object as an extra item, imported with its axes, labels and units; the axes of a mapped object may describe its first dimensions only, so that one mapping applies to the counters and detectors of a scan

### 🔄 Changes ###

**Curve readability:**

* Signal curves are easier to read, especially when many curves are shown (closes [Issue #453](https://github.com/DataLab-Platform/DataLab/issues/453)):
  * Curve colors now stand out on both light and dark backgrounds: several colors of the former palette (orange, pink, yellow-green, cyan) were pale on a white background, and gray curves could not be told apart from the grid
  * After the first 10 curves, curves are now drawn with dashed lines only: dash-dot and dash-dot-dot lines, which drew only half of the curve, are no longer used
  * The plot grid now uses discreet solid lines instead of gray dotted lines, which competed with dashed curves; it also follows color theme changes
* Curve colors and line styles now depend on the position of the curves among those shown on the plot, in the order of the signals in the panel: the first curve shown is always drawn in the first color. Opening the ROI editor, **View in a new window** or other dialogs no longer shifts the styles of the next curves, which could reach dotted styles with only a few curves shown
* Automatic curve styles are no longer saved in workspaces: only a color or a line style chosen by the user is saved and always kept, whatever its value. When opening workspaces and history files saved with DataLab 1.3 or earlier, which saved the automatic style of every curve, these styles are removed so that curves are displayed with the new automatic styles; a style chosen with these versions is removed as well when it is identical to one of their automatic styles. Metadata imported from a file are applied as they are
* **View > Reset curve styles** now gives the selected signals automatic curve styles again, removing the colors and line styles chosen for them
* On high-DPI screens (e.g. with a 150 % display scaling), curves of width 1 are now drawn one screen pixel wide, which makes their rendering much faster: noisy signals were drawn up to several hundred times slower than at 100 % (requires PlotPy ≥ 2.12.0)

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

**Regions of interest:**

* A rectangular region of interest now selects the pixels whose centers lie inside the rectangle, bounds included, like circular and polygonal ROIs and like the display: its bounds were rounded to the nearest pixel, which could shift the selection by up to half a pixel. A rectangle whose bounds lie exactly on pixel centers now includes the pixels on both bounds, i.e. one more row and one more column than before. This applies to processing and analysis in ROIs, to ROI extraction and to the background area of the offset correction
* Circular and polygonal regions of interest now also select the pixels whose centers lie inside their shape, boundary included: polygon vertices were rounded to the nearest pixel, so that the selected pixels could differ from the polygon drawn on the image, and circles were drawn in pixel units on images with non-uniform coordinates
* A rectangular ROI defined in pixel indices now covers its pixels up to their outer edges, so that it keeps the same pixels when converted to physical coordinates and back
* The cells of ROI grids (**ROI > Create ROI grid...**) are now made of whole pixels, so that adjacent cells never share a pixel and a full-size grid covers every pixel exactly once, whatever the image size; on images with non-uniform coordinates, the grid now spans the whole image, whose last rows and columns were left out
* Regions of interest saved with earlier versions are not converted: they select pixels according to these rules when they are used again, so that processing and analysis in these regions may give slightly different results when recomputed

**Image profiles in physical coordinates:**

* Line, segment, average and radial profiles (**Analysis > Intensity profiles**) are now defined in the physical coordinates of the image, like regions of interest and analysis results, instead of pixel indices: the profile parameters show coordinates in the units of the image axes, and the X axis of the resulting signals holds physical coordinates or distances with the matching unit. On images without calibration (origin 0, pixel size 1), results are unchanged
* The radial profile is now circular on images with rectangular pixels or non-uniform coordinates, and its center is correct on calibrated images, where the image center was misplaced
* Segment and radial profiles are refused on images whose X and Y units differ, since distances are then undefined; coordinates outside the image are refused instead of being silently moved to its edge. Lines and segments drawn beyond the image in the profile dialog are brought back to its outermost pixels
* Scripts, macros and plugins must use the new parameter names (`y`/`x` for line profiles, `x1`, `y1`, `x2`, `y2` for segment and average profiles, `xc`/`yc` for radial profiles) and physical coordinates. Profile computations recorded in the provenance of workspaces saved with earlier versions are reported as not verifiable, instead of being replayed with a different meaning

**HDF5 browser:**

* Browsing an HDF5 file no longer reads its datasets entirely: they are listed and described from their shape and data type, and the value previews read only their first elements. Large files open faster and use less memory; data is read when a signal or an image is previewed or imported
* The HDF5 browser stays responsive while a preview is read, and items selected meanwhile are not read; selecting an item that cannot be previewed (a group, for example) clears the preview. Ordinary 1D arrays and 2D images, NeXus and mapped objects above 2²⁴ values (about 16.8 million) are previewed with a step along each axis (the axes of NeXus images are still read entirely, to check them as on import); compound datasets and signals stored in 2D tables still use complete objects for graphical previews
* Datasets with three dimensions or more (image stacks, spectral cubes...) are now listed in the HDF5 browser and imported as one image or signal per frame: a view selector chooses the displayed axes, the frames (first, last and step along another axis) and the index of the other axes, previews the first frame, and imports the frames once **Import these frames** is checked (with a confirmation beyond 100 frames). Frames of NeXus datasets keep their axes, and their title gives their position (e.g. the motor position of each detector image of a scan)
* Dataset attributes with more than 10 000 values are no longer copied to the metadata of imported objects

See the [HDF5 browser guide](../features/common/h5browser.rst) for scientific object import, experimental origin and preview limits.

**Scripts, macros and plugins:**

* An object or a group looked up by its title, from a macro, a plugin, a remote client (e.g. `get_object("My signal")`) or the Web API, is no longer silently replaced by the first one found when several objects or groups share this title (e.g. files imported twice, "Untitled" objects): an error now names the matching objects with their UUID, which identifies them unambiguously (HTTP 409 for the Web API)
* For plugin developers: `patch_title_with_ids` no longer takes the third argument (`id_func`), and the helpers of the former `s001`-style identifiers (`SHORT_ID_REGEX`, `find_by_short_id`, `find_short_ids_in_title`, `replace_short_ids_by_uuids_in_titles`, `replace_uuids_by_short_ids_in_titles`) are removed from `datalab.objectmodel`, as titles now refer to objects by their UUID
* For developers of custom DataLab controllers derived from `BaseProxy`: the new `export_capsule` and `open_capsule` methods must be implemented

### 🛠️ Bug Fixes ###

**Performance with many objects:**

* Fixed processing becoming progressively slower as signals or images accumulate in a panel: each new result rewrote the titles of every existing object, rebuilt the whole object tree and measured every title again, so creating 1000 signals one after the other took about 39 minutes; it now takes about 3 minutes (a processing step with 1000 objects in the panel dropped from 2.3 s to about 0.3 s)
* Image panels no longer slow down because of cross-section curves kept for hidden images (requires PlotPy ≥ 2.11.2)
* Fixed titles referring to objects numbered 1000 or above (e.g., `s1000`) being corrupted when objects were added or reordered

**Window layout:**

* Fixed the width of the signal/image panels not being restored at startup: after narrowing the panels and restarting DataLab, they came back wider (up to their maximum width) instead of keeping the width chosen by the user

**Display scaling:**

* Fixed the splash screen, the DataLab logo of the welcome page and of the guided tour, and the watermark of empty plots being blurry with a Windows display scaling greater than 100% (e.g., 125%): they are now displayed at the resolution of the screen
* Fixed most of the text of the user interface (welcome page, object properties, panel tabs, dialog boxes, ...) keeping its 100% size with a Windows display scaling greater than 100% (e.g., 125%), while menus and the status bar were enlarged. All the text now follows the display scaling and the Windows text size setting, and uses the standard font of Windows applications ("Segoe UI"). This affected DataLab running with Qt 5, like the standalone application (requires guidata ≥ 3.16.1)

**Plot watermark:**

* Fixed the watermark of empty plots being read from another installation of DataLab sharing the same user configuration (e.g., DataLab installed in another Python environment): the watermark disappeared once this other installation was removed, and stayed blurry on high-DPI screens when it was an older version. The watermark of the running DataLab is now always used

**Signal display:**

* Fixed auto downsampling of signals (**Settings > Visualization > Signals**) only starting above 5 times the configured maximum number of points: with the default value of 100,000 points, signals of a few hundred thousand points were drawn in full, which slowed down the display. Signals with more points than the maximum are now downsampled so that at most this number of points is drawn, and the Settings dialog and the documentation now give the actual default value (100,000 instead of 10,000) (fixes [Issue #455](https://github.com/DataLab-Platform/DataLab/issues/455))
* Fixed changes of the auto downsampling and line width settings only applying to the curves of the selected signals: the other curves kept the former settings, even once selected again. These settings now apply to all curves as soon as the Settings dialog is closed
* Fixed the X axis ticks being shifted relative to the vertical grid lines and the curves after opening a signal or changing the selection, for signals whose X axis does not start and end on a graduation (e.g., `step.npy` from the Sigima test data): the shift remained until the view was panned or zoomed. The X axis now stays aligned with the grid and the curves (fixes [Issue #457](https://github.com/DataLab-Platform/DataLab/issues/457))

**Custom colormaps:**

* Fixed custom colormaps (created with the colormap manager) being saved outside DataLab's configuration directory, in a `.plotpy` or `.none` directory of the user's configuration directory shared with other applications based on PlotPy, depending on how DataLab was started. They are now saved in the `plotpy` subdirectory of DataLab's configuration directory; the first time, custom colormaps saved in the former directories are copied there (requires PlotPy ≥ 2.12.0)

**HDF5 import:**

* Fixed images with unsigned 32-bit values above 2 147 483 647 being clipped when imported from an HDF5 file ("Clipping uint32 data to int32" warning): such images are now imported as 64-bit floating-point images keeping every value exactly. Other integer images that no supported integer type can hold also use a sufficiently precise floating-point type when possible, instead of rounding to 32-bit floating-point values beyond 16 777 216; a warning is emitted when no supported type guarantees exact conversion (in particular beyond ±2⁵³ for float64)
* Fixed single-precision complex (`complex64`) 1D datasets losing their imaginary part when imported from an HDF5 file: they are now imported as complex signals. Integer signals beyond ±2⁵³ now also emit the precision warning instead of being rounded silently
* Fixed the import of several datasets from HDF5 files stopping at the first dataset that could not be imported: the other datasets are now imported, and the datasets that could not be are listed in a single message at the end
* Fixed HDF5 files containing broken links failing to open in the HDF5 browser, for example synchrotron master files whose detector files are not available (external links to a missing file): the broken links are now skipped and the rest of the file can be browsed and imported
* Fixed HDF5 files containing a group linked into itself or into one of its subgroups failing to open in the HDF5 browser (the file was browsed endlessly until an error): such links back to a parent group are now skipped, while other aliases of a group or dataset are still shown at each of their paths

**Workspace saving:**

* Fixed a failed save destroying the previous version of the workspace file: the file was emptied before the data was written, so that an error during the save, a full disk or an interruption left neither the previous nor the new workspace on disk. Workspaces, capsules and History files (`.dlhist`) are now written to a temporary file, which replaces the previous file only once completely written; if the save fails, the previous file is left untouched

**Metadata:**

* Adding, pasting or deleting metadata now marks the workspace as modified, so DataLab asks to save these changes before closing
* Importing metadata from a file into an object (**Edit > Metadata > Import metadata**) no longer gives it the identity of the object the metadata were exported from: two objects sharing an identity confused source references in titles, **Select source objects** and the saving of the workspace

**Processing:**

* Results of the extraction of several regions of interest into a single object can be processed again: the **Processing** tab showed "Unknown computing feature: extract_rois" with **Reapply** disabled, and replaying the History skipped the action
* **Edit > Recompute** now also recomputes results of operations combining several objects (e.g., difference of two signals, average of several images), which were skipped
* Fixed **Processing > Resampling** of images shifting every value by half a pixel: resampling an image with the default parameters, i.e. onto its own grid, filled its last row and column with NaN. The X/Y min/max parameters are now the coordinates of the centers of the first and last output pixels (by default, those of the image), and the 'Pixel size' mode keeps the requested pixel size exactly
* Fixed the sum, average, product and standard deviation of several signals, and their combination into an image, depending on the order of the signals: when a real signal came first, the imaginary parts of the complex signals were discarded
* The sum, average and product of several images now accept complex images, which raised an error
* Fixed **Processing > Resampling** of complex images with the default parameters, and the Z-axis **Polynomial calibration** of complex images, discarding the imaginary parts: complex images now stay complex
* Fixed modifying the result of a diagonal, horizontal or vertical flip or a 90° rotation of an image in place (e.g. from a macro or a plugin) also modifying its source image
* Fixed the extraction of several regions of interest into a single image giving wrong coordinates to images with non-uniform coordinates (the X coordinates were taken from the extracted rows and the Y coordinates from the extracted columns)
* The arbitrary rotation of images (**Processing > Geometry > Rotate by...**) now refuses images with non-uniform coordinates, like resizing and resampling, instead of producing an image whose coordinates did not match its pixels
* Fixed the uncertainty of the product of signals being lost where one of the signals is zero: it is now propagated there too
* Fixed ROIs, annotations and analysis results (centroids, detected peaks, contours...) being shifted by one pixel after a flip or a 90° rotation of an image; after a 90° rotation of a non-square image, or of an image whose coordinates do not start at zero, analysis results even ended up far from the features they describe. They now move exactly with the pixels; for images with non-uniform coordinates, circles, ellipses and polygons are moved approximately when the pixels are reversed along a non-uniform axis (a circle does not stay a circle). ROIs defined in pixel indices are also moved correctly, and inverse ROIs (which mask the inside of their shape) no longer become normal ROIs after these operations, a translation or a distribution of images on a grid
* Fixed **ROI > Extract...** giving the extracted image a wrong origin when the coordinates of the source image do not start at zero, and coordinates not matching the extracted pixels for images with non-uniform coordinates
* Fixed the coordinates of analysis results computed in a ROI being off by up to half a pixel
* Fixed the analysis results of images with non-uniform coordinates (**Analysis > Centroid...**, **Minimum enclosing circle center...**, **2D peak detection**, **Contour detection...**, **Circle Hough transform...** and **Blob detection**) being given in pixel indices, or according to the former pixel size and origin of an image switched to non-uniform coordinates, so that they were drawn away from the features they describe: they are now placed at the positions of the pixels on the image axes, with radii scaled by the local pixel size. The results of images with uniform coordinates are unchanged
* **ROI > Extract...** now reports an explicit error for a region of interest holding no pixel of the image (for example a cell of a ROI grid narrower than a pixel, or a region outside the image) instead of silently creating an empty image; the other regions of interest are still extracted
* Fixed **Processing > Level adjustment > Offset correction** using a background area truncated to integer coordinates: the area was wrong for images whose coordinates are not in pixels (e.g. in millimeters), and empty when smaller than one unit

**Plugins:**

* Plugins organized as a folder, such as those created by `datalab-plugin create --project`, are now loaded when their folder is added to the plugin directories, without being installed first: they were silently ignored, unless an earlier installation had left package information in the project. The project folder itself can now be added, instead of its `src` subfolder (fixes [Issue #411](https://github.com/DataLab-Platform/DataLab/issues/411))

**Plugins and History:**

* Replaying a History action or re-processing a result created by a plugin feature that shares its name with a built-in feature now runs the plugin feature instead of the built-in one

**History Panel:**

* Fixed an oversized warning appearing when deleting a group of signals or images whose processing results were kept in another group, even when the History panel was not recording: it listed every deleted object, often as an unreadable identifier, and stated that "operations could not be reconnected" although there was nothing to reconnect. Deleting such objects now shows no message, and the results depending on them are left unchanged as before. The warning is now shown only when a recorded processing step loses its source, and it lists each step once, in a message of bounded size (fixes [Issue #451](https://github.com/DataLab-Platform/DataLab/issues/451))

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
* Fixed images with non-uniform coordinates losing them when transferred through the Web API (`GET` and `PUT /objects/{name}/data`): they were rebuilt with default uniform coordinates. The NPZ data format now includes these coordinates; archives written by earlier versions remain readable. Clients with their own NPZ reader, such as DataLab-Kernel, must support the new entries to take advantage of this

**AI Assistant:**

* Fixed the **Save to Macros** link missing when a conversation is reopened from **History...**: a macro proposed by the assistant in a previous session could no longer be saved. Reopened conversations also showed every tool call as successful, including the failed ones; they now show the actual outcome (fixes [Issue #447](https://github.com/DataLab-Platform/DataLab/issues/447))
* Saving a macro proposed by the assistant now offers a **Show in Macro panel** link: the Macro panel is tabbed with the AI Assistant, so the saved macro stayed hidden behind the conversation
* Fixed the assistant answering a request to *write* a macro with the code in its reply, which could not be saved to the Macro panel: it now proposes the macro without running it (the confirmation dialog states that it will not be executed) and offers the **Save to Macros** link. A **Save to Macros** link is also offered under each Python code block of its replies, unless macro creation is disabled

**Security:**

* Opening a workspace received from someone else can no longer make DataLab load or run other code installed on the computer. The parameters stored with objects, analysis results and History actions name their class, and DataLab imported whatever module they named and called the object named there, which a modified file could use, for example, to make DataLab quit. Only parameter classes of DataLab, Sigima, guidata and installed plugins are now used; other stored parameters are reported as unreadable (fixes [Issue #407](https://github.com/DataLab-Platform/DataLab/issues/407))
