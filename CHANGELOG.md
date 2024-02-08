# Changelog #

See DataLab [roadmap page](https://codra-ingenierie-informatique.github.io/DataLab/en/dev/roadmap.html)
for future and past milestones.

## DataLab Version 0.12.0 ##

🧹 Clarity-Enhanced Interface Update:

* The tabs used to switch between the data panels (signals and images) and the
  visualization components ("Curve panel" and "Image panel") have been renamed
  to "Signal Panel" and "Image Panel" (instead of "Signals" and "Images")
* The visualization components have been renamed to "Signal View" and "Image View"
  (instead of "Curve panel" and "Image panel")
* The data panel toolbar has been renamed to "Signal Toolbar" and "Image Toolbar"
  (instead of "Signal Processing Toolbar" and "Image Processing Toolbar")
* Ergonomics improvements: the "Signal Panel" and "Image Panel" are now displayed
  on the left side of the main window, and the "Signal View" and "Image View" are
  displayed on the right side of the main window. This reduces the distance between
  the list of objects (signals and images) and the associated actions (toolbars and
  menus), and makes the interface more intuitive and easier to use

✨ New tour and demo feature:

* When starting DataLab for the first time, an optional tour is now shown to the user
  to introduce the main features of the application
* The tour can be started again at any time from the "?" menu
* Also added a new "Demo" feature to the "?" menu

🚀 New Binder environment to test DataLab online without installing anything

📚 Documentation:

* New text tutorials are available:
  * Measuring Laser Beam Size
  * DataLab and Spyder: a perfect match
* "Getting started" section: added more explanations and links to the tutorials
* New "Contributing" section explaining how to contribute to DataLab, whether you
  are a developer or not
* Added "Copy" button to code blocks in the documentation

💥 New features:

* Added menu on the "Signal Panel" and "Image Panel" tabs corner to quickly access the
  most used features (e.g. "Add", "Remove", "Duplicate", etc.)
* Intensity profile extraction feature:
  * Added graphical user interface to extract intensity profiles from images, for
    both line and averaged profiles
  * Parameters are still directly editable by the user ("Edit profile parameters"
    button)
  * Parameters are now stored from one profile extraction to another
* Statistics feature:
  * Added `<y>/σ(y)` to the signal "Statistics" result table
    (in addition to the mean, median, standard deviation, etc.)
  * Added `peak-to-peak` to the signal and image "Statistics" result table
* Curve fitting feature: fit results are now stored in a dictionary in the signal
  metadata (instead of being stored individually in the signal metadata)
* Window state:
  * The toolbars and dock widgets state (visibility, position, etc.) are now stored
    in the configuration file and restored at startup (size and position were already
    stored and restored)
  * This implements part of [Issue #30](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/30) - Save/restore main window layout

🛠️ Bug fixes:

* Fixed [Issue #41](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/41) - Radial profile extraction: unable to enter user-defined center coordinates
* Fixed [Issue #49](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/49) - Error when trying to open a (UTF-8 BOM) text file as an image
* Fixed [Issue #51](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/51) - Unexpected dimensions when adding new ROI on an image with X/Y arbitrary units (not pixels)

## DataLab Version 0.11.0 ##

💥 New features:

* Signals and images may now be reordered in the tree view:
  * Using the new "Move up" and "Move down" actions in the "Edit" menu (or using the
    corresponding toolbar buttons):
  * This fixes [Issue #22](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/22) - Add "move up/down" actions in "Edit" menu, for signals/images and groups
* Signals and images may also be reordered using drag and drop:
  * Signals and images can be dragged and dropped inside their own panel to change
    their order
  * Groups can also be dragged and dropped inside their panel
  * The feature also supports multi-selection (using the standard Ctrl and Shift
    modifiers), so that multiple signals/images/groups can be moved at once, not
    necessarily with contiguous positions
  * This fixes [Issue #17](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/17) - Add Drag and Drop feature to Signals/Images tree views
* New 1D interpolation features:
  * Added "Interpolation" feature to signal panel's "Processing" menu
  * Methods available: linear, spline, quadratic, cubic, barycentric and PCHIP
  * Thanks to [@marcel-goldschen-ohm](https://github.com/marcel-goldschen-ohm) for the contribution to spline interpolation
  * This fixes [Issue #20](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/20) - Add 1D interpolation features
* New 1D resampling feature:
  * Added "Resampling" feature to signal panel's "Processing" menu
  * Same interpolation methods as for the "Interpolation" feature
  * Possibility to specify the resampling step or the number of points
  * This fixes [Issue #21](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/21) - Add 1D resampling feature
* New 1D convolution feature:
  * Added "Convolution" feature to signal panel's "Operation" menu
  * This fixes [Issue #23](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/23) - Add 1D convolution feature
* New 1D detrending feature:
  * Added "Detrending" feature to signal panel's "Processing" menu
  * Methods available: linear or constant
  * This fixes [Issue #24](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/24) - Add 1D detrending feature
* 2D computing results:
  * Before this release, 2D computing results such as contours, blobs, etc. were
    stored in image metadata dictionary as coordinates (x0, y0, x1, y1, ...) even
    for circles and ellipses (i.e. the coordinates of the bounding rectangles).
  * For convenience, the circle and ellipse coordinates are now stored in image
    metadata dictionary as (x0, y0, radius) and (x0, y0, a, b, theta) respectively.
  * These results are also shown as such in the "Results" dialog box (either at the
    end of the computing process or when clicking on the "Show results" button).
  * This fixes [Issue #32](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/32) - Contour detection: show circle `(x, y, r)` and ellipse `(x, y, a, b, theta)` instead of `(x0, y0, x1, x1, ...)`
* 1D and 2D computing results:
  * Additionnaly to the previous enhancement, more computing results are now shown
    in the "Results" dialog box
  * This concerns both 1D (FHWM, ...) and 2D computing results (contours, blobs, ...):
    * Segment results now also show length (L) and center coordinates (Xc, Yc)
    * Circle and ellipse results now also show area (A)
* Added "Plot results" entry in "Computing" menu:
  * This feature allows to plot computing results (1D or 2D)
  * It creates a new signal with X and Y axes corresponding to user-defined
    parameters (e.g. X = indexes and Y = radius for circle results)
* Increased default width of the object selection dialog box:
  * The object selection dialog box is now wider by default, so that the full
    signal/image/group titles may be more easily readable
* Delete metadata feature:
  * Before this release, the feature was deleting all metadata, including the Regions
    Of Interest (ROI) metadata, if any.
  * Now a confirmation dialog box is shown to the user before deleting all metadata if
    the signal/image has ROI metadata: this allows to keep the ROI metadata if needed.
* Image profile extraction feature: added support for masked images (when defining
  regions of interest, the areas outside the ROIs are masked, and the profile is
  extracted only on the unmasked areas, or averaged on the unmasked areas in the case
  of average profile extraction)
* Curve style: added "Reset curve styles" in "View" menu.
  This feature allows to reset the curve style cycle to its initial state.
* Plugin base classe `PluginBase`:
  * Added `edit_new_signal_parameters` method for showing a dialog box to edit
    parameters for a new signal
  * Added `edit_new_image_parameters` method for showing a dialog box to edit
    parameters for a new image (updated the *cdl_testdata.py* plugin accordingly)
* Signal and image computations API (`cdl.core.computations`):
  * Added wrappers for signal and image 1 -> 1 computations
  * These wrappers aim at simplifying the creation of a basic computation function
    operating on DataLab's native objects (`SignalObj` and `ImageObj`) from a
    function operating on NumPy arrays
  * This simplifies DataLab's internals and makes it easier to create new computing
    features inside plugins
  * See the *cdl_custom_func.py* example plugin for a practical use case
* Added "Radial profile extraction" feature to image panel's "Operation" menu:
  * This feature allows to extract a radially averaged profile from an image
  * The profile is extracted around a user-defined center (x0, y0)
  * The center may also be computed (centroid or image center)
* Automated test suite:
  * Since version 0.10, DataLab's proxy object has a `toggle_auto_refresh` method
    to toggle the "Auto-refresh" feature. This feature may be useful to improve
    performance during the execution of test scripts
  * Test scenarios on signals and images are now using this feature to improve
    performance
* Signal and image metadata:
  * Added "source" entry to the metadata dictionary, to store the source file path
    when importing a signal or an image from a file
  * This field is kept while processing the signal/image, in order to keep track of
    the source file path

📚 Documentation:

* New [Tutorial section](https://codra-ingenierie-informatique.github.io/DataLab/en/intro/tutorials/index.html) in the documentation:
  * This section provides a set of tutorials to learn how to use DataLab
  * The following video tutorials are available:
    * Quick demo
    * Adding your own features
  * The following text tutorials are available:
    * Processing a spectrum
    * Detecting blobs on an image
    * Measuring Fabry-Perot fringes
    * Prototyping a custom processing pipeline
* New [API section](https://codra-ingenierie-informatique.github.io/DataLab/en/api/index.html) in the documentation:
  * This section explains how to use DataLab as a Python library, by covering the
  following topics:
    * How to use DataLab algorithms on NumPy arrays
    * How to use DataLab computation features on DataLab objects (signals and images)
    * How to use DataLab I/O features
    * How to use proxy objects to control DataLab remotely
  * This section also provides a complete API reference for DataLab objects and
  features
  * This fixes [Issue #19](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/19) - Add API documentation (data model, functions on arrays or signal/image objects, ...)

🛠️ Bug fixes:

* Fixed [Issue #29](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/29) - Polynomial fit error: `QDialog [...] argument 1 has an unexpected type 'SignalProcessor'`
* Image ROI extraction feature:
  * Before this release, when extracting a single circular ROI from an image with the
    "Extract all regions of interest into a single image object" option enabled, the
    result was a single image without the ROI mask (the ROI mask was only available
    when extracting ROI with the option disabled)
  * This was leading to an unexpected behavior, because one could interpret the result
    (a square image without the ROI mask) as the result of a single rectangular ROI
  * Now, when extracting a single circular ROI from an image with the "Extract all
    regions of interest into a single image object" option enabled, the result is a
    single image with the ROI mask (as if the option was disabled)
  * This fixes [Issue #31](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/31) - Single circular ROI extraction: automatically switch to `extract_single_roi` function
* Computing on circular ROI:
  * Before this release, when running computations on a circular ROI,
    the results were unexpected in terms of coordinates (results seemed to be computed
    in a region located above the actual ROI).
  * This was due to a regression introduced in an earlier release.
  * Now, when defining a circular ROI and running computations on it, the results are
    computed on the actual ROI
  * This fixes [Issue #33](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/33) - Computing on circular ROI: unexpected results
* Contour detection on ROI:
  * Before this release, when running contour detection on a ROI, some
    contours were detected outside the ROI (it may be due to a limitation of the
    scikit-image `find_contours` function).
  * Now, thanks a workaround, the erroneous contours are filtered out.
  * A new test module `cdl.tests.features.images.contour_fabryperot_app` has been
    added to test the contour detection feature on a Fabry-Perot image (thanks to
    [@emarin2642](https://github.com/emarin2642) for the contribution)
  * This fixes [Issue #34](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/34) - Contour detection: unexpected results outside ROI
* Computing result merging:
  * Before this release, when doing a `1->N` computation (sum, average, product) on
    a group of signals/images, the computing results associated to each signal/image
    were merged into a single result, but only the type of result present in the
    first signal/image was kept.
  * Now, the computing results associated to each signal/image are merged into a
    single result, whatever the type of result is.
* Fixed [Issue #36](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/36) - "Delete all" action enable state is sometimes not refreshed
* Image X/Y swap: when swapping X and Y axes, the regions of interest (ROI) were not
  removed and not swapped either (ROI are now removed, until we implement the swap
  feature, if requested)
* "Properties" group box: the "Apply" button was enabled by default, even when no
  property was modified, which was confusing for the user (the "Apply" button is now
  disabled by default, and is enabled only when a property is modified)
* Fixed proxy `get_object` method when there is no object to return
  (`None` is returned instead of an exception)
* Fixed `IndexError: list index out of range` when performing some operations or
  computations on groups of signals/images (e.g. "ROI extraction", "Peak detection",
  "Resize", etc.)
* Drag and drop from a file manager: filenames are now sorted alphabetically

## DataLab Version 0.10.1 ##

*Note*: V0.10.0 was almost immediately replaced by V0.10.1 due to a last minute
bug fix

💥 New features:

* Features common to signals and images:
  * Added "Real part" and "Imaginary part" features to "Operation" menu
  * Added "Convert data type" feature to "Operation" menu
* Features added following user requests (12/18/2023 meetup @ CEA):
  * Curve and image styles are now saved in the HDF5 file:
    * Curve style covers the following properties: color, line style, line width,
      marker style, marker size, marker edge color, marker face color, etc.
    * Image style covers the following properties: colormap, interpolation, etc.
    * Those properties were already persistent during the working session, but
      were lost when saving and reloading the HDF5 file
    * Now, those properties are saved in the HDF5 file and are restored when
      reloading the HDF5 file
  * New profile extraction features for images:
    * Added "Line profile" to "Operations" menu, to extract a profile from
      an image along a row or a column
    * Added "Average profile" to "Operations" menu, to extract the
      average profile on a rectangular area of an image, along a row or a column
  * Image LUT range (contrast/brightness settings) is now saved in the HDF5 file:
    * As for curve and image styles, the LUT range was already persistent during
      the working session, but was lost when saving and reloading the HDF5 file
    * Now, the LUT range is saved in the HDF5 file and is restored when reloading it
  * Added "Auto-refresh" and "Refresh manually" actions in "View" menu
    (and main toolbar):
    * When "Auto-refresh" is enabled (default), the plot view is automatically refreshed
      when a signal/image is modified, added or removed. Even though the refresh is
      optimized, this may lead to performance issues when working with large
      datasets.
    * When disabled, the plot view is not automatically refreshed. The user
      must manually refresh the plot view by clicking on the "Refresh manually" button
      in the main toolbar or by pressing the standard refresh key (e.g. "F5").
  * Added `toggle_auto_refresh` method to DataLab proxy object:
    * This method allows to toggle the "Auto-refresh" feature from a macro-command,
      a plugin or a remote control client.
    * A context manager `context_no_refresh` is also available to temporarily disable
      the "Auto-refresh" feature from a macro-command, a plugin or a remote control
      client. Typical usage:

      ```python
      with proxy.context_no_refresh():
          # Do something without refreshing the plot view
          proxy.compute_fft() # (...)
      ```

  * Improved curve readability:
    * Until this release, the curve style was automatically set by cycling through
      **PlotPy** predefined styles
    * However, some styles are not suitable for curve readability (e.g. "cyan" and
      "yellow" colors are not readable on a white background, especially when combined
      with a "dashed" line style)
    * This release introduces a new curve style management with colors which are
      distinguishable and accessible, even to color vision deficiency people
* Added "Curve anti-aliasing" feature to "View" menu (and toolbar):
  * This feature allows to enable/disable curve anti-aliasing (default: enabled)
  * When enabled, the curve rendering is smoother but may lead to performance issues
    when working with large datasets (that's why it can be disabled)
* Added `toggle_show_titles` method to DataLab proxy object. This method allows to
  toggle the "Show graphical object titles" feature from a macro-command, a plugin
  or a remote control client.
* Remote client is now checking the server version and shows a warning message if
  the server version may not be fully compatible with the client version.

🛠️ Bug fixes:

* Image contour detection feature ("Computing" menu):
  * The contour detection feature was not taking into account the "shape" parameter
    (circle, ellipse, polygon) when computing the contours. The parameter was stored
    but really used only when calling the feature a second time.
  * This unintentional behavior led to an `AssertionError` when choosing "polygon"
    as the contour shape and trying to compute the contours for the first time.
  * This is now fixed (see [Issue #9](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/9) - Image contour detection: `AssertionError` when choosing "polygon" as the contour shape)
* Keyboard shortcuts:
  * The keyboard shortcuts for "New", "Open", "Save", "Duplicate", "Remove",
    "Delete all" and "Refresh manually" actions were not working properly.
  * Those shortcuts were specific to each signal/image panel, and were working only
    when the panel on which the shortcut was pressed for the first time was active
    (when activated from another panel, the shortcut was not working and a warning
    message was displayed in the console,
    e.g. `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
  * Besides, the shortcuts were not working at startup (when no panel had focus).
  * This is now fixed: the shortcuts are now working whatever the active panel is,
    and even at startup (see [Issue #10](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/10) - Keyboard shortcuts not working properly: `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
* "Show graphical object titles" and "Auto-refresh" actions were not working properly:
  * The "Show graphical object titles" and "Auto-refresh" actions were only working on
    the active signal/image panel, and not on all panels.
  * This is now fixed (see [Issue #11](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/11) - "Show graphical object titles" and "Auto-refresh" actions were working only on current signal/image panel)
* Fixed [Issue #14](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/14) - Saving/Reopening HDF5 project without cleaning-up leads to `ValueError`
* Fixed [Issue #15](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/15) - MacOS: 1. `pip install cdl` error - 2. Missing menus:
  * Part 1: `pip install cdl` error on MacOS was actually an issue from **PlotPy** (see
    [this issue](https://github.com/PlotPyStack/PlotPy/issues/9)), and has been fixed
    in PlotPy v2.0.3 with an additional compilation flag indicating to use C++11 standard
  * Part 2: Missing menus on MacOS was due to a PyQt/MacOS bug regarding dynamic menus
* HDF5 file format: when importing an HDF5 dataset as a signal or an image, the
  dataset attributes were systematically copied to signal/image metadata: we now
  only copy the attributes which match standard data types (integers, floats, strings)
  to avoid errors when serializing/deserializing the signal/image object
* Installation/configuration viewer: improved readability (removed syntax highlighting)
* PyInstaller specification file: added missing `skimage` data files manually in order
  to continue supporting Python 3.8 (see [Issue #12](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/12) - Stand-alone version on Windows 7: missing `api-ms-win-core-path-l1-1-0.dll`)
* Fixed [Issue #13](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/13) - ArchLinux: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found`

## DataLab Version 0.9.2 ##

🛠️ Bug fixes:

* Region of interest (ROI) extraction feature for images:
  * ROI extraction was not working properly when the "Extract all regions of interest
    into a single image object" option was enabled if there was only one defined ROI.
    The result was an image positioned at the origin (0, 0) instead of the expected
    position (x0, y0) and the ROI rectangle itself was not removed as expected.
    This is now fixed (see [Issue #6](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/6) - 'Extract multiple ROI' feature: unexpected result for a single ROI)
  * ROI rectangles with negative coordinates were not properly handled:
    ROI extraction was raising a `ValueError` exception, and the image mask was not
    displayed properly.
    This is now fixed (see [Issue #7](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/7) - Image ROI extraction: `ValueError: zero-size array to reduction operation minimum which has no identity`)
  * ROI extraction was not taking into account the pixel size (dx, dy) and the origin
    (x0, y0) of the image.
    This is now fixed (see [Issue #8](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/8) - Image ROI extraction: take into account pixel size)
* Macro-command console is now read-only:
  * The macro-command panel Python console is currently not supporting standard input
    stream (`stdin`) and this is intended (at least for now)
  * Set Python console read-only to avoid confusion

## DataLab Version 0.9.1 ##

🛠️ Bug fixes:

* French translation is not available on Windows/Stand alone version:
  * Locale was not properly detected on Windows for stand-alone version (frozen
    with `pyinstaller`) due to an issue with `locale.getlocale()` (function
    returning `None` instead of the expected locale on frozen applications)
  * This is ultimately a `pyinstaller` issue, but a workaround has been
    implemented in `guidata` V3.2.2 (see [guidata issue #68](https://github.com/PlotPyStack/guidata/issues/68) - Windows: gettext translation is not working on frozen applications)
  * [Issue #2](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/2) - French translation is not available on Windows Stand alone version
* Saving image to JPEG2000 fails for non integer data:
  * JPEG2000 encoder does not support non integer data or signed integer data
  * Before, DataLab was showing an error message when trying to save incompatible
    data to JPEG2000: this was not a consistent behavior with other standard image
    formats (e.g. PNG, JPG, etc.) for which DataLab was automatically converting
    data to the appropriate format (8-bit unsigned integer)
  * Current behavior is now consistent with other standard image formats: when
    saving to JPEG2000, DataLab automatically converts data to 8-bit unsigned
    integer or 16-bit unsigned integer (depending on the original data type)
  * [Issue #3](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/3) - Save image to JPEG2000: 'OSError: encoder error -2 when writing image file'
* Windows stand-alone version shortcuts not showing in current user start menu:
  * When installing DataLab on Windows from a non-administrator account, the
    shortcuts were not showing in the current user start menu but in the
    administrator start menu instead (due to the elevated privileges of the
    installer and the fact that the installer does not support installing
    shortcuts for all users)
  * Now, the installer *does not* ask for elevated privileges anymore, and
    shortcuts are installed in the current user start menu (this also means that
    the current user must have write access to the installation directory)
  * In future releases, the installer will support installing shortcuts for all
    users if there is a demand for it (see [Issue #5](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/5))
  * [Issue #4](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/4) - Windows: stand-alone version shortcuts not showing in current user start menu
* Installation and configuration window for stand-alone version:
  * Do not show ambiguous error message 'Invalid dependencies' anymore
  * Dependencies are supposed to be checked when building the stand-alone version
* Added PDF documentation to stand-alone version:
  * The PDF documentation was missing in previous release
  * Now, the PDF documentation (in English and French) is included in the
    stand-alone version

## DataLab Version 0.9.0 ##

New dependencies:

* DataLab is now powered by [PlotPyStack](https://github.com/PlotPyStack):
  * [PythonQwt](https://github.com/PlotPyStack/PythonQwt)
  * [guidata](https://github.com/PlotPyStack/guidata)
  * [PlotPy](https://github.com/PlotPyStack/PlotPy)
* [opencv-python](https://pypi.org/project/opencv-python/) (algorithms for image processing)

New reference platform:

* DataLab is validated on Windows 11 with Python 3.11 and PyQt 5.15
* DataLab is also compatible with other OS (Linux, MacOS) and other Python-Qt
  bindings and versions (Python 3.8-3.12, PyQt6, PySide6)

New features:

* DataLab is a platform:
  * Added support for plugins
    * Custom processing features available in the "Plugins" menu
    * Custom I/O features: new file formats can be added to the standard I/O
      features for signals and images
    * Custom HDF5 features: new HDF5 file formats can be added to the standard
      HDF5 import feature
    * More features to come...
  * Added remote control feature: DataLab can be controlled remotely via a
    TCP/IP connection (see [Remote control](https://codra-ingenierie-informatique.github.io/DataLab/en/remote_control.html))
  * Added macro commands: DataLab can be controlled via a macro file (see
    [Macro commands](https://codra-ingenierie-informatique.github.io/DataLab/en/macro_commands.html))
* General features:
  * Added settings dialog box (see "Settings" entry in "File" menu):
    * General settings
    * Visualization settings
    * Processing settings
    * Etc.
  * New default layout: signal/image panels are on the right side of the main
    window, visualization panels are on the left side with a vertical toolbar
* Signal/Image features:
  * Added process isolation: each signal/image is processed in a separate
    process, so that DataLab does not freeze anymore when processing large
    signals/images
  * Added support for groups: signals and images can be grouped together, and
    operations can be applied to all objects in a group, or between groups
  * Added warning and error dialogs with detailed traceback links to the source
    code (warnings may be optionally ignored)
  * Drastically improved performance when selecting objects
  * Optimized performance when showing large images
  * Added support for dropping files on signal/image panel
  * Added "Computing parameters" group box to show last result input parameters
  * Added "Copy titles to clipboard" feature in "Edit" menu
  * For every single processing feature (operation, processing and computing menus),
    the entered parameters (dialog boxes) are stored in cache to be used as defaults
    the next time the feature is used
* Signal processing:
  * Added support for optional FFT shift (see Settings dialog box)
* Image processing:
  * Added pixel binning operation (X/Y binning factors, operation: sum, mean, ...)
  * Added "Distribute on a grid" and "Reset image positions" in operation menu
  * Added Butterworth filter
  * Added exposure processing features:
    * Gamma correction
    * Logarithmic correction
    * Sigmoïd correction
  * Added restoration processing features:
    * Total variation denoising filter (TV Chambolle)
    * Bilateral filter (denoising)
    * Wavelet denoising filter
    * White Top-Hat denoising filter
  * Added morphological transforms (disk footprint):
    * White Top-Hat
    * Black Top-Hat
    * Erosion
    * Dilation
    * Opening
    * Closing
  * Added edge detection features:
    * Roberts filter
    * Prewitt filter (vertical, horizontal, both)
    * Sobel filter (vertical, horizontal, both)
    * Scharr filter (vertical, horizontal, both)
    * Farid filter (vertical, horizontal, both)
    * Laplace filter
    * Canny filter
  * Contour detection: added support for polygonal contours (in addition to
    circle and ellipse contours)
  * Added circle Hough transform (circle detection)
  * Added image intensity levels rescaling
  * Added histogram equalization
  * Added adaptative histogram equalization
  * Added blob detection methods:
    * Difference of Gaussian
    * Determinant of Hessian method
    * Laplacian of Gaussian
    * Blob detection using OpenCV
  * Result shapes and annotations are now transformed (instead of removed) when
    executing one of the following operations:
    * Rotation (arbitrary angle, +90°, -90°)
    * Symetry (vertical/horizontal)
  * Added support for optional FFT shift (see Settings dialog box)
* Console: added configurable external editor (default: VSCode) to follow the
  traceback links to the source code
