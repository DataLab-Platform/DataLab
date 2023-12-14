# Changelog #

See DataLab [roadmap page](https://cdlapp.readthedocs.io/en/latest/dev/roadmap.html)
for future and past milestones.

## DataLab Version 0.9.2 ##

🛠️ Bug fixes:

* Region of interest (ROI) extraction feature for images:
  * ROI extraction was not working properly when the "Extract all regions of interest
    into a single image object" option was enabled if there was only one defined ROI.
    The result was an image positioned at the origin (0, 0) instead of the expected
    position (x0, y0) and the ROI rectangle itself was not removed as expected.
    This is now fixed (see [Issue #6](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/6) - 'Extract multiple ROI' feature: unexpected result for a single ROI)
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
    TCP/IP connection (see [Remote control](https://cdlapp.readthedocs.io/en/latest/remote_control.html))
  * Added macro commands: DataLab can be controlled via a macro file (see
    [Macro commands](https://cdlapp.readthedocs.io/en/latest/macro_commands.html))
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

## Older releases ##

### CodraFT Version 2.2.0 ###

New features:

* Images: added support for XYZ image files
* All shapes: removed shape drag symbols, so that background image is no longer
  masked by small-sized shapes
* At startup, restoring last current panel (image or signal panel)
* Plot cleanup and shape management: greatly optimized performance
* After removing object(s) (signal/image), the previous object in the list is selected
* Added default image visualization settings in .INI configuration file
* Using guiqwt v4.3.2: fixed pixel position (first pixel is centered at (0,0) coords)

### CodraFT Version 2.1.4 ###

Bug fixes:

* HDF5 import/browser features: added support for non-ASCII dataset names
* ANDOR SIF files:
  * Fixed compatibility issues for various SIF files
  * Fixed unicode error
* Image Contour detection:
  * Fixed level default value for 8-bit data
  * Added missing "level" parameter
* Dev/VSCode: simplified `launch.json` and fixed environment variable substitution issue

Other changes:

* Alpha/beta release: fixed installer, added warning

### CodraFT Version 2.1.3 ###

Bug fixes:

* Panel's object list `select_rows` method: fixed plot refresh behavior in case of
multiple selection (refresh widget only once)
* LMJ-formatted HDF5 file: now reading invalid compound datasets
* [Issue #16](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/16) - Embedding DataLab: "add_object" method call with invalid data should lead to app crash
  * Panel's `add_object` method (public API): check data type before adding object to
panel - this prevents DataLab from crashing when trying to plot invalid data type
afterwards
  * Now handling exceptions in `add_object` and `insert_object` methods
* Multigaussian curve fitting: fixed default fit parameters
* Improved I/O application test with respect to unsupported filetypes

Other changes:

* Images: added support for `numpy.int32` datatype
* Added unit tests for all curve fitting dialogs

### CodraFT Version 2.1.2 ###

Bug fixes:

* [Pull Request #2](https://github.com/Codra-Ingenierie-Informatique/DataLab/pull/2) - Load / Save conventional CSVs, by [@aanastasiou](https://github.com/aanastasiou)
* [Issue #3](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/3) - Wrong units/titles are displayed
* [Issue #6](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/6) - 2D peak detection: GUI freezes when creating ROIs
* [Issue #4](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/4) - Processing multiple images/signals: avoid unnecessary time-consuming plot updates
* [Issue #7](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/7) - Image/Circular ROI: IndexError when circle exceeds the image size
* [Issue #5](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/5) - ROI dialog box: unable to remove all ROIs and validate
* [Issue #8](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/8) - HDF5 import: unable to easily distinguish datasets with the same name but different path
* Average operation now merges ROI data (i.e. same behavior as sum)
* Fixed multiple regressions with ROI management (adding, removing ROI, ...)

Other changes:

* Optimized load time (especially for images): avoid unnecessary refresh when adding objects
* Added "Remove regions of interest" entry to "Computing" menu (and context menu)
* Signal/image list: added tooltip showing a summary of metadata values (e.g. when
  importing data from HDF5, this shows HDF5 filename and HDF5 dataset path) - Issue #8
* Dependencies hash check: feature is now OS-dependent (+ more explicit messages)
* Slightly improved test coverage

### CodraFT Version 2.1.1 ###

Changes:

* Image Regions Of Interest (ROI):
  * ROIs are now shown as masks (areas outside ROIs are shaded)
  * Added support for circular ROIs
  * ROIs now take into account pixel size (dx, dy) as well as origin (x0, y0)
* Signal and Image ROIs:
  * New default extract mode: creating as many signals/images as ROIs (each ROI is
    extracted into a single signal/image)
  * The old extract mode (single signal/image output) is still available and may be
    enabled using the new checkbox added in ROI extraction dialog box
* Image visualization:
  * Added "Show contrast panel" option in toolbar and view menu
  * By default, contrast panel is now visible
  * When multiple images are selected, the first image LUT range is applied to all
* "View in a new window": now opens non-modal dialogs, thus allowing to visualize
  multiple signals or images in separate windows
* Added demo mode (from command line, simply run: cdl-demo)
* Command line option --h5 is now a positionnal argument (h5)
* Added command line option -b (or --h5browser) to browse a HDF5 file at startup
* Added command line option --version to show DataLab version

Bug fixes:

* Image computations now takes into account origin (x0, y0), pixel size (dx, dy) as
  well as regions of interest (related features: centroid, enclosing circle, 2D peak
  detection and contour detection)
* Image ROI definition dialog: maximum rows and columns were erroneously truncated
* Centralized argument parsing in DataLab exec env object, thus avoiding conflicts

### CodraFT Version 2.0.3 ###

Bug fixes:

* Fixed pen.setWidth TypeError on Linux

Other changes:

* Added an option to ignore dependency check warning at startup
* Installation configuration viewer: added info on dependency check result
* Ignore when unable to save h5 in ima/sig test scenarios

### CodraFT Version 2.0.2 ###

The following major changes were introduced with DataLab V2:

* Fully automated high-level processing features for internal testing purpose, as well
as embedding DataLab in a third-party software
* Extensive test suite (unit tests and application tests) with 90% feature coverage
* Segmentation fault and Python exception logging
* Customizable annotations for both signals and images

### Release key features ####

* New data visualization and processing features:

  | Signal | Image | Feature                                                |
  |:------:|:-----:|--------------------------------------------------------|
  |        |   •   | Automatic 2D-peak detection                            |
  |        |   •   | Automatic contour extraction (circle/ellipse fit)      |
  |    •   |   •   | Multiple Regions of Interest (ROIs)                    |
  |        |   •   | User-defined annotations (labels and geometric shapes) |
  |    •   |   •   | "Statistics" computing feature                         |

* Automation of high-level processing features: added fully automated high-level test
scenarios, and enhanced public API for embedding DataLab into a third-party application
* Test Driven Development with high quality standards
(pylint score >= 9.8/10, test coverage >= 90%)

### Detailed feature list ####

New data visualization and processing features:

* Image:
  * New automatic image contour detection feature returning fitted circle/ellipse
  * New automatic 2D peak detection feature (optionally create ROIs)
* "View in a new window": added customizable "Annotations" support for both signal
and image panels - supports user-defined annotations (points, segments, circles,
ellipses, labels,...) which are serialized in image metadata
* Added "Show graphical object titles" option in "View" menu to show or hide the title
(or subtitle) of ROIs or any other graphical object
* Added support for **multiple** Regions of Interest (ROI):
  * All "Computing" menu features apply to multiple ROIs
  * Computation result arrays now contains ROI index (first column) and one row per ROI
  * ROI are merged when summing objects (signals or images)
  * ROI can be removed, modified or added at any time
* Added option "Show graphical object titles" ("View" menu) to show or hide ROI titles or any
other geometrical shapes title (or subtitle)
* New computing "Statistics" feature showing a table with statistics on image/signal
and eventually regions of interest (min, max, mean, standard deviation, sum, ...)

New general purpose features:

* Memory management:
  * New available memory indicator on main window status bar
  * New warning dialog box when trying to open/create data if available memory is below
  the "available_memory_threshold" defined in DataLab configuration file (default: 500MB)
* Error handling:
  * New integrated log file viewer
  * New warning dialog box at startup suggesting to view log files when logs were
  generated during last session
  * Logging segmentation faults in ".DataLab_faulthandler.log"
  * Logging Python exceptions in ".DataLabL_traceback.log"
* Signal/Image metadata:
  * New copy/paste feature: update object metadata from another one
  * New import/export feature: import-export object metadata (JSON text file) using the
  new "Import metadata into" / "Export metadata from" entries in "File" menu
* HDF5 browser feature: complete redesign (better compatibility, evolutive design, ...)
* Added support for multiple HDF5 files opening at once
* Added `.DataLab.ini` configuration file (user home directory):
  * New configuration file entry: current working directory
  * New configuration file entry: current main window size and position
  * New configuration file entry: embedded Python console enabled state
  * New configuration file entry: available memory alarm threshold

New test-related features:

* Added non-interactive tests, opening the way for unit tests with better coverage
* Added "unattended" and "screenshot" execution modes respectively for testing and documentation purpose
* Added automated high-level test scenarios (signal and image processing)
* Tests are now splitted in two categories: unit tests (`*_unit.py`) and application tests (`*_app.py`).
* Added Coverage.py support
* Added "all_tests.py" to run all tests in unattended mode

New dependencies:

* [scikit-image](https://pypi.org/project/scikit-image/)
* [psutil](https://pypi.org/project/psutil/)

Other changes (on existing features):

* Image and Signal:
  * Object properties panel: added data type information (feature refactored upstream to guidata)
  * New random signal/image: added support for both Normal and Uniform distributions
  * Operations "sum" and "average" now merge metadata results
  * Computed titles "s/i000" are now renamed after inserting/removing an object
  * Computing results (geometrical shapes: segment, circle, ellipse): numerical results
  are now automatically added to metadata (respectively: length, center and radius,
  center, a and b)
* Image:
  * Added support for image origin and pixel size
  * Flat field correction: added threshold parameter
  * "New image" now creates an image with the same data type as selected image
  * "New image" now supports uint16 data type
* Signal:
  * Peak detection: added minimal distance parameter
  * Fit dialog / plot: do auto scale at startup
  * Peak detection dialog: preselect horizontal cursor at startup
* `cdl.core.gui` code refactoring: added subpackage `core.gui.processor`
* Added "Browse HDF5" action to main window ("Open HDF5" now imports all data)

Bug fixes:

* HDF5 file import: converted `bytes` metadata to `str`
* Added h5py to requirements (setup.py)
* Plot: reintroduced pure white background in light mode (white background was removed
  unintentionally when introducing dark mode)
* Image:
  * "Clean-up data view" feature was accidently removing grid
  * Fixed hard crash when trying to visualize images with NaNs (use case: result of
  any filter on `uint8` image)
  * Fixed hard crash when using image Z-axis log scale on some images
  * Fixed DICOM support
  * Fixed hard crash in "to_codraft" (cross section item with empty data)
  * Fixed image visualization parameters update from metadata
  * MinEnclosingCircle: fixed sqrt(2) error
* Signal:
  * "Clean-up data view" feature was accidently removing legend box and grid
  * Fixed integral (missing initial point)
  * Fixed plotting support for complex data
  * Fixed signal visualization parameters update from metadata

### CodraFT Version 1.7.2 ###

Bug fixes:

* Fixed unit test "app1_test.py" (create a single QApp)
* Fixed progress bar cancel issues (when passing HDF5 files to `app.run` function)
* Fixed random hard crash when opening curve fitting dialog
* Fixed curve fitting dialog parenting
* ROI metadata is now removed (because potentially invalid) after performing a
  computation that changes X-axis or Y-axis data (e.g. ROI extraction, image flip,
  image rotation, etc.)
* Fixed image creation features (broken since major refactoring)

Other changes:

* Removed deprecated Qt `exec_` calls (replaced by `exec`)
* Added more infos on uninstaller registry keys
* Added documentation on key features

### CodraFT Version 1.7.1 ###

Added first page of documentation (there is a beginning to everything...).

Bug fixes:

* Cross section tool was working only on first image in item list
* Separate view was broken since major refactoring

### CodraFT Version 1.7.0 ###

New features:

* Python 3.8 is now the reference Python release
* Dropped Python 2 and PyQt 4 support
* Major code cleaning and refactoring
* Reorganized the whole code base
* Added more unit tests
* Added GUI-based test launcher
* Added isort/black code formatting
* Switched from cx_Freeze to pyinstaller for generating the stand-alone version
* Improved pylint score up to 9.90/10 with strict quality criteria

### CodraFT Version 1.6.0 ###

New features:

* Added dependencies check on startup: warn the user if at least one dependency
  has been altered (i.e. the application has not been qualified in this context)
* Added py3compat (since QtPy is dropping Python 3 support)

### CodraFT Version 1.5.0 ###

New features:

* Sum, average, difference, multiplication: re-converting data to initial type.

* Now supporting PySide2/PyQt4/PyQt5 compatibility thanks to
  guidata >= v1.7.9 (using QtPy).

* Now supporting Python 3.9 and NumPy 1.20.

Bug fixes:

* Fixed cross section retrieval feature: in stand-alone mode, a new DataLab
  window was created (that is not the expected behavior).

* Fixed crash when enabling cross sections on main window (needs PythonQwt 0.9.2).

* Fixed ValueError when generating a 2D-gaussian image with floats.

* HDF5 file import feature:

  * Fixed unit processing (parsing) with Python 3.

  * Fixed critical bug when clicking on "Check all".

### CodraFT Version 1.4.4 ###

New experimental features:

* Experimental support for PySide2/PyQt4/PyQt5 thanks to guidata >= v1.7.9 (using QtPy).

* Experimental support for Python 3.9 and NumPy 1.20.

New minor features:

* ZAxisLogTool: update automatically Z-axis scale (+ showing real value)

* Added contrast test (following issues with "eliminate_outliers")

### CodraFT Version 1.4.3 ###

New minor features:

* New test script for global application test (test_app.py).
* Improved DataLab launcher (app.py).

### CodraFT Version 1.4.2 ###

New minor features:

* LMJ-formatted HDF5 file import: tree widget item's tooltip now
  shows item data "description".

Bug fixes:

* Fixed runtime warnings when computing centroid coordinates on
  an image ROI filled with zeros.

* LMJ-formatted HDF5 file support: fixed truncated units.

### CodraFT Version 1.4.1 ###

Bug fixes:

* Fixed LMJ-formatted HDF5 files: strings are encoded in "latin-1"
  which is not the expected behavior ("utf-8" is the expected
  encoding for ensuring better compatibility).

### CodraFT Version 1.4.0 ###

New features:

* LMJ-formatted HDF5 file import: added support for axis units and labels.

* New curve style behavior (more readable): unselecting items by default,
  circling over curve colors when selecting multiple curve items.

Bug fixes:

* Fixed LMJ-formatted HDF5 file support in DataLab data import feature.

### CodraFT Version 1.3.1 ###

Bug fixes:

* Improved support for LMJ-formatted HDF5 files.

* Z-axis logscale feature: freeing memory when mode is off.

* CDLMainWindow.get_instance: create instance if it doesn't already exist.

* to_codraft: show DataLab main window on top, if not already visible.

* Patch/guiqwt.histogram: removing histogram curve (if necessary)
  when image item has been removed.

### CodraFT Version 1.3.0 ###

New features:

* Image computations: added "Smallest enclosing circle center" computation.
* Added support for FXD image file type.

Bug fixes:

* Fixed image levels "Log scale" feature for Python 3 compatibility.

### CodraFT Version 1.2.2 ###

New features:

* Added "Delete all" entry to "Edit" menu: this removes all objects (signals or
  images) from current view.

* Added an option "hide_on_close" to CDLMainWindow class constructor
  (default value is False): when set to True, DataLab main window will simply
  hide when "Close" button is clicked, which is the expected behavior when
  embedding DataLab in another application.

Bug fixes:

* The memory leak fix in app.py was accidentally commented before commit.

### CodraFT Version 1.2.1 ###

Bug fixes:

* When quitting DataLab, objects were not deleted: this was causing a memory
  leak when embedding DataLab in another Qt window.

* When canceling HDF5 import dialog box after selecting at least one signal or
  image, the progress bar was shown even if no data was being imported.

* When closing HDF5 import dialog box, preview signal/image widgets were not
  deleted, hence causing another memory leak.

### CodraFT Version 1.2.0 ###

New features:

* Added support for uint32 images (converting to int32 data)

* Added "Z-axis logarithmic scale" feature for image items (check out the new
  entries in standard image toolbar and context menu)

* Added "HDF5 I/O Toolbar" to avoid a frequently reported user confusion
  between HDF5 I/O icons and Signal/Image specific I/O icons (i.e. open and
  save actions)

* Cross-section panels are now configured to show only cross-section curves
  associated to the currently selected image (instead of showing all curves,
  including those associated to hidden images)

* Image subtraction: now handling integer underflow

Bug fixes:

* When "Clean up data view" option was enabled, image histogram was not updated
  properly when changing image selection (histogram was the sum of all images
  histograms).

* Changed default image levels histogram "eliminate outliers" value: .1% instead
  of 2% to avoid display bug for noise background images for example (i.e.
  images with high contrast and very narrow histogram levels)

### CodraFT Version 1.1.2 ###

Bug fixes:

* When the X/Y Cross Section widget is embedded into a main window other than
  DataLab's, clicking on the "Process signal" button will send the signal to
  DataLab's signal panel for further processing, as expected.

### CodraFT Version 1.1.1 ###

Bug fixes:

* Fixed a bug leading to "None" titles when importing signals/images from HDF5
  files created outside DataLab.

### CodraFT Version 1.1.0 ###

New features:

* Added new icons.

* Images:

  * Added support for SPIRICON image files (single-frame support only).

Bug fixes:

* Fixed a critical bug when opening HDF5 file (bug from "guidata" package).
  Now guidata is patched inside DataLab to take into account the unusual/risky
  PyQt patch from Taurus package (PyQt API is set to 2 for QString objects and
  instead of raising an ImportError when importing QString from PyQt4.QtCore,
  QString still exists and is replaced by "str"...).

* Images:

  * Centroid feature: coordinates were mixed up in DataLab application.

* Signals:

  * Curve fitting (gaussian and lorentzian): fixed amplitude initial value
      for automatic fitting feature
  * FWHM and FW1/e²: fixed amplitude computation for input fit parameters
      and output results

### CodraFT Version 1.0.0 ###

First release of `CodraFT`.

New features:

* Added support for both Python 3 and Python 2.7, and both PyQt5 and PyQt4.

* Added HDF5 file reading support, using a new HDF5 browser with embedded
  curve and image preview.

* Signal and Image:

  * Added menu "Computing" for computing scalar values from signals/images.
  * Added "ROI definition" for "Computing" features
  * Added absolute value operation.
  * Added 10 base logarithm operation.
  * Added moving average/median filtering feature.

* Images:

  * Added support for Andor SIF image files (support multiple frames).
  * Added centroid computing feature.
  * Added support for images containing NaN values.

* Signals:

  * Added FWHM computing feature (based on curve fitting)
  * Added Full Width at 1/e² computing feature (based on gaussian fitting)
  * Added derivative and integral computation features.
  * Added "lorentzian" and "Voigt" to "new signals" available.

* Added curve fitting feature supporting various models (polynomial,
  gaussian, lorentzian, Voigt and multi-gaussian). Computed fitting
  parameters are stored in signal's metadata (a new dictionary item
  for the Signal objects)

* Edit menu: added a new "View in a new window" action

* Added standard keyboard shortcuts (new, open, copy, etc.)

* "New image": added new 2D-gaussian creation feature

* Added a GUI-based ROI extraction feature for both signal and image views

* Added a pop-up dialog when double-clicking on a signal/image to allow
  visualizing things on a possibly large window

* Added a peak detection feature

* Added centroid coordinates in image statistics tool

* Added support for curve/image titles, axis labels and axis units (those can
  be modified through the editable form within the "Properties" groupbox)

* Added support for cross section extraction from the image widget to the
  signal tab ; the extracted curve's title shows the associated coordinates

* Added deployment script for building self-consistent executable distribution
  using the cx_Freeze tool

* Improved curve visual: background is now flat and white

Bug fixes:

* Console dockwidget is now created after the `View` menu so that it appears
  in it, as expected. It is now hidden by default.

* Improved curve visual when selected: instead of adding big black squares
  along a selected curve, the curve line is simply broader when selected.
