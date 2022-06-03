# CodraFT Roadmap #

## Past milestones ##

### CodraFT 2.0 ###

* New data processing and visualization features (see below)
* Fully automated high-level processing features for internal testing purpose, as well as embedding CodraFT in a third-party software
* Extensive test suite (unit tests and application tests) with 90% feature coverage

### CodraFT 1.7 ###

* Major redesign
* Python 3.8 is the new reference
* Dropped Python 2 support

### CodraFT 1.6 ###

* Last release supporting Python 2

----

# CodraFT Releases #

## Version 2.0.2 ##

The following major changes were introduced with CodraFT V2:

* Fully automated high-level processing features for internal testing purpose, as well
as embedding CodraFT in a third-party software
* Extensive test suite (unit tests and application tests) with 90% feature coverage
* Segmentation fault and Python exception logging
* Dropped Python 2 support (CodraFT V1.7, the last release supporting Python 2,
will still be maintained for critical bug fixes, i.e. without introducing new features)

### Release key features ###

* New data visualization and processing features:

  | Signal | Image | Feature                                                |
  |:------:|:-----:|--------------------------------------------------------|
  |        |   •   | Automatic 2D-peak detection                            |
  |        |   •   | Automatic contour extraction (circle/ellipse fit)      |
  |    •   |   •   | Multiple Regions of Interest (ROIs)                    |
  |        |   •   | User-defined annotations (labels and geometric shapes) |
  |    •   |   •   | "Statistics" computing feature                         |

* Automation of high-level processing features: added fully automated high-level test
scenarios, and enhanced public API for embedding CodraFT into a third-party application
* Test Driven Development with high quality standards
(pylint score >= 9.8/10, test coverage >= 90%)

### Detailed feature list ###

New data visualization and processing features:

* Image:
  * Added "Edit Annotations" button on image properties group box:
  supports user-defined annotations (points, segments, circles, ellipses, labels,...)
  which are serialized in image metadata
  * New automatic image contour detection feature returning fitted circle/ellipse
  * New automatic 2D peak detection feature (optionally create ROIs)
* Added "Graphical object titles" option in "View" menu to show or hide the title
(or subtitle) of ROIs or any other graphical object
* Added support for **multiple** Regions of Interest (ROI):
  * All "Computing" menu features apply to multiple ROIs
  * Computation result arrays now contains ROI index (first column) and one row per ROI
  * ROI are merged when summing objects (signals or images)
  * ROI can be removed, modified or added at any time
* Added option "Show annotations" ("View" menu) to show or hide ROI titles or any
other geometrical shapes title (or subtitle)
* New computing "Statistics" feature showing a table with statistics on image/signal
and eventually regions of interest (min, max, mean, standard deviation, sum, ...)

New general purpose features:

* Memory management:
  * New available memory indicator on main window status bar
  * New warning dialog box when trying to open/create data if available memory is below
  the "available_memory_threshold" defined in CodraFT configuration file (default: 500MB)
* Error handling:
  * New integrated log file viewer
  * New warning dialog box at startup suggesting to view log files when logs were
  generated during last session
  * Logging segmentation faults in ".CodraFT_faulthandler.log"
  * Logging Python exceptions in ".CodraFT_traceback.log"
* Signal/Image metadata:
  * New copy/paste feature: update object metadata from another one
  * New import/export feature: import-export object metadata (JSON text file) using the
  new "Import metadata into" / "Export metadata from" entries in "File" menu
* HDF5 browser feature: complete redesign (better compatibility, evolutive design, ...)
* Added support for multiple HDF5 files opening at once
* Added `.CodraFT.ini` configuration file (user home directory):
  * New configuration file entry: current working directory
  * New configuration file entry: current main window size and position
  * New configuration file entry: embedded Python console enable state
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
* `codraft.core.gui` code refactoring: added subpackage `core.gui.processor`
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

## Version 1.7.2 ##

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

## Version 1.7.1 ##

Added first page of documentation (there is a beginning to everything...).

Bug fixes:

* Cross section tool was working only on first image in item list
* Separate view was broken since major refactoring

## Version 1.7.0 ##

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

## Version 1.6.0 ##

New features:

* Added dependencies check on startup: warn the user if at least one dependency
  has been altered (i.e. the application has not been qualified in this context)
* Added py3compat (since QtPy is dropping Python 3 support)

## Version 1.5.0 ##

New features:

* Sum, average, difference, multiplication: re-converting data to initial type.

* Now supporting PySide2/PyQt4/PyQt5 compatibility thanks to
  guidata >= v1.7.9 (using QtPy).

* Now supporting Python 3.9 and NumPy 1.20.

Bug fixes:

* Fixed cross section retrieval feature: in stand-alone mode, a new CodraFT
  window was created (that is not the expected behavior).

* Fixed crash when enabling cross sections on main window (needs PythonQwt 0.9.2).

* Fixed ValueError when generating a 2D-gaussian image with floats.

* HDF5 file import feature:

  * Fixed unit processing (parsing) with Python 3.

  * Fixed critical bug when clicking on "Check all".

## Version 1.4.4 ##

New experimental features:

* Experimental support for PySide2/PyQt4/PyQt5 thanks to guidata >= v1.7.9 (using QtPy).

* Experimental support for Python 3.9 and NumPy 1.20.

New minor features:

* ZAxisLogTool: update automatically Z-axis scale (+ showing real value)

* Added contrast test (following issues with "eliminate_outliers")

## Version 1.4.3 ##

New minor features:

* New test script for global application test (test_app.py).
* Improved CodraFT launcher (app.py).

## Version 1.4.2 ##

New minor features:

* LMJ-formatted HDF5 file import: tree widget item's tooltip now
  shows item data "description".

Bug fixes:

* Fixed runtime warnings when computing centroid coordinates on
  an image ROI filled with zeros.

* LMJ-formatted HDF5 file support: fixed truncated units.

## Version 1.4.1 ##

Bug fixes:

* Fixed LMJ-formatted HDF5 files: strings are encoded in "latin-1"
  which is not the expected behavior ("utf-8" is the expected
  encoding for ensuring better compatibility).

## Version 1.4.0 ##

New features:

* LMJ-formatted HDF5 file import: added support for axis units and labels.

* New curve style behavior (more readable): unselecting items by default,
  circling over curve colors when selecting multiple curve items.

Bug fixes:

* Fixed LMJ-formatted HDF5 file support in CodraFT data import feature.

## Version 1.3.1 ##

Bug fixes:

* Improved support for LMJ-formatted HDF5 files.

* Z-axis logscale feature: freeing memory when mode is off.

* CodraFTMainWindow.get_instance: create instance if it doesn't already exist.

* to_codraft: show CodraFT main window on top, if not already visible.

* Patch/guiqwt.histogram: removing histogram curve (if necessary)
  when image item has been removed.

## Version 1.3.0 ##

New features:

* Image computations: added "Smallest enclosing circle center" computation.
* Added support for FXD image file type.

Bug fixes:

* Fixed image levels "Log scale" feature for Python 3 compatibility.

## Version 1.2.2 ##

New features:

* Added "Delete all" entry to "Edit" menu: this removes all objects (signals or
  images) from current view.

* Added an option "hide_on_close" to CodraFTMainWindow class constructor
  (default value is False): when set to True, CodraFT main window will simply
  hide when "Close" button is clicked, which is the expected behavior when
  embedding CodraFT in another application.

Bug fixes:

* The memory leak fix in app.py was accidentally commented before commit.

## Version 1.2.1 ##

Bug fixes:

* When quitting CodraFT, objects were not deleted: this was causing a memory
  leak when embedding CodraFT in another Qt window.

* When canceling HDF5 import dialog box after selecting at least one signal or
  image, the progress bar was shown even if no data was being imported.

* When closing HDF5 import dialog box, preview signal/image widgets were not
  deleted, hence causing another memory leak.

## Version 1.2.0 ##

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

## Version 1.1.2 ##

Bug fixes:

* When the X/Y Cross Section widget is embedded into a main window other than
  CodraFT's, clicking on the "Process signal" button will send the signal to
  CodraFT's signal panel for further processing, as expected.

## Version 1.1.1 ##

Bug fixes:

* Fixed a bug leading to "None" titles when importing signals/images from HDF5
  files created outside CodraFT.

## Version 1.1.0 ##

New features:

* Added new icons.

* Images:

  * Added support for SPIRICON image files (single-frame support only).

Bug fixes:

* Fixed a critical bug when opening HDF5 file (bug from "guidata" package).
  Now guidata is patched inside CodraFT to take into account the unusual/risky
  PyQt patch from Taurus package (PyQt API is set to 2 for QString objects and
  instead of raising an ImportError when importing QString from PyQt4.QtCore,
  QString still exists and is replaced by "str"...).

* Images:

  * Centroid feature: coordinates were mixed up in CodraFT application.

* Signals:

  * Curve fitting (gaussian and lorentzian): fixed amplitude initial value
      for automatic fitting feature
  * FWHM and FW1/e²: fixed amplitude computation for input fit parameters
      and output results

## Version 1.0.0 ##

Copyright © 2018 Codra, Pierre Raybaut, licensed under the terms of the
CECILL License v2.1.

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
  parameters are stored in signal's metadata (a new dictionnary item
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
