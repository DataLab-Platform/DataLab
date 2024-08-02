# Changelog #

See DataLab [roadmap page](https://datalab-platform.com/en/contributing/roadmap.html) for future and past milestones.

## DataLab Version 0.17.0 ##

ℹ️ PlotPy v2.5 is required for this release.

💥 New features and enhancements:

* Menu "Computing" was renamed to "Analysis" for both Signal and Image panels, to better reflect the nature of the features in this menu
* Regions Of Interest (ROIs) are now taken into account everywhere in the application where it makes sense, and not only for the old "Computing" menu (now "Analysis") features. This closes [Issue #93](https://github.com/DataLab-Platform/DataLab/issues/93). If a signal or an image has an ROI defined:
  * Operations are done on the ROI only (except if the operation changes the data shape, or the pixel size for images)
  * Processing features are done on the ROI only (if the destination object data type is compatible with the source object data type, which excludes thresholding, for instance)
  * Analysis features are done on the ROI only, like before
* As a consequence of previous point, and for clarity:
  * The "Edit Regions of interest" and "Remove all Regions of interest" features have been moved from the old "Computing" (now "Analysis") menu to the "Edit" menu where all metadata-related features are located
  * The "Edit Regions of interest" action has been added to both Signal and Image View vertical toolbars (in second position, after the "View in a new window" action)
* Following the bug fix on image data type conversion issues with basic operations, a new "Arithmetic operation" feature has been added to the "Operations" menu for both Signal and Image panels. This feature allows to perform linear operations on signals and images, with the following operations:
  * Addition: ``obj3 = (obj1 + obj2) * a + b``
  * Subtraction: ``obj3 = (obj1 - obj2) * a + b``
  * Multiplication: ``obj3 = (obj1 * obj2) * a + b``
  * Division: ``obj3 = (obj1 / obj2) * a + b``
* Improved "View in a new window" and "ROI editor" dialog boxes size management: default size won't be larger than DataLab's main window size
* ROI editor:
  * Added toolbars for both Signal and Image ROI editors, to allow to zoom in and out, and to reset the zoom level easily
  * Rearranged the buttons in the ROI editor dialog box for better ergonomics and consistency with the Annotations editor ("View in a new window" dialog box)
* Application color theme:
  * Added support for color theme (auto, light, dark) in the "Settings" dialog box
  * The color theme is applied without restarting the application

🛠️ Bug fixes:

* Intensity profile / Segment profile extraction:
  * When extracting a profile on an image with a ROI defined, the associated PlotPy feature show a warning message ('UserWarning: Warning: converting a masked element to nan.') but the profile is correctly extracted and displayed, with NaN values where the ROI is not defined.
  * NaN values are now removed from the profile before plotting it
* Simple processing features with a one-to-on mapping with a Python function (e.g. `numpy.absolute`, `numpy.log10`, etc.) and without parameters: fix result object title which was systematically ending with "|" (the character that usually precedes the list of parameters)
* Butterworth filter: fix cutoff frequency ratio default value and valid range
* Fix actions refresh issue in Image View vertical toolbar:
  * When starting DataLab with the Signal Panel active, switching to the Image View was showing "View in a new window" or "Edit Regions of interest" actions enabled in the vertical toolbar, even if no image was displayed in the Image View
  * The Image View vertical toolbar is now correctly updated at startup
* View in a new window: cross section tools (intensity profiles) stayed disabled unless the user selected an image through the item list - this is now fixed
* Image View: "Show contrast panel" toolbar button was not enabled at startup, and was only enabled when at least one image was displayed in the Image View - it is now always enabled, as expected
* Image data type conversion:
  * Previously, the data type conversion feature was common to signal and image processing features, i.e. a simple conversion of the data type using NumPy's `astype` method
  * This was not sufficient for image processing features, in particular for integer images, because even if the result was correct from a numerical point of view, underflow or overflow could be legitimately seen as a bug from a mathematical point of view
  * The image data type conversion feature now relies on the internal `clip_astype` function, which clips the data to the valid range of the target data type before converting it (in the case of integer images)
* Image ROI extraction issues:
  * Multiple regressions were introduced in version 0.16.0:
    * Single circular ROI extraction was not working as expected (a rectangular ROI was extracted, with unexpected coordinates)
    * Multiple circular ROI extraction lead to a rectangular ROI extraction
    * Multiple ROI extraction was no longer cropping the image to the overall bounding box of the ROIs
  * These issues are now fixed, and unit tests have been added to prevent regressions:
    * An independent test algorithm has been implemented to check the correctness of the ROI extraction in all cases mentioned above
    * Tests cover both single and multiple ROI extraction, with circular and rectangular ROIs
* Overflow and underflow issues in some operations on integer images:
  * When processing integer images, some features were causing overflow or underflow issues, leading to unexpected results (correct results from a numerical point of view, but not from a mathematical point of view)
  * This issue only concerned basic operations (addition, subtraction, multiplication, division, and constant operations) - all the other features were already working as expected
  * This is now fixed as result output are now floating point images
  * Unit tests have been added to prevent regressions for all these operations

## DataLab Version 0.16.4 ##

This is a minor maintenance release.

🛠️ Bug fixes:

* Requires PlotPy v2.4.1 or later to fix the following issues related to the contrast adjustment feature:
  * A regression was introduced in an earlier version of PlotPy: levels histogram was no longer removed from contrast adjustment panel when the associated image was removed from the plot
  * This is now fixed: when an image is removed, the histogram is removed as well and the contrast panel is refreshed (which was not the case even before the regression)
* Ignore `AssertionError` in *config_unit_test.py* when executing test suite on WSL

📚 Documentation:

* Fix class reference in `Wrap11Func` documentation

## DataLab Version 0.16.3 ##

🛠️ Bug fixes:

* Fixed [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84) - Build issues with V0.16.1: `signal` name conflict, ...
  * This issue was intended to be fixed in version 0.16.2, but the fix was not complete
  * Thanks to [@rolandmas](https://github.com/rolandmas) for reporting the issue and for the help in investigating the problem and testing the fix
* Fixed [Issue #85](https://github.com/DataLab-Platform/DataLab/issues/85) - Test data paths may be added multiple times to `cdl.utils.tests.TST_PATH`
  * This issue is related to [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84)
  * Adding the test data paths multiple times to `cdl.utils.tests.TST_PATH` was causing the test data to be loaded multiple times, which lead to some tests failing (a simple workaround was added to V0.16.2: this issue is now fixed)
  * Thanks again to [@rolandmas](https://github.com/rolandmas) for reporting the issue in the context of the Debian packaging
* Fixed [Issue #86](https://github.com/DataLab-Platform/DataLab/issues/86) - Average of N integer images overflows data type
* Fixed [Issue #87](https://github.com/DataLab-Platform/DataLab/issues/87) - Image average profile extraction: `AttributeError` when trying to edit profile parameters
* Fixed [Issue #88](https://github.com/DataLab-Platform/DataLab/issues/88) - Image segment profile: point coordinates inversion

## DataLab Version 0.16.2 ##

This release requires PlotPy v2.4.0 or later, which brings the following bug fixes and new features:

* New constrast adjustment features and bug fixes:
  * New layout: the vertical toolbar (which was constrained in a small area on the right side of the panel) is now a horizontal toolbar at the top of the panel, beside the title
  * New "Set range" button: allows the user to set manually the minimum and maximum values of the histogram range
  * Fixed histogram update issues when no image was currently selected (even if the an image was displayed and was selected before)
  * Histogram range was not updated when either the minimum or maximum value was set using the "Minimum value" or "Maximum value" buttons (which have been renamed to "Min." and "Max." in this release)
  * Histogram range was not updated when the "Set full range" button was clicked, or when the LUT range was modified using the "Scales / LUT range" form in "Properties" group box

* Image view context menu: new "Reverse X axis" feature

ℹ️ Minor new features and enhancements:

* Image file types:
  * Added native support for reading .SPE, .GEL, .NDPI and .REC image files
  * Added support for any `imageio`-supported file format through configuration file (entry `imageio_formats` may be customized to complement the default list of supported formats: see [documentation](https://datalab-platform.com/en/features/image/menu_file.html#open-image) for more details)

🛠️ Bug fixes:

* Image Fourier analysis:
  * Fixed logarithmic scale for the magnitude spectrum (computing dB instead of natural logarithm)
  * Fixed PSD computation with logarithmic scale (computing dB instead of natural logarithm)
  * Updated the documentation to explicitly mention that the logarithmic scale is in dB

* Fixed [Issue #82](https://github.com/DataLab-Platform/DataLab/issues/82) - Macros are not renamed in DataLab after exporting them to Python scripts

* `ResultProperties` object can now be added to `SignalObj` or `ImageObj` metadata even outside a Qt event loop (because the label item is no longer created right away)

* Progress bar is now automatically closed as expected when an error occurrs during a long operation (e.g. when opening a file)

* Difference, division, ...: dialog box for the second operand selection was allowing to select a group (only a signal or an image should be selected)

* When doing an operation which involves an object (signal or image) with higher order number than the current object (e.g. when subtracting an image with an image from a group below the current image), the resulting object's title now correctly refers to the order numbers of the objects involved in the operation (e.g., to continue with the subtraction example mentioned above, the resulting object's title was previously referring to the order number before the insertion of the resulting image)

* Added support for additional test data folder thanks to the `CDL_DATA` environment variable (useful for testing purposes, and especially in the context of Debian packaging)

## DataLab Version 0.16.1 ##

Since version 0.16.0, many validation functions have been added to the test suite. The percentage of validated compute functions has increased from 37% to 84% in this release.

NumPy 2.0 support has been added with this release.

ℹ️ Minor new features and enhancements:

* Signal and image moving average and median filters:
  * Added "Mode" parameter to choose the mode of the filter (e.g. "reflect", "constant", "nearest", "mirror", "wrap")
  * The default mode is "reflect" for moving average and "nearest" for moving median
  * This allows to handle edge effects when filtering signals and images

🛠️ Bug fixes:

* Fixed Canny edge detection to return binary image as `uint8` instead of `bool` (for consistency with other image processing features)

* Fixed Image normalization: lower bound was wrongly set for `maximum` method

* Fixed `ValueError` when computing PSD with logarithmic scale

* Fixed Signal derivative algorithm: now using `numpy.gradient` instead of a custom implementation

* Fixed SciPy's `cumtrapz` deprecation: use `cumulative_trapezoid` instead

* Curve selection now shows the individual points of the curve (before, only the curve line width was broadened)

* Windows installer: add support for unstable releases (e.g., 0.16.1.dev0), thus allowing to easily install the latest development version of DataLab on Windows

* Fixed [Issue #81](https://github.com/DataLab-Platform/DataLab/issues/81) - When opening files, show progress dialog only if necessary

* Fixed [Issue #80](https://github.com/DataLab-Platform/DataLab/issues/80) - Plotting results: support for two use cases
  * The features of the "Analysis" menu produce *results* (scalars): blob detection (circle coordinates), 2D peak detection (point coordinates), etc. Depending on the feature, result tables are displayed in the "Results" dialog box, and the results are also stored in the signal or image metadata: each line of the result table is an individual result, and each column is a property of the result - some results may consist only of a single individual result (e.g., image centroid or curve FHWM), while others may consist of multiple individual results (e.g., blob detection, contour detection, etc.).
  * Before this change, the "Plot results" feature only supported plotting the first individual result of a result table, as a function of the index (of the signal or image objects) or any of the columns of the result table. This was not sufficient for some use cases, where the user wanted to plot multiple individual results of a result table.
  * Now, the "Plot results" feature supports two use cases:
    * "One curve per result title": Plotting the first individual result of a result table, as before
    * "One curve per object (or ROI) and per result title": Plotting all individual results of a result table, as a function of the index (of the signal or image objects) or any of the columns of the result table
  * The selection of the use case is done in the "Plot results" dialog box
  * The default use case is "One curve per result title" if the result table has only one line, and "One curve per object (or ROI) and per result title" otherwise

## DataLab Version 0.16.0 ##

💥 New features and enhancements:

* Major user interface overhaul:
  * The menu bar and toolbars have been reorganized to make the application more intuitive and easier to use
  * Operations and processing features have been regrouped in submenus
  * All visualization-related actions are now grouped in the plot view vertical toolbar
  * Clarified the "Annotations" management (new buttons, toolbar action, ...)

* New validation process for signal and image features:
  * Before this release, DataLab's validation process was exclusively done from the programmer's point of view, by writing unit tests and integration tests, thus ensuring that the code was working as expected (i.e. that no exception was raised and that the behavior was correct)
  * With this release, a new validation process has been introduced, from the user's point of view, by adding new validation functions (marked with the `@pytest.mark.validation` decorator) in the test suite
  * A new "Validation" section in the documentation explains how validation is done and contains a list of all validation functions with the statistics of the validation process (generated from the test suite)
  * The validation process is a work in progress and will be improved in future versions

* "Properties" group box:
  * Added "Scales" tab, to show and set the plot scales:
    * X, Y for signals
    * X, Y, Z (LUT range) for images

* View options:
  * New "Show first only" option in the "View" menu, to show only the first curve (or image) when multiple curves (or images) are displayed in the plot view
  * New (movable) label for FWHM computations, additional to the existing segment annotation

* I/O features:
  * Added support for reading and writing .MAT files (MATLAB format)
  * Create a new group when opening a file containing multiple signals or images (e.g. CSV file with multiple curves)

* Add support for binary images
* Signal ROI extraction: added new dialog box to manually edit the ROI lower and upper bounds after defining the ROI graphically

ℹ️ New **Signal** operations, processing and analysis features:

| Menu        | Submenu      |Features                                                 |
|-------------|--------------|---------------------------------------------------------|
| New | New signal | Exponential, pulse, polynomial, experimental (manual input)            |
| Operations  | | Exponential, Square root, Power |
| Operations  | Operations with a constant | +, -, *, / |
| Processing  | Axis Transformation | Reverse X-axis |
| Processing  | Level Adjustment | Offset correction |
| Processing  | Fourier analysis | Power spectrum, Phase spectrum, Magnitude spectrum, Power spectral density |
| Processing  | Frequency filters | Low-pass, High-pass, Band-pass, Band-stop |
| Processing  | | Windowing (Hanning, Hamming, Blackman, Blackman-Harris, Nuttall, Flat-top, ...) |
| Processing  | Fit | Linear fit, Sinusoidal fit, Exponential fit, CDF fit |
| Analysis   | | FWHM (Zero-crossing method), X value @ min/max, Sampling period/frequency, Dynamic parameters (ENOB, SNR, SINAD, THD, SFDR), -3dB bandwidth, Contrast |

ℹ️ New **Image** operations, processing and analysis features:

| Menu        | Submenu      |Features                                                 |
|-------------|--------------|---------------------------------------------------------|
| Operations  | | Exponential |
| Operations  | Intensity profiles | Profile along a segment |
| Operations  | Operations with a constant | +, -, *, / |
| Processing  | Level Adjustment | Normalization, Clipping, Offset correction |
| Processing  | Fourier analysis | Power spectrum, Phase spectrum, Magnitude spectrum, Power spectral density |
| Processing  | Thresholding | Parametric, ISODATA, Li, Mean, Minimum, Otsu, Triangle, Yen |

🛠️ Bug fixes:

* Fixed a performance issue due to an unnecessary refresh of the plot view when adding a new signal or image
* Fixed [Issue #77](https://github.com/DataLab-Platform/DataLab/issues/77) - Intensity profiles: unable to accept dialog the second time
* Fixed [Issue #75](https://github.com/DataLab-Platform/DataLab/issues/75) - View in a new window: curve anti-aliasing is not enabled by default
* Annotations visibility is now correctly saved and restored:
  * Before this release, when modifying the annotations visibility in the separate plot view, the visibility was not saved and restored when reopening the plot view
  * This has been [fixed upstream](https://github.com/PlotPyStack/PlotPy/commit/03faaa42e5d6d4016ea8c99334c29d46a5963467) in PlotPy (v2.3.3)

## DataLab Version 0.15.1 ##

🛠️ Bug fixes:

* Fixed [Issue #68](https://github.com/DataLab-Platform/DataLab/issues/68) - Slow loading of even simple plots:
  * On macOS, the user experience was degraded when handling even simple plots
  * This was due to the way macOS handles the pop-up windows, e.g. when refreshing the plot view ("Creating plot items" progress bar), hence causing a very annoying flickering effect and a global slowdown of the application
  * This is now fixed by showing the progress bar only after a short delay (1s), that is when it is really needed (i.e. for long operations)
  * Thanks to [@marcel-goldschen-ohm](https://github.com/marcel-goldschen-ohm) for the very thorough feedback and the help in testing the fix
* Fixed [Issue #69](https://github.com/DataLab-Platform/DataLab/issues/69) - Annotations should be read-only in Signal/Image View
  * Regarding the annotations, DataLab's current behavior is the following:
    * Annotations are created only when showing the signal/image in a separate window (double-click on the object, or "View" > "View in a new window")
    * When displaying the objects in either the "Signal View" or the "Image View", the annotations should be read-only (i.e. not movable, nor resizable or deletable)
  * However, some annotations were still deletable in the "Signal View" and the "Image View": this is now fixed
  * Note that the fact that annotations can't be created in the "Signal View" or the "Image View" is a limitation of the current implementation, and may be improved in future versions

## DataLab Version 0.15.0 ##

🎁 New installer for the stand-alone version on Windows:

* The stand-alone version on Windows is now distributed as an MSI installer (instead of an EXE installer)
* This avoids the false positive detection of the stand-alone version as a potential threat by some antivirus software
* The program will install files and shortcuts:
  * For current user, if the user has no administrator privileges
  * For all users, if the user has administrator privileges
  * Installation directory may be customized
* MSI installer allows to integrate DataLab's installation seemlessly in an organization's deployment system

💥 New features and enhancements:

* Added support for large text/CSV files:
  * Files over 1 GB (and with reasonable number of lines) can now be imported as signals or images without crashing the application or even slowing it down
  * The file is read by chunks and, for signals, the data is downsampled to a reasonable number of points for visualization
  * Large files are supported when opening a file (or dragging and dropping a file in the Signal Panel) and when importing a file in the Text Import Wizard
* Auto downsampling feature:
  * Added "Auto downsampling" feature to signal visualization settings (see "Settings" dialog box)
  * This feature allows to automatically downsample the signal data for visualization when the number of points is too high and would lead to a slow rendering
  * The downsampling factor is automatically computed based on the configured maximum number of points to display
  * This feature is enabled by default and may be disabled in the signal visualization settings
* CSV format handling:
  * Improved support for CSV files with a header row (column names)
  * Added support for CSV files with empty columns
* Open/save file error handling:
  * Error messages are now more explicit when opening or saving a file fails
  * Added a link to the folder containing the file in the error message
* Added "Plugins and I/O formats" page to the Installation and Configuration Viewer (see "Help" menu)
* Reset DataLab configuration:
  * In some cases, it may be useful to reset the DataLab configuration file to its default values (e.g. when the configuration file is corrupted)
  * Added new `--reset` command line option to remove the configuration folder
  * Added new "Reset DataLab" Start Menu shortcut to the Windows installer

🛠️ Bug fixes:

* Fixed [Issue #64](https://github.com/DataLab-Platform/DataLab/issues/64) - HDF5 browser does not show datasets with 1x1 size:
  * HDF5 datasets with a size of 1x1 were not shown in the HDF5 browser
  * Even if those datasets should not be considered as signals or images, they are now shown in the HDF5 browser (but not checkable, i.e. not importable as signals or images)

## DataLab Version 0.14.2 ##

⚠️ API changes required for fixing support for multiple signals loading feature:

* Merged `open_object` and `open_objects` methods to `load_from_files` in proxy classes, main window and data panels
* For consistency's sake: merged `save_object` and `save_objects` into `save_to_files`
* To sum up, those changes lead to the following situation:
  * `load_from_files`: load a sequence of objects from multiple files
  * `save_to_files`: save a sequence of objects to multiple files (at the moment, it only supports saving a single object to a single file, but it may be extended in the future to support saving multiple objects to a single file)

🛠️ Bug fixes:

* Fixed [Issue #61](https://github.com/DataLab-Platform/DataLab/issues/61) - Text file import wizard: application crash when importing a multiple curve text file:
  * This issue concerns a use case where the text file contains multiple curves
  * This is now fixed and an automatic test has been added to prevent regressions

## DataLab Version 0.14.1 ##

🎉 New domain name: [datalab-platform.com](https://datalab-platform.com)

💥 New features:

* Added support for colormap inversion in Image View:
  * New "Invert colormap" entry in plot context menu, image parameters, and in the default image view settings
  * This requires `PlotPy` v2.3 or later
* HDF5 Browser:
  * Added "Show array" button at the corner of the "Group" and "Attributes" tabs, to show the array in a separate window (useful for copy/pasting data to other applications, for instance)
  * Attributes: added support for more scalar data types
* Testability and maintainability:
  * DataLab's unit tests are now using [pytest](https://pytest.org). This has required a lot of work for the transition, especially to readapt the tests so that they may be executed in the same process. For instance, a particular attention has been given to sandboxing the tests, so that they do not interfere with each other.
  * Added continuous integration (CI) with GitHub Actions
  * For this release, test coverage is 87%
* Text file import assistant:
  * Drastically improved the performance of the array preview when importing large text files (no more progress bar, and the preview is now displayed almost instantaneously)

🛠️ Bug fixes:

* XML-RPC server was not shut down properly when closing DataLab
* Fixed test-related issues: some edge cases were hidden by the old test suite, and have been revealed by the transition to `pytest`. This has led to some bug fixes and improvements in the code.
* On Linux, when running a computation on a signal or an image, and on rare occasions, the computation was stuck as if it was running indefinitely. Even though the graphical user interface was still responsive, the computation was not progressing and the user had to cancel the operation and restart it. This was due to the start method of the separate process used for the computation (default method was "fork" on Linux). This is now fixed by using the "spawn" method instead, which is the recommended method for latest versions of Python on Linux when multithreading is involved.
* Fixed [Issue #60](https://github.com/DataLab-Platform/DataLab/issues/60) - `OSError: Invalid HDF5 file [...]` when trying to open an HDF5 file with an extension other than ".h5"
* Image Region of Interest (ROI) extraction: when modifying the image bounds in the confirmation dialog box, the ROI was not updated accordingly until the operation was run again
* Deprecation issues:
  * Fixed `scipy.ndimage.filters` deprecation warning
  * Fixed `numpy.fromstring` deprecation warning

## DataLab Version 0.14.0 ##

💥 New features:

* New "Histogram" feature in "Analysis" menu:
  * Added histogram computation feature for both signals and images
  * The histogram is computed on the regions of interest (ROI) if any, or on the whole signal/image if no ROI is defined
  * Editable parameters: number of bins, lower and upper bounds
* HDF5 browser:
  * Improved tree view layout (more compact and readable)
  * Multiple files can now be opened at once, using the file selection dialog box
  * Added tabs with information below the graphical preview:
    * Group info: path, textual preview, etc.
    * Attributes info: name, value
  * Added "Show only supported data" check box: when checked, only supported data (signals and images) are shown in the tree view
  * Added "Show values" check box, to show/hide the values in the tree view
* Macro Panel:
  * Macro commands are now numbered, starting from 1, like signals and images
* Remote control API (`RemoteProxy` and `LocalProxy`):
  * `get_object_titles` method now accepts "macro" as panel name and returns the list of macro titles
  * New `run_macro`, `stop_macro` and `import_macro_from_file` methods

🛠️ Bug fixes:

* Stand-alone version - Integration in Windows start menu:
  * Fixed "Uninstall" shortcut (unclickable due to a generic name)
  * Translated "Browse installation directory" and "Uninstall" shortcuts
* Fixed [Issue #55](https://github.com/DataLab-Platform/DataLab/issues/55) - Changing image bounds in Image View has no effect on the associated image object properties
* Fixed [Issue #56](https://github.com/DataLab-Platform/DataLab/issues/56) - "Test data" plugin: `AttributeError: 'NoneType' object has no attribute 'data'` when canceling "Create image with peaks"
* Fixed [Issue #57](https://github.com/DataLab-Platform/DataLab/issues/57) - Circle and ellipse result shapes are not transformed properly
* Curve color and style cycle:
  * Before this release, this cycle was handled by the same mechanism either for the Signal Panel or the HDF5 Browser, which was not the expected behavior
  * Now, the cycle is handled separately: the HDF5 Browser or the Text Import Wizard use always the same color and style for curves, and they don't interfere with the Signal Panel cycle

## DataLab Version 0.12.0 ##

🧹 Clarity-Enhanced Interface Update:

* The tabs used to switch between the data panels (signals and images) and the visualization components ("Curve panel" and "Image panel") have been renamed to "Signal Panel" and "Image Panel" (instead of "Signals" and "Images")
* The visualization components have been renamed to "Signal View" and "Image View" (instead of "Curve panel" and "Image panel")
* The data panel toolbar has been renamed to "Signal Toolbar" and "Image Toolbar" (instead of "Signal Processing Toolbar" and "Image Processing Toolbar")
* Ergonomics improvements: the "Signal Panel" and "Image Panel" are now displayed on the left side of the main window, and the "Signal View" and "Image View" are displayed on the right side of the main window. This reduces the distance between the list of objects (signals and images) and the associated actions (toolbars and menus), and makes the interface more intuitive and easier to use

✨ New tour and demo feature:

* When starting DataLab for the first time, an optional tour is now shown to the user to introduce the main features of the application
* The tour can be started again at any time from the "?" menu
* Also added a new "Demo" feature to the "?" menu

🚀 New Binder environment to test DataLab online without installing anything

📚 Documentation:

* New text tutorials are available:
  * Measuring Laser Beam Size
  * DataLab and Spyder: a perfect match
* "Getting started" section: added more explanations and links to the tutorials
* New "Contributing" section explaining how to contribute to DataLab, whether you are a developer or not
* New "Macros" section explaining how to use the macro commands feature
* Added "Copy" button to code blocks in the documentation

💥 New features:

* New "Text file import assistant" feature:
  * This feature allows to import text files as signals or images
  * The user can define the source (clipboard or texte file)
  * Then, it is possible to define the delimiter, the number of rows to skip, the destination data type, etc.
* Added menu on the "Signal Panel" and "Image Panel" tabs corner to quickly access the most used features (e.g. "Add", "Remove", "Duplicate", etc.)
* Intensity profile extraction feature:
  * Added graphical user interface to extract intensity profiles from images, for both line and averaged profiles
  * Parameters are still directly editable by the user ("Edit profile parameters" button)
  * Parameters are now stored from one profile extraction to another
* Statistics feature:
  * Added `<y>/σ(y)` to the signal "Statistics" result table (in addition to the mean, median, standard deviation, etc.)
  * Added `peak-to-peak` to the signal and image "Statistics" result table
* Curve fitting feature: fit results are now stored in a dictionary in the signal metadata (instead of being stored individually in the signal metadata)
* Window state:
  * The toolbars and dock widgets state (visibility, position, etc.) are now stored in the configuration file and restored at startup (size and position were already stored and restored)
  * This implements part of [Issue #30](https://github.com/DataLab-Platform/DataLab/issues/30) - Save/restore main window layout

🛠️ Bug fixes:

* Fixed [Issue #41](https://github.com/DataLab-Platform/DataLab/issues/41) - Radial profile extraction: unable to enter user-defined center coordinates
* Fixed [Issue #49](https://github.com/DataLab-Platform/DataLab/issues/49) - Error when trying to open a (UTF-8 BOM) text file as an image
* Fixed [Issue #51](https://github.com/DataLab-Platform/DataLab/issues/51) - Unexpected dimensions when adding new ROI on an image with X/Y arbitrary units (not pixels)
* Improved plot item style serialization management:
  * Before this release, the plot item style was stored in the signal/image metadata only when saving the workspace to an HDF5 file. So, when modifying the style of a signal/image from the "Parameters" button (view toolbar), the style was not kept in some cases (e.g. when duplicating the signal/image).
  * Now, the plot item style is stored in the signal/image metadata whenever the style is modified, and is restored when reloading the workspace
* Handled `ComplexWarning` cast warning when adding regions of interest (ROI) to a signal with complex data

## DataLab Version 0.11.0 ##

💥 New features:

* Signals and images may now be reordered in the tree view:
  * Using the new "Move up" and "Move down" actions in the "Edit" menu (or using the corresponding toolbar buttons):
  * This fixes [Issue #22](https://github.com/DataLab-Platform/DataLab/issues/22) - Add "move up/down" actions in "Edit" menu, for signals/images and groups
* Signals and images may also be reordered using drag and drop:
  * Signals and images can be dragged and dropped inside their own panel to change their order
  * Groups can also be dragged and dropped inside their panel
  * The feature also supports multi-selection (using the standard Ctrl and Shift modifiers), so that multiple signals/images/groups can be moved at once, not necessarily with contiguous positions
  * This fixes [Issue #17](https://github.com/DataLab-Platform/DataLab/issues/17) - Add Drag and Drop feature to Signals/Images tree views
* New 1D interpolation features:
  * Added "Interpolation" feature to signal panel's "Processing" menu
  * Methods available: linear, spline, quadratic, cubic, barycentric and PCHIP
  * Thanks to [@marcel-goldschen-ohm](https://github.com/marcel-goldschen-ohm) for the contribution to spline interpolation
  * This fixes [Issue #20](https://github.com/DataLab-Platform/DataLab/issues/20) - Add 1D interpolation features
* New 1D resampling feature:
  * Added "Resampling" feature to signal panel's "Processing" menu
  * Same interpolation methods as for the "Interpolation" feature
  * Possibility to specify the resampling step or the number of points
  * This fixes [Issue #21](https://github.com/DataLab-Platform/DataLab/issues/21) - Add 1D resampling feature
* New 1D convolution feature:
  * Added "Convolution" feature to signal panel's "Operation" menu
  * This fixes [Issue #23](https://github.com/DataLab-Platform/DataLab/issues/23) - Add 1D convolution feature
* New 1D detrending feature:
  * Added "Detrending" feature to signal panel's "Processing" menu
  * Methods available: linear or constant
  * This fixes [Issue #24](https://github.com/DataLab-Platform/DataLab/issues/24) - Add 1D detrending feature
* 2D analysis results:
  * Before this release, 2D analysis results such as contours, blobs, etc. were stored in image metadata dictionary as coordinates (x0, y0, x1, y1, ...) even for circles and ellipses (i.e. the coordinates of the bounding rectangles).
  * For convenience, the circle and ellipse coordinates are now stored in image metadata dictionary as (x0, y0, radius) and (x0, y0, a, b, theta) respectively.
  * These results are also shown as such in the "Results" dialog box (either at the end of the computing process or when clicking on the "Show results" button).
  * This fixes [Issue #32](https://github.com/DataLab-Platform/DataLab/issues/32) - Contour detection: show circle `(x, y, r)` and ellipse `(x, y, a, b, theta)` instead of `(x0, y0, x1, x1, ...)`
* 1D and 2D analysis results:
  * Additionnaly to the previous enhancement, more analysis results are now shown in the "Results" dialog box
  * This concerns both 1D (FHWM, ...) and 2D analysis results (contours, blobs, ...):
    * Segment results now also show length (L) and center coordinates (Xc, Yc)
    * Circle and ellipse results now also show area (A)
* Added "Plot results" entry in "Analysis" menu:
  * This feature allows to plot analysis results (1D or 2D)
  * It creates a new signal with X and Y axes corresponding to user-defined parameters (e.g. X = indexes and Y = radius for circle results)
* Increased default width of the object selection dialog box:
  * The object selection dialog box is now wider by default, so that the full signal/image/group titles may be more easily readable
* Delete metadata feature:
  * Before this release, the feature was deleting all metadata, including the Regions Of Interest (ROI) metadata, if any.
  * Now a confirmation dialog box is shown to the user before deleting all metadata if the signal/image has ROI metadata: this allows to keep the ROI metadata if needed.
* Image profile extraction feature: added support for masked images (when defining regions of interest, the areas outside the ROIs are masked, and the profile is extracted only on the unmasked areas, or averaged on the unmasked areas in the case of average profile extraction)
* Curve style: added "Reset curve styles" in "View" menu. This feature allows to reset the curve style cycle to its initial state.
* Plugin base classe `PluginBase`:
  * Added `edit_new_signal_parameters` method for showing a dialog box to edit parameters for a new signal
  * Added `edit_new_image_parameters` method for showing a dialog box to edit parameters for a new image (updated the *cdl_testdata.py* plugin accordingly)
* Signal and image computations API (`cdl.computations`):
  * Added wrappers for signal and image 1 -> 1 computations
  * These wrappers aim at simplifying the creation of a basic computation function operating on DataLab's native objects (`SignalObj` and `ImageObj`) from a function operating on NumPy arrays
  * This simplifies DataLab's internals and makes it easier to create new computing features inside plugins
  * See the *cdl_custom_func.py* example plugin for a practical use case
* Added "Radial profile extraction" feature to image panel's "Operation" menu:
  * This feature allows to extract a radially averaged profile from an image
  * The profile is extracted around a user-defined center (x0, y0)
  * The center may also be computed (centroid or image center)
* Automated test suite:
  * Since version 0.10, DataLab's proxy object has a `toggle_auto_refresh` method to toggle the "Auto-refresh" feature. This feature may be useful to improve performance during the execution of test scripts
  * Test scenarios on signals and images are now using this feature to improve performance
* Signal and image metadata:
  * Added "source" entry to the metadata dictionary, to store the source file path when importing a signal or an image from a file
  * This field is kept while processing the signal/image, in order to keep track of the source file path

📚 Documentation:

* New [Tutorial section](https://datalab-platform.com/en/intro/tutorials/index.html) in the documentation:
  * This section provides a set of tutorials to learn how to use DataLab
  * The following video tutorials are available:
    * Quick demo
    * Adding your own features
  * The following text tutorials are available:
    * Processing a spectrum
    * Detecting blobs on an image
    * Measuring Fabry-Perot fringes
    * Prototyping a custom processing pipeline
* New [API section](https://datalab-platform.com/en/api/index.html) in the documentation:
  * This section explains how to use DataLab as a Python library, by covering the following topics:
    * How to use DataLab algorithms on NumPy arrays
    * How to use DataLab computation features on DataLab objects (signals and images)
    * How to use DataLab I/O features
    * How to use proxy objects to control DataLab remotely
  * This section also provides a complete API reference for DataLab objects and features
  * This fixes [Issue #19](https://github.com/DataLab-Platform/DataLab/issues/19) - Add API documentation (data model, functions on arrays or signal/image objects, ...)

🛠️ Bug fixes:

* Fixed [Issue #29](https://github.com/DataLab-Platform/DataLab/issues/29) - Polynomial fit error: `QDialog [...] argument 1 has an unexpected type 'SignalProcessor'`
* Image ROI extraction feature:
  * Before this release, when extracting a single circular ROI from an image with the "Extract all regions of interest into a single image object" option enabled, the result was a single image without the ROI mask (the ROI mask was only available when extracting ROI with the option disabled)
  * This was leading to an unexpected behavior, because one could interpret the result (a square image without the ROI mask) as the result of a single rectangular ROI
  * Now, when extracting a single circular ROI from an image with the "Extract all regions of interest into a single image object" option enabled, the result is a single image with the ROI mask (as if the option was disabled)
  * This fixes [Issue #31](https://github.com/DataLab-Platform/DataLab/issues/31) - Single circular ROI extraction: automatically switch to `extract_single_roi` function
* Analysis on circular ROI:
  * Before this release, when running computations on a circular ROI, the results were unexpected in terms of coordinates (results seemed to be computed in a region located above the actual ROI).
  * This was due to a regression introduced in an earlier release.
  * Now, when defining a circular ROI and running computations on it, the results are computed on the actual ROI
  * This fixes [Issue #33](https://github.com/DataLab-Platform/DataLab/issues/33) - Analysis on circular ROI: unexpected results
* Contour detection on ROI:
  * Before this release, when running contour detection on a ROI, some contours were detected outside the ROI (it may be due to a limitation of the scikit-image `find_contours` function).
  * Now, thanks a workaround, the erroneous contours are filtered out.
  * A new test module `cdl.tests.features.images.contour_fabryperot_app` has been added to test the contour detection feature on a Fabry-Perot image (thanks to [@emarin2642](https://github.com/emarin2642) for the contribution)
  * This fixes [Issue #34](https://github.com/DataLab-Platform/DataLab/issues/34) - Contour detection: unexpected results outside ROI
* Analysis result merging:
  * Before this release, when doing a `1->N` computation (sum, average, product) on a group of signals/images, the analysis results associated to each signal/image were merged into a single result, but only the type of result present in the first signal/image was kept.
  * Now, the analysis results associated to each signal/image are merged into a single result, whatever the type of result is.
* Fixed [Issue #36](https://github.com/DataLab-Platform/DataLab/issues/36) - "Delete all" action enable state is sometimes not refreshed
* Image X/Y swap: when swapping X and Y axes, the regions of interest (ROI) were not removed and not swapped either (ROI are now removed, until we implement the swap feature, if requested)
* "Properties" group box: the "Apply" button was enabled by default, even when no property was modified, which was confusing for the user (the "Apply" button is now disabled by default, and is enabled only when a property is modified)
* Fixed proxy `get_object` method when there is no object to return (`None` is returned instead of an exception)
* Fixed `IndexError: list index out of range` when performing some operations or computations on groups of signals/images (e.g. "ROI extraction", "Peak detection", "Resize", etc.)
* Drag and drop from a file manager: filenames are now sorted alphabetically

## DataLab Version 0.10.1 ##

*Note*: V0.10.0 was almost immediately replaced by V0.10.1 due to a last minute bug fix

💥 New features:

* Features common to signals and images:
  * Added "Real part" and "Imaginary part" features to "Operation" menu
  * Added "Convert data type" feature to "Operation" menu
* Features added following user requests (12/18/2023 meetup @ CEA):
  * Curve and image styles are now saved in the HDF5 file:
    * Curve style covers the following properties: color, line style, line width, marker style, marker size, marker edge color, marker face color, etc.
    * Image style covers the following properties: colormap, interpolation, etc.
    * Those properties were already persistent during the working session, but were lost when saving and reloading the HDF5 file
    * Now, those properties are saved in the HDF5 file and are restored when reloading the HDF5 file
  * New profile extraction features for images:
    * Added "Line profile" to "Operations" menu, to extract a profile from an image along a row or a column
    * Added "Average profile" to "Operations" menu, to extract the average profile on a rectangular area of an image, along a row or a column
  * Image LUT range (contrast/brightness settings) is now saved in the HDF5 file:
    * As for curve and image styles, the LUT range was already persistent during the working session, but was lost when saving and reloading the HDF5 file
    * Now, the LUT range is saved in the HDF5 file and is restored when reloading it
  * Added "Auto-refresh" and "Refresh manually" actions in "View" menu (and main toolbar):
    * When "Auto-refresh" is enabled (default), the plot view is automatically refreshed when a signal/image is modified, added or removed. Even though the refresh is optimized, this may lead to performance issues when working with large datasets.
    * When disabled, the plot view is not automatically refreshed. The user must manually refresh the plot view by clicking on the "Refresh manually" button in the main toolbar or by pressing the standard refresh key (e.g. "F5").
  * Added `toggle_auto_refresh` method to DataLab proxy object:
    * This method allows to toggle the "Auto-refresh" feature from a macro-command, a plugin or a remote control client.
    * A context manager `context_no_refresh` is also available to temporarily disable the "Auto-refresh" feature from a macro-command, a plugin or a remote control client. Typical usage:

      ```python
      with proxy.context_no_refresh():
          # Do something without refreshing the plot view
          proxy.compute_fft() # (...)      ```

  * Improved curve readability:
    * Until this release, the curve style was automatically set by cycling through
      **PlotPy** predefined styles
    * However, some styles are not suitable for curve readability (e.g. "cyan" and "yellow" colors are not readable on a white background, especially when combined with a "dashed" line style)
    * This release introduces a new curve style management with colors which are distinguishable and accessible, even to color vision deficiency people
* Added "Curve anti-aliasing" feature to "View" menu (and toolbar):
  * This feature allows to enable/disable curve anti-aliasing (default: enabled)
  * When enabled, the curve rendering is smoother but may lead to performance issues when working with large datasets (that's why it can be disabled)
* Added `toggle_show_titles` method to DataLab proxy object. This method allows to toggle the "Show graphical object titles" feature from a macro-command, a plugin or a remote control client.
* Remote client is now checking the server version and shows a warning message if the server version may not be fully compatible with the client version.

🛠️ Bug fixes:

* Image contour detection feature ("Analysis" menu):
  * The contour detection feature was not taking into account the "shape" parameter (circle, ellipse, polygon) when computing the contours. The parameter was stored but really used only when calling the feature a second time.
  * This unintentional behavior led to an `AssertionError` when choosing "polygon" as the contour shape and trying to compute the contours for the first time.
  * This is now fixed (see [Issue #9](https://github.com/DataLab-Platform/DataLab/issues/9) - Image contour detection: `AssertionError` when choosing "polygon" as the contour shape)
* Keyboard shortcuts:
  * The keyboard shortcuts for "New", "Open", "Save", "Duplicate", "Remove", "Delete all" and "Refresh manually" actions were not working properly.
  * Those shortcuts were specific to each signal/image panel, and were working only when the panel on which the shortcut was pressed for the first time was active (when activated from another panel, the shortcut was not working and a warning message was displayed in the console, e.g. `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
  * Besides, the shortcuts were not working at startup (when no panel had focus).
  * This is now fixed: the shortcuts are now working whatever the active panel is, and even at startup (see [Issue #10](https://github.com/DataLab-Platform/DataLab/issues/10) - Keyboard shortcuts not working properly: `QAction::event: Ambiguous shortcut overload: Ctrl+C`)
* "Show graphical object titles" and "Auto-refresh" actions were not working properly:
  * The "Show graphical object titles" and "Auto-refresh" actions were only working on the active signal/image panel, and not on all panels.
  * This is now fixed (see [Issue #11](https://github.com/DataLab-Platform/DataLab/issues/11) - "Show graphical object titles" and "Auto-refresh" actions were working only on current signal/image panel)
* Fixed [Issue #14](https://github.com/DataLab-Platform/DataLab/issues/14) - Saving/Reopening HDF5 project without cleaning-up leads to `ValueError`
* Fixed [Issue #15](https://github.com/DataLab-Platform/DataLab/issues/15) - MacOS: 1. `pip install cdl` error - 2. Missing menus:
  * Part 1: `pip install cdl` error on MacOS was actually an issue from **PlotPy** (see [this issue](https://github.com/PlotPyStack/PlotPy/issues/9)), and has been fixed in PlotPy v2.0.3 with an additional compilation flag indicating to use C++11 standard
  * Part 2: Missing menus on MacOS was due to a PyQt/MacOS bug regarding dynamic menus
* HDF5 file format: when importing an HDF5 dataset as a signal or an image, the dataset attributes were systematically copied to signal/image metadata: we now only copy the attributes which match standard data types (integers, floats, strings) to avoid errors when serializing/deserializing the signal/image object
* Installation/configuration viewer: improved readability (removed syntax highlighting)
* PyInstaller specification file: added missing `skimage` data files manually in order to continue supporting Python 3.8 (see [Issue #12](https://github.com/DataLab-Platform/DataLab/issues/12) - Stand-alone version on Windows 7: missing `api-ms-win-core-path-l1-1-0.dll`)
* Fixed [Issue #13](https://github.com/DataLab-Platform/DataLab/issues/13) - ArchLinux: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found`

## DataLab Version 0.9.2 ##

🛠️ Bug fixes:

* Region of interest (ROI) extraction feature for images:
  * ROI extraction was not working properly when the "Extract all regions of interest into a single image object" option was enabled if there was only one defined ROI. The result was an image positioned at the origin (0, 0) instead of the expected position (x0, y0) and the ROI rectangle itself was not removed as expected. This is now fixed (see [Issue #6](https://github.com/DataLab-Platform/DataLab/issues/6) - 'Extract multiple ROI' feature: unexpected result for a single ROI)
  * ROI rectangles with negative coordinates were not properly handled: ROI extraction was raising a `ValueError` exception, and the image mask was not displayed properly. This is now fixed (see [Issue #7](https://github.com/DataLab-Platform/DataLab/issues/7) - Image ROI extraction: `ValueError: zero-size array to reduction operation minimum which has no identity`)
  * ROI extraction was not taking into account the pixel size (dx, dy) and the origin (x0, y0) of the image. This is now fixed (see [Issue #8](https://github.com/DataLab-Platform/DataLab/issues/8) - Image ROI extraction: take into account pixel size)
* Macro-command console is now read-only:
  * The macro-command panel Python console is currently not supporting standard input stream (`stdin`) and this is intended (at least for now)
  * Set Python console read-only to avoid confusion

## DataLab Version 0.9.1 ##

🛠️ Bug fixes:

* French translation is not available on Windows/Stand alone version:
  * Locale was not properly detected on Windows for stand-alone version (frozen with `pyinstaller`) due to an issue with `locale.getlocale()` (function returning `None` instead of the expected locale on frozen applications)
  * This is ultimately a `pyinstaller` issue, but a workaround has been implemented in `guidata` V3.2.2 (see [guidata issue #68](https://github.com/PlotPyStack/guidata/issues/68) - Windows: gettext translation is not working on frozen applications)
  * [Issue #2](https://github.com/DataLab-Platform/DataLab/issues/2) - French translation is not available on Windows Stand alone version
* Saving image to JPEG2000 fails for non integer data:
  * JPEG2000 encoder does not support non integer data or signed integer data
  * Before, DataLab was showing an error message when trying to save incompatible data to JPEG2000: this was not a consistent behavior with other standard image formats (e.g. PNG, JPG, etc.) for which DataLab was automatically converting data to the appropriate format (8-bit unsigned integer)
  * Current behavior is now consistent with other standard image formats: when saving to JPEG2000, DataLab automatically converts data to 8-bit unsigned integer or 16-bit unsigned integer (depending on the original data type)
  * [Issue #3](https://github.com/DataLab-Platform/DataLab/issues/3) - Save image to JPEG2000: 'OSError: encoder error -2 when writing image file'
* Windows stand-alone version shortcuts not showing in current user start menu:
  * When installing DataLab on Windows from a non-administrator account, the shortcuts were not showing in the current user start menu but in the administrator start menu instead (due to the elevated privileges of the installer and the fact that the installer does not support installing shortcuts for all users)
  * Now, the installer *does not* ask for elevated privileges anymore, and shortcuts are installed in the current user start menu (this also means that the current user must have write access to the installation directory)
  * In future releases, the installer will support installing shortcuts for all users if there is a demand for it (see [Issue #5](https://github.com/DataLab-Platform/DataLab/issues/5))
  * [Issue #4](https://github.com/DataLab-Platform/DataLab/issues/4) - Windows: stand-alone version shortcuts not showing in current user start menu
* Installation and configuration window for stand-alone version:
  * Do not show ambiguous error message 'Invalid dependencies' anymore
  * Dependencies are supposed to be checked when building the stand-alone version
* Added PDF documentation to stand-alone version:
  * The PDF documentation was missing in previous release
  * Now, the PDF documentation (in English and French) is included in the stand-alone version

## DataLab Version 0.9.0 ##

New dependencies:

* DataLab is now powered by [PlotPyStack](https://github.com/PlotPyStack):
  * [PythonQwt](https://github.com/PlotPyStack/PythonQwt)
  * [guidata](https://github.com/PlotPyStack/guidata)
  * [PlotPy](https://github.com/PlotPyStack/PlotPy)
* [opencv-python](https://pypi.org/project/opencv-python/) (algorithms for image processing)

New reference platform:

* DataLab is validated on Windows 11 with Python 3.11 and PyQt 5.15
* DataLab is also compatible with other OS (Linux, MacOS) and other Python-Qt bindings and versions (Python 3.8-3.12, PyQt6, PySide6)

New features:

* DataLab is a platform:
  * Added support for plugins
    * Custom processing features available in the "Plugins" menu
    * Custom I/O features: new file formats can be added to the standard I/O features for signals and images
    * Custom HDF5 features: new HDF5 file formats can be added to the standard HDF5 import feature
    * More features to come...
  * Added remote control feature: DataLab can be controlled remotely via a TCP/IP connection (see [Remote control](https://datalab-platform.com/en/remote_control.html))
  * Added macro commands: DataLab can be controlled via a macro file (see [Macro commands](https://datalab-platform.com/en/macro_commands.html))
* General features:
  * Added settings dialog box (see "Settings" entry in "File" menu):
    * General settings
    * Visualization settings
    * Processing settings
    * Etc.
  * New default layout: signal/image panels are on the right side of the main window, visualization panels are on the left side with a vertical toolbar
* Signal/Image features:
  * Added process isolation: each signal/image is processed in a separate process, so that DataLab does not freeze anymore when processing large signals/images
  * Added support for groups: signals and images can be grouped together, and operations can be applied to all objects in a group, or between groups
  * Added warning and error dialogs with detailed traceback links to the source code (warnings may be optionally ignored)
  * Drastically improved performance when selecting objects
  * Optimized performance when showing large images
  * Added support for dropping files on signal/image panel
  * Added "Analysis parameters" group box to show last result input parameters
  * Added "Copy titles to clipboard" feature in "Edit" menu
  * For every single processing feature (operation, processing and analysis menus), the entered parameters (dialog boxes) are stored in cache to be used as defaults the next time the feature is used
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
  * Contour detection: added support for polygonal contours (in addition to circle and ellipse contours)
  * Added circle Hough transform (circle detection)
  * Added image intensity levels rescaling
  * Added histogram equalization
  * Added adaptative histogram equalization
  * Added blob detection methods:
    * Difference of Gaussian
    * Determinant of Hessian method
    * Laplacian of Gaussian
    * Blob detection using OpenCV
  * Result shapes and annotations are now transformed (instead of removed) when executing one of the following operations:
    * Rotation (arbitrary angle, +90°, -90°)
    * Symetry (vertical/horizontal)
  * Added support for optional FFT shift (see Settings dialog box)
* Console: added configurable external editor (default: VSCode) to follow the traceback links to the source code
