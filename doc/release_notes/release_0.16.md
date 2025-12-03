# Version 0.16 #

## DataLab Version 0.16.4 (2024-07-09) ##

This is a minor maintenance release.

🛠️ Bug fixes:

* Requires PlotPy v2.4.1 or later to fix the following issues related to the contrast adjustment feature:
  * A regression was introduced in an earlier version of PlotPy: levels histogram was no longer removed from contrast adjustment panel when the associated image was removed from the plot
  * This is now fixed: when an image is removed, the histogram is removed as well and the contrast panel is refreshed (which was not the case even before the regression)
* Ignore `AssertionError` in *config_unit_test.py* when executing test suite on WSL

📚 Documentation:

* Fix class reference in `Wrap11Func` documentation

## DataLab Version 0.16.3 (2024-07-03) ##

🛠️ Bug fixes:

* Fixed [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84) - Build issues with V0.16.1: `signal` name conflict, ...
  * This issue was intended to be fixed in version 0.16.2, but the fix was not complete
  * Thanks to [@rolandmas](https://github.com/rolandmas) for reporting the issue and for the help in investigating the problem and testing the fix
* Fixed [Issue #85](https://github.com/DataLab-Platform/DataLab/issues/85) - Test data paths may be added multiple times to `datalab.utils.tests.TST_PATH`
  * This issue is related to [Issue #84](https://github.com/DataLab-Platform/DataLab/issues/84)
  * Adding the test data paths multiple times to `datalab.utils.tests.TST_PATH` was causing the test data to be loaded multiple times, which lead to some tests failing (a simple workaround was added to V0.16.2: this issue is now fixed)
  * Thanks again to [@rolandmas](https://github.com/rolandmas) for reporting the issue in the context of the Debian packaging
* Fixed [Issue #86](https://github.com/DataLab-Platform/DataLab/issues/86) - Average of N integer images overflows data type
* Fixed [Issue #87](https://github.com/DataLab-Platform/DataLab/issues/87) - Image average profile extraction: `AttributeError` when trying to edit profile parameters
* Fixed [Issue #88](https://github.com/DataLab-Platform/DataLab/issues/88) - Image segment profile: point coordinates inversion

## DataLab Version 0.16.2 (2024-07-01) ##

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

* Difference, division...: dialog box for the second operand selection was allowing to select a group (only a signal or an image should be selected)

* When doing an operation which involves an object (signal or image) with higher order number than the current object (e.g. when subtracting an image with an image from a group below the current image), the resulting object's title now correctly refers to the order numbers of the objects involved in the operation (e.g., to continue with the subtraction example mentioned above, the resulting object's title was previously referring to the order number before the insertion of the resulting image)

* Added support for additional test data folder thanks to the `DATALAB_DATA` environment variable (useful for testing purposes, and especially in the context of Debian packaging)

## DataLab Version 0.16.1 (2024-06-21) ##

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

## DataLab Version 0.16.0 (2024-06-13) ##

💥 New features and enhancements:

* Major user interface overhaul:
  * The menu bar and toolbars have been reorganized to make the application more intuitive and easier to use
  * Operations and processing features have been regrouped in submenus
  * All visualization-related actions are now grouped in the plot view vertical toolbar
  * Clarified the "Annotations" management (new buttons, toolbar action...)

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
| New | New signal | Exponential, pulse, polynomial, custom (manual input)            |
| Operations  | | Exponential, Square root, Power |
| Operations  | Operations with a constant | +, -, *, / |
| Processing  | Axis Transformation | Reverse X-axis |
| Processing  | Level Adjustment | Offset correction |
| Processing  | Fourier analysis | Power spectrum, Phase spectrum, Magnitude spectrum, Power spectral density |
| Processing  | Frequency filters | Low-pass, High-pass, Band-pass, Band-stop |
| Processing  | | Windowing (Hanning, Hamming, Blackman, Blackman-Harris, Nuttall, Flat-top...) |
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
