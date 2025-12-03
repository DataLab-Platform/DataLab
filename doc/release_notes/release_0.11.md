# Version 0.11 #

## DataLab Version 0.11.0 (2024-01-23) ##

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
  * This concerns both 1D (FHWM...) and 2D analysis results (contours, blobs...):
    * Segment results now also show length (L) and center coordinates (Xc, Yc)
    * Circle and ellipse results now also show area (A)
* Added "Plot results" entry in "Analysis" menu:
  * This feature allows to plot analysis results (1D or 2D)
  * It creates a new signal with X and Y axes corresponding to user-defined parameters (e.g. X = indices and Y = radius for circle results)
* Increased default width of the object selection dialog box:
  * The object selection dialog box is now wider by default, so that the full signal/image/group titles may be more easily readable
* Delete metadata feature:
  * Before this release, the feature was deleting all metadata, including the Regions Of Interest (ROI) metadata, if any.
  * Now a confirmation dialog box is shown to the user before deleting all metadata if the signal/image has ROI metadata: this allows to keep the ROI metadata if needed.
* Image profile extraction feature: added support for masked images (when defining regions of interest, the areas outside the ROIs are masked, and the profile is extracted only on the unmasked areas, or averaged on the unmasked areas in the case of average profile extraction)
* Curve style: added "Reset curve styles" in "View" menu. This feature allows to reset the curve style cycle to its initial state.
* Plugin base classe `PluginBase`:
  * Added `edit_new_signal_parameters` method for showing a dialog box to edit parameters for a new signal
  * Added `edit_new_image_parameters` method for showing a dialog box to edit parameters for a new image (updated the *datalab_testdata.py* plugin accordingly)
* Signal and image computations API (`datalab.computations`):
  * Added wrappers for signal and image 1 -> 1 computations
  * These wrappers aim at simplifying the creation of a basic computation function operating on DataLab's native objects (`SignalObj` and `ImageObj`) from a function operating on NumPy arrays
  * This simplifies DataLab's internals and makes it easier to create new computing features inside plugins
  * See the *datalab_custom_func.py* example plugin for a practical use case
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
  * Before this release, when extracting a single circular ROI from an image with the "Extract all ROIs into a single image object" option enabled, the result was a single image without the ROI mask (the ROI mask was only available when extracting ROI with the option disabled)
  * This was leading to an unexpected behavior, because one could interpret the result (a square image without the ROI mask) as the result of a single rectangular ROI
  * Now, when extracting a single circular ROI from an image with the "Extract all ROIs into a single image object" option enabled, the result is a single image with the ROI mask (as if the option was disabled)
  * This fixes [Issue #31](https://github.com/DataLab-Platform/DataLab/issues/31) - Single circular ROI extraction: automatically switch to `compute_extract_roi` function
* Analysis on circular ROI:
  * Before this release, when running computations on a circular ROI, the results were unexpected in terms of coordinates (results seemed to be computed in a region located above the actual ROI).
  * This was due to a regression introduced in an earlier release.
  * Now, when defining a circular ROI and running computations on it, the results are computed on the actual ROI
  * This fixes [Issue #33](https://github.com/DataLab-Platform/DataLab/issues/33) - Analysis on circular ROI: unexpected results
* Contour detection on ROI:
  * Before this release, when running contour detection on a ROI, some contours were detected outside the ROI (it may be due to a limitation of the scikit-image `find_contours` function).
  * Now, thanks a workaround, the erroneous contours are filtered out.
  * A new test module `datalab.tests.features.images.contour_fabryperot_app` has been added to test the contour detection feature on a Fabry-Perot image (thanks to [@emarin2642](https://github.com/emarin2642) for the contribution)
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
