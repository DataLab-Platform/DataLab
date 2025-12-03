# Version 0.17 #

## DataLab Version 0.17.1 (2024-10-01) ##

ℹ️ PlotPy v2.6.2 is required for this release.

💥 New features and enhancements:

* Image View:
  * Before this release, when selecting a high number of images (e.g. when selecting a group of images), the application was very slow because all the images were displayed in the image view, even if they were all superimposed on the same image
  * The workaround was to enable the "Show first only" option
  * Now, to improve performance, if multiple images are selected, only the last image of the selection is displayed in the image view if this last image has no transparency and if the other images are completely covered by this last image
* Clarification: action "Show first only" was renamed to "Show first object only", and a new icon was added to the action
* API: added `width` and `height` properties to `ImageObj` class (returns the width and height of the image in physical units)
* Windows launcher "start.pyw": writing a log file "datalab_error.log" when an exception occurs at startup

🛠️ Bug fixes:

* Changing the color theme now correctly updates all DataLab's user interface components without the need to restart the application

ℹ️ Other changes:

* OpenCV is now an optional dependency:
  * This change is motivated by the fact that the OpenCV conda package is not maintained on Windows (at least), which leads to an error when installing DataLab with conda
  * When OpenCV is not installed, only the "OpenCV blob detection" feature won't work, and a warning message will be displayed when trying to use this feature

## DataLab Version 0.17.0 (2024-08-02) ##

ℹ️ PlotPy v2.6 is required for this release.

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
