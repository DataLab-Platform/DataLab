# Version 0.18 #

## DataLab Version 0.18.2 (2025-03-16) ##

ℹ️ General information:

* Python 3.13 is now supported, since the availability of the scikit-image V0.25 (see [Issue #104](https://github.com/DataLab-Platform/DataLab/issues/104) - Python 3.13: `KeyError: 'area_bbox'`)

💥 Enhancements:

* Added new "Keep results after computation" option in "Processing" section:
  * Before this change, when applying a processing feature (e.g. a filter, a threshold, etc.) on a signal or an image, the analysis results were removed from the object
  * This new option allows to keep the analysis results after applying a processing feature on a signal or an image. Even if the analysis results are not updated, they might be relevant in some use cases (e.g. when using the 2D peak detection feature on an image, and then applying a filter on the image, or summing two images, etc.)

🛠️ Bug fixes:

* Fixed [Issue #138](https://github.com/DataLab-Platform/DataLab/issues/138) - Image colormaps were no longer stored in metadata (and serialized in HDF5 files) since PlotPy v2.6.3 (this commit, specifically: [PlotPyStack/PlotPy@a37af8a](https://github.com/PlotPyStack/PlotPy/commit/a37af8ae8392e5e3655e5c34b67a7cd1544ea845))
* Fixed [Issue #137](https://github.com/DataLab-Platform/DataLab/issues/137) - Arithmetic operations and signal interpolation: dialog box with parameters is not displayed
* Fixed [Issue #136](https://github.com/DataLab-Platform/DataLab/issues/136) - When processing a signal or an image, the analysis result is kept from original object
  * Before this fix, when processing a signal or an image (e.g. when applying a filter, a threshold, etc.), the analysis result was kept from the original object, and was not updated with the new data. Thus the analysis result was not meaningful anymore, and was misleading the user.
  * This is now fixed: the analysis result is now removed when processing a signal or an image. However it is not recalculated automatically, because there is no way to know which analysis result should be recalculated (e.g. if the user has applied a filter, should the FWHM be recalculated?) - besides, the current implementation of the analysis features does not allow to recalculate the analysis results automatically when the data is modified. The user has to recalculate the analysis results manually if needed.
* Fixed [Issue #132](https://github.com/DataLab-Platform/DataLab/issues/132) - Plot analysis results: "One curve per result title" mode ignores ROIs
  * Before this fix, the "One curve per result title" mode was ignoring ROIs, and was plotting the selected result for all objects (signals or images) without taking into account the ROI defined on the objects
  * This is now fixed: the "One curve per result title" mode now takes into account the ROI defined on the objects, and plots the selected result for each object (signal or image) and for each ROI defined on the object
* Fixed [Issue #128](https://github.com/DataLab-Platform/DataLab/issues/128) - Support long object titles in Signal and Image panels
* Fixed [Issue #133](https://github.com/DataLab-Platform/DataLab/issues/133) - Remove specific analysis results from metadata clipboard during copy operation
* Fixed [Issue #135](https://github.com/DataLab-Platform/DataLab/issues/135) - Allow to edit ROI on multiple signals or images at once
  * Before this fix, the ROI editor was disabled when multiple signals or images were selected
  * This is now fixed: the ROI editor is now enabled when multiple signals or images are selected, and the ROI is applied to all selected signals or images (only the ROI of the first selected signal or image is taken into account)
  * This new behavior is consistent with the ROI extraction feature, which allows to extract the ROI on multiple signals or images at once, based on the ROI defined on the first selected signal or image
* Image ROI features:
  * Fixed [Issue #120](https://github.com/DataLab-Platform/DataLab/issues/120) - ROI extraction on multiple images: defined ROI should not be saved in the first selected object. The design choice is to save the defined ROI neither in the first nor in any of the selected objects: the ROI is only used for the extraction, and is not saved in any object
  * Fixed [Issue #121](https://github.com/DataLab-Platform/DataLab/issues/121) - `AttributeError` when extracting multiple ROIs on a single image, if more than one image is selected
  * Fixed [Issue #122](https://github.com/DataLab-Platform/DataLab/issues/122) - Image masks are not refreshed when removing metadata except for the active image
  * Fixed [Issue #123](https://github.com/DataLab-Platform/DataLab/issues/123) - Image masks are not refreshed when pasting metadata on multiple images, except for the last image
* Text and CSV files:
  * Enhance text file reading by detecting data headers (using a list of typical headers from scientific instruments) and by allowing to skip the header when reading the file
  * Ignore encoding errors when reading files in both open feature and import wizard, hence allowing to read files with special characters without raising an exception
  * Fixed [Issue #124](https://github.com/DataLab-Platform/DataLab/issues/124) - Text files: support locale decimal separator (different than `.`)
* Signal analysis features: fixed duplicate results when no ROI is defined
* Fixed [Issue #113](https://github.com/DataLab-Platform/DataLab/issues/113) - Call to `RemoteClient.open_h5_files` (and `import_h5_file`) fails without passing the optional arguments
* Fixed [Issue #116](https://github.com/DataLab-Platform/DataLab/issues/116) - `KeyError` exception when trying to remove a group after opening an HDF5 file

## DataLab Version 0.18.1 (2024-12-13) ##

💥 Enhancements:

* FWHM computation now raises an exception when less than two points are found with zero-crossing method
* Improved result validation for array-like results by checking the data type of the result

🛠️ Bug fixes:

* Fixed [Issue #106](https://github.com/DataLab-Platform/DataLab/issues/106) - Analysis: coordinate shifted results on images with ROIs and shifted origin
* Fixed [Issue #107](https://github.com/DataLab-Platform/DataLab/issues/107) - Wrong indices when extracting a profile from an image with a ROI
* Fixed [Issue #111](https://github.com/DataLab-Platform/DataLab/issues/111) - Proxy `add_object` method does not support signal/image metadata (e.g. ROI)
* Test data plugin / "Create 2D noisy gauss image": fixed amplitude calculation in `datalab.tests.data.create_2d_random` for non-integer data types

📚 Documentation:

* Fixed path separators in plugin directory documentation
* Corrected left and right area descriptions in workspace documentation
* Updated Google style link in contributing guidelines
* Fixed various French translations in the documentation

## DataLab Version 0.18.0 (2024-11-14) ##

ℹ️ General information:

* PlotPy v2.7 is required for this release.
* Dropped support for Python 3.8.
* Python 3.13 is not supported yet, due to the fact that some dependencies are not compatible with this version.

💥 New features and enhancements:

* New operation mode feature:
  * Added "Operation mode" feature to the "Processing" tab in the "Settings" dialog box
  * This feature allows to choose between "single" and "pairwise" operation modes for all basic operations (addition, subtraction, multiplication, division, etc.):
    * "Single" mode: single operand mode (default mode: the operation is done on each object independently)
    * "Pairwise" mode: pairwise operand mode (the operation is done on each pair of objects)
  * This applies to both signals and images, and to computations taking *N* inputs
  * Computations taking *N* inputs are the ones where:
    * *N(>=2)* objects in give *N* objects out
    * *N(>=1)* object(s) + 1 object in give N objects out

* New ROI (Region Of Interest) features:
  * New polygonal ROI feature
  * Complete redesign of the ROI editor user interfaces, improving ergonomics and consistency with the rest of the application
  * Major internal refactoring of the ROI system to make it more robust (more tests) and easier to maintain

* Implemented [Issue #102](https://github.com/DataLab-Platform/DataLab/issues/102) - Launch DataLab using `datalab` instead of `datalab`. Note that the `datalab` command is still available for backward compatibility.

* Implemented [Issue #101](https://github.com/DataLab-Platform/DataLab/issues/101) - Configuration: set default image interpolation to anti-aliasing (`5` instead of `0` for nearest). This change is motivated by the fact that a performance improvement was made in PlotPy v2.7 on Windows, which allows to use anti-aliasing interpolation by default without a significant performance impact.

* Implemented [Issue #100](https://github.com/DataLab-Platform/DataLab/issues/100) - Use the same installer and executable on Windows 7 SP1, 8, 10, 11. Before this change, a specific installer was required for Windows 7 SP1, due to the fact that Python 3.9 and later versions are not supported on this platform. A workaround was implemented to make DataLab work on Windows 7 SP1 with Python 3.9.

🛠️ Bug fixes:

* Fixed [Issue #103](https://github.com/DataLab-Platform/DataLab/issues/103) - `proxy.add_annotations_from_items`: circle shape color seems to be ignored.
