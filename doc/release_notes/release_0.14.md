# Version 0.14 #

## DataLab Version 0.14.2 (2024-03-22) ##

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

## DataLab Version 0.14.1 (2024-03-18) ##

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

## DataLab Version 0.14.0 (2024-03-05) ##

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
