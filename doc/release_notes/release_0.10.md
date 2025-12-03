# Version 0.10 #

## DataLab Version 0.10.1 (2023-12-22) ##

*Note*: V0.10.0 (released 2023-12-22) was almost immediately replaced by V0.10.1 due to a last minute bug fix

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
          proxy.compute_fft() # (...)
      ```

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
