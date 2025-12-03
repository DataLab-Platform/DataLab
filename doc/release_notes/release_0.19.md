# Version 0.19 #

## DataLab Version 0.19.2 (2025-04-28) ##

🛠️ Bug fixes:

* Fixed [Issue #172](https://github.com/DataLab-Platform/DataLab/issues/172) - Image profiles: when moving/resizing image, profile plots are not refreshed (fixed in PlotPy v2.7.4)
* Fixed [Issue #173](https://github.com/DataLab-Platform/DataLab/issues/173) - Phase spectrum: add unit (degree) and function reference (`numpy.angle`) to the documentation
* Fixed [Issue #177](https://github.com/DataLab-Platform/DataLab/issues/177) - "Open from directory" feature: unexpected group name (a group named "." is created instead of the root folder name)
* Fixed [Issue #169](https://github.com/DataLab-Platform/DataLab/issues/169) - Signal / Fourier analysis: magnitude spectrum feature does not work as expected with logarithmic scale enabled
* Fixed [Issue #168](https://github.com/DataLab-Platform/DataLab/issues/168) - Average profile visualization: empty profile is displayed when the target rectangular area is outside the image area (this has been fixed upstream, in PlotPy v2.7.4, and so requires the latest version of PlotPy)

## DataLab Version 0.19.1 (2025-04-08) ##

🛠️ Bug fixes:

* Pairwise operation mode:
  * Fixed an unexpected behavior when using the pairwise operation mode with functions that take a single second operand (e.g. for images: difference, division, arithmetic operations, and flatfield correction)
  * If only one set of operands was selected in a single group, a warning message was displayed "In pairwise mode, you need to select objects in at least two groups.", which is correct for functions that are symmetric (e.g. addition, multiplication, etc.), but not for functions that are not symmetric (e.g. difference, division, etc.).
  * This is now fixed: the warning message is only displayed for functions that are symmetric (e.g. addition, multiplication, etc.).
  * This closes [Issue #157](https://github.com/DataLab-Platform/DataLab/issues/157) - Pairwise operation mode: unexpected behavior with functions that take a single second operand
* Fixed [Issue #152](https://github.com/DataLab-Platform/DataLab/issues/152) - Ignore `nan` values for image normalization, flatfield correction, offset correction, and centroid computation
* Fixed [Issue #153](https://github.com/DataLab-Platform/DataLab/issues/153) - Ignore `nan` values for signal normalization and statistics computations (both analysis result and interactive tool)
* Fixed [Issue #158](https://github.com/DataLab-Platform/DataLab/issues/158) - When editing ROI of a list of images, the first image of the selection is shown (instead of the last as in the image panel)
* Fixed [Issue #159](https://github.com/DataLab-Platform/DataLab/issues/159) - When selecting multiple images just after opening an HDF5 file, the "View in a new window" feature does not work (`KeyError` exception)
* Fixed [Issue #160](https://github.com/DataLab-Platform/DataLab/issues/160) - When selecting multiple images and clearing ROI in ROI editor, only the first image is affected
* Fixed [Issue #161](https://github.com/DataLab-Platform/DataLab/issues/161) - Refresh image items only if necessary (when editing ROI, pasting/deleting metadata)
* Fixed [Issue #162](https://github.com/DataLab-Platform/DataLab/issues/162) - View in a new window: when displaying multiple images, the item list panel should be visible
* Fixed [Issue #163](https://github.com/DataLab-Platform/DataLab/issues/163) - Open from directory: expected one group per folder when loading multiple files
* Fixed [Issue #164](https://github.com/DataLab-Platform/DataLab/issues/164) - Open from directory: unsupported files should be ignored when loading files recursively, to avoid warning popup dialog boxes
* Fixed [Issue #165](https://github.com/DataLab-Platform/DataLab/issues/165) - When opening a file, the default signal/image title must be set to the file name, instead of the relative path to the file name

## DataLab Version 0.19.0 (2025-03-31) ##

💥 New features and enhancements:

* Image operation features ("Operations" menu):
  * Renamed "Rotation" submenu to "Flip or rotation"
  * New "Flip diagonally" feature
* Signal processing features ("Processing" menu):
  * New "Convert to Cartesian coordinates" feature
  * New "Convert to polar coordinates" feature
* Signal analysis features ("Analysis" menu):
  * Renamed "X values at min/max" to "Abscissa of the minimum and maximum"
  * New "Abscissa at y=..." feature
* New "Open from directory" feature:
  * This feature allows to open multiple files from a directory at once, recursively (only the files with the supported extensions by the current panel are opened)
  * Add "Open from directory" action to the "File" menu for both Signal and Image panels
  * Add support for folders when dropping files in the Signal and Image panels
* Add `1/x` operation to the "Operations" menu for both Signal and Image panels:
  * This feature relies on the `numpy.reciprocal` function, and handles the case where the denominator is zero by catching warnings and replacing the `np.inf` values with `np.nan` values
  * Add `compute_inverse` method for image and signal processors
  * This closes [Issue #143](https://github.com/DataLab-Platform/DataLab/issues/143) - New feature: `1/x` for signals and images
* Public API (local or remote):
  * Add `add_group` method with `title` and `select` arguments to create a new group in a data panel (e.g. Signal or Image panel) and eventually select it after creation:
    * Method was added to the following classes: `AbstractDLControl`, `BaseDataPanel` and `RemoteClient`
    * This closes the following issues:
      * [Issue #131](https://github.com/DataLab-Platform/DataLab/issues/131) - `BaseDataPanel.add_group`: add `select` argument
      * [Issue #47](https://github.com/DataLab-Platform/DataLab/issues/47) - Remote proxy / Public API: add `add_group` method
  * `AbstractDLControl.get_object_uuids`: add an optional `group` argument (group ID, title or number) to eventually filter the objects by group (this closes [Issue #130](https://github.com/DataLab-Platform/DataLab/issues/130))
* When opening an HDF5 file, the confirmation dialog box asking if current workspace should be cleared has a new possible answer "Ignore":
  * Choosing "Ignore" will prevent the confirmation dialog box from being displayed again, and will choose the current setting (i.e. clear or not the workspace) for all subsequent file openings
  * Added a new "Clear workspace before loading HDF5 file" option in the "Settings" dialog box, to allow the user to change the current setting (i.e. clear or not the workspace) for all subsequent file openings
  * Added a new "Ask before clearing workspace" option in the "Settings" dialog box, to allow the user to disable or re-enable the confirmation dialog box asking if current workspace should be cleared when opening an HDF5 file
  * This closes [Issue #146](https://github.com/DataLab-Platform/DataLab/issues/146) - Ask before clearing workspace when opening HDF5 file: add "Ignore" option to prevent dialog from being displayed again
* Object and group title renaming:
  * Removed "Rename group" feature from the "Edit" menu and context menu
  * Added "Rename object" feature to the "Edit" menu and context menu, with F2 shortcut, to rename the title of the selected object or group
  * This closes [Issue #148](https://github.com/DataLab-Platform/DataLab/issues/148) - Rename signal/image/group title by pressing F2
* Region of Interest editor:
  * Regrouped the graphical actions (new rectangular ROI, new circular ROI, new polygonal ROI) in a single menu "Graphical ROI"
  * Added new "Coordinate-based ROI" menu to create a ROI using manual input of the coordinates:
    * For signals, the ROI is defined by the start and end coordinates
    * For images:
      * The rectangular ROI is defined by the top-left and bottom-right coordinates
      * The circular ROI is defined by the center and radius coordinates
      * The polygonal ROI is not supported yet
    * This closes [Issue #145](https://github.com/DataLab-Platform/DataLab/issues/145) - ROI editor: add manual input of the coordinates

🛠️ Bug fixes:

* Fixed [Issue #141](https://github.com/DataLab-Platform/DataLab/issues/141) - Image analysis: mask `nan` values when computing statistics, for example
* Fixed [Issue #144](https://github.com/DataLab-Platform/DataLab/issues/144) - Average profile extraction: `ValueError` when selection rectangle is larger than the image
