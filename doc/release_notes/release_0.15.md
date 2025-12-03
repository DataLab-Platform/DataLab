# Version 0.15 #

## DataLab Version 0.15.1 (2024-05-03) ##

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

## DataLab Version 0.15.0 (2024-04-11) ##

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
