# Version 0.20 #

## DataLab Version 0.20.1 (2025-07-15) ##

🛠️ Bug fixes:

* Fixed [Issue #233](https://github.com/DataLab-Platform/DataLab/issues/233) - Hard crash when trying to activate the curve stats tool on a zero signal
* Fixed [Issue #184](https://github.com/DataLab-Platform/DataLab/issues/184) - Curve marker style unexpectedly changes to "Square" after validating "Parameters…" dialog
* Fixed [Issue #117](https://github.com/DataLab-Platform/DataLab/issues/117) - DataLab's signal moving median crashes on Linux with `mode='mirror'`: `free(): invalid next size (normal)` (this is a bug in SciPy v1.15.0 to v1.15.2, which was fixed in SciPy v1.15.3)
* Fixed [Issue #186](https://github.com/DataLab-Platform/DataLab/issues/186) - Image text files with a comma as decimal separator cannot be opened directly (only the Import Wizard does support this)
* Fixed [Issue #238](https://github.com/DataLab-Platform/DataLab/issues/238) - Image text files with a trailing delimiter leads to data with a superfluous column when opened directly (does not happen in the Import Wizard)
* Fixed [Issue #239](https://github.com/DataLab-Platform/DataLab/issues/239) - Text Import Wizard does not preserve user-defined titles and units
* Fixed [Issue #240](https://github.com/DataLab-Platform/DataLab/issues/240) - Text Import Wizard does not preserve user-defined data type (e.g. `int16`, `float32`, etc.)
* Fixed [Issue #235](https://github.com/DataLab-Platform/DataLab/issues/235) - Text Import Wizard: add support for importing signal files with integer values
* Fixed [Issue #236](https://github.com/DataLab-Platform/DataLab/issues/236) - Text Import Wizard: add support for `.mca` files
* Fixed [Issue #243](https://github.com/DataLab-Platform/DataLab/issues/243) - View in a new window (image): intensity profile tools are sometimes disabled (fixed in PlotPy v2.7.5)

## DataLab Version 0.20.0 (2025-04-28) ##

💥 New features and enhancements:

* ANDOR SIF Images:
  * Added support for background images in ANDOR SIF files
  * This closes [Issue #178](https://github.com/DataLab-Platform/DataLab/issues/178) - Add support for ANDOR SIF files with background image
* Array editor (results, signal and image data, ...):
  * New "Copy all" button in the array editor dialog box, to copy all the data in the clipboard, including row and column headers
  * New "Export" button in the array editor dialog box, to export the data in a CSV file, including row and column headers
  * New "Paste" button in the array editor dialog box, to paste the data from the clipboard into the array editor (this feature is not available for read-only data, such as analysis results)
  * The features above require guidata v3.9.0 or later
  * This closes [Issue #174](https://github.com/DataLab-Platform/DataLab/issues/174), [Issue #175](https://github.com/DataLab-Platform/DataLab/issues/175) and [Issue #176](https://github.com/DataLab-Platform/DataLab/issues/176)
* Fourier analysis features ("Processing" menu):
  * New "Zero padding" feature
  * Implementation for signals:
    * Choose a zero padding strategy (Next power of 2, Double the length, Triple the length, Custom length)
    * Or manually set the zero padding length (if "Custom length" is selected)
  * Implementation for images:
    * Choose a zero padding strategy (Next power of 2, Next multiple of 64, Custom length)
    * Or manually set the zero padding row and column lengths (if "Custom length" is selected)
    * Set the position of the zero padding (bottom-right, centered)
  * This closes [Issue #170](https://github.com/DataLab-Platform/DataLab/issues/170) - Fourier analysis: add zero padding feature for signals and images
* Region of Interest (ROI) editor:
  * This concerns the "Edit Regions of Interest" feature for both signals and images
  * New behavior:
    * Signals: the range ROI selection tool is now active by default, and the user can select right away the range of the signal to be used as a ROI
    * Images: the rectangular ROI selection tool is now active by default, and the user can select right away the rectangular ROI to be used as a ROI
    * This closes [Issue #154](https://github.com/DataLab-Platform/DataLab/issues/154) - ROI editor: activate ROI selection tool by default, so that the user can select right away the area to be used as a ROI
  * Added the "Select tool" to editor's toolbar, to allow the user to switch between the "Select" and "Draw" tools easily without having to use the plot toolbar on the top of the window
* Signal processing features ("Processing" menu):
  * New "X-Y mode" feature: this feature simulates the behavior of the X-Y mode of an oscilloscope, i.e. it allows to plot one signal as a function of another signal (e.g. X as a function of Y)
  * New abscissa and ordinate find features:
    * "First abscissa at y=..." feature: this feature allows to find the first abscissa value of a signal at a given y value (e.g. the abscissa value of a signal at y=0)
    * "Ordinate at x=..." feature: this feature allows to find the ordinate value of a signal at a given x value (e.g. the ordinate value of a signal at x=0)
    * Each feature has its own dialog box, which allows to set the y or x value to be used for the search with a slider or a text box
    * This closes [Issue #125](https://github.com/DataLab-Platform/DataLab/issues/125) and [Issue #126](https://github.com/DataLab-Platform/DataLab/issues/126)
  * New full width at given y feature:
    * The "Full width at y=..." feature allows to find the full width of a signal at a given y value (e.g. the full width of a signal at y=0)
    * A specific dialog box allows to set the y value to be used for the search with a slider or a text box
    * This closes [Issue #127](https://github.com/DataLab-Platform/DataLab/issues/127)
* Public API (local or remote):
  * Add `group_id` and `set_current` arguments to `add_signal`, `add_image` and `add_object` methods:
    * This concerns the `LocalProxy`, `AbstractDLControl`, `RemoteClient`, `RemoteServer` and `DLMainWindow` classes
    * `group_id` argument allows to specify the group ID where the signal or image should be added (if not specified, the signal or image is added to the current group)
    * `set_current` argument allows to specify if the signal or image should be set as current after being added (default is `True`)
    * This closes [Issue #151](https://github.com/DataLab-Platform/DataLab/issues/151) - Public API: add a keyword `group_id` to `add_signal` and `add_image`
