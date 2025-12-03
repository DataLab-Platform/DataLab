# Version 0.12 #

## DataLab Version 0.12.0 (2024-02-16) ##

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
