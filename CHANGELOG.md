# Changelog #

See DataLab [roadmap page](https://cdlapp.readthedocs.io/en/latest/dev/roadmap.html)
for future and past milestones.

## DataLab Version 0.9.3 ##

💥 New features:

* Features common to signals and images:
  * Added "Real part" and "Imaginary part" features to "Operation" menu

## DataLab Version 0.9.2 ##

🛠️ Bug fixes:

* Region of interest (ROI) extraction feature for images:
  * ROI extraction was not working properly when the "Extract all regions of interest
    into a single image object" option was enabled if there was only one defined ROI.
    The result was an image positioned at the origin (0, 0) instead of the expected
    position (x0, y0) and the ROI rectangle itself was not removed as expected.
    This is now fixed (see [Issue #6](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/6) - 'Extract multiple ROI' feature: unexpected result for a single ROI)
  * ROI rectangles with negative coordinates were not properly handled:
    ROI extraction was raising a `ValueError` exception, and the image mask was not
    displayed properly.
    This is now fixed (see [Issue #7](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/7) - Image ROI extraction: `ValueError: zero-size array to reduction operation minimum which has no identity`)
  * ROI extraction was not taking into account the pixel size (dx, dy) and the origin
    (x0, y0) of the image.
    This is now fixed (see [Issue #8](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/8) - Image ROI extraction: take into account pixel size)
* Macro-command console is now read-only:
  * The macro-command panel Python console is currently not supporting standard input
    stream (`stdin`) and this is intended (at least for now)
  * Set Python console read-only to avoid confusion

## DataLab Version 0.9.1 ##

🛠️ Bug fixes:

* French translation is not available on Windows/Stand alone version:
  * Locale was not properly detected on Windows for stand-alone version (frozen
    with `pyinstaller`) due to an issue with `locale.getlocale()` (function
    returning `None` instead of the expected locale on frozen applications)
  * This is ultimately a `pyinstaller` issue, but a workaround has been
    implemented in `guidata` V3.2.2 (see [guidata issue #68](https://github.com/PlotPyStack/guidata/issues/68) - Windows: gettext translation is not working on frozen applications)
  * [Issue #2](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/2) - French translation is not available on Windows Stand alone version
* Saving image to JPEG2000 fails for non integer data:
  * JPEG2000 encoder does not support non integer data or signed integer data
  * Before, DataLab was showing an error message when trying to save incompatible
    data to JPEG2000: this was not a consistent behavior with other standard image
    formats (e.g. PNG, JPG, etc.) for which DataLab was automatically converting
    data to the appropriate format (8-bit unsigned integer)
  * Current behavior is now consistent with other standard image formats: when
    saving to JPEG2000, DataLab automatically converts data to 8-bit unsigned
    integer or 16-bit unsigned integer (depending on the original data type)
  * [Issue #3](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/3) - Save image to JPEG2000: 'OSError: encoder error -2 when writing image file'
* Windows stand-alone version shortcuts not showing in current user start menu:
  * When installing DataLab on Windows from a non-administrator account, the
    shortcuts were not showing in the current user start menu but in the
    administrator start menu instead (due to the elevated privileges of the
    installer and the fact that the installer does not support installing
    shortcuts for all users)
  * Now, the installer *does not* ask for elevated privileges anymore, and
    shortcuts are installed in the current user start menu (this also means that
    the current user must have write access to the installation directory)
  * In future releases, the installer will support installing shortcuts for all
    users if there is a demand for it (see [Issue #5](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/5))
  * [Issue #4](https://github.com/Codra-Ingenierie-Informatique/DataLab/issues/4) - Windows: stand-alone version shortcuts not showing in current user start menu
* Installation and configuration window for stand-alone version:
  * Do not show ambiguous error message 'Invalid dependencies' anymore
  * Dependencies are supposed to be checked when building the stand-alone version
* Added PDF documentation to stand-alone version:
  * The PDF documentation was missing in previous release
  * Now, the PDF documentation (in English and French) is included in the
    stand-alone version

## DataLab Version 0.9.0 ##

New dependencies:

* DataLab is now powered by [PlotPyStack](https://github.com/PlotPyStack):
  * [PythonQwt](https://github.com/PlotPyStack/PythonQwt)
  * [guidata](https://github.com/PlotPyStack/guidata)
  * [PlotPy](https://github.com/PlotPyStack/PlotPy)
* [opencv-python](https://pypi.org/project/opencv-python/) (algorithms for image processing)

New reference platform:

* DataLab is validated on Windows 11 with Python 3.11 and PyQt 5.15
* DataLab is also compatible with other OS (Linux, MacOS) and other Python-Qt
  bindings and versions (Python 3.8-3.12, PyQt6, PySide6)

New features:

* DataLab is a platform:
  * Added support for plugins
    * Custom processing features available in the "Plugins" menu
    * Custom I/O features: new file formats can be added to the standard I/O
      features for signals and images
    * Custom HDF5 features: new HDF5 file formats can be added to the standard
      HDF5 import feature
    * More features to come...
  * Added remote control feature: DataLab can be controlled remotely via a
    TCP/IP connection (see [Remote control](https://cdlapp.readthedocs.io/en/latest/remote_control.html))
  * Added macro commands: DataLab can be controlled via a macro file (see
    [Macro commands](https://cdlapp.readthedocs.io/en/latest/macro_commands.html))
* General features:
  * Added settings dialog box (see "Settings" entry in "File" menu):
    * General settings
    * Visualization settings
    * Processing settings
    * Etc.
  * New default layout: signal/image panels are on the right side of the main
    window, visualization panels are on the left side with a vertical toolbar
* Signal/Image features:
  * Added process isolation: each signal/image is processed in a separate
    process, so that DataLab does not freeze anymore when processing large
    signals/images
  * Added support for groups: signals and images can be grouped together, and
    operations can be applied to all objects in a group, or between groups
  * Added warning and error dialogs with detailed traceback links to the source
    code (warnings may be optionally ignored)
  * Drastically improved performance when selecting objects
  * Optimized performance when showing large images
  * Added support for dropping files on signal/image panel
  * Added "Computing parameters" group box to show last result input parameters
  * Added "Copy titles to clipboard" feature in "Edit" menu
  * For every single processing feature (operation, processing and computing menus),
    the entered parameters (dialog boxes) are stored in cache to be used as defaults
    the next time the feature is used
* Signal processing:
  * Added support for optional FFT shift (see Settings dialog box)
* Image processing:
  * Added pixel binning operation (X/Y binning factors, operation: sum, mean, ...)
  * Added "Distribute on a grid" and "Reset image positions" in operation menu
  * Added Butterworth filter
  * Added exposure processing features:
    * Gamma correction
    * Logarithmic correction
    * Sigmoïd correction
  * Added restoration processing features:
    * Total variation denoising filter (TV Chambolle)
    * Bilateral filter (denoising)
    * Wavelet denoising filter
    * White Top-Hat denoising filter
  * Added morphological transforms (disk footprint):
    * White Top-Hat
    * Black Top-Hat
    * Erosion
    * Dilation
    * Opening
    * Closing
  * Added edge detection features:
    * Roberts filter
    * Prewitt filter (vertical, horizontal, both)
    * Sobel filter (vertical, horizontal, both)
    * Scharr filter (vertical, horizontal, both)
    * Farid filter (vertical, horizontal, both)
    * Laplace filter
    * Canny filter
  * Contour detection: added support for polygonal contours (in addition to
    circle and ellipse contours)
  * Added circle Hough transform (circle detection)
  * Added image intensity levels rescaling
  * Added histogram equalization
  * Added adaptative histogram equalization
  * Added blob detection methods:
    * Difference of Gaussian
    * Determinant of Hessian method
    * Laplacian of Gaussian
    * Blob detection using OpenCV
  * Result shapes and annotations are now transformed (instead of removed) when
    executing one of the following operations:
    * Rotation (arbitrary angle, +90°, -90°)
    * Symetry (vertical/horizontal)
  * Added support for optional FFT shift (see Settings dialog box)
* Console: added configurable external editor (default: VSCode) to follow the
  traceback links to the source code
