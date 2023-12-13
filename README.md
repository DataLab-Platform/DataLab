![DataLab](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/DataLab-banner.png)

[![pypi version](https://img.shields.io/pypi/v/cdl.svg)](https://pypi.org/project/CDL/)
[![PyPI status](https://img.shields.io/pypi/status/cdl.svg)](https://github.com/Codra-Ingenierie-Informatique/DataLab)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/cdl.svg)](https://pypi.python.org/pypi/CDL/)

ℹ️ Created by [Codra](https://codra.net/)/[Pierre Raybaut](https://github.com/PierreRaybaut) in 2023, developed and maintained by DataLab open-source project team with the support of [Codra](https://codra.net/).

![DataLab](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/DataLab-Screenshot.png)

ℹ️ DataLab is powered by [PlotPyStack](https://github.com/PlotPyStack) 🚀.

![PlotPyStack](https://raw.githubusercontent.com/PlotPyStack/.github/main/data/plotpy-stack-powered.png)

ℹ️ DataLab is built on Python and scientific libraries.

![Python](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/Python.png) ![NumPy](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/NumPy.png) ![SciPy](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/SciPy.png) ![scikit-image](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/scikit-image.png) ![OpenCV](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/OpenCV.png) ![PlotPyStack](https://raw.githubusercontent.com/CODRA-Ingenierie-Informatique/DataLab/gh-pages/logos/plotpystack.png)

----

## Overview

DataLab is a generic signal and image processing software based on Python scientific
libraries (such as NumPy, SciPy or scikit-image) and Qt graphical user interfaces
(thanks to the powerful [PlotPyStack](https://github.com/PlotPyStack) - mostly the
[guidata](https://github.com/PlotPyStack/guidata) and
[PlotPy](https://github.com/PlotPyStack/PlotPy) libraries).

DataLab is available as a **stand-alone** application (see for example our all-in-one
Windows installer) or as an **addon to your Python-Qt application** thanks to advanced
automation and embedding features.

✨ Add features to DataLab by writing your own [plugin](https://cdlapp.readthedocs.io/en/latest/features/general/plugins.html)
(see [plugin examples](https://github.com/Codra-Ingenierie-Informatique/DataLab/tree/main/plugins/examples))

✨ DataLab may be remotely controlled from a third-party application (such as Jupyter,
Spyder or any IDE):

* Using the integrated [remote control](https://cdlapp.readthedocs.io/en/latest/features/general/remote.html)
feature (this requires to install on your environment DataLab as a Python package and all its dependencies)

* Using the lightweight [DataLab Simple Client](https://github.com/Codra-Ingenierie-Informatique/DataLabSimpleClient) (`pip install cdlclient`)

See [home page](https://codra-ingenierie-informatique.github.io/DataLab/) and
[documentation](https://cdlapp.readthedocs.io/en/latest/) for more details on
the library and [changelog](https://github.com/Codra-Ingenierie-Informatique/DataLab/blob/main/CHANGELOG.md)
for recent history of changes.

### New in DataLab 0.9

New key features in DataLab 0.9:

* New third-party plugin system to add your own features to DataLab
* New process isolation feature to run computations safely in a separate process
* New remote control features to interact with DataLab from Spyder, Jupyter or any IDE
* New remote control features to run computations with DataLab from a third-party application
* New data processing and visualization features (see details in [changelog](CHANGELOG.md))
* Fully automated high-level processing features for internal testing purpose, as well as embedding DataLab in a third-party software
* Extensive test suite (unit tests and application tests) with >80% feature coverage

### Credits

Copyrights and licensing:

* Copyright © 2023 [Codra](https://codra.net/), [Pierre Raybaut](https://github.com/PierreRaybaut).
* Licensed under the terms of the BSD 3-Clause (see [LICENSE](https://github.com/Codra-Ingenierie-Informatique/DataLab/blob/main/LICENSE)).

----

## Key features

### Data visualization

| Signal |  Image | Feature                        |
|:------:|:------:|--------------------------------|
|    •   |    •   | Screenshots (save, copy)       |
|    •   | Z-axis | Lin/log scales                 |
|    •   |    •   | Data table editing             |
|    •   |    •   | Statistics on user-defined ROI |
|    •   |    •   | Markers                        |
|        |    •   | Aspect ratio (1:1, custom)     |
|        |    •   | 50+ available colormaps        |
|        |    •   | X/Y raw/averaged profiles      |
|    •   |    •   | User-defined annotations       |

![1D-Peak detection](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/peak_detection.png)

![2D-Peak detection](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/2dpeak_detection.png)

### Data processing

| Signal | Image | Feature                                            |
|:------:|:-----:|----------------------------------------------------|
|    •   |   •   | Process isolation (for runnning computations)      |
|    •   |   •   | Remote control from Jupyter, Spyder or any IDE     |
|    •   |   •   | Remote control from a third-party application      |
|    •   |   •   | Multiple ROI support                               |
|    •   |   •   | Sum, average, difference, product, ...             |
|    •   |   •   | ROI extraction, Swap X/Y axes                      |
|    •   |       | Semi-automatic multi-peak detection                |
|        |   •   | Rotation (flip, rotate), resize, ...               |
|        |   •   | Flat-field correction                              |
|    •   |       | Normalize, derivative, integral                    |
|    •   |   •   | Linear calibration                                 |
|        |   •   | Thresholding, clipping                             |
|    •   |   •   | Gaussian filter, Wiener filter                     |
|    •   |   •   | Moving average, moving median                      |
|    •   |   •   | FFT, inverse FFT                                   |
|    •   |       | Interactive fit: Gauss, Lorenzt, Voigt, polynomial |
|    •   |       | Interactive multigaussian fit                      |
|    •   |   •   | Computing on custom ROI                            |
|    •   |       | FWHM, FW @ 1/e²                                    |
|        |   •   | Centroid (robust method w/r noise)                 |
|        |   •   | Minimum enclosing circle center                    |
|        |   •   | Automatic 2D-peak detection                        |
|        |   •   | Automatic contour extraction (circle/ellipse fit)  |

![Contour detection](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/contour_detection.png)

![Multi-gaussian fit](https://raw.githubusercontent.com/Codra-Ingenierie-Informatique/DataLab/main/doc/images/multi_gaussian_fit.png)

----

## Installation

### From the installer

DataLab is available as a stand-alone application, which does not require any Python
distribution to be installed. Just run the installer and you're good to go!

The installer package is available in the [Releases](https://github.com/Codra-Ingenierie-Informatique/DataLab/releases) section.

### Dependencies and other installation methods

See [Installation](https://cdlapp.readthedocs.io/en/latest/intro/installation.html)
section in the documentation for more details.
