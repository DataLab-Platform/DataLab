# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab
=======

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `PlotPyStack`_ libraries).

.. _PlotPyStack: https://github.com/PlotPyStack
"""

import os

__version__ = "0.21.0"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/DataLab/issues/new/choose"

os.environ["DATALAB_VERSION"] = __version__

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
