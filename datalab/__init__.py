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


# Compatibility implementations for removed Sigima methods:
#
# 1. Geometry transformations (replaces "transform_shapes"):
#    - Implemented in datalab.utils.geometry_transforms.apply_geometry_transform
#    - Used by image geometric operations (rotate, rotate90, rotate270, fliph, flipv)
#    - Automatically applied via wrapper functions in ImageProcessor
#
# 2. Result deletion (replaces "delete_results"):
#    - Implemented via adapter methods in datalab.adapters_metadata
#    - TableAdapter.remove_all_from() and GeometryAdapter.remove_all_from()
#    - Used by data panels for result management
#
# 3. Geometry removal (replaces "remove_all_shapes"):
#    - Implemented in datalab.utils.geometry_transforms.remove_all_geometry_results
#    - Used by image distribute_on_grid and reset_positions operations
#
# 4. Geometry/Table result compatibility (replaces monkey-patched methods):
#    - Implemented via adapter classes in datalab.adapters_metadata
#    - Provides clean API for result creation and metadata management
