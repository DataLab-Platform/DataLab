# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
sigima_
======

Scientific computing engine for 1D signals and 2D images,
part of the DataLab open-source platform.
"""

# TODO: Use `numpy.typing.NDArray` for more precise type annotations once NumPy >= 1.21
# can be safely required (e.g. after raising the minimum required version of
# scikit-image to >= 0.19).

# The following comments are used to track the migration process of the `sigima_`
# package, in the context of the DataLab Core Architecture Redesign project funded by
# the NLnet Foundation.

# -------- Point of no return after creating an independent `sigima_` package ----------
# TODO: In `cdl` Python package, remove modifications related to the inclusion of the
#       `sigima_` module within the `cdl` package (e.g., see TODOs in pyproject.toml,
#       VSCode tasks, Pylint configuration, etc.)
# TODO: Move `cdl.tests.sigima_tests` to external `sigima.tests` module
# ** Task 1. Core Architecture Redesign **
# **   Milestone 1.b. Decouple I/O features (including I/O plugins) **
# TODO: Implement a I/O plugin system similar to the `cdl.plugins` module
# **   Milestone 1.c. Redesign the API for the new core library **
# TODO: Remove "sigima_*" from "cdl" pyproject.toml (`include = ["cdl*", "sigima_*"]`)
# TODO: Rename "cdl" package to "datalab" (at last! finally!)
# TODO: Rename "sigima_" (temporary name until the package is fully migrated)
#       to "sigima_" when the migration is complete.
# TODO: Add local translations for the `sigima_` package
#
# ** Task 2. Technical Validation and Testing **
# TODO: Add `pytest` infrastructure. Step 2: migrate `cdl/tests/sigima_tests`
#       to `sigima_/tests` directory and create a `conftest.py` file using the
#       `cdl/tests/conftest.py` file as a template (see TODOs in that file).
# TODO: Handle the `CDL_DATA` environment variable in the `sigima_` package
#       and its documentation (at the moment, it has been replaced by `SIGIMA_DATA`
#       in sigima_\tests\helpers.py)
#
# ** Task 3. Documentation and Training Materials **
# TODO: Add documentation. Step 1: initiate `sigima_` package documentation
# TODO: Add documentation. Step 2: migrate parts of `cdl` package documentation
# --------------------------------------------------------------------------------------

# pylint:disable=unused-import
# flake8: noqa

from sigima_.io import (
    read_images,
    read_signals,
    write_image,
    write_signal,
    read_image,
    read_signal,
)
from sigima_.obj import (
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussLorentzVoigtParam,
    ImageDatatypes,
    ImageROI,
    ImageObj,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalRandomParam,
    PeriodicParam,
    PolygonalROI,
    RectangularROI,
    ResultProperties,
    ResultShape,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    ShapeTypes,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TypeObj,
    TypeROI,
    UniformRandomParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
)

__version__ = "0.0.1"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/sigima_/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
