# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Public API's object model module
----------------------------------------

This modules aims at providing all the necessary classes and functions to
create and manipulate DataLab signal and image objects.

Those classes and functions are defined in other modules:
    - :mod:`cdl.core.model.base`
    - :mod:`cdl.core.model.image`
    - :mod:`cdl.core.model.signal`
    - :mod:`cdl.core.io`

This module is thus a convenient way to import all the objects at once.
"""

# pylint:disable=unused-import

from cdl.core.io import read_image, read_signal
from cdl.core.model.base import (
    NormalRandomParam,
    ResultShape,
    ShapeTypes,
    UniformRandomParam,
)
from cdl.core.model.image import (
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageTypes,
    NewImageParam,
    RoiDataGeometries,
    create_image,
    create_image_from_param,
    new_image_param,
)
from cdl.core.model.signal import (
    GaussLorentzVoigtParam,
    NewSignalParam,
    PeriodicParam,
    SignalObj,
    SignalTypes,
    StepParam,
    create_signal,
    create_signal_from_param,
    new_signal_param,
)
