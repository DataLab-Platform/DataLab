# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Object model (:mod:`cdl.obj`)
-----------------------------

The :mod:`cdl.obj` module aims at providing all the necessary classes and functions
to create and manipulate DataLab signal and image objects.

Those classes and functions are defined in other modules:
    - :mod:`cdl.core.model.base`
    - :mod:`cdl.core.model.image`
    - :mod:`cdl.core.model.signal`
    - :mod:`cdl.core.io`

The :mod:`cdl.obj` module is thus a convenient way to import all the objects at once.
As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from cdl.core.model.signal import SignalObj
    from cdl.core.model.image import ImageObj

    # Equivalent import statement
    from cdl.obj import SignalObj, ImageObj

Common objects
^^^^^^^^^^^^^^

.. autoclass:: cdl.obj.ResultProperties
    :members:
.. autoclass:: cdl.obj.ResultShape
    :members:
.. autoclass:: cdl.obj.ShapeTypes
    :members:
.. autoclass:: cdl.obj.UniformRandomParam
.. autoclass:: cdl.obj.NormalRandomParam
.. autoclass:: cdl.obj.BaseProcParam

Signal model
^^^^^^^^^^^^

.. autodataset:: cdl.obj.SignalObj
    :members:
    :inherited-members:
.. autofunction:: cdl.obj.read_signal
.. autofunction:: cdl.obj.read_signals
.. autofunction:: cdl.obj.create_signal
.. autofunction:: cdl.obj.create_signal_from_param
.. autofunction:: cdl.obj.new_signal_param
.. autoclass:: cdl.obj.SignalTypes
.. autodataset:: cdl.obj.NewSignalParam
.. autodataset:: cdl.obj.GaussLorentzVoigtParam
.. autodataset:: cdl.obj.StepParam
.. autodataset:: cdl.obj.PeriodicParam
.. autodataset:: cdl.obj.ROI1DParam

Image model
^^^^^^^^^^^

.. autodataset:: cdl.obj.ImageObj
    :members:
    :inherited-members:
.. autofunction:: cdl.obj.read_image
.. autofunction:: cdl.obj.read_images
.. autofunction:: cdl.obj.create_image
.. autofunction:: cdl.obj.create_image_from_param
.. autofunction:: cdl.obj.new_image_param
.. autoclass:: cdl.obj.ImageTypes
.. autodataset:: cdl.obj.NewImageParam
.. autodataset:: cdl.obj.Gauss2DParam
.. autoclass:: cdl.obj.RoiDataGeometries
.. autodataset:: cdl.obj.ROI2DParam
.. autoclass:: cdl.obj.ImageDatatypes
.. autoclass:: cdl.obj.ImageRoiDataItem
"""

# pylint:disable=unused-import
# flake8: noqa

from cdl.core.io import read_image, read_images, read_signal, read_signals
from cdl.core.model.base import (
    BaseProcParam,
    NormalRandomParam,
    ResultShape,
    ResultProperties,
    ShapeTypes,
    UniformRandomParam,
)
from cdl.core.model.image import (
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageRoiDataItem,
    ImageTypes,
    NewImageParam,
    RoiDataGeometries,
    ROI2DParam,
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
    ROI1DParam,
    StepParam,
    create_signal,
    create_signal_from_param,
    new_signal_param,
)
