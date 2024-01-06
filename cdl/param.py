# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Parameters (:mod:`cdl.param`)
-----------------------------

The :mod:`cdl.param` module aims at providing all the dataset parameters that are used
by the :mod:`cdl.core.computation` and :mod:`cdl.core.gui.processor` packages.

Those datasets (:py:class:`guidata.dataset.datatypes.Dataset` subclasses) are defined
in other modules:

    - :mod:`cdl.core.computation.base`
    - :mod:`cdl.core.computation.image`
    - :mod:`cdl.core.computation.signal`

The :mod:`cdl.param` module is thus a convenient way to import all the sets of
parameters at once.

As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from cdl.core.computation.base import MovingAverageParam
    from cdl.core.computation.signal import PolynomialFitParam
    from cdl.core.computation.image.exposure import EqualizeHistParam

    # Equivalent import statement
    from cdl.param import MovingAverageParam, PolynomialFitParam, EqualizeHistParam

Common parameters
^^^^^^^^^^^^^^^^^

.. autoclass:: cdl.param.ClipParam
    :no-index:
.. autoclass:: cdl.param.FFTParam
    :no-index:
.. autoclass:: cdl.param.GaussianParam
    :no-index:
.. autoclass:: cdl.param.MovingAverageParam
    :no-index:
.. autoclass:: cdl.param.MovingMedianParam
    :no-index:
.. autoclass:: cdl.param.ROIDataParam
    :no-index:
.. autoclass:: cdl.param.ThresholdParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autoclass:: cdl.param.DataTypeSParam
    :no-index:
.. autoclass:: cdl.param.FWHMParam
    :no-index:
.. autoclass:: cdl.param.NormalizeYParam
    :no-index:
.. autoclass:: cdl.param.PeakDetectionParam
    :no-index:
.. autoclass:: cdl.param.PolynomialFitParam
    :no-index:
.. autoclass:: cdl.param.XYCalibrateParam
    :no-index:
.. autoclass:: cdl.param.InterpolationParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.AverageProfileParam
    :no-index:
.. autoclass:: cdl.param.BinningParam
    :no-index:
.. autoclass:: cdl.param.ButterworthParam
    :no-index:
.. autoclass:: cdl.param.DataTypeIParam
    :no-index:
.. autoclass:: cdl.param.FlatFieldParam
    :no-index:
.. autoclass:: cdl.param.GridParam
    :no-index:
.. autoclass:: cdl.param.HoughCircleParam
    :no-index:
.. autoclass:: cdl.param.LogP1Param
    :no-index:
.. autoclass:: cdl.param.ProfileParam
    :no-index:
.. autoclass:: cdl.param.ResizeParam
    :no-index:
.. autoclass:: cdl.param.RotateParam
    :no-index:
.. autoclass:: cdl.param.ZCalibrateParam
    :no-index:

Exposure correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.AdjustGammaParam
    :no-index:
.. autoclass:: cdl.param.AdjustLogParam
    :no-index:
.. autoclass:: cdl.param.AdjustSigmoidParam
    :no-index:
.. autoclass:: cdl.param.EqualizeAdaptHistParam
    :no-index:
.. autoclass:: cdl.param.EqualizeHistParam
    :no-index:
.. autoclass:: cdl.param.RescaleIntensityParam
    :no-index:

Restoration parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.DenoiseBilateralParam
    :no-index:
.. autoclass:: cdl.param.DenoiseTVParam
    :no-index:
.. autoclass:: cdl.param.DenoiseWaveletParam
    :no-index:

Morphological parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.MorphologyParam
    :no-index:

Edge detection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.CannyParam
    :no-index:

Detection parameters
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cdl.param.BlobDOGParam
    :no-index:
.. autoclass:: cdl.param.BlobDOHParam
    :no-index:
.. autoclass:: cdl.param.BlobLOGParam
    :no-index:
.. autoclass:: cdl.param.BlobOpenCVParam
    :no-index:
.. autoclass:: cdl.param.ContourShapeParam
    :no-index:
.. autoclass:: cdl.param.Peak2DDetectionParam
    :no-index:
"""

# pylint:disable=unused-import
# flake8: noqa

from cdl.core.computation.base import (
    ClipParam,
    FFTParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ROIDataParam,
    ThresholdParam,
)
from cdl.core.computation.image import (
    AverageProfileParam,
    BinningParam,
    ButterworthParam,
    DataTypeIParam,
    FlatFieldParam,
    GridParam,
    HoughCircleParam,
    LogP1Param,
    ProfileParam,
    ResizeParam,
    RotateParam,
    ZCalibrateParam,
)
from cdl.core.computation.image.detection import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    Peak2DDetectionParam,
)
from cdl.core.computation.image.edges import CannyParam
from cdl.core.computation.image.exposure import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    RescaleIntensityParam,
)
from cdl.core.computation.image.morphology import MorphologyParam
from cdl.core.computation.image.restoration import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from cdl.core.computation.signal import (
    DataTypeSParam,
    FWHMParam,
    InterpolationParam,
    NormalizeYParam,
    PeakDetectionParam,
    PolynomialFitParam,
    XYCalibrateParam,
)
