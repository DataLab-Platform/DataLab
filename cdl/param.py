# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`cdl.param`)
-----------------------------

The :mod:`cdl.param` module aims at providing all the dataset parameters that are used
by the :mod:`cdl.core.computation` and :mod:`cdl.core.gui.processor` packages.

Those datasets are defined in other modules:

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

Introduction to `DataSet` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The datasets listed in the following sections are used to define the parameters
necessary for the various computations and processing operations available in DataLab.

Each dataset is a subclass of :py:class:`guidata.dataset.datatypes.DataSet` and thus
needs to be instantiated before being used.

Here is a complete example of how to instantiate a dataset and access its parameters
with the :py:class:`cdl.param.BinningParam` dataset:

    .. autodataset:: cdl.param.BinningParam
        :no-index:
        :shownote:

Common parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: cdl.param.ClipParam
    :no-index:
.. autodataset:: cdl.param.FFTParam
    :no-index:
.. autodataset:: cdl.param.GaussianParam
    :no-index:
.. autodataset:: cdl.param.MovingAverageParam
    :no-index:
.. autodataset:: cdl.param.MovingMedianParam
    :no-index:
.. autodataset:: cdl.param.ROIDataParam
    :no-index:
.. autodataset:: cdl.param.ThresholdParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: cdl.param.DataTypeSParam
    :no-index:
.. autodataset:: cdl.param.FWHMParam
    :no-index:
.. autodataset:: cdl.param.NormalizeYParam
    :no-index:
.. autodataset:: cdl.param.PeakDetectionParam
    :no-index:
.. autodataset:: cdl.param.PolynomialFitParam
    :no-index:
.. autodataset:: cdl.param.XYCalibrateParam
    :no-index:
.. autodataset:: cdl.param.InterpolationParam
    :no-index:
.. autodataset:: cdl.param.ResamplingParam
    :no-index:
.. autodataset:: cdl.param.DetrendingParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.AverageProfileParam
    :no-index:
.. autodataset:: cdl.param.RadialProfileParam
    :no-index:
.. autodataset:: cdl.param.BinningParam
    :no-index:
.. autodataset:: cdl.param.ButterworthParam
    :no-index:
.. autodataset:: cdl.param.DataTypeIParam
    :no-index:
.. autodataset:: cdl.param.FlatFieldParam
    :no-index:
.. autodataset:: cdl.param.GridParam
    :no-index:
.. autodataset:: cdl.param.HoughCircleParam
    :no-index:
.. autodataset:: cdl.param.LogP1Param
    :no-index:
.. autodataset:: cdl.param.ProfileParam
    :no-index:
.. autodataset:: cdl.param.ResizeParam
    :no-index:
.. autodataset:: cdl.param.RotateParam
    :no-index:
.. autodataset:: cdl.param.ZCalibrateParam
    :no-index:

Exposure correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.AdjustGammaParam
    :no-index:
.. autodataset:: cdl.param.AdjustLogParam
    :no-index:
.. autodataset:: cdl.param.AdjustSigmoidParam
    :no-index:
.. autodataset:: cdl.param.EqualizeAdaptHistParam
    :no-index:
.. autodataset:: cdl.param.EqualizeHistParam
    :no-index:
.. autodataset:: cdl.param.RescaleIntensityParam
    :no-index:

Restoration parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.DenoiseBilateralParam
    :no-index:
.. autodataset:: cdl.param.DenoiseTVParam
    :no-index:
.. autodataset:: cdl.param.DenoiseWaveletParam
    :no-index:

Morphological parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.MorphologyParam
    :no-index:

Edge detection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.CannyParam
    :no-index:

Detection parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.BlobDOGParam
    :no-index:
.. autodataset:: cdl.param.BlobDOHParam
    :no-index:
.. autodataset:: cdl.param.BlobLOGParam
    :no-index:
.. autodataset:: cdl.param.BlobOpenCVParam
    :no-index:
.. autodataset:: cdl.param.ContourShapeParam
    :no-index:
.. autodataset:: cdl.param.Peak2DDetectionParam
    :no-index:
"""

# pylint:disable=unused-import
# flake8: noqa

from cdl.core.computation.base import (
    ClipParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
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
    RadialProfileParam,
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
    DetrendingParam,
    FWHMParam,
    InterpolationParam,
    NormalizeYParam,
    PeakDetectionParam,
    PolynomialFitParam,
    ResamplingParam,
    XYCalibrateParam,
)
