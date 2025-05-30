# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`sigima.param`)
-----------------------------

The :mod:`sigima.param` module aims at providing all the dataset parameters that are used
by the :mod:`sigima.computation` and :mod:`cdl.core.gui.processor` packages.

Those datasets are defined in other modules:

    - :mod:`sigima.base`
    - :mod:`sigima.image`
    - :mod:`sigima.signal`

The :mod:`sigima.param` module is thus a convenient way to import all the sets of
parameters at once.

As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from sigima.base import MovingAverageParam
    from sigima.signal import PolynomialFitParam
    from sigima.image.exposure import EqualizeHistParam

    # Equivalent import statement
    from sigima.param import MovingAverageParam, PolynomialFitParam, EqualizeHistParam

Introduction to `DataSet` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The datasets listed in the following sections are used to define the parameters
necessary for the various computations and processing operations available in DataLab.

Each dataset is a subclass of :py:class:`guidata.dataset.datatypes.DataSet` and thus
needs to be instantiated before being used.

Here is a complete example of how to instantiate a dataset and access its parameters
with the :py:class:`sigima.param.BinningParam` dataset:

    .. autodataset:: sigima.param.BinningParam
        :no-index:
        :shownote:

Common parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima.param.ArithmeticParam
    :no-index:
.. autodataset:: sigima.param.ClipParam
    :no-index:
.. autodataset:: sigima.param.ConstantParam
    :no-index:
.. autodataset:: sigima.param.FFTParam
    :no-index:
.. autodataset:: sigima.param.GaussianParam
    :no-index:
.. autodataset:: sigima.param.HistogramParam
    :no-index:
.. autodataset:: sigima.param.MovingAverageParam
    :no-index:
.. autodataset:: sigima.param.MovingMedianParam
    :no-index:
.. autodataset:: sigima.param.NormalizeParam
    :no-index:
.. autodataset:: sigima.param.SpectrumParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima.param.AllanVarianceParam
    :no-index:
.. autodataset:: sigima.param.AngleUnitParam
    :no-index:
.. autodataset:: sigima.param.BandPassFilterParam
    :no-index:
.. autodataset:: sigima.param.BandStopFilterParam
    :no-index:
.. autodataset:: sigima.param.DataTypeSParam
    :no-index:
.. autodataset:: sigima.param.DetrendingParam
    :no-index:
.. autodataset:: sigima.param.DynamicParam
    :no-index:
.. autodataset:: sigima.param.AbscissaParam
    :no-index:
.. autodataset:: sigima.param.OrdinateParam
    :no-index:
.. autodataset:: sigima.param.FWHMParam
    :no-index:
.. autodataset:: sigima.param.HighPassFilterParam
    :no-index:
.. autodataset:: sigima.param.InterpolationParam
    :no-index:
.. autodataset:: sigima.param.LowPassFilterParam
    :no-index:
.. autodataset:: sigima.param.PeakDetectionParam
    :no-index:
.. autodataset:: sigima.param.PolynomialFitParam
    :no-index:
.. autodataset:: sigima.param.PowerParam
    :no-index:
.. autodataset:: sigima.param.ResamplingParam
    :no-index:
.. autodataset:: sigima.param.WindowingParam
    :no-index:
.. autodataset:: sigima.param.XYCalibrateParam
    :no-index:
.. autodataset:: sigima.param.ZeroPadding1DParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.AverageProfileParam
    :no-index:
.. autodataset:: sigima.param.BinningParam
    :no-index:
.. autodataset:: sigima.param.ButterworthParam
    :no-index:
.. autodataset:: sigima.param.DataTypeIParam
    :no-index:
.. autodataset:: sigima.param.FlatFieldParam
    :no-index:
.. autodataset:: sigima.param.GridParam
    :no-index:
.. autodataset:: sigima.param.HoughCircleParam
    :no-index:
.. autodataset:: sigima.param.LineProfileParam
    :no-index:
.. autodataset:: sigima.param.LogP1Param
    :no-index:
.. autodataset:: sigima.param.RadialProfileParam
    :no-index:
.. autodataset:: sigima.param.ResizeParam
    :no-index:
.. autodataset:: sigima.param.RotateParam
    :no-index:
.. autodataset:: sigima.param.SegmentProfileParam
    :no-index:
.. autodataset:: sigima.param.ZCalibrateParam
    :no-index:
.. autodataset:: sigima.param.ZeroPadding2DParam
    :no-index:

Detection parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.BlobDOGParam
    :no-index:
.. autodataset:: sigima.param.BlobDOHParam
    :no-index:
.. autodataset:: sigima.param.BlobLOGParam
    :no-index:
.. autodataset:: sigima.param.BlobOpenCVParam
    :no-index:
.. autodataset:: sigima.param.ContourShapeParam
    :no-index:
.. autodataset:: sigima.param.Peak2DDetectionParam
    :no-index:

Edge detection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.CannyParam
    :no-index:

Exposure correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.AdjustGammaParam
    :no-index:
.. autodataset:: sigima.param.AdjustLogParam
    :no-index:
.. autodataset:: sigima.param.AdjustSigmoidParam
    :no-index:
.. autodataset:: sigima.param.EqualizeAdaptHistParam
    :no-index:
.. autodataset:: sigima.param.EqualizeHistParam
    :no-index:
.. autodataset:: sigima.param.RescaleIntensityParam
    :no-index:

Morphological parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.MorphologyParam
    :no-index:

Restoration parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.DenoiseBilateralParam
    :no-index:
.. autodataset:: sigima.param.DenoiseTVParam
    :no-index:
.. autodataset:: sigima.param.DenoiseWaveletParam
    :no-index:

Threshold parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima.param.ThresholdParam
    :no-index:
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima.base import (
    ArithmeticParam,
    ClipParam,
    ConstantParam,
    FFTParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    NormalizeParam,
    SpectrumParam,
)
from sigima.signal import (
    AllanVarianceParam,
    AbscissaParam,
    AngleUnitParam,
    BandPassFilterParam,
    BandStopFilterParam,
    DataTypeSParam,
    DetrendingParam,
    DynamicParam,
    OrdinateParam,
    FWHMParam,
    HighPassFilterParam,
    InterpolationParam,
    LowPassFilterParam,
    PeakDetectionParam,
    PolynomialFitParam,
    PowerParam,
    ResamplingParam,
    WindowingParam,
    XYCalibrateParam,
    ZeroPadding1DParam,
)

from sigima.image import (
    GridParam,
)
from sigima.image import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    HoughCircleParam,
    Peak2DDetectionParam,
)
from sigima.image import CannyParam
from sigima.image import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    RescaleIntensityParam,
    ZCalibrateParam,
)
from sigima.image import (
    AverageProfileParam,
    LineProfileParam,
    RadialProfileParam,
    SegmentProfileParam,
)
from sigima.image import ButterworthParam
from sigima.image import ZeroPadding2DParam
from sigima.image import BinningParam, ResizeParam, RotateParam
from sigima.image import DataTypeIParam, LogP1Param
from sigima.image import MorphologyParam
from sigima.image import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from sigima.image import ThresholdParam
