# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`sigima_.param`)
-----------------------------

The :mod:`sigima_.param` module aims at providing all the dataset parameters that are
used by the :mod:`sigima_.computation` and :mod:`cdl.gui.processor` packages.

Those datasets are defined in other modules:

    - :mod:`sigima_.computation.base`
    - :mod:`sigima_.computation.image`
    - :mod:`sigima_.computation.signal`

The :mod:`sigima_.param` module is thus a convenient way to import all the sets of
parameters at once.

As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from sigima_.computation.base import MovingAverageParam
    from sigima_.computation.signal import PolynomialFitParam
    from sigima_.computation.image.exposure import EqualizeHistParam

    # Equivalent import statement
    from sigima_.param import MovingAverageParam, PolynomialFitParam, EqualizeHistParam

Introduction to `DataSet` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The datasets listed in the following sections are used to define the parameters
necessary for the various computations and processing operations available in DataLab.

Each dataset is a subclass of :py:class:`guidata.dataset.datatypes.DataSet` and thus
needs to be instantiated before being used.

Here is a complete example of how to instantiate a dataset and access its parameters
with the :py:class:`sigima_.param.BinningParam` dataset:

    .. autodataset:: sigima_.param.BinningParam
        :no-index:
        :shownote:

Common parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima_.param.ArithmeticParam
    :no-index:
.. autodataset:: sigima_.param.ClipParam
    :no-index:
.. autodataset:: sigima_.param.ConstantParam
    :no-index:
.. autodataset:: sigima_.param.FFTParam
    :no-index:
.. autodataset:: sigima_.param.GaussianParam
    :no-index:
.. autodataset:: sigima_.param.HistogramParam
    :no-index:
.. autodataset:: sigima_.param.MovingAverageParam
    :no-index:
.. autodataset:: sigima_.param.MovingMedianParam
    :no-index:
.. autodataset:: sigima_.param.NormalizeParam
    :no-index:
.. autodataset:: sigima_.param.SpectrumParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: sigima_.param.AllanVarianceParam
    :no-index:
.. autodataset:: sigima_.param.AngleUnitParam
    :no-index:
.. autodataset:: sigima_.param.BandPassFilterParam
    :no-index:
.. autodataset:: sigima_.param.BandStopFilterParam
    :no-index:
.. autodataset:: sigima_.param.DataTypeSParam
    :no-index:
.. autodataset:: sigima_.param.DetrendingParam
    :no-index:
.. autodataset:: sigima_.param.DynamicParam
    :no-index:
.. autodataset:: sigima_.param.AbscissaParam
    :no-index:
.. autodataset:: sigima_.param.OrdinateParam
    :no-index:
.. autodataset:: sigima_.param.FWHMParam
    :no-index:
.. autodataset:: sigima_.param.HighPassFilterParam
    :no-index:
.. autodataset:: sigima_.param.InterpolationParam
    :no-index:
.. autodataset:: sigima_.param.LowPassFilterParam
    :no-index:
.. autodataset:: sigima_.param.PeakDetectionParam
    :no-index:
.. autodataset:: sigima_.param.PolynomialFitParam
    :no-index:
.. autodataset:: sigima_.param.PowerParam
    :no-index:
.. autodataset:: sigima_.param.ResamplingParam
    :no-index:
.. autodataset:: sigima_.param.WindowingParam
    :no-index:
.. autodataset:: sigima_.param.XYCalibrateParam
    :no-index:
.. autodataset:: sigima_.param.ZeroPadding1DParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.AverageProfileParam
    :no-index:
.. autodataset:: sigima_.param.BinningParam
    :no-index:
.. autodataset:: sigima_.param.ButterworthParam
    :no-index:
.. autodataset:: sigima_.param.DataTypeIParam
    :no-index:
.. autodataset:: sigima_.param.FlatFieldParam
    :no-index:
.. autodataset:: sigima_.param.GridParam
    :no-index:
.. autodataset:: sigima_.param.HoughCircleParam
    :no-index:
.. autodataset:: sigima_.param.LineProfileParam
    :no-index:
.. autodataset:: sigima_.param.LogP1Param
    :no-index:
.. autodataset:: sigima_.param.RadialProfileParam
    :no-index:
.. autodataset:: sigima_.param.ResizeParam
    :no-index:
.. autodataset:: sigima_.param.RotateParam
    :no-index:
.. autodataset:: sigima_.param.SegmentProfileParam
    :no-index:
.. autodataset:: sigima_.param.ZCalibrateParam
    :no-index:
.. autodataset:: sigima_.param.ZeroPadding2DParam
    :no-index:

Detection parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.BlobDOGParam
    :no-index:
.. autodataset:: sigima_.param.BlobDOHParam
    :no-index:
.. autodataset:: sigima_.param.BlobLOGParam
    :no-index:
.. autodataset:: sigima_.param.BlobOpenCVParam
    :no-index:
.. autodataset:: sigima_.param.ContourShapeParam
    :no-index:
.. autodataset:: sigima_.param.Peak2DDetectionParam
    :no-index:

Edge detection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.CannyParam
    :no-index:

Exposure correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.AdjustGammaParam
    :no-index:
.. autodataset:: sigima_.param.AdjustLogParam
    :no-index:
.. autodataset:: sigima_.param.AdjustSigmoidParam
    :no-index:
.. autodataset:: sigima_.param.EqualizeAdaptHistParam
    :no-index:
.. autodataset:: sigima_.param.EqualizeHistParam
    :no-index:
.. autodataset:: sigima_.param.RescaleIntensityParam
    :no-index:

Morphological parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.MorphologyParam
    :no-index:

Restoration parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.DenoiseBilateralParam
    :no-index:
.. autodataset:: sigima_.param.DenoiseTVParam
    :no-index:
.. autodataset:: sigima_.param.DenoiseWaveletParam
    :no-index:

Threshold parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: sigima_.param.ThresholdParam
    :no-index:
"""

# pylint:disable=unused-import
# flake8: noqa

from sigima_.computation.base import (
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
from sigima_.computation.signal import (
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

from sigima_.computation.image import (
    GridParam,
)
from sigima_.computation.image import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    HoughCircleParam,
    Peak2DDetectionParam,
)
from sigima_.computation.image import CannyParam
from sigima_.computation.image import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    FlatFieldParam,
    RescaleIntensityParam,
    ZCalibrateParam,
)
from sigima_.computation.image import (
    AverageProfileParam,
    LineProfileParam,
    RadialProfileParam,
    SegmentProfileParam,
)
from sigima_.computation.image import ButterworthParam
from sigima_.computation.image import ZeroPadding2DParam
from sigima_.computation.image import BinningParam, ResizeParam, RotateParam
from sigima_.computation.image import DataTypeIParam, LogP1Param
from sigima_.computation.image import MorphologyParam
from sigima_.computation.image import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from sigima_.computation.image import ThresholdParam
