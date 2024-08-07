# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameters (:mod:`cdl.param`)
-----------------------------

The :mod:`cdl.param` module aims at providing all the dataset parameters that are used
by the :mod:`cdl.computation` and :mod:`cdl.core.gui.processor` packages.

Those datasets are defined in other modules:

    - :mod:`cdl.computation.base`
    - :mod:`cdl.computation.image`
    - :mod:`cdl.computation.signal`

The :mod:`cdl.param` module is thus a convenient way to import all the sets of
parameters at once.

As a matter of fact, the following import statement is equivalent to the previous one:

.. code-block:: python

    # Original import statement
    from cdl.computation.base import MovingAverageParam
    from cdl.computation.signal import PolynomialFitParam
    from cdl.computation.image.exposure import EqualizeHistParam

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

.. autodataset:: cdl.param.ArithmeticParam
    :no-index:
.. autodataset:: cdl.param.ClipParam
    :no-index:
.. autodataset:: cdl.param.FFTParam
    :no-index:
.. autodataset:: cdl.param.SpectrumParam
    :no-index:
.. autodataset:: cdl.param.GaussianParam
    :no-index:
.. autodataset:: cdl.param.MovingAverageParam
    :no-index:
.. autodataset:: cdl.param.MovingMedianParam
    :no-index:
.. autodataset:: cdl.param.ROIDataParam
    :no-index:
.. autodataset:: cdl.param.ConstantParam
    :no-index:

Signal parameters
^^^^^^^^^^^^^^^^^

.. autodataset:: cdl.param.DataTypeSParam
    :no-index:
.. autodataset:: cdl.param.FWHMParam
    :no-index:
.. autodataset:: cdl.param.NormalizeParam
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
.. autodataset:: cdl.param.PowerParam
    :no-index:
.. autodataset:: cdl.param.WindowingParam
    :no-index:
.. autodataset:: cdl.param.LowPassFilterParam
    :no-index:
.. autodataset:: cdl.param.HighPassFilterParam
    :no-index:
.. autodataset:: cdl.param.BandPassFilterParam
    :no-index:
.. autodataset:: cdl.param.BandStopFilterParam
    :no-index:
.. autodataset:: cdl.param.DynamicParam
    :no-index:

Image parameters
^^^^^^^^^^^^^^^^

Base image parameters
~~~~~~~~~~~~~~~~~~~~~

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
.. autodataset:: cdl.param.LineProfileParam
    :no-index:
.. autodataset:: cdl.param.SegmentProfileParam
    :no-index:
.. autodataset:: cdl.param.AverageProfileParam
    :no-index:
.. autodataset:: cdl.param.RadialProfileParam
    :no-index:
.. autodataset:: cdl.param.ResizeParam
    :no-index:
.. autodataset:: cdl.param.RotateParam
    :no-index:
.. autodataset:: cdl.param.ZCalibrateParam
    :no-index:

Threshold parameters
~~~~~~~~~~~~~~~~~~~~

.. autodataset:: cdl.param.ThresholdParam
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

from cdl.computation.base import (
    ArithmeticParam,
    ClipParam,
    FFTParam,
    SpectrumParam,
    GaussianParam,
    HistogramParam,
    MovingAverageParam,
    MovingMedianParam,
    ROIDataParam,
    ConstantParam,
    NormalizeParam,
)
from cdl.computation.image import (
    AverageProfileParam,
    BinningParam,
    ButterworthParam,
    DataTypeIParam,
    FlatFieldParam,
    GridParam,
    HoughCircleParam,
    LogP1Param,
    LineProfileParam,
    SegmentProfileParam,
    RadialProfileParam,
    ResizeParam,
    RotateParam,
    ZCalibrateParam,
)
from cdl.computation.image.detection import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    Peak2DDetectionParam,
)
from cdl.computation.image.edges import CannyParam
from cdl.computation.image.threshold import ThresholdParam
from cdl.computation.image.exposure import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    RescaleIntensityParam,
)
from cdl.computation.image.morphology import MorphologyParam
from cdl.computation.image.restoration import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from cdl.computation.signal import (
    DataTypeSParam,
    DetrendingParam,
    FWHMParam,
    InterpolationParam,
    PeakDetectionParam,
    PolynomialFitParam,
    ResamplingParam,
    XYCalibrateParam,
    PowerParam,
    WindowingParam,
    LowPassFilterParam,
    HighPassFilterParam,
    BandPassFilterParam,
    BandStopFilterParam,
    DynamicParam,
)
