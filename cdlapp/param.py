# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
DataLab Base Computation parameters module
------------------------------------------

This modules aims at providing all the dataset parameters that are used
by the :mod:`cdlapp.core.gui.processor` module.

Those datasets are defined other modules:
    - :mod:`cdlapp.core.computation.base`
    - :mod:`cdlapp.core.computation.image`
    - :mod:`cdlapp.core.computation.signal`

This module is thus a convenient way to import all the parameters at once.
"""

# pylint:disable=unused-import

from cdlapp.core.computation.base import (
    ClipParam,
    FFTParam,
    GaussianParam,
    MovingAverageParam,
    MovingMedianParam,
    ROIDataParam,
    ThresholdParam,
)
from cdlapp.core.computation.image import (
    BinningParam,
    ButterworthParam,
    FlatFieldParam,
    GridParam,
    HoughCircleParam,
    LogP1Param,
    ResizeParam,
    RotateParam,
    ZCalibrateParam,
)
from cdlapp.core.computation.image.detection import (
    BlobDOGParam,
    BlobDOHParam,
    BlobLOGParam,
    BlobOpenCVParam,
    ContourShapeParam,
    Peak2DDetectionParam,
)
from cdlapp.core.computation.image.edges import CannyParam
from cdlapp.core.computation.image.exposure import (
    AdjustGammaParam,
    AdjustLogParam,
    AdjustSigmoidParam,
    EqualizeAdaptHistParam,
    EqualizeHistParam,
    RescaleIntensityParam,
)
from cdlapp.core.computation.image.morphology import MorphologyParam
from cdlapp.core.computation.image.restoration import (
    DenoiseBilateralParam,
    DenoiseTVParam,
    DenoiseWaveletParam,
)
from cdlapp.core.computation.signal import (
    FWHMParam,
    NormalizeYParam,
    PeakDetectionParam,
    PolynomialFitParam,
    XYCalibrateParam,
)
