# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Base Computation module
-------------------------------

This module defines the base parameters and functions used by the
:mod:`cdl.core.gui.processor` module.

It is based on the :mod:`cdl.algorithms` module, which defines the algorithms
that are applied to the data, and on the :mod:`cdl.core.model` module, which
defines the data model.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

from cdl.config import _


class GaussianParam(gds.DataSet):
    """Gaussian filter parameters"""

    sigma = gds.FloatItem("σ", default=1.0)


class MovingAverageParam(gds.DataSet):
    """Moving average parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1)


class MovingMedianParam(gds.DataSet):
    """Moving median parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1, even=False)


class ThresholdParam(gds.DataSet):
    """Threshold parameters"""

    value = gds.FloatItem(_("Threshold"))


class ClipParam(gds.DataSet):
    """Data clipping parameters"""

    value = gds.FloatItem(_("Clipping value"))


class ROIDataParam(gds.DataSet):
    """ROI Editor data"""

    roidata = gds.FloatArrayItem(_("ROI data"))
    singleobj = gds.BoolItem(_("Single object"))
    modified = gds.BoolItem(_("Modified")).set_prop("display", hide=True)

    # pylint: disable=arguments-differ
    @classmethod
    def create(cls, roidata: np.ndarray | None = None, singleobj: bool | None = None):
        """Create ROIDataParam instance"""
        if roidata is not None:
            roidata = np.array(roidata, dtype=int)
        instance = cls()
        instance.roidata = roidata
        instance.singleobj = singleobj
        return instance

    @property
    def is_empty(self) -> bool:
        """Return True if there is no ROI"""
        return self.roidata is None or self.roidata.size == 0


class FFTParam(gds.DataSet):
    """FFT parameters"""

    shift = gds.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))
