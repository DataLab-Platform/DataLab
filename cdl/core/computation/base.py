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

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np

from cdl.config import Conf, _


class GaussianParam(gdt.DataSet):
    """Gaussian filter parameters"""

    sigma = gdi.FloatItem("σ", default=1.0)


class MovingAverageParam(gdt.DataSet):
    """Moving average parameters"""

    n = gdi.IntItem(_("Size of the moving window"), default=3, min=1)


class MovingMedianParam(gdt.DataSet):
    """Moving median parameters"""

    n = gdi.IntItem(_("Size of the moving window"), default=3, min=1, even=False)


class ThresholdParam(gdt.DataSet):
    """Threshold parameters"""

    value = gdi.FloatItem(_("Threshold"))


class ClipParam(gdt.DataSet):
    """Data clipping parameters"""

    value = gdi.FloatItem(_("Clipping value"))


class ROIDataParam(gdt.DataSet):
    """ROI Editor data"""

    roidata = gdi.FloatArrayItem(_("ROI data"))
    singleobj = gdi.BoolItem(_("Single object"))
    modified = gdi.BoolItem(_("Modified")).set_prop("display", hide=True)

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


class FFTParam(gdt.DataSet):
    """FFT parameters"""

    shift = gdi.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))
