# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module patching *guidata* and *plotpy* to adapt it to DataLab
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# Allows accessing protecting members, unused arguments, unused variables
# pylint: disable=W0212,W0613,W0612,E0102

from __future__ import annotations

import sys

import numpy as np
import plotpy.items
import plotpy.plot
import plotpy.tools
from plotpy._scaler import INTERP_NEAREST, _scale_rect
from plotpy.mathutils.arrayfuncs import get_nan_range
from qtpy import QtCore as QC
from qwt import QwtLinearScaleEngine, QwtScaleDraw
from qwt import QwtLogScaleEngine as QwtLog10ScaleEngine

from cdl.config import _


def monkeypatch_method(cls, patch_name):
    # This function's code was inspired from the following thread:
    # "[Python-Dev] Monkeypatching idioms -- elegant or ugly?"
    # by Robert Brewer <fumanchu at aminus.org>
    # (Tue Jan 15 19:13:25 CET 2008)
    """
    Add the decorated method to the given class; replace as needed.

    If the named method already exists on the given class, it will
    be replaced, and a reference to the old method is created as
    cls._old<patch_name><name>. If the "_old_<patch_name>_<name>" attribute
    already exists, KeyError is raised.
    """

    def decorator(func):
        """Decorateur wrapper function"""
        fname = func.__name__
        old_func = getattr(cls, fname, None)
        if old_func is not None:
            # Add the old func to a list of old funcs.
            old_ref = f"_old_{patch_name}_{fname}"
            # print old_ref, old_func
            old_attr = getattr(cls, old_ref, None)
            if old_attr is None:
                setattr(cls, old_ref, old_func)
            else:
                print(
                    f"Warning: {cls.__name__}.{fname} already patched",
                    file=sys.stderr,
                )
        setattr(cls, fname, func)
        return func

    return decorator


# ==============================================================================
#  Adding support for z-axis logarithmic scale
# ==============================================================================
class ZAxisLogTool(plotpy.tools.ToggleTool):
    """Patched tools.ToggleTool"""

    def __init__(self, manager):
        title = _("Z-Axis logarithmic scale")
        super().__init__(
            manager,
            title=title,
            toolbar_id=plotpy.tools.DefaultToolbarID,
            icon="zlog.svg",
        )

    def activate_command(self, plot, checked):
        """Reimplement tools.ToggleTool method"""
        for item in self.get_supported_items(plot):
            item.set_zaxis_log_state(not item.get_zaxis_log_state())
        plot.replot()
        self.update_status(plot)

    def get_supported_items(self, plot):
        """Reimplement tools.ToggleTool method"""
        items = [
            item
            for item in plot.get_items()
            if isinstance(item, plotpy.items.ImageItem)
            and not item.is_empty()
            and hasattr(item, "get_zaxis_log_state")
        ]
        if len(items) > 1:
            items = [item for item in items if item in plot.get_selected_items()]
        if items:
            self.action.setChecked(items[0].get_zaxis_log_state())
        return items

    def update_status(self, plot):
        """Reimplement tools.ToggleTool method"""
        self.action.setEnabled(len(self.get_supported_items(plot)) > 0)


@monkeypatch_method(plotpy.plot.PlotManager, "PlotManager")
def register_image_tools(self):
    """Reimplement plotpy.plot.PlotManager method"""
    self._old_PlotManager_register_image_tools()
    self.add_tool(ZAxisLogTool)


@monkeypatch_method(plotpy.items.ImageItem, "ImageItem")
def __init__(self, data=None, param=None):
    self._log_data = None
    self._lin_lut_range = None
    self._is_zaxis_log = False
    self._old_ImageItem___init__(data=data, param=param)


class ZLogScaleDraw(QwtScaleDraw):
    """Patched QwtScaleDraw"""

    def label(self, value):
        """Reimplement QwtScaleDraw method"""
        logvalue = int(10 ** min([100.0, value]) - 1)
        return super().label(logvalue)
        # XXX: [P3] This is not the cleanest way to show log-Z scale
        # (we should be able to trick PyQwt at another stage of the
        #  rendering process, in order to choose scale data properly)


@monkeypatch_method(plotpy.items.ImageItem, "ImageItem")
def set_zaxis_log_state(self, state):
    """Reimplement image.ImageItem method"""
    self._is_zaxis_log = state
    plot = self.plot()
    if state:
        self._lin_lut_range = self.get_lut_range()
        if self._log_data is None:
            self._log_data = np.array(
                np.log10(self.data.clip(1)), dtype=np.float64, copy=True
            )
        self.set_lut_range(get_nan_range(self._log_data))
        plot.setAxisScaleDraw(plot.yRight, ZLogScaleDraw())
        plot.setAxisScaleEngine(plot.yRight, QwtLog10ScaleEngine())
    else:
        self._log_data = None
        self.set_lut_range(self._lin_lut_range)
        plot.setAxisScaleDraw(plot.yRight, QwtScaleDraw())
        plot.setAxisScaleEngine(plot.yRight, QwtLinearScaleEngine())
    plot.update_colormap_axis(self)


@monkeypatch_method(plotpy.items.ImageItem, "ImageItem")
def get_zaxis_log_state(self):
    """Reimplement image.ImageItem method"""
    return self._is_zaxis_log


@monkeypatch_method(plotpy.items.ImageItem, "ImageItem")
def draw_image(self, painter, canvasRect, src_rect, dst_rect, xMap, yMap):
    """Reimplement image.ImageItem method"""
    if self.data is None:
        return
    src2 = self._rescale_src_rect(src_rect)
    dst_rect = tuple(int(i) for i in dst_rect)

    # Not the most efficient way to do it, but it works...
    # --------------------------------------------------------------------------
    if self.get_zaxis_log_state():
        data = self._log_data
    else:
        data = self.data
    # --------------------------------------------------------------------------

    dest = _scale_rect(
        data, src2, self._offscreen, dst_rect, self.lut, (INTERP_NEAREST,)
    )
    qrect = QC.QRectF(QC.QPointF(dest[0], dest[1]), QC.QPointF(dest[2], dest[3]))
    painter.drawImage(qrect, self._image, qrect)
