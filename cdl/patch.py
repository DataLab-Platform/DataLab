# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module patching *guidata* and *plotpy* to adapt it to DataLab
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# Allows accessing protecting members, unused arguments, unused variables
# pylint: disable=W0212,W0613,W0612,E0102

from __future__ import annotations

import sys
import warnings

import numpy as np
import plotpy.items
import plotpy.plot
import plotpy.tools
import scipy.integrate as spt
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from plotpy._scaler import INTERP_NEAREST, _scale_rect
from plotpy.events import StatefulEventFilter
from plotpy.mathutils.arrayfuncs import get_nan_range
from plotpy.panels.csection import csplot, cswidget
from plotpy.tools.cursor import BaseCursorTool
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy.QtWidgets import QApplication, QMainWindow
from qwt import QwtLinearScaleEngine, QwtScaleDraw
from qwt import QwtLogScaleEngine as QwtLog10ScaleEngine

from cdl.algorithms.signal import fwhm
from cdl.config import APP_NAME, _
from cdl.core.model.signal import create_signal


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


# Patching AnnotatedSegment "get_infos" method for a more compact text
@monkeypatch_method(plotpy.items.AnnotatedSegment, "AnnotatedSegment")
def get_infos(self):
    """Return formatted string with informations on current shape"""
    return "Δ = " + self.x_to_str(self.get_tr_length())


#  Patching curve stats to show additional information
@monkeypatch_method(plotpy.tools.CurveStatsTool, "CurveStatsTool")
def move(self, filter: StatefulEventFilter, event: QG.QMouseEvent) -> None:
    """Move tool action

    Args:
        filter: StatefulEventFilter instance
        event: Qt mouse event
    """
    BaseCursorTool.move(self, filter, event)

    # The following import is here to avoid circular imports
    # pylint: disable=import-outside-toplevel
    from plotpy.builder import make

    if self.label is None:
        plot = filter.plot
        curve = self.get_associated_item(plot)

        def fwhm_info(x, y):
            """Return FWHM information string"""
            with warnings.catch_warnings(record=True) as w:
                x0, y0, x1, y1 = fwhm((x, y), "zero-crossing")
                wstr = " ⚠️" if w else ""
            return f"{x1 - x0:g}{wstr}"

        self.label = make.computations(
            self.shape,
            "TL",
            [
                (
                    curve,
                    "%g &lt; x &lt; %g",
                    lambda *args: (args[0].min(), args[0].max()),
                ),
                (
                    curve,
                    "%g &lt; y &lt; %g",
                    lambda *args: (args[1].min(), args[1].max()),
                ),
                (curve, "&lt;y&gt;=%g", lambda *args: args[1].mean()),
                (curve, "σ(y)=%g", lambda *args: args[1].std()),
                (curve, "∑(y)=%g", lambda *args: spt.trapezoid(args[1])),
                (curve, "∫ydx=%g<br>", lambda *args: spt.trapezoid(args[1], args[0])),
                (curve, "FWHM = %s", fwhm_info),
            ],
        )
        self.label.attach(plot)
        self.label.setZ(plot.get_max_z() + 1)
        self.label.setVisible(True)


#  Adding centroid parameter to the image stats tool
@monkeypatch_method(plotpy.items.BaseImageItem, "ImageItem")
def get_stats(
    self,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    show_surface: bool = False,
    show_integral: bool = False,
) -> str:
    """Return formatted string with stats on image rectangular area
    (output should be compatible with AnnotatedShape.get_infos)

    Args:
        x0: X0
        y0: Y0
        x1: X1
        y1: Y1
        show_surface: Show surface (Default value = False)
        show_integral: Show integral (Default value = False)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        txt = self._old_ImageItem_get_stats(x0, y0, x1, y1, show_surface, show_integral)
    ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
    data = self.data[iy0:iy1, ix0:ix1]

    # pylint: disable=C0415
    from cdl.algorithms.image import get_centroid_fourier

    c_i, c_j = get_centroid_fourier(data)
    c_x, c_y = self.get_plot_coordinates(c_j + ix0, c_i + iy0)
    xfmt = self.param.xformat
    yfmt = self.param.yformat
    return (
        txt
        + "<br>"
        + "<br>".join(
            [
                "C|x = " + xfmt % c_x,
                "C|y = " + yfmt % c_y,
            ]
        )
    )


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


# ==============================================================================
#  Cross section : add a button to send curve to DataLab's signal panel
# ==============================================================================
def profile_to_signal(
    cs_plot: csplot.HorizontalCrossSectionPlot | csplot.VerticalCrossSectionPlot,
) -> None:
    """Send cross section curve to DataLab's signal list"""
    win = None
    for win in QApplication.topLevelWidgets():
        if isinstance(win, QMainWindow):
            break
    if win is None or win.objectName() != APP_NAME:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from cdl.core.gui import main

        # Note : this is the only way to retrieve the DataLab main window instance
        # when the CrossSectionItem object is embedded into an image widget
        # parented to another main window.
        win = main.CDLMainWindow.get_instance()
        assert win is not None  # Should never happen

    for item in cs_plot.get_items():
        if not isinstance(item, plotpy.items.CurveItem):
            continue
        x, y, _dx, _dy = item.get_data()
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue

        signal = create_signal(item.param.label)

        image_item = None
        for image_item, curve_item in cs_plot.known_items.items():
            if curve_item is item:
                break
        image_plot = image_item.plot()

        if isinstance(cs_plot, csplot.VerticalCrossSectionPlot):
            signal.set_xydata(y, x)
            xaxis_name = "left"
            xunit = image_plot.get_axis_unit("bottom")
            if xunit:
                signal.title += " " + xunit
        else:
            signal.set_xydata(x, y)
            xaxis_name = "bottom"
            yunit = image_plot.get_axis_unit("left")
            if yunit:
                signal.title += " " + yunit

        signal.ylabel = image_plot.get_axis_title("right")
        signal.yunit = image_plot.get_axis_unit("right")
        signal.xlabel = image_plot.get_axis_title(xaxis_name)
        signal.xunit = image_plot.get_axis_unit(xaxis_name)

        win.signalpanel.add_object(signal)

    # Show DataLab main window on top, if not already visible
    win.show()
    win.raise_()


@monkeypatch_method(cswidget.XCrossSection, "XCrossSection")
def add_actions_to_toolbar(self):
    """Add actions to toolbar"""
    to_signal_ac = create_action(
        self,
        _("Process signal"),
        icon=get_icon("to_signal.svg"),
        triggered=lambda: profile_to_signal(self.cs_plot),
    )
    add_actions(self.toolbar, (to_signal_ac, None))
    self._old_XCrossSection_add_actions_to_toolbar()
