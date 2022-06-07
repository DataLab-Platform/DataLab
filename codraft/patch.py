# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Module patching *guiqwt* to adapt it to CodraFT
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# Allows accessing protecting members, unused arguments, unused variables
# pylint: disable=W0212,W0613,W0612

import sys

import guiqwt.curve
import guiqwt.histogram
import guiqwt.image
import guiqwt.plot
import guiqwt.tools
import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guiqwt import cross_section as cs
from guiqwt.transitional import QwtLinearScaleEngine
from qtpy.QtWidgets import QApplication, QMainWindow
from qwt import QwtLogScaleEngine as QwtLog10ScaleEngine
from qwt import QwtScaleDraw

from codraft.config import APP_NAME, _
from codraft.core.model.signal import create_signal


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


#  Patching CurveItem's "select" method to avoid showing giant ugly squares
@monkeypatch_method(guiqwt.curve.CurveItem, "CurveItem")
def select(self):
    """Select item"""
    self.selected = True
    plot = self.plot()
    if plot is not None:
        plot.blockSignals(True)
    pen = self.curveparam.line.build_pen()
    pen.setWidth(2.0)
    self.setPen(pen)
    if plot is not None:
        plot.blockSignals(False)
    self.invalidate_plot()


#  Adding centroid parameter to the image stats tool
@monkeypatch_method(guiqwt.image.BaseImageItem, "ImageItem")
def get_stats(self, x0, y0, x1, y1):
    """Return formatted string with stats on image rectangular area
    (output should be compatible with AnnotatedShape.get_infos)"""
    txt = self._old_ImageItem_get_stats(x0, y0, x1, y1)
    ix0, iy0, ix1, iy1 = self.get_closest_index_rect(x0, y0, x1, y1)
    data = self.data[iy0:iy1, ix0:ix1]

    # pylint: disable=C0415
    from codraft.core.computation.image import get_centroid_fourier

    c_i, c_j = get_centroid_fourier(data)
    c_x, c_y = self.get_plot_coordinates(c_j + ix0, c_i + iy0)
    xfmt = self.imageparam.xformat
    yfmt = self.imageparam.yformat
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
class ZAxisLogTool(guiqwt.tools.ToggleTool):
    """Patched tools.ToggleTool"""

    def __init__(self, manager):
        title = _("Z-Axis logarithmic scale")
        super().__init__(
            manager,
            title=title,
            toolbar_id=guiqwt.tools.DefaultToolbarID,
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
            if isinstance(item, guiqwt.image.ImageItem)
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


@monkeypatch_method(guiqwt.plot.PlotManager, "PlotManager")
def register_image_tools(self):
    """Reimplement guiqwt.plot.PlotManager method"""
    self._old_PlotManager_register_image_tools()
    self.add_tool(ZAxisLogTool)


@monkeypatch_method(guiqwt.image.ImageItem, "ImageItem")
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


@monkeypatch_method(guiqwt.image.ImageItem, "ImageItem")
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
        self.set_lut_range(
            [guiqwt.image._nanmin(self._log_data), guiqwt.image._nanmax(self._log_data)]
        )
        plot.setAxisScaleDraw(plot.yRight, ZLogScaleDraw())
        plot.setAxisScaleEngine(plot.yRight, QwtLog10ScaleEngine())
    else:
        self._log_data = None
        self.set_lut_range(self._lin_lut_range)
        plot.setAxisScaleDraw(plot.yRight, QwtScaleDraw())
        plot.setAxisScaleEngine(plot.yRight, QwtLinearScaleEngine())
    plot.update_colormap_axis(self)


@monkeypatch_method(guiqwt.image.ImageItem, "ImageItem")
def get_zaxis_log_state(self):
    """Reimplement image.ImageItem method"""
    return self._is_zaxis_log


@monkeypatch_method(guiqwt.image.ImageItem, "ImageItem")
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

    dest = guiqwt.image._scale_rect(
        data, src2, self._offscreen, dst_rect, self.lut, (guiqwt.image.INTERP_NEAREST,)
    )
    qrect = guiqwt.image.QRectF(
        guiqwt.image.QPointF(dest[0], dest[1]), guiqwt.image.QPointF(dest[2], dest[3])
    )
    painter.drawImage(qrect, self._image, qrect)


# ==============================================================================
#  Cross section : add a button to send curve to CodraFT's signal panel
# ==============================================================================
def to_codraft(cs_plot):
    """Send cross section curve to CodraFT's signal list"""
    win = None
    for win in QApplication.topLevelWidgets():
        if isinstance(win, QMainWindow):
            break
    if win is None or win.objectName() != APP_NAME:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=cyclic-import
        from codraft.core.gui import main

        # Note : this is the only way to retrieve the CodraFT main window instance
        # when the CrossSectionItem object is embedded into an image widget
        # parented to another main window.
        win = main.CodraFTMainWindow.get_instance()
        assert win is not None  # Should never happen

    for item in cs_plot.get_items():
        if not isinstance(item, guiqwt.curve.CurveItem):
            continue
        x, y, _dx, _dy = item.get_data()
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue

        signal = create_signal(item.curveparam.label)

        image_item = None
        for image_item, curve_item in cs_plot.known_items.items():
            if curve_item is item:
                break
        image_plot = image_item.plot()

        if isinstance(cs_plot, cs.VerticalCrossSectionPlot):
            xydata = np.vstack((y, x))
            xaxis_name = "left"
            xunit = image_plot.get_axis_unit("bottom")
            if xunit:
                signal.title += " " + xunit
        else:
            xaxis_name = "bottom"
            xydata = np.vstack((x, y))
            yunit = image_plot.get_axis_unit("left")
            if yunit:
                signal.title += " " + yunit

        signal.ylabel = image_plot.get_axis_title("right")
        signal.yunit = image_plot.get_axis_unit("right")
        signal.xlabel = image_plot.get_axis_title(xaxis_name)
        signal.xunit = image_plot.get_axis_unit(xaxis_name)

        signal.xydata = xydata
        win.signalpanel.add_object(signal)

    # Show CodraFT main window on top, if not already visible
    win.show()
    win.raise_()


@monkeypatch_method(cs.XCrossSection, "XCrossSection")
def add_actions_to_toolbar(self):
    """Add actions to toolbar"""
    to_codraft_ac = create_action(
        self,
        _("Process signal"),
        icon=get_icon("to_signal.svg"),
        triggered=lambda: to_codraft(self.cs_plot),
    )
    add_actions(self.toolbar, (to_codraft_ac, None))
    self._old_XCrossSection_add_actions_to_toolbar()
