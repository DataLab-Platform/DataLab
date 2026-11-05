# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
X axis scale draw unit tests
----------------------------

The X axis of the signal view keeps its scale draw when the view is refreshed, so
that its ticks stay aligned with the grid and the curves (the scale draw is only
replaced when leaving the date/time mode).
"""

# guitest: show

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from plotpy.plot import BasePlot
from qwt import QwtDateTimeScaleDraw
from sigima.objects import SignalObj, create_signal

from datalab.tests import datalab_test_app_context

# X bounds which are not graduations of the axis once autoscaled (as for the
# 'step.npy' test file): end labels do not overflow the axis ends
X_MIN, X_MAX = 3.1e-7, 4.08e-7


def _create_step_signal() -> SignalObj:
    """Create a signal whose X axis bounds are not graduations"""
    x = np.linspace(X_MIN, X_MAX, 500)
    return create_signal("Step", x, np.tanh((x - 3.15e-7) * 1e8))


def _create_datetime_signal() -> SignalObj:
    """Create a signal with a date/time X axis"""
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(seconds=i) for i in range(100)]
    obj = create_signal("Date/time", np.arange(100.0), np.sin(np.arange(100.0) / 10))
    obj.set_x_from_datetime(timestamps, unit="s")
    return obj


def _x_axis_ends(plot: BasePlot) -> tuple[list[float], list[float]]:
    """Return the ends of the X axis scale and of the X canvas map, in canvas
    coordinates"""
    widget = plot.axisWidget(plot.xBottom)
    offset = widget.x() - plot.canvas().x()
    smap = widget.scaleDraw().scaleMap()
    cmap = plot.canvasMap(plot.xBottom)
    return (
        sorted((smap.p1() + offset, smap.p2() + offset)),
        sorted((cmap.p1(), cmap.p2())),
    )


def test_refresh_keeps_x_scale_draw() -> None:
    """Refreshing the signal view keeps the scale draw of the X axis"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        plot = panel.plothandler.plot
        panel.add_object(_create_step_signal())
        scale_draw = plot.axisScaleDraw(plot.xBottom)
        panel.manual_refresh()
        assert plot.axisScaleDraw(plot.xBottom) is scale_draw
        panel.add_object(_create_step_signal())
        assert plot.axisScaleDraw(plot.xBottom) is scale_draw


def test_x_axis_aligned_with_canvas() -> None:
    """The X axis ticks are aligned with the grid and the curves after loading"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        plot = panel.plothandler.plot
        panel.add_object(_create_step_signal())
        scale_ends, canvas_ends = _x_axis_ends(plot)
        assert scale_ends == pytest.approx(canvas_ends, abs=0.5)
        panel.manual_refresh()
        scale_ends, canvas_ends = _x_axis_ends(plot)
        assert scale_ends == pytest.approx(canvas_ends, abs=0.5)


def test_datetime_x_axis_restored() -> None:
    """The date/time X axis formatting is removed when showing regular signals"""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        plot = panel.plothandler.plot
        panel.add_object(_create_datetime_signal())
        assert isinstance(plot.axisScaleDraw(plot.xBottom), QwtDateTimeScaleDraw)
        panel.add_object(_create_step_signal())
        scale_draw = plot.axisScaleDraw(plot.xBottom)
        assert not isinstance(scale_draw, QwtDateTimeScaleDraw)
        panel.manual_refresh()
        assert plot.axisScaleDraw(plot.xBottom) is scale_draw


if __name__ == "__main__":
    test_refresh_keeps_x_scale_draw()
    test_x_axis_aligned_with_canvas()
    test_datetime_x_axis_restored()
