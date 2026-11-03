# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction dialog: physical coordinates
-----------------------------------------------

Profile parameters are defined in the physical coordinates of the image: the
dialog commits the drawn shapes as they are (points beyond the image being brought
back to the nearest pixel centers), and draws the initial shape from them.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
import sigima.params
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from sigima.objects import ImageObj, create_image

from datalab.gui.profiledialog import ProfileExtractionDialog
from datalab.tests import datalab_test_app_context

NROWS, NCOLS = 20, 30
X0, Y0, DX, DY = -10.0, 5.0, 0.5, 2.0
XMAX, YMAX = X0 + (NCOLS - 1) * DX, Y0 + (NROWS - 1) * DY  # Last pixel centers


def make_image() -> ImageObj:
    """Return a calibrated image with anisotropic pixels"""
    obj = create_image("I", np.arange(NROWS * NCOLS, dtype=float).reshape(NROWS, -1))
    obj.x0, obj.y0, obj.dx, obj.dy = X0, Y0, DX, DY
    return obj


def draw(dialog: ProfileExtractionDialog, *coords: float) -> None:
    """Simulate the drawing of the profile shape (plot coordinates)"""
    zero = QC.QPointF(0.0, 0.0)
    dialog.cstool.add_shape_to_plot(dialog.get_plot(), zero, zero)
    shape = dialog.cstool.get_last_final_shape()
    if len(coords) == 2:
        shape.set_pos(*coords)
    else:
        shape.set_rect(*coords)
    dialog.cstool.SIG_TOOL_JOB_FINISHED.emit()


def test_profile_dialog_commits_physical_coordinates() -> None:
    """Drawn shapes are committed in physical coordinates"""
    with qt_app_context():
        param = sigima.params.LineProfileParam()
        dialog = ProfileExtractionDialog("line", param)
        dialog.set_obj(make_image())
        draw(dialog, -3.2, 21.7)
        assert (param.x, param.y) == pytest.approx((-3.2, 21.7))
        # Beyond the image: brought back to the nearest pixel centers
        draw(dialog, 100.0, -100.0)
        assert (param.x, param.y) == pytest.approx((XMAX, Y0))
        dialog.reject()

        param = sigima.params.SegmentProfileParam()
        dialog = ProfileExtractionDialog("segment", param)
        dialog.set_obj(make_image())
        # Drawn from right to left: committed from left to right
        draw(dialog, 2.0, 30.0, -20.0, 9.0)
        assert (param.x1, param.y1, param.x2, param.y2) == pytest.approx(
            (X0, 9.0, 2.0, 30.0)
        )
        dialog.reject()

        param = sigima.params.AverageProfileParam()
        dialog = ProfileExtractionDialog("rectangle", param)
        dialog.set_obj(make_image())
        draw(dialog, 1.5, 30.0, -4.0, 9.0)
        assert (param.x1, param.y1, param.x2, param.y2) == pytest.approx(
            (-4.0, 9.0, 1.5, 30.0)
        )
        dialog.reject()


def test_profile_dialog_initial_shape() -> None:
    """The initial shape is drawn from the parameters, fitted into the image"""
    with qt_app_context():
        # Parameters of a profile extracted from another image
        param = sigima.params.LineProfileParam.create(x=-50.0, y=12.0)
        dialog = ProfileExtractionDialog("line", param, add_initial_shape=True)
        dialog.set_obj(make_image())
        assert (param.x, param.y) == pytest.approx((X0, 12.0))
        assert dialog.shape.get_pos() == pytest.approx((X0, 12.0))
        dialog.reject()

        # Undefined bounds stand for the edges of the image
        param = sigima.params.AverageProfileParam.create(x1=-5.0)
        dialog = ProfileExtractionDialog("rectangle", param, add_initial_shape=True)
        dialog.set_obj(make_image())
        assert dialog.shape.get_rect() == pytest.approx(
            (-5.0, Y0 - DY / 2, XMAX + DX / 2, YMAX + DY / 2)
        )
        dialog.reject()


def test_profile_processor_keeps_last_position() -> None:
    """The last profile position is reused, instead of the center of the image"""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        panel.add_object(make_image())
        proc = panel.processor
        # pylint: disable=protected-access
        init = proc._ImageProcessor__init_profile_param
        saved = proc.PARAM_DEFAULTS.pop("LineProfileParam", None)
        try:
            edit, param, initial = init(None, sigima.params.LineProfileParam, "P")
            assert edit and not initial
            obj = panel.objview.get_current_object()
            assert (param.x, param.y) == pytest.approx((obj.xc, obj.yc))
            param.x, param.y = 1.0, 15.0
            edit, param, initial = init(None, sigima.params.LineProfileParam, "P")
            assert edit and initial
            assert (param.x, param.y) == pytest.approx((1.0, 15.0))
        finally:
            proc.PARAM_DEFAULTS.pop("LineProfileParam", None)
            if saved is not None:
                proc.PARAM_DEFAULTS["LineProfileParam"] = saved


if __name__ == "__main__":
    test_profile_dialog_commits_physical_coordinates()
    test_profile_dialog_initial_shape()
    test_profile_processor_keeps_last_position()
