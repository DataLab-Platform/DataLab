# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

from __future__ import annotations

import numpy as np
import sigima.objects
import sigima.params
from sigima.config import options as sigima_options
from sigima.objects import GeometryResult
from sigima.objects.scalar import KindShape
from sigima.tests import data as test_data

from datalab.adapters_metadata import GeometryAdapter
from datalab.tests import datalab_test_app_context


def create_image_with_geometry_results() -> sigima.objects.ImageObj:
    """Create test image with geometry results"""
    param = sigima.objects.Gauss2DParam.create(
        height=600,
        width=600,
        title="Test image (with result shapes)",
        dtype=sigima.objects.ImageDatatypes.UINT16,
        x0=2,
        y0=3,
    )
    image = sigima.objects.create_image_from_param(param)
    # Create geometry results for testing

    # Create a point geometry directly
    point_geom = GeometryResult(
        title="Point Test",
        kind=KindShape.POINT,
        coords=np.array([[10.0, 20.0]]),
        roi_indices=np.array([0], dtype=int),
        attrs={},
    )
    GeometryAdapter(point_geom).add_to(image)

    # Create a rectangle geometry directly
    rect_geom = GeometryResult(
        title="Rectangle Test",
        kind=KindShape.RECTANGLE,
        coords=np.array([[10.0, 20.0, 30.0, 40.0]]),
        roi_indices=np.array([0], dtype=int),
        attrs={},
    )
    GeometryAdapter(rect_geom).add_to(image)
    return image


def __check_geometry_results_merge(
    obj1: sigima.objects.SignalObj | sigima.objects.ImageObj,
    obj2: sigima.objects.SignalObj | sigima.objects.ImageObj,
) -> None:
    """Check if geometry results merge properly: the scenario is to duplicate an object,
    then compute average. We thus have to check if the second object (average) has the
    expected geometry results (i.e. twice the number of geometry results of the original
    object for each geometry result type).

    Args:
        obj1: Original object
        obj2: Merged object
    """
    rsl1, rsl2 = (
        list(GeometryAdapter.iterate_from_obj(obj1)),
        list(GeometryAdapter.iterate_from_obj(obj2)),
    )
    assert len(rsl2) == len(rsl1), (
        "Merged object the same number of result shapes as original object, "
        "but expected twice the number of result shapes for each type."
    )
    for rs1, rs2 in zip(rsl1, rsl2):
        assert rs1.array.shape[0] * 2 == rs2.array.shape[0], (
            f"Result shape array length mismatch: {rs1.array.shape[0]} * 2 != "
            f"{rs2.array.shape[0]}"
        )
        assert np.all(np.vstack([rs1.array, rs1.array])[:, 1:] == rs2.array[:, 1:])


def __check_roi_merge(
    obj1: sigima.objects.SignalObj | sigima.objects.ImageObj,
    obj2: sigima.objects.SignalObj | sigima.objects.ImageObj,
) -> None:
    """Check if ROI merge properly: the scenario is to duplicate an object,
    then compute average. We thus have to check if the second object (average) has the
    expected ROI (i.e. the union of the original object's ROI).

    Args:
        obj1: Original object
        obj2: Merged object
    """
    roi1 = obj1.roi
    roi2 = obj2.roi
    for single_roi2 in roi2:
        assert roi1.get_single_roi(0) == single_roi2


def test_geometry_results() -> None:
    """Geometry results test"""
    with datalab_test_app_context() as win:
        obj1 = test_data.create_sincos_image()
        obj2 = create_image_with_geometry_results()
        obj2.roi = sigima.objects.create_image_roi("rectangle", [10, 10, 50, 400])
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_noisy_signal(noised=noised)
            panel.add_object(sig)
            panel.processor.run_feature("fwhm", sigima.params.FWHMParam())
            panel.processor.run_feature("fw1e2")
        panel.objview.select_objects((1, 2))
        panel.show_results()
        panel.plot_results()
        win.set_current_panel("image")
        panel = win.imagepanel
        for obj in (obj1, obj2):
            panel.add_object(obj)
        panel.show_results()
        panel.plot_results()
        with sigima_options.keep_results.context(True):
            # Test merging result shapes (duplicate obj, then compute average):
            for panel in (win.signalpanel, win.imagepanel):
                panel.objview.select_objects((2,))
                panel.duplicate_object()
                panel.objview.select_objects((2, len(panel)))
                panel.processor.run_feature("average")
                __check_geometry_results_merge(panel[2], panel[len(panel)])
                if panel is win.imagepanel:
                    __check_roi_merge(panel[2], panel[len(panel)])


if __name__ == "__main__":
    test_geometry_results()
