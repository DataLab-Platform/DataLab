# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

from __future__ import annotations

import pandas as pd
import sigima.objects
import sigima.params
from sigima.tests import data as test_data

from datalab.adapters_metadata import GeometryAdapter
from datalab.config import Conf
from datalab.tests import datalab_test_app_context


def __create_image_with_geometry_results() -> sigima.objects.ImageObj:
    """Create test image with geometry results"""
    param = sigima.objects.Gauss2DParam.create(
        height=600,
        width=600,
        title="Test image (with geometry results)",
        dtype=sigima.objects.ImageDatatypes.UINT16,
        x0=2,
        y0=3,
    )
    image = sigima.objects.create_image_from_param(param)
    for geometry in test_data.generate_geometry_results():
        GeometryAdapter(geometry).add_to(image)
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
        f"Result shapes length mismatch: {len(rsl1)} != {len(rsl2)}"
    )
    for rs1, rs2 in zip(rsl1, rsl2):
        df1 = rs1.to_dataframe()
        df2 = rs2.to_dataframe()
        assert len(df1) * 2 == len(df2), (
            f"Result shape dataframe length mismatch: {len(df1)} * 2 != {len(df2)}"
        )
        # Check that the second dataframe contains double the data
        # (original geometry result concatenated with itself)
        coord_cols = [col for col in df1.columns if col != "roi_index"]
        df1_doubled = pd.concat([df1, df1], ignore_index=True)
        pd.testing.assert_frame_equal(
            df2[coord_cols].sort_values(coord_cols[0]).reset_index(drop=True),
            df1_doubled[coord_cols].sort_values(coord_cols[0]).reset_index(drop=True),
        )


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
        obj2 = __create_image_with_geometry_results()
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
        for keep_results in (False, True):
            with Conf.proc.keep_results.temp(keep_results):
                # Test merging result shapes (duplicate obj, then compute average):
                for panel in (win.signalpanel, win.imagepanel):
                    panel.objview.select_objects((2,))
                    panel.duplicate_object()
                    panel.objview.select_objects((2, len(panel)))
                    panel.processor.run_feature("average")
                    last_obj = panel[len(panel)]
                    if keep_results:
                        __check_geometry_results_merge(panel[2], last_obj)
                        if panel is win.imagepanel:
                            __check_roi_merge(panel[2], last_obj)
                    else:
                        assert (
                            len(list(GeometryAdapter.iterate_from_obj(last_obj))) == 0
                        )


if __name__ == "__main__":
    test_geometry_results()
