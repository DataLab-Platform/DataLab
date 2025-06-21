# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

from __future__ import annotations

import numpy as np

import sigima_.obj as so
import sigima_.param
from cdl.config import Conf
from cdl.tests import cdltest_app_context
from cdl.tests import data as test_data


def create_image_with_resultshapes() -> so.ImageObj:
    """Create test image with resultshapes"""
    newparam = so.NewImageParam.create(
        height=600,
        width=600,
        title="Test image (with result shapes)",
        itype=so.ImageTypes.GAUSS,
        dtype=so.ImageDatatypes.UINT16,
    )
    addparam = so.Gauss2DParam.create(x0=2, y0=3)
    image = so.create_image_from_param(newparam, addparam)
    for mshape in test_data.create_resultshapes():
        mshape.add_to(image)
    return image


def __check_resultshapes_merge(
    obj1: so.SignalObj | so.ImageObj,
    obj2: so.SignalObj | so.ImageObj,
) -> None:
    """Check if result shapes merge properly: the scenario is to duplicate an object,
    then compute average. We thus have to check if the second object (average) has the
    expected result shapes (i.e. twice the number of result shapes of the original
    object for each result shape type).

    Args:
        obj1: Original object
        obj2: Merged object
    """
    for rs1, rs2 in zip(obj1.iterate_resultshapes(), obj2.iterate_resultshapes()):
        assert np.all(np.vstack([rs1.array, rs1.array])[:, 1:] == rs2.array[:, 1:])


def __check_roi_merge(
    obj1: so.SignalObj | so.ImageObj,
    obj2: so.SignalObj | so.ImageObj,
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


def test_resultshapes() -> None:
    """Result shapes test"""
    with cdltest_app_context(console=False) as win:
        obj1 = test_data.create_sincos_image()
        obj2 = create_image_with_resultshapes()
        obj2.roi = so.create_image_roi("rectangle", [10, 10, 50, 400])
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_noisy_signal(noised=noised)
            panel.add_object(sig)
            panel.processor.run_feature("fwhm", sigima_.param.FWHMParam())
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
        with Conf.proc.keep_results.temp(True):
            # Test merging result shapes (duplicate obj, then compute average):
            for panel in (win.signalpanel, win.imagepanel):
                panel.objview.select_objects((2,))
                panel.duplicate_object()
                panel.objview.select_objects((2, len(panel)))
                panel.processor.run_feature("average")
                __check_resultshapes_merge(panel[2], panel[len(panel)])
                if panel is win.imagepanel:
                    __check_roi_merge(panel[2], panel[len(panel)])


if __name__ == "__main__":
    test_resultshapes()
