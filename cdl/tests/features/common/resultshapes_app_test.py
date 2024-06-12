# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result shapes application test:

  - Create an image with metadata shapes and ROI
  - Further tests to be done manually: check if copy/paste metadata works
"""

# guitest: show

from __future__ import annotations

import numpy as np

import cdl.obj
import cdl.param
from cdl.tests import cdltest_app_context
from cdl.tests import data as test_data


def create_image_with_resultshapes():
    """Create test image with resultshapes"""
    newparam = cdl.obj.new_image_param(
        height=600,
        width=600,
        title="Test image (with result shapes)",
        itype=cdl.obj.ImageTypes.GAUSS,
        dtype=cdl.obj.ImageDatatypes.UINT16,
    )
    addparam = cdl.obj.Gauss2DParam.create(x0=2, y0=3)
    image = cdl.obj.create_image_from_param(newparam, addparam)
    for mshape in test_data.create_resultshapes():
        mshape.add_to(image)
    return image


def __check_resultshapes_merge(
    obj1: cdl.obj.SignalObj | cdl.obj.ImageObj,
    obj2: cdl.obj.SignalObj | cdl.obj.ImageObj,
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
    obj1: cdl.obj.SignalObj | cdl.obj.ImageObj,
    obj2: cdl.obj.SignalObj | cdl.obj.ImageObj,
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
    assert np.all(roi1 == roi2[: len(roi1)])
    assert np.all(roi1 == roi2[len(roi1) : len(roi1) * 2])
    assert roi1.shape[0] * 2 == roi2.shape[0]


def test_resultshapes():
    """Result shapes test"""
    with cdltest_app_context(console=False) as win:
        obj1 = test_data.create_sincos_image()
        obj2 = create_image_with_resultshapes()
        obj2.roi = np.array([[10, 10, 60, 400]], int)
        panel = win.signalpanel
        for noised in (False, True):
            sig = test_data.create_noisy_signal(noised=noised)
            panel.add_object(sig)
            panel.processor.compute_fwhm(cdl.param.FWHMParam())
            panel.processor.compute_fw1e2()
        panel.objview.select_objects((1, 2))
        panel.show_results()
        panel.plot_results()
        win.set_current_panel("image")
        panel = win.imagepanel
        for obj in (obj1, obj2):
            panel.add_object(obj)
        panel.show_results()
        panel.plot_results()
        # Test merging result shapes (duplicate obj, then compute average):
        for panel in (win.signalpanel, win.imagepanel):
            panel.objview.select_objects((2,))
            panel.duplicate_object()
            panel.objview.select_objects((2, len(panel)))
            panel.processor.compute_average()
            __check_resultshapes_merge(panel[2], panel[len(panel)])
            if panel is win.imagepanel:
                __check_roi_merge(panel[2], panel[len(panel)])


if __name__ == "__main__":
    test_resultshapes()
