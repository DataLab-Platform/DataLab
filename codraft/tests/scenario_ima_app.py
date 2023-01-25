# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Unit test scenario Image 01

Testing the following:
  - Add image object at startup
  - Create image (random type)
  - Sum two images
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from codraft.core.gui.main import CodraFTMainWindow
from codraft.core.gui.processor.image import (
    ContourShapeParam,
    DenoiseTVParam,
    LogP1Param,
    PeakDetectionParam,
    ResizeParam,
    RotateParam,
    ZCalibrateParam,
)
from codraft.core.model.base import UniformRandomParam
from codraft.core.model.image import ImageTypes, create_image, new_image_param
from codraft.tests import codraft_app_context
from codraft.tests.data import PeakDataParam, create_test_image1, get_peak2d_data
from codraft.tests.newobject_unit import iterate_image_creation
from codraft.tests.scenario_sig_app import test_common_operations

SHOW = True  # Show test in GUI-based test launcher


def test_image_features(win: CodraFTMainWindow, data_size: int = 150) -> None:
    """Testing signal features"""
    win.switch_to_image_panel()
    panel = win.imagepanel

    for image in iterate_image_creation(data_size, non_zero=True):
        panel.add_object(create_test_image1(data_size))
        panel.add_object(image)
        test_common_operations(panel)
        panel.remove_all_objects()

    ima1 = create_test_image1(data_size)
    panel.add_object(ima1)

    # Add new image based on i0
    panel.objlist.set_current_row(0)
    newparam = new_image_param(itype=ImageTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    param = ZCalibrateParam()
    param.a, param.b = 1.2, 0.1
    panel.processor.calibrate(param)

    param = DenoiseTVParam()
    panel.processor.compute_denoise_tv(param)

    param = LogP1Param()
    param.n = 1
    panel.processor.compute_logp1(param)

    panel.processor.rotate_90()
    panel.processor.rotate_270()
    panel.processor.flip_horizontally()
    panel.processor.flip_vertically()

    param = RotateParam()
    param.angle = 5.0
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.rotate_arbitrarily(param)

    param = ResizeParam()
    param.zoom = 1.3
    panel.processor.resize_image(param)

    n = data_size // 10
    panel.processor.extract_roi([[n, n, data_size - n, data_size - n]])

    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()

    data = get_peak2d_data(PeakDataParam(size=data_size))
    ima = create_image("Test image with peaks", data)
    panel.add_object(ima)
    param = PeakDetectionParam()
    param.create_rois = True
    panel.processor.compute_peak_detection(param)

    param = ContourShapeParam()
    panel.processor.compute_contour_shape(param)


def test():
    """Run image unit test scenario 1"""
    with codraft_app_context(save=True) as win:
        test_image_features(win)
        win.imagepanel.open_separate_view((0, 1, 2, 3))


if __name__ == "__main__":
    test()
