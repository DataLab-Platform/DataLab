# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image processing test scenario
------------------------------

Testing all the image processing features.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import cdl.param
from cdl.config import Conf
from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.obj import ImageTypes, UniformRandomParam, create_image, new_image_param
from cdl.tests import cdl_app_context
from cdl.tests.data import PeakDataParam, create_test_image1, get_peak2d_data
from cdl.tests.features.common.newobject_unit import iterate_image_creation
from cdl.tests.scenarios.scenario_sig_app import test_common_operations

SHOW = True  # Show test in GUI-based test launcher


def test_image_features(
    win: CDLMainWindow, data_size: int = 150, all_types: bool = True
) -> None:
    """Testing signal features"""
    win.switch_to_panel("image")
    panel = win.imagepanel

    if all_types:
        for image in iterate_image_creation(data_size, non_zero=True):
            panel.add_object(create_test_image1(data_size))
            panel.add_object(image)
            test_common_operations(panel)
            panel.remove_all_objects()

    ima1 = create_test_image1(data_size)
    panel.add_object(ima1)

    # Add new image based on i0
    panel.objview.set_current_object(ima1)
    newparam = new_image_param(itype=ImageTypes.UNIFORMRANDOM)
    addparam = UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    param = cdl.param.ZCalibrateParam()
    param.a, param.b = 1.2, 0.1
    panel.processor.compute_calibration(param)

    param = cdl.param.DenoiseTVParam()
    panel.processor.compute_denoise_tv(param)

    param = cdl.param.DenoiseBilateralParam()
    panel.processor.compute_denoise_bilateral(param)

    param = cdl.param.DenoiseWaveletParam()
    panel.processor.compute_denoise_wavelet(param)

    param = cdl.param.AdjustGammaParam()
    param.gamma = 0.5
    panel.processor.compute_adjust_gamma(param)

    param = cdl.param.AdjustLogParam()
    param.gain = 0.5
    panel.processor.compute_adjust_log(param)

    param = cdl.param.AdjustSigmoidParam()
    param.gain = 0.5
    panel.processor.compute_adjust_sigmoid(param)

    param = cdl.param.EqualizeHistParam()
    panel.processor.compute_equalize_hist(param)

    param = cdl.param.EqualizeAdaptHistParam()
    panel.processor.compute_equalize_adapthist(param)

    param = cdl.param.RescaleIntensityParam()
    panel.processor.compute_rescale_intensity(param)

    param = cdl.param.MorphologyParam()
    param.radius = 10
    panel.processor.compute_denoise_tophat(param)
    panel.processor.compute_white_tophat(param)
    panel.processor.compute_black_tophat(param)
    param.radius = 1
    panel.processor.compute_erosion(param)
    panel.processor.compute_dilation(param)
    panel.processor.compute_opening(param)
    panel.processor.compute_closing(param)

    param = cdl.param.ButterworthParam()
    param.order = 2
    param.cut_off = 0.5
    panel.processor.compute_butterworth(param)

    ima2 = create_test_image1(data_size)
    param = cdl.param.CannyParam()
    panel.processor.compute_canny(param)
    panel.add_object(ima2)

    panel.processor.compute_roberts()

    panel.processor.compute_prewitt()
    panel.processor.compute_prewitt_h()
    panel.processor.compute_prewitt_v()

    panel.processor.compute_sobel()
    panel.processor.compute_sobel_h()
    panel.processor.compute_sobel_v()

    panel.processor.compute_scharr()
    panel.processor.compute_scharr_h()
    panel.processor.compute_scharr_v()

    panel.processor.compute_farid()
    panel.processor.compute_farid_h()
    panel.processor.compute_farid_v()

    panel.processor.compute_laplace()

    param = cdl.param.LogP1Param()
    param.n = 1
    panel.processor.compute_logp1(param)

    panel.processor.compute_rotate90()
    panel.processor.compute_rotate270()
    panel.processor.compute_fliph()
    panel.processor.compute_flipv()

    param = cdl.param.RotateParam()
    param.angle = 5.0
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.compute_rotate(param)

    param = cdl.param.ResizeParam()
    param.zoom = 1.3
    panel.processor.compute_resize(param)

    n = data_size // 10
    panel.processor.extract_roi([[n, n, data_size - n, data_size - n]])

    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()

    data = get_peak2d_data(PeakDataParam(size=data_size))
    ima = create_image("Test image with peaks", data)
    panel.add_object(ima)
    param = cdl.param.Peak2DDetectionParam()
    param.create_rois = True
    panel.processor.compute_peak_detection(param)

    param = cdl.param.ContourShapeParam()
    panel.processor.compute_contour_shape(param)


def test() -> None:
    """Run image unit test scenario 1"""
    assert (
        Conf.main.process_isolation_enabled.get()
    ), "Process isolation must be enabled"
    with cdl_app_context(save=True) as win:
        execenv.print("Testing image features without process isolation...")
        win.set_process_isolation_enabled(False)
        test_image_features(win)
        win.imagepanel.remove_all_objects()
        execenv.print("Testing image features *with* process isolation...")
        win.set_process_isolation_enabled(True)
        test_image_features(win, all_types=False)
        oids = win.imagepanel.objmodel.get_object_ids()
        win.imagepanel.open_separate_view(oids[:4])


if __name__ == "__main__":
    test()
