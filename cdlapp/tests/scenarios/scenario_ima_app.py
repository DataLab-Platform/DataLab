# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Image processing test scenario
------------------------------

Testing all the image processing features.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import cdlapp.obj as dlo
import cdlapp.param as dlp
from cdlapp.config import Conf
from cdlapp.core.gui.main import CDLMainWindow
from cdlapp.env import execenv
from cdlapp.tests import cdl_app_context
from cdlapp.tests.data import create_peak2d_image, create_sincos_image
from cdlapp.tests.features.common.newobject_unit import iterate_image_creation
from cdlapp.tests.scenarios.scenario_sig_app import test_common_operations


def test_image_features(
    win: CDLMainWindow, data_size: int = 150, all_types: bool = True
) -> None:
    """Testing signal features"""
    win.switch_to_panel("image")
    panel = win.imagepanel

    newparam = dlo.new_image_param(height=data_size, width=data_size)

    if all_types:
        for image in iterate_image_creation(data_size, non_zero=True):
            panel.add_object(create_sincos_image(newparam))
            panel.add_object(image)
            test_common_operations(panel)
            panel.remove_all_objects()

    ima1 = create_sincos_image(newparam)
    panel.add_object(ima1)

    # Add new image based on i0
    panel.objview.set_current_object(ima1)
    newparam = dlo.new_image_param(itype=dlo.ImageTypes.UNIFORMRANDOM)
    addparam = dlo.UniformRandomParam()
    addparam.set_from_datatype(ima1.data.dtype)
    addparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(newparam, addparam=addparam, edit=False)

    test_common_operations(panel)

    param = dlp.ZCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.compute_calibration(param)

    param = dlp.DenoiseTVParam()
    panel.processor.compute_denoise_tv(param)

    param = dlp.DenoiseBilateralParam()
    panel.processor.compute_denoise_bilateral(param)

    param = dlp.DenoiseWaveletParam()
    panel.processor.compute_denoise_wavelet(param)

    panel.processor.compute_abs()  # Avoid neg. values for skimage correction methods

    param = dlp.AdjustGammaParam.create(gamma=0.5)
    panel.processor.compute_adjust_gamma(param)

    param = dlp.AdjustLogParam.create(gain=0.5)
    panel.processor.compute_adjust_log(param)

    param = dlp.AdjustSigmoidParam.create(gain=0.5)
    panel.processor.compute_adjust_sigmoid(param)

    param = dlp.EqualizeHistParam()
    panel.processor.compute_equalize_hist(param)

    param = dlp.EqualizeAdaptHistParam()
    panel.processor.compute_equalize_adapthist(param)

    param = dlp.RescaleIntensityParam()
    panel.processor.compute_rescale_intensity(param)

    param = dlp.MorphologyParam.create(radius=10)
    panel.processor.compute_denoise_tophat(param)
    panel.processor.compute_white_tophat(param)
    panel.processor.compute_black_tophat(param)
    param.radius = 1
    panel.processor.compute_erosion(param)
    panel.processor.compute_dilation(param)
    panel.processor.compute_opening(param)
    panel.processor.compute_closing(param)

    param = dlp.ButterworthParam.create(order=2, cut_off=0.5)
    panel.processor.compute_butterworth(param)

    ima2 = create_sincos_image(newparam)
    param = dlp.CannyParam()
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

    param = dlp.LogP1Param.create(n=1)
    panel.processor.compute_logp1(param)

    panel.processor.compute_rotate90()
    panel.processor.compute_rotate270()
    panel.processor.compute_fliph()
    panel.processor.compute_flipv()

    param = dlp.RotateParam.create(angle=5.0)
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.compute_rotate(param)

    param = dlp.ResizeParam.create(zoom=1.3)
    panel.processor.compute_resize(param)

    n = data_size // 10
    panel.processor.compute_roi_extraction(
        dlp.ROIDataParam.create([[n, n, data_size - n, data_size - n]])
    )

    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()

    ima = create_peak2d_image(newparam)
    panel.add_object(ima)
    param = dlp.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.compute_peak_detection(param)

    param = dlp.ContourShapeParam()
    panel.processor.compute_contour_shape(param)

    param = dlp.BinningParam.create(binning_x=2, binning_y=2, operation="average")
    panel.processor.compute_binning(param)


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
        execenv.print("==> OK")
        execenv.print("Testing image features *with* process isolation...")
        win.set_process_isolation_enabled(True)
        test_image_features(win, all_types=False)
        oids = win.imagepanel.objmodel.get_object_ids()
        win.imagepanel.open_separate_view(oids[:4])
        execenv.print("==> OK")


if __name__ == "__main__":
    test()
