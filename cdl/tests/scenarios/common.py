# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Scenarios common functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from __future__ import annotations

import numpy as np

import cdl.obj as dlo
import cdl.param as dlp
from cdl.config import _
from cdl.core.gui.main import CDLMainWindow
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.tests.data import (
    create_paracetamol_signal,
    create_peak2d_image,
    create_sincos_image,
)
from cdl.tests.features.common.newobject_unit_test import (
    iterate_image_creation,
    iterate_signal_creation,
)
from cdl.widgets import fitdialog


def __compute_11_operations(panel: SignalPanel | ImagePanel, number: int) -> None:
    """Test compute_11 type operations on a signal or image

    Requires that one signal or image has been added at index."""
    assert len(panel) >= number - 1
    panel.objview.select_objects((number,))
    panel.processor.compute_gaussian_filter(dlp.GaussianParam())
    panel.processor.compute_moving_average(dlp.MovingAverageParam())
    panel.processor.compute_moving_median(dlp.MovingMedianParam())
    panel.processor.compute_wiener()
    panel.processor.compute_fft()
    panel.processor.compute_ifft()
    panel.processor.compute_abs()
    panel.remove_object()
    panel.processor.compute_re()
    panel.remove_object()
    panel.processor.compute_im()
    panel.remove_object()
    panel.processor.compute_astype(dlp.DataTypeIParam.create(dtype="float64"))
    panel.processor.compute_log10()
    panel.processor.compute_exp()
    panel.processor.compute_swap_axes()
    panel.processor.compute_swap_axes()


def compute_common_operations(panel: SignalPanel | ImagePanel) -> None:
    """Test operations common to signal/image

    Requires that two (and only two) signals/images are created/added to panel

    First signal/image is supposed to be always the same (reference)
    Second signal/image is the tested object
    """
    assert len(panel) == 2

    panel.objview.select_objects((2,))
    panel.processor.compute_difference(panel[1])  # difference with obj #1
    panel.remove_object()
    panel.objview.select_objects((2,))
    panel.processor.compute_quadratic_difference(panel[2])
    panel.delete_metadata()

    const_oper_param = dlp.ConstantOperationParam.create(value=2.0)
    for const_oper in (
        panel.processor.compute_sum_constant,
        panel.processor.compute_difference_constant,
        panel.processor.compute_product_by_constant,
        panel.processor.compute_division_by_contant,
    ):
        panel.objview.select_objects((3,))
        const_oper(param=const_oper_param)

    panel.objview.select_objects((3,))
    panel.remove_object()

    panel.objview.select_objects((1, 2))
    panel.processor.compute_sum()
    panel.objview.select_objects((1, 2))
    panel.processor.compute_sum()
    panel.objview.select_objects((1, 2))
    panel.processor.compute_product()

    param = dlp.ConstantOperationParam()
    param.value = 2.0
    panel.processor.compute_sum_constant(param)
    panel.processor.compute_difference_constant(param)
    panel.processor.compute_product_by_constant(param)
    panel.processor.compute_division_by_contant(param)

    obj = panel.objmodel.get_groups()[0][-1]
    param = dlp.ThresholdParam()
    param.value = (obj.data.max() - obj.data.min()) * 0.2 + obj.data.min()
    panel.processor.compute_threshold(param)
    param = dlp.ClipParam()  # Clipping before division...
    param.value = (obj.data.max() - obj.data.min()) * 0.8 + obj.data.min()
    panel.processor.compute_clip(param)

    param = dlp.NormalizeParam()
    for method_value, _method_name in param.methods:
        param.method = method_value
        panel.processor.compute_normalize(param)

    panel.objview.select_objects((3, 7))
    panel.processor.compute_division()
    panel.objview.select_objects((1, 2, 3))
    panel.processor.compute_average()

    panel.add_label_with_title()

    __compute_11_operations(panel, 2)


def run_signal_computations(
    win: CDLMainWindow, data_size: int = 500, all_types: bool = True
) -> None:
    """Testing signal features"""
    panel = win.signalpanel
    win.set_current_panel("signal")

    if all_types:
        for signal in iterate_signal_creation(data_size, non_zero=True):
            panel.add_object(create_paracetamol_signal(data_size))
            panel.add_object(signal)
            compute_common_operations(panel)
            panel.remove_all_objects()

    sig1 = create_paracetamol_signal(data_size)
    win.add_object(sig1)

    # Add new signal based on s0
    panel.objview.set_current_object(sig1)
    newparam = dlo.new_signal_param(
        _("Random function"), stype=dlo.SignalTypes.UNIFORMRANDOM
    )
    addparam = dlo.UniformRandomParam.create(vmin=0, vmax=sig1.y.max() * 0.2)
    noiseobj1 = panel.new_object(newparam, addparam=addparam, edit=False)

    compute_common_operations(panel)

    # Signal specific operations
    panel.processor.compute_sqrt()
    panel.processor.compute_pow(dlp.PowParam.create(exponent=2))
    panel.processor.compute_reverse_x()
    panel.processor.compute_reverse_x()

    # Test filter methods
    for filter_func, paramclass in (
        (panel.processor.compute_lowpass, dlp.LowPassFilterParam),
        (panel.processor.compute_highpass, dlp.HighPassFilterParam),
        (panel.processor.compute_bandpass, dlp.BandPassFilterParam),
        (panel.processor.compute_bandstop, dlp.BandStopFilterParam),
    ):
        for method_value, _method_name in paramclass.methods:
            panel.objview.set_current_object(sig1)
            param = paramclass.create(method=method_value)
            param.update_from_signal(sig1)  # Use default cut-off frequencies
            filter_func(param)

    # Test windowing methods
    noiseobj2 = noiseobj1.copy()
    win.add_object(noiseobj2)
    param = dlp.WindowingParam()
    for method_value, _method_name in param.methods:
        panel.objview.set_current_object(noiseobj2)
        param.method = method_value
        panel.processor.compute_windowing(param)

    win.add_object(create_paracetamol_signal(data_size))

    param = dlp.XYCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.compute_calibration(param)

    panel.processor.compute_derivative()
    panel.processor.compute_integral()

    param = dlp.PeakDetectionParam()
    panel.processor.compute_peak_detection(param)

    panel.processor.compute_multigaussianfit()

    panel.objview.select_objects([-3])
    sig = panel.objview.get_sel_objects()[0]
    i1 = data_size // 10
    i2 = len(sig.y) - i1
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create(roidata=[[i1, i2]]))

    param = dlp.PolynomialFitParam()
    panel.processor.compute_polyfit(param)
    panel.processor.compute_linearfit()
    panel.processor.compute_expfit()
    panel.processor.compute_cdffit()
    panel.processor.compute_sinfit()

    panel.processor.compute_fit(_("Gaussian fit"), fitdialog.gaussianfit)
    panel.processor.compute_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
    panel.processor.compute_fit(_("Voigt fit"), fitdialog.voigtfit)

    newparam = dlo.new_signal_param(_("Gaussian"), stype=dlo.SignalTypes.GAUSS)
    sig = dlo.create_signal_from_param(
        newparam, dlo.GaussLorentzVoigtParam(), edit=False
    )
    panel.add_object(sig)

    param = dlp.FWHMParam()
    for fittype, _name in param.fittypes:
        param.fittype = fittype
        panel.processor.compute_fwhm(param)
    panel.processor.compute_fw1e2()

    # Create a new signal which X values are a subset of sig1
    x = np.linspace(sig1.x.min(), sig1.x.max(), data_size // 2)[: data_size // 4]
    y = x * 0.0
    sig2 = dlo.create_signal("X values for interpolation", x, y)
    panel.add_object(sig2)

    # Test interpolation
    # pylint: disable=protected-access
    for method_choice_tuple in dlp.InterpolationParam._methods:
        method = method_choice_tuple[0]
        for fill_value in (None, 0.0):
            panel.objview.set_current_object(sig1)
            param = dlp.InterpolationParam.create(method=method, fill_value=fill_value)
            panel.processor.compute_interpolation(sig2, param)

    # Test resampling
    xmin, xmax = x[0], x[-1]
    for mode, dx, nbpts in (("dx", 0.1, 10), ("nbpts", 0.0, 100)):
        panel.objview.set_current_object(sig1)
        param = dlp.ResamplingParam.create(
            xmin=xmin, xmax=xmax, mode=mode, dx=dx, nbpts=nbpts
        )
        panel.processor.compute_resampling(param)

    # Test convolution
    panel.objview.set_current_object(sig1)
    panel.processor.compute_derivative()
    panel.processor.compute_convolution(sig1)

    # Test detrending
    panel.objview.set_current_object(sig1)
    # pylint: disable=protected-access
    for method_choice_tuple in dlp.DetrendingParam._methods:
        param = dlp.DetrendingParam.create(method=method_choice_tuple[0])
        panel.processor.compute_detrending(param)

    # Test histogram
    panel.objview.set_current_object(sig1)
    param = dlp.HistogramParam.create(bins=100)
    panel.processor.compute_histogram(param)


def run_image_computations(
    win: CDLMainWindow, data_size: int = 150, all_types: bool = True
) -> None:
    """Testing signal features"""
    win.set_current_panel("image")
    panel = win.imagepanel

    newparam = dlo.new_image_param(height=data_size, width=data_size)

    if all_types:
        for image in iterate_image_creation(data_size, non_zero=True):
            panel.add_object(create_sincos_image(newparam))
            panel.add_object(image)
            compute_common_operations(panel)
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

    compute_common_operations(panel)

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
        dlp.ROIDataParam.create(roidata=[[n, n, data_size - n, data_size - n]])
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

    # Test histogram
    panel.objview.set_current_object(ima)
    param = dlp.HistogramParam.create(bins=100)
    panel.processor.compute_histogram(param)
