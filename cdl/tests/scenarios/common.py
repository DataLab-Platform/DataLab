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
    GaussianNoiseParam,
    create_noisy_signal,
    create_paracetamol_signal,
    create_peak2d_image,
    create_sincos_image,
)
from cdl.tests.features.common.newobject_unit_test import (
    iterate_image_creation,
    iterate_signal_creation,
)
from cdl.widgets import fitdialog


def __compute_1_to_1_operations(panel: SignalPanel | ImagePanel, number: int) -> None:
    """Test `compute_1_to_1` type operations on a signal or image

    Requires that one signal or image has been added at index."""
    assert len(panel) >= number - 1
    panel.objview.select_objects((number,))
    panel.processor.compute("gaussian_filter", dlp.GaussianParam())
    panel.processor.compute("moving_average", dlp.MovingAverageParam())
    panel.processor.compute("moving_median", dlp.MovingMedianParam())
    panel.processor.compute("wiener")
    panel.processor.compute("fft")
    panel.processor.compute("ifft")
    panel.processor.compute("abs")
    panel.processor.compute("magnitude_spectrum")
    panel.processor.compute("phase_spectrum")
    panel.processor.compute("psd")
    panel.remove_object()
    panel.processor.compute("re")
    panel.remove_object()
    panel.processor.compute("im")
    panel.remove_object()
    panel.processor.compute("astype", dlp.DataTypeIParam.create(dtype_str="float64"))
    panel.processor.compute("log10")
    panel.processor.compute("exp")
    panel.processor.compute("swap_axes")
    panel.processor.compute("swap_axes")


def compute_common_operations(panel: SignalPanel | ImagePanel) -> None:
    """Test operations common to signal/image

    Requires that two (and only two) signals/images are created/added to panel

    First signal/image is supposed to be always the same (reference)
    Second signal/image is the tested object
    """
    assert len(panel) == 2

    panel.objview.select_objects((2,))
    panel.processor.compute("difference", panel[1])  # difference with obj #1
    panel.remove_object()
    panel.objview.select_objects((2,))
    panel.processor.compute("quadratic_difference", panel[2])
    panel.delete_metadata()

    const_oper_param = dlp.ConstantParam.create(value=2.0)
    for const_oper in (
        "addition_constant",
        "difference_constant",
        "product_constant",
        "division_constant",
    ):
        panel.objview.select_objects((3,))
        panel.processor.compute(const_oper, const_oper_param)

    panel.objview.select_objects((3,))
    panel.remove_object()

    panel.objview.select_objects((1, 2))
    panel.processor.compute("addition")
    panel.objview.select_objects((1, 2))
    panel.processor.compute("addition")
    panel.objview.select_objects((1, 2))
    panel.processor.compute("product")

    param = dlp.ConstantParam()
    param.value = 2.0
    panel.processor.compute("addition_constant", param)
    panel.processor.compute("difference_constant", param)
    panel.processor.compute("product_constant", param)
    panel.processor.compute("division_constant", param)

    obj = panel.objmodel.get_groups()[0][-1]
    param = dlp.ClipParam()  # Clipping before division...
    param.upper = (obj.data.max() - obj.data.min()) * 0.8 + obj.data.min()
    panel.processor.compute("clip", param)

    param = dlp.NormalizeParam()
    for method_value, _method_name in param.methods:
        param.method = method_value
        panel.processor.compute("normalize", param)

    panel.objview.select_objects((3, 7))
    panel.processor.compute("division")
    panel.objview.select_objects((1, 2, 3))
    panel.processor.compute("average")

    panel.add_label_with_title()

    __compute_1_to_1_operations(panel, 2)


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
    panel.processor.compute("sqrt")
    panel.processor.compute("power", dlp.PowerParam.create(power=2))
    panel.processor.compute("reverse_x")
    panel.processor.compute("reverse_x")

    # Test filter methods
    for filter_func_name, paramclass in (
        ("lowpass", dlp.LowPassFilterParam),
        ("highpass", dlp.HighPassFilterParam),
        ("bandpass", dlp.BandPassFilterParam),
        ("bandstop", dlp.BandStopFilterParam),
    ):
        for method_value, _method_name in paramclass.methods:
            panel.objview.set_current_object(sig1)
            param = paramclass.create(method=method_value)
            param.update_from_obj(sig1)  # Use default cut-off frequencies
            panel.processor.compute(filter_func_name, param)

    # Test windowing methods
    noiseobj2 = noiseobj1.copy()
    win.add_object(noiseobj2)
    param = dlp.WindowingParam()
    for method_value, _method_name in param.methods:
        panel.objview.set_current_object(noiseobj2)
        param.method = method_value
        panel.processor.compute("windowing", param)

    win.add_object(sig1.copy())

    param = dlp.XYCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.compute("calibration", param)

    panel.processor.compute("derivative")
    panel.processor.compute("integral")

    param = dlp.PeakDetectionParam()
    panel.processor.compute_peak_detection(param)

    panel.processor.compute_multigaussianfit()

    panel.objview.select_objects([-3])
    sig = panel.objview.get_sel_objects()[0]
    i1 = data_size // 10
    i2 = len(sig.y) - i1
    roi = dlo.create_signal_roi([i1, i2], indices=True)
    panel.processor.compute_roi_extraction(roi)

    sig = create_noisy_signal(GaussianNoiseParam.create(sigma=5.0))
    panel.add_object(sig)
    param = dlp.PolynomialFitParam()
    panel.processor.compute_polyfit(param)
    for fittitle, fitfunc in (
        (_("Gaussian fit"), fitdialog.gaussianfit),
        (_("Lorentzian fit"), fitdialog.lorentzianfit),
        (_("Voigt fit"), fitdialog.voigtfit),
        (_("Linear fit"), fitdialog.linearfit),
        (_("Exponential fit"), fitdialog.exponentialfit),
        (_("CDF fit"), fitdialog.cdffit),
        (_("Sinusoidal fit"), fitdialog.sinusoidalfit),
    ):
        panel.objview.set_current_object(sig)
        panel.processor.compute_fit(fittitle, fitfunc)

    newparam = dlo.new_signal_param(_("Gaussian"), stype=dlo.SignalTypes.GAUSS)
    sig = dlo.create_signal_from_param(
        newparam, dlo.GaussLorentzVoigtParam(), edit=False
    )
    panel.add_object(sig)

    param = dlp.FWHMParam()
    for method_value, _method_name in param.methods:
        param.method = method_value
        panel.processor.compute("fwhm", param)
    panel.processor.compute("fw1e2")

    # Create a new signal which X values are a subset of sig1
    x = np.linspace(sig1.x.min(), sig1.x.max(), data_size // 2)[: data_size // 4]
    y = x * 0.0
    sig2 = dlo.create_signal("X values for interpolation", x, y)
    panel.add_object(sig2)

    # Test interpolation
    # pylint: disable=protected-access
    for method_choice_tuple in dlp.InterpolationParam.methods:
        method = method_choice_tuple[0]
        for fill_value in (None, 0.0):
            panel.objview.set_current_object(sig1)
            param = dlp.InterpolationParam.create(method=method, fill_value=fill_value)
            panel.processor.compute("interpolation", sig2, param)

    # Test resampling
    xmin, xmax = x[0], x[-1]
    for mode, dx, nbpts in (("dx", 0.1, 10), ("nbpts", 0.0, 100)):
        panel.objview.set_current_object(sig1)
        param = dlp.ResamplingParam.create(
            xmin=xmin, xmax=xmax, mode=mode, dx=dx, nbpts=nbpts
        )
        panel.processor.compute("resampling", param)

    # Test convolution
    panel.objview.set_current_object(sig1)
    panel.processor.compute("derivative")
    panel.processor.compute("convolution", sig1)

    # Test detrending
    panel.objview.set_current_object(sig1)
    # pylint: disable=protected-access
    for method_choice_tuple in dlp.DetrendingParam.methods:
        param = dlp.DetrendingParam.create(method=method_choice_tuple[0])
        panel.processor.compute("detrending", param)

    # Test histogram
    panel.objview.set_current_object(sig1)
    param = dlp.HistogramParam.create(bins=100)
    panel.processor.compute("histogram", param)

    # Test bandwidth and dynamic parameters
    panel.processor.compute("bandwidth_3db")
    panel.processor.compute("dynamic_parameters")


def run_image_computations(
    win: CDLMainWindow, data_size: int = 150, all_types: bool = True
) -> None:
    """Test image features"""
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

    # Test denoising methods
    param = dlp.ZCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.compute("calibration", param)
    param = dlp.DenoiseTVParam()
    panel.processor.compute("denoise_tv", param)
    param = dlp.DenoiseBilateralParam()
    panel.processor.compute("denoise_bilateral", param)
    param = dlp.DenoiseWaveletParam()
    panel.processor.compute("denoise_wavelet", param)

    # Test exposure methods
    ima2 = create_sincos_image(newparam)
    panel.add_object(ima2)
    panel.processor.compute("abs")  # Avoid neg. values for skimage correction methods
    param = dlp.AdjustGammaParam.create(gamma=0.5)
    panel.processor.compute("adjust_gamma", param)
    param = dlp.AdjustLogParam.create(gain=0.5)
    panel.processor.compute("adjust_log", param)
    param = dlp.AdjustSigmoidParam.create(gain=0.5)
    panel.processor.compute("adjust_sigmoid", param)
    param = dlp.EqualizeHistParam()
    panel.processor.compute("equalize_hist", param)
    param = dlp.EqualizeAdaptHistParam()
    panel.processor.compute("equalize_adapthist", param)
    param = dlp.RescaleIntensityParam()
    panel.processor.compute("rescale_intensity", param)

    # Test morphology methods
    param = dlp.MorphologyParam.create(radius=10)
    panel.processor.compute("denoise_tophat", param)
    panel.processor.compute("white_tophat", param)
    panel.processor.compute("black_tophat", param)
    param.radius = 1
    panel.processor.compute("erosion", param)
    panel.processor.compute("dilation", param)
    panel.processor.compute("opening", param)
    panel.processor.compute("closing", param)

    param = dlp.ButterworthParam.create(order=2, cut_off=0.5)
    panel.processor.compute("butterworth", param)

    param = dlp.CannyParam()
    panel.processor.compute("canny", param)

    # Test threshold methods
    ima2 = create_sincos_image(newparam)
    panel.add_object(ima2)
    param = dlp.ThresholdParam()
    for method_value, _method_name in param.methods:
        panel.objview.set_current_object(ima2)
        param = dlp.ThresholdParam.create(method=method_value)
        if method_value == "manual":
            param.value = (ima2.data.max() - ima2.data.min()) * 0.5 + ima2.data.min()
        panel.processor.compute("threshold", param)
    for func_name in (
        "threshold_isodata",
        "threshold_li",
        "threshold_mean",
        "threshold_minimum",
        "threshold_otsu",
        "threshold_triangle",
        "threshold_yen",
    ):
        panel.objview.set_current_object(ima2)
        panel.processor.compute(func_name)

    # Test edge detection methods
    ima2 = create_sincos_image(newparam)
    panel.add_object(ima2)
    for func_name in (
        "roberts",
        "prewitt",
        "prewitt_h",
        "prewitt_v",
        "sobel",
        "sobel_h",
        "sobel_v",
        "scharr",
        "scharr_h",
        "scharr_v",
        "farid",
        "farid_h",
        "farid_v",
        "laplace",
    ):
        panel.processor.compute(func_name)

    param = dlp.LogP1Param.create(n=1)
    panel.processor.compute("logp1", param)

    panel.processor.compute("rotate90")
    panel.processor.compute("rotate270")
    panel.processor.compute("fliph")
    panel.processor.compute("flipv")

    param = dlp.RotateParam.create(angle=5.0)
    for boundary in param.boundaries[:-1]:
        param.mode = boundary
        panel.processor.compute("rotate", param)

    param = dlp.ResizeParam.create(zoom=1.3)
    panel.processor.compute("resize", param)

    n = data_size // 10
    roi = dlo.create_image_roi(
        "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
    )
    panel.processor.compute_roi_extraction(roi)

    panel.processor.compute("centroid")
    panel.processor.compute("enclosing_circle")

    ima = create_peak2d_image(newparam)
    panel.add_object(ima)
    param = dlp.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.compute_peak_detection(param)

    param = dlp.ContourShapeParam()
    panel.processor.compute("contour_shape", param)

    param = dlp.BinningParam.create(sx=2, sy=2, operation="average")
    panel.processor.compute("binning", param)

    # Test histogram
    panel.objview.set_current_object(ima)
    param = dlp.HistogramParam.create(bins=100)
    panel.processor.compute("histogram", param)
