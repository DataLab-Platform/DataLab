# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Scenarios common functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from __future__ import annotations

import numpy as np
import sigima.enums
import sigima.objects
import sigima.params
from sigima.tests.data import (
    create_noisy_signal,
    create_paracetamol_signal,
    create_peak_image,
    create_sincos_image,
    iterate_image_creation,
    iterate_signal_creation,
)

from datalab.config import _
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalPanel
from datalab.widgets import fitdialog


def __compute_1_to_1_operations(panel: SignalPanel | ImagePanel, number: int) -> None:
    """Test `compute_1_to_1` type operations on a signal or image

    Requires that one signal or image has been added at index."""
    assert len(panel) >= number - 1
    panel.objview.select_objects((number,))
    panel.processor.run_feature("gaussian_filter", sigima.params.GaussianParam())
    panel.processor.run_feature("moving_average", sigima.params.MovingAverageParam())
    panel.processor.run_feature("moving_median", sigima.params.MovingMedianParam())
    panel.processor.run_feature("wiener")
    panel.processor.run_feature("fft")
    panel.processor.run_feature("ifft")
    panel.processor.run_feature("absolute")
    panel.processor.run_feature("magnitude_spectrum")
    panel.processor.run_feature("phase_spectrum")
    panel.processor.run_feature("psd")
    panel.remove_object()
    panel.processor.run_feature("real")
    panel.remove_object()
    panel.processor.run_feature("imag")
    panel.remove_object()
    panel.processor.run_feature(
        "astype",
        sigima.params.DataTypeIParam.create(dtype_str="float64"),
    )
    panel.processor.run_feature("log10")
    panel.processor.run_feature("exp")
    panel.processor.run_feature("transpose")
    panel.processor.run_feature("transpose")


def compute_common_operations(panel: SignalPanel | ImagePanel) -> None:
    """Test operations common to signal/image

    Requires that two (and only two) signals/images are created/added to panel

    First signal/image is supposed to be always the same (reference)
    Second signal/image is the tested object
    """
    assert len(panel) == 2

    panel.objview.select_objects((2,))
    panel.processor.run_feature("difference", panel[1])  # difference with obj #1
    panel.remove_object()
    panel.objview.select_objects((2,))
    panel.processor.run_feature("quadratic_difference", panel[2])
    panel.delete_metadata()

    const_oper_param = sigima.params.ConstantParam.create(value=2.0)
    for const_oper in (
        "addition_constant",
        "difference_constant",
        "product_constant",
        "division_constant",
    ):
        panel.objview.select_objects((3,))
        panel.processor.run_feature(const_oper, const_oper_param)

    panel.objview.select_objects((3,))
    panel.remove_object()

    panel.objview.select_objects((1, 2))
    panel.processor.run_feature("addition")
    panel.objview.select_objects((1, 2))
    panel.processor.run_feature("addition")
    panel.objview.select_objects((1, 2))
    panel.processor.run_feature("product")

    param = sigima.params.ConstantParam.create(value=2.0)
    panel.processor.run_feature("addition_constant", param)
    panel.processor.run_feature("difference_constant", param)
    panel.processor.run_feature("product_constant", param)
    panel.processor.run_feature("division_constant", param)

    obj = panel.objmodel.get_groups()[0][-1]
    param = sigima.params.ClipParam()  # Clipping before division...
    param.upper = (obj.data.max() - obj.data.min()) * 0.8 + obj.data.min()
    panel.processor.run_feature("clip", param)

    param = sigima.params.NormalizeParam()
    for method in sigima.enums.NormalizationMethod:
        param.method = method
        panel.processor.run_feature("normalize", param)

    panel.objview.select_objects((3, 7))
    panel.processor.run_feature("division")
    for feature_name in ("average", "standard_deviation"):
        panel.objview.select_objects((1, 2, 3))
        panel.processor.run_feature(feature_name)

    panel.add_label_with_title()

    __compute_1_to_1_operations(panel, 2)


def run_signal_computations(
    win: DLMainWindow, data_size: int = 500, all_types: bool = True
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
    param = sigima.objects.UniformDistribution1DParam.create(
        title=_("Random function"), vmin=0, vmax=sig1.y.max() * 0.2
    )
    noiseobj1 = panel.new_object(param, edit=False)

    compute_common_operations(panel)

    # Signal specific operations
    panel.processor.run_feature("sqrt")
    panel.processor.run_feature("power", sigima.params.PowerParam.create(power=2))
    panel.processor.run_feature("reverse_x")
    panel.processor.run_feature("reverse_x")

    # Test filter methods
    for filter_func_name, paramclass in (
        ("lowpass", sigima.params.LowPassFilterParam),
        ("highpass", sigima.params.HighPassFilterParam),
        ("bandpass", sigima.params.BandPassFilterParam),
        ("bandstop", sigima.params.BandStopFilterParam),
    ):
        for method in sigima.enums.FrequencyFilterMethod:
            panel.objview.set_current_object(sig1)
            param = paramclass.create(method=method)
            param.update_from_obj(sig1)  # Use default cut-off frequencies
            panel.processor.run_feature(filter_func_name, param)

    # Test windowing methods
    noiseobj2 = noiseobj1.copy()
    win.add_object(noiseobj2)
    param = sigima.params.WindowingParam()
    for method in sigima.enums.WindowingMethod:
        panel.objview.set_current_object(noiseobj2)
        param.method = method
        panel.processor.run_feature("apply_window", param)

    win.add_object(sig1.copy())

    param = sigima.params.XYCalibrateParam.create(a=1.2, b=0.1)
    panel.processor.run_feature("calibration", param)

    panel.processor.run_feature("derivative")
    panel.processor.run_feature("integral")

    param = sigima.params.PeakDetectionParam()
    panel.processor.compute_peak_detection(param)

    panel.processor.compute_multigaussianfit()

    panel.objview.select_objects([-3])
    sig = panel.objview.get_sel_objects()[0]
    i1 = data_size // 10
    i2 = len(sig.y) - i1
    roi = sigima.objects.create_signal_roi([i1, i2], indices=True)
    panel.processor.compute_roi_extraction(roi)

    sig = create_noisy_signal(sigima.objects.NormalDistributionParam.create(sigma=5.0))
    panel.add_object(sig)
    param = sigima.params.PolynomialFitParam()
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

    param = sigima.objects.GaussParam.create(title=_("Gaussian"))
    sig = sigima.objects.create_signal_from_param(param)
    panel.add_object(sig)

    param = sigima.params.FWHMParam()
    for method_value, _method_name in param.methods:
        param.method = method_value
        panel.processor.run_feature("fwhm", param)
    panel.processor.run_feature("fw1e2")

    # Create a new signal which X values are a subset of sig1
    x = np.linspace(sig1.x.min(), sig1.x.max(), data_size // 2)[: data_size // 4]
    y = x * 0.0
    sig2 = sigima.objects.create_signal("X values for interpolation", x, y)
    panel.add_object(sig2)

    # Test interpolation
    # pylint: disable=protected-access
    for method in sigima.enums.Interpolation1DMethod:
        for fill_value in (None, 0.0):
            panel.objview.set_current_object(sig1)
            param = sigima.params.InterpolationParam.create(
                method=method, fill_value=fill_value
            )
            panel.processor.run_feature("interpolate", sig2, param)

    # Test resampling
    xmin, xmax = x[0], x[-1]
    for mode, dx, nbpts in (("dx", 0.1, 10), ("nbpts", 0.0, 100)):
        panel.objview.set_current_object(sig1)
        param = sigima.params.Resampling1DParam.create(
            xmin=xmin, xmax=xmax, mode=mode, dx=dx, nbpts=nbpts
        )
        panel.processor.run_feature("resampling", param)

    # Test convolution
    panel.objview.set_current_object(sig1)
    panel.processor.run_feature("derivative")
    panel.processor.run_feature("convolution", sig1)

    # Test detrending
    panel.objview.set_current_object(sig1)
    # pylint: disable=protected-access
    for method_choice_tuple in sigima.params.DetrendingParam.methods:
        param = sigima.params.DetrendingParam.create(method=method_choice_tuple[0])
        panel.processor.run_feature("detrending", param)

    # Test histogram
    panel.objview.set_current_object(sig1)
    param = sigima.params.HistogramParam.create(bins=100)
    panel.processor.run_feature("histogram", param)

    # Test bandwidth and dynamic parameters
    panel.processor.run_feature("bandwidth_3db")
    panel.processor.run_feature("dynamic_parameters")


def run_image_computations(
    win: DLMainWindow, data_size: int = 150, all_types: bool = True
) -> None:
    """Test image features"""
    win.set_current_panel("image")
    panel = win.imagepanel

    newparam = sigima.objects.NewImageParam.create(height=data_size, width=data_size)

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
    unifparam = sigima.objects.UniformDistribution2DParam()
    unifparam.set_from_datatype(ima1.data.dtype)
    unifparam.vmax = int(ima1.data.max() * 0.2)
    panel.new_object(unifparam, edit=False)

    compute_common_operations(panel)

    # Test resampling
    w, h = ima1.data.shape[1], ima1.data.shape[0]
    for method, mode, dx, dy, width_param, height_param in (
        (sigima.enums.Interpolation2DMethod.NEAREST, "dxy", 0.5, 0.5, 10, 10),
        (sigima.enums.Interpolation2DMethod.LINEAR, "shape", 0.0, 0.0, w // 2, h // 2),
        (sigima.enums.Interpolation2DMethod.CUBIC, "shape", 0.0, 0.0, w * 2, h // 2),
    ):
        panel.objview.set_current_object(ima1)
        param = sigima.params.Resampling2DParam.create(
            method=method,
            mode=mode,
            dx=dx,
            dy=dy,
            width=width_param,
            height=height_param,
            xmin=ima1.x0,
            xmax=ima1.x0 + ima1.width,
            ymin=ima1.y0,
            ymax=ima1.y0 + ima1.height,
        )
        panel.processor.run_feature("resampling", param)

    # Test denoising methods
    param = sigima.params.XYZCalibrateParam.create(axis="z", a=1.2, b=0.1)
    panel.processor.run_feature("calibration", param)
    param = sigima.params.DenoiseTVParam()
    panel.processor.run_feature("denoise_tv", param)
    param = sigima.params.DenoiseBilateralParam()
    panel.processor.run_feature("denoise_bilateral", param)
    param = sigima.params.DenoiseWaveletParam()
    panel.processor.run_feature("denoise_wavelet", param)

    # Test exposure methods
    ima2 = create_sincos_image(newparam)
    panel.add_object(ima2)
    panel.processor.run_feature(
        "absolute"
    )  # Avoid neg. values for skimage correction methods
    param = sigima.params.AdjustGammaParam.create(gamma=0.5)
    panel.processor.run_feature("adjust_gamma", param)
    param = sigima.params.AdjustLogParam.create(gain=0.5)
    panel.processor.run_feature("adjust_log", param)
    param = sigima.params.AdjustSigmoidParam.create(gain=0.5)
    panel.processor.run_feature("adjust_sigmoid", param)
    param = sigima.params.EqualizeHistParam()
    panel.processor.run_feature("equalize_hist", param)
    param = sigima.params.EqualizeAdaptHistParam()
    panel.processor.run_feature("equalize_adapthist", param)
    param = sigima.params.RescaleIntensityParam()
    panel.processor.run_feature("rescale_intensity", param)

    # Test morphology methods
    param = sigima.params.MorphologyParam.create(radius=10)
    panel.processor.run_feature("denoise_tophat", param)
    panel.processor.run_feature("white_tophat", param)
    panel.processor.run_feature("black_tophat", param)
    param.radius = 1
    panel.processor.run_feature("erosion", param)
    panel.processor.run_feature("dilation", param)
    panel.processor.run_feature("opening", param)
    panel.processor.run_feature("closing", param)

    param = sigima.params.ButterworthParam.create(order=2, cut_off=0.5)
    panel.processor.run_feature("butterworth", param)

    param = sigima.params.CannyParam()
    panel.processor.run_feature("canny", param)

    # Test threshold methods
    ima2 = create_sincos_image(newparam)
    panel.add_object(ima2)
    param = sigima.params.ThresholdParam()
    for method_value, _method_name in param.methods:
        panel.objview.set_current_object(ima2)
        param = sigima.params.ThresholdParam.create(method=method_value)
        if method_value == "manual":
            param.value = (ima2.data.max() - ima2.data.min()) * 0.5 + ima2.data.min()
        panel.processor.run_feature("threshold", param)
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
        panel.processor.run_feature(func_name)

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
        panel.processor.run_feature(func_name)

    param = sigima.params.LogP1Param.create(n=1)
    panel.processor.run_feature("logp1", param)

    panel.processor.run_feature("rotate90")
    panel.processor.run_feature("rotate270")
    panel.processor.run_feature("fliph")
    panel.processor.run_feature("flipv")

    param = sigima.params.RotateParam.create(angle=5.0)
    for boundary in sigima.enums.BorderMode:
        if boundary is sigima.enums.BorderMode.MIRROR:
            continue
        param.mode = boundary
        panel.processor.run_feature("rotate", param)

    param = sigima.params.ResizeParam.create(zoom=1.3)
    panel.processor.run_feature("resize", param)

    n = data_size // 10
    roi = sigima.objects.create_image_roi(
        "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
    )
    panel.processor.compute_roi_extraction(roi)

    panel.processor.run_feature("centroid")
    panel.processor.run_feature("enclosing_circle")

    ima = create_peak_image(newparam)
    panel.add_object(ima)
    param = sigima.params.Peak2DDetectionParam.create(create_rois=True)
    panel.processor.compute_peak_detection(param)

    param = sigima.params.ContourShapeParam()
    panel.processor.run_feature("contour_shape", param)

    param = sigima.params.BinningParam.create(sx=2, sy=2, operation="average")
    panel.processor.run_feature("binning", param)

    # Test histogram
    panel.objview.set_current_object(ima)
    param = sigima.params.HistogramParam.create(bins=100)
    panel.processor.run_feature("histogram", param)
