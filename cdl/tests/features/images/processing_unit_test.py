# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image processing functions
-----------------------------------------

Features from the "Processing" menu are covered by this test.
The "Processing" menu contains functions to process images, such as
denoising, FFT, thresholding, etc.

Some of the functions are tested here, such as the image clipping.
Other functions may be tested in different files, depending on the
complexity of the function.

[1] Implementation note regarding scikit-image methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following note applies to:
- thresholding methods (isodata, li, mean, minimum, otsu, triangle, yen)
- exposure methods (adjust_gamma, adjust_log, adjust_sigmoid, rescale_intensity,
  equalize_hist, equalize_adapthist)
- restoration methods (denoise_tv, denoise_bilateral, denoise_wavelet)
- morphology methods (white_tophat, black_tophat, erosion, dilation, opening, closing)
- edge detection methods (canny, roberts, prewitt, sobel, scharr, farid, laplace)

The thresholding, morphological, and edge detection methods are implemented
in the scikit-image library: those algorithms are considered to be validated,
so we can use them as reference.
As a consequence, the only purpose of the associated validation tests is to check
if the methods are correctly called and if the results are consistent with
the reference implementation.

In other words, we are not testing the correctness of the algorithms, but
the correctness of the interface between the DataLab and the scikit-image
libraries.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi
import scipy.signal as sps
from skimage import exposure, feature, filters, morphology, restoration, util

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from sigima_.tests.data import get_test_image
from sigima_.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_calibration() -> None:
    """Validation test for the image calibration processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.ZCalibrateParam()

    # Test with a = 1 and b = 0: should do nothing
    p.a, p.b = 1.0, 0.0
    dst = sigima_image.calibration(src, p)
    exp = np.array(src.data, dtype=float)
    check_array_result("Calibration[identity]", dst.data, exp)

    # Testing with random values of a and b
    p.a, p.b = 0.5, 0.1
    exp = p.a * src.data + p.b
    dst = sigima_image.calibration(src, p)
    check_array_result(f"Calibration[a={p.a},b={p.b}]", dst.data, exp)


@pytest.mark.validation
def test_image_swap_axes() -> None:
    """Validation test for the image axes swapping processing."""
    src = get_test_image("flower.npy")
    dst = sigima_image.swap_axes(src)
    exp = np.swapaxes(src.data, 0, 1)
    check_array_result("SwapAxes", dst.data, exp)


@pytest.mark.validation
def test_image_normalize() -> None:
    """Validation test for the image normalization processing."""
    src = get_test_image("flower.npy")
    src.data = np.array(src.data, dtype=float)
    src.data[20:30, 20:30] = np.nan  # Adding NaN values to the image
    p = sigima_.param.NormalizeParam()

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method_value, _method_name in p.methods:
        p.method = method_value
        dst = sigima_image.normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        exp_min, exp_max = None, None
        if p.method == "maximum":
            exp_min, exp_max = np.nanmin(src.data) / np.nanmax(src.data), 1.0
        elif p.method == "amplitude":
            exp_min, exp_max = 0.0, 1.0
        elif p.method == "area":
            area = np.nansum(src.data)
            exp_min, exp_max = np.nanmin(src.data) / area, np.nanmax(src.data) / area
        elif p.method == "energy":
            energy = np.sqrt(np.nansum(np.abs(src.data) ** 2))
            exp_min, exp_max = (
                np.nanmin(src.data) / energy,
                np.nanmax(src.data) / energy,
            )
        elif p.method == "rms":
            rms = np.sqrt(np.nanmean(np.abs(src.data) ** 2))
            exp_min, exp_max = np.nanmin(src.data) / rms, np.nanmax(src.data) / rms
        check_scalar_result(f"{title}|min", np.nanmin(dst.data), exp_min)
        check_scalar_result(f"{title}|max", np.nanmax(dst.data), exp_max)


@pytest.mark.validation
def test_image_clip() -> None:
    """Validation test for the image clipping processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (50, 100)):
        p.lower, p.upper = lower, upper
        dst = sigima_image.clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Validation test for the image offset correction processing."""
    src = get_test_image("flower.npy")
    # Defining the ROI that will be used to estimate the offset
    p = sigima_.obj.ROI2DParam.create(x0=0, y0=0, dx=50, dy=20)
    dst = sigima_image.offset_correction(src, p)
    ix0, iy0 = int(p.x0), int(p.y0)
    ix1, iy1 = int(p.x0 + p.dx), int(p.y0 + p.dy)
    exp = src.data - np.mean(src.data[iy0:iy1, ix0:ix1])
    check_array_result("OffsetCorrection", dst.data, exp)


@pytest.mark.validation
def test_image_gaussian_filter() -> None:
    """Validation test for the image Gaussian filter processing."""
    src = get_test_image("flower.npy")
    for sigma in (10.0, 50.0):
        p = sigima_.param.GaussianParam.create(sigma=sigma)
        dst = sigima_image.gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_average() -> None:
    """Validation test for the image moving average processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.MovingAverageParam.create(n=30)
    for mode in p.modes:
        p.mode = mode
        dst = sigima_image.moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_median() -> None:
    """Validation test for the image moving median processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.MovingMedianParam.create(n=5)
    for mode in p.modes:
        p.mode = mode
        dst = sigima_image.moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_wiener() -> None:
    """Validation test for the image Wiener filter processing."""
    src = get_test_image("flower.npy")
    dst = sigima_image.wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


@pytest.mark.validation
def test_threshold() -> None:
    """Validation test for the image threshold processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.ThresholdParam.create(value=100)
    dst = sigima_image.threshold(src, p)
    exp = util.img_as_ubyte(src.data > p.value)
    check_array_result(f"Threshold[{p.value}]", dst.data, exp)


def __generic_threshold_validation(method: str) -> None:
    """Generic test for thresholding methods."""
    # See [1] for more information about the validation of thresholding methods.
    src = get_test_image("flower.npy")
    dst = sigima_image.threshold(
        src, sigima_.param.ThresholdParam.create(method=method)
    )
    exp = util.img_as_ubyte(
        src.data > getattr(filters, f"threshold_{method}")(src.data)
    )
    check_array_result(f"Threshold{method.capitalize()}", dst.data, exp)


@pytest.mark.validation
def test_threshold_isodata() -> None:
    """Validation test for the image threshold Isodata processing."""
    __generic_threshold_validation("isodata")


@pytest.mark.validation
def test_threshold_li() -> None:
    """Validation test for the image threshold Li processing."""
    __generic_threshold_validation("li")


@pytest.mark.validation
def test_threshold_mean() -> None:
    """Validation test for the image threshold Mean processing."""
    __generic_threshold_validation("mean")


@pytest.mark.validation
def test_threshold_minimum() -> None:
    """Validation test for the image threshold Minimum processing."""
    __generic_threshold_validation("minimum")


@pytest.mark.validation
def test_threshold_otsu() -> None:
    """Validation test for the image threshold Otsu processing."""
    __generic_threshold_validation("otsu")


@pytest.mark.validation
def test_threshold_triangle() -> None:
    """Validation test for the image threshold Triangle processing."""
    __generic_threshold_validation("triangle")


@pytest.mark.validation
def test_threshold_yen() -> None:
    """Validation test for the image threshold Yen processing."""
    __generic_threshold_validation("yen")


@pytest.mark.validation
def test_adjust_gamma() -> None:
    """Validation test for the image gamma adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for gamma, gain in ((0.5, 1.0), (1.0, 2.0), (1.5, 0.5)):
        p = sigima_.param.AdjustGammaParam.create(gamma=gamma, gain=gain)
        dst = sigima_image.adjust_gamma(src, p)
        exp = exposure.adjust_gamma(src.data, gamma=gamma, gain=gain)
        check_array_result(f"AdjustGamma[gamma={gamma},gain={gain}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_log() -> None:
    """Validation test for the image logarithmic adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for gain, inv in ((1.0, False), (2.0, True)):
        p = sigima_.param.AdjustLogParam.create(gain=gain, inv=inv)
        dst = sigima_image.adjust_log(src, p)
        exp = exposure.adjust_log(src.data, gain=gain, inv=inv)
        check_array_result(f"AdjustLog[gain={gain},inv={inv}]", dst.data, exp)


@pytest.mark.validation
def test_adjust_sigmoid() -> None:
    """Validation test for the image sigmoid adjustment processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for cutoff, gain, inv in ((0.5, 1.0, False), (0.25, 2.0, True)):
        p = sigima_.param.AdjustSigmoidParam.create(cutoff=cutoff, gain=gain, inv=inv)
        dst = sigima_image.adjust_sigmoid(src, p)
        exp = exposure.adjust_sigmoid(src.data, cutoff=cutoff, gain=gain, inv=inv)
        check_array_result(
            f"AdjustSigmoid[cutoff={cutoff},gain={gain},inv={inv}]", dst.data, exp
        )


@pytest.mark.validation
def test_rescale_intensity() -> None:
    """Validation test for the image intensity rescaling processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    p = sigima_.param.RescaleIntensityParam.create(in_range=(0, 255), out_range=(0, 1))
    dst = sigima_image.rescale_intensity(src, p)
    exp = exposure.rescale_intensity(
        src.data, in_range=p.in_range, out_range=p.out_range
    )
    check_array_result("RescaleIntensity", dst.data, exp)


@pytest.mark.validation
def test_equalize_hist() -> None:
    """Validation test for the image histogram equalization processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for nbins in (256, 512):
        p = sigima_.param.EqualizeHistParam.create(nbins=nbins)
        dst = sigima_image.equalize_hist(src, p)
        exp = exposure.equalize_hist(src.data, nbins=nbins)
        check_array_result(f"EqualizeHist[nbins={nbins}]", dst.data, exp)


@pytest.mark.validation
def test_equalize_adapthist() -> None:
    """Validation test for the image adaptive histogram equalization processing."""
    # See [1] for more information about the validation of exposure methods.
    src = get_test_image("flower.npy")
    for clip_limit in (0.01, 0.1):
        p = sigima_.param.EqualizeAdaptHistParam.create(clip_limit=clip_limit)
        dst = sigima_image.equalize_adapthist(src, p)
        exp = exposure.equalize_adapthist(src.data, clip_limit=clip_limit)
        check_array_result(f"AdaptiveHist[clip_limit={clip_limit}]", dst.data, exp)


@pytest.mark.validation
def test_denoise_tv() -> None:
    """Validation test for the image Total Variation denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for weight, eps, mni in ((0.1, 0.0002, 200), (0.5, 0.0001, 100)):
        p = sigima_.param.DenoiseTVParam.create(
            weight=weight, eps=eps, max_num_iter=mni
        )
        dst = sigima_image.denoise_tv(src, p)
        exp = restoration.denoise_tv_chambolle(src.data, weight, eps, mni)
        check_array_result(
            f"DenoiseTV[weight={weight},eps={eps},max_num_iter={mni}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_bilateral() -> None:
    """Validation test for the image bilateral denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    for sigma, mode in ((1.0, "constant"), (2.0, "edge")):
        p = sigima_.param.DenoiseBilateralParam.create(sigma_spatial=sigma, mode=mode)
        dst = sigima_image.denoise_bilateral(src, p)
        exp = restoration.denoise_bilateral(src.data, sigma_spatial=sigma, mode=mode)
        check_array_result(
            f"DenoiseBilateral[sigma_spatial={sigma},mode={mode}]",
            dst.data,
            exp,
        )


@pytest.mark.validation
def test_denoise_wavelet() -> None:
    """Validation test for the image wavelet denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    src.data = src.data[::8, ::8]
    p = sigima_.param.DenoiseWaveletParam()
    for wavelets in ("db1", "db2", "db3"):
        for mode in p.modes:
            for method in ("BayesShrink",):
                p.wavelets, p.mode, p.method = wavelets, mode, method
                dst = sigima_image.denoise_wavelet(src, p)
                exp = restoration.denoise_wavelet(
                    src.data, wavelet=wavelets, mode=mode, method=method
                )
                check_array_result(
                    f"DenoiseWavelet[wavelets={wavelets},mode={mode},method={method}]",
                    dst.data,
                    exp,
                    atol=0.1,
                )


@pytest.mark.validation
def test_denoise_tophat() -> None:
    """Validation test for the image top-hat denoising processing."""
    # See [1] for more information about the validation of restoration methods.
    src = get_test_image("flower.npy")
    p = sigima_.param.MorphologyParam.create(radius=10)
    dst = sigima_image.denoise_tophat(src, p)
    footprint = morphology.disk(p.radius)
    exp = src.data - morphology.white_tophat(src.data, footprint=footprint)
    check_array_result(f"DenoiseTophat[radius={p.radius}]", dst.data, exp)


def __generic_morphology_validation(method: str) -> None:
    """Generic test for morphology methods."""
    # See [1] for more information about the validation of morphology methods.
    src = get_test_image("flower.npy")
    p = sigima_.param.MorphologyParam.create(radius=10)
    dst: sigima_.obj.ImageObj = getattr(sigima_image, method)(src, p)
    exp = getattr(morphology, method)(src.data, footprint=morphology.disk(p.radius))
    check_array_result(f"{method.capitalize()}[radius={p.radius}]", dst.data, exp)


@pytest.mark.validation
def test_white_tophat() -> None:
    """Validation test for the image white top-hat processing."""
    __generic_morphology_validation("white_tophat")


@pytest.mark.validation
def test_black_tophat() -> None:
    """Validation test for the image black top-hat processing."""
    __generic_morphology_validation("black_tophat")


@pytest.mark.validation
def test_erosion() -> None:
    """Validation test for the image erosion processing."""
    __generic_morphology_validation("erosion")


@pytest.mark.validation
def test_dilation() -> None:
    """Validation test for the image dilation processing."""
    __generic_morphology_validation("dilation")


@pytest.mark.validation
def test_opening() -> None:
    """Validation test for the image opening processing."""
    __generic_morphology_validation("opening")


@pytest.mark.validation
def test_closing() -> None:
    """Validation test for the image closing processing."""
    __generic_morphology_validation("closing")


@pytest.mark.validation
def test_canny() -> None:
    """Validation test for the image Canny edge detection processing."""
    # See [1] for more information about the validation of edge detection methods.
    src = get_test_image("flower.npy")
    p = sigima_.param.CannyParam.create(
        sigma=1.0, low_threshold=0.1, high_threshold=0.2
    )
    dst = sigima_image.canny(src, p)
    exp = util.img_as_ubyte(
        feature.canny(
            src.data,
            sigma=p.sigma,
            low_threshold=p.low_threshold,
            high_threshold=p.high_threshold,
            use_quantiles=p.use_quantiles,
            mode=p.mode,
            cval=p.cval,
        )
    )
    check_array_result(
        f"Canny[sigma={p.sigma},low_threshold={p.low_threshold},"
        f"high_threshold={p.high_threshold}]",
        dst.data,
        exp,
    )


def __generic_edge_validation(method: str) -> None:
    """Generic test for edge detection methods."""
    # See [1] for more information about the validation of edge detection methods.
    src = get_test_image("flower.npy")
    dst: sigima_.obj.ImageObj = getattr(sigima_image, method)(src)
    exp = getattr(filters, method)(src.data)
    check_array_result(f"{method.capitalize()}", dst.data, exp)


@pytest.mark.validation
def test_roberts() -> None:
    """Validation test for the image Roberts edge detection processing."""
    __generic_edge_validation("roberts")


@pytest.mark.validation
def test_prewitt() -> None:
    """Validation test for the image Prewitt edge detection processing."""
    __generic_edge_validation("prewitt")


@pytest.mark.validation
def test_prewitt_h() -> None:
    """Validation test for the image horizontal Prewitt edge detection processing."""
    __generic_edge_validation("prewitt_h")


@pytest.mark.validation
def test_prewitt_v() -> None:
    """Validation test for the image vertical Prewitt edge detection processing."""
    __generic_edge_validation("prewitt_v")


@pytest.mark.validation
def test_sobel() -> None:
    """Validation test for the image Sobel edge detection processing."""
    __generic_edge_validation("sobel")


@pytest.mark.validation
def test_sobel_h() -> None:
    """Validation test for the image horizontal Sobel edge detection processing."""
    __generic_edge_validation("sobel_h")


@pytest.mark.validation
def test_sobel_v() -> None:
    """Validation test for the image vertical Sobel edge detection processing."""
    __generic_edge_validation("sobel_v")


@pytest.mark.validation
def test_scharr() -> None:
    """Validation test for the image Scharr edge detection processing."""
    __generic_edge_validation("scharr")


@pytest.mark.validation
def test_scharr_h() -> None:
    """Validation test for the image horizontal Scharr edge detection processing."""
    __generic_edge_validation("scharr_h")


@pytest.mark.validation
def test_scharr_v() -> None:
    """Validation test for the image vertical Scharr edge detection processing."""
    __generic_edge_validation("scharr_v")


@pytest.mark.validation
def test_farid() -> None:
    """Validation test for the image Farid edge detection processing."""
    __generic_edge_validation("farid")


@pytest.mark.validation
def test_farid_h() -> None:
    """Validation test for the image horizontal Farid edge detection processing."""
    __generic_edge_validation("farid_h")


@pytest.mark.validation
def test_farid_v() -> None:
    """Validation test for the image vertical Farid edge detection processing."""
    __generic_edge_validation("farid_v")


@pytest.mark.validation
def test_laplace() -> None:
    """Validation test for the image Laplace edge detection processing."""
    __generic_edge_validation("laplace")


@pytest.mark.validation
def test_butterworth() -> None:
    """Validation test for the image Butterworth filter processing."""
    src = get_test_image("flower.npy")
    p = sigima_.param.ButterworthParam.create(order=2, cut_off=0.5, high_pass=False)
    dst = sigima_image.butterworth(src, p)
    exp = filters.butterworth(src.data, p.cut_off, p.high_pass, p.order)
    check_array_result(
        f"Butterworth[order={p.order},cut_off={p.cut_off},high_pass={p.high_pass}]",
        dst.data,
        exp,
    )


if __name__ == "__main__":
    test_image_calibration()
    test_image_swap_axes()
    test_image_normalize()
    test_image_clip()
    test_image_offset_correction()
    test_image_gaussian_filter()
    test_image_moving_average()
    test_image_moving_median()
    test_image_wiener()
    test_threshold()
    test_threshold_isodata()
    test_threshold_li()
    test_threshold_mean()
    test_threshold_minimum()
    test_threshold_otsu()
    test_threshold_triangle()
    test_threshold_yen()
    test_adjust_gamma()
    test_adjust_log()
    test_adjust_sigmoid()
    test_rescale_intensity()
    test_equalize_hist()
    test_equalize_adapthist()
    test_denoise_tv()
    test_denoise_bilateral()
    test_denoise_wavelet()
    test_denoise_tophat()
    test_white_tophat()
    test_black_tophat()
    test_erosion()
    test_dilation()
    test_opening()
    test_closing()
    test_canny()
    test_roberts()
    test_prewitt()
    test_prewitt_h()
    test_prewitt_v()
    test_sobel()
    test_sobel_h()
    test_sobel_v()
    test_scharr()
    test_scharr_h()
    test_scharr_v()
    test_farid()
    test_farid_h()
    test_farid_v()
    test_laplace()
    test_butterworth()
