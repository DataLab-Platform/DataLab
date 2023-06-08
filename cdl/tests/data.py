# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Test data functions

Functions creating test data: curves, images, ...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import dataclasses

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np

import cdl.obj
from cdl.config import _
from cdl.utils.tests import get_test_fnames


def get_test_signal(filename: str) -> cdl.obj.SignalObj:
    """Return test signal

    Args:
        filename (str): Filename

    Returns:
        SignalObj: Signal object
    """
    return cdl.obj.read_signal(get_test_fnames(filename)[0])


def get_test_image(filename: str) -> cdl.obj.ImageObj:
    """Return test image

    Args:
        filename (str): Filename

    Returns:
        ImageObj: Image object
    """
    return cdl.obj.read_image(get_test_fnames(filename)[0])


def create_paracetamol_signal(
    size: int | None = None, title: str | None = None
) -> cdl.obj.SignalObj:
    """Create test signal (Paracetamol molecule spectrum)

    Args:
        size (int, optional): Size of the data. Defaults to None.
        title (str, optional): Title of the signal. Defaults to None.

    Returns:
        SignalObj: Signal object
    """
    obj = cdl.obj.read_signal(get_test_fnames("paracetamol.txt")[0])
    if title is not None:
        obj.title = title
    if size is not None:
        x0, y0 = obj.xydata
        x1 = np.linspace(x0[0], x0[-1], size)
        y1 = np.interp(x1, x0, y0)
        obj.set_xydata(x1, y1)
    return obj


class GaussianNoiseParam(gdt.DataSet):
    """Gaussian noise parameters"""

    mu = gdi.FloatItem(
        _("Mean"),
        default=0.0,
        min=-100.0,
        max=100.0,
        help=_("Mean of the Gaussian distribution"),
    )
    sigma = gdi.FloatItem(
        _("Standard deviation"),
        default=0.1,
        min=0.0,
        max=100.0,
        help=_("Standard deviation of the Gaussian distribution"),
    )
    seed = gdi.IntItem(
        _("Seed"),
        default=1,
        min=0,
        max=1000000,
        help=_("Seed for random number generator"),
    )


def add_gaussian_noise_to_signal(
    signal: cdl.obj.SignalObj, p: GaussianNoiseParam | None = None
) -> None:
    """Add Gaussian (Normal-law) random noise to data

    Args:
        signal (cdl.obj.SignalObj): Signal object
        p (GaussianNoiseParam, optional): Gaussian noise parameters.
    """
    if p is None:
        p = GaussianNoiseParam()
    rng = np.random.default_rng(p.seed)
    signal.data += rng.normal(p.mu, p.sigma, size=signal.data.shape)


def create_noisy_signal(
    noiseparam: GaussianNoiseParam | None = None,
    newparam: cdl.obj.NewSignalParam | None = None,
    addparam: cdl.obj.GaussLorentzVoigtParam | None = None,
    title: str | None = None,
    noised: bool | None = None,
) -> cdl.obj.SignalObj:
    """Create curve data, optionally noised

    Args:
        noiseparam (GaussianNoiseParam, optional): Noise parameters. Default:
            None: No noise
        newparam (cdl.obj.NewSignalParam, optional): New signal parameters.
            Default: Gaussian, size=500, xmin=-10, xmax=10
        addparam (cdl.obj.GaussLorentzVoigtParam, optional): Additional parameters.
            Default: a=1.0, sigma=1.0, mu=0.0, ymin=0.0
        title (str, optional): Title of the signal. Default: None
            If not None, overrides the title in newparam
        noised (bool, optional): If True, add noise to the signal.
            Default: None (use noiseparam)
            If True, eventually creates a new noiseparam if None

    Returns:
        cdl.obj.SignalObj: Signal object
    """
    if newparam is None:
        newparam = cdl.obj.NewSignalParam()
        newparam.type = cdl.obj.SignalTypes.GAUSS
    if title is not None:
        newparam.title = title
    newparam.title = "Test signal (noisy)" if newparam.title is None else newparam.title
    if addparam is None:
        addparam = cdl.obj.GaussLorentzVoigtParam()
    if noised is not None and noiseparam is None:
        noiseparam = GaussianNoiseParam()
        noiseparam.sigma = 5.0
    sig = cdl.obj.create_signal_from_param(newparam, addparam)
    if noiseparam is not None:
        add_gaussian_noise_to_signal(sig, noiseparam)
    return sig


def create_2d_steps_data(size: int, width: int, dtype: np.dtype) -> np.ndarray:
    """Creating 2D steps data for testing purpose

    Args:
        size (int): Size of the data
        width (int): Width of the steps
        dtype (np.dtype): Data type

    Returns:
        np.ndarray: 2D data
    """
    data = np.zeros((size, size), dtype=dtype)
    value = 1
    for col in range(0, size - width + 1, width):
        data[:, col : col + width] = value
        value *= 10
    data2 = np.zeros_like(data)
    value = 1
    for row in range(0, size - width + 1, width):
        data2[row : row + width, :] = value
        value *= 10
    data += data2
    return data


def create_2d_random(
    size: int, dtype: np.dtype, level: float = 0.1, seed: int = 1
) -> np.ndarray:
    """Creating 2D Uniform-law random image

    Args:
        size (int): Size of the data
        dtype (np.dtype): Data type
        level (float, optional): Level of the random noise. Defaults to 0.1.
        seed (int, optional): Seed for random number generator. Defaults to 1.

    Returns:
        np.ndarray: 2D data
    """
    rng = np.random.default_rng(seed)
    amp = np.iinfo(dtype).max * level
    return np.array(rng.random((size, size)) * amp, dtype=dtype)


def create_2d_gaussian(
    size: int,
    dtype: np.dtype,
    x0: float = 0,
    y0: float = 0,
    mu: float = 0.0,
    sigma: float = 2.0,
    amp: float | None = None,
) -> np.ndarray:
    """Creating 2D Gaussian (-10 <= x <= 10 and -10 <= y <= 10)

    Args:
        size (int): Size of the data
        dtype (np.dtype): Data type
        x0 (float, optional): x0. Defaults to 0.
        y0 (float, optional): y0. Defaults to 0.
        mu (float, optional): mu. Defaults to 0.0.
        sigma (float, optional): sigma. Defaults to 2.0.
        amp (float, optional): Amplitude. Defaults to None.

    Returns:
        np.ndarray: 2D data
    """
    xydata = np.linspace(-10, 10, size)
    x, y = np.meshgrid(xydata, xydata)
    if amp is None:
        amp = np.iinfo(dtype).max * 0.5
    return np.array(
        amp
        * np.exp(
            -((np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - mu) ** 2) / (2.0 * sigma**2)
        ),
        dtype=dtype,
    )


def get_laser_spot_data() -> list[np.ndarray]:
    """Return a list of NumPy arrays containing images which are relevant for
    testing laser spot image processing features

    Returns:
        list[np.ndarray]: List of NumPy arrays
    """
    znoise = create_2d_random(2000, np.uint16)
    zgauss = create_2d_gaussian(2000, np.uint16, x0=2.0, y0=-3.0)
    return [zgauss + znoise] + [
        cdl.obj.read_image(fname).data for fname in get_test_fnames("*.scor-data")
    ]


@dataclasses.dataclass
class PeakDataParam:
    """Peak data test image parameters

    Attributes:
        size (int): Size of the data
        n_points (int): Number of points
        sigma_gauss2d (float): Sigma of the 2D Gaussian
        amp_gauss2d (int): Amplitude of the 2D Gaussian
        mu_noise (int): Mean of the Gaussian distribution
        sigma_noise (int): Standard deviation of the Gaussian distribution
        dx0 (float): x0
        dy0 (float): y0
        att (float): Attenuation
    """

    size: int = 2000
    n_points: int = 4
    sigma_gauss2d: float = 0.06
    amp_gauss2d: int = 1900
    mu_noise: int = 845
    sigma_noise: int = 25
    dx0: float = 0.0
    dy0: float = 0.0
    att: float = 1.0


def get_peak2d_data(
    p: PeakDataParam | None = None, seed: int | None = None, multi: bool = False
) -> np.ndarray:
    """Return a list of NumPy arrays containing images which are relevant for
    testing 2D peak detection or similar image processing features

    Args:
        p (PeakDataParam, optional): Peak data test image parameters. Defaults to None.
        seed (int, optional): Seed for random number generator. Defaults to None.
        multi (bool, optional): If True, multiple peaks are generated.
            Defaults to False.

    Returns:
        np.ndarray: 2D data
    """
    if p is None:
        p = PeakDataParam()
    delta = 0.1
    rng = np.random.default_rng(seed)
    coords = (rng.random((p.n_points, 2)) - 0.5) * 10 * (1 - delta)
    data = rng.normal(p.mu_noise, p.sigma_noise, size=(p.size, p.size))
    multi_nb = 2 if multi else 1
    for x0, y0 in coords:
        for idx in range(multi_nb):
            if idx != 0:
                p.dx0 = 0.08 + rng.random() * 0.08
                p.dy0 = 0.08 + rng.random() * 0.08
                p.att = 0.2 + rng.random() * 0.8
            data += create_2d_gaussian(
                p.size,
                np.uint16,
                x0=x0 + p.dx0,
                y0=y0 + p.dy0,
                sigma=p.sigma_gauss2d,
                amp=p.amp_gauss2d / multi_nb * p.att,
            )
    return data


def __set_default_size_dtype(
    p: cdl.obj.NewImageParam | None = None,
) -> cdl.obj.NewImageParam:
    """Set default shape and dtype

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.
            If None, a new object is created.

    Returns:
        cdl.obj.NewImageParam: Image parameters
    """
    if p is None:
        p = cdl.obj.NewImageParam()
    p.height = 2000 if p.height is None else p.height
    p.width = 2000 if p.width is None else p.width
    p.dtype = cdl.obj.ImageDatatypes.UINT16 if p.dtype is None else p.dtype
    return p


def add_gaussian_noise_to_image(
    image: cdl.obj.ImageObj, param: cdl.obj.NormalRandomParam
) -> None:
    """Add Gaussian noise to image

    Args:
        src (cdl.obj.ImageObj): Source image
        param (cdl.obj.NormalRandomParam): Parameters for the normal distribution
    """
    newparam = cdl.obj.new_image_param(
        height=image.data.shape[0],
        width=image.data.shape[1],
        dtype=cdl.obj.ImageDatatypes.from_dtype(image.data.dtype),
        itype=cdl.obj.ImageTypes.NORMALRANDOM,
    )
    noise = cdl.obj.create_image_from_param(newparam, param)
    image.data = image.data + noise.data


def create_2dstep_image(p: cdl.obj.NewImageParam | None = None) -> cdl.obj.ImageObj:
    """Creating 2D step image

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.

    Returns:
        cdl.obj.ImageObj: Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D step)" if p.title is None else p.title
    obj = cdl.obj.create_image_from_param(p)
    obj.data = create_2d_steps_data(p.height, p.height // 10, p.dtype.value)
    return obj


def create_peak2d_image(p: cdl.obj.NewImageParam | None = None) -> cdl.obj.ImageObj:
    """Creating 2D peak image

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.

    Returns:
        cdl.obj.ImageObj: Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D peaks)" if p.title is None else p.title
    obj = cdl.obj.create_image_from_param(p)
    param = PeakDataParam()
    if p.height is not None and p.width is not None:
        param.size = max(p.height, p.width)
    obj.data = get_peak2d_data(param)
    return obj


def create_sincos_image(p: cdl.obj.NewImageParam | None = None) -> cdl.obj.ImageObj:
    """Creating test image (sin(x)+cos(y))

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.

    Returns:
        cdl.obj.ImageObj: Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (sin(x)+cos(y))" if p.title is None else p.title
    dtype = p.dtype.value
    x, y = np.meshgrid(np.linspace(0, 10, p.width), np.linspace(0, 10, p.height))
    raw_data = 0.5 * (np.sin(x) + np.cos(y)) + 0.5
    dmin = np.iinfo(dtype).min * 0.95
    dmax = np.iinfo(dtype).max * 0.95
    obj = cdl.obj.create_image_from_param(p)
    obj.data = np.array(raw_data * (dmax - dmin) + dmin, dtype=dtype)
    return obj


def create_noisygauss_image(p: cdl.obj.NewImageParam | None = None) -> cdl.obj.ImageObj:
    """Create test image (2D noisy gaussian)

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.

    Returns:
        ImageObj: Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (noisy 2D Gaussian)" if p.title is None else p.title
    dtype = p.dtype.value
    size = p.width
    obj = cdl.obj.create_image_from_param(p)
    obj.data = create_2d_gaussian(size, dtype=dtype, x0=2.0, y0=3.0) + create_2d_random(
        size, dtype
    )
    obj.set_annotations_from_file(get_test_fnames("annotations.json")[0])
    return obj


def create_multigauss_image(p: cdl.obj.NewImageParam | None = None) -> cdl.obj.ImageObj:
    """Create test image (multiple 2D-gaussian peaks)

    Args:
        p (cdl.obj.NewImageParam, optional): Image parameters. Defaults to None.

    Returns:
        ImageObj: Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (multi-2D-gaussian)" if p.title is None else p.title
    dtype = p.dtype.value
    size = p.width
    obj = cdl.obj.create_image_from_param(p)
    obj.data = (
        create_2d_gaussian(size, dtype, x0=0.5, y0=3.0)
        + create_2d_gaussian(size, dtype, x0=-1.0, y0=-1.0, sigma=1.0)
        + create_2d_gaussian(size, dtype, x0=7.0, y0=8.0)
    )
    return obj


def create_annotated_image(title: str | None = None) -> cdl.obj.ImageObj:
    """Create test image with annotations

    Returns:
        ImageObj: Image object
    """
    data = create_2d_gaussian(600, np.uint16, x0=2.0, y0=3.0)
    title = "Test image (with metadata)" if title is None else title
    image = cdl.obj.create_image(title, data)
    image.set_annotations_from_file(get_test_fnames("annotations.json")[0])
    return image


def create_resultshapes() -> tuple[cdl.obj.ResultShape, ...]:
    """Create test result shapes (core.model.base.ResultShape test objects)

    Returns:
        tuple[ResultShape, ...]: Tuple of ResultShape objects
    """
    RShape, SType = cdl.obj.ResultShape, cdl.obj.ShapeTypes
    return (
        RShape(
            SType.CIRCLE,
            [[0, 100, 100, 400, 400], [0, 150, 150, 350, 350]],
            "circle",
        ),
        RShape(SType.RECTANGLE, [0, 300, 200, 700, 700], "rectangle"),
        RShape(SType.SEGMENT, [0, 50, 250, 400, 400], "segment"),
        RShape(SType.POINT, [[0, 500, 500], [0, 15, 400]], "point"),
    )
