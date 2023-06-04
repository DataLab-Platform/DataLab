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

import numpy as np

from cdl.algorithms import fit
from cdl.obj import (
    ImageObj,
    ResultShape,
    ShapeTypes,
    SignalObj,
    create_image,
    create_signal,
    read_image,
    read_signal,
)
from cdl.utils.tests import get_test_fnames


def add_gaussian_noise(
    data: np.ndarray, mu: float = 0.0, sigma: float = 0.1, seed: int = 1
) -> None:
    """Add Gaussian (Normal-law) random noise to data

    Args:
        data (np.ndarray): Data to be noised
        mu (float, optional): Mean of the Gaussian distribution. Defaults to 0.0.
        sigma (float, optional): Standard deviation of the Gaussian distribution.
            Defaults to 0.1.
        seed (int, optional): Seed for random number generator. Defaults to 1.
    """
    rng = np.random.default_rng(seed)
    data += rng.normal(mu, sigma, size=data.shape)


def create_1d_gaussian(
    size,
    amp: float = 50.0,
    sigma: float = 2.0,
    x0: float = 0.0,
    y0: float = 0.0,
    noise_mu: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create Gaussian curve data (optionally noised, if noise_sigma != 0.0)

    Args:
        size (int): Number of points
        amp (float, optional): Amplitude. Defaults to 50.0.
        sigma (float, optional): Standard deviation. Defaults to 2.0.
        x0 (float, optional): x0. Defaults to 0.0.
        y0 (float, optional): y0. Defaults to 0.0.
        noise_mu (float, optional): Mean of the Gaussian distribution. Defaults to 0.0.
        noise_sigma (float, optional): Standard deviation of the Gaussian distribution.
            Defaults to 0.0.
        seed (int, optional): Seed for random number generator. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: x, y
    """
    x = np.linspace(-10, 10, size)
    y = fit.GaussianModel.func(x, amp, sigma, x0, y0)
    if noise_sigma:
        add_gaussian_noise(y, mu=noise_mu, sigma=noise_sigma, seed=seed)
    return x, y


def create_test_2d_data(size: int, dtype: np.dtype) -> np.ndarray:
    """Creating 2D test data

    Args:
        size (int): Size of the data
        dtype (np.dtype): Data type

    Returns:
        np.ndarray: 2D data
    """
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    raw_data = 0.5 * (np.sin(x) + np.cos(y)) + 0.5
    dmin = np.iinfo(dtype).min * 0.95
    dmax = np.iinfo(dtype).max * 0.95
    return np.array(raw_data * (dmax - dmin) + dmin, dtype=dtype)


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
        read_image(fname).data for fname in get_test_fnames("*.scor-data")
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


def __get_default_data_size(size: int | None = None) -> int:
    """Return default data size

    Args:
        size (int, optional): Size of the data. Defaults to None.

    Returns:
        int: Size of the data
    """
    return 500 if size is None else size


def create_test_signal1(size: int | None = None, title: str | None = None) -> SignalObj:
    """Create test signal (Paracetamol molecule spectrum)

    Args:
        size (int, optional): Size of the data. Defaults to None.
        title (str, optional): Title of the signal. Defaults to None.

    Returns:
        SignalObj: Signal object
    """
    obj = read_signal(get_test_fnames("paracetamol.txt")[0])
    if title is not None:
        obj.title = title
    size = __get_default_data_size(size)
    if size is not None:
        x0, y0 = obj.xydata
        x1 = np.linspace(x0[0], x0[-1], size)
        y1 = np.interp(x1, x0, y0)
        obj.set_xydata(x1, y1)
    return obj


def create_test_signal2(
    size: int | None = None, title: str | None = None, noised: bool = False
) -> SignalObj:
    """Create test signal (Gaussian curve, optionally noised)

    Args:
        size (int, optional): Size of the data. Defaults to None.
        title (str, optional): Title of the signal. Defaults to None.
        noised (bool, optional): If True, the signal is noised. Defaults to False.

    Returns:
        SignalObj: Signal object
    """
    size = __get_default_data_size(size)
    default_title = "Gaussian curve" + (" with noise" if noised else "")
    obj = create_signal(default_title if title is None else title)
    x, y = create_1d_gaussian(size=size, noise_sigma=5.0 if noised else 0.0)
    obj.set_xydata(x, y)
    return obj


def __get_default_size_dtype(
    size: int | None = None, dtype: np.dtype | None = None
) -> tuple[int, np.dtype]:
    """Return default shape and dtype

    Args:
        size (int, optional): Size of the data. Defaults to None.
        dtype (np.dtype, optional): Data type. Defaults to None.

    Returns:
        tuple[int, np.dtype]: Size and data type
    """
    size = 2000 if size is None else size
    dtype = np.uint16 if dtype is None else dtype
    return size, dtype


def create_test_image1(
    size: int | None = None, dtype: np.dtype | None = None, title: str | None = None
) -> ImageObj:
    """Create test image (sin(x)+cos(y))

    Args:
        size (int, optional): Size of the data. Defaults to None.
        dtype (np.dtype, optional): Data type. Defaults to None.
        title (str, optional): Title of the image. Defaults to None.

    Returns:
        ImageObj: Image object
    """
    size, dtype = __get_default_size_dtype(size, dtype)
    title = "sin(x)+cos(y)" if title is None else title
    return create_image(title, create_test_2d_data(size, dtype=dtype))


def create_test_image2(
    size: int | None = None,
    dtype: np.dtype | None = None,
    with_annotations: bool = True,
    title: str | None = None,
) -> ImageObj:
    """Create test image (2D noisy gaussian)

    Args:
        size (int, optional): Size of the data. Defaults to None.
        dtype (np.dtype, optional): Data type. Defaults to None.
        with_annotations (bool, optional): If True, the image is annotated.
            Defaults to True.
        title (str, optional): Title of the image. Defaults to None.

    Returns:
        ImageObj: Image object
    """
    size, dtype = __get_default_size_dtype(size, dtype)
    title = "2D Gaussian" if title is None else title
    zgauss = create_2d_gaussian(size, dtype=dtype, x0=2.0, y0=3.0)
    znoise = create_2d_random(size, dtype)
    data = zgauss + znoise
    image = create_image(title, data)
    if with_annotations:
        with open(get_test_fnames("annotations.json")[0], mode="rb") as fdesc:
            image.annotations = fdesc.read().decode()
    return image


def create_test_image3(
    size: int | None = None, dtype: np.dtype | None = None, title: str | None = None
) -> ImageObj:
    """Create test image (multiple 2D-gaussian peaks)

    Args:
        size (int, optional): Size of the data. Defaults to None.
        dtype (np.dtype, optional): Data type. Defaults to None.
        title (str, optional): Title of the image. Defaults to None.

    Returns:
        ImageObj: Image object
    """
    size, dtype = __get_default_size_dtype(size, dtype)
    data = (
        create_2d_gaussian(size, dtype, x0=0.5, y0=3.0)
        + create_2d_gaussian(size, dtype, x0=-1.0, y0=-1.0, sigma=1.0)
        + create_2d_gaussian(size, dtype, x0=7.0, y0=8.0)
    )
    return create_image("Multiple 2D-gaussian peaks" if title is None else title, data)


def create_resultshapes() -> tuple[ResultShape, ...]:
    """Create test result shapes (core.model.base.ResultShape test objects)

    Returns:
        tuple[ResultShape, ...]: Tuple of ResultShape objects
    """
    return (
        ResultShape(
            ShapeTypes.CIRCLE,
            [[0, 100, 100, 400, 400], [0, 150, 150, 350, 350]],
            "circle",
        ),
        ResultShape(ShapeTypes.RECTANGLE, [0, 300, 200, 700, 700], "rectangle"),
        ResultShape(ShapeTypes.SEGMENT, [0, 50, 250, 400, 400], "segment"),
        ResultShape(ShapeTypes.POINT, [[0, 500, 500], [0, 15, 400]], "point"),
    )


def create_image_with_annotations(title: str | None = None) -> ImageObj:
    """Create test image with annotations

    Returns:
        ImageObj: Image object
    """
    data = create_2d_gaussian(600, np.uint16, x0=2.0, y0=3.0)
    title = "Test image with metadata" if title is None else title
    image = create_image(title, data)
    image.set_annotations_from_file(get_test_fnames("annotations.json")[0])
    return image


def add_annotations_to_image(image: ImageObj):
    """Add annotations to image

    Args:
        image (ImageObj): Image object
    """
    image.add_annotations_from_file(get_test_fnames("annotations.json")[0])
