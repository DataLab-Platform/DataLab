# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test data functions

Functions creating test data: curves, images, ...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from __future__ import annotations

from typing import Generator

import guidata.dataset as gds
import numpy as np

from sigima_.config import _
from sigima_.io import read_image, read_signal
from sigima_.obj import (
    GaussLorentzVoigtParam,
    ImageDatatypes,
    ImageObj,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalRandomParam,
    PeriodicParam,
    ResultProperties,
    ResultShape,
    SignalObj,
    SignalTypes,
    create_image,
    create_image_from_param,
    create_signal_from_param,
)
from sigima_.tests.helpers import get_test_fnames


def get_test_signal(filename: str) -> SignalObj:
    """Return test signal

    Args:
        filename: Filename

    Returns:
        Signal object
    """
    return read_signal(get_test_fnames(filename)[0])


def get_test_image(filename: str) -> ImageObj:
    """Return test image

    Args:
        filename: Filename

    Returns:
        Image object
    """
    return read_image(get_test_fnames(filename)[0])


def create_paracetamol_signal(
    size: int | None = None, title: str | None = None
) -> SignalObj:
    """Create test signal (Paracetamol molecule spectrum)

    Args:
        size: Size of the data. Defaults to None.
        title: Title of the signal. Defaults to None.

    Returns:
        Signal object
    """
    obj = read_signal(get_test_fnames("paracetamol.txt")[0])
    if title is not None:
        obj.title = title
    if size is not None:
        x0, y0 = obj.xydata
        x1 = np.linspace(x0[0], x0[-1], size)
        y1 = np.interp(x1, x0, y0)
        obj.set_xydata(x1, y1)
    return obj


class GaussianNoiseParam(gds.DataSet):
    """Gaussian noise parameters"""

    mu = gds.FloatItem(
        _("Mean"),
        default=0.0,
        min=-100.0,
        max=100.0,
        help=_("Mean of the Gaussian distribution"),
    )
    sigma = gds.FloatItem(
        _("Standard deviation"),
        default=0.1,
        min=0.0,
        max=100.0,
        help=_("Standard deviation of the Gaussian distribution"),
    )
    seed = gds.IntItem(
        _("Seed"),
        default=1,
        min=0,
        max=1000000,
        help=_("Seed for random number generator"),
    )


def add_gaussian_noise_to_signal(
    signal: SignalObj, p: GaussianNoiseParam | None = None
) -> None:
    """Add Gaussian (Normal-law) random noise to data

    Args:
        signal: Signal object
        p: Gaussian noise parameters.
    """
    if p is None:
        p = GaussianNoiseParam()
    rng = np.random.default_rng(p.seed)
    signal.data += rng.normal(p.mu, p.sigma, size=signal.data.shape)
    signal.title = f"GaussNoise({signal.title}, µ={p.mu}, σ={p.sigma})"


def create_noisy_signal(
    noiseparam: GaussianNoiseParam | None = None,
    newparam: NewSignalParam | None = None,
    addparam: GaussLorentzVoigtParam | None = None,
    title: str | None = None,
    noised: bool | None = None,
) -> SignalObj:
    """Create curve data, optionally noised

    Args:
        noiseparam: Noise parameters. Default: None: No noise
        newparam: New signal parameters.
         Default: Gaussian, size=500, xmin=-10, xmax=10
        addparam: Additional parameters.
         Default: a=1.0, sigma=1.0, mu=0.0, ymin=0.0
        title: Title of the signal. Default: None
         If not None, overrides the title in newparam
        noised: If True, add noise to the signal.
         Default: None (use noiseparam)
         If True, eventually creates a new noiseparam if None

    Returns:
        Signal object
    """
    if newparam is None:
        newparam = NewSignalParam()
        newparam.stype = SignalTypes.GAUSS
    if title is not None:
        newparam.title = title
    newparam.title = "Test signal (noisy)" if newparam.title is None else newparam.title
    if addparam is None:
        addparam = GaussLorentzVoigtParam()
    if noised is not None and noised and noiseparam is None:
        noiseparam = GaussianNoiseParam()
        noiseparam.sigma = 5.0
    sig = create_signal_from_param(newparam, addparam)
    if noiseparam is not None:
        add_gaussian_noise_to_signal(sig, noiseparam)
    return sig


def create_periodic_signal(
    shape: SignalTypes,
    freq: float = 50.0,
    size: int = 10000,
    xmin: float = -10.0,
    xmax: float = 10.0,
    a: float = 1.0,
) -> SignalObj:
    """Create a periodic signal

    Args:
        shape: Shape of the signal
        freq: Frequency of the signal. Defaults to 50.0.
        size: Size of the signal. Defaults to 10000.
        xmin: Minimum value of the signal. Defaults to None.
        xmax: Maximum value of the signal. Defaults to None.
        a: Amplitude of the signal. Defaults to 1.0.

    Returns:
        Signal object
    """
    newparam = NewSignalParam.create(stype=shape, size=size, xmin=xmin, xmax=xmax)
    addparam = PeriodicParam.create(freq=freq, a=a)
    return create_signal_from_param(newparam, addparam)


def create_2d_steps_data(size: int, width: int, dtype: np.dtype) -> np.ndarray:
    """Creating 2D steps data for testing purpose

    Args:
        size: Size of the data
        width: Width of the steps
        dtype: Data type

    Returns:
        2D data
    """
    data = np.zeros((size, size), dtype=dtype)
    value = 1
    for col in range(0, size - width + 1, width):
        data[:, col : col + width] = np.array(value).astype(dtype)
        value *= 10
    data2 = np.zeros_like(data)
    value = 1
    for row in range(0, size - width + 1, width):
        data2[row : row + width, :] = np.array(value).astype(dtype)
        value *= 10
    data += data2
    return data


def create_2d_random(
    size: int, dtype: np.dtype, level: float = 0.1, seed: int = 1
) -> np.ndarray:
    """Creating 2D Uniform-law random image

    Args:
        size: Size of the data
        dtype: Data type
        level: Level of the random noise. Defaults to 0.1.
        seed: Seed for random number generator. Defaults to 1.

    Returns:
        2D data
    """
    rng = np.random.default_rng(seed)
    amp = (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0) * level
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
        size: Size of the data
        dtype: Data type
        x0: x0. Defaults to 0.
        y0: y0. Defaults to 0.
        mu: mu. Defaults to 0.0.
        sigma: sigma. Defaults to 2.0.
        amp: Amplitude. Defaults to None.

    Returns:
        2D data
    """
    xydata = np.linspace(-10, 10, size)
    x, y = np.meshgrid(xydata, xydata)
    if amp is None:
        try:
            amp = np.iinfo(dtype).max * 0.5
        except ValueError:
            # dtype is not integer
            amp = 1.0
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
        List of NumPy arrays
    """
    znoise = create_2d_random(2000, np.uint16)
    zgauss = create_2d_gaussian(2000, np.uint16, x0=2.0, y0=-3.0)
    return [zgauss + znoise] + [
        read_image(fname).data for fname in get_test_fnames("*.scor-data")
    ]


class PeakDataParam(gds.DataSet):
    """Peak data test image parameters"""

    size = gds.IntItem(_("Size"), default=2000, min=1)
    n_points = gds.IntItem(_("Number"), default=4, min=1, help=_("Number of points"))
    sigma_gauss2d = gds.FloatItem(
        "σ<sub>Gauss2D</sub>", default=0.06, help=_("Sigma of the 2D Gaussian")
    )
    amp_gauss2d = gds.IntItem(
        "A<sub>Gauss2D</sub>", default=1900, help=_("Amplitude of the 2D Gaussian")
    )
    mu_noise = gds.IntItem(
        "μ<sub>noise</sub>", default=845, help=_("Mean of the Gaussian distribution")
    )
    sigma_noise = gds.IntItem(
        "σ<sub>noise</sub>",
        default=25,
        help=_("Standard deviation of the Gaussian distribution"),
    )
    dx0 = gds.FloatItem("dx0", default=0.0)
    dy0 = gds.FloatItem("dy0", default=0.0)
    att = gds.FloatItem(_("Attenuation"), default=1.0)


def get_peak2d_data(
    p: PeakDataParam | None = None, seed: int | None = None, multi: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return a list of NumPy arrays containing images which are relevant for
    testing 2D peak detection or similar image processing features

    Args:
        p: Peak data test image parameters. Defaults to None.
        seed: Seed for random number generator. Defaults to None.
        multi: If True, multiple peaks are generated. Defaults to False.

    Returns:
        A tuple containing the image data and coordinates of the peaks.
    """
    if p is None:
        p = PeakDataParam()
    delta = 0.1
    rng = np.random.default_rng(seed)
    coords_phys = (rng.random((p.n_points, 2)) - 0.5) * 10 * (1 - delta)
    data = rng.normal(p.mu_noise, p.sigma_noise, size=(p.size, p.size))
    multi_nb = 2 if multi else 1
    for x0, y0 in coords_phys:
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
    # Convert coordinates to indices
    coords = []
    for x0, y0 in coords_phys:
        x = int((x0 + 10) / 20 * p.size)
        y = int((y0 + 10) / 20 * p.size)
        if 0 <= x < p.size and 0 <= y < p.size:
            coords.append((x, y))
    return data, np.array(coords)


def __set_default_size_dtype(
    p: NewImageParam | None = None,
) -> NewImageParam:
    """Set default shape and dtype

    Args:
        p: Image parameters. Defaults to None. If None, a new object is created.

    Returns:
        Image parameters
    """
    if p is None:
        p = NewImageParam()
    p.height = 2000 if p.height is None else p.height
    p.width = 2000 if p.width is None else p.width
    p.dtype = ImageDatatypes.UINT16 if p.dtype is None else p.dtype
    return p


def add_gaussian_noise_to_image(image: ImageObj, param: NormalRandomParam) -> None:
    """Add Gaussian noise to image

    Args:
        src: Source image
        param: Parameters for the normal distribution
    """
    newparam = NewImageParam.create(
        height=image.data.shape[0],
        width=image.data.shape[1],
        dtype=ImageDatatypes.from_dtype(image.data.dtype),
        itype=ImageTypes.NORMALRANDOM,
    )
    noise = create_image_from_param(newparam, param)
    image.data = image.data + noise.data


def create_checkerboard(p: NewImageParam | None = None, num_checkers=8) -> ImageObj:
    """Generate a checkerboard pattern

    Args:
        p: Image parameters. Defaults to None.
        num_checkers: Number of checkers. Defaults to 8.
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (checkerboard)" if p.title is None else p.title
    obj = create_image_from_param(p)
    re = np.r_[num_checkers * [0, 1]]  # one row of the checkerboard
    board = np.vstack(num_checkers * (re, re ^ 1))  # build the checkerboard
    board = np.kron(
        board, np.ones((p.height // num_checkers, p.height // num_checkers))
    )  # scale up the board
    obj.data = board
    return obj


def create_2dstep_image(
    p: NewImageParam | None = None, extra_param: gds.DataSet | None = None
) -> ImageObj:
    """Creating 2D step image

    Args:
        p: Image parameters. Defaults to None.
        extra_param: Extra parameters for the image creation

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D step)" if p.title is None else p.title
    obj = create_image_from_param(p, extra_param=extra_param)
    obj.data = create_2d_steps_data(p.height, p.height // 10, p.dtype.value)
    return obj


class RingParam(gds.DataSet):
    """Parameters for creating a ring image"""

    size = gds.IntItem(_("Size"), default=1000)
    ring_x0 = gds.IntItem(_("X<sub>center</sub>"), default=500)
    ring_y0 = gds.IntItem(_("Y<sub>center</sub>"), default=500)
    ring_width = gds.IntItem(_("Width"), default=10)
    ring_radius = gds.IntItem(_("Radius"), default=250)
    ring_intensity = gds.IntItem(_("Intensity"), default=1000)


def create_ring_data(
    size: int, x0: int, y0: int, width: int, radius: int, intensity: int
) -> np.ndarray:
    """Create 2D ring data

    Args:
        size: Size of the image
        x0: Center x coordinate
        y0: Center y coordinate
        width: Width of the ring
        radius: Radius of the ring
        intensity: Intensity of the ring

    Returns:
        2D data
    """
    data = np.zeros((size, size), dtype=np.uint16)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if (x - x0) ** 2 + (y - y0) ** 2 >= (radius - width) ** 2 and (
                x - x0
            ) ** 2 + (y - y0) ** 2 <= (radius + width) ** 2:
                data[x, y] = intensity
    return data


def create_ring_image(p: RingParam | None = None) -> ImageObj:
    """Creating 2D ring image

    Args:
        p: Ring image parameters. Defaults to None.

    Returns:
        Image object
    """
    if p is None:
        p = RingParam()
    obj = create_image(
        f"Ring(size={p.size},x0={p.ring_x0},y0={p.ring_y0},width={p.ring_width},"
        f"radius={p.ring_radius},intensity={p.ring_intensity})"
    )
    obj.data = create_ring_data(
        p.size,
        p.ring_x0,
        p.ring_y0,
        p.ring_width,
        p.ring_radius,
        p.ring_intensity,
    )
    return obj


def create_peak2d_image(
    p: NewImageParam | None = None, extra_param: gds.DataSet | None = None
) -> ImageObj:
    """Creating 2D peak image

    Args:
        p: Image parameters. Defaults to None
        extra_param: Extra parameters for the image creation

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (2D peaks)" if p.title is None else p.title
    obj = create_image_from_param(p, extra_param=extra_param)
    param = PeakDataParam()
    if p.height is not None and p.width is not None:
        param.size = max(p.height, p.width)
    obj.data, coords = get_peak2d_data(param)
    obj.metadata["peak_coords"] = coords
    return obj


def create_sincos_image(
    p: NewImageParam | None = None, extra_param: gds.DataSet | None = None
) -> ImageObj:
    """Creating test image (sin(x)+cos(y))

    Args:
        p: Image parameters. Defaults to None
        extra_param: Extra parameters

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (sin(x)+cos(y))" if p.title is None else p.title
    dtype = p.dtype.value
    x, y = np.meshgrid(np.linspace(0, 10, p.width), np.linspace(0, 10, p.height))
    raw_data = 0.5 * (np.sin(x) + np.cos(y)) + 0.5
    obj = create_image_from_param(p, extra_param=extra_param)
    if np.issubdtype(dtype, np.floating):
        obj.data = raw_data
        return obj
    dmin = np.iinfo(dtype).min * 0.95
    dmax = np.iinfo(dtype).max * 0.95
    obj.data = np.array(raw_data * (dmax - dmin) + dmin, dtype=dtype)
    return obj


def add_annotations_from_file(obj: SignalObj | ImageObj, filename: str) -> None:
    """Add annotations from a file to a Signal or Image object

    Args:
        obj: Signal or Image object to which annotations will be added
        filename: Filename containing annotations
    """
    with open(filename, "r", encoding="utf-8") as file:
        json_str = file.read()
    if obj.annotations:
        json_str = obj.annotations[:-1] + "," + json_str[1:]
    obj.annotations = json_str


def create_noisygauss_image(
    p: NewImageParam | None = None,
    center: tuple[float, float] | None = None,
    level: float = 0.1,
    add_annotations: bool = False,
    extra_param: gds.DataSet | None = None,
) -> ImageObj:
    """Create test image (2D noisy gaussian)

    Args:
        p: Image parameters. Defaults to None.
        center: Center of the gaussian. Defaults to None.
        level: Level of the random noise. Defaults to 0.1.
        add_annotations: If True, add annotations. Defaults to False.
        extra_param: Extra parameters for the image creation. Defaults to None.

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (noisy 2D Gaussian)" if p.title is None else p.title
    dtype = p.dtype.value
    size = p.width
    obj = create_image_from_param(p, extra_param=extra_param)
    if center is None:
        # Default center
        x0, y0 = 2.0, 3.0
    else:
        x0, y0 = center
    obj.data = create_2d_gaussian(size, dtype=dtype, x0=x0, y0=y0)
    if level:
        obj.data += create_2d_random(size, dtype, level)
    if add_annotations:
        add_annotations_from_file(obj, get_test_fnames("annotations.json")[0])
    return obj


def create_multigauss_image(
    p: NewImageParam | None = None, extra_param: gds.DataSet | None = None
) -> ImageObj:
    """Create test image (multiple 2D-gaussian peaks)

    Args:
        p: Image parameters. Defaults to None.
        extra_param: Extra parameters for the image creation. Defaults to None.

    Returns:
        Image object
    """
    p = __set_default_size_dtype(p)
    p.title = "Test image (multi-2D-gaussian)" if p.title is None else p.title
    dtype = p.dtype.value
    size = p.width
    obj = create_image_from_param(p, extra_param=extra_param)
    obj.data = (
        create_2d_gaussian(size, dtype, x0=0.5, y0=3.0)
        + create_2d_gaussian(size, dtype, x0=-1.0, y0=-1.0, sigma=1.0)
        + create_2d_gaussian(size, dtype, x0=7.0, y0=8.0)
    )
    return obj


def create_annotated_image(title: str | None = None) -> ImageObj:
    """Create test image with annotations

    Returns:
        Image object
    """
    data = create_2d_gaussian(600, np.uint16, x0=2.0, y0=3.0)
    title = "Test image (with metadata)" if title is None else title
    image = create_image(title, data)
    add_annotations_from_file(image, get_test_fnames("annotations.json")[0])
    return image


def create_resultshapes() -> Generator[ResultShape, None, None]:
    """Create test result shapes (core.base.ResultShape test objects)

    Yields:
        ResultShape object
    """
    for shape, data in (
        ("circle", [[0, 250, 250, 200], [0, 250, 250, 140]]),
        ("rectangle", [0, 300, 200, 700, 700]),
        ("segment", [0, 50, 250, 400, 400]),
        ("point", [[0, 500, 500], [0, 15, 400]]),
        (
            "polygon",
            [0, 100, 100, 150, 100, 150, 150, 200, 100, 250, 50],
        ),
    ):
        yield ResultShape(shape, data, shape, add_label=shape == "segment")


def create_resultproperties() -> Generator[ResultProperties, None, None]:
    """Create test result properties (core.base.ResultProperties test object)

    Returns:
        ResultProperties object
    """
    for title, data, labels in (
        ("TestProperties1", [0, 2.5, -30909, 1.0, 0.0], ["A", "B", "C", "D"]),
        (
            "TestProperties2",
            [[0, 1.232325, -9, 0, 10], [0, 250, -3, 12.0, 530.0]],
            ["P1", "P2", "P3", "P4"],
        ),
    ):
        yield ResultProperties(title, data, labels)
