# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Fourier computation module
--------------------------

This module implements Fourier transform operations and related spectral analysis tools
for images.

Main features include:
- Forward and inverse Fast Fourier Transform (FFT)
- Magnitude and phase spectrum calculation
- Power spectral density (PSD) computation

Fourier analysis is commonly used for frequency-domain filtering and periodicity
analysis in images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.computation.image` module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np

import sigima_.algorithms.image as alg
from sigima_.computation import computation_function
from sigima_.computation.base import FFTParam, SpectrumParam, dst_1_to_1
from sigima_.computation.image.base import Wrap1to1Func
from sigima_.config import _
from sigima_.obj.image import ImageObj


class ZeroPadding2DParam(gds.DataSet):
    """Zero padding parameters for 2D images"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.choice_callback(None, self.strategy)

    def choice_callback(self, item, value):  # pylint: disable=unused-argument
        """Callback to update padding values"""
        if self.__obj is None:
            return
        rows, cols = self.__obj.data.shape
        if value == "next_pow2":
            self.rows = 2 ** int(np.ceil(np.log2(rows))) - rows
            self.cols = 2 ** int(np.ceil(np.log2(cols))) - cols
        elif value == "multiple_of_64":
            self.rows = (64 - rows % 64) if rows % 64 != 0 else 0
            self.cols = (64 - cols % 64) if cols % 64 != 0 else 0

    strategies = ("next_pow2", "multiple_of_64", "custom")
    _prop = gds.GetAttrProp("strategy")
    strategy = gds.ChoiceItem(
        _("Padding strategy"), zip(strategies, strategies), default=strategies[-1]
    ).set_prop("display", store=_prop, callback=choice_callback)

    _func_prop = gds.FuncProp(_prop, lambda x: x == "custom")
    rows = gds.IntItem(_("Rows to add"), min=0).set_prop("display", active=_func_prop)
    cols = gds.IntItem(_("Columns to add"), min=0).set_prop(
        "display", active=_func_prop
    )

    positions = ("bottom-right", "center")
    position = gds.ChoiceItem(
        _("Padding position"), zip(positions, positions), default=positions[0]
    )


@computation_function()
def zero_padding(src: ImageObj, p: ZeroPadding2DParam) -> ImageObj:
    """
    Compute zero padding for an image using `sigima_.algorithms.image.zero_padding`.

    Args:
        src: source image object
        p: parameters

    Returns:
        New padded image object
    """
    if p.strategy == "custom":
        suffix = f"rows={p.rows}, cols={p.cols}"
    else:
        suffix = f"strategy={p.strategy}"
    suffix += f", position={p.position}"
    dst = dst_1_to_1(src, "zero_padding", suffix)
    result = alg.zero_padding(
        src.data,
        rows=p.rows,
        cols=p.cols,
        position=p.position,
    )
    dst.data = result
    return dst


@computation_function()
def fft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute FFT with :py:func:`sigima_.algorithms.image.fft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "fft")
    dst.data = alg.fft2d(src.data, shift=True if p is None else p.shift)
    dst.save_attr_to_metadata("xunit", "")
    dst.save_attr_to_metadata("yunit", "")
    dst.save_attr_to_metadata("zunit", "")
    dst.save_attr_to_metadata("xlabel", _("Frequency"))
    dst.save_attr_to_metadata("ylabel", _("Frequency"))
    return dst


@computation_function()
def ifft(src: ImageObj, p: FFTParam | None = None) -> ImageObj:
    """Compute inverse FFT with :py:func:`sigima_.algorithms.image.ifft2d`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "ifft")
    dst.data = alg.ifft2d(src.data, shift=True if p is None else p.shift)
    dst.restore_attr_from_metadata("xunit", "")
    dst.restore_attr_from_metadata("yunit", "")
    dst.restore_attr_from_metadata("zunit", "")
    dst.restore_attr_from_metadata("xlabel", "")
    dst.restore_attr_from_metadata("ylabel", "")
    return dst


@computation_function()
def magnitude_spectrum(src: ImageObj, p: SpectrumParam | None = None) -> ImageObj:
    """Compute magnitude spectrum
    with :py:func:`sigima_.algorithms.image.magnitude_spectrum`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "magnitude_spectrum")
    log_scale = p is not None and p.log
    dst.data = alg.magnitude_spectrum(src.data, log_scale=log_scale)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


@computation_function()
def phase_spectrum(src: ImageObj) -> ImageObj:
    """Compute phase spectrum
    with :py:func:`sigima_.algorithms.image.phase_spectrum`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    dst = Wrap1to1Func(alg.phase_spectrum)(src)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst


@computation_function()
def psd(src: ImageObj, p: SpectrumParam | None = None) -> ImageObj:
    """Compute power spectral density
    with :py:func:`sigima_.algorithms.image.psd`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "psd")
    log_scale = p is not None and p.log
    dst.data = alg.psd(src.data, log_scale=log_scale)
    dst.xunit = dst.yunit = dst.zunit = ""
    dst.xlabel = dst.ylabel = _("Frequency")
    return dst
