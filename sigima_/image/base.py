# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base computation module
-----------------------

This module provides core classes and utility functions that serve as building blocks
for the other computation modules.

Main features include:
- Generic helper functions used across image processing modules
- Core wrappers and infrastructure for computation functions

Intended primarily for internal use, these tools support consistent API design
and code reuse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from sigima_.algorithms.datatypes import is_integer_dtype
from sigima_.base import dst_1_to_1, new_signal_result
from sigima_.obj.base import ResultShape
from sigima_.obj.image import ImageObj
from sigima_.obj.signal import SignalObj


def restore_data_outside_roi(dst: ImageObj, src: ImageObj) -> None:
    """Restore data outside the Region Of Interest (ROI) of the input image
    after a computation, only if the input image has a ROI,
    and if the output image has the same ROI as the input image,
    and if the data types are compatible,
    and if the shapes are the same.
    Otherwise, do nothing.

    Args:
        dst: output image object
        src: input image object
    """
    if src.maskdata is not None and dst.maskdata is not None:
        if (
            np.array_equal(src.maskdata, dst.maskdata)
            and (
                dst.data.dtype == src.data.dtype or not is_integer_dtype(dst.data.dtype)
            )
            and dst.data.shape == src.data.shape
        ):
            dst.data[src.maskdata] = src.data[src.maskdata]


class Wrap1to1Func:
    """Wrap a 1 array → 1 array function to produce a 1 image → 1 image function,
    which can be used inside DataLab's infrastructure to perform computations with
    :class:`cdl.gui.processor.image.ImageProcessor`.

    This wrapping mechanism using a class is necessary for the resulted function to be
    pickable by the ``multiprocessing`` module.

    The instance of this wrapper is callable and returns a :class:`cdl.obj.ImageObj`
    object.

    Example:

        >>> import numpy as np
        >>> from sigima_.image import Wrap1to1Func
        >>> import cdl.obj
        >>> def add_noise(data):
        ...     return data + np.random.random(data.shape)
        >>> compute_add_noise = Wrap1to1Func(add_noise)
        >>> data= np.ones((100, 100))
        >>> ima0 = cdl.obj.create_image("Example", data)
        >>> ima1 = compute_add_noise(ima0)

    Args:
        func: 1 array → 1 array function
        *args: Additional positional arguments to pass to the function
        **kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__call__.__func__.__doc__ = self.func.__doc__

    def __call__(self, src: ImageObj) -> ImageObj:
        """Compute the function on the input image and return the result image

        Args:
            src: input image object

        Returns:
            Output image object
        """
        suffix = ", ".join(
            [str(arg) for arg in self.args]
            + [f"{k}={v}" for k, v in self.kwargs.items() if v is not None]
        )
        dst = dst_1_to_1(src, self.func.__name__, suffix)
        dst.data = self.func(src.data, *self.args, **self.kwargs)
        restore_data_outside_roi(dst, src)
        return dst


def dst_1_to_1_signal(src: ImageObj, name: str, suffix: str | None = None) -> SignalObj:
    """Create a result signal object, as returned by the callback function of the
    :func:`cdl.gui.processor.base.BaseProcessor.compute_1_to_1` method

    Args:
        src: input image object
        name: name of the processing function

    Returns:
        Output signal object
    """
    return new_signal_result(
        src, name, suffix, (src.xunit, src.zunit), (src.xlabel, src.zlabel)
    )


def calc_resultshape(
    title: str,
    shape: Literal[
        "rectangle", "circle", "ellipse", "segment", "marker", "point", "polygon"
    ],
    obj: ImageObj,
    func: Callable,
    *args: Any,
    add_label: bool = False,
) -> ResultShape | None:
    """Calculate result shape by executing a computation function on an image object,
    taking into account the image origin (x0, y0), scale (dx, dy) and ROIs.

    Args:
        title: result title
        shape: result shape kind
        obj: input image object
        func: computation function
        *args: computation function arguments
        add_label: if True, add a label item (and the geometrical shape) to plot
         (default to False)

    Returns:
        Result shape object or None if no result is found

    .. warning::

        The computation function must take either a single argument (the data) or
        multiple arguments (the data followed by the computation parameters).

        Moreover, the computation function must return a single value or a NumPy array
        containing the result of the computation. This array contains the coordinates
        of points, polygons, circles or ellipses in the form [[x, y], ...], or
        [[x0, y0, x1, y1, ...], ...], or [[x0, y0, r], ...], or
        [[x0, y0, a, b, theta], ...].
    """
    res: list[np.ndarray] = []
    num_cols: list[int] = []
    for i_roi in obj.iterate_roi_indices():
        data_roi = obj.get_data(i_roi)
        if args is None:
            coords: np.ndarray = func(data_roi)
        else:
            coords: np.ndarray = func(data_roi, *args)

        # This is a very long condition, but it's still quite readable, so we keep it
        # as is and disable the pylint warning.
        #
        # pylint: disable=too-many-boolean-expressions
        if not isinstance(coords, np.ndarray) or (
            (
                coords.ndim != 2
                or coords.shape[1] < 2
                or (coords.shape[1] > 5 and coords.shape[1] % 2 != 0)
            )
            and coords.size > 0
        ):
            raise ValueError(
                f"Computation function {func.__name__} must return a NumPy array "
                f"containing coordinates of points, polygons, circles or ellipses "
                f"(in the form [[x, y], ...], or [[x0, y0, x1, y1, ...], ...], or "
                f"[[x0, y0, r], ...], or [[x0, y0, a, b, theta], ...]), or an empty "
                f"array."
            )

        if coords.size:
            coords = np.array(coords, dtype=float)
            if coords.shape[1] % 2 == 0:
                # Coordinates are in the form [x0, y0, x1, y1, ...]
                colx, coly = slice(None, None, 2), slice(1, None, 2)
            else:
                # Circle [x0, y0, r] or ellipse coordinates [x0, y0, a, b, theta]
                colx, coly = 0, 1
            coords[:, colx] = obj.dx * coords[:, colx] + obj.x0
            coords[:, coly] = obj.dy * coords[:, coly] + obj.y0
            if obj.roi is not None:
                x0, y0, _x1, _y1 = obj.roi.get_single_roi(i_roi).get_bounding_box(obj)
                coords[:, colx] += x0 - obj.x0
                coords[:, coly] += y0 - obj.y0
            idx = np.ones((coords.shape[0], 1)) * (0 if i_roi is None else i_roi)
            coords = np.hstack([idx, coords])
            res.append(coords)
            num_cols.append(coords.shape[1])
    if res:
        if len(set(num_cols)) != 1:
            # This happens when the number of columns is not the same for all ROIs.
            # As of now, this happens only for polygon contours.
            # We need to pad the arrays with NaNs.
            max_cols = max(num_cols)
            num_rows = sum(coords.shape[0] for coords in res)
            array = np.full((num_rows, max_cols), np.nan)
            row = 0
            for coords in res:
                array[row : row + coords.shape[0], : coords.shape[1]] = coords
                row += coords.shape[0]
        else:
            array = np.vstack(res)
        return ResultShape(title, array, shape, add_label=add_label)
    return None
