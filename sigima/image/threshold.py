# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Threshold computation module
----------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.image` module.

from __future__ import annotations

import guidata.dataset as gds
import skimage.util
from skimage import filters

from cdl.config import _
from cdl.obj import ImageObj
from sigima import computation_function
from sigima.base import dst_1_to_1
from sigima.image.base import restore_data_outside_roi


class ThresholdParam(gds.DataSet):
    """Histogram threshold parameters"""

    methods = (
        ("manual", _("Manual")),
        ("isodata", "ISODATA"),
        ("li", "Li"),
        ("mean", _("Mean")),
        ("minimum", _("Minimum")),
        ("otsu", "Otsu"),
        ("triangle", _("Triangle")),
        ("yen", "Yen"),
    )

    _method_prop = gds.GetAttrProp("method")
    method = gds.ChoiceItem(_("Threshold method"), methods, default="manual").set_prop(
        "display", store=_method_prop
    )
    bins = gds.IntItem(_("Number of bins"), default=256, min=1).set_prop(
        "display",
        active=gds.FuncProp(_method_prop, lambda x: x not in ("li", "mean", "manual")),
    )
    value = gds.FloatItem(_("Threshold value"), default=0.0).set_prop(
        "display", active=gds.FuncProp(_method_prop, lambda x: x == "manual")
    )
    operation = gds.ChoiceItem(
        _("Operation"),
        ((">", _("Greater than")), ("<", _("Less than"))),
        default=">",
    )


@computation_function
def threshold(src: ImageObj, p: ThresholdParam) -> ImageObj:
    """Compute the threshold, using one of the available algorithms:

    - Manual: a fixed threshold value
    - ISODATA: :py:func:`skimage.filters.threshold_isodata`
    - Li: :py:func:`skimage.filters.threshold_li`
    - Mean: :py:func:`skimage.filters.threshold_mean`
    - Minimum: :py:func:`skimage.filters.threshold_minimum`
    - Otsu: :py:func:`skimage.filters.threshold_otsu`
    - Triangle: :py:func:`skimage.filters.threshold_triangle`
    - Yen: :py:func:`skimage.filters.threshold_yen`

    Args:
        src: input image object
        p: parameters

    Returns:
        Output image object
    """
    if p.method == "manual":
        suffix = f"value={p.value}"
        value = p.value
    else:
        suffix = f"method={p.method}"
        if p.method not in ("li", "mean"):
            suffix += f", nbins={p.bins}"
        func = getattr(filters, f"threshold_{p.method}")
        args = [] if p.method in ("li", "mean") else [p.bins]
        value = func(src.data, *args)
    dst = dst_1_to_1(src, "threshold", suffix)
    data = src.data > value if p.operation == ">" else src.data < value
    dst.data = skimage.util.img_as_ubyte(data)
    dst.zscalemin, dst.zscalemax = 0, 255  # LUT range
    dst.metadata["colormap"] = "gray"
    restore_data_outside_roi(dst, src)
    return dst


@computation_function
def threshold_isodata(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Isodata algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_isodata`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="isodata"))


@computation_function
def threshold_li(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Li algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_li`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="li"))


@computation_function
def threshold_mean(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Mean algorithm,
    see :py:func:`skimage.filters.threshold_mean`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="mean"))


@computation_function
def threshold_minimum(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Minimum algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_minimum`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="minimum"))


@computation_function
def threshold_otsu(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Otsu algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_otsu`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="otsu"))


@computation_function
def threshold_triangle(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Triangle algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_triangle`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="triangle"))


@computation_function
def threshold_yen(src: ImageObj) -> ImageObj:
    """Compute the threshold using the Yen algorithm with default parameters,
    see :py:func:`skimage.filters.threshold_yen`

    Args:
        src: input image object

    Returns:
        Output image object
    """
    return threshold(src, ThresholdParam.create(method="yen"))
