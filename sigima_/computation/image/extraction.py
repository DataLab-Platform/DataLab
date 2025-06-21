# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Extraction computation module
-----------------------------

This module provides functions to extract sub-regions
and intensity profiles from images.

Main features include:
- Extraction of regions of interest (ROIs)
- Extraction of line, segment, average, and radial intensity profiles

These functions are useful for isolating specific image zones and for analyzing signal
intensity along defined paths.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported
#   in the `sigima_.param` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima_.image` module.

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
from numpy import ma

# Import as "csline" to avoid the function to be interpreted as a validation function
# in the context of DataLab's validation process:
from plotpy.panels.csection.csitem import compute_line_section as csline

import sigima_.algorithms.image as alg
from cdl.config import _
from sigima_ import computation_function
from sigima_.computation.base import dst_1_to_1
from sigima_.computation.image.base import dst_1_to_1_signal
from sigima_.obj.image import ImageObj, ImageROI, ROI2DParam
from sigima_.obj.signal import SignalObj


@computation_function()
def extract_rois(src: ImageObj, params: list[ROI2DParam]) -> ImageObj:
    """Extract multiple regions of interest from data

    Args:
        src: input image object
        params: list of ROI parameters

    Returns:
        Output image object
    """
    # Initialize x0, y0 with maximum values:
    y0, x0 = ymax, xmax = src.data.shape
    # Initialize x1, y1 with minimum values:
    y1, x1 = ymin, xmin = 0, 0
    for p in params:
        x0i, y0i, x1i, y1i = p.get_bounding_box_indices()
        x0, y0, x1, y1 = min(x0, x0i), min(y0, y0i), max(x1, x1i), max(y1, y1i)
    x0, y0 = max(x0, xmin), max(y0, ymin)
    x1, y1 = min(x1, xmax), min(y1, ymax)

    suffix = None
    if len(params) == 1:
        p = params[0]
        suffix = p.get_suffix()
    dst = dst_1_to_1(src, "extract_rois", suffix)
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    dst.roi = None

    src2 = src.copy()
    src2.roi = ImageROI.from_params(src2, params)
    src2.data[src2.maskdata] = 0
    dst.data = src2.data[y0:y1, x0:x1]
    return dst


@computation_function()
def extract_roi(src: ImageObj, p: ROI2DParam) -> ImageObj:
    """Extract single ROI

    Args:
        src: input image object
        p: ROI parameters

    Returns:
        Output image object
    """
    dst = dst_1_to_1(src, "extract_roi", p.get_suffix())
    dst.data = p.get_data(src).copy()
    dst.roi = p.get_extracted_roi(src)
    x0, y0, _x1, _y1 = p.get_bounding_box_indices()
    dst.x0 += x0 * src.dx
    dst.y0 += y0 * src.dy
    return dst


class LineProfileParam(gds.DataSet):
    """Horizontal or vertical profile parameters"""

    _prop = gds.GetAttrProp("direction")
    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True).set_prop(
        "display", store=_prop
    )
    row = gds.IntItem(_("Row"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "horizontal")
    )
    col = gds.IntItem(_("Column"), default=0, min=0).set_prop(
        "display", active=gds.FuncProp(_prop, lambda x: x == "vertical")
    )


@computation_function()
def line_profile(src: ImageObj, p: LineProfileParam) -> SignalObj:
    """Compute horizontal or vertical profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the profile
    """
    data = src.get_masked_view()
    p.row = min(p.row, data.shape[0] - 1)
    p.col = min(p.col, data.shape[1] - 1)
    if p.direction == "horizontal":
        suffix, shape_index, pdata = f"row={p.row}", 1, data[p.row, :]
    else:
        suffix, shape_index, pdata = f"col={p.col}", 0, data[:, p.col]
    pdata: ma.MaskedArray
    x = np.arange(data.shape[shape_index])[~pdata.mask]
    y = np.array(pdata, dtype=float)[~pdata.mask]
    dst = dst_1_to_1_signal(src, "profile", suffix)
    dst.set_xydata(x, y)
    return dst


class SegmentProfileParam(gds.DataSet):
    """Segment profile parameters"""

    row1 = gds.IntItem(_("Start row"), default=0, min=0)
    col1 = gds.IntItem(_("Start column"), default=0, min=0)
    row2 = gds.IntItem(_("End row"), default=0, min=0)
    col2 = gds.IntItem(_("End column"), default=0, min=0)


@computation_function()
def segment_profile(src: ImageObj, p: SegmentProfileParam) -> SignalObj:
    """Compute segment profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the segment profile
    """
    data = src.get_masked_view()
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"({p.row1}, {p.col1})-({p.row2}, {p.col2})"
    x, y = csline(data, p.row1, p.col1, p.row2, p.col2)
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]  # Remove NaN values
    dst = dst_1_to_1_signal(src, "segment_profile", suffix)
    dst.set_xydata(np.array(x, dtype=float), np.array(y, dtype=float))
    return dst


class AverageProfileParam(gds.DataSet):
    """Average horizontal or vertical profile parameters"""

    _directions = (("horizontal", _("horizontal")), ("vertical", _("vertical")))
    direction = gds.ChoiceItem(_("Direction"), _directions, radio=True)
    _hgroup_begin = gds.BeginGroup(_("Profile rectangular area"))
    row1 = gds.IntItem(_("Row 1"), default=0, min=0)
    row2 = gds.IntItem(_("Row 2"), default=-1, min=-1)
    col1 = gds.IntItem(_("Column 1"), default=0, min=0)
    col2 = gds.IntItem(_("Column 2"), default=-1, min=-1)
    _hgroup_end = gds.EndGroup(_("Profile rectangular area"))


@computation_function()
def average_profile(src: ImageObj, p: AverageProfileParam) -> SignalObj:
    """Compute horizontal or vertical average profile

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the average profile
    """
    data = src.get_masked_view()
    if p.row2 == -1:
        p.row2 = data.shape[0] - 1
    if p.col2 == -1:
        p.col2 = data.shape[1] - 1
    if p.row1 > p.row2:
        p.row1, p.row2 = p.row2, p.row1
    if p.col1 > p.col2:
        p.col1, p.col2 = p.col2, p.col1
    p.row1 = min(p.row1, data.shape[0] - 1)
    p.row2 = min(p.row2, data.shape[0] - 1)
    p.col1 = min(p.col1, data.shape[1] - 1)
    p.col2 = min(p.col2, data.shape[1] - 1)
    suffix = f"{p.direction}, rows=[{p.row1}, {p.row2}], cols=[{p.col1}, {p.col2}]"
    if p.direction == "horizontal":
        x, axis = np.arange(p.col1, p.col2 + 1), 0
    else:
        x, axis = np.arange(p.row1, p.row2 + 1), 1
    y = ma.mean(data[p.row1 : p.row2 + 1, p.col1 : p.col2 + 1], axis=axis)
    dst = dst_1_to_1_signal(src, "average_profile", suffix)
    dst.set_xydata(x, y)
    return dst


class RadialProfileParam(gds.DataSet):
    """Radial profile parameters"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__obj: ImageObj | None = None

    def update_from_obj(self, obj: ImageObj) -> None:
        """Update parameters from image"""
        self.__obj = obj
        self.x0 = obj.xc
        self.y0 = obj.yc

    def choice_callback(self, item, value):
        """Callback for choice item"""
        if value == "centroid":
            self.y0, self.x0 = alg.get_centroid_fourier(self.__obj.get_masked_view())
        elif value == "center":
            self.x0, self.y0 = self.__obj.xc, self.__obj.yc

    _prop = gds.GetAttrProp("center")
    center = gds.ChoiceItem(
        _("Center position"),
        (
            ("centroid", _("Image centroid")),
            ("center", _("Image center")),
            ("user", _("User-defined")),
        ),
        default="centroid",
    ).set_prop("display", store=_prop, callback=choice_callback)

    _func_prop = gds.FuncProp(_prop, lambda x: x == "user")
    _xyl = "<sub>" + _("Center") + "</sub>"
    x0 = gds.FloatItem(f"X{_xyl}", unit="pixel").set_prop("display", active=_func_prop)
    y0 = gds.FloatItem(f"X{_xyl}", unit="pixel").set_prop("display", active=_func_prop)


@computation_function()
def radial_profile(src: ImageObj, p: RadialProfileParam) -> SignalObj:
    """Compute radial profile around the centroid
    with :py:func:`sigima_.algorithms.image.get_radial_profile`

    Args:
        src: input image object
        p: parameters

    Returns:
        Signal object with the radial profile
    """
    data = src.get_masked_view()
    if p.center == "centroid":
        y0, x0 = alg.get_centroid_fourier(data)
    elif p.center == "center":
        x0, y0 = src.xc, src.yc
    else:
        x0, y0 = p.x0, p.y0
    suffix = f"center=({x0:.3f}, {y0:.3f})"
    dst = dst_1_to_1_signal(src, "radial_profile", suffix)
    x, y = alg.get_radial_profile(data, (x0, y0))
    dst.set_xydata(x, y)
    return dst
