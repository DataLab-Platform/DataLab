# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image object and related classes
--------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import abc
import enum
import re
from collections.abc import ByteString, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Type
from uuid import uuid4

import guidata.dataset as gds
import numpy as np
from guidata.configtools import get_icon
from guidata.dataset import update_dataset
from plotpy.builder import make
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedPolygon,
    AnnotatedRectangle,
    MaskedImageItem,
)
from skimage import draw

from cdl.algorithms.datatypes import clip_astype
from cdl.algorithms.image import scale_data_to_min_max
from cdl.config import Conf, _
from cdl.core.model import base

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW


def to_builtin(obj) -> str | int | float | list | dict | np.ndarray | None:
    """Convert an object implementing a numeric value or collection
    into the corresponding builtin/NumPy type.

    Return None if conversion fails."""
    try:
        return int(obj) if int(obj) == float(obj) else float(obj)
    except (TypeError, ValueError):
        pass
    if isinstance(obj, ByteString):
        return str(obj)
    if isinstance(obj, Sequence):
        return str(obj) if len(obj) == len(str(obj)) else list(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray):
        return obj
    return None


class ROI2DParam(base.BaseROIParam["ImageObj", "BaseSingleImageROI"]):
    """Image ROI parameters"""

    # Note: the ROI coordinates are expressed in pixel coordinates (integers)
    # => That is the only way to handle ROI parametrization for image objects.
    #    Otherwise, we would have to ask the user to systematically provide the
    #    physical coordinates: that would be cumbersome and error-prone.

    _geometry_prop = gds.GetAttrProp("geometry")
    _rfp = gds.FuncProp(_geometry_prop, lambda x: x != "rectangle")
    _cfp = gds.FuncProp(_geometry_prop, lambda x: x != "circle")
    _pfp = gds.FuncProp(_geometry_prop, lambda x: x != "polygon")

    # Do not declare it as a static method: not supported on Python 3.8
    def _lbl(name: str, index: int):  # pylint: disable=no-self-argument
        """Returns name<sub>index</sub>"""
        return f"{name}<sub>{index}</sub>"

    _ut = "pixels"

    geometries = ("rectangle", "circle", "polygon")
    geometry = gds.ChoiceItem(
        _("Geometry"), list(zip(geometries, geometries)), default="rectangle"
    ).set_prop("display", store=_geometry_prop, hide=True)

    # Parameters for rectangular ROI geometry:
    _tlcorner = gds.BeginGroup(_("Top left corner")).set_prop("display", hide=_rfp)
    x0 = gds.IntItem(_lbl("X", 0), unit=_ut).set_prop("display", hide=_rfp)
    y0 = gds.IntItem(_lbl("Y", 0), unit=_ut).set_pos(1).set_prop("display", hide=_rfp)
    _e_tlcorner = gds.EndGroup(_("Top left corner"))
    dx = gds.IntItem("ΔX", unit=_ut).set_prop("display", hide=_rfp)
    dy = gds.IntItem("ΔY", unit=_ut).set_pos(1).set_prop("display", hide=_rfp)

    # Parameters for circular ROI geometry:
    _cgroup = gds.BeginGroup(_("Center coordinates")).set_prop("display", hide=_cfp)
    xc = gds.IntItem(_lbl("X", "C"), unit=_ut).set_prop("display", hide=_cfp)
    yc = gds.IntItem(_lbl("Y", "C"), unit=_ut).set_pos(1).set_prop("display", hide=_cfp)
    _e_cgroup = gds.EndGroup(_("Center coordinates"))
    r = gds.IntItem(_("Radius"), unit=_ut).set_prop("display", hide=_cfp)

    # Parameters for polygonal ROI geometry:
    points = gds.FloatArrayItem(_("Coordinates") + f" ({_ut})").set_prop(
        "display", hide=_pfp
    )

    def to_single_roi(
        self, obj: ImageObj, title: str = ""
    ) -> PolygonalROI | RectangularROI | CircularROI:
        """Convert parameters to single ROI

        Args:
            obj: image object (used for conversion of pixel to physical coordinates)
            title: ROI title

        Returns:
            Single ROI
        """
        if self.geometry == "rectangle":
            return RectangularROI.from_param(obj, self)
        if self.geometry == "circle":
            return CircularROI.from_param(obj, self)
        if self.geometry == "polygon":
            return PolygonalROI.from_param(obj, self)
        raise ValueError(f"Unknown ROI geometry type: {self.geometry}")

    def get_suffix(self) -> str:
        """Get suffix text representation for ROI extraction"""
        if self.geometry == "rectangle":
            return f"x0={self.x0},y0={self.y0},dx={self.dx},dy={self.dy}"
        if self.geometry == "circle":
            return f"xc={self.xc},yc={self.yc},r={self.r}"
        if self.geometry == "polygon":
            return "polygon"
        raise ValueError(f"Unknown ROI geometry type: {self.geometry}")

    def get_extracted_roi(self, obj: ImageObj) -> ImageROI | None:
        """Get extracted ROI, i.e. the remaining ROI after extracting ROI from image.

        Args:
            obj: image object (used for conversion of pixel to physical coordinates)

        When extracting ROIs from an image to multiple images (i.e. one image per ROI),
        this method returns the ROI that has to be kept in the destination image. This
        is not necessary for a rectangular ROI: the destination image is simply a crop
        of the source image according to the ROI coordinates. But for a circular ROI or
        a polygonal ROI, the destination image is a crop of the source image according
        to the bounding box of the ROI. Thus, to avoid any loss of information, a ROI
        has to be defined for the destination image: this is the ROI returned by this
        method. It's simply the same as the source ROI, but with coordinates adjusted
        to the destination image. One may called this ROI the "extracted ROI".
        """
        if self.geometry == "rectangle":
            return None
        single_roi = self.to_single_roi(obj)
        x0, y0, _x1, _y1 = self.get_bounding_box_indices()
        single_roi.translate(obj, -x0, -y0)
        roi = ImageROI()
        roi.add_roi(single_roi)
        return roi

    def get_bounding_box_indices(self) -> tuple[int, int, int, int]:
        """Get bounding box (pixel coordinates)"""
        if self.geometry == "circle":
            x0, y0 = self.xc - self.r, self.yc - self.r
            x1, y1 = self.xc + self.r, self.yc + self.r
        elif self.geometry == "rectangle":
            x0, y0, x1, y1 = self.x0, self.y0, self.x0 + self.dx, self.y0 + self.dy
        else:
            self.points: np.ndarray
            x0, y0 = self.points[::2].min(), self.points[1::2].min()
            x1, y1 = self.points[::2].max(), self.points[1::2].max()
        return x0, y0, x1, y1

    def get_data(self, obj: ImageObj) -> np.ndarray:
        """Get data in ROI

        Args:
            obj: image object

        Returns:
            Data in ROI
        """
        x0, y0, x1, y1 = self.get_bounding_box_indices()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(obj.data.shape[1], x1), min(obj.data.shape[0], y1)
        return obj.data[y0:y1, x0:x1]


class BaseSingleImageROI(base.BaseSingleROI["ImageObj", ROI2DParam], abc.ABC):
    """Base class for single image ROI

    Args:
        coords: ROI edge coordinates (floats)
        title: ROI title

    .. note::

        The image ROI coords are expressed in physical coordinates (floats). The
        conversion to pixel coordinates is done in :class:`cdl.obj.ImageObj`
        (see :meth:`cdl.obj.ImageObj.physical_to_indices`). Most of the time,
        the physical coordinates are the same as the pixel coordinates, but this
        is not always the case (e.g. after image binning), so it's better to keep the
        physical coordinates in the ROI object: this will help reusing the ROI with
        different images (e.g. with different pixel sizes).
    """

    @abc.abstractmethod
    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """

    @abc.abstractmethod
    def translate(self, obj: ImageObj, dx: int, dy: int) -> None:
        """Translate ROI

        Args:
            obj: image object
            dx: translation along X-axis
            dy: translation along Y-axis
        """


class PolygonalROI(BaseSingleImageROI):
    """Polygonal ROI

    Args:
        coords: ROI edge coordinates
        title: title

    Raises:
        ValueError: if number of coordinates is odd

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) % 2 != 0:
            raise ValueError("Edge indices must be pairs of X, Y values")

    # pylint: disable=unused-argument
    @classmethod
    def from_param(cls: PolygonalROI, obj: ImageObj, param: ROI2DParam) -> PolygonalROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        indices = True  # ROI coordinates are in pixel coordinates in `ROI2DParam`
        return cls(param.points, indices=indices, title=param.get_title())

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        coords = self.get_physical_coords(obj)
        x_edges, y_edges = coords[::2], coords[1::2]
        return min(x_edges), min(y_edges), max(x_edges), max(y_edges)

    def translate(self, obj: ImageObj, dx: int, dy: int) -> None:
        """Translate ROI

        Args:
            obj: image object
            dx: translation along X-axis
            dy: translation along Y-axis
        """
        coords = self.get_indices_coords(obj)
        coords[::2] += int(dx)
        coords[1::2] += int(dy)
        self.set_indices_coords(obj, coords)

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        indices = self.get_indices_coords(obj)
        rows, cols = indices[1::2], indices[::2]
        rr, cc = draw.polygon(rows, cols, shape=obj.data.shape)
        roi_mask[rr, cc] = False
        return roi_mask

    def to_param(self, obj: ImageObj, title: str | None = None) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: ROI title
        """
        param = ROI2DParam(title="ROI" if title is None else title)
        param.geometry = "polygon"
        param.points = self.get_indices_coords(obj)
        return param

    def to_plot_item(self, obj: ImageObj, title: str) -> AnnotatedPolygon:
        """Make and return the annnotated polygon associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """
        item = AnnotatedPolygon(self.get_physical_coords(obj))
        item.annotationparam.title = title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @classmethod
    def from_plot_item(cls: PolygonalROI, item: AnnotatedPolygon) -> PolygonalROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        return cls(item.get_points(), False, item.annotationparam.title)


class RectangularROI(PolygonalROI):
    """Rectangular ROI

    Args:
        coords: ROI edge coordinates (x0, y0, dx, dy)
        title: title

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 4:
            raise ValueError("Rectangle ROI requires 4 coordinates")

    @classmethod
    def from_param(
        cls: RectangularROI, obj: ImageObj, param: ROI2DParam
    ) -> RectangularROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        ix0, iy0, ix1, iy1 = param.get_bounding_box_indices()
        coords = [ix0, iy0, ix1 - ix0, iy1 - iy0]
        indices = True  # ROI coordinates are in pixel coordinates in `ROI2DParam`
        return cls(coords, indices=indices, title=param.get_title())

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        x0, y0, dx, dy = self.get_physical_coords(obj)
        return x0, y0, x0 + dx, y0 + dy

    def translate(self, obj: ImageObj, dx: int, dy: int) -> None:
        """Translate ROI

        Args:
            obj: image object
            dx: translation along X-axis
            dy: translation along Y-axis
        """
        coords = self.get_indices_coords(obj)
        coords[0] += int(dx)
        coords[1] += int(dy)
        self.set_indices_coords(obj, coords)

    def get_physical_coords(self, obj: ImageObj) -> np.ndarray:
        """Return physical coords

        Args:
            obj: image object

        Returns:
            Physical coords
        """
        if self.indices:
            ix0, iy0, idx, idy = self.coords
            x0, y0, x1, y1 = obj.indices_to_physical([ix0, iy0, ix0 + idx, iy0 + idy])
            return [x0, y0, x1 - x0, y1 - y0]
        return self.coords

    def get_indices_coords(self, obj: ImageObj) -> np.ndarray:
        """Return indices coords

        Args:
            obj: image object

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords
        ix0, iy0, ix1, iy1 = obj.physical_to_indices(self.get_bounding_box(obj))
        return [ix0, iy0, ix1 - ix0, iy1 - iy0]

    def set_indices_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            ix0, iy0, idx, idy = coords
            x0, y0, x1, y1 = obj.indices_to_physical([ix0, iy0, ix0 + idx, iy0 + idy])
            self.coords = [x0, y0, x1 - x0, y1 - y0]

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        x0, y0, dx, dy = self.get_indices_coords(obj)
        roi_mask[max(y0, 0) : y0 + dy, max(x0, 0) : x0 + dx] = False
        return roi_mask

    def to_param(self, obj: ImageObj, title: str | None = None) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: ROI title
        """
        param = ROI2DParam(title="ROI" if title is None else title)
        param.geometry = "rectangle"
        param.x0, param.y0, param.dx, param.dy = self.get_indices_coords(obj)
        return param

    def to_plot_item(self, obj: ImageObj, title: str) -> AnnotatedRectangle:
        """Make and return the annnotated rectangle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """

        def info_callback(item: AnnotatedRectangle) -> str:
            """Return info string for rectangular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            x0, y0, dx, dy = self.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"X0, Y0 = {x0:g}, {y0:g}",
                    f"ΔX x ΔY  = {dx:g} x {dy:g}",
                ]
            )

        x0, y0, dx, dy = self.get_physical_coords(obj)
        x1, y1 = x0 + dx, y0 + dy
        roi_item: AnnotatedRectangle = make.annotated_rectangle(x0, y0, x1, y1, title)
        roi_item.set_info_callback(info_callback)
        param = roi_item.label.labelparam
        param.anchor = "BL"
        param.xc, param.yc = 5, -5
        param.update_item(roi_item.label)
        return roi_item

    @staticmethod
    def rect_to_coords(
        x0: int | float, y0: int | float, x1: int | float, y1: int | float
    ) -> np.ndarray:
        """Convert rectangle to coordinates

        Args:
            x0: x0 (top-left corner)
            y0: y0 (top-left corner)
            x1: x1 (bottom-right corner)
            y1: y1 (bottom-right corner)

        Returns:
            Rectangle coordinates
        """
        return np.array([x0, y0, x1 - x0, y1 - y0], dtype=type(x0))

    @classmethod
    def from_plot_item(cls: RectangularROI, item: AnnotatedRectangle) -> RectangularROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        rect = item.get_rect()
        return cls(cls.rect_to_coords(*rect), False, item.annotationparam.title)


class CircularROI(BaseSingleImageROI):
    """Circular ROI

    Args:
        coords: ROI edge coordinates (xc, yc, r)
        title: title

    .. note:: The image ROI coords are expressed in physical coordinates (floats)
    """

    # pylint: disable=unused-argument
    @classmethod
    def from_param(cls: CircularROI, obj: ImageObj, param: ROI2DParam) -> CircularROI:
        """Create ROI from parameters

        Args:
            obj: image object
            param: parameters
        """
        ix0, iy0, ix1, iy1 = param.get_bounding_box_indices()
        ixc, iyc = (ix0 + ix1) * 0.5, (iy0 + iy1) * 0.5
        ir = (ix1 - ix0) * 0.5
        indices = True  # ROI coordinates are in pixel coordinates in `ROI2DParam`
        return cls([ixc, iyc, ir], indices=indices, title=param.get_title())

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 3:
            raise ValueError("Circle ROI requires 3 coordinates")

    def get_bounding_box(self, obj: ImageObj) -> tuple[float, float, float, float]:
        """Get bounding box (physical coordinates)

        Args:
            obj: image object
        """
        xc, yc, r = self.get_physical_coords(obj)
        return xc - r, yc - r, xc + r, yc + r

    def translate(self, obj: ImageObj, dx: int, dy: int) -> None:
        """Translate ROI

        Args:
            obj: image object
            dx: translation along X-axis
            dy: translation along Y-axis
        """
        coords = self.get_indices_coords(obj)
        coords[0] += int(dx)
        coords[1] += int(dy)
        self.set_indices_coords(obj, coords)

    def get_physical_coords(self, obj: ImageObj) -> np.ndarray:
        """Return physical coords

        Args:
            obj: image object

        Returns:
            Physical coords
        """
        if self.indices:
            ixc, iyc, ir = self.coords
            x0, y0, x1, y1 = obj.indices_to_physical(
                [ixc - ir, iyc - ir, ixc + ir, iyc + ir]
            )
            return [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)]
        return self.coords

    def get_indices_coords(self, obj: ImageObj) -> np.ndarray:
        """Return indices coords

        Args:
            obj: image object

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords
        ix0, iy0, ix1, iy1 = obj.physical_to_indices(self.get_bounding_box(obj))
        ixc, iyc = int((ix0 + ix1) * 0.5), int((iy0 + iy1) * 0.5)
        ir = int((ix1 - ix0) * 0.5)
        return [ixc, iyc, ir]

    def set_indices_coords(self, obj: ImageObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            ixc, iyc, ir = coords
            x0, y0, x1, y1 = obj.indices_to_physical(
                [ixc - ir, iyc - ir, ixc + ir, iyc + ir]
            )
            self.coords = [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)]

    def to_mask(self, obj: ImageObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        roi_mask = np.ones_like(obj.data, dtype=bool)
        ixc, iyc, ir = self.get_indices_coords(obj)
        yxratio = obj.dy / obj.dx
        rr, cc = draw.ellipse(iyc, ixc, ir / yxratio, ir, shape=obj.data.shape)
        roi_mask[rr, cc] = False
        return roi_mask

    def to_param(self, obj: ImageObj, title: str | None = None) -> ROI2DParam:
        """Convert ROI to parameters

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: ROI title
        """
        param = ROI2DParam(title="ROI" if title is None else title)
        param.geometry = "circle"
        param.xc, param.yc, param.r = self.get_indices_coords(obj)
        return param

    def to_plot_item(self, obj: ImageObj, title: str) -> AnnotatedCircle:
        """Make and return the annnotated circle associated to ROI

        Args:
            obj: object (image), for physical-indices coordinates conversion
            title: title
        """

        def info_callback(item: AnnotatedCircle) -> str:
            """Return info string for circular ROI"""
            x0, y0, x1, y1 = item.get_rect()
            if self.indices:
                x0, y0, x1, y1 = obj.physical_to_indices([x0, y0, x1, y1])
            xc, yc, r = self.rect_to_coords(x0, y0, x1, y1)
            return "<br>".join(
                [
                    f"Center = {xc:g}, {yc:g}",
                    f"Radius = {r:g}",
                ]
            )

        xc, yc, r = self.get_physical_coords(obj)
        item = AnnotatedCircle(xc - r, yc, xc + r, yc)
        item.set_info_callback(info_callback)
        item.annotationparam.title = title
        item.annotationparam.update_item(item)
        item.set_style("plot", "shape/drag")
        return item

    @staticmethod
    def rect_to_coords(
        x0: int | float, y0: int | float, x1: int | float, y1: int | float
    ) -> np.ndarray:
        """Convert rectangle to circle coordinates

        Args:
            x0: x0 (top-left corner)
            y0: y0 (top-left corner)
            x1: x1 (bottom-right corner)
            y1: y1 (bottom-right corner)

        Returns:
            Circle coordinates
        """
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        r = 0.5 * ((x1 - x0) + (y1 - y0))
        return np.array([xc, yc, r], dtype=type(x0))

    @classmethod
    def from_plot_item(cls: CircularROI, item: AnnotatedCircle) -> CircularROI:
        """Create ROI from plot item

        Args:
            item: plot item
        """
        rect = item.get_rect()
        return cls(cls.rect_to_coords(*rect), False, item.annotationparam.title)


class ImageROI(base.BaseROI["ImageObj", BaseSingleImageROI, ROI2DParam]):
    """Image Regions of Interest

    Args:
        singleobj: if True, when extracting data defined by ROIs, only one object
         is created (default to True). If False, one object is created per single ROI.
         If None, the value is get from the user configuration
        inverse: if True, ROI is outside the region
    """

    PREFIX = "i"

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[BaseSingleImageROI]]:
        """Return compatible single ROI classes"""
        return [RectangularROI, CircularROI, PolygonalROI]

    def to_mask(self, obj: ImageObj) -> np.ndarray[bool]:
        """Create mask from ROI

        Args:
            obj: image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.data, dtype=bool)
        for roi in self.single_rois:
            mask &= roi.to_mask(obj)
        return mask


def create_image_roi(
    geometry: Literal["rectangle", "circle", "polygon"],
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = True,
    singleobj: bool | None = None,
    inverse: bool = False,
    title: str = "",
) -> ImageROI:
    """Create Image Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        geometry: ROI type ('rectangle', 'circle', 'polygon')
        coords: ROI coords (physical coordinates), `[x0, y0, dx, dy]` for a rectangle,
         `[xc, yc, r]` for a circle, or `[x0, y0, x1, y1, ...]` for a polygon (lists or
         NumPy arrays are accepted). For multiple ROIs, nested lists or NumPy arrays are
         accepted but with a common geometry type (e.g.
         `[[xc1, yc1, r1], [xc2, yc2, r2], ...]` for circles).
        indices: if True, coordinates are indices, if False, they are physical values
         (default to True for images)
        singleobj: if True, when extracting data defined by ROIs, only one object
         is created (default to True). If False, one object is created per single ROI.
         If None, the value is get from the user configuration
        inverse: if True, ROI is outside the region
        title: title

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if ROI type is unknown or if the number of coordinates is invalid
    """
    coords = np.array(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    roi = ImageROI(singleobj, inverse)
    if geometry == "rectangle":
        if coords.shape[1] != 4:
            raise ValueError("Rectangle ROI requires 4 coordinates")
        for row in coords:
            roi.add_roi(RectangularROI(row, indices, title))
    elif geometry == "circle":
        if coords.shape[1] != 3:
            raise ValueError("Circle ROI requires 3 coordinates")
        for row in coords:
            roi.add_roi(CircularROI(row, indices, title))
    elif geometry == "polygon":
        if coords.shape[1] % 2 != 0:
            raise ValueError("Polygon ROI requires pairs of X, Y coordinates")
        for row in coords:
            roi.add_roi(PolygonalROI(row, indices, title))
    else:
        raise ValueError(f"Unknown ROI type: {geometry}")
    return roi


class ImageObj(gds.DataSet, base.BaseObj[ImageROI]):
    """Image object"""

    PREFIX = "i"
    CONF_FMT = Conf.view.ima_format
    DEFAULT_FMT = ".1f"
    VALID_DTYPES = (
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
        np.complex128,
    )

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title: title
            comment: comment
            icon: icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)
        self.regenerate_uuid()
        self._dicom_template = None

    @staticmethod
    def get_roi_class() -> Type[ImageROI]:
        """Return ROI class"""
        return ImageROI

    def regenerate_uuid(self):
        """Regenerate UUID

        This method is used to regenerate UUID after loading the object from a file.
        This is required to avoid UUID conflicts when loading objects from file
        without clearing the workspace first.
        """
        self.uuid = str(uuid4())

    def __add_metadata(self, key: str, value: Any) -> None:
        """Add value to metadata if value can be converted into builtin/NumPy type

        Args:
            key: key
            value: value
        """
        stored_val = to_builtin(value)
        if stored_val is not None:
            self.metadata[key] = stored_val

    def set_metadata_from(self, obj: Mapping | dict) -> None:
        """Set metadata from object: dict-like (only string keys are considered)
        or any other object (iterating over supported attributes)

        Args:
            obj: object
        """
        self.reset_metadata_to_defaults()
        ptn = r"__[\S_]*__$"
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if isinstance(key, str) and not re.match(ptn, key):
                    self.__add_metadata(key, value)
        else:
            for attrname in dir(obj):
                if attrname != "GroupLength" and not re.match(ptn, attrname):
                    try:
                        attr = getattr(obj, attrname)
                        if not callable(attr) and attr:
                            self.__add_metadata(attrname, attr)
                    except AttributeError:
                        pass

    @property
    def dicom_template(self):
        """Get DICOM template"""
        return self._dicom_template

    @dicom_template.setter
    def dicom_template(self, template):
        """Set DICOM template"""
        if template is not None:
            ipp = getattr(template, "ImagePositionPatient", None)
            if ipp is not None:
                self.x0, self.y0 = float(ipp[0]), float(ipp[1])
            pxs = getattr(template, "PixelSpacing", None)
            if pxs is not None:
                self.dy, self.dx = float(pxs[0]), float(pxs[1])
            self.set_metadata_from(template)
            self._dicom_template = template

    uuid = gds.StringItem("UUID").set_prop("display", hide=True)

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data"))
    data = gds.FloatArrayItem(_("Data"))
    metadata = gds.DictItem(_("Metadata"), default={})
    _e_datag = gds.EndGroup(_("Data"))

    _dxdyg = gds.BeginGroup(f'{_("Origin")} / {_("Pixel spacing")}')
    _origin = gds.BeginGroup(_("Origin"))
    x0 = gds.FloatItem("X<sub>0</sub>", default=0.0)
    y0 = gds.FloatItem("Y<sub>0</sub>", default=0.0).set_pos(col=1)
    _e_origin = gds.EndGroup(_("Origin"))
    _pixel_spacing = gds.BeginGroup(_("Pixel spacing"))
    dx = gds.FloatItem("Δx", default=1.0, nonzero=True)
    dy = gds.FloatItem("Δy", default=1.0, nonzero=True).set_pos(col=1)
    _e_pixel_spacing = gds.EndGroup(_("Pixel spacing"))
    _e_dxdyg = gds.EndGroup(f'{_("Origin")} / {_("Pixel spacing")}')

    _unitsg = gds.BeginGroup(f'{_("Titles")} / {_("Units")}')
    title = gds.StringItem(_("Image title"), default=_("Untitled"))
    _tabs_u = gds.BeginTabGroup("units")
    _unitsx = gds.BeginGroup(_("X-axis"))
    xlabel = gds.StringItem(_("Title"), default="")
    xunit = gds.StringItem(_("Unit"), default="")
    _e_unitsx = gds.EndGroup(_("X-axis"))
    _unitsy = gds.BeginGroup(_("Y-axis"))
    ylabel = gds.StringItem(_("Title"), default="")
    yunit = gds.StringItem(_("Unit"), default="")
    _e_unitsy = gds.EndGroup(_("Y-axis"))
    _unitsz = gds.BeginGroup(_("Z-axis"))
    zlabel = gds.StringItem(_("Title"), default="")
    zunit = gds.StringItem(_("Unit"), default="")
    _e_unitsz = gds.EndGroup(_("Z-axis"))
    _e_tabs_u = gds.EndTabGroup("units")
    _e_unitsg = gds.EndGroup(f'{_("Titles")} / {_("Units")}')

    _scalesg = gds.BeginGroup(_("Scales"))
    _prop_autoscale = gds.GetAttrProp("autoscale")
    autoscale = gds.BoolItem(_("Auto scale"), default=True).set_prop(
        "display", store=_prop_autoscale
    )
    _tabs_b = gds.BeginTabGroup("bounds")
    _boundsx = gds.BeginGroup(_("X-axis"))
    xscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    xscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    xscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsx = gds.EndGroup(_("X-axis"))
    _boundsy = gds.BeginGroup(_("Y-axis"))
    yscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    yscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    yscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsy = gds.EndGroup(_("Y-axis"))
    _boundsz = gds.BeginGroup(_("LUT range"))
    zscalemin = gds.FloatItem(_("Lower bound"), check=False)
    zscalemax = gds.FloatItem(_("Upper bound"), check=False)
    _e_boundsz = gds.EndGroup(_("LUT range"))
    _e_tabs_b = gds.EndTabGroup("bounds")
    _e_scalesg = gds.EndGroup(_("Scales"))

    _e_tabs = gds.EndTabGroup("all")

    @property
    def width(self) -> float:
        """Return image width, i.e. number of columns multiplied by pixel size"""
        return self.data.shape[1] * self.dx

    @property
    def height(self) -> float:
        """Return image height, i.e. number of rows multiplied by pixel size"""
        return self.data.shape[0] * self.dy

    @property
    def xc(self) -> float:
        """Return image center X-axis coordinate"""
        return self.x0 + 0.5 * self.width

    @property
    def yc(self) -> float:
        """Return image center Y-axis coordinate"""
        return self.y0 + 0.5 * self.height

    def get_data(self, roi_index: int | None = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Masked data
        """
        if self.roi is None or roi_index is None:
            return self.data
        single_roi = self.roi.get_single_roi(roi_index)
        x0, y0, x1, y1 = self.physical_to_indices(single_roi.get_bounding_box(self))
        return self.get_masked_view()[y0:y1, x0:x1]

    def copy(self, title: str | None = None, dtype: np.dtype | None = None) -> ImageObj:
        """Copy object.

        Args:
            title: title
            dtype: data type

        Returns:
            Copied object
        """
        title = self.title if title is None else title
        obj = ImageObj(title=title)
        obj.title = title
        obj.xlabel = self.xlabel
        obj.ylabel = self.ylabel
        obj.xunit = self.xunit
        obj.yunit = self.yunit
        obj.zunit = self.zunit
        obj.x0 = self.x0
        obj.y0 = self.y0
        obj.dx = self.dx
        obj.dy = self.dy
        obj.metadata = base.deepcopy_metadata(self.metadata)
        obj.data = np.array(self.data, copy=True, dtype=dtype)
        obj.dicom_template = self.dicom_template
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:
        """Change data type.
        If data type is integer, clip values to the new data type's range, thus avoiding
        overflow or underflow.

        Args:
            Data type
        """
        self.data = clip_astype(self.data, dtype)

    def __viewable_data(self) -> np.ndarray:
        """Return viewable data"""
        data = self.data.real
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, posinf=0, neginf=0)
        return data

    def update_plot_item_parameters(self, item: MaskedImageItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        for axis in ("x", "y", "z"):
            unit = getattr(self, axis + "unit")
            fmt = r"%.1f"
            if unit:
                fmt = r"%.1f (" + unit + ")"
            setattr(item.param, axis + "format", fmt)
        # Updating origin and pixel spacing
        has_origin = self.x0 is not None and self.y0 is not None
        has_pixelspacing = self.dx is not None and self.dy is not None
        if has_origin or has_pixelspacing:
            x0, y0, dx, dy = 0.0, 0.0, 1.0, 1.0
            if has_origin:
                x0, y0 = self.x0, self.y0
            if has_pixelspacing:
                dx, dy = self.dx, self.dy
            shape = self.data.shape
            item.param.xmin, item.param.xmax = x0, x0 + dx * shape[1]
            item.param.ymin, item.param.ymax = y0, y0 + dy * shape[0]
        zmin, zmax = item.get_lut_range()
        if self.zscalemin is not None or self.zscalemax is not None:
            zmin = zmin if self.zscalemin is None else self.zscalemin
            zmax = zmax if self.zscalemax is None else self.zscalemax
            item.set_lut_range([zmin, zmax])
        super().update_plot_item_parameters(item)

    def update_metadata_from_plot_item(self, item: MaskedImageItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        super().update_metadata_from_plot_item(item)
        # Updating the LUT range:
        self.zscalemin, self.zscalemax = item.get_lut_range()
        # Updating origin and pixel spacing:
        shape = self.data.shape
        param = item.param
        xmin, xmax, ymin, ymax = param.xmin, param.xmax, param.ymin, param.ymax
        if xmin == 0 and ymin == 0 and xmax == shape[1] and ymax == shape[0]:
            self.x0, self.y0, self.dx, self.dy = 0.0, 0.0, 1.0, 1.0
        else:
            self.x0, self.y0 = xmin, ymin
            self.dx, self.dy = (xmax - xmin) / shape[1], (ymax - ymin) / shape[0]

    def make_item(self, update_from: MaskedImageItem | None = None) -> MaskedImageItem:
        """Make plot item from data.

        Args:
            update_from: update from plot item

        Returns:
            Plot item
        """
        data = self.__viewable_data()
        item = make.maskedimage(
            data,
            self.maskdata,
            title=self.title,
            colormap="viridis",
            eliminate_outliers=Conf.view.ima_eliminate_outliers.get(),
            interpolation="nearest",
            show_mask=True,
        )
        if update_from is None:
            self.update_plot_item_parameters(item)
        else:
            update_dataset(item.param, update_from.param)
            item.param.update_item(item)
        return item

    def update_item(self, item: MaskedImageItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """
        if data_changed:
            item.set_data(self.__viewable_data(), lut_range=[item.min, item.max])
        item.set_mask(self.maskdata)
        item.param.label = self.title
        self.update_plot_item_parameters(item)
        item.plot().update_colormap_axis(item)

    def new_roi_item(
        self,
        fmt: str,
        lbl: bool,
        editable: bool,
        geometry: Literal["rectangle", "circle"] = "rectangle",
    ) -> MaskedImageItem:
        """Return a new ROI item from scratch

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable
            geometry: ROI geometry
        """
        frac = 0.2
        height, width = self.data.shape
        x0, x1 = frac * width, (1 - frac) * width
        y0, y1 = frac * height, (1 - frac) * height
        if geometry == "rectangle":
            coords = np.array([x0, y0, x1 - x0, y1 - y0], int)
        elif geometry == "circle":
            xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            r = (x1 - x0) * 0.5
            coords = np.array([xc, yc, r], int)
        else:
            raise ValueError(f"Unknown ROI geometry: {geometry}")
        roi = create_image_roi(geometry, coords, indices=True)
        item = roi.get_single_roi(0).to_plot_item(self, "ROI")
        return base.configure_roi_item(item, fmt, lbl, editable, option="i")

    def physical_to_indices(self, coords: list[float]) -> np.ndarray:
        """Convert coordinates from physical (real world) to (array) indices (pixel)

        Args:
            coords: coordinates

        Returns:
            Indices
        """
        indices = np.array(coords, float)
        ndim = indices.ndim
        if ndim == 1:
            indices = indices.reshape(1, -1)
        if indices.size > 0:
            indices[:, ::2] -= self.x0 + 0.5 * self.dx
            indices[:, ::2] /= self.dx
            indices[:, 1::2] -= self.y0 + 0.5 * self.dy
            indices[:, 1::2] /= self.dy
        if ndim == 1:
            indices = indices.flatten()
        return np.array(indices, int)

    def indices_to_physical(
        self, indices: list[float | int] | np.ndarray
    ) -> np.ndarray:
        """Convert coordinates from (array) indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
        """
        coords = np.array(indices, float)
        ndim = coords.ndim
        if ndim == 1:
            coords = coords.reshape(1, -1)
        if coords.size > 0:
            coords[:, ::2] *= self.dx
            coords[:, ::2] += self.x0 + 0.5 * self.dx
            coords[:, 1::2] *= self.dy
            coords[:, 1::2] += self.y0 + 0.5 * self.dy
        if ndim == 1:
            coords = coords.flatten()
        return coords

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use image title)
        """
        title = self.title if title is None else title
        if title:
            label = make.label(title, (self.x0, self.y0), (10, 10), "TL")
            self.add_annotations_from_items([label])


def create_image(
    title: str,
    data: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple | None = None,
    labels: tuple | None = None,
) -> ImageObj:
    """Create a new Image object

    Args:
        title: image title
        data: image data
        metadata: image metadata
        units: X, Y, Z units (tuple of strings)
        labels: X, Y, Z labels (tuple of strings)

    Returns:
        Image object
    """
    assert isinstance(title, str)
    assert data is None or isinstance(data, np.ndarray)
    image = ImageObj(title=title)
    image.title = title
    image.data = data
    if units is not None:
        image.xunit, image.yunit, image.zunit = units
    if labels is not None:
        image.xlabel, image.ylabel, image.zlabel = labels
    if metadata is not None:
        image.metadata.update(metadata)
    return image


class ImageDatatypes(base.Choices):
    """Image data types"""

    @classmethod
    def from_dtype(cls, dtype):
        """Return member from NumPy dtype"""
        return getattr(cls, str(dtype).upper(), cls.UINT8)

    @classmethod
    def check(cls):
        """Check if data types are valid"""
        for member in cls:
            assert hasattr(np, member.value)

    #: Unsigned integer number stored with 8 bits
    UINT8 = enum.auto()
    #: Unsigned integer number stored with 16 bits
    UINT16 = enum.auto()
    #: Signed integer number stored with 16 bits
    INT16 = enum.auto()
    #: Float number stored with 32 bits
    FLOAT32 = enum.auto()
    #: Float number stored with 64 bits
    FLOAT64 = enum.auto()


ImageDatatypes.check()


class ImageTypes(base.Choices):
    """Image types"""

    #: Image filled with zeros
    ZEROS = _("zeros")
    #: Empty image (filled with data from memory state)
    EMPTY = _("empty")
    #: 2D Gaussian image
    GAUSS = _("gaussian")
    #: Image filled with random data (uniform law)
    UNIFORMRANDOM = _("random (uniform law)")
    #: Image filled with random data (normal law)
    NORMALRANDOM = _("random (normal law)")


class NewImageParam(gds.DataSet):
    """New image dataset"""

    hide_image_dtype = False
    hide_image_type = False

    title = gds.StringItem(_("Title"))
    height = gds.IntItem(
        _("Height"), help=_("Image height (total number of rows)"), min=1
    )
    width = gds.IntItem(
        _("Width"), help=_("Image width (total number of columns)"), min=1
    )
    dtype = gds.ChoiceItem(_("Data type"), ImageDatatypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_image_dtype")
    )
    itype = gds.ChoiceItem(_("Type"), ImageTypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_image_type")
    )


DEFAULT_TITLE = _("Untitled image")


def new_image_param(
    title: str | None = None,
    itype: ImageTypes | None = None,
    height: int | None = None,
    width: int | None = None,
    dtype: ImageDatatypes | None = None,
) -> NewImageParam:
    """Create a new Image dataset instance.

    Args:
        title: dataset title (default: None, uses default title)
        itype: image type (default: None, uses default type)
        height: image height (default: None, uses default height)
        width: image width (default: None, uses default width)
        dtype: image data type (default: None, uses default data type)

    Returns:
        New image dataset instance
    """
    title = DEFAULT_TITLE if title is None else title
    param = NewImageParam(title=title, icon=get_icon("new_image.svg"))
    param.title = title
    if height is not None:
        param.height = height
    if width is not None:
        param.width = width
    if dtype is not None:
        param.dtype = dtype
    if itype is not None:
        param.itype = itype
    return param


IMG_NB = 0


class Gauss2DParam(gds.DataSet):
    """2D Gaussian parameters"""

    a = gds.FloatItem("Norm")
    xmin = gds.FloatItem("Xmin", default=-10).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=1.0)
    xmax = gds.FloatItem("Xmax", default=10).set_pos(col=1)
    mu = gds.FloatItem("μ", default=0.0)
    ymin = gds.FloatItem("Ymin", default=-10).set_pos(col=1)
    x0 = gds.FloatItem("X0", default=0)
    ymax = gds.FloatItem("Ymax", default=10).set_pos(col=1)
    y0 = gds.FloatItem("Y0", default=0).set_pos(col=0, colspan=1)


def create_image_from_param(
    newparam: NewImageParam,
    addparam: gds.DataSet | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> ImageObj | None:
    """Create a new Image object from dialog box.

    Args:
        newparam: new image parameters
        addparam: additional parameters
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        New image object or None if user cancelled
    """
    global IMG_NB  # pylint: disable=global-statement
    if newparam is None:
        newparam = new_image_param()
    if newparam.height is None:
        newparam.height = 500
    if newparam.width is None:
        newparam.width = 500
    if newparam.dtype is None:
        newparam.dtype = ImageDatatypes.UINT16
    incr_sig_nb = not newparam.title
    if incr_sig_nb:
        newparam.title = f"{newparam.title} {IMG_NB + 1:d}"
    if not edit or addparam is not None or newparam.edit(parent=parent):
        prefix = newparam.itype.name.lower()
        if incr_sig_nb:
            IMG_NB += 1
        image = create_image(newparam.title)
        shape = (newparam.height, newparam.width)
        dtype = newparam.dtype.value
        p = addparam
        if newparam.itype == ImageTypes.ZEROS:
            image.data = np.zeros(shape, dtype=dtype)
        elif newparam.itype == ImageTypes.EMPTY:
            image.data = np.empty(shape, dtype=dtype)
        elif newparam.itype == ImageTypes.GAUSS:
            if p is None:
                p = Gauss2DParam(_("2D-gaussian image"))
            if p.a is None:
                try:
                    p.a = np.iinfo(dtype).max / 2.0
                except ValueError:
                    p.a = 10.0
            if edit and not p.edit(parent=parent):
                return None
            x, y = np.meshgrid(
                np.linspace(p.xmin, p.xmax, shape[1]),
                np.linspace(p.ymin, p.ymax, shape[0]),
            )
            zgauss = p.a * np.exp(
                -((np.sqrt((x - p.x0) ** 2 + (y - p.y0) ** 2) - p.mu) ** 2)
                / (2.0 * p.sigma**2)
            )
            image.data = np.array(zgauss, dtype=dtype)
            if image.title == DEFAULT_TITLE:
                image.title = (
                    f"{prefix}(a={p.a:g},μ={p.mu:g},σ={p.sigma:g}),"
                    f"x0={p.x0:g},y0={p.y0:g})"
                )
        elif newparam.itype in (ImageTypes.UNIFORMRANDOM, ImageTypes.NORMALRANDOM):
            pclass = {
                ImageTypes.UNIFORMRANDOM: base.UniformRandomParam,
                ImageTypes.NORMALRANDOM: base.NormalRandomParam,
            }[newparam.itype]
            if p is None:
                p = pclass(_("Image") + " - " + newparam.itype.value)
                p.set_from_datatype(dtype)
            if edit and not p.edit(parent=parent):
                return None
            rng = np.random.default_rng(p.seed)
            if newparam.itype == ImageTypes.UNIFORMRANDOM:
                data = rng.random(shape)
                image.data = scale_data_to_min_max(data, p.vmin, p.vmax)
                if image.title == DEFAULT_TITLE:
                    image.title = (
                        f"{prefix}(vmin={p.vmin:g},vmax={p.vmax:g},seed={p.seed})"
                    )
            elif newparam.itype == ImageTypes.NORMALRANDOM:
                image.data = rng.normal(p.mu, p.sigma, size=shape)
                if image.title == DEFAULT_TITLE:
                    image.title = f"{prefix}(μ={p.mu:g},σ={p.sigma:g},seed={p.seed})"
            else:
                raise NotImplementedError(f"New param type: {newparam.itype.value}")
        return image
    return None
