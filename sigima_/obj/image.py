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
from typing import Any, Literal, Type

import guidata.dataset as gds
import numpy as np
from numpy import ma
from skimage import draw

from cdl.config import _
from sigima_.algorithms.datatypes import clip_astype
from sigima_.algorithms.image import scale_data_to_min_max
from sigima_.obj import base


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

    # Do not declare it as a static method: not supported by Python 3.9
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
        conversion to pixel coordinates is done in :class:`sigima_.obj.ImageObj`
        (see :meth:`sigima_.obj.ImageObj.physical_to_indices`). Most of the time,
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
        coords = np.array(self.get_indices_coords(obj))
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
        param = ROI2DParam(title=self.title if title is None else title)
        param.geometry = "polygon"
        param.points = np.array(self.get_indices_coords(obj))
        return param


class RectangularROI(BaseSingleImageROI):
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

    # pylint: disable=unused-argument
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
        param = ROI2DParam(title=self.title if title is None else title)
        param.geometry = "rectangle"
        param.x0, param.y0, param.dx, param.dy = self.get_indices_coords(obj)
        return param

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
        param = ROI2DParam(title=self.title if title is None else title)
        param.geometry = "circle"
        param.xc, param.yc, param.r = self.get_indices_coords(obj)
        return param

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
        xc, yc, r = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0)
        return np.array([xc, yc, r], dtype=type(x0))


class ImageROI(
    base.BaseROI[
        "ImageObj",
        BaseSingleImageROI,
        ROI2DParam,
    ]
):
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

    def to_mask(self, obj: ImageObj) -> np.ndarray:
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
        self._dicom_template = None

    @staticmethod
    def get_roi_class() -> Type[ImageROI]:
        """Return ROI class"""
        return ImageROI

    def __add_metadata(self, key: str, value: Any) -> None:
        """Add value to metadata if value can be converted into builtin/NumPy type

        Args:
            key: key
            value: value
        """
        stored_val = to_builtin(value)
        if stored_val is not None:
            self.metadata[key] = stored_val

    def __set_metadata_from(self, obj: Mapping | dict) -> None:
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
            self.__set_metadata_from(template)
            self._dicom_template = template

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data"))
    data = gds.FloatArrayItem(_("Data"))  # type: ignore[assignment]
    metadata = gds.DictItem(_("Metadata"), default={})  # type: ignore[assignment]
    annotations = gds.StringItem(_("Annotations"), default="").set_prop(
        "display",
        hide=True,
    )  # Annotations as a serialized JSON string  # type: ignore[assignment]
    _e_datag = gds.EndGroup(_("Data"))

    _dxdyg = gds.BeginGroup(f"{_('Origin')} / {_('Pixel spacing')}")
    _origin = gds.BeginGroup(_("Origin"))
    x0 = gds.FloatItem("X<sub>0</sub>", default=0.0)
    y0 = gds.FloatItem("Y<sub>0</sub>", default=0.0).set_pos(col=1)
    _e_origin = gds.EndGroup(_("Origin"))
    _pixel_spacing = gds.BeginGroup(_("Pixel spacing"))
    dx = gds.FloatItem("Δx", default=1.0, nonzero=True)
    dy = gds.FloatItem("Δy", default=1.0, nonzero=True).set_pos(col=1)
    _e_pixel_spacing = gds.EndGroup(_("Pixel spacing"))
    _e_dxdyg = gds.EndGroup(f"{_('Origin')} / {_('Pixel spacing')}")

    _unitsg = gds.BeginGroup(f"{_('Titles')} / {_('Units')}")
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
    _e_unitsg = gds.EndGroup(f"{_('Titles')} / {_('Units')}")

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
            view = self.data.view(ma.MaskedArray)
            view.mask = np.isnan(self.data)
            return view
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
        obj.annotations = self.annotations
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

    def physical_to_indices(self, coords: list[float], clip: bool = False) -> list[int]:
        """Convert coordinates from physical (real world) to indices (pixel)

        Args:
            coords: flat list of physical coordinates [x0, y0, x1, y1, ...]
            clip: if True, clip values to image boundaries

        Returns:
            Indices
        """
        indices = np.array(coords, float)
        if indices.size > 0:
            indices[::2] -= self.x0 + 0.5 * self.dx
            indices[::2] /= self.dx
            indices[1::2] -= self.y0 + 0.5 * self.dy
            indices[1::2] /= self.dy
        if clip:
            indices[::2] = np.clip(indices[::2], 0, self.data.shape[1] - 1)
            indices[1::2] = np.clip(indices[1::2], 0, self.data.shape[0] - 1)
        return np.array(indices, int).tolist()

    def indices_to_physical(self, indices: list[float]) -> list[int]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: flat list of indices [x0, y0, x1, y1, ...]

        Returns:
            Coordinates
        """
        coords = np.array(indices, float)
        if coords.size > 0:
            coords[::2] *= self.dx
            coords[::2] += self.x0 + 0.5 * self.dx
            coords[1::2] *= self.dy
            coords[1::2] += self.y0 + 0.5 * self.dy
        return coords.tolist()


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


DEFAULT_TITLE = _("Untitled image")


class NewImageParam(gds.DataSet):
    """New image dataset"""

    hide_image_height = False
    hide_image_dtype = False
    hide_image_type = False

    title = gds.StringItem(_("Title"), default=DEFAULT_TITLE)
    height = gds.IntItem(
        _("Height"), help=_("Image height: number of rows"), min=1
    ).set_prop("display", hide=gds.GetAttrProp("hide_image_height"))
    width = gds.IntItem(_("Width"), help=_("Image width: number of columns"), min=1)
    dtype = gds.ChoiceItem(_("Data type"), ImageDatatypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_image_dtype")
    )
    itype = gds.ChoiceItem(_("Type"), ImageTypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_image_type")
    )


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


IMG_NB = 0


def get_next_image_number():
    """Get the next image number.

    This function is used to keep track of the number of signals created.
    It is typically used to generate unique titles for new signals.

    Returns:
        int: new image number
    """
    global IMG_NB  # pylint: disable=global-statement
    IMG_NB += 1
    return IMG_NB


def create_image_from_param(
    base_param: NewImageParam,
    extra_param: gds.DataSet | None = None,
) -> ImageObj:
    """Create a new Image object from parameters.

    Args:
        base_param: new image parameters
        extra_param: additional parameters (optional)

    Returns:
        Image object

    Raises:
        ValueError: if `extra_param` is required but not provided
        NotImplementedError: if the image type is not supported
    """
    if base_param.height is None:
        base_param.height = 500
    if base_param.width is None:
        base_param.width = 500
    if base_param.dtype is None:
        base_param.dtype = ImageDatatypes.UINT16
    incr_img_nb = not base_param.title
    prefix = base_param.itype.name.lower()
    title = base_param.title = base_param.title or DEFAULT_TITLE
    if incr_img_nb:
        title = f"{title} {get_next_image_number()}"

    shape = (base_param.height, base_param.width)
    dtype = base_param.dtype.value
    ep = extra_param

    if base_param.itype == ImageTypes.ZEROS:
        data = np.zeros(shape, dtype=dtype)

    elif base_param.itype == ImageTypes.EMPTY:
        data = np.empty(shape, dtype=dtype)

    elif base_param.itype == ImageTypes.GAUSS:
        if ep is None:
            raise ValueError("extra_param (Gauss2DParam) required.")
        assert isinstance(ep, Gauss2DParam)
        if ep.a is None:
            try:
                ep.a = np.iinfo(dtype).max / 2.0
            except ValueError:
                ep.a = 10.0
        x, y = np.meshgrid(
            np.linspace(ep.xmin, ep.xmax, shape[1]),
            np.linspace(ep.ymin, ep.ymax, shape[0]),
        )
        zgauss = ep.a * np.exp(
            -((np.sqrt((x - ep.x0) ** 2 + (y - ep.y0) ** 2) - ep.mu) ** 2)
            / (2.0 * ep.sigma**2)
        )
        data = np.array(zgauss, dtype=dtype)
        title = (
            f"{prefix}(a={ep.a:g},μ={ep.mu:g},"
            f"σ={ep.sigma:g}),x0={ep.x0:g},y0={ep.y0:g})"
        )

    elif base_param.itype == ImageTypes.UNIFORMRANDOM:
        if ep is None:
            raise ValueError("extra_param (UniformRandomParam) required.")
        assert isinstance(ep, base.UniformRandomParam)
        rng = np.random.default_rng(ep.seed)
        data = scale_data_to_min_max(rng.random(shape), ep.vmin, ep.vmax)
        title = f"{prefix}(vmin={ep.vmin:g},vmax={ep.vmax:g},seed={ep.seed})"

    elif base_param.itype == ImageTypes.NORMALRANDOM:
        if ep is None:
            raise ValueError("extra_param (NormalRandomParam) required.")
        assert isinstance(ep, base.NormalRandomParam)
        rng = np.random.default_rng(ep.seed)
        data = rng.normal(ep.mu, ep.sigma, size=shape)
        title = f"{prefix}(μ={ep.mu:g},σ={ep.sigma:g},seed={ep.seed})"

    else:
        raise NotImplementedError(f"Image type '{base_param.itype}' not implemented.")

    title = title if base_param.title == DEFAULT_TITLE else base_param.title
    image = create_image(title, data)
    return image
