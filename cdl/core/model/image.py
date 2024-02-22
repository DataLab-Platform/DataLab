# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Image object and related classes
--------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import enum
import re
from collections.abc import ByteString, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import guidata.dataset as gds
import numpy as np
from guidata.configtools import get_icon
from guidata.dataset import update_dataset
from numpy import ma
from plotpy.builder import make
from plotpy.items import AnnotatedCircle, AnnotatedRectangle, MaskedImageItem
from skimage import draw

from cdl.algorithms.image import scale_data_to_min_max
from cdl.config import Conf, _
from cdl.core.model import base

if TYPE_CHECKING:  # pragma: no cover
    from qtpy import QtWidgets as QW


def make_roi_rectangle(
    x0: int, y0: int, x1: int, y1: int, title: str
) -> AnnotatedRectangle:
    """Make and return the annnotated rectangle associated to ROI

    Args:
        x0: top left corner X coordinate
        y0: top left corner Y coordinate
        x1: bottom right corner X coordinate
        y1: bottom right corner Y coordinate
        title: title
    """
    roi_item: AnnotatedRectangle = make.annotated_rectangle(x0, y0, x1, y1, title)
    param = roi_item.label.labelparam
    param.anchor = "BL"
    param.xc, param.yc = 5, -5
    param.update_label(roi_item.label)
    return roi_item


def make_roi_circle(x0: int, y0: int, x1: int, y1: int, title: str) -> AnnotatedCircle:
    """Make and return the annnotated circle associated to ROI

    Args:
        x0: top left corner X coordinate
        y0: top left corner Y coordinate
        x1: bottom right corner X coordinate
        y1: bottom right corner Y coordinate
        title: title
    """
    item = AnnotatedCircle(x0, y0, x1, y1)
    item.annotationparam.title = title
    item.annotationparam.update_annotation(item)
    item.set_style("plot", "shape/drag")
    return item


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


class RoiDataGeometries(enum.Enum):
    """ROI data geometry types"""

    RECTANGLE = 0
    CIRCLE = 1


class RoiDataItem:
    """Object representing an image ROI.

    Args:
        data: ROI data
    """

    def __init__(self, data: np.ndarray | list | tuple):
        self._data = data

    @classmethod
    def from_image(cls, obj: ImageObj, geometry: RoiDataGeometries) -> RoiDataItem:
        """Construct roi data item from image object: called for making new ROI items

        Args:
            obj: image object
            geometry: ROI geometry
        """
        width, height = obj.data.shape[1] * obj.dx, obj.data.shape[0] * obj.dy
        frac = 0.2
        x0, x1 = obj.x0 + frac * width, obj.x0 + (1 - frac) * width
        if geometry is RoiDataGeometries.RECTANGLE:
            y0, y1 = obj.y0 + frac * height, obj.y0 + (1 - frac) * height
        else:
            y0 = y1 = obj.yc
        coords = x0, y0, x1, y1
        return cls(coords)

    @property
    def geometry(self) -> RoiDataGeometries:
        """ROI geometry"""
        _x0, y0, _x1, y1 = self._data
        if y0 == y1:
            return RoiDataGeometries.CIRCLE
        return RoiDataGeometries.RECTANGLE

    def get_rect(self) -> tuple[int, int, int, int]:
        """Get rectangle coordinates"""
        x0, y0, x1, y1 = self._data
        if self.geometry is RoiDataGeometries.CIRCLE:
            radius = int(round(0.5 * (x1 - x0)))
            y0 -= radius
            y1 += radius
        return x0, y0, x1, y1

    def get_image_masked_view(self, obj: ImageObj) -> np.ndarray:
        """Return masked view for data

        Args:
            obj: image object
        """
        x0, y0, x1, y1 = self.get_rect()
        return obj.get_masked_view()[y0:y1, x0:x1]

    def apply_mask(self, data: np.ndarray, yxratio: float) -> np.ndarray:
        """Apply ROI to data as a mask and return masked array

        Args:
            data: data
            yxratio: Y/X ratio
        """
        roi_mask = np.ones_like(data, dtype=bool)
        x0, y0, x1, y1 = self.get_rect()
        if self.geometry is RoiDataGeometries.RECTANGLE:
            roi_mask[max(y0, 0) : y1, max(x0, 0) : x1] = False
        else:
            xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            radius = 0.5 * (x1 - x0)
            rr, cc = draw.ellipse(yc, xc, radius / yxratio, radius, shape=data.shape)
            roi_mask[rr, cc] = False
        return roi_mask

    def make_roi_item(
        self, index: int | None, fmt: str, lbl: bool, editable: bool = True
    ):
        """Make ROI plot item

        Args:
            index: ROI index
            fmt: format string
            lbl: if True, show label
            editable: if True, ROI is editable
        """
        coords = self._data
        if self.geometry is RoiDataGeometries.RECTANGLE:
            func = make_roi_rectangle
        else:
            func = make_roi_circle
        title = "ROI" if index is None else f"ROI{index:02d}"
        return base.make_roi_item(
            func, coords, title, fmt, lbl, editable, option="shape/drag"
        )


def roi_label(name: str, index: int):
    """Returns name<sub>index</sub>"""
    return f"{name}<sub>{index}</sub>"


class RectangleROIParam(gds.DataSet):
    """ROI parameters"""

    geometry = RoiDataGeometries.RECTANGLE

    def get_suffix(self):
        """Get suffix text representation for ROI extraction"""
        return f"x={self.x0}:{self.x1},y={self.y0}:{self.y1}"

    def get_coords(self):
        """Get ROI coordinates"""
        return self.x0, self.y0, self.x1, self.y1

    _tlcorner = gds.BeginGroup(_("Top left corner"))
    x0 = gds.IntItem(roi_label("X", 0), unit="pixel")
    y0 = gds.IntItem(roi_label("Y", 0), unit="pixel").set_pos(1)
    _e_tlcorner = gds.EndGroup(_("Top left corner"))
    _brcorner = gds.BeginGroup(_("Bottom right corner"))
    x1 = gds.IntItem(roi_label("X", 1), unit="pixel")
    y1 = gds.IntItem(roi_label("Y", 1), unit="pixel").set_pos(1)
    _e_brcorner = gds.EndGroup(_("Bottom right corner"))


class CircularROIParam(gds.DataSet):
    """ROI parameters"""

    geometry = RoiDataGeometries.CIRCLE

    def get_single_roi(self):
        """Get single circular ROI, i.e. after extracting ROI from image"""
        return np.array([(0, self.r, self.x1 - self.x0, self.r)], int)

    def get_suffix(self):
        """Get suffix text representation for ROI extraction"""
        return f"xc={self.xc},yc={self.yc},r={self.r}"

    def get_coords(self):
        """Get ROI coordinates"""
        return self.x0, self.yc, self.x1, self.yc

    @property
    def x0(self):
        """Return rectangle top left corner X coordinate"""
        return self.xc - self.r

    @property
    def x1(self):
        """Return rectangle bottom right corner X coordinate"""
        return self.xc + self.r

    @property
    def y0(self):
        """Return rectangle top left corner Y coordinate"""
        return self.yc - self.r

    @property
    def y1(self):
        """Return rectangle bottom right corner Y coordinate"""
        return self.yc + self.r

    _tlcorner = gds.BeginGroup(_("Center coordinates"))
    xc = gds.IntItem(roi_label("X", "C"), unit="pixel")
    yc = gds.IntItem(roi_label("Y", "C"), unit="pixel").set_pos(1)
    _e_tlcorner = gds.EndGroup(_("Center coordinates"))
    r = gds.IntItem(_("Radius"), unit="pixel")


class ImageObj(gds.DataSet, base.BaseObj):
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
        self._maskdata_cache = None

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

    _e_tabs = gds.EndTabGroup("all")

    @property
    def xc(self) -> float:
        """Return image center X-axis coordinate"""
        return self.x0 + 0.5 * self.data.shape[1] * self.dx

    @property
    def yc(self) -> float:
        """Return image center Y-axis coordinate"""
        return self.y0 + 0.5 * self.data.shape[0] * self.dy

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
        roidataitem = RoiDataItem(self.roi[roi_index])
        return roidataitem.get_image_masked_view(self)

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
        obj.x0 = self.x0
        obj.y0 = self.y0
        obj.dx = self.dx
        obj.dy = self.dy

        # Copying metadata, but not the LUT range (which is specific to the data:
        # when processing the image, the LUT range may not be appropriate anymore):
        obj.metadata = deepcopy(self.metadata)
        obj.metadata.pop("lut_range", None)

        obj.data = np.array(self.data, copy=True, dtype=dtype)
        obj.dicom_template = self.dicom_template
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:
        """Change data type.

        Args:
            Data type
        """
        self.data = np.array(self.data, dtype=dtype)

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
        lut_range = self.metadata.get("lut_range")
        if lut_range is not None:
            item.set_lut_range(lut_range)
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
        # Storing the LUT range in metadata:
        lut_range = list(item.get_lut_range())
        self.metadata["lut_range"] = lut_range
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
            colormap="jet",
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

    def get_roi_param(self, title, *defaults) -> gds.DataSet:
        """Return ROI parameters dataset.

        Args:
            title: title
            *defaults: default values
        """
        roidataitem = RoiDataItem(defaults)
        xd0, yd0, xd1, yd1 = defaults
        if roidataitem.geometry is RoiDataGeometries.RECTANGLE:
            param = RectangleROIParam(title)
            param.x0 = xd0
            param.y0 = yd0
            param.x1 = xd1
            param.y1 = yd1
        else:
            param = CircularROIParam(title)
            param.xc = int(0.5 * (xd0 + xd1))
            param.yc = yd0
            param.r = int(0.5 * (xd1 - xd0))
        return param

    @staticmethod
    def params_to_roidata(params: gds.DataSetGroup) -> np.ndarray | None:
        """Convert ROI dataset group to ROI array data.

        Args:
            params: ROI dataset group

        Returns:
            ROI array data
        """
        roilist = []
        for roiparam in params.datasets:
            roiparam: RectangleROIParam | CircularROIParam
            roilist.append(roiparam.get_coords())
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(
        self, fmt: str, lbl: bool, editable: bool, geometry: RoiDataGeometries
    ) -> MaskedImageItem:
        """Return a new ROI item from scratch

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable
            geometry: ROI geometry
        """
        roidataitem = RoiDataItem.from_image(self, geometry)
        return roidataitem.make_roi_item(None, fmt, lbl, editable)

    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes.

        Args:
            coords: coordinates

        Returns:
            Indexes
        """
        indexes = np.array(coords)
        if indexes.size > 0:
            indexes[:, ::2] -= self.x0 + 0.5 * self.dx
            indexes[:, ::2] /= self.dx
            indexes[:, 1::2] -= self.y0 + 0.5 * self.dy
            indexes[:, 1::2] /= self.dy
        return np.array(indexes, int)

    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True) -> Iterator:
        """Make plot item representing a Region of Interest.

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable

        Yields:
            Plot item
        """
        if self.roi is not None:
            roicoords = np.array(self.roi, float)
            roicoords[:, ::2] *= self.dx
            roicoords[:, ::2] += self.x0 - 0.5 * self.dx
            roicoords[:, 1::2] *= self.dy
            roicoords[:, 1::2] += self.y0 - 0.5 * self.dy
            for index, coords in enumerate(roicoords):
                roidataitem = RoiDataItem(coords)
                yield roidataitem.make_roi_item(index, fmt, lbl, editable)

    @property
    def maskdata(self) -> np.ndarray:
        """Return masked data (areas outside defined regions of interest)

        Returns:
            Masked data
        """
        roi_changed = self.roi_has_changed()
        if self.roi is None:
            if roi_changed:
                self._maskdata_cache = None
        elif roi_changed or self._maskdata_cache is None:
            mask = np.ones_like(self.data, dtype=bool)
            for roirow in self.roi:
                roidataitem = RoiDataItem(roirow)
                roi_mask = roidataitem.apply_mask(self.data, yxratio=self.dy / self.dx)
                mask &= roi_mask
            self._maskdata_cache = mask
        return self._maskdata_cache

    def get_masked_view(self) -> ma.MaskedArray:
        """Return masked view for data

        Returns:
            Masked view
        """
        self.data: np.ndarray
        view = self.data.view(ma.MaskedArray)
        view.mask = self.maskdata
        return view

    def invalidate_maskdata_cache(self) -> None:
        """Invalidate mask data cache: force to rebuild it"""
        self._maskdata_cache = None

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

    UINT8 = enum.auto()
    UINT16 = enum.auto()
    INT16 = enum.auto()
    FLOAT32 = enum.auto()
    FLOAT64 = enum.auto()


ImageDatatypes.check()


class ImageTypes(base.Choices):
    """Image types"""

    ZEROS = _("zeros")
    EMPTY = _("empty")
    GAUSS = _("gaussian")
    UNIFORMRANDOM = _("random (uniform law)")
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
