# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT Datasets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import enum
import re
from collections import abc

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
from guidata.configtools import get_icon
from guidata.utils import update_dataset
from guiqwt.builder import make
from guiqwt.image import ImageItem

from codraft.config import Conf, _
from codraft.core.computation.image import scale_data_to_min_max
from codraft.core.model import base


def make_roi_rectangle(x0: int, y0: int, x1: int, y1: int, title: str):
    """Make and return the annnotated rectangle associated to ROI"""
    return make.annotated_rectangle(x0, y0, x1, y1, title)


def to_builtin(obj):
    """Convert an object implementing a numeric value or collection
    into the corresponding builtin/NumPy type.

    Return None if conversion fails."""
    try:
        return int(obj) if int(obj) == float(obj) else float(obj)
    except (TypeError, ValueError):
        pass
    if isinstance(obj, abc.ByteString):
        return str(obj)
    if isinstance(obj, abc.Sequence):
        return str(obj) if len(obj) == len(str(obj)) else list(obj)
    if isinstance(obj, abc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray):
        return obj
    return None


class ImageParam(gdt.DataSet, base.ObjectItf):
    """Image dataset"""

    CONF_FMT = Conf.view.ima_format
    DEFAULT_FMT = ".1f"

    def __init__(self, title=None, comment=None, icon=""):
        gdt.DataSet.__init__(self, title, comment, icon)
        self._dicom_template = None

    @property
    def size(self):
        """Returns (width, height)"""
        return self.data.shape[1], self.data.shape[0]

    def __add_metadata(self, key, value):
        """Add value to metadata if value can be converted into builtin/NumPy type"""
        stored_val = to_builtin(value)
        if stored_val is not None:
            self.metadata[key] = stored_val

    def set_metadata_from(self, obj):
        """Set metadata from object: dict-like (only string keys are considered)
        or any other object (iterating over supported attributes)"""
        self.metadata = {}
        ptn = r"__[\S_]*__$"
        if isinstance(obj, abc.Mapping):
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

    _tabs = gdt.BeginTabGroup("all")

    _datag = gdt.BeginGroup(_("Data and metadata"))
    data = gdi.FloatArrayItem(_("Data"))
    metadata = base.MetadataItem(_("Metadata"), default={})
    _e_datag = gdt.EndGroup(_("Data and metadata"))

    _dxdyg = gdt.BeginGroup(_("Origin and pixel spacing"))
    _origin = gdt.BeginGroup(_("Origin"))
    x0 = gdi.FloatItem("X<sub>0</sub>", default=0.0)
    y0 = gdi.FloatItem("Y<sub>0</sub>", default=0.0).set_pos(col=1)
    _e_origin = gdt.EndGroup(_("Origin"))
    _pixel_spacing = gdt.BeginGroup(_("Pixel spacing"))
    dx = gdi.FloatItem("Δx", default=1.0, nonzero=True)
    dy = gdi.FloatItem("Δy", default=1.0, nonzero=True).set_pos(col=1)
    _e_pixel_spacing = gdt.EndGroup(_("Pixel spacing"))
    _e_dxdyg = gdt.EndGroup(_("Origin and pixel spacing"))

    _unitsg = gdt.BeginGroup(_("Titles and units"))
    title = gdi.StringItem(_("Image title"), default=_("Untitled"))
    _tabs_u = gdt.BeginTabGroup("units")
    _unitsx = gdt.BeginGroup(_("X-axis"))
    xlabel = gdi.StringItem(_("Title"))
    xunit = gdi.StringItem(_("Unit"))
    _e_unitsx = gdt.EndGroup(_("X-axis"))
    _unitsy = gdt.BeginGroup(_("Y-axis"))
    ylabel = gdi.StringItem(_("Title"))
    yunit = gdi.StringItem(_("Unit"))
    _e_unitsy = gdt.EndGroup(_("Y-axis"))
    _unitsz = gdt.BeginGroup(_("Z-axis"))
    zlabel = gdi.StringItem(_("Title"))
    zunit = gdi.StringItem(_("Unit"))
    _e_unitsz = gdt.EndGroup(_("Z-axis"))
    _e_tabs_u = gdt.EndTabGroup("units")
    _e_unitsg = gdt.EndGroup(_("Titles and units"))

    _e_tabs = gdt.EndTabGroup("all")

    def get_data(self, roi_index: int = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).
        """
        if self.roi is None or roi_index is None:
            return self.data
        x0, y0, x1, y1 = self.roi[roi_index]
        return self.data[y0:y1, x0:x1]

    def copy_data_from(self, other, dtype=None):
        """Copy data from other dataset instance"""
        self.x0 = other.x0
        self.y0 = other.y0
        self.dx = other.dx
        self.dy = other.dy
        self.metadata = other.metadata.copy()
        self.data = np.array(other.data, copy=True, dtype=dtype)
        self.dicom_template = other.dicom_template

    def set_data_type(self, dtype):
        """Change data type"""
        self.data = np.array(self.data, dtype=dtype)

    def __viewable_data(self):
        """Return viewable data"""
        data = self.data.real
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, posinf=0, neginf=0)
        return data

    def make_item(self, update_from=None):
        """Make plot item from data"""
        data = self.__viewable_data()
        item = make.image(
            data,
            title=self.title,
            colormap="jet",
            eliminate_outliers=0.1,
            interpolation="nearest",
        )
        if update_from is not None:
            update_dataset(item.imageparam, update_from.imageparam)
            item.imageparam.update_image(item)
        return item

    def update_item(self, item: ImageItem):
        """Update plot item from data"""
        data = self.__viewable_data()
        item.set_data(data, lut_range=[item.min, item.max])
        item.imageparam.label = self.title
        for axis in ("x", "y", "z"):
            unit = getattr(self, axis + "unit")
            fmt = r"%.1f"
            if unit:
                fmt = r"%.1f (" + unit + ")"
            setattr(item.imageparam, axis + "format", fmt)

        # Updating origin and pixel spacing
        has_origin = self.x0 is not None and self.y0 is not None
        has_pixelspacing = self.dx is not None and self.dy is not None
        if has_origin or has_pixelspacing:
            x0, y0, dx, dy = 0.0, 0.0, 1.0, 1.0
            if has_origin:
                x0, y0 = self.x0, self.y0
            if has_pixelspacing:
                dx, dy = self.dx, self.dy
            item.imageparam.xmin, item.imageparam.xmax = x0, x0 + dx * data.shape[1]
            item.imageparam.ymin, item.imageparam.ymax = y0, y0 + dy * data.shape[0]

        update_dataset(item.imageparam, self.metadata)
        item.imageparam.update_image(item)

        item.plot().update_colormap_axis(item)

    def get_roi_param(self, title, *defaults):
        """Return ROI parameters dataset"""
        shape = self.data.shape
        xd0, yd0, xd1, yd1 = defaults
        xd0, yd0 = max(0, xd0), max(0, yd0)
        ymax, xmax = shape[0] - 1, shape[1] - 1
        xd1, yd1 = min(xmax, xd1), min(ymax, yd1)

        class ROIParam(gdt.DataSet):
            """ROI parameters"""

            _tlcorner = gdt.BeginGroup(_("Top left corner"))
            x0 = gdi.IntItem("X<sub>0</sub>", default=xd0, min=-1, max=xmax)
            y0 = gdi.IntItem("Y<sub>0</sub>", default=yd0, min=-1, max=ymax).set_pos(1)
            _e_tlcorner = gdt.EndGroup(_("Top left corner"))
            _brcorner = gdt.BeginGroup(_("Bottom right corner"))
            x1 = gdi.IntItem("X<sub>1</sub>", default=xd1, min=-1, max=xmax)
            y1 = gdi.IntItem("Y<sub>1</sub>", default=yd1, min=-1, max=ymax).set_pos(1)
            _e_brcorner = gdt.EndGroup(_("Bottom right corner"))

        return ROIParam(title)

    @staticmethod
    def params_to_roidata(params: gdt.DataSetGroup) -> np.ndarray:
        """Convert list of dataset parameters to ROI data"""
        roilist = []
        for roiparam in params.datasets:
            roilist.append([roiparam.x0, roiparam.y0, roiparam.x1, roiparam.y1])
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(self, fmt, lbl, editable):
        """Return a new ROI item from scratch"""
        coords = self.x0, self.y0, self.size[0] + self.x0, self.size[1] + self.y0
        return self.make_roi_item(make_roi_rectangle, coords, "ROI", fmt, lbl, editable)

    def roi_indexes_to_coords(self) -> np.ndarray:
        """Convert ROI indexes to coordinates"""
        coords = np.array(self.roi, float)
        coords[:, ::2] += self.x0
        coords[:, 1::2] += self.y0
        return coords

    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes"""
        indexes = np.array(coords)
        if indexes.size > 0:
            indexes[:, ::2] -= self.x0
            indexes[:, 1::2] -= self.y0
        return np.array(indexes, int)

    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True):
        """Iterate over plot items representing Regions of Interest"""
        if self.roi is None:
            yield self.new_roi_item(fmt, lbl, editable)
        else:
            for index, coords in enumerate(self.roi_indexes_to_coords()):
                yield self.make_roi_item(
                    make_roi_rectangle, coords, f"ROI{index:02d}", fmt, lbl, editable
                )


def create_image(
    title,
    data: np.ndarray = None,
    metadata: dict = None,
    units: tuple = None,
    labels: tuple = None,
):
    """Create a new Image object

    :param str title: image title
    :param numpy.ndarray data: image data
    :param dict metadata: image metadata
    :param tuple units: X, Y, Z units (tuple of strings)
    :param tuple labels: X, Y, Z labels (tuple of strings)
    """
    assert isinstance(title, str)
    assert data is None or isinstance(data, np.ndarray)
    image = ImageParam()
    image.title = title
    image.data = data
    if units is not None:
        image.xunit, image.yunit, image.zunit = units
    if labels is not None:
        image.xlabel, image.ylabel, image.zlabel = labels
    image.metadata = {} if metadata is None else metadata
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


class ImageParamNew(gdt.DataSet):
    """New image dataset"""

    title = gdi.StringItem(_("Title"), default=_("Untitled"))
    height = gdi.IntItem(
        _("Height"), help=_("Image height (total number of rows)"), min=1, default=500
    )
    width = gdi.IntItem(
        _("Width"), help=_("Image width (total number of columns)"), min=1, default=500
    )
    dtype = gdi.ChoiceItem(
        _("Data type"), ImageDatatypes.get_choices(), default=ImageDatatypes.UINT16
    )
    type = gdi.ChoiceItem(_("Type"), ImageTypes.get_choices())


def new_image_param(title=None, itype=None, height=None, width=None, dtype=None):
    """Create a new Image dataset instance.

    :param str title: dataset title (default: None, uses default title)"""
    if title is None:
        title = _("Untitled image")
    param = ImageParamNew(title=title, icon=get_icon("new_image.svg"))
    param.title = title
    if height is not None:
        param.height = height
    if width is not None:
        param.width = width
    if dtype is not None:
        param.dtype = dtype
    if itype is not None:
        param.type = itype
    return param


IMG_NB = 0


class Gauss2DParam(gdt.DataSet):
    """2D Gaussian parameters"""

    a = gdi.FloatItem("Norm")
    xmin = gdi.FloatItem("Xmin", default=-10).set_pos(col=1)
    sigma = gdi.FloatItem("σ", default=1.0)
    xmax = gdi.FloatItem("Xmax", default=10).set_pos(col=1)
    mu = gdi.FloatItem("μ", default=0.0)
    ymin = gdi.FloatItem("Ymin", default=-10).set_pos(col=1)
    x0 = gdi.FloatItem("X0", default=0)
    ymax = gdi.FloatItem("Ymax", default=10).set_pos(col=1)
    y0 = gdi.FloatItem("Y0", default=0).set_pos(col=0, colspan=1)


def create_image_from_param(newparam, addparam=None, edit=False, parent=None):
    """Create a new Image object from dialog box.

    :param ImageParamNew param: new image parameters
    :param guidata.dataset.datatypes.DataSet addparam: additional parameters
    :param bool edit: Open a dialog box to edit parameters (default: False)
    :param QWidget parent: parent widget
    """
    global IMG_NB  # pylint: disable=global-statement
    if newparam is None:
        newparam = new_image_param()
    incr_sig_nb = not newparam.title
    if incr_sig_nb:
        newparam.title = f"{newparam.title} {IMG_NB + 1:d}"
    if not edit or addparam is not None or newparam.edit(parent=parent):
        if incr_sig_nb:
            IMG_NB += 1
        image = create_image(newparam.title)
        shape = (newparam.height, newparam.width)
        dtype = newparam.dtype.value
        p = addparam
        if newparam.type == ImageTypes.ZEROS:
            image.data = np.zeros(shape, dtype=dtype)
        elif newparam.type == ImageTypes.EMPTY:
            image.data = np.empty(shape, dtype=dtype)
        elif newparam.type == ImageTypes.GAUSS:
            if p is None:
                p = Gauss2DParam(_("New 2D-gaussian image"))
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
        elif newparam.type in (ImageTypes.UNIFORMRANDOM, ImageTypes.NORMALRANDOM):
            pclass = {
                ImageTypes.UNIFORMRANDOM: base.UniformRandomParam,
                ImageTypes.NORMALRANDOM: base.NormalRandomParam,
            }[newparam.type]
            if p is None:
                p = pclass(_("Image") + " - " + newparam.type.value)
                p.set_from_datatype(dtype)
            if edit and not p.edit(parent=parent):
                return None
            rng = np.random.default_rng(p.seed)
            if newparam.type == ImageTypes.UNIFORMRANDOM:
                data = rng.random(shape)
                image.data = scale_data_to_min_max(data, p.vmin, p.vmax)
            elif newparam.type == ImageTypes.NORMALRANDOM:
                image.data = rng.normal(p.mu, p.sigma, size=shape)
            else:
                raise NotImplementedError(f"New param type: {newparam.type.value}")
        return image
    return None
