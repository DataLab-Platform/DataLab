# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Datasets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import enum
import json
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
import numpy as np
from guidata.dataset import update_dataset
from guidata.dataset.io import JSONHandler, JSONReader, JSONWriter
from plotpy.builder import make
from plotpy.io import load_items, save_items
from plotpy.items import AnnotatedPoint, AnnotatedShape, LabelItem

from cdl.algorithms import coordinates
from cdl.algorithms.datatypes import is_integer_dtype
from cdl.config import Conf, _

if TYPE_CHECKING:
    from plotpy.items import CurveItem, MaskedImageItem

ROI_KEY = "_roi_"
ANN_KEY = "_ann_"


@enum.unique
class Choices(enum.Enum):
    """Object associating an enum to guidata.dataset.ChoiceItem choices"""

    # Reimplement enum.Enum method as suggested by Python documentation:
    # https://docs.python.org/3/library/enum.html#enum.Enum._generate_next_value_
    # Here, it is only needed for ImageDatatypes (see core/model/image.py).
    # pylint: disable=unused-argument,no-self-argument,no-member
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    @classmethod
    def get_choices(cls):
        """Return tuple of (key, value) choices to be used as parameter of
        guidata.dataset.ChoiceItem"""
        return tuple((member, member.value) for member in cls)


class BaseProcParam(gds.DataSet):
    """Base class for processing parameters"""

    def __init__(self, title=None, comment=None, icon=""):
        super().__init__(title, comment, icon)
        self.set_global_prop("data", min=None, max=None)

    def apply_integer_range(self, vmin, vmax):  # pylint: disable=unused-argument
        """Do something in case of integer min-max range"""

    def apply_float_range(self, vmin, vmax):  # pylint: disable=unused-argument
        """Do something in case of float min-max range"""

    def set_from_datatype(self, dtype):
        """Set min/max range from NumPy datatype"""
        if is_integer_dtype(dtype):
            info = np.iinfo(dtype)
            self.apply_integer_range(info.min, info.max)
        else:
            info = np.finfo(dtype)
            self.apply_float_range(info.min, info.max)
        self.set_global_prop("data", min=info.min, max=info.max)


class BaseRandomParam(BaseProcParam):
    """Random signal/image parameters"""

    seed = gds.IntItem(_("Seed"), default=1)


class UniformRandomParam(BaseRandomParam):
    """Uniform-law random signal/image parameters"""

    def apply_integer_range(self, vmin, vmax):
        """Do something in case of integer min-max range"""
        self.vmin, self.vmax = vmin, vmax

    vmin = gds.FloatItem(
        "V<sub>min</sub>", default=-0.5, help=_("Uniform distribution lower bound")
    )
    vmax = gds.FloatItem(
        "V<sub>max</sub>", default=0.5, help=_("Uniform distribution higher bound")
    ).set_pos(col=1)


class NormalRandomParam(BaseRandomParam):
    """Normal-law random signal/image parameters"""

    DEFAULT_RELATIVE_MU = 0.1
    DEFAULT_RELATIVE_SIGMA = 0.02

    def apply_integer_range(self, vmin, vmax):
        """Do something in case of integer min-max range"""
        delta = vmax - vmin
        self.mu = int(self.DEFAULT_RELATIVE_MU * delta + vmin)
        self.sigma = int(self.DEFAULT_RELATIVE_SIGMA * delta)

    mu = gds.FloatItem(
        "μ", default=DEFAULT_RELATIVE_MU, help=_("Normal distribution mean")
    )
    sigma = gds.FloatItem(
        "σ",
        default=DEFAULT_RELATIVE_SIGMA,
        help=_("Normal distribution standard deviation"),
    ).set_pos(col=1)


@enum.unique
class ShapeTypes(enum.Enum):
    """Shape types for image metadata"""

    # Reimplement enum.Enum method as suggested by Python documentation:
    # https://docs.python.org/3/library/enum.html#enum.Enum._generate_next_value_
    # Here, it is only needed for ImageDatatypes (see core/model/image.py).
    # pylint: disable=unused-argument,no-self-argument,no-member
    def _generate_next_value_(name, start, count, last_values):
        return f"_{name.lower()[:3]}_"

    RECTANGLE = enum.auto()
    CIRCLE = enum.auto()
    ELLIPSE = enum.auto()
    SEGMENT = enum.auto()
    MARKER = enum.auto()
    POINT = enum.auto()
    POLYGON = enum.auto()


def config_annotated_shape(
    item: AnnotatedShape, fmt: str, lbl: bool, option: str, cmp: bool | None = None
):
    """Configurate annotated shape.

    Args:
        item: Annotated shape item
        fmt: Format string
        lbl: Show label
        option: Shape style option (e.g. "shape/drag")
        cmp: Show computations
    """
    param = item.annotationparam
    param.format = fmt
    param.show_label = lbl
    if cmp is not None:
        param.show_computations = cmp
    param.update_annotation(item)
    item.set_style("plot", option)


def set_plot_item_editable(item, state):
    """Set plot item editable state.

    Args:
        item: Plot item
        state: State
    """
    item.set_movable(state)
    item.set_resizable(state)
    item.set_rotatable(state)
    item.set_readonly(not state)


class ResultShape:
    """Object representing a geometrical shape serializable in signal/image metadata.

    Result `array` is a NumPy 2-D array: each row is a result, optionnally associated
    to a ROI (first column value).

    ROI index is starting at 0 (or is simply 0 if there is no ROI).

    Args:
        shapetype: shape type
        array: shape coordinates (multiple shapes: one shape per row),
         first column is ROI index (0 if there is no ROI)
        label: shape label

    Raises:
        AssertionError: invalid argument
    """

    def __init__(self, shapetype: ShapeTypes, array: np.ndarray, label: str = ""):
        assert isinstance(label, str)
        assert isinstance(shapetype, ShapeTypes)
        self.label = self.show_label = label
        self.shapetype = shapetype
        if isinstance(array, (list, tuple)):
            if isinstance(array[0], (list, tuple)):
                array = np.array(array)
            else:
                array = np.array([array])
        assert isinstance(array, np.ndarray)
        self.array = array
        if label.endswith("s"):
            self.show_label = label[:-1]
        self.check_array()

    @classmethod
    def label_shapetype_from_key(cls, key: str):
        """Return metadata shape label and shapetype from metadata key"""
        for member in ShapeTypes:
            if key.startswith(member.value):
                label = key[len(member.value) :]
                return label, member
        raise ValueError(f"Invalid metadata key `{key}`")

    @classmethod
    def from_metadata_entry(cls, key, value) -> ResultShape | None:
        """Create metadata shape object from (key, value) metadata entry"""
        if isinstance(key, str) and isinstance(value, np.ndarray):
            try:
                label, shapetype = cls.label_shapetype_from_key(key)
                return cls(shapetype, value, label)
            except ValueError:
                pass
        return None

    @classmethod
    def match(cls, key, value) -> bool:
        """Return True if metadata dict entry (key, value) is a metadata result"""
        return cls.from_metadata_entry(key, value) is not None

    @property
    def key(self) -> str:
        """Return metadata key associated to result"""
        return self.shapetype.value + self.label

    @property
    def shown_xlabels(self) -> tuple[str]:
        """Return labels for result array columns"""
        if self.shapetype in (ShapeTypes.MARKER, ShapeTypes.POINT):
            labels = "ROI", "x", "y"
        elif self.shapetype in (
            ShapeTypes.RECTANGLE,
            ShapeTypes.SEGMENT,
            ShapeTypes.POLYGON,
        ):
            labels = ["ROI"]
            for index in range(0, self.array.shape[1] - 1, 2):
                labels += [f"x{index//2}", f"y{index//2}"]
            labels = tuple(labels)
        elif self.shapetype is ShapeTypes.CIRCLE:
            labels = "ROI", "x", "y", "r"
        elif self.shapetype is ShapeTypes.ELLIPSE:
            labels = "ROI", "x", "y", "a", "b", "θ"
        else:
            raise NotImplementedError(f"Unsupported shapetype {self.shapetype}")
        labels += self.__get_complementary_xlabels() or ()
        return labels[-self.shown_array.shape[1] :]

    @property
    def shown_array(self) -> np.ndarray:
        """Return array of shown results, i.e. including the complementary array

        Returns:
            Array of shown results
        """
        arr = self.array
        comp_arr = self.__get_complementary_array()
        if comp_arr is None:
            return arr
        return np.hstack([arr, comp_arr])

    def __get_complementary_xlabels(self) -> tuple[str] | None:
        """Return complementary labels for result array columns

        Returns:
            Complementary labels for result array columns, or None if there is no
            complementary labels
        """
        if self.shapetype is ShapeTypes.SEGMENT:
            return ("L", "Xc", "Yc")
        if self.shapetype in (ShapeTypes.CIRCLE, ShapeTypes.ELLIPSE):
            return ("A",)
        return None

    def __get_complementary_array(self) -> np.ndarray | None:
        """Return the complementary array of results, e.g. the array of lengths
        for a segment result shape, or the array of areas for a circle result shape

        Returns:
            Complementary array of results, or None if there is no complementary array
        """
        arr = self.array
        if self.shapetype is ShapeTypes.SEGMENT:
            dx1, dy1 = arr[:, 3] - arr[:, 1], arr[:, 4] - arr[:, 2]
            length = np.linalg.norm(np.vstack([dx1, dy1]).T, axis=1)
            xc = (arr[:, 1] + arr[:, 3]) / 2
            yc = (arr[:, 2] + arr[:, 4]) / 2
            return np.vstack([length, xc, yc]).T
        if self.shapetype is ShapeTypes.CIRCLE:
            area = np.pi * arr[:, 3] ** 2
            return area.reshape(-1, 1)
        if self.shapetype is ShapeTypes.ELLIPSE:
            area = np.pi * arr[:, 3] * arr[:, 4]
            return area.reshape(-1, 1)
        return None

    def add_to(self, obj: BaseObj):
        """Add metadata shape to object (signal/image)"""
        obj.metadata[self.key] = self.array
        if self.shapetype in (
            ShapeTypes.SEGMENT,
            ShapeTypes.CIRCLE,
            ShapeTypes.ELLIPSE,
        ):
            #  Automatically adds segment norm / circle area to object metadata
            arr = self.array
            comp_arr = self.__get_complementary_array()
            comp_lbl = self.__get_complementary_xlabels()
            assert comp_lbl is not None and comp_arr is not None
            for index, label in enumerate(comp_lbl):
                results = np.zeros((arr.shape[0], 2), dtype=arr.dtype)
                results[:, 0] = arr[:, 0]  # ROI indexes
                results[:, 1] = comp_arr[:, index]
                obj.metadata[self.label + label] = results

    def merge_with(self, obj: BaseObj, other_obj: BaseObj | None = None):
        """Merge object resultshape with another's: obj <-- other_obj
        or simply merge this resultshape with obj if other_obj is None"""
        if other_obj is None:
            other_obj = obj
        other_value = other_obj.metadata.get(self.key)
        if other_value is not None:
            other = ResultShape.from_metadata_entry(self.key, other_value)
            assert other is not None
            other_array = np.array(other.array, copy=True)
            if other.is_first_column_roi_index():  # Column 0 is the ROI index
                other_array[:, 0] += self.array[-1, 0] + 1  # Adding ROI index offset
            if other_array.shape[1] != self.array.shape[1]:
                # This can only happen if the shape is a polygon
                assert self.shapetype is ShapeTypes.POLYGON
                # We must padd the array with NaNs
                max_colnb = max(self.array.shape[1], other_array.shape[1])
                new_array = np.full(
                    (self.array.shape[0] + other_array.shape[0], max_colnb), np.nan
                )
                new_array[: self.array.shape[0], : self.array.shape[1]] = self.array
                new_array[self.array.shape[0] :, : other_array.shape[1]] = other_array
                self.array = new_array
            else:
                self.array = np.vstack([self.array, other_array])
        self.add_to(obj)

    @property
    def data_colnb(self):
        """Return raw data results column number"""
        if self.shapetype == ShapeTypes.POLYGON:
            raise ValueError("Polygon has an undefined number of data columns")
        return {
            ShapeTypes.MARKER: 2,
            ShapeTypes.POINT: 2,
            ShapeTypes.RECTANGLE: 4,
            ShapeTypes.CIRCLE: 3,
            ShapeTypes.SEGMENT: 4,
            ShapeTypes.ELLIPSE: 5,
        }[self.shapetype]

    def is_first_column_roi_index(self) -> bool:
        """Return True if first column is ROI index"""
        if self.shapetype is ShapeTypes.POLYGON:
            # Polygon is a special case: the number of data columns is variable
            # (2 columns per point). So we only check if the number of columns
            # is odd, which means that the first column is the ROI index, followed
            # by an even number of data columns (flattened x, y coordinates).
            return self.array.shape[1] % 2 == 1
        return self.array.shape[1] == self.data_colnb + 1

    @property
    def data(self):
        """Return raw data (array without ROI informations)"""
        if self.is_first_column_roi_index():
            # Column 0 is the ROI index
            return self.array[:, 1:]
        # No ROI index
        return self.array

    def transform_coordinates(self, func: Callable[[np.ndarray], None]) -> None:
        """Transform shape coordinates.

        Args:
            func: function to transform coordinates
        """
        if self.shapetype in (
            ShapeTypes.MARKER,
            ShapeTypes.POINT,
            ShapeTypes.POLYGON,
            ShapeTypes.RECTANGLE,
            ShapeTypes.SEGMENT,
        ):
            func(self.data)
        elif self.shapetype is ShapeTypes.CIRCLE:
            coords = coordinates.array_circle_to_diameter(self.data)
            func(coords)
            self.data[:] = coordinates.array_circle_to_center_radius(coords)
        elif self.shapetype is ShapeTypes.ELLIPSE:
            coords = coordinates.array_ellipse_to_diameters(self.data)
            func(coords)
            self.data[:] = coordinates.array_ellipse_to_center_axes_angle(coords)
        else:
            raise NotImplementedError(f"Unsupported shapetype {self.shapetype}")

    def check_array(self):
        """Check if array is valid"""
        assert len(self.array.shape) == 2
        assert self.is_first_column_roi_index()

    def iterate_plot_items(self, fmt: str, lbl: bool, option: str) -> Iterable:
        """Iterate over metadata shape plot items.

        Args:
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option (e.g. "shape/drag")

        Yields:
            Plot item
        """
        for args in self.data:
            yield self.create_plot_item(args, fmt, lbl, option)

    def create_plot_item(self, args: np.ndarray, fmt: str, lbl: bool, option: str):
        """Make plot item.

        Args:
            args: shape data
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option (e.g. "shape/drag")

        Returns:
            Plot item
        """
        if self.shapetype is ShapeTypes.MARKER:
            item = self.make_marker_item(args, fmt)
        elif self.shapetype is ShapeTypes.POINT:
            item = AnnotatedPoint(*args)
            sparam = item.shape.shapeparam
            sparam.symbol.marker = "Ellipse"
            sparam.symbol.size = 6
            sparam.sel_symbol.marker = "Ellipse"
            sparam.sel_symbol.size = 6
            sparam.update_shape(item.shape)
            param = item.annotationparam
            param.title = self.show_label
            param.update_annotation(item)
        elif self.shapetype is ShapeTypes.RECTANGLE:
            x0, y0, x1, y1 = args
            item = make.annotated_rectangle(x0, y0, x1, y1, title=self.show_label)
        elif self.shapetype is ShapeTypes.CIRCLE:
            xc, yc, r = args
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.annotated_circle(x0, y0, x1, y1, title=self.show_label)
        elif self.shapetype is ShapeTypes.SEGMENT:
            x0, y0, x1, y1 = args
            item = make.annotated_segment(x0, y0, x1, y1, title=self.show_label)
        elif self.shapetype is ShapeTypes.ELLIPSE:
            xc, yc, a, b, t = args
            coords = coordinates.ellipse_to_diameters(xc, yc, a, b, t)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords
            item = make.annotated_ellipse(
                x0, y0, x1, y1, x2, y2, x3, y3, title=self.show_label
            )
        elif self.shapetype is ShapeTypes.POLYGON:
            x, y = args[::2], args[1::2]
            item = make.polygon(x, y, title=self.show_label, closed=False)
        else:
            print(f"Warning: unsupported item {self.shapetype}", file=sys.stderr)
            return None
        if isinstance(item, AnnotatedShape):
            config_annotated_shape(item, fmt, lbl, option)
        set_plot_item_editable(item, False)
        return item

    def make_marker_item(self, args, fmt):
        """Make marker item"""
        x0, y0 = args
        if np.isnan(x0):
            mstyle = "-"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.show_label + ": " + fmt) % y

        elif np.isnan(y0):
            mstyle = "|"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.show_label + ": " + fmt) % x

        else:
            mstyle = "+"
            txt = self.show_label + ": (" + fmt + ", " + fmt + ")"

            def label(x, y):
                return txt % (x, y)

        return make.marker(
            position=(x0, y0),
            markerstyle=mstyle,
            label_cb=label,
            linestyle="DashLine",
            color="yellow",
        )


def make_roi_item(
    func, coords: list, title: str, fmt: str, lbl: bool, editable: bool, option: str
):
    """Make ROI item shape.

    Args:
        func: function to create ROI item
        coords: coordinates
        title: title
        fmt: numeric format (e.g. "%.3f")
        lbl: if True, show shape labels
        editable: if True, make shape editable
        option: shape style option (e.g. "shape/drag")

    Returns:
        Plot item
    """
    item = func(*coords, title)
    if not editable:
        if isinstance(item, AnnotatedShape):
            config_annotated_shape(item, fmt, lbl, option, cmp=editable)
            item.set_style("plot", "shape/mask")
        item.set_movable(False)
        item.set_resizable(False)
        item.set_readonly(True)
    return item


def items_to_json(items: list) -> str | None:
    """Convert plot items to JSON string.

    Args:
        items: list of plot items

    Returns:
        JSON string or None if items is empty
    """
    if items:
        writer = JSONWriter(None)
        save_items(writer, items)
        return writer.get_json(indent=4)
    return None


def json_to_items(json_str: str | None) -> list:
    """Convert JSON string to plot items.

    Args:
        json_str: JSON string or None

    Returns:
        List of plot items
    """
    items = []
    if json_str:
        try:
            for item in load_items(JSONReader(json_str)):
                items.append(item)
        except json.decoder.JSONDecodeError:
            pass
    return items


class BaseObjMeta(abc.ABCMeta, gds.DataSetMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseObj(metaclass=BaseObjMeta):
    """Object (signal/image) interface"""

    PREFIX = ""  # This is overriden in children classes

    DEFAULT_FMT = "s"  # This is overriden in children classes
    CONF_FMT = Conf.view.sig_format  # This is overriden in children classes

    # This is overriden in children classes with a gds.DictItem instance:
    metadata: dict[str, Any] = {}

    VALID_DTYPES = ()

    def __init__(self):
        self.__onb = 0
        self.__roi_changed: bool | None = None
        self.__metadata_options: dict[str, Any] | None = None
        self.reset_metadata_to_defaults()

    @property
    def number(self) -> int:
        """Return object number (used for short ID)"""
        return self.__onb

    @number.setter
    def number(self, onb: int):
        """Set object number (used for short ID).

        Args:
            onb: object number
        """
        self.__onb = onb

    @property
    def short_id(self):
        """Short object ID"""
        return f"{self.PREFIX}{self.__onb:03d}"

    @property
    @abc.abstractmethod
    def data(self):
        """Data"""

    @classmethod
    def get_valid_dtypenames(cls) -> list[str]:
        """Get valid data type names

        Returns:
            Valid data type names supported by this class
        """
        return [
            dtname
            for dtname in np.sctypeDict
            if dtname in (dtype.__name__ for dtype in cls.VALID_DTYPES)
        ]

    def check_data(self):
        """Check if data is valid, raise an exception if that's not the case

        Raises:
            TypeError: if data type is not supported
        """
        if self.data is not None:
            if self.data.dtype not in self.VALID_DTYPES:
                raise TypeError(f"Unsupported data type: {self.data.dtype}")

    def iterate_roi_indexes(self):
        """Iterate over object ROI indexes ([0] if there is no ROI)"""
        if self.roi is None:
            yield 0
        else:
            yield from range(len(self.roi))

    @abc.abstractmethod
    def get_data(self, roi_index: int | None = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Data
        """

    @abc.abstractmethod
    def copy(self, title: str | None = None, dtype: np.dtype | None = None) -> BaseObj:
        """Copy object.

        Args:
            title: title
            dtype: data type

        Returns:
            Copied object
        """

    @abc.abstractmethod
    def set_data_type(self, dtype):
        """Change data type.

        Args:
            dtype: data type
        """

    @abc.abstractmethod
    def make_item(self, update_from=None):
        """Make plot item from data.

        Args:
            update_from: update

        Returns:
            Plot item
        """

    @abc.abstractmethod
    def update_item(self, item, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """

    @abc.abstractmethod
    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes.

        Args:
            coords: coordinates

        Returns:
            Indexes
        """

    @abc.abstractmethod
    def get_roi_param(self, title, *defaults):
        """Return ROI parameters dataset.

        Args:
            title: title
            *defaults: default values
        """

    def roidata_to_params(self, roidata: np.ndarray) -> gds.DataSetGroup:
        """Convert ROI array data to ROI dataset group.

        Args:
            roidata: ROI array data

        Returns:
            ROI dataset group
        """
        roi_params = []
        for index, parameters in enumerate(roidata):
            roi_param = self.get_roi_param(f"ROI{index:02d}", *parameters)
            roi_params.append(roi_param)
        group = gds.DataSetGroup(roi_params, title=_("Regions of interest"))
        return group

    @staticmethod
    @abc.abstractmethod
    def params_to_roidata(params: gds.DataSetGroup) -> np.ndarray:
        """Convert ROI dataset group to ROI array data.

        Args:
            params: ROI dataset group

        Returns:
            ROI array data
        """

    def roi_has_changed(self) -> bool:
        """Return True if ROI has changed since last call to this method.

        The first call to this method will return True if ROI has not yet been set,
        or if ROI has been set and has changed since the last call to this method.
        The next call to this method will always return False if ROI has not changed
        in the meantime.

        Returns:
            True if ROI has changed
        """
        if self.__roi_changed is None:
            self.__roi_changed = True
        returned_value = self.__roi_changed
        self.__roi_changed = False
        return returned_value

    @property
    def roi(self) -> np.ndarray | None:
        """Return object regions of interest array (one ROI per line).

        Returns:
            Regions of interest array
        """
        roidata = self.metadata.get(ROI_KEY)
        assert roidata is None or isinstance(roidata, np.ndarray)
        return roidata

    @roi.setter
    def roi(self, roidata: np.ndarray):
        """Set object regions of interest array, using a list or ROI dataset params.

        Args:
            roidata: regions of interest array
        """
        if roidata is None:
            if ROI_KEY in self.metadata:
                self.metadata.pop(ROI_KEY)
        else:
            self.metadata[ROI_KEY] = np.array(roidata, int)
        self.__roi_changed = True

    def add_resultshape(
        self,
        label: str,
        shapetype: ShapeTypes,
        array: np.ndarray,
        param: gds.DataSet | None = None,
    ) -> ResultShape:
        """Add geometric shape as metadata entry, and return ResultShape instance.

        Args:
            label: label
            shapetype: shape type
            array: array
            param: parameters

        Returns:
            Result shape
        """
        mshape = ResultShape(shapetype, array, label)
        mshape.add_to(self)
        if param is not None:
            self.metadata[f"{label}Param"] = str(param)
        return mshape

    def iterate_resultshapes(self):
        """Iterate over object result shapes.

        Yields:
            Result shape
        """
        for key, value in self.metadata.items():
            if ResultShape.match(key, value):
                yield ResultShape.from_metadata_entry(key, value)

    def update_resultshapes_from(self, other: BaseObj) -> None:
        """Update geometric shape from another object (merge metadata).

        Args:
            other: other object, from which to update this object
        """
        # The following code is merging the result shapes of the `other` object
        # with the result shapes of this object, but it is merging only the result
        # shapes of the same type (`mshape.key`). Thus, if the `other` object has
        # a result shape that is not present in this object, it will not be merged,
        # and we will have to add it to this object manually.
        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            mshape.merge_with(self, other)
        # Iterating on `other` object result shapes to find result shapes that are
        # not present in this object, and add them to this object.
        for mshape in other.iterate_resultshapes():
            assert mshape is not None
            if mshape.key not in self.metadata:
                mshape.add_to(self)

    def transform_shapes(self, orig, func, param=None):
        """Apply transform function to result shape / annotations coordinates.

        Args:
            orig: original object
            func: transform function
            param: transform function parameter
        """

        def transform(coords: np.ndarray):
            """Transform coordinates"""
            if param is None:
                func(self, orig, coords)
            else:
                func(self, orig, coords, param)

        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            mshape.transform_coordinates(transform)
        items = []
        for item in json_to_items(self.annotations):
            if isinstance(item, AnnotatedShape):
                transform(item.shape.points)
                item.set_label_position()
            elif isinstance(item, LabelItem):
                x, y = item.G
                points = np.array([[x, y]], float)
                transform(points)
                x, y = points[0]
                item.set_pos(x, y)
            items.append(item)
        if items:
            self.annotations = items_to_json(items)

    @abc.abstractmethod
    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True):
        """Make plot item representing a Region of Interest.

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable

        Yields:
            Plot item
        """

    def __set_annotations(self, annotations: str | None) -> None:
        """Set object annotations (JSON string describing annotation plot items)

        Args:
            annotations: JSON string describing annotation plot items,
             or None to remove annotations
        """
        if annotations is None:
            if ANN_KEY in self.metadata:
                self.metadata.pop(ANN_KEY)
        else:
            self.metadata[ANN_KEY] = annotations

    def __get_annotations(self) -> str:
        """Get object annotations (JSON string describing annotation plot items)"""
        return self.metadata.get(ANN_KEY, "")

    annotations = property(__get_annotations, __set_annotations)

    def set_annotations_from_file(self, filename: str) -> None:
        """Set object annotations from file (JSON).

        Args:
            filename: filename
        """
        with open(filename, mode="rb") as fdesc:
            self.annotations = fdesc.read().decode()

    def add_annotations_from_items(self, items: list) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
        """
        ann_items = json_to_items(self.annotations)
        ann_items.extend(items)
        if ann_items:
            self.annotations = items_to_json(ann_items)

    def add_annotations_from_file(self, filename: str) -> None:
        """Add object annotations from file (JSON).

        Args:
            filename: filename
        """
        items = load_items(JSONReader(filename))
        self.add_annotations_from_items(items)

    @abc.abstractmethod
    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use object title)
        """

    def iterate_shape_items(self, editable: bool = False):
        """Iterate over computing items encoded in metadata (if any).

        Args:
            editable: if True, ROI is editable

        Yields:
            Plot item
        """
        fmt = self.get_metadata_option("format")
        lbl = self.get_metadata_option("showlabel")
        for key, value in self.metadata.items():
            if key == ROI_KEY:
                yield from self.iterate_roi_items(fmt=fmt, lbl=lbl, editable=False)
            elif ResultShape.match(key, value):
                mshape = ResultShape.from_metadata_entry(key, value)
                yield from mshape.iterate_plot_items(
                    fmt, lbl, f"shape/result/{self.PREFIX}"
                )
        if self.annotations:
            try:
                for item in load_items(JSONReader(self.annotations)):
                    set_plot_item_editable(item, editable)
                    if isinstance(item, AnnotatedShape):
                        config_annotated_shape(item, fmt, lbl, "shape/annotation")
                    yield item
            except json.decoder.JSONDecodeError:
                pass

    def remove_all_shapes(self) -> None:
        """Remove metadata shapes and ROIs"""
        for key, value in list(self.metadata.items()):
            resultshape = ResultShape.from_metadata_entry(key, value)
            if resultshape is not None or key == ROI_KEY:
                # Metadata entry is a metadata shape or a ROI
                self.metadata.pop(key)
        self.annotations = None

    def get_metadata_option(self, name: str) -> Any:
        """Return metadata option value

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name

        Returns:
            Option value

        Valid option names:
            'format': format string
            'showlabel': show label
        """
        if name not in self.__metadata_options:
            raise ValueError(f"Invalid metadata option name `{name}`")
        default = self.__metadata_options[name]
        return self.metadata.get(f"__{name}", default)

    def set_metadata_option(self, name: str, value: Any) -> None:
        """Set metadata option value

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name
            value: option value

        Valid option names:
            'format': format string
            'showlabel': show label
        """
        if name not in self.__metadata_options:
            raise ValueError(f"Invalid metadata option name `{name}`")
        self.metadata[f"__{name}"] = value

    def reset_metadata_to_defaults(self) -> None:
        """Reset metadata to default values"""
        self.__metadata_options = {
            "format": "%" + self.CONF_FMT.get(self.DEFAULT_FMT),
            "showlabel": Conf.view.show_label.get(False),
        }
        self.metadata = {}
        for name, value in self.__metadata_options.items():
            self.set_metadata_option(name, value)
        self.update_metadata_view_settings()

    def __get_def_dict(self) -> dict[str, Any]:
        """Return default visualization settings dictionary"""
        return Conf.view.get_def_dict(self.__class__.__name__[:3].lower())

    def update_metadata_view_settings(self) -> None:
        """Update metadata view settings from Conf.view"""
        self.metadata.update(self.__get_def_dict())

    def update_plot_item_parameters(self, item: CurveItem | MaskedImageItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        # Subclasses have to override this method to update plot item parameters,
        # then call this implementation of the method to update plot item.
        update_dataset(item.param, self.metadata)
        item.param.update_item(item)
        if item.selected:
            item.select()

    def update_metadata_from_plot_item(self, item: CurveItem | MaskedImageItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        for key in self.__get_def_dict():
            if hasattr(item.param, key):  # In case the PlotPy version is not up-to-date
                self.metadata[key] = getattr(item.param, key)
        # Subclasses may override this method to update metadata from plot item,
        # then call this implementation of the method to update metadata standard
        # entries.

    def export_metadata_to_file(self, filename: str) -> None:
        """Export object metadata to file (JSON).

        Args:
            filename: filename
        """
        handler = JSONHandler(filename)
        handler.set_json_dict(self.metadata)
        handler.save()

    def import_metadata_from_file(self, filename: str) -> None:
        """Import object metadata from file (JSON).

        Args:
            filename: filename
        """
        handler = JSONHandler(filename)
        handler.load()
        self.metadata = handler.get_json_dict()

    def metadata_to_html(self) -> str:
        """Convert metadata to human-readable string.

        Returns:
            HTML string
        """
        textlines = []
        for key, value in self.metadata.items():
            if len(textlines) > 5:
                textlines.append("[...]")
                break
            if not key.startswith("_"):
                vlines = str(value).splitlines()
                if vlines:
                    text = f"<b>{key}:</b> {vlines[0]}"
                    if len(vlines) > 1:
                        text += " [...]"
                    textlines.append(text)
        if textlines:
            ptit = _("Object metadata")
            psub = _("(click on Metadata button for more details)")
            prefix = f"<i><u>{ptit}:</u> {psub}</i><br>"
            return f"<p style='white-space:pre'>{prefix}{'<br>'.join(textlines)}</p>"
        return ""
