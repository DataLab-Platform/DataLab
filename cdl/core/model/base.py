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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
import numpy as np
from guidata.dataset import update_dataset
from guidata.dataset.io import JSONHandler, JSONReader, JSONWriter
from plotpy.builder import make
from plotpy.io import load_items, save_items
from plotpy.items import AnnotatedPoint, AnnotatedShape, LabelItem

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
        item (AnnotatedShape): Annotated shape item
        fmt (str): Format string
        lbl (bool): Show label
        option (str): Shape style option (e.g. "shape/drag")
        cmp (bool | None): Show computations
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
        item (PlotItem): Plot item
        state (bool): State
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
    def xlabels(self) -> tuple[str]:
        """Return labels for result array columns"""
        if self.shapetype in (ShapeTypes.MARKER, ShapeTypes.POINT):
            labels = "ROI", "x", "y"
        elif self.shapetype in (
            ShapeTypes.RECTANGLE,
            ShapeTypes.CIRCLE,
            ShapeTypes.SEGMENT,
            ShapeTypes.ELLIPSE,
            ShapeTypes.POLYGON,
        ):
            labels = ["ROI"]
            for index in range(0, self.array.shape[1] - 1, 2):
                labels += [f"x{index//2}", f"y{index//2}"]
            labels = tuple(labels)
        else:
            raise NotImplementedError(f"Unsupported shapetype {self.shapetype}")
        return labels[-self.array.shape[1] :]

    def add_to(self, obj: BaseObj):
        """Add metadata shape to object (signal/image)"""
        obj.metadata[self.key] = self.array
        if self.shapetype in (
            ShapeTypes.SEGMENT,
            ShapeTypes.CIRCLE,
            ShapeTypes.ELLIPSE,
        ):
            #  Automatically adds segment norm / circle diameter to object metadata
            colnb = 2
            if self.shapetype is ShapeTypes.ELLIPSE:
                colnb += 1
            arr = self.array
            results = np.zeros((arr.shape[0], colnb), dtype=arr.dtype)
            results[:, 0] = arr[:, 0]  # ROI indexes
            dx1, dy1 = arr[:, 3] - arr[:, 1], arr[:, 4] - arr[:, 2]
            results[:, 1] = np.linalg.norm(np.vstack([dx1, dy1]).T, axis=1)
            if self.shapetype is ShapeTypes.ELLIPSE:
                dx2, dy2 = arr[:, 7] - arr[:, 5], arr[:, 8] - arr[:, 6]
                results[:, 2] = np.linalg.norm(np.vstack([dx2, dy2]).T, axis=1)
            label = self.label
            if self.shapetype is ShapeTypes.CIRCLE:
                label += "Diameter"
            if self.shapetype is ShapeTypes.ELLIPSE:
                label += "Diameters"
            obj.metadata[label] = results

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
            if other_array.shape[1] % 2:  # Column 0 is the ROI index
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
            ShapeTypes.CIRCLE: 4,
            ShapeTypes.SEGMENT: 4,
            ShapeTypes.ELLIPSE: 8,
        }[self.shapetype]

    @property
    def data(self):
        """Return raw data (array without ROI informations)"""
        if self.array.shape[1] % 2:
            # Column 0 is the ROI index
            return self.array[:, 1:]
        # No ROI index
        return self.array

    def check_array(self):
        """Check if array is valid"""
        assert len(self.array.shape) == 2
        if self.shapetype is ShapeTypes.POLYGON:
            # Polygon is a special case: the number of data columns is variable
            # (2 columns per point). So we only check if the number of columns
            # is odd, which means that the first column is the ROI index, followed
            # by an even number of data columns (flattened x, y coordinates).
            assert self.array.shape[1] % 2 == 1
        else:
            assert self.array.shape[1] == self.data_colnb + 1

    def iterate_plot_items(self, fmt: str, lbl: bool, option: str) -> Iterable:
        """Iterate over metadata shape plot items.

        Args:
            fmt (str): numeric format (e.g. "%.3f")
            lbl (bool): if True, show shape labels
            option (str): shape style option (e.g. "shape/drag")

        Yields:
            PlotItem: plot item
        """
        for args in self.data:
            yield self.create_plot_item(args, fmt, lbl, option)

    def create_plot_item(self, args: np.ndarray, fmt: str, lbl: bool, option: str):
        """Make plot item.

        Args:
            args (numpy.ndarray): shape data
            fmt (str): numeric format (e.g. "%.3f")
            lbl (bool): if True, show shape labels
            option (str): shape style option (e.g. "shape/drag")

        Returns:
            PlotItem: plot item
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
            x0, y0, x1, y1 = args
            item = make.annotated_circle(x0, y0, x1, y1, title=self.show_label)
        elif self.shapetype is ShapeTypes.SEGMENT:
            x0, y0, x1, y1 = args
            item = make.annotated_segment(x0, y0, x1, y1, title=self.show_label)
        elif self.shapetype is ShapeTypes.ELLIPSE:
            x0, y0, x1, y1, x2, y2, x3, y3 = args
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
        func (function): function to create ROI item
        coords (list): coordinates
        title (str): title
        fmt (str): numeric format (e.g. "%.3f")
        lbl (bool): if True, show shape labels
        editable (bool): if True, make shape editable
        option (str): shape style option (e.g. "shape/drag")

    Returns:
        PlotItem: plot item
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
        items (list): list of plot items

    Returns:
        str: JSON string or None if items is empty
    """
    if items:
        writer = JSONWriter(None)
        save_items(writer, items)
        return writer.get_json(indent=4)
    return None


def json_to_items(json_str: str | None) -> list:
    """Convert JSON string to plot items.

    Args:
        json_str (str): JSON string or None

    Returns:
        list: list of plot items
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
            onb (int): object number
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
    def get_valid_dtypenames(cls) -> tuple[str]:
        """Get valid data type names

        Returns:
            Valid data type names supported by this class
        """
        return tuple(
            [
                dtname
                for dtname in np.sctypeDict.keys()
                if dtname in (dtype.__name__ for dtype in cls.VALID_DTYPES)
            ]
        )

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
            roi_index (int): ROI index

        Returns:
            numpy.ndarray: data
        """

    @abc.abstractmethod
    def copy(self, title: str | None = None, dtype: np.dtype | None = None) -> BaseObj:
        """Copy object.

        Args:
            title (str): title
            dtype (numpy.dtype): data type

        Returns:
            BaseObj: copied object
        """

    @abc.abstractmethod
    def set_data_type(self, dtype):
        """Change data type.

        Args:
            dtype (numpy.dtype): data type
        """

    @abc.abstractmethod
    def make_item(self, update_from=None):
        """Make plot item from data.

        Args:
            update_from (BaseObj): update

        Returns:
            PlotItem: plot item
        """

    @abc.abstractmethod
    def update_item(self, item, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item (PlotItem): plot item
            data_changed (bool): if True, data has changed
        """

    @abc.abstractmethod
    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes.

        Args:
            coords (list): coordinates

        Returns:
            numpy.ndarray: indexes
        """

    @abc.abstractmethod
    def get_roi_param(self, title, *defaults):
        """Return ROI parameters dataset.

        Args:
            title (str): title
            *defaults: default values
        """

    def roidata_to_params(self, roidata: np.ndarray) -> gds.DataSetGroup:
        """Convert ROI array data to ROI dataset group.

        Args:
            roidata (numpy.ndarray): ROI array data

        Returns:
            DataSetGroup: ROI dataset group
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
            params (DataSetGroup): ROI dataset group

        Returns:
            numpy.ndarray: ROI array data
        """

    def roi_has_changed(self) -> bool:
        """Return True if ROI has changed since last call to this method.

        The first call to this method will return True if ROI has not yet been set,
        or if ROI has been set and has changed since the last call to this method.
        The next call to this method will always return False if ROI has not changed
        in the meantime.

        Returns:
            bool: True if ROI has changed
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
            numpy.ndarray: regions of interest array
        """
        roidata = self.metadata.get(ROI_KEY)
        assert roidata is None or isinstance(roidata, np.ndarray)
        return roidata

    @roi.setter
    def roi(self, roidata: np.ndarray):
        """Set object regions of interest array, using a list or ROI dataset params.

        Args:
            roidata (numpy.ndarray): regions of interest array
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
            label (str): label
            shapetype (ShapeTypes): shape type
            array (numpy.ndarray): array
            param (guidata.dataset.DataSet): parameters

        Returns:
            ResultShape: result shape
        """
        mshape = ResultShape(shapetype, array, label)
        mshape.add_to(self)
        if param is not None:
            self.metadata[f"{label}Param"] = str(param)
        return mshape

    def iterate_resultshapes(self):
        """Iterate over object result shapes.

        Yields:
            ResultShape: result shape
        """
        for key, value in self.metadata.items():
            if ResultShape.match(key, value):
                yield ResultShape.from_metadata_entry(key, value)

    def update_resultshapes_from(self, other):
        """Update geometric shape from another object (merge metadata).

        Args:
            other (BaseObj): other object
        """
        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            mshape.merge_with(self, other)

    def transform_shapes(self, orig, func, param=None):
        """Apply transform function to result shape / annotations coordinates.

        Args:
            orig (BaseObj): original object
            func (callable): transform function
            param (object): transform function parameter
        """

        def transform(coords: np.ndarray):
            """Transform coordinates"""
            if param is None:
                func(self, orig, coords)
            else:
                func(self, orig, coords, param)

        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            transform(mshape.data)
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
            fmt (str): format string
            lbl (bool): if True, add label
            editable (bool): if True, ROI is editable

        Yields:
            PlotItem: plot item
        """

    def __set_annotations(self, annotations: str | None) -> None:
        """Set object annotations (JSON string describing annotation plot items)

        Args:
            annotations (str | None): JSON string describing annotation plot items,
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
            filename (str): filename
        """
        with open(filename, mode="rb") as fdesc:
            self.annotations = fdesc.read().decode()

    def add_annotations_from_items(self, items: list) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items (list): annotation plot items
        """
        ann_items = json_to_items(self.annotations)
        ann_items.extend(items)
        if ann_items:
            self.annotations = items_to_json(ann_items)

    def add_annotations_from_file(self, filename: str) -> None:
        """Add object annotations from file (JSON).

        Args:
            filename (str): filename
        """
        items = load_items(JSONReader(filename))
        self.add_annotations_from_items(items)

    @abc.abstractmethod
    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title (str): title (if None, use object title)
        """

    def iterate_shape_items(self, editable: bool = False):
        """Iterate over computing items encoded in metadata (if any).

        Args:
            editable (bool): if True, ROI is editable

        Yields:
            PlotItem: plot item
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
            name (str): option name

        Returns:
            Option value

        Valid option names:
            'format': format string (str)
            'showlabel': show label (bool)
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
            name (str): option name
            value (Any): option value

        Valid option names:
            'format': format string (str)
            'showlabel': show label (bool)
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
            self.metadata[key] = getattr(item.param, key)
        # Subclasses may override this method to update metadata from plot item,
        # then call this implementation of the method to update metadata standard
        # entries.

    def export_metadata_to_file(self, filename: str) -> None:
        """Export object metadata to file (JSON).

        Args:
            filename (str): filename
        """
        handler = JSONHandler(filename)
        handler.set_json_dict(self.metadata)
        handler.save()

    def import_metadata_from_file(self, filename: str) -> None:
        """Import object metadata from file (JSON).

        Args:
            filename (str): filename
        """
        handler = JSONHandler(filename)
        handler.load()
        self.metadata = handler.get_json_dict()

    def metadata_to_html(self) -> str:
        """Convert metadata to human-readable string.

        Returns:
            str: HTML string
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
