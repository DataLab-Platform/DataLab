# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base model classes for signals and images.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import enum
import sys
from collections.abc import Callable, Generator, Iterable
from copy import deepcopy
from typing import Any, Generic, Iterator, Literal, Type, TypeVar

import guidata.dataset as gds
import numpy as np
import pandas as pd
from numpy import ma

from sigima_.algorithms import coordinates
from sigima_.config import _

if sys.version_info >= (3, 11):
    # Use Self from typing module in Python 3.11+
    from typing import Self
else:
    # Use Self from typing_extensions module in Python < 3.11
    from typing_extensions import Self

ROI_KEY = "_roi_"


def deepcopy_metadata(
    metadata: dict[str, Any], special_keys: set[str] | None = None
) -> dict[str, Any]:
    """Deepcopy metadata, except keys starting with "_" (private keys)
    with the exception of "_roi_" and "_ann_" keys.

    Args:
        metadata: Metadata dictionary to deepcopy.
        special_keys: Set of keys that should not be removed even if they
         start with "_".

    Returns:
        A new dictionary with deepcopied metadata, excluding private keys
        except those in `special_keys`.
    """
    if special_keys is None:
        special_keys = set()
    special_keys = set([ROI_KEY]) | special_keys
    mdcopy = deepcopy(metadata)
    for key, value in metadata.items():
        rshape = ResultShape.from_metadata_entry(key, value)
        if rshape is None and key.startswith("_") and key not in special_keys:
            mdcopy.pop(key)
    return mdcopy


@enum.unique
class Choices(enum.Enum):
    """Object associating an enum to guidata.dataset.ChoiceItem choices"""

    # Reimplement enum.Enum method as suggested by Python documentation:
    # https://docs.python.org/3/library/enum.html#enum.Enum._generate_next_value_
    # Here, it is only needed for ImageDatatypes (see core/model/image.py).
    # pylint: disable=unused-argument,no-self-argument,no-member
    def _generate_next_value_(name, start, count, last_values):
        return str(name).lower()

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
        if np.issubdtype(dtype, np.integer):
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
    # pylint: disable=unused-argument,no-self-argument,no-member
    def _generate_next_value_(name, start, count, last_values):
        return f"_{str(name).lower()[:3]}_"

    #: Rectangle shape
    RECTANGLE = enum.auto()
    #: Circle shape
    CIRCLE = enum.auto()
    #: Ellipse shape
    ELLIPSE = enum.auto()
    #: Segment shape
    SEGMENT = enum.auto()
    #: Marker shape
    MARKER = enum.auto()
    #: Point shape
    POINT = enum.auto()
    #: Polygon shape
    POLYGON = enum.auto()


class BaseResult(abc.ABC):
    """Base class for results, i.e. objects returned by computation functions
    used by :py:class`cdl.gui.processor.base.BaseProcessor.compute_1_to_0` method.

    Args:
        title: result title
        category: result category
        array: result array (one row per ROI, first column is ROI index)
        labels: result labels (one label per column of result array)
    """

    PREFIX = ""  # To be overriden in children classes
    METADATA_ATTRS = ()  # To be overriden in children classes

    def __init__(
        self,
        title: str,
        array: np.ndarray,
        labels: list[str] | None = None,
    ) -> None:
        assert isinstance(title, str)
        self.title = title
        self.array = array
        self.xunit: str = ""
        self.yunit: str = ""
        if labels is None:
            labels = []
        self.__labels = labels
        self.check_array()

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Return result category"""

    def check_array(self) -> None:
        """Check if array attribute is valid

        Raises:
            AssertionError: invalid array
        """
        # Allow to pass a list of lists or a NumPy array.
        # For instance, the following are equivalent:
        #   array = [[1, 2], [3, 4]]
        #   array = np.array([[1, 2], [3, 4]])
        # Or, for only one line (one single result), the following are equivalent:
        #   array = [1, 2]
        #   array = [[1, 2]]
        #   array = np.array([[1, 2]])
        if isinstance(self.array, (list, tuple)):
            if isinstance(self.array[0], (list, tuple)):
                self.array = np.array(self.array)
            else:
                self.array = np.array([self.array])
        assert isinstance(self.array, np.ndarray)
        assert len(self.array.shape) == 2

    @property
    def labels(self) -> list[str]:
        """Return result labels (one label per column of result array)"""
        return self.__labels

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents, i.e. a tuple of couples of (index, text)
        where index is the column of raw_data and text is the associated
        label format string"""
        return tuple(enumerate(self.labels))

    @property
    def headers(self) -> list[str]:
        """Return result headers (one header per column of result array)"""
        # Default implementation: return labels
        return self.__labels

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame from properties array"""
        return pd.DataFrame(self.shown_array, columns=list(self.headers))

    @property
    @abc.abstractmethod
    def shown_array(self) -> np.ndarray:
        """Return array of shown results, i.e. including complementary array (if any)

        Returns:
            Array of shown results
        """

    @property
    def raw_data(self):
        """Return raw data (array without ROI informations)"""
        return self.array[:, 1:]

    @property
    def key(self) -> str:
        """Return metadata key associated to result"""
        return self.PREFIX + self.title

    @classmethod
    def from_metadata_entry(cls, key: str, value: dict[str, Any]) -> Self | None:
        """Create metadata shape object from (key, value) metadata entry"""
        if (
            isinstance(key, str)
            and key.startswith(cls.PREFIX)
            and isinstance(value, dict)
        ):
            try:
                title = key[len(cls.PREFIX) :]
                instance = cls(title, **value)
                return instance
            except (ValueError, TypeError):
                pass
        return None

    @classmethod
    def match(cls, key, value) -> bool:
        """Return True if metadata dict entry (key, value) is a metadata result"""
        return cls.from_metadata_entry(key, value) is not None

    def add_to(self, obj: BaseObj) -> None:
        """Add result to object metadata

        Args:
            obj: object (signal/image)
        """
        self.set_obj_metadata(obj)

    def set_obj_metadata(self, obj: BaseObj) -> None:
        """Set object metadata with properties

        Args:
            obj: object
        """
        obj.metadata[self.key] = {
            key: getattr(self, key) for key in self.METADATA_ATTRS
        }

    def get_text(self, obj: TypeObj) -> str:
        """Return text representation of result"""
        text = ""
        for i_row in range(self.array.shape[0]):
            suffix = f"|ROI{i_row}" if i_row > 0 else ""
            text += f"<u>{self.title}{suffix}</u>:"
            for i_col, label in self.label_contents:
                # "label" may contains "<" and ">" characters which are interpreted
                # as HTML tags by the LabelItem. We must escape them.
                label = label.replace("<", "&lt;").replace(">", "&gt;")
                if "%" not in label:
                    label += " = %g"
                text += (
                    "<br>" + label.strip().format(obj) % self.shown_array[i_row, i_col]
                )
            if i_row < self.shown_array.shape[0] - 1:
                text += "<br><br>"
        return text


class ResultProperties(BaseResult):
    """Object representing properties serializable in signal/image metadata.

    Result `array` is a NumPy 2-D array: each row is a list of properties, optionnally
    associated to a ROI (first column value).

    ROI index is starting at 0 (or is simply 0 if there is no ROI).

    Args:
        title: properties title
        array: properties array
        labels: properties labels (one label per column of result array)

    .. note::

        The `array` argument can be a list of lists or a NumPy array. For instance,
        the following are equivalent:

        - ``array = [[1, 2], [3, 4]]``
        - ``array = np.array([[1, 2], [3, 4]])``

        Or for only one line (one single result), the following are equivalent:

        - ``array = [1, 2]``
        - ``array = [[1, 2]]``
        - ``array = np.array([[1, 2]])``
    """

    PREFIX = "_properties_"
    METADATA_ATTRS = ("array", "labels")

    def __init__(
        self,
        title: str,
        array: np.ndarray,
        labels: list[str] | None,
    ) -> None:
        super().__init__(title, array, labels)
        if labels is not None:
            assert len(labels) == self.array.shape[1] - 1

    @property
    def category(self) -> str:
        """Return result category"""
        return _("Properties") + f" | {self.title}"

    @property
    def headers(self) -> list[str]:
        """Return result headers (one header per column of result array)"""
        # ResultProperties implementation: return labels without units or "=" sign
        return [label.split("=")[0].strip() for label in self.labels]

    @property
    def shown_array(self) -> np.ndarray:
        """Return array of shown results, i.e. including complementary array (if any)

        Returns:
            Array of shown results
        """
        return self.raw_data


class ResultShape(BaseResult):
    """Object representing a geometrical shape serializable in signal/image metadata.

    Result `array` is a NumPy 2-D array: each row is a result, optionnally associated
    to a ROI (first column value).

    ROI index is starting at 0 (or is simply 0 if there is no ROI).

    Args:
        title: result shape title
        array: shape coordinates (multiple shapes: one shape per row),
         first column is ROI index (0 if there is no ROI)
        shape: shape kind
        add_label: if True, add a label item (and the geometrical shape) to plot
         (default to False)

    Raises:
        AssertionError: invalid argument

    .. note::

        The `array` argument can be a list of lists or a NumPy array. For instance,
        the following are equivalent:

        - ``array = [[1, 2], [3, 4]]``
        - ``array = np.array([[1, 2], [3, 4]])``

        Or for only one line (one single result), the following are equivalent:

        - ``array = [1, 2]``
        - ``array = [[1, 2]]``
        - ``array = np.array([[1, 2]])``
    """

    PREFIX = "_shapes_"
    METADATA_ATTRS = ("array", "shape", "add_label")

    def __init__(
        self,
        title: str,
        array: np.ndarray,
        shape: Literal[
            "rectangle", "circle", "ellipse", "segment", "marker", "point", "polygon"
        ],
        add_label: bool = False,
    ) -> None:
        self.shape = shape
        try:
            self.shapetype = ShapeTypes[shape.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid shapetype {shape}") from exc
        self.add_label = add_label
        super().__init__(title, array)

    @property
    def category(self) -> str:
        """Return result category"""
        return self.shape.upper()

    def check_array(self) -> None:
        """Check if array attribute is valid

        Raises:
            AssertionError: invalid array
        """
        super().check_array()
        if self.shapetype is ShapeTypes.POLYGON:
            # Polygon is a special case: the number of data columns is variable
            # (2 columns per point). So we only check if the number of columns
            # is odd, which means that the first column is the ROI index, followed
            # by an even number of data columns (flattened x, y coordinates).
            assert self.array.shape[1] % 2 == 1
        else:
            data_colnb = len(self.__get_coords_labels())
            # `data_colnb` is the number of data columns depends on the shape type,
            # not counting the ROI index, hence the +1 in the following assertion
            assert self.array.shape[1] == data_colnb + 1

    def __get_coords_labels(self) -> list[str]:
        """Return shape coordinates labels

        Returns:
            Shape coordinates labels
        """
        if self.shapetype is ShapeTypes.POLYGON:
            labels = []
            for i in range(0, self.array.shape[1] - 1, 2):
                labels += [f"x{i // 2}", f"y{i // 2}"]
            return labels
        try:
            return {
                ShapeTypes.MARKER: ["x", "y"],
                ShapeTypes.POINT: ["x", "y"],
                ShapeTypes.RECTANGLE: ["x0", "y0", "x1", "y1"],
                ShapeTypes.CIRCLE: ["x", "y", "r"],
                ShapeTypes.SEGMENT: ["x0", "y0", "x1", "y1"],
                ShapeTypes.ELLIPSE: ["x", "y", "a", "b", "θ"],
            }[self.shapetype]
        except KeyError as exc:
            raise NotImplementedError(
                f"Unsupported shapetype {self.shapetype}"
            ) from exc

    def __get_complementary_xlabels(self) -> list[str]:
        """Return complementary labels for result array columns

        Returns:
            Complementary labels for result array columns, or empty list if there
             is no complementary array
        """
        if self.shapetype is ShapeTypes.SEGMENT:
            return ["L", "Xc", "Yc"]
        if self.shapetype in (ShapeTypes.CIRCLE, ShapeTypes.ELLIPSE):
            return ["A"]
        return []

    def __get_complementary_array(self) -> np.ndarray | None:
        """Return the complementary array of results, e.g. the array of lengths
        for a segment result shape, or the array of areas for a circle result shape

        Returns:
            Complementary array of results, or None if there is no complementary array
        """
        array = self.array
        if self.shapetype is ShapeTypes.SEGMENT:
            dx1, dy1 = array[:, 3] - array[:, 1], array[:, 4] - array[:, 2]
            length = np.linalg.norm(np.vstack([dx1, dy1]).T, axis=1)
            xc = (array[:, 1] + array[:, 3]) / 2
            yc = (array[:, 2] + array[:, 4]) / 2
            return np.vstack([length, xc, yc]).T
        if self.shapetype is ShapeTypes.CIRCLE:
            area = np.pi * array[:, 3] ** 2
            return area.reshape(-1, 1)
        if self.shapetype is ShapeTypes.ELLIPSE:
            area = np.pi * array[:, 3] * array[:, 4]
            return area.reshape(-1, 1)
        return None

    @property
    def headers(self) -> list[str]:
        """Return result headers (one header per column of result array)"""
        labels = self.__get_coords_labels() + self.__get_complementary_xlabels()
        return labels[-self.shown_array.shape[1] :]

    @property
    def shown_array(self) -> np.ndarray:
        """Return array of shown results, i.e. including complementary array (if any)

        Returns:
            Array of shown results
        """
        comp_array = self.__get_complementary_array()
        if comp_array is None:
            return self.raw_data
        return np.hstack([self.raw_data, comp_array])

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents, i.e. a tuple of couples of (index, text)
        where index is the column of raw_data and text is the associated
        label format string"""
        contents = []
        for idx, lbl in enumerate(self.__get_complementary_xlabels()):
            contents.append((idx + self.raw_data.shape[1], lbl))
        return tuple(contents)

    def merge_with(self, obj: BaseObj, other_metadata: dict[str, Any]) -> None:
        """Merge object resultshape with another's metadata (obj <-- other obj's
        metadata)

        Args:
            obj: object (signal/image)
            other_metadata: other object metadata
        """
        other_value = other_metadata.get(self.key)
        if other_value is not None:
            other = ResultShape.from_metadata_entry(self.key, other_value)
            assert other is not None
            other_array = np.array(other.array, copy=True)
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
            func(self.raw_data)
        elif self.shapetype is ShapeTypes.CIRCLE:
            coords = coordinates.array_circle_to_diameter(self.raw_data)
            func(coords)
            self.raw_data[:] = coordinates.array_circle_to_center_radius(coords)
        elif self.shapetype is ShapeTypes.ELLIPSE:
            coords = coordinates.array_ellipse_to_diameters(self.raw_data)
            func(coords)
            self.raw_data[:] = coordinates.array_ellipse_to_center_axes_angle(coords)
        else:
            raise NotImplementedError(f"Unsupported shapetype {self.shapetype}")


TypeObj = TypeVar("TypeObj", bound="BaseObj")
TypeROIParam = TypeVar("TypeROIParam", bound="BaseROIParam")
TypeSingleROI = TypeVar("TypeSingleROI", bound="BaseSingleROI")
TypeROI = TypeVar("TypeROI", bound="BaseROI")


class BaseObjMeta(abc.ABCMeta, gds.DataSetMeta):
    """Mixed metaclass to avoid conflicts"""


class NoDefaultOption:
    """Marker class for metadata option without default value"""


class BaseObj(Generic[TypeROI], metaclass=BaseObjMeta):
    """Object (signal/image) interface"""

    PREFIX = ""  # This is overriden in children classes

    # This is overriden in children classes with a gds.DictItem instance:
    metadata: dict[str, Any] = {}
    annotations: str = ""

    VALID_DTYPES = (np.float64,)  # To be overriden in children classes

    def __init__(self):
        self.__roi_changed: bool | None = None
        self._maskdata_cache: np.ndarray | None = None
        self.__metadata_options_defaults: dict[str, Any] = {}

    @staticmethod
    @abc.abstractmethod
    def get_roi_class() -> Type[TypeROI]:
        """Return ROI class"""

    @property
    @abc.abstractmethod
    def data(self) -> np.ndarray | None:
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
            if isinstance(dtname, str)
            and dtname in (dtype.__name__ for dtype in cls.VALID_DTYPES)
        ]

    def check_data(self):
        """Check if data is valid, raise an exception if that's not the case

        Raises:
            TypeError: if data type is not supported
        """
        if self.data is not None:
            if self.data.dtype not in self.VALID_DTYPES:
                raise TypeError(f"Unsupported data type: {self.data.dtype}")

    def iterate_roi_indices(self) -> Generator[int | None, None, None]:
        """Iterate over object ROI indices (if there is no ROI, yield None)"""
        if self.roi is None:
            yield None
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
    def copy(self, title: str | None = None, dtype: np.dtype | None = None) -> Self:
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
    def physical_to_indices(self, coords: list[float]) -> list[int]:
        """Convert coordinates from physical (real world) to indices

        Args:
            coords: coordinates

        Returns:
            Indices
        """

    @abc.abstractmethod
    def indices_to_physical(self, indices: list[int]) -> list[float]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
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
    def roi(self) -> TypeROI | None:
        """Return object regions of interest object.

        Returns:
            Regions of interest object
        """
        roidata = self.metadata.get(ROI_KEY)
        if roidata is None:
            return None
        if not isinstance(roidata, dict):
            # Old or unsupported format: remove it
            self.metadata.pop(ROI_KEY)
            return None
        return self.get_roi_class().from_dict(roidata)

    @roi.setter
    def roi(self, roi: TypeROI | None) -> None:
        """Set object regions of interest.

        Args:
            roi: regions of interest object
        """
        if roi is None:
            if ROI_KEY in self.metadata:
                self.metadata.pop(ROI_KEY)
        else:
            self.metadata[ROI_KEY] = roi.to_dict()
        self.__roi_changed = True

    @property
    def maskdata(self) -> np.ndarray | None:
        """Return masked data (areas outside defined regions of interest)

        Returns:
            Masked data
        """
        roi_changed = self.roi_has_changed()
        if self.roi is None:
            if roi_changed:
                self._maskdata_cache = None
        elif roi_changed or self._maskdata_cache is None:
            self._maskdata_cache = self.roi.to_mask(self)
        return self._maskdata_cache

    def get_masked_view(self) -> ma.MaskedArray:
        """Return masked view for data

        Returns:
            Masked view
        """
        assert isinstance(self.data, np.ndarray)
        view = self.data.view(ma.MaskedArray)
        if self.maskdata is None:
            view.mask = np.isnan(self.data)
        else:
            view.mask = self.maskdata | np.isnan(self.data)
        return view

    def invalidate_maskdata_cache(self) -> None:
        """Invalidate mask data cache: force to rebuild it"""
        self._maskdata_cache = None

    def iterate_resultshapes(self) -> Iterable[ResultShape]:
        """Iterate over object result shapes.

        Yields:
            Result shape
        """
        for key, value in self.metadata.items():
            if ResultShape.match(key, value):
                result_shape = ResultShape.from_metadata_entry(key, value)
                assert result_shape is not None
                yield result_shape

    def iterate_resultproperties(self) -> Iterable[ResultProperties]:
        """Iterate over object result properties.

        Yields:
            Result properties
        """
        for key, value in self.metadata.items():
            if ResultProperties.match(key, value):
                result_properties = ResultProperties.from_metadata_entry(key, value)
                assert result_properties is not None
                yield result_properties

    def delete_results(self) -> None:
        """Delete all object results (shapes and properties)"""
        for key in list(self.metadata.keys()):
            if ResultShape.match(key, self.metadata[key]) or ResultProperties.match(
                key, self.metadata[key]
            ):
                self.metadata.pop(key)

    def update_resultshapes_from(self, other: BaseObj[TypeROI]) -> None:
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
            mshape.merge_with(self, other.metadata)
        # Iterating on `other` object result shapes to find result shapes that are
        # not present in this object, and add them to this object.
        for mshape in other.iterate_resultshapes():
            assert mshape is not None
            if mshape.key not in self.metadata:
                mshape.add_to(self)

    def transform_coords(
        self,
        coords: np.ndarray,
        orig: BaseObj[TypeROI],
        func: Callable[[BaseObj[TypeROI], BaseObj[TypeROI], np.ndarray], None]
        | Callable[[BaseObj[TypeROI], BaseObj[TypeROI], np.ndarray, gds.DataSet], None],
        param: gds.DataSet | None,
    ) -> None:
        """Transform coordinates

        Args:
            coords: coordinates to transform
            orig: original object (signal/image)
            func: transform function
            param: transform function parameter (optional)
        """
        if param is None:
            func(self, orig, coords)
        else:
            func(self, orig, coords, param)

    def transform_shapes(
        self,
        orig: BaseObj[TypeROI],
        func: Callable[[BaseObj[TypeROI], BaseObj[TypeROI], np.ndarray], None]
        | Callable[[BaseObj[TypeROI], BaseObj[TypeROI], np.ndarray, gds.DataSet], None],
        param: gds.DataSet | None = None,
    ) -> None:
        """Apply transform function to result shape / annotations coordinates.

        Args:
            orig: original object
            func: transform function
            param: transform function parameter
        """
        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            mshape.transform_coordinates(
                lambda coords: self.transform_coords(coords, orig, func, param)
            )

    def remove_all_shapes(self) -> None:
        """Remove metadata shapes and ROIs"""
        for key, value in list(self.metadata.items()):
            resultshape = ResultShape.from_metadata_entry(key, value)
            if resultshape is not None or key == ROI_KEY:
                # Metadata entry is a metadata shape or a ROI
                self.metadata.pop(key)

    def update_metadata_from(self, other_metadata: dict[str, Any]) -> None:
        """Update metadata from another object's metadata (merge result shapes and
        annotations, and update the rest of the metadata).

        Args:
            other_metadata: other object metadata
        """
        other_metadata = other_metadata.copy()
        # Merge result shapes
        for mshape in self.iterate_resultshapes():
            assert mshape is not None
            mshape.merge_with(self, other_metadata)
        for key, value in other_metadata.copy().items():
            if ResultShape.match(key, value):
                mshape = ResultShape.from_metadata_entry(key, value)
                assert mshape is not None
                if mshape.key not in self.metadata:
                    mshape.add_to(self)
                other_metadata.pop(key)
        # Updating the rest of the metadata
        self.metadata.update(other_metadata)

    # Method to set the default values of metadata options:
    def set_metadata_options_defaults(
        self, defaults: dict[str, Any], overwrite: bool = False
    ) -> None:
        """Set default values for metadata options

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        .. note::

            This will not overwrite existing metadata options
            (unless `overwrite` is True).
            It will only set the default values for options that are not already set.*
            Use `reset_metadata_to_defaults` method to reset all metadata options
            to their default values.

        Args:
            defaults: dictionary of default values for metadata options
            overwrite: whether to overwrite existing metadata options (default: False)
        """
        self.__metadata_options_defaults.update(defaults)
        for key, value in defaults.items():
            self.set_metadata_option(key, value, overwrite)

    def get_metadata_options_defaults(self) -> dict[str, Any]:
        """Return default values for metadata options

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        Returns:
            Dictionary of default values for metadata options
        """
        return self.__metadata_options_defaults

    def get_metadata_option(self, name: str, default: Any = NoDefaultOption) -> Any:
        """Return metadata option value

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name
            default: default value if option is not set (optional)

        Returns:
            Option value

        Raises:
            ValueError: if option name is invalid
        """
        if (
            default is not NoDefaultOption
            and name not in self.__metadata_options_defaults
        ):
            # If default is provided, store it in defaults
            # and set it as the option value
            self.__metadata_options_defaults[name] = default
            self.set_metadata_option(name, default, overwrite=False)
        try:
            value = self.metadata[f"__{name}"]
        except KeyError as exc:
            defaults = self.get_metadata_options_defaults()
            if name in defaults:
                value = defaults[name]
            else:
                raise ValueError(
                    f"Invalid metadata option name `{name}` "
                    f"(valid names: {', '.join(defaults.keys())})"
                ) from exc
        return value

    def set_metadata_option(
        self, name: str, value: Any, overwrite: bool = True
    ) -> None:
        """Set metadata option value

        A metadata option is a metadata entry starting with an underscore.
        It is a way to store application-specific options in object metadata.

        Args:
            name: option name
            value: option value
            overwrite: whether to overwrite existing metadata options (default: True)

        Raises:
            ValueError: if option name is invalid
        """
        if overwrite or f"__{name}" not in self.metadata:
            self.metadata[f"__{name}"] = value

    def get_metadata_options(self) -> dict[str, Any]:
        """Return metadata options
        A metadata option is a metadata entry starting with an underscore.

        Returns:
            Dictionary of metadata options (name: value)
        """
        options = {}
        for name, value in self.metadata.items():
            if name.startswith("__"):
                options[name[2:]] = value
        return options

    def reset_metadata_to_defaults(self) -> None:
        """Reset metadata to default values"""
        self.metadata = {}
        defaults = self.get_metadata_options_defaults()
        for name, value in defaults.items():
            self.set_metadata_option(name, value)

    def save_attr_to_metadata(self, attrname: str, new_value: Any) -> None:
        """Save attribute to metadata

        Args:
            attrname: attribute name
            new_value: new value
        """
        value = getattr(self, attrname)
        if value:
            self.metadata[f"orig_{attrname}"] = value
        setattr(self, attrname, new_value)

    def restore_attr_from_metadata(self, attrname: str, default: Any) -> None:
        """Restore attribute from metadata

        Args:
            attrname: attribute name
            default: default value
        """
        value = self.metadata.pop(f"orig_{attrname}", default)
        setattr(self, attrname, value)


class BaseROIParamMeta(abc.ABCMeta, gds.DataSetMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIParam(
    gds.DataSet,
    Generic[TypeObj, TypeSingleROI],  # type: ignore
    metaclass=BaseROIParamMeta,
):
    """Base class for ROI parameters"""

    @abc.abstractmethod
    def to_single_roi(self, obj: TypeObj, title: str = "") -> TypeSingleROI:
        """Convert parameters to single ROI

        Args:
            obj: object (signal/image)
            title: ROI title

        Returns:
            Single ROI
        """


class BaseSingleROI(Generic[TypeObj, TypeROIParam], abc.ABC):  # type: ignore
    """Base class for single ROI

    Args:
        coords: ROI edge (physical coordinates for signal)
        indices: if True, coords are indices (pixels) instead of physical coordinates
        title: ROI title
    """

    def __init__(
        self,
        coords: np.ndarray | list[int] | list[float],
        indices: bool,
        title: str = "ROI",
    ) -> None:
        self.coords = np.array(coords, int if indices else float)
        self.indices = indices
        self.title = title
        self.check_coords()

    def __eq__(self, other: BaseSingleROI) -> bool:
        """Test equality with another single ROI"""
        return (
            np.array_equal(self.coords, other.coords) and self.indices == other.indices
        )

    def get_physical_coords(self, obj: TypeObj) -> list[float]:
        """Return physical coords

        Args:
            obj: object (signal/image)

        Returns:
            Physical coords
        """
        if self.indices:
            return obj.indices_to_physical(self.coords.tolist())
        return self.coords.tolist()

    def get_indices_coords(self, obj: TypeObj) -> list[int]:
        """Return indices coords

        Args:
            obj: object (signal/image)

        Returns:
            Indices coords
        """
        if self.indices:
            return self.coords.tolist()
        return obj.physical_to_indices(self.coords.tolist())

    def set_indices_coords(self, obj: TypeObj, coords: np.ndarray) -> None:
        """Set indices coords

        Args:
            obj: object (signal/image)
            coords: indices coords
        """
        if self.indices:
            self.coords = coords
        else:
            self.coords = np.array(obj.indices_to_physical(self.coords.tolist()))

    @abc.abstractmethod
    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """

    @abc.abstractmethod
    def to_mask(self, obj: TypeObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal or image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """

    @abc.abstractmethod
    def to_param(self, obj: TypeObj, title: str | None = None) -> TypeROIParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal/image), for physical-indices coordinates conversion
            title: ROI title
        """

    def to_dict(self) -> dict:
        """Convert ROI to dictionary

        Returns:
            Dictionary
        """
        return {
            "coords": self.coords,
            "indices": self.indices,
            "title": self.title,
            "type": type(self).__name__,
        }

    @classmethod
    def from_dict(cls: Type[TypeSingleROI], dictdata: dict) -> TypeSingleROI:
        """Convert dictionary to ROI

        Args:
            dictdata: dictionary

        Returns:
            ROI
        """
        return cls(dictdata["coords"], dictdata["indices"], dictdata["title"])


class BaseROI(Generic[TypeObj, TypeSingleROI, TypeROIParam], abc.ABC):  # type: ignore
    """Abstract base class for ROIs (Regions of Interest)

    Args:
        singleobj: if True, when extracting data defined by ROIs, only one object
         is created (default to True). If False, one object is created per single ROI.
         If None, the value is get from the user configuration
        inverse: if True, ROI is outside the region of interest
    """

    PREFIX = ""  # This is overriden in children classes

    def __init__(self, singleobj: bool | None = None, inverse: bool = False) -> None:
        self.single_rois: list[TypeSingleROI] = []
        self.singleobj = singleobj
        self.inverse = inverse

    @staticmethod
    @abc.abstractmethod
    def get_compatible_single_roi_classes() -> list[Type[BaseSingleROI]]:
        """Return compatible single ROI classes"""

    def __len__(self) -> int:
        """Return number of ROIs"""
        return len(self.single_rois)

    def __iter__(self) -> Iterator[TypeSingleROI]:
        """Iterate over single ROIs"""
        return iter(self.single_rois)

    def get_single_roi(self, index: int) -> TypeSingleROI:
        """Return single ROI at index

        Args:
            index: ROI index
        """
        return self.single_rois[index]

    def is_empty(self) -> bool:
        """Return True if no ROI is defined"""
        return len(self) == 0

    @classmethod
    def create(
        cls: Type[BaseROI], single_roi: TypeSingleROI
    ) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Create Regions of Interest object from a single ROI.

        Args:
            single_roi: single ROI

        Returns:
            Regions of Interest object
        """
        roi = cls()
        roi.add_roi(single_roi)
        return roi

    def copy(self) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Return a copy of ROIs"""
        return deepcopy(self)

    def empty(self) -> None:
        """Empty ROIs"""
        self.single_rois.clear()

    def add_roi(
        self, roi: TypeSingleROI | BaseROI[TypeObj, TypeSingleROI, TypeROIParam]
    ) -> None:
        """Add ROI.

        Args:
            roi: ROI

        Raises:
            TypeError: if roi type is not supported (not a single ROI or a ROI)
            ValueError: if `singleobj` or `inverse` values are incompatible
        """
        if isinstance(roi, BaseSingleROI):
            self.single_rois.append(roi)
        elif isinstance(roi, BaseROI):
            self.single_rois.extend(roi.single_rois)
            if roi.singleobj != self.singleobj:
                raise ValueError("Incompatible `singleobj` values")
            if roi.inverse != self.inverse:
                raise ValueError("Incompatible `inverse` values")
        else:
            raise TypeError(f"Unsupported ROI type: {type(roi)}")

    @abc.abstractmethod
    def to_mask(self, obj: TypeObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal or image object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """

    def to_params(self, obj: TypeObj) -> list[TypeROIParam]:
        """Convert ROIs to a list of parameters

        Args:
            obj: object (signal/image), for physical to pixel conversion

        Returns:
            ROI parameters
        """
        return [iroi.to_param(obj, f"ROI{idx:02d}") for idx, iroi in enumerate(self)]

    @classmethod
    def from_params(
        cls: Type[BaseROI],
        obj: TypeObj,
        params: list[TypeROIParam],
        singleobj: bool | None = None,
        inverse: bool = False,
    ) -> BaseROI[TypeObj, TypeSingleROI, TypeROIParam]:
        """Create ROIs from parameters

        Args:
            obj: object (signal/image)
            params: ROI parameters
            singleobj: If True, extract all ROIs into a single object
            inverse: If True, extract the inverse of the ROIs

        Returns:
            ROIs
        """
        roi = cls()
        for param in params:
            assert isinstance(param, BaseROIParam), "Invalid ROI parameter type"
            roi.add_roi(param.to_single_roi(obj))
        roi.singleobj = singleobj
        roi.inverse = inverse
        return roi

    def to_dict(self) -> dict:
        """Convert ROIs to dictionary

        Returns:
            Dictionary
        """
        return {
            "singleobj": self.singleobj,
            "inverse": self.inverse,
            "single_rois": [roi.to_dict() for roi in self.single_rois],
        }

    @classmethod
    def from_dict(cls: Type[TypeROI], dictdata: dict) -> TypeROI:
        """Convert dictionary to ROIs

        Args:
            dictdata: dictionary

        Returns:
            ROIs
        """
        instance = cls()
        instance.singleobj = dictdata["singleobj"]
        instance.inverse = dictdata["inverse"]
        instance.single_rois = []
        for single_roi in dictdata["single_rois"]:
            for single_roi_class in instance.get_compatible_single_roi_classes():
                if single_roi["type"] == single_roi_class.__name__:
                    instance.single_rois.append(single_roi_class.from_dict(single_roi))
                    break
            else:
                raise ValueError(f"Unsupported single ROI type: {single_roi['type']}")
        return instance
