# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Base Module
--------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import json
import sys
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    Literal,
    TypeVar,
)

import numpy as np
from guidata.configtools import get_font
from guidata.dataset import update_dataset
from guidata.io import JSONReader, JSONWriter
from plotpy.builder import make
from plotpy.io import load_items, save_items
from plotpy.items import (
    AbstractLabelItem,
    AnnotatedPoint,
    AnnotatedSegment,
    AnnotatedShape,
    LabelItem,
    PolygonShape,
)
from sigima.objects.base import (
    ROI_KEY,
    BaseObj,
    BaseROI,
    ResultProperties,
    ResultShape,
    ShapeTypes,
    TypeObj,
    TypeROI,
    TypeROIParam,
    TypeSingleROI,
    get_generic_roi_title,
)
from sigima.tools import coordinates

from datalab.config import PLOTPY_CONF, Conf

if TYPE_CHECKING:
    from plotpy.items import (
        AbstractShape,
        AnnotatedCircle,
        AnnotatedEllipse,
        AnnotatedPolygon,
        AnnotatedRectangle,
        AnnotatedXRange,
        CurveItem,
        Marker,
        MaskedImageItem,
    )
    from plotpy.styles import AnnotationParam, ShapeParam


def config_annotated_shape(
    item: AnnotatedShape,
    fmt: str,
    lbl: bool,
    section: str | None = None,
    option: str | None = None,
    show_computations: bool | None = None,
):
    """Configurate annotated shape

    Args:
        item: Annotated shape item
        fmt: Format string
        lbl: Show label
        section: Shape style section (e.g. "plot")
        option: Shape style option (e.g. "shape/drag")
        show_computations: Show computations
    """
    param: AnnotationParam = item.annotationparam
    param.format = fmt
    param.show_label = lbl
    if show_computations is not None:
        param.show_computations = show_computations

    if isinstance(item, AnnotatedSegment):
        item.label.labelparam.anchor = "T"
        item.label.labelparam.update_item(item.label)

    param.update_item(item)
    if section is not None and option is not None:
        item.set_style(section, option)


# TODO: [P3] Move this function as a method of plot items in PlotPy
def set_plot_item_editable(
    item: AbstractShape | AbstractLabelItem | AnnotatedShape, state
):
    """Set plot item editable state

    Args:
        item: Plot item
        state: State
    """
    item.set_movable(state)
    item.set_resizable(state and not isinstance(item, AbstractLabelItem))
    item.set_rotatable(state and not isinstance(item, AbstractLabelItem))
    item.set_readonly(not state)
    item.set_selectable(state)


def items_to_json(items: list) -> str | None:
    """Convert plot items to JSON string

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
    """Convert JSON string to plot items

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


class ResultPlotPyAdapter:
    """Adapter for converting `sigima` result properties or result shape to PlotPy

    Args:
        result: Result properties or result shape
    """

    def __init__(self, result: ResultProperties | ResultShape) -> None:
        self.result = result
        self.item_json = ""  # JSON representation of the item

    def update_obj_metadata_from_item(
        self, obj: BaseObj, item: LabelItem | None
    ) -> None:
        """Update object metadata with label item

        Args:
            obj: object (signal/image)
            item: label item
        """
        if item is not None:
            self.item_json = items_to_json([item])
        self.result.set_obj_metadata(obj)

    def create_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Create label item

        Args:
            obj: object (signal/image)

        Returns:
            Label item

        .. note::

            The signal or image object is required as argument to create the label
            item because the label text may contain format strings that need to be
            filled with the object properties. For instance, the label text may contain
            the signal or image units.
        """
        text = self.result.get_text(obj)
        item = make.label(text, "TL", (0, 0), "TL", title=self.result.title)
        font = get_font(PLOTPY_CONF, "properties", "label/font")
        item.set_style("properties", "label")
        item.labelparam.font.update_param(font)
        item.labelparam.update_item(item)
        return item

    def get_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Return label item associated to this result

        Args:
            obj: object (signal/image)

        Returns:
            Label item

        .. note::

            The signal or image object is required as argument to eventually create
            the label item if it has not been created yet.
            See :py:meth:`create_label_item`.
        """
        if not self.item_json:
            # Label item has not been created yet
            item = self.create_label_item(obj)
            if item is not None:
                self.update_obj_metadata_from_item(obj, item)
        if self.item_json:
            item = json_to_items(self.item_json)[0]
            assert isinstance(item, LabelItem)
            return item
        return None


class ResultPropertiesPlotPyAdapter(ResultPlotPyAdapter):
    """Adapter for converting `sigima` result properties to PlotPy

    Args:
        result: Result properties

    Raises:
        AssertionError: invalid argument
    """

    def __init__(self, result: ResultProperties) -> None:
        assert isinstance(result, ResultProperties)
        super().__init__(result)


class ResultShapePlotPyAdapter(ResultPlotPyAdapter):
    """Adapter for converting `sigima` result shapes to PlotPy

    Args:
        result: Result shape

    Raises:
        AssertionError: invalid argument
    """

    def __init__(self, result: ResultShape) -> None:
        assert isinstance(result, ResultShape)
        super().__init__(result)

    def create_label_item(self, obj: BaseObj) -> LabelItem | None:
        """Create label item

        Returns:
            Label item
        """
        if self.result.add_label:
            return super().create_label_item(obj)
        return None

    def iterate_plot_items(
        self, fmt: str, lbl: bool, option: Literal["s", "i"]
    ) -> Iterable:
        """Iterate over metadata shape plot items.

        Args:
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option ("s" for signal, "i" for image)

        Yields:
            Plot item
        """
        for coords in self.result.raw_data:
            yield self.create_shape_item(coords, fmt, lbl, option)

    def create_shape_item(
        self, coords: np.ndarray, fmt: str, lbl: bool, option: Literal["s", "i"]
    ) -> (
        AnnotatedPoint
        | Marker
        | AnnotatedRectangle
        | AnnotatedCircle
        | AnnotatedSegment
        | AnnotatedEllipse
        | PolygonShape
        | None
    ):
        """Make geometrical shape plot item adapted to the shape type.

        Args:
            coords: shape data
            fmt: numeric format (e.g. "%.3f")
            lbl: if True, show shape labels
            option: shape style option ("s" for signal, "i" for image)

        Returns:
            Plot item
        """
        if self.result.shapetype is ShapeTypes.MARKER:
            x0, y0 = coords
            item = self.__make_marker_item(x0, y0, fmt)
        elif self.result.shapetype is ShapeTypes.POINT:
            x0, y0 = coords
            item = AnnotatedPoint(x0, y0)
            sparam: ShapeParam = item.shape.shapeparam
            sparam.symbol.marker = "Ellipse"
            sparam.symbol.size = 6
            sparam.sel_symbol.marker = "Ellipse"
            sparam.sel_symbol.size = 6
            aparam = item.annotationparam
            aparam.title = self.result.title
            sparam.update_item(item.shape)
            aparam.update_item(item)
        elif self.result.shapetype is ShapeTypes.RECTANGLE:
            x0, y0, x1, y1 = coords
            item = make.annotated_rectangle(x0, y0, x1, y1, title=self.result.title)
        elif self.result.shapetype is ShapeTypes.CIRCLE:
            xc, yc, r = coords
            x0, y0, x1, y1 = coordinates.circle_to_diameter(xc, yc, r)
            item = make.annotated_circle(x0, y0, x1, y1, title=self.result.title)
        elif self.result.shapetype is ShapeTypes.SEGMENT:
            x0, y0, x1, y1 = coords
            item = make.annotated_segment(x0, y0, x1, y1, title=self.result.title)
        elif self.result.shapetype is ShapeTypes.ELLIPSE:
            xc, yc, a, b, t = coords
            coords = coordinates.ellipse_to_diameters(xc, yc, a, b, t)
            x0, y0, x1, y1, x2, y2, x3, y3 = coords
            item = make.annotated_ellipse(
                x0, y0, x1, y1, x2, y2, x3, y3, title=self.result.title
            )
        elif self.result.shapetype is ShapeTypes.POLYGON:
            x, y = coords[::2], coords[1::2]
            item = make.polygon(x, y, title=self.result.title, closed=False)
        else:
            print(f"Warning: unsupported item {self.result.shapetype}", file=sys.stderr)
            return None
        if isinstance(item, AnnotatedShape):
            config_annotated_shape(item, fmt, lbl, "results", option)
        set_plot_item_editable(item, False)
        return item

    def __make_marker_item(self, x0: float, y0: float, fmt: str) -> Marker:
        """Make marker item

        Args:
            x0: x coordinate
            y0: y coordinate
            fmt: numeric format (e.g. '%.3f')
        """
        if np.isnan(x0):
            mstyle = "-"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.result.title + ": " + fmt) % y

        elif np.isnan(y0):
            mstyle = "|"

            def label(x, y):  # pylint: disable=unused-argument
                return (self.result.title + ": " + fmt) % x

        else:
            mstyle = "+"
            txt = self.result.title + ": (" + fmt + ", " + fmt + ")"

            def label(x, y):
                return txt % (x, y)

        return make.marker(
            position=(x0, y0),
            markerstyle=mstyle,
            label_cb=label,
            linestyle="DashLine",
            color="yellow",
        )


TypeSingleROIPlotPyAdapter = TypeVar(
    "TypeSingleROIPlotPyAdapter",
    bound="BaseSingleROIPlotPyAdapter[TypeSingleROI, TypeROIItem]",
)
TypeROIItem = TypeVar(
    "TypeROIItem",
    bound="AnnotatedXRange | AnnotatedPolygon | AnnotatedRectangle | AnnotatedCircle",
)


class BaseSingleROIPlotPyAdapter(Generic[TypeSingleROI, TypeROIItem], abc.ABC):
    """Base class for single ROI plot item adapter

    Args:
        single_roi: single ROI object
    """

    def __init__(self, single_roi: TypeSingleROI) -> None:
        self.single_roi = single_roi

    @abc.abstractmethod
    def to_plot_item(self, obj: TypeObj) -> TypeROIItem:
        """Make ROI plot item from ROI.

        Args:
            obj: object (signal/image), for physical-indices coordinates conversion

        Returns:
            Plot item
        """

    @classmethod
    @abc.abstractmethod
    def from_plot_item(cls, item: TypeROIItem) -> TypeSingleROI:
        """Create single ROI from plot item

        Args:
            item: plot item

        Returns:
            Single ROI
        """


def configure_roi_item(
    item,
    fmt: str,
    lbl: bool,
    editable: bool,
    option: Literal["s", "i"],
):
    """Configure ROI plot item.

    Args:
        item: plot item
        fmt: numeric format (e.g. "%.3f")
        lbl: if True, show shape labels
        editable: if True, make shape editable
        option: shape style option ("s" for signal, "i" for image)

    Returns:
        Plot item
    """
    option += "/" + ("editable" if editable else "readonly")
    if not editable:
        if isinstance(item, AnnotatedShape):
            config_annotated_shape(
                item, fmt, lbl, "roi", option, show_computations=editable
            )
        item.set_movable(False)
        item.set_resizable(False)
        item.set_readonly(True)
    item.set_style("roi", option)
    return item


class BaseROIPlotPyAdapter(Generic[TypeROI], abc.ABC):
    """ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def __init__(self, roi: BaseROI[TypeObj, TypeSingleROI, TypeROIParam]) -> None:
        self.roi = roi

    @abc.abstractmethod
    def to_plot_item(self, single_roi: TypeSingleROI, obj: TypeObj) -> TypeROIItem:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion

        Returns:
            Plot item
        """

    def iterate_roi_items(
        self, obj: TypeObj, fmt: str, lbl: bool, editable: bool = True
    ) -> Iterator[TypeROIItem]:
        """Iterate over ROI plot items associated to each single ROI composing
        the object.

        Args:
            obj: object (signal/image), for physical-indices coordinates conversion
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable

        Yields:
            Plot item
        """
        for index, single_roi in enumerate(self.roi.single_rois):
            roi_item = self.to_plot_item(single_roi, obj)
            item = configure_roi_item(
                roi_item, fmt, lbl, editable, option=self.roi.PREFIX
            )
            item.setTitle(single_roi.title or get_generic_roi_title(index))
            yield item


TypePlotItem = TypeVar("TypePlotItem", bound="CurveItem | MaskedImageItem")


class BaseObjPlotPyAdapter(Generic[TypeObj, TypePlotItem]):
    """Object (signal/image) plot item adapter class"""

    DEFAULT_FMT = "s"  # This is overriden in children classes
    CONF_FMT = Conf.view.sig_format  # This is overriden in children classes

    def __init__(self, obj: TypeObj) -> None:
        """Initialize the adapter with the object.

        Args:
            obj: object (signal/image)
        """
        self.obj = obj
        self.__default_options = {
            "format": "%" + self.CONF_FMT.get(self.DEFAULT_FMT),
            "showlabel": Conf.view.show_label.get(False),
        }

    def get_obj_option(self, name: str) -> Any:
        """Get object option value.
        Args:
            name: option name

        Returns:
            Option value
        """
        default = self.__default_options[name]
        return self.obj.get_metadata_option(name, default)

    @abc.abstractmethod
    def make_item(self, update_from: TypePlotItem | None = None) -> TypePlotItem:
        """Make plot item from data.

        Args:
            update_from: update

        Returns:
            Plot item
        """

    @abc.abstractmethod
    def update_item(self, item: TypePlotItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """

    def add_annotations_from_items(self, items: list) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
        """
        ann_items = json_to_items(self.obj.annotations)
        ann_items.extend(items)
        if ann_items:
            self.obj.annotations = items_to_json(ann_items)

    @abc.abstractmethod
    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use object title)
        """

    def iterate_shape_items(self, editable: bool = False):
        """Iterate over shape items encoded in metadata (if any).

        Args:
            editable: if True, annotations are editable

        Yields:
            Plot item
        """
        # pylint: disable=import-outside-toplevel
        from datalab.adapters_plotpy.factories import create_adapter_from_object

        fmt = self.get_obj_option("format")
        lbl = self.get_obj_option("showlabel")
        for key, value in self.obj.metadata.items():
            if key == ROI_KEY:
                roi = self.obj.roi
                if roi is not None:
                    yield from create_adapter_from_object(roi).iterate_roi_items(
                        self.obj, fmt=fmt, lbl=lbl, editable=False
                    )
            elif ResultShape.match(key, value):
                mshape: ResultShape = ResultShape.from_metadata_entry(key, value)
                yield from create_adapter_from_object(mshape).iterate_plot_items(
                    fmt, lbl, self.obj.PREFIX
                )
        if self.obj.annotations:
            try:
                for item in json_to_items(self.obj.annotations):
                    if isinstance(item, AnnotatedShape):
                        config_annotated_shape(item, fmt, lbl)
                    set_plot_item_editable(item, editable)
                    yield item
            except json.decoder.JSONDecodeError:
                pass

    def update_plot_item_parameters(self, item: TypePlotItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        def_dict = Conf.view.get_def_dict(self.__class__.__name__[:3].lower())
        self.obj.set_metadata_options_defaults(def_dict, overwrite=False)

        # Subclasses have to override this method to update plot item parameters,
        # then call this implementation of the method to update plot item.
        update_dataset(item.param, self.obj.get_metadata_options())
        item.param.update_item(item)
        if item.selected:
            item.select()

    def update_metadata_from_plot_item(self, item: TypePlotItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        def_dict = Conf.view.get_def_dict(self.__class__.__name__[:3].lower())
        for key in def_dict:
            if hasattr(item.param, key):  # In case the PlotPy version is not up-to-date
                self.obj.set_metadata_option(key, getattr(item.param, key))
        # Subclasses may override this method to update metadata from plot item,
        # then call this implementation of the method to update metadata standard
        # entries.
