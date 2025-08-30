# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Base ROI Module
------------------------------
"""

import abc
from typing import TYPE_CHECKING, Generic, Iterator, Literal, TypeVar

from plotpy.items import AnnotatedShape
from sigima.objects.base import (
    BaseROI,
    TypeObj,
    TypeROI,
    TypeROIParam,
    TypeSingleROI,
    get_generic_roi_title,
)

from datalab.adapters_plotpy.base import config_annotated_shape

if TYPE_CHECKING:
    from plotpy.items import (
        AnnotatedCircle,
        AnnotatedPolygon,
        AnnotatedRectangle,
        AnnotatedXRange,
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
