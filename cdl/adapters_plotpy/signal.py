# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Signal Module
----------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import numpy as np
from guidata.dataset import restore_dataset, update_dataset
from plotpy.builder import make
from plotpy.items import CurveItem, XRangeSelection
from sigima.obj import SegmentROI, SignalObj, SignalROI

from cdl.adapters_plotpy.base import (
    BaseObjPlotPyAdapter,
    BaseROIPlotPyAdapter,
    BaseSingleROIPlotPyAdapter,
)
from cdl.config import Conf

if TYPE_CHECKING:
    from plotpy.styles import CurveParam


class SegmentROIPlotPyAdapter(BaseSingleROIPlotPyAdapter[SegmentROI, XRangeSelection]):
    """Segment ROI plot item adapter

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    # pylint: disable=unused-argument
    def to_plot_item(self, obj: SignalObj, title: str | None = None) -> XRangeSelection:
        """Make and return the annnotated segment associated with the ROI

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            title: title
        """
        xmin, xmax = self.single_roi.get_physical_coords(obj)
        item = make.range(xmin, xmax)
        return item

    @classmethod
    def from_plot_item(cls, item: XRangeSelection) -> SegmentROI:
        """Create ROI from plot item

        Args:
            item: plot item

        Returns:
            ROI
        """
        if not isinstance(item, XRangeSelection):
            raise TypeError("Invalid plot item type")
        return SegmentROI(item.get_range(), False)


class SignalROIPlotPyAdapter(BaseROIPlotPyAdapter[SignalROI]):
    """Signal ROI plot item adapter class

    Args:
        roi: ROI object
    """

    def to_plot_item(
        self, single_roi: SegmentROI, obj: SignalObj, title: str | None = None
    ) -> XRangeSelection:
        """Make ROI plot item from single ROI

        Args:
            single_roi: single ROI object
            obj: object (signal/image), for physical-indices coordinates conversion
            title: ROI title

        Returns:
            Plot item
        """
        return SegmentROIPlotPyAdapter(single_roi).to_plot_item(obj, title)


class CurveStyles:
    """Object to manage curve styles"""

    #: Curve colors
    COLORS = (
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    )
    #: Curve line styles
    LINESTYLES = ("SolidLine", "DashLine", "DashDotLine", "DashDotDotLine")

    def __init__(self) -> None:
        self.__suspend = False
        self.curve_style = self.style_generator()

    @staticmethod
    def style_generator() -> Generator[tuple[str, str], None, None]:
        """Cycling through curve styles"""
        while True:
            for linestyle in CurveStyles.LINESTYLES:
                for color in CurveStyles.COLORS:
                    yield (color, linestyle)

    def apply_style(self, param: CurveParam) -> None:
        """Apply style to curve"""
        if self.__suspend:
            # Suspend mode: always apply the first style
            color, linestyle = CurveStyles.COLORS[0], CurveStyles.LINESTYLES[0]
        else:
            color, linestyle = next(self.curve_style)
        param.line.color = color
        param.line.style = linestyle
        param.symbol.marker = "NoSymbol"

    def reset_styles(self) -> None:
        """Reset styles"""
        self.curve_style = self.style_generator()

    @contextmanager
    def alternative(
        self, other_style_generator: Generator[tuple[str, str], None, None]
    ) -> Generator[None, None, None]:
        """Use an alternative style generator"""
        old_style_generator = self.curve_style
        self.curve_style = other_style_generator
        yield
        self.curve_style = old_style_generator

    @contextmanager
    def suspend(self) -> Generator[None, None, None]:
        """Suspend style generator"""
        self.__suspend = True
        yield
        self.__suspend = False


CURVESTYLES = CurveStyles()  # This is the unique instance of the CurveStyles class


def apply_downsampling(item: CurveItem, do_not_update: bool = False) -> None:
    """Apply downsampling to curve item

    Args:
        item: curve item
        do_not_update: if True, do not update the item even if the downsampling
         parameters have changed
    """
    old_use_dsamp = item.param.use_dsamp
    item.param.use_dsamp = False
    if Conf.view.sig_autodownsampling.get():
        nbpoints = item.get_data()[0].size
        maxpoints = Conf.view.sig_autodownsampling_maxpoints.get()
        if nbpoints > 5 * maxpoints:
            item.param.use_dsamp = True
            item.param.dsamp_factor = nbpoints // maxpoints
    if not do_not_update and old_use_dsamp != item.param.use_dsamp:
        item.update_data()


class SignalObjPlotPyAdapter(BaseObjPlotPyAdapter[SignalObj, CurveItem]):
    """Signal object plot item adapter class"""

    CONF_FMT = Conf.view.sig_format
    DEFAULT_FMT = "g"

    def update_plot_item_parameters(self, item: CurveItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        update_dataset(item.param.line, self.obj.metadata)
        update_dataset(item.param.symbol, self.obj.metadata)
        super().update_plot_item_parameters(item)

    def update_metadata_from_plot_item(self, item: CurveItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        super().update_metadata_from_plot_item(item)
        restore_dataset(item.param.line, self.obj.metadata)
        restore_dataset(item.param.symbol, self.obj.metadata)

    def make_item(self, update_from: CurveItem | None = None) -> CurveItem:
        """Make plot item from data.

        Args:
            update_from: plot item to update from

        Returns:
            Plot item
        """
        o = self.obj
        if len(o.xydata) in (2, 3, 4):
            assert isinstance(o.xydata, np.ndarray)
            if len(o.xydata) == 2:  # x, y signal
                x, y = o.xydata
                assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                item = make.mcurve(x.real, y.real, label=o.title)
            elif len(o.xydata) == 3:  # x, y, dy error bar signal
                x, y, dy = o.xydata
                assert (
                    isinstance(x, np.ndarray)
                    and isinstance(y, np.ndarray)
                    and isinstance(dy, np.ndarray)
                )
                item = make.merror(x.real, y.real, dy.real, label=o.title)
            elif len(o.xydata) == 4:  # x, y, dx, dy error bar signal
                x, y, dx, dy = o.xydata
                assert (
                    isinstance(x, np.ndarray)
                    and isinstance(y, np.ndarray)
                    and isinstance(dx, np.ndarray)
                    and isinstance(dy, np.ndarray)
                )
                item = make.merror(x.real, y.real, dx.real, dy.real, label=o.title)
            CURVESTYLES.apply_style(item.param)
            apply_downsampling(item, do_not_update=True)
        else:
            raise RuntimeError("data not supported")
        if update_from is None:
            self.update_plot_item_parameters(item)
        else:
            update_dataset(item.param, update_from.param)
            item.update_params()
        return item

    def update_item(self, item: CurveItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """
        o = self.obj
        if data_changed:
            assert isinstance(o.xydata, np.ndarray)
            if len(o.xydata) == 2:  # x, y signal
                x, y = o.xydata
                assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                item.set_data(x.real, y.real)
            elif len(o.xydata) == 3:  # x, y, dy error bar signal
                x, y, dy = o.xydata
                assert (
                    isinstance(x, np.ndarray)
                    and isinstance(y, np.ndarray)
                    and isinstance(dy, np.ndarray)
                )
                item.set_data(x.real, y.real, dy=dy.real)
            elif len(o.xydata) == 4:  # x, y, dx, dy error bar signal
                x, y, dx, dy = o.xydata
                assert (
                    isinstance(x, np.ndarray)
                    and isinstance(y, np.ndarray)
                    and isinstance(dx, np.ndarray)
                    and isinstance(dy, np.ndarray)
                )
                item.set_data(x.real, y.real, dx.real, dy.real)
        item.param.label = o.title
        apply_downsampling(item)
        self.update_plot_item_parameters(item)

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use signal title)
        """
        title = self.obj.title if title is None else title
        if title:
            label = make.label(title, "TL", (0, 0), "TL")
            self.add_annotations_from_items([label])
