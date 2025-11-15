# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Signal Module
----------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import numpy as np
from guidata.dataset import restore_dataset, update_dataset
from plotpy.builder import make
from plotpy.items import CurveItem
from sigima.objects import SignalObj

from datalab.adapters_plotpy.objects.base import (
    BaseObjPlotPyAdapter,
)
from datalab.config import Conf


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

    def apply_style(self, item: CurveItem) -> None:
        """Apply style to curve

        Args:
            item: curve item
        """
        if self.__suspend:
            # Suspend mode: always apply the first style
            color, linestyle = CurveStyles.COLORS[0], CurveStyles.LINESTYLES[0]
        else:
            color, linestyle = next(self.curve_style)
        item.param.line.color = color
        item.param.line.style = linestyle
        item.param.symbol.marker = "NoSymbol"
        # Note: line width is set separately via apply_line_width()
        # to ensure it's always recalculated based on current data size and settings

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


def apply_line_width(item: CurveItem) -> None:
    """Apply line width to curve item with smart clamping for large datasets

    Args:
        item: curve item
    """
    # Get data size
    data_size = item.get_data()[0].size

    # Get configured line width
    line_width = Conf.view.sig_linewidth.get()

    # For large datasets, clamp linewidth to 1.0 for performance
    # (thick lines cause ~10x rendering slowdown due to Qt raster engine)
    threshold = Conf.view.sig_linewidth_perfs_threshold.get()
    if data_size > threshold and line_width > 1.0:
        line_width = 1.0

    # Apply the line width
    item.param.line.width = line_width


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
        if len(o.xydata) in (2, 4):
            assert isinstance(o.xydata, np.ndarray)
            if len(o.xydata) == 2:  # x, y signal
                x, y = o.xydata
                item = make.mcurve(x.real, y.real, label=o.title)
            else:  # x, y, dx, dy error bar signal
                x, y, dx, dy = o.xydata
                if o.dx is None and o.dy is None:  # x, y signal with no error
                    item = make.mcurve(x.real, y.real, label=o.title)
                elif o.dx is None:  # x, y, dy error bar signal with y error
                    item = make.merror(x.real, y.real, dy.real, label=o.title)
                else:  # x, y, dx, dy error bar signal with x error
                    dy = np.zeros_like(y) if dy is None else dy
                    item = make.merror(x.real, y.real, dx.real, dy.real, label=o.title)
            # Apply style (without linewidth, will be set separately)
            CURVESTYLES.apply_style(item)
            apply_downsampling(item, do_not_update=True)
            # Apply linewidth with smart clamping based on actual data size
            apply_line_width(item)
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
        # Reapply linewidth with smart clamping (data size may have changed)
        apply_line_width(item)
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
