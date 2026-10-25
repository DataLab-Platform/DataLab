# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile dialog
==============

The :mod:`datalab.gui.profiledialog` module provides the profile extraction dialog.

.. autoclass:: ProfileExtractionDialog
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import qtpy.QtCore as QC
import sigima.params
from guidata.configtools import get_icon
from plotpy.interfaces import ICurveItemType
from plotpy.items import AnnotatedPoint, AnnotatedRectangle, AnnotatedSegment
from plotpy.panels.csection.cswidget import LineCrossSection
from plotpy.plot import PlotDialog, PlotOptions
from plotpy.tools import AverageCrossSectionTool, CrossSectionTool, LineCrossSectionTool
from qtpy import QtWidgets as QW
from qtpy.QtWidgets import QWidget

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import _
from datalab.utils.qthelpers import resize_widget_to_parent

if TYPE_CHECKING:
    from plotpy.items import CurveItem
    from plotpy.panels import XCrossSection, YCrossSection
    from plotpy.plot import BasePlot
    from sigima.objects import ImageObj


class ProfileExtractionDialog(PlotDialog):
    """Profile extraction dialog

    Args:
        mode: "line", "segment" or "rectangle"
        options: Plot options
        parent: Parent widget
    """

    def __init__(
        self,
        mode: Literal["line", "segment", "rectangle"],
        param: sigima.params.LineProfileParam
        | sigima.params.SegmentProfileParam
        | sigima.params.AverageProfileParam,
        options: PlotOptions | None = None,
        parent: QWidget | None = None,
        add_initial_shape: bool = False,
    ) -> None:
        self.__obj: ImageObj | None = None
        self.__default_curve_color: str | None = None
        assert mode in ("line", "segment", "rectangle")
        self.mode = mode
        self.param = param
        title = param.get_title()
        self.__add_initial_shape = add_initial_shape
        if options is None:
            options = PlotOptions(show_contrast=True, show_axes_tab=False)
        options.show_xsection = options.show_ysection = True
        toolklass = {
            "line": CrossSectionTool,
            "segment": LineCrossSectionTool,
            "rectangle": AverageCrossSectionTool,
        }[mode]
        super().__init__(title=title, edit=True, options=options, parent=parent)
        self.setObjectName("profileextraction")
        self.setWindowIcon(get_icon("DataLab.svg"))
        resize_widget_to_parent(self, parent, aspect_ratio=1.0)
        mgr = self.get_manager()
        tool: (
            CrossSectionTool | LineCrossSectionTool | AverageCrossSectionTool | None
        ) = mgr.get_tool(toolklass)
        if tool is None:
            tool = mgr.add_tool(toolklass)
        self.cstool = tool
        self.shape: AnnotatedPoint | AnnotatedSegment | AnnotatedRectangle | None = None
        self.param_btn = pbtn = QW.QPushButton(_("Edit profile parameters"))
        pbtn.setIcon(get_icon("edit.svg"))
        pbtn.clicked.connect(self.edit_values)
        pbtn.setEnabled(False)
        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(False)
        tool.SIG_TOOL_JOB_FINISHED.connect(self.__tool_job_finished)
        self.button_layout.insertWidget(0, pbtn)
        rbtn = QW.QPushButton(_("Reset selection"))
        rbtn.setIcon(get_icon("reset.svg"))
        rbtn.clicked.connect(self.reset_to_initial)
        self.button_layout.insertWidget(1, rbtn)
        self.update_cs_panels_state()
        self.get_plot().SIG_ITEMS_CHANGED.connect(self.update_cs_panels_state)

    def populate_plot_layout(self):
        """Populate the plot layout"""
        super().populate_plot_layout()
        lcs_panel = LineCrossSection(self)
        splitter = self.plot_widget.xcsw_splitter
        splitter.addWidget(lcs_panel)
        splitter.setSizes([0, 1, 0])
        self.manager.add_panel(lcs_panel)
        # Install event filters on cross-section plots to detect clicks
        mgr = self.get_manager()
        xcs_panel = mgr.get_xcs_panel()
        ycs_panel = mgr.get_ycs_panel()
        if xcs_panel is not None:
            xcs_panel.cs_plot.canvas().installEventFilter(self)
            # Store reference to know which plot was clicked
            xcs_panel.cs_plot.canvas().setProperty("cs_direction", "horizontal")
        if ycs_panel is not None:
            ycs_panel.cs_plot.canvas().installEventFilter(self)
            ycs_panel.cs_plot.canvas().setProperty("cs_direction", "vertical")

    def __set_curve_enable_state(self, curve: CurveItem, state: bool) -> None:
        """Set curve enable state

        Args:
            curve: Curve item
            state: Enable state
        """
        if self.__default_curve_color is None:
            self.__default_curve_color = curve.param.line.color
        color = self.__default_curve_color if state else "#666666"
        curve.param.line.color = color
        curve.param.update_item(curve)

    def __get_curve(self, plot: BasePlot) -> CurveItem | None:
        """Get cross-section curve

        Args:
            plot: Plot
        """
        curves = plot.get_items(item_type=ICurveItemType)
        if curves:
            return curves[0]
        return None

    def __update_xycs_panel_state(
        self, cs_panel: XCrossSection | YCrossSection, state: bool
    ) -> None:
        """Update cross-section panel state

        Args:
            cs_panel: Cross-section panel
            state: Enable state
        """
        cs_panel.setEnabled(state)
        cs_plot = cs_panel.cs_plot
        curve = self.__get_curve(cs_plot)
        if curve:
            self.__set_curve_enable_state(curve, state)

    def __on_cs_plot_clicked(self, direction: str) -> None:
        """Handle click on cross-section plot to switch direction

        Args:
            direction: "horizontal" or "vertical"
        """
        # Only handle clicks for line and rectangle modes
        if self.mode not in ("line", "rectangle"):
            return
        # Switch direction if different
        if self.param.direction != direction:
            self.param.direction = direction
            self.update_cs_panels_state()
            self.get_plot().replot()

    def eventFilter(self, obj: QC.QObject, event: QC.QEvent) -> bool:
        """Event filter to detect clicks on cross-section plots

        Args:
            obj: Object
            event: Event

        Returns:
            True if event was handled, False otherwise
        """
        # Check if this is a cross-section plot canvas
        direction = obj.property("cs_direction")
        if direction:
            # Handle mouse press to switch direction
            if event.type() == QC.QEvent.MouseButtonPress:
                self.__on_cs_plot_clicked(direction)
                # Consume the event to prevent curve selection
                return True
        return super().eventFilter(obj, event)

    def update_cs_panels_state(self):
        """Enable or disable X or Y cross-section panels depending on the direction
        of the desired profile extraction."""
        mgr = self.get_manager()
        lcs_panel = mgr.get_panel(LineCrossSection.PANEL_ID)
        xcs_panel, ycs_panel = mgr.get_xcs_panel(), mgr.get_ycs_panel()
        xymode = self.mode in ("line", "rectangle")
        lcs_panel.setVisible(not xymode)
        xcs_panel.setVisible(xymode)
        ycs_panel.setVisible(xymode)
        if xymode:
            p = self.param
            self.__update_xycs_panel_state(xcs_panel, p.direction == "horizontal")
            self.__update_xycs_panel_state(ycs_panel, p.direction == "vertical")

    def accept(self) -> None:
        """Accept"""
        if self.shape is not None:
            self.shape_to_param()
        super().accept()

    def reset_to_initial(self) -> None:
        """Reset to initial"""
        self.param_btn.setEnabled(False)
        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(False)
        # Reset parameters *in place* instead of replacing `self.param` with a
        # new instance: callers (e.g. `ImageProcessor.compute_average_profile`)
        # keep their own reference to the original `param` object and reuse it
        # once the dialog is closed. Replacing the instance here would break
        # that reference, so that further updates made by `shape_to_param()`
        # (triggered when a new selection is drawn) would silently be lost,
        # and the caller would keep using the discarded selection's
        # coordinates (see issue #322).
        self.param.set_defaults()
        plot = self.get_plot()
        if self.shape is not None:
            plot.del_item(self.shape)
            self.shape = None
        self.cstool.activate()
        self.update_cs_panels_state()
        plot.replot()

    def __tool_job_finished(self):
        """Tool job finished"""
        self.param_btn.setEnabled(True)
        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(True)
        self.shape = self.cstool.get_last_final_shape()
        assert self.shape is not None
        self.shape.set_readonly(True)
        self.shape_to_param()
        self.update_cs_panels_state()

    def __clip_to_pixel_centers(self, x: float, y: float) -> tuple[float, float]:
        """Clip a point to the range of the pixel centers of the image

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Clipped coordinates (physical units)
        """
        assert self.__obj is not None
        obj = self.__obj
        nrows, ncols = obj.data.shape
        xb = obj.axis_to_physical(np.array([0.0, ncols - 1.0]), "x")
        yb = obj.axis_to_physical(np.array([0.0, nrows - 1.0]), "y")
        x = float(np.clip(x, np.min(xb), np.max(xb)))
        y = float(np.clip(y, np.min(yb), np.max(yb)))
        return x, y

    def __fit_param_to_image(self) -> None:
        """Bring the profile parameters back into the image (e.g. the parameters
        of the last profile, extracted from another image)"""
        assert self.__obj is not None
        clip = self.__clip_to_pixel_centers
        p = self.param
        if isinstance(p, sigima.params.LineProfileParam):
            p.x, p.y = clip(p.x, p.y)
        elif isinstance(p, sigima.params.SegmentProfileParam):
            (p.x1, p.y1), (p.x2, p.y2) = clip(p.x1, p.y1), clip(p.x2, p.y2)
        else:
            nrows, ncols = self.__obj.data.shape
            x0, y0, x1, y1 = self.__obj.pixel_range_to_physical([0, 0, ncols, nrows])
            xmin, xmax, ymin, ymax = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            for name, vmin, vmax in (
                ("x1", xmin, xmax),
                ("x2", xmin, xmax),
                ("y1", ymin, ymax),
                ("y2", ymin, ymax),
            ):
                value = getattr(p, name)
                if value is not None:
                    setattr(p, name, float(np.clip(value, vmin, vmax)))

    def shape_to_param(self) -> None:
        """Shape to param (profile parameters are in physical coordinates)"""
        assert self.__obj is not None
        clip = self.__clip_to_pixel_centers
        p = self.param
        if isinstance(self.shape, AnnotatedPoint):
            assert isinstance(p, sigima.params.LineProfileParam)
            p.x, p.y = clip(*self.shape.get_pos())
        elif isinstance(self.shape, AnnotatedSegment):
            assert isinstance(p, sigima.params.SegmentProfileParam)
            x1, y1, x2, y2 = self.shape.get_rect()
            (p.x1, p.y1), (p.x2, p.y2) = clip(x1, y1), clip(x2, y2)
            if p.x1 > p.x2:
                p.x1, p.x2 = p.x2, p.x1
                p.y1, p.y2 = p.y2, p.y1
        else:
            assert isinstance(p, sigima.params.AverageProfileParam)
            x1, y1, x2, y2 = self.shape.get_rect()
            p.x1, p.x2 = sorted((float(x1), float(x2)))
            p.y1, p.y2 = sorted((float(y1), float(y2)))

    def param_to_shape(self) -> None:
        """Param to shape"""
        assert self.__obj is not None
        p = self.param
        if isinstance(self.shape, AnnotatedPoint):
            assert isinstance(p, sigima.params.LineProfileParam)
            self.shape.set_pos(p.x, p.y)
        elif isinstance(self.shape, AnnotatedSegment):
            assert isinstance(p, sigima.params.SegmentProfileParam)
            self.shape.set_rect(p.x1, p.y1, p.x2, p.y2)
        else:
            assert isinstance(p, sigima.params.AverageProfileParam)
            # Undefined bounds stand for the edges of the image
            nrows, ncols = self.__obj.data.shape
            edges = self.__obj.pixel_range_to_physical([0, 0, ncols, nrows])
            coords = [p.x1, p.y1, p.x2, p.y2]
            x1, y1, x2, y2 = [e if c is None else c for c, e in zip(coords, edges)]
            self.shape.set_rect(x1, y1, x2, y2)

    def edit_values(self) -> None:
        """Edit values"""
        self.shape_to_param()
        if self.param.edit(parent=self, apply=lambda _param: self.apply_callback()):
            self.param_to_shape()
            self.update_cs_panels_state()
            self.get_plot().replot()

    def apply_callback(self) -> None:
        """Apply callback"""
        self.param_to_shape()
        self.update_cs_panels_state()
        self.get_plot().replot()

    def set_obj(self, obj: ImageObj):
        """Set object

        Args:
            obj: Image object
        """
        self.__obj = obj
        item = create_adapter_from_object(obj).make_item()
        item.set_readonly(True)
        item.set_resizable(False)
        item.set_rotatable(False)
        item.set_selectable(False)
        plot = self.get_plot()
        plot.add_item(item)
        plot.set_active_item(item)
        item.unselect()
        if self.__add_initial_shape:
            plot = self.get_plot()
            self.show()  # To be able to convert axes to canvas coordinates
            p0 = QC.QPointF(0.0, 0.0)
            self.cstool.add_shape_to_plot(plot, p0, p0)
            self.shape = self.cstool.get_last_final_shape()
            self.__fit_param_to_image()
            self.param_to_shape()
            self.param_btn.setEnabled(True)
            self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.cstool.activate()
