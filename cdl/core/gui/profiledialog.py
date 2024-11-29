# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile dialog
==============

The :mod:`cdl.core.gui.profiledialog` module provides the profile extraction dialog.

.. autoclass:: ProfileExtractionDialog
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import qtpy.QtCore as QC
from guidata.configtools import get_icon
from plotpy.interfaces import ICurveItemType
from plotpy.items import AnnotatedPoint, AnnotatedRectangle, AnnotatedSegment
from plotpy.panels.csection.cswidget import LineCrossSection
from plotpy.plot import PlotDialog, PlotOptions
from plotpy.tools import AverageCrossSectionTool, CrossSectionTool, LineCrossSectionTool
from qtpy import QtWidgets as QW
from qtpy.QtWidgets import QWidget

import cdl.param
from cdl.config import _

if TYPE_CHECKING:
    from plotpy.items import CurveItem
    from plotpy.panels import XCrossSection, YCrossSection
    from plotpy.plot import BasePlot

    from cdl.core.model.image import ImageObj


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
        param: cdl.param.LineProfileParam
        | cdl.param.SegmentProfileParam
        | cdl.param.AverageProfileParam,
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
            options = PlotOptions(show_contrast=True)
        options.show_xsection = options.show_ysection = True
        toolklass = {
            "line": CrossSectionTool,
            "segment": LineCrossSectionTool,
            "rectangle": AverageCrossSectionTool,
        }[mode]
        super().__init__(title=title, edit=True, options=options, parent=parent)
        self.setObjectName("profileextraction")
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.resize(800, 800)
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
        if self.mode == "line":
            self.param = cdl.param.LineProfileParam()
        elif self.mode == "segment":
            self.param = cdl.param.SegmentProfileParam()
        else:
            self.param = cdl.param.AverageProfileParam()
        plot = self.get_plot()
        if self.shape is not None:
            plot.del_item(self.shape)
        self.cstool.activate()
        self.update_cs_panels_state()
        self.get_plot().replot()

    def __tool_job_finished(self):
        """Tool job finished"""
        self.param_btn.setEnabled(True)
        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(True)
        self.shape = self.cstool.get_last_final_shape()
        assert self.shape is not None
        self.shape.set_readonly(True)
        self.shape_to_param()
        self.update_cs_panels_state()

    def shape_to_param(self) -> None:
        """Shape to param"""
        assert self.__obj is not None
        p2i = self.__obj.physical_to_indices
        p = self.param
        if isinstance(self.shape, AnnotatedPoint):
            assert isinstance(p, cdl.param.LineProfileParam)
            p.col, p.row = p2i(self.shape.get_pos())
        elif isinstance(self.shape, AnnotatedSegment):
            assert isinstance(p, cdl.param.SegmentProfileParam)
            p.col1, p.row1, p.col2, p.row2 = p2i(self.shape.get_rect())
            if p.col1 > p.col2:
                p.col1, p.col2 = p.col2, p.col1
                p.row1, p.row2 = p.row2, p.row1
        else:
            assert isinstance(p, cdl.param.AverageProfileParam)
            p.col1, p.row1, p.col2, p.row2 = p2i(self.shape.get_rect())
            if p.col1 > p.col2:
                p.col1, p.col2 = p.col2, p.col1
                p.row1, p.row2 = p.row2, p.row1

    def param_to_shape(self) -> None:
        """Param to shape"""
        assert self.__obj is not None
        i2p = self.__obj.indices_to_physical
        p = self.param
        if isinstance(self.shape, AnnotatedPoint):
            assert isinstance(p, cdl.param.LineProfileParam)
            x, y = i2p([p.col, p.row])
            self.shape.set_pos(x, y)
        elif isinstance(self.shape, AnnotatedSegment):
            assert isinstance(p, cdl.param.SegmentProfileParam)
            x1, y1, x2, y2 = i2p([p.col1, p.row1, p.col2, p.row2])
            self.shape.set_rect(x1, y1, x2, y2)
        else:
            assert isinstance(p, cdl.param.AverageProfileParam)
            x1, y1, x2, y2 = i2p([p.col1, p.row1, p.col2, p.row2])
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
        item = obj.make_item()
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
            self.param_to_shape()
            self.param_btn.setEnabled(True)
            self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.cstool.activate()
