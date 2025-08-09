# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""ROI Grid editor"""

from __future__ import annotations

from typing import Any

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
from plotpy.plot import PlotDialog, PlotOptions
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, ImageROI, RectangularROI
from sigima.objects.base import ChoiceEnum

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.config import _


class Direction(ChoiceEnum):
    """Direction choice"""

    INCREASING = _("increasing")
    DECREASING = _("decreasing")


class ROIGridParam(gds.DataSet):
    """ROI Grid parameters"""

    def __init__(self, roi_editor: ImageGridROIEditor | None = None) -> None:
        super().__init__()
        self.roi_editor = roi_editor

    def geometry_changed(self, item, value) -> None:
        """Emit geometry changed signal."""
        assert self.roi_editor is not None, "ROI editor is not set"
        if self.roi_editor.is_ready():
            self.roi_editor.SIG_GEOMETRY_CHANGED.emit()

    _b_group0 = gds.BeginGroup(_("Geometry"))
    ny = gds.IntItem(
        "N<sub>y</sub> (%s)" % _("rows"), default=3, nonzero=True
    ).set_prop("display", callback=geometry_changed)
    nx = (
        gds.IntItem("N<sub>x</sub> (%s)" % _("columns"), default=3, nonzero=True)
        .set_prop("display", callback=geometry_changed)
        .set_pos(col=1)
    )
    xtranslation = gds.IntItem(
        _("X translation"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    ytranslation = gds.IntItem(
        _("Y translation"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    xsize = gds.IntItem(
        "X size (%s)" % _("column size"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    ysize = gds.IntItem(
        "Y size (%s)" % _("row size"),
        default=50,
        min=0,
        max=100,
        unit="%",
        slider=True,
    ).set_prop("display", callback=geometry_changed)
    _e_group0 = gds.EndGroup(_("Geometry"))
    _b_group1 = gds.BeginGroup(_("ROI titles"))
    base_name = gds.StringItem(_("Base name"), default="ROI").set_prop(
        "display", callback=geometry_changed
    )
    name_pattern = gds.StringItem(
        _("Name pattern"), default="{base}({r},{c})"
    ).set_prop("display", callback=geometry_changed)
    xdirection = gds.ChoiceItem(_("X direction"), Direction.choices()).set_prop(
        "display", callback=geometry_changed
    )
    ydirection = (
        gds.ChoiceItem(_("Y direction"), Direction.choices())
        .set_prop("display", callback=geometry_changed)
        .set_pos(col=1)
    )
    _e_group1 = gds.EndGroup(_("ROI titles"))


class DisplayParam(gds.DataSet):
    """ROI Grid display parameters"""

    def __init__(self, roi_editor: ImageGridROIEditor | None = None) -> None:
        super().__init__()
        self.roi_editor = roi_editor

    def display_changed(self, item, value) -> None:
        """Emit display changed signal."""
        assert self.roi_editor is not None, "ROI editor is not set"
        if self.roi_editor.is_ready():
            self.roi_editor.SIG_DISPLAY_CHANGED.emit()

    show_mask = gds.BoolItem(_("Show mask"), default=True).set_prop(
        "display", callback=display_changed
    )
    _prop_show_roi = gds.ValueProp(False)
    show_roi = gds.BoolItem(_("Show ROI"), default=True).set_prop(
        "display", callback=display_changed, store=_prop_show_roi
    )
    show_names = gds.BoolItem(_("Show names"), default=True).set_prop(
        "display", callback=display_changed, active=_prop_show_roi
    )


def add_grid_roi_to_image(obj: ImageObj, p: ROIGridParam) -> None:
    """Add a grid ROI to the image object.

    Args:
        obj: The image object to create the ROI for.
        p: ROIGridParam object containing the grid parameters.
    """
    roi = ImageROI()
    dx_cell = obj.width / p.nx
    dy_cell = obj.height / p.ny
    dx = dx_cell * p.xsize / 100.0
    dy = dy_cell * p.ysize / 100.0
    xtrans = obj.width * (p.xtranslation - 50.0) / 100.0
    ytrans = obj.height * (p.ytranslation - 50.0) / 100.0
    lbl_rows = range(p.ny)
    if p.ydirection == Direction.DECREASING:
        lbl_rows = range(p.ny - 1, -1, -1)
    lbl_cols = range(p.nx)
    if p.xdirection == Direction.DECREASING:
        lbl_cols = range(p.nx - 1, -1, -1)
    ptn: str = p.name_pattern
    for ir in range(p.ny):
        for ic in range(p.nx):
            x0 = obj.x0 + (ic + 0.5) * dx_cell + xtrans - 0.5 * dx
            y0 = obj.y0 + (ir + 0.5) * dy_cell + ytrans - 0.5 * dy
            nir, nic = lbl_rows[ir], lbl_cols[ic]
            try:
                title = ptn.format(base=p.base_name, r=nir + 1, c=nic + 1)
            except Exception:  # pylint: disable=broad-except
                title = f"ROI({nir + 1},{nic + 1})"
            roi.add_roi(RectangularROI([x0, y0, dx, dy], indices=False, title=title))
    obj.roi = roi


class ImageGridROIEditor(PlotDialog):
    """Image Grid ROI Editor.

    Args:
        parent: Parent plot dialog
        obj: Object to edit (:class:`sigima.objects.ImageObj`)
        item: Optional plot item to add to the plot (if None, a new item is created
         from the object)
    """

    SIG_GEOMETRY_CHANGED = QC.Signal()
    SIG_DISPLAY_CHANGED = QC.Signal()
    ADDITIONAL_OPTIONS = {"show_itemlist": False, "show_contrast": False}

    def __init__(
        self,
        parent: QW.QWidget | None,
        obj: ImageObj,
        gridparam: ROIGridParam | None = None,
        displayparam: DisplayParam | None = None,
        options: PlotOptions | dict[str, Any] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.__is_ready = False
        self.obj = obj = obj.copy()  # Avoid modifying the original object
        obj.roi = None  # Clear the ROI to avoid conflicts with the editor
        gridparam = gridparam or ROIGridParam()
        gridparam.roi_editor = self
        displayparam = displayparam or DisplayParam()
        displayparam.roi_editor = self
        self.gridparamwidget = gdq.DataSetEditGroupBox(
            _("Grid parameters"),
            ROIGridParam,
            show_button=False,
            roi_editor=self,
        )
        gds.update_dataset(self.gridparamwidget.dataset, gridparam)
        self.displayparamwidget = gdq.DataSetEditGroupBox(
            _("Display parameters"), DisplayParam, show_button=False, roi_editor=self
        )
        gds.update_dataset(self.displayparamwidget.dataset, displayparam)
        self.update_obj(update_item=False)

        if options is None:
            options = self.ADDITIONAL_OPTIONS
        else:
            options = options.copy(self.ADDITIONAL_OPTIONS)
        roi_s = _("ROI grid")
        super().__init__(
            parent=parent,
            toolbar=False,
            options=options,
            title=f"{roi_s} - {obj.title}",
            icon="DataLab.svg",
            edit=True,
            size=size,
        )
        self.setObjectName("i_grid_roi_editor")
        self.setup_editor_layout()
        self.update_items()
        self.SIG_GEOMETRY_CHANGED.connect(self.update_obj)
        self.SIG_DISPLAY_CHANGED.connect(self.update_items)
        self.__is_ready = True

    def is_ready(self) -> bool:
        """Check if the editor is ready for use"""
        return self.__is_ready

    def setup_layout(self) -> None:  # Reimplement PlotDialog method
        """Populate the plot layout"""
        super().setup_layout()
        self.editor_layout = QW.QVBoxLayout()
        self.plot_layout.addLayout(self.editor_layout, 0, 1)
        self.plot_layout.setColumnStretch(0, 2)
        self.plot_layout.setColumnStretch(1, 1)

    def setup_editor_layout(self) -> None:
        """Setup editor layout"""
        self.editor_layout.addWidget(self.gridparamwidget)
        self.editor_layout.addWidget(self.displayparamwidget)
        self.editor_layout.addStretch()
        self.editor_layout.addSpacing(100)

    def update_items(self) -> None:
        """Setup items"""
        dp = self.displayparamwidget.dataset
        obj = self.obj
        plot = self.get_plot()
        plot.del_all_items()
        item = create_adapter_from_object(obj).make_item()
        item.set_mask_visible(dp.show_mask)
        item.set_selectable(False)
        item.set_readonly(True)
        plot.add_item(item)
        plot.set_active_item(item)
        item.unselect()
        if dp.show_roi:
            fmt = create_adapter_from_object(obj).get_obj_option("format")
            roi_adapter = create_adapter_from_object(obj.roi)
            for ritem in roi_adapter.iterate_roi_items(obj, fmt, dp.show_names, False):
                plot.add_item(ritem)
        plot.replot()

    def update_obj(self, update_item: bool = True) -> None:
        """Update the object with the current parameters"""
        add_grid_roi_to_image(self.obj, self.gridparamwidget.dataset)
        if update_item:
            self.update_items()

    def get_roi(self) -> ImageROI:
        """Get the current ROI"""
        return self.obj.roi.copy()
