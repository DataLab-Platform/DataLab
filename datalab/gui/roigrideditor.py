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
from datalab.config import DEBUG, _
from datalab.env import execenv


class DirectionChoice(ChoiceEnum):
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

    base_name = gds.StringItem(_("Base name"), default="ROI").set_prop(
        "display", callback=geometry_changed
    )
    name_pattern = gds.StringItem(
        _("Name pattern"), default="{base}({r},{c})"
    ).set_prop("display", callback=geometry_changed)
    ny = gds.IntItem(
        "N<sub>y</sub> (%s)" % _("rows"), default=1, nonzero=True
    ).set_prop("display", callback=geometry_changed)
    nx = (
        gds.IntItem("N<sub>x</sub> (%s)" % _("columns"), default=1, nonzero=True)
        .set_prop("display", callback=geometry_changed)
        .set_pos(col=1)
    )
    xdirection = gds.ChoiceItem(_("X direction"), DirectionChoice.choices()).set_prop(
        "display", callback=geometry_changed
    )
    ydirection = (
        gds.ChoiceItem(
            _("Y direction"),
            DirectionChoice.choices(),
            default=DirectionChoice.DECREASING.value,
        )
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
        self.init_param()

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

    def init_param(self) -> None:
        """Initialize parameters"""
        rows, cols = self.obj.data.shape
        gp = self.gridparamwidget.dataset
        gp.nx = gp.ny = 3
        gp.xtranslation = gp.ytranslation = gp.xsize = gp.ysize = 50
        self.gridparamwidget.get()
        self.update_obj(update_item=False)

    def __get_roi_coords(
        self, i_row: int, i_col: int
    ) -> list[float, float, float, float]:
        """Get the coordinates of the ROI for the given row and column indices"""
        obj = self.obj
        gp = self.gridparamwidget.dataset

        xtrans = obj.width * (float(gp.xtranslation) - 50.0) / 100.0
        ytrans = obj.height * (float(gp.ytranslation) - 50.0) / 100.0

        dx_max = obj.width / gp.nx
        dx = dx_max * float(gp.xsize) / 100.0
        dy_max = obj.height / gp.ny
        dy = dy_max * float(gp.ysize) / 100.0

        x1 = obj.x0 + (i_col + 0.5) * dx_max + xtrans - 0.5 * dx
        y1 = obj.y0 + (i_row + 0.5) * dy_max + ytrans - 0.5 * dy

        return [x1, y1, dx, dy]

    def update_obj(self, update_item: bool = True) -> None:
        """Update the object with the current parameters"""
        roi = ImageROI()
        gp = self.gridparamwidget.dataset
        # Iterate over grid cells, taking into account the number of columns and rows,
        # the direction of increasing rows and columns
        row_list = list(range(gp.ny))
        col_list = list(range(gp.nx))
        if gp.ydirection == DirectionChoice.DECREASING.value:
            row_list = list(range(gp.ny - 1, -1, -1))
        if gp.xdirection == DirectionChoice.DECREASING.value:
            col_list = list(range(gp.nx - 1, -1, -1))
        ptn: str = gp.name_pattern
        for i_row in row_list:
            for i_col in col_list:
                try:
                    title = ptn.format(base=gp.base_name, r=i_row + 1, c=i_col + 1)
                except Exception:  # pylint: disable=broad-except
                    title = f"{gp.base_name}({i_row + 1}, {i_col + 1})"
                x0, y0, dx, dy = self.__get_roi_coords(i_row, i_col)
                coords = [x0, y0, dx, dy]
                if DEBUG:
                    execenv.print(f"Creating ROI: {title} with coords {coords}")
                single_roi = RectangularROI(coords, indices=False, title=title)
                roi.add_roi(single_roi)
        self.obj.roi = roi
        if update_item:
            self.update_items()

    def get_roi(self) -> ImageROI:
        """Get the current ROI"""
        return self.obj.roi.copy()
