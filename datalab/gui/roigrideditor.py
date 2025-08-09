# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""ROI Grid editor"""

from __future__ import annotations

from typing import Any

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
from plotpy.plot import PlotDialog, PlotOptions
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, ImageROI
from sigima.params import ROIGridParam
from sigima.proc.image import generate_image_grid_roi

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.config import _


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
        self.__roi = ImageROI()
        self.__is_ready = False
        self.obj = obj = obj.copy()  # Avoid modifying the original object
        obj.roi = None  # Clear the ROI to avoid conflicts with the editor
        gridparam = gridparam or ROIGridParam()
        displayparam = displayparam or DisplayParam()
        self.gridparamwidget = gdq.DataSetEditGroupBox(
            _("Grid parameters"), ROIGridParam, show_button=False
        )
        self.gridparamwidget.dataset.on_geometry_changed = (
            lambda: self.SIG_GEOMETRY_CHANGED.emit()
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

    def get_roi(self) -> ImageROI:
        """Get the current ROI"""
        return self.__roi

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
        roi = generate_image_grid_roi(self.obj, self.gridparamwidget.dataset)
        self.__roi = self.obj.roi = roi
        if update_item:
            self.update_items()
