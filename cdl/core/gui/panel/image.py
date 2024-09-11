# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image panel (see parent package :mod:`cdl.core.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from plotpy.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    LabelTool,
)

from cdl.config import Conf, _
from cdl.core.gui import roieditor
from cdl.core.gui.actionhandler import ImageActionHandler
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.plothandler import ImagePlotHandler
from cdl.core.gui.processor.image import ImageProcessor
from cdl.core.io.image import ImageIORegistry
from cdl.core.model.image import (
    ImageDatatypes,
    ImageObj,
    ImageROI,
    create_image_from_param,
    new_image_param,
)

if TYPE_CHECKING:
    import guidata.dataset as gds
    from plotpy.plot import BasePlot
    from qtpy import QtWidgets as QW

    from cdl.core.gui.docks import DockablePlotWidget
    from cdl.core.model.image import NewImageParam


class ImagePanel(BaseDataPanel[ImageObj, ImageROI, roieditor.ImageROIEditor]):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = _("Image Panel")
    PANEL_STR_ID = "image"
    PARAMCLASS = ImageObj
    MINDIALOGSIZE = (800, 800)
    ANNOTATION_TOOLS = (
        AnnotatedCircleTool,
        AnnotatedSegmentTool,
        AnnotatedRectangleTool,
        AnnotatedPointTool,
        AnnotatedEllipseTool,
        LabelTool,
    )
    IO_REGISTRY = ImageIORegistry
    H5_PREFIX = "DataLab_Ima"
    ROIDIALOGOPTIONS = {"show_itemlist": True, "show_contrast": False}

    # pylint: disable=duplicate-code

    @staticmethod
    def get_roieditor_class() -> Type[roieditor.ImageROIEditor]:
        """Return ROI editor class"""
        return roieditor.ImageROIEditor

    def __init__(
        self,
        parent: QW.QWidget,
        dockableplotwidget: DockablePlotWidget,
        panel_toolbar: QW.QToolBar,
    ) -> None:
        super().__init__(parent)
        self.plothandler = ImagePlotHandler(self, dockableplotwidget.plotwidget)
        self.processor = ImageProcessor(self, dockableplotwidget.plotwidget)
        view_toolbar = dockableplotwidget.toolbar
        self.acthandler = ImageActionHandler(self, panel_toolbar, view_toolbar)

    # ------Refreshing GUI--------------------------------------------------------------
    def plot_lut_changed(self, plot: BasePlot) -> None:
        """The LUT of the plot has changed: updating image objects accordingly

        Args:
            plot: Plot object
        """
        zmin, zmax = plot.get_axis_limits(plot.colormap_axis)
        for obj in self.objview.get_sel_objects():
            obj.zscalemin, obj.zscalemax = zmin, zmax
            if obj is self.objview.get_current_object():
                self.objprop.update_properties_from(obj)

    # ------Creating, adding, removing objects------------------------------------------
    def get_newparam_from_current(
        self, newparam: NewImageParam | None = None, title: str | None = None
    ) -> NewImageParam | None:
        """Get new object parameters from the current object.

        Args:
            newparam (guidata.dataset.DataSet): new object parameters.
             If None, create a new one.
            title: new object title. If None, use the current object title, or the
             default title.

        Returns:
            New object parameters
        """
        curobj: ImageObj = self.objview.get_current_object()
        if newparam is None:
            newparam = new_image_param(title=title)
        if curobj is not None:
            newparam.width, newparam.height = curobj.data.shape
            newparam.dtype = ImageDatatypes.from_dtype(curobj.data.dtype)
        return newparam

    def new_object(
        self,
        newparam: NewImageParam | None = None,
        addparam: gds.DataSet | None = None,
        edit: bool = True,
        add_to_panel: bool = True,
    ) -> ImageObj | None:
        """Create a new object (image).

        Args:
            newparam (Daguidata.dataset.datatypes.DataSettaSet): new object parameters
            addparam (guidata.dataset.DataSet): additional parameters
            edit (bool): Open a dialog box to edit parameters (default: True)
            add_to_panel (bool): Add the object to the panel (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        newparam = self.get_newparam_from_current(newparam)
        image = create_image_from_param(
            newparam, addparam=addparam, edit=edit, parent=self.parent()
        )
        if image is None:
            return None
        if add_to_panel:
            self.add_object(image)
        return image

    def toggle_show_contrast(self, state: bool) -> None:
        """Toggle show contrast option"""
        Conf.view.show_contrast.set(state)
        self.SIG_REFRESH_PLOT.emit("selected", True)
