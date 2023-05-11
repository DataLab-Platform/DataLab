# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""DataLab Image Panel"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

from guiqwt.plot import ImageDialog
from guiqwt.tools import (
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
    ImageParam,
    create_image_from_param,
    new_image_param,
)

if TYPE_CHECKING:
    import guidata.dataset.datatypes as gdt
    from guiqwt.plot import ImageWidget
    from qtpy import QtWidgets as QW

    from cdl.core.model.image import ImageParamNew


class ImagePanel(BaseDataPanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = _("Image panel")
    PARAMCLASS = ImageParam
    DIALOGCLASS = ImageDialog
    DIALOGSIZE = (800, 800)
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
    ROIDIALOGOPTIONS = dict(show_itemlist=True, show_contrast=False)
    ROIDIALOGCLASS = roieditor.ImageROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent: QW.QWidget, plotwidget: ImageWidget, toolbar) -> None:
        super().__init__(parent, plotwidget, toolbar)
        self.plothandler = ImagePlotHandler(self, plotwidget)
        self.processor = ImageProcessor(self, plotwidget)
        self.acthandler = ImageActionHandler(self, toolbar)

    # ------Refreshing GUI--------------------------------------------------------------
    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: updating signal"""
        obj = self.objview.get_current_object()
        if obj is not None:
            obj.invalidate_maskdata_cache()
            super().properties_changed()

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(
        self,
        newparam: ImageParamNew | None = None,
        addparam: gdt.DataSet | None = None,
        edit: bool = True,
    ) -> ImageParam | None:
        """Create a new object (image).

        Args:
            newparam (DataSet): new object parameters
            addparam (DataSet): additional parameters
            edit (bool): Open a dialog box to edit parameters (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        curobj: ImageParam = self.objview.get_current_object()
        if curobj is not None:
            newparam = newparam if newparam is not None else new_image_param()
            newparam.width, newparam.height = curobj.size
            newparam.dtype = ImageDatatypes.from_dtype(curobj.data.dtype)
        image = create_image_from_param(
            newparam, addparam=addparam, edit=edit, parent=self
        )
        if image is None:
            return None
        self.add_object(image)
        return image

    def toggle_show_contrast(self, state: bool) -> None:
        """Toggle show contrast option"""
        Conf.view.show_contrast.set(state)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
