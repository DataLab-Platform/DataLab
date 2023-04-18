# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""CobraDataLab Image Panel"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

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
from cdl.core.gui import actionhandler, plotitemlist, roieditor
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.processor.image import ImageProcessor
from cdl.core.io.image import ImageIORegistry
from cdl.core.model.image import (
    ImageDatatypes,
    ImageParam,
    create_image_from_param,
    new_image_param,
)


class ImagePanel(BaseDataPanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = _("Image List")
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
    PREFIX = "i"
    IO_REGISTRY = ImageIORegistry
    H5_PREFIX = "CDL_Ima"
    ROIDIALOGOPTIONS = dict(show_itemlist=True, show_contrast=False)
    ROIDIALOGCLASS = roieditor.ImageROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = plotitemlist.ImageItemList(self, self.objlist, plotwidget)
        self.processor = proc = ImageProcessor(self, self.objlist, plotwidget)
        self.acthandler = actionhandler.ImageActionHandler(self, proc, toolbar)

    # ------Refreshing GUI--------------------------------------------------------------
    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.objlist.currentRow()
        self.objlist[row].invalidate_maskdata_cache()
        super().properties_changed()

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new image.
        :param cdl.core.model.image.ImageNewParam newparam: new image parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """
        if not self.mainwindow.confirm_memory_state():
            return
        curobj: ImageParam = self.objlist.get_sel_object(-1)
        if curobj is not None:
            newparam = newparam if newparam is not None else new_image_param()
            newparam.width, newparam.height = curobj.size
            newparam.dtype = ImageDatatypes.from_dtype(curobj.data.dtype)
        image = create_image_from_param(
            newparam, addparam=addparam, edit=edit, parent=self
        )
        if image is not None:
            self.add_object(image)

    def toggle_show_contrast(self, state: bool) -> None:
        """Toggle show contrast option"""
        Conf.view.show_contrast.set(state)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
