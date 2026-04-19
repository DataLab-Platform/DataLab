# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image panel (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Type
from weakref import ReferenceType, ref

from plotpy.interfaces import IVoiImageItemType
from plotpy.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    LabelTool,
)
from sigima.io.image import ImageIORegistry
from sigima.objects import ImageDatatypes, ImageObj, ImageROI, NewImageParam

from datalab.config import Conf, _
from datalab.gui import roieditor
from datalab.gui.actionhandler import ImageActionHandler
from datalab.gui.newobject import create_image_gui
from datalab.gui.panel.base import BaseDataPanel
from datalab.gui.plothandler import ImagePlotHandler
from datalab.gui.processor.image import ImageProcessor
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from plotpy.plot import BasePlot
    from qtpy import QtWidgets as QW

    from datalab.gui.docks import DockablePlotWidget


class ImagePanel(BaseDataPanel[ImageObj, ImageROI, roieditor.ImageROIEditor]):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = _("Image Panel")
    PANEL_STR_ID = "image"
    PARAMCLASS = ImageObj
    MINDIALOGSIZE = (800, 800)

    # The following tools are used to create annotations on images. The annotation
    # items are created using PlotPy's default settings. Those appearance settings
    # may be modified in the configuration (see `datalab.config`).
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

    # pylint: disable=duplicate-code

    @staticmethod
    def get_roi_class() -> Type[ImageROI]:
        """Return ROI class"""
        return ImageROI

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
        self._contrast_sync_in_progress = False
        self._contrast_editors: dict[
            str, list[ReferenceType[roieditor.ImageROIEditor]]
        ] = {}
        self.plothandler = ImagePlotHandler(self, dockableplotwidget.plotwidget)
        self.processor = ImageProcessor(self, dockableplotwidget.plotwidget)
        view_toolbar = dockableplotwidget.toolbar
        self.acthandler = ImageActionHandler(self, panel_toolbar, view_toolbar)

    def register_contrast_editor(
        self, obj: ImageObj, editor: roieditor.ImageROIEditor
    ) -> None:
        """Register an image ROI editor for contrast synchronization."""
        obj_uuid = get_uuid(obj)
        editors = self._contrast_editors.setdefault(obj_uuid, [])
        for editor_ref in list(editors):
            current_editor = editor_ref()
            if current_editor is None:
                editors.remove(editor_ref)
            elif current_editor is editor:
                return
        editors.append(ref(editor))
        item = self.plothandler.get(obj_uuid)
        if item is not None:
            zmin, zmax = item.get_lut_range()
            editor.apply_shared_contrast(zmin, zmax)

    def _update_contrast_panel_range(self, zmin: float, zmax: float) -> None:
        """Update contrast panel range without re-emitting LUT signals."""
        contrast = self.plothandler.plotwidget.manager.get_contrast_panel()
        if contrast is None:
            return
        contrast.histogram.range.set_range(zmin, zmax, dosignal=False)
        contrast.histogram.replot()

    def _update_object_contrast_state(
        self, obj: ImageObj, zmin: float, zmax: float, update_panel: bool = False
    ) -> None:
        """Update object and current panel state for a contrast change."""
        obj.zscalemin, obj.zscalemax = zmin, zmax
        if obj is self.objview.get_current_object():
            self.objprop.update_properties_from(obj)
            if update_panel:
                self._update_contrast_panel_range(zmin, zmax)

    def apply_shared_contrast(
        self,
        obj: ImageObj,
        zmin: float,
        zmax: float,
        source: roieditor.ImageROIEditor | None = None,
    ) -> None:
        """Apply a contrast change coming from another view."""
        del source  # unused: kept for API symmetry with _notify_contrast_editors
        self._update_object_contrast_state(obj, zmin, zmax, update_panel=True)
        item = self.plothandler.get(get_uuid(obj))
        if item is None:
            return
        if item.get_lut_range() == (zmin, zmax):
            return
        self._contrast_sync_in_progress = True
        try:
            item.set_lut_range((zmin, zmax))
            plot = self.plothandler.plot
            plot.update_colormap_axis(item)
            plot.notify_colormap_changed()
        finally:
            self._contrast_sync_in_progress = False

    def _notify_contrast_editors(
        self,
        obj: ImageObj,
        zmin: float,
        zmax: float,
        source: roieditor.ImageROIEditor | None = None,
    ) -> None:
        """Propagate a contrast change to all ROI editors of an image."""
        obj_uuid = get_uuid(obj)
        editors = self._contrast_editors.get(obj_uuid)
        if not editors:
            return
        alive_editors: list[ReferenceType[roieditor.ImageROIEditor]] = []
        for editor_ref in editors:
            editor = editor_ref()
            if editor is None:
                continue
            alive_editors.append(editor_ref)
            if editor is not source:
                editor.apply_shared_contrast(zmin, zmax)
        if alive_editors:
            self._contrast_editors[obj_uuid] = alive_editors
        else:
            self._contrast_editors.pop(obj_uuid, None)

    def _get_lut_changed_objects(
        self, plot: BasePlot
    ) -> list[tuple[ImageObj, float, float]]:
        """Return image objects affected by a LUT change on the plot."""
        changed_objects: list[tuple[ImageObj, float, float]] = []
        items = plot.get_selected_items(item_type=IVoiImageItemType)
        if not items:
            active_item = plot.get_last_active_item(IVoiImageItemType)
            items = [] if active_item is None else [active_item]
        for item in items:
            obj = self.plothandler.get_obj_from_item(item)
            if not isinstance(obj, ImageObj):
                continue
            zmin, zmax = item.get_lut_range()
            changed_objects.append((obj, zmin, zmax))
        return changed_objects

    # ------Refreshing GUI--------------------------------------------------------------
    def plot_lut_changed(self, plot: BasePlot) -> None:
        """The LUT of the plot has changed: updating image objects accordingly

        Args:
            plot: Plot object
        """
        for obj, zmin, zmax in self._get_lut_changed_objects(plot):
            self._update_object_contrast_state(obj, zmin, zmax, update_panel=True)
            if not self._contrast_sync_in_progress:
                self._notify_contrast_editors(obj, zmin, zmax)

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
            newparam = NewImageParam()
        if title is not None:
            newparam.title = title
        if curobj is not None and Conf.proc.use_image_dims.get(True):
            # Use current image dimensions for new image:
            newparam.height, newparam.width = curobj.data.shape
            newparam.dtype = ImageDatatypes.from_numpy_dtype(curobj.data.dtype)
        return newparam

    def new_object(
        self,
        param: NewImageParam | None = None,
        edit: bool = False,
        add_to_panel: bool = True,
    ) -> ImageObj | None:
        """Create a new object (image).

        Args:
            param (guidata.dataset.DataSet): new object parameters
            edit (bool): Open a dialog box to edit parameters (default: False).
             When False, the object is created with default parameters and creation
             parameters are stored in metadata for interactive editing.
            add_to_panel (bool): Add the object to the panel (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        param = self.get_newparam_from_current(param)
        image = create_image_gui(param, edit=edit, parent=self.parentWidget())
        if image is None:
            return None
        if add_to_panel:
            self.add_object(image)
        return image

    def toggle_show_contrast(self, state: bool) -> None:
        """Toggle show contrast option"""
        Conf.view.show_contrast.set(state)
        self.refresh_plot("selected", True, False)
