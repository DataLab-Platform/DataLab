# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Image GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import os.path as osp

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions
from guiqwt.io import imread, imwrite, iohandler
from guiqwt.plot import ImageDialog
from guiqwt.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    FreeFormTool,
    LabelTool,
)
from qtpy import QtWidgets as QW
from qtpy.compat import getsavefilename

from codraft.config import Conf, _
from codraft.core.gui import base as guibase
from codraft.core.gui.processor.image import ImageProcessor
from codraft.core.model.image import (
    ImageDatatypes,
    ImageParam,
    create_image,
    create_image_from_param,
    new_image_param,
)
from codraft.utils.qthelpers import qt_try_loadsave_file, save_restore_stds


class ImageItemList(guibase.BaseItemList):
    """Object handling image plot items, plot dialogs, plot options"""

    def cleanup_dataview(self):
        """Clean up data view"""
        for widget in (self.plotwidget.xcsw, self.plotwidget.ycsw):
            widget.hide()
        super().cleanup_dataview()

    def get_current_plot_options(self):
        """
        Return standard signal/image plot options

        :return: Dictionary containing plot arguments for CurveDialog/ImageDialog
        """
        options = super().get_current_plot_options()
        options.update(
            dict(
                zlabel=self.plot.get_axis_title("right"),
                zunit=self.plot.get_axis_unit("right"),
                show_contrast=True,
            )
        )
        return options


class ImageActionHandler(guibase.BaseActionHandler):
    """Object handling image panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("image")

    def create_operation_actions(self):
        """Create operation actions"""
        base_actions = super().create_operation_actions()
        proc = self.processor
        rotate_menu = QW.QMenu(_("Rotation"), self.panel)
        hflip_act = self.cra(
            _("Flip horizontally"),
            triggered=proc.flip_horizontally,
            icon=get_icon("flip_horizontally.svg"),
        )
        vflip_act = self.cra(
            _("Flip vertically"),
            triggered=proc.flip_vertically,
            icon=get_icon("flip_vertically.svg"),
        )
        rot90_act = self.cra(
            _("Rotate %s right") % "90°",  # pylint: disable=consider-using-f-string
            triggered=proc.rotate_270,
            icon=get_icon("rotate_right.svg"),
        )
        rot270_act = self.cra(
            _("Rotate %s left") % "90°",  # pylint: disable=consider-using-f-string
            triggered=proc.rotate_90,
            icon=get_icon("rotate_left.svg"),
        )
        rotate_act = self.cra(
            _("Rotate arbitrarily..."), triggered=proc.rotate_arbitrarily
        )
        resize_act = self.cra(_("Resize"), triggered=proc.resize_image)
        logp1_act = self.cra("Log10(z+n)", triggered=proc.compute_logp1)
        flatfield_act = self.cra(
            _("Flat-field correction"), triggered=proc.flat_field_correction
        )
        self.actlist_2 += [flatfield_act]
        self.actlist_1more += [
            resize_act,
            hflip_act,
            vflip_act,
            logp1_act,
            rot90_act,
            rot270_act,
            rotate_act,
        ]
        add_actions(
            rotate_menu, [hflip_act, vflip_act, rot90_act, rot270_act, rotate_act]
        )
        roi_actions = self.operation_end_actions
        actions = [
            logp1_act,
            flatfield_act,
            None,
            rotate_menu,
            None,
            resize_act,
        ]
        return base_actions + actions + roi_actions

    def create_computing_actions(self):
        """Create computing actions"""
        base_actions = super().create_computing_actions()
        proc = self.processor
        # TODO: [P3] Add "Create ROI grid..." action to create a regular grid or ROIs
        cent_act = self.cra(
            _("Centroid"), proc.compute_centroid, tip=_("Compute image centroid")
        )
        encl_act = self.cra(
            _("Minimum enclosing circle center"),
            proc.compute_enclosing_circle,
            tip=_("Compute smallest enclosing circle center"),
        )
        peak_act = self.cra(
            _("2D peak detection"),
            proc.compute_peak_detection,
            tip=_("Compute automatic 2D peak detection"),
        )
        contour_act = self.cra(
            _("Contour detection"),
            proc.compute_contour_shape,
            tip=_("Compute contour shape fit"),
        )
        self.actlist_1more += [cent_act, encl_act, peak_act, contour_act]
        return base_actions + [cent_act, encl_act, peak_act, contour_act]


class ImageROIEditor(guibase.BaseROIEditor):
    """Image ROI Editor"""

    ICON_NAME = "image_roi_new.svg"

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        for index, roi_item in enumerate(self.roi_items):
            roi_item.annotationparam.title = f"ROI{index:02d}"
            roi_item.annotationparam.update_annotation(roi_item)

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        return roi_item.get_rect()


class ImagePanel(guibase.BasePanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = "Image Panel"
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
        FreeFormTool,
    )
    PREFIX = "i"
    OPEN_FILTERS = iohandler.get_filters("load", dtype=None)
    H5_PREFIX = "CodraFT_Ima"
    ROIDIALOGOPTIONS = dict(show_itemlist=True, show_contrast=False)
    ROIDIALOGCLASS = ImageROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = ImageItemList(self, self.objlist, plotwidget)
        self.processor = ImageProcessor(self, self.objlist)
        self.acthandler = ImageActionHandler(
            self, self.objlist, self.itmlist, self.processor, toolbar
        )
        self.setup_panel()

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new image.

        :param codraft.core.model.image.ImageNewParam newparam: new image parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """
        if not self.mainwindow.confirm_memory_state():
            return
        curobj = self.objlist.get_sel_object(-1)
        if curobj is not None:
            newparam = newparam if newparam is not None else new_image_param()
            newparam.width, newparam.height = curobj.size
            newparam.dtype = ImageDatatypes.from_dtype(curobj.data.dtype)
        image = create_image_from_param(
            newparam, addparam=addparam, edit=edit, parent=self
        )
        if image is not None:
            self.add_object(image)

    def open_object(self, filename: str) -> None:
        """Open object from file (signal/image)"""
        data = imread(filename, to_grayscale=False)
        if filename.lower().endswith(".sif") and len(data.shape) == 3:
            for idx in range(data.shape[0]):
                image = create_image(
                    osp.basename(filename) + "_Im" + str(idx), data[idx, ::]
                )
                self.add_object(image)
        else:
            if data.ndim == 3:
                # Converting to grayscale
                data = data[..., :4].mean(axis=2)
            image = create_image(osp.basename(filename), data)
            if osp.splitext(filename)[1].lower() == ".dcm":
                from pydicom import dicomio  # pylint: disable=C0415,E0401

                image.dicom_template = dicomio.read_file(
                    filename, stop_before_pixels=True, force=True
                )
            self.add_object(image)

    def save_object(self, obj, filename: str = None) -> None:
        """Save object to file (signal/image)"""
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getsavefilename(  # pylint: disable=duplicate-code
                    self,
                    _("Save as"),
                    basedir,
                    iohandler.get_filters(
                        "save", dtype=obj.data.dtype, template=obj.dicom_template
                    ),
                )
        if filename:
            Conf.main.base_dir.set(filename)
            kwargs = {}
            if osp.splitext(filename)[1].lower() == ".dcm":
                kwargs["template"] = obj.dicom_template
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                imwrite(filename, obj.data, **kwargs)
