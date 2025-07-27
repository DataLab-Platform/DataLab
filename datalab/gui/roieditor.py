# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor
==========

The :mod:`datalab.gui.roieditor` module provides the ROI editor widgets
for signals and images.

Signal ROI editor
-----------------

.. autoclass:: SignalROIEditor
    :members:

Image ROI editor
----------------

.. autoclass:: ImageROIEditor
    :members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from plotpy.builder import make
from plotpy.items import (
    AnnotatedCircle,
    AnnotatedPolygon,
    AnnotatedRectangle,
    CurveItem,
    MaskedImageItem,
    ObjectInfo,
    XRangeSelection,
)
from plotpy.plot import PlotDialog, PlotManager, PlotOptions
from plotpy.tools import CircleTool, HRangeTool, PolygonTool, RectangleTool, SelectTool
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import (
    CircularROI,
    ImageObj,
    ImageROI,
    PolygonalROI,
    RectangularROI,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    SignalObj,
    SignalROI,
    TypeObj,
    TypeROI,
)

from datalab.adapters_plotpy import (
    TypePlotItem,
    TypeROIItem,
    configure_roi_item,
)
from datalab.adapters_plotpy.converters import (
    plotitem_to_singleroi,
)
from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.config import Conf, _
from datalab.env import execenv

if TYPE_CHECKING:
    from plotpy.plot import BasePlot
    from plotpy.tools.base import InteractiveTool


def configure_roi_item_in_tool(shape, obj: SignalObj | ImageObj) -> None:
    """Configure ROI item in tool"""
    fmt = create_adapter_from_object(obj).get_obj_option("format")
    configure_roi_item(shape, fmt, lbl=True, editable=True, option=obj.PREFIX)


def tool_deselect_items(tool: InteractiveTool) -> None:
    """Deselect all items in plot associated with the tool"""
    plot = tool.get_active_plot()
    plot.select_some_items([])  # Deselect all items


def tool_setup_shape(shape: TypeROIItem, obj: SignalObj | ImageObj) -> None:
    """Tool setup shape"""
    shape.setTitle("ROI")
    configure_roi_item_in_tool(shape, obj)


class ROISegmentTool(HRangeTool):
    """ROI segment tool"""

    TITLE = _("Range ROI")
    ICON = "signal_roi.svg"

    def __init__(self, manager: PlotManager, obj: SignalObj) -> None:
        super().__init__(manager, switch_to_default_tool=False, toolbar_id=None)
        self.roi = SegmentROI([0, 1], False)
        self.obj = obj

    def activate(self):
        """Activate tool"""
        tool_deselect_items(self)
        super().activate()

    def create_shape(self) -> XRangeSelection:
        """Create shape"""
        shape = create_adapter_from_object(self.roi).to_plot_item(self.obj)
        configure_roi_item_in_tool(shape, self.obj)
        return shape


class ROIRectangleTool(RectangleTool):
    """ROI rectangle tool"""

    TITLE = _("Rectangular ROI")
    ICON = "roi_new_rectangle.svg"

    def __init__(self, manager: PlotManager, obj: ImageObj) -> None:
        super().__init__(
            manager,
            switch_to_default_tool=False,
            toolbar_id=None,
            setup_shape_cb=tool_setup_shape,
        )
        self.roi = RectangularROI([0, 0, 1, 1], True)
        self.obj = obj

    def activate(self):
        """Activate tool"""
        tool_deselect_items(self)
        super().activate()

    def create_shape(self) -> tuple[AnnotatedRectangle, int, int]:
        """Create shape"""
        item = create_adapter_from_object(self.roi).to_plot_item(self.obj)
        return item, 0, 2

    def setup_shape(self, shape: AnnotatedRectangle) -> None:
        """Setup shape"""
        tool_setup_shape(shape, self.obj)


class ROICircleTool(CircleTool):
    """ROI circle tool"""

    TITLE = _("Circular ROI")
    ICON = "roi_new_circle.svg"

    def __init__(self, manager: PlotManager, obj: ImageObj) -> None:
        super().__init__(
            manager,
            switch_to_default_tool=False,
            toolbar_id=None,
            setup_shape_cb=tool_setup_shape,
        )
        self.roi = CircularROI([0, 0, 1], True)
        self.obj = obj

    def activate(self):
        """Activate tool"""
        tool_deselect_items(self)
        super().activate()

    def create_shape(self) -> tuple[AnnotatedCircle, int, int]:
        """Create shape"""
        item = create_adapter_from_object(self.roi).to_plot_item(self.obj)
        return item, 0, 1

    def setup_shape(self, shape: AnnotatedCircle) -> None:
        """Setup shape"""
        tool_setup_shape(shape, self.obj)


class ROIPolygonTool(PolygonTool):
    """ROI polygon tool"""

    TITLE = _("Polygonal ROI")
    ICON = "roi_new_polygon.svg"

    def __init__(self, manager: PlotManager, obj: ImageObj) -> None:
        super().__init__(
            manager,
            switch_to_default_tool=False,
            toolbar_id=None,
            setup_shape_cb=tool_setup_shape,
        )
        self.roi = PolygonalROI([[0, 0], [1, 0], [1, 1], [0, 1]], True)
        self.obj = obj

    def activate(self):
        """Activate tool"""
        tool_deselect_items(self)
        super().activate()

    def create_shape(self) -> tuple[AnnotatedPolygon, int, int]:
        """Create shape"""
        return create_adapter_from_object(self.roi).to_plot_item(self.obj)

    def setup_shape(self, shape: AnnotatedPolygon) -> None:
        """Setup shape"""
        tool_setup_shape(shape, self.obj)


TypeROIEditor = TypeVar("TypeROIEditor", bound="BaseROIEditor")


class BaseROIEditorMeta(type(PlotDialog), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(
    PlotDialog,
    Generic[TypeObj, TypeROI, TypePlotItem, TypeROIItem],  # type: ignore
    metaclass=BaseROIEditorMeta,
):
    """ROI Editor

    Args:
        parent: Parent plot dialog
        obj: Object to edit (:class:`sigima.objects.SignalObj` or
         :class:`sigima.objects.ImageObj`)
        mode: Mode of operation, either "apply" (define ROI, then apply it to
         selected objects), "extract" (define ROI, then extract data from it),
         or "define" (define ROI without applying or extracting).
        item: Optional plot item to add to the plot (if None, a new item is created
         from the object)
    """

    ICON_NAME = None
    OBJ_NAME = None
    ROI_ITEM_TYPES = ()
    ADDITIONAL_OPTIONS: dict[str, Any] | None = None

    def __init__(
        self,
        parent: QW.QWidget | None,
        obj: SignalObj | ImageObj,
        mode: Literal["apply", "extract", "define"] = "apply",
        item: TypePlotItem | None = None,
        options: PlotOptions | dict[str, Any] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        assert mode in ("apply", "extract", "define"), (
            f"Invalid mode: {mode}. Must be either 'apply', 'extract' or 'define'."
        )
        self.editor_layout: QW.QHBoxLayout | None = None
        self.remove_all_action: QW.QAction | None = None
        self.singleobj_btn: QW.QToolButton | None = None
        self.obj = obj
        self.mode = mode
        if item is None:
            item = create_adapter_from_object(obj).make_item()
        self.main_item = item
        self.__modified: bool | None = None
        self._tools: list[InteractiveTool] = []

        # Create ROI associated with the object
        if mode == "define":
            # In "define" mode, we do not want to show the object's ROI to avoid any
            # confusion with the ROI being defined
            roi = None
        else:
            # In "apply" or "extract" mode, we want to show the object's ROI
            # if it exists, otherwise create a new empty ROI
            roi = obj.roi
        if roi is None:
            roi = self.get_obj_roi_class()()
        if roi.singleobj is None:
            roi.singleobj = Conf.proc.extract_roi_singleobj.get()
        self.__roi: TypeROI = roi

        roi_s = _("Regions of interest")
        super().__init__(
            parent=parent,
            toolbar=mode != "define",
            options=options,
            title=f"{roi_s} - {obj.title}",
            icon="DataLab.svg",
            edit=True,
            size=size,
        )
        self.setObjectName(f"{obj.PREFIX}_roi_dialog")
        self.accepted.connect(self.dialog_accepted)

        # Create the ROI items
        fmt = create_adapter_from_object(obj).get_obj_option("format")
        roi_adapter = create_adapter_from_object(self.__roi)
        self.roi_items: list[TypeROIItem] = list(
            roi_adapter.iterate_roi_items(obj, fmt, True, True)
        )

        self.toolbar = QW.QToolBar(self)

        self.add_tools_to_plot_dialog()
        self.setup_editor_layout()
        self.setup_items()

        # force update of ROI titles and remove_all_btn state
        self.items_changed(self.get_plot())

        self.get_plot().SIG_ITEMS_CHANGED.connect(self.items_changed)
        self.get_plot().SIG_ITEM_REMOVED.connect(self.item_removed)
        self.get_plot().SIG_RANGE_CHANGED.connect(
            lambda _rng, _min, _max: self.item_moved()
        )
        self.get_plot().SIG_ANNOTATION_CHANGED.connect(lambda _plt: self.item_moved())

        #  In "extract mode", the dialog box OK button should always been enabled
        #  when at least one ROI is defined,
        #  whereas in non-extract mode (when editing ROIs) the OK button is by default
        #  disabled (until ROI data is modified)
        if self.mode == "extract":
            self.modified = len(self.roi_items) > 0
        else:
            self.modified = False

    def setup_layout(self) -> None:  # Reimplement PlotDialog method
        """Populate the plot layout"""
        super().setup_layout()
        self.editor_layout = QW.QHBoxLayout()
        self.button_layout.insertLayout(0, self.editor_layout)

    def setup_editor_layout(self) -> None:
        """Setup editor layout"""
        self.toolbar.setToolButtonStyle(QC.Qt.ToolButtonTextUnderIcon)
        add_actions(self.toolbar, self.create_actions())
        for action in self.toolbar.actions():
            if action.menu() is not None:
                widget = self.toolbar.widgetForAction(action)
                widget.setPopupMode(QW.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.editor_layout.addWidget(self.toolbar)
        if self.mode == "extract":
            self.singleobj_btn = QW.QCheckBox(
                _("Extract all ROIs\ninto a single %s") % self.OBJ_NAME,
                self,
            )
            self.editor_layout.addWidget(self.singleobj_btn)
            self.singleobj_btn.setChecked(self.__roi.singleobj)
        self.editor_layout.addStretch()

    def setup_items(self) -> None:
        """Setup items"""
        self.main_item.set_selectable(False)
        self.main_item.set_readonly(True)
        self.get_plot().add_item(self.main_item)
        for roi_item in self.roi_items:
            self.get_plot().add_item(roi_item)
            self.get_plot().set_active_item(roi_item)

    @abc.abstractmethod
    def get_obj_roi_class(self) -> type[TypeROI]:
        """Get object ROI class"""

    def add_tools_to_plot_dialog(self) -> None:
        """Add tools to plot dialog"""
        mgr = self.get_manager()
        select_tool = mgr.get_tool(SelectTool)
        add_actions(self.toolbar, [select_tool.action])

    @property
    def modified(self) -> bool:
        """Return dialog modified state"""
        return self.__modified

    @modified.setter
    def modified(self, value: bool) -> None:
        """Set dialog modified state"""
        self.__modified = value
        if self.mode in ("extract", "define"):
            # In "extract" mode or in "define" mode, OK button is enabled
            # when at least one ROI is defined
            value = value and len(self.roi_items) > 0
        self.button_box.button(QW.QDialogButtonBox.Ok).setEnabled(value)

    def dialog_accepted(self) -> None:
        """Parent dialog was accepted: updating ROI Editor data"""
        self.__roi.empty()
        for roi_item in self.roi_items:
            self.__roi.add_roi(plotitem_to_singleroi(roi_item))
        if self.singleobj_btn is not None:
            singleobj = self.singleobj_btn.isChecked()
            self.__roi.singleobj = singleobj
            Conf.proc.extract_roi_singleobj.set(singleobj)

    def get_roieditor_results(self) -> tuple[TypeROI, bool]:
        """Get ROI editor results

        Returns:
            A tuple containing the ROI data parameters and ROI modified state.
            ROI modified state is True if the ROI data has been modified within
            the dialog box.
        """
        return self.__roi, self.modified

    def create_coordinate_based_roi_actions(self) -> list[QW.QAction]:
        """Create coordinate-based ROI actions"""
        return []

    def __new_action_menu(
        self, title: str, icon: str, actions: list[QW.QAction]
    ) -> QW.QAction:
        """Create a new action menu"""
        menu = QW.QMenu(title)
        for action in actions:
            menu.addAction(action)
        action = QW.QAction(get_icon(icon), title, self)
        action.setMenu(menu)
        return action

    def create_actions(self) -> list[QW.QAction]:
        """Create actions"""
        g_menu_act = self.__new_action_menu(
            _("Graphical ROI"),
            "roi_graphical.svg",
            [tool.action for tool in self._tools],
        )
        c_menu_act = self.__new_action_menu(
            _("Coordinate-based ROI"),
            "roi_coordinate.svg",
            self.create_coordinate_based_roi_actions(),
        )
        self.remove_all_action = create_action(
            self,
            _("Remove all"),
            icon=get_icon("roi_delete.svg"),
            triggered=self.remove_all_rois,
        )
        return [g_menu_act, c_menu_act, None, self.remove_all_action]

    def remove_all_rois(self) -> None:
        """Remove all ROIs"""
        if (
            execenv.unattended
            or QW.QMessageBox.question(
                self,
                _("Remove all ROIs"),
                _("Are you sure you want to remove all ROIs?"),
            )
            == QW.QMessageBox.Yes
        ):
            self.get_plot().del_items(self.roi_items)

    @abc.abstractmethod
    def update_roi_titles(self) -> None:
        """Update ROI annotation titles"""

    def update_roi_items(self) -> None:
        """Update ROI items"""
        old_nb_items = len(self.roi_items)
        self.roi_items = [
            item
            for item in self.get_plot().get_items()
            if isinstance(item, self.ROI_ITEM_TYPES)
        ]
        self.get_plot().select_some_items([])
        self.update_roi_titles()
        if old_nb_items != len(self.roi_items):
            self.modified = True

    def items_changed(self, _plot: BasePlot) -> None:
        """Items have changed"""
        self.update_roi_items()
        self.remove_all_action.setEnabled(len(self.roi_items) > 0)

    def item_removed(self, item) -> None:
        """Item was removed. Since all items are read-only except ROIs...
        this must be an ROI."""
        assert item in self.roi_items
        self.update_roi_items()
        self.modified = True

    def item_moved(self) -> None:
        """ROI plot item has just been moved"""
        self.modified = True


class ROIRangeInfo(ObjectInfo):
    """ObjectInfo for ROI selection"""

    def __init__(self, roi_items: list[TypeROIItem]) -> None:
        self.roi_items = roi_items

    def get_text(self) -> str:
        textlist = []
        for index, roi_item in enumerate(self.roi_items):
            x0, x1 = roi_item.get_range()
            textlist.append(f"ROI{index:02d}: {x0} ≤ x ≤ {x1}")
        return "<br>".join(textlist)


class SignalROIEditor(BaseROIEditor[SignalObj, SignalROI, CurveItem, XRangeSelection]):
    """Signal ROI Editor

    Args:
        parent: Parent plot dialog
        obj: Object to edit (:class:`sigima.objects.SignalObj` or
         :class:`sigima.objects.ImageObj`)
        mode: Mode of operation, either "apply" (define ROI, then apply it to
         selected objects), "extract" (define ROI, then extract data from it),
         or "define" (define ROI without applying or extracting).
        item: Optional plot item to add to the plot (if None, a new item is created
         from the object)
    """

    ICON_NAME = "signal_roi.svg"
    OBJ_NAME = _("signal")
    ROI_ITEM_TYPES = (XRangeSelection,)

    def get_obj_roi_class(self) -> type[SignalROI]:
        """Get object ROI class"""
        return SignalROI

    def add_tools_to_plot_dialog(self) -> None:
        """Add tools to plot dialog"""
        super().add_tools_to_plot_dialog()
        mgr = self.get_manager()
        segm_tool = mgr.add_tool(ROISegmentTool, self.obj)
        self._tools.append(segm_tool)
        segm_tool.activate()

    def manually_add_roi(self) -> None:
        """Manually add segment ROI"""
        param = ROI1DParam()
        if param.edit(parent=self):
            segment_roi = param.to_single_roi(self.obj)
            shape = create_adapter_from_object(segment_roi).to_plot_item(self.obj)
            configure_roi_item_in_tool(shape, self.obj)
            self.get_plot().add_item(shape)
            self.get_plot().set_active_item(shape)

    def create_coordinate_based_roi_actions(self) -> list[QW.QAction]:
        """Create coordinate-based ROI actions"""
        segcoord_act = create_action(
            self,
            _("Range ROI"),
            icon=get_icon("signal_roi.svg"),
            triggered=self.manually_add_roi,
        )
        return [segcoord_act]

    def setup_items(self) -> None:
        """Setup items"""
        super().setup_items()
        info = ROIRangeInfo(self.roi_items)
        info_label = make.info_label("BL", info, title=_("Regions of interest"))
        self.get_plot().add_item(info_label)
        self.info_label = info_label

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        super().update_roi_titles()
        self.info_label.update_text()


class ImageROIEditor(
    BaseROIEditor[
        ImageObj,
        ImageROI,
        MaskedImageItem,
        # `Union` is mandatory here for Python 3.9-3.10 compatibility:
        Union[AnnotatedPolygon, AnnotatedRectangle, AnnotatedCircle],
    ]
):
    """Image ROI Editor

    Args:
        parent: Parent plot dialog
        obj: Object to edit (:class:`sigima.objects.SignalObj` or
         :class:`sigima.objects.ImageObj`)
        mode: Mode of operation, either "apply" (define ROI, then apply it to
         selected objects), "extract" (define ROI, then extract data from it),
         or "define" (define ROI without applying or extracting).
        item: Optional plot item to add to the plot (if None, a new item is created
         from the object)
    """

    ICON_NAME = "image_roi.svg"
    OBJ_NAME = _("image")
    ROI_ITEM_TYPES = (AnnotatedRectangle, AnnotatedCircle, AnnotatedPolygon)
    ADDITIONAL_OPTIONS = {"show_itemlist": True, "show_contrast": False}

    def get_obj_roi_class(self) -> type[ImageROI]:
        """Get object ROI class"""
        return ImageROI

    def add_tools_to_plot_dialog(self) -> None:
        """Add tools to plot dialog"""
        super().add_tools_to_plot_dialog()
        mgr = self.get_manager()
        rect_tool = mgr.add_tool(ROIRectangleTool, self.obj)
        circ_tool = mgr.add_tool(ROICircleTool, self.obj)
        poly_tool = mgr.add_tool(ROIPolygonTool, self.obj)
        self._tools.extend([rect_tool, circ_tool, poly_tool])
        rect_tool.activate()

    def manually_add_roi(
        self, roi_type: Literal["rectangle", "circle", "polygon"]
    ) -> None:
        """Manually add image ROI"""
        assert roi_type in ("rectangle", "circle", "polygon")
        if roi_type == "polygon":
            raise NotImplementedError("Manual polygonal ROI creation is not supported")
        param = ROI2DParam()
        param.geometry = roi_type
        if param.edit(parent=self):
            roi = param.to_single_roi(self.obj)
            shape = create_adapter_from_object(roi).to_plot_item(self.obj)
            configure_roi_item_in_tool(shape, self.obj)
            self.get_plot().add_item(shape)
            self.get_plot().set_active_item(shape)

    def create_coordinate_based_roi_actions(self) -> list[QW.QAction]:
        """Create coordinate-based ROI actions"""
        rectcoord_act = create_action(
            self,
            _("Rectangular ROI"),
            icon=get_icon("roi_new_rectangle.svg"),
            triggered=lambda: self.manually_add_roi("rectangle"),
        )
        circcoord_act = create_action(
            self,
            _("Circular ROI"),
            icon=get_icon("roi_new_circle.svg"),
            triggered=lambda: self.manually_add_roi("circle"),
        )
        return [rectcoord_act, circcoord_act]

    def setup_items(self) -> None:
        """Setup items"""
        super().setup_items()
        # In "define" mode, we want to show the mask that is associated with the
        # image's ROI, if it exists, because the ROI that is being defined has nothing
        # to do with the image's ROI: it is just a temporary ROI that will for example
        # be used in a processing function, so the user need to visualize the mask that
        # corresponds to the data on which the processing function will be applied.
        self.main_item.set_mask_visible(self.mode == "define")

    def update_roi_titles(self) -> None:
        """Update ROI annotation titles"""
        super().update_roi_titles()
        for index, roi_item in enumerate(self.roi_items):
            roi_item: AnnotatedRectangle | AnnotatedCircle | AnnotatedPolygon
            roi_item.annotationparam.title = f"ROI{index:02d}"
            roi_item.annotationparam.update_item(roi_item)
