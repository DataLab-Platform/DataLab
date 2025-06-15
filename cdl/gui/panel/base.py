# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Base panel objects (see parent package :mod:`cdl.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import dataclasses
import glob
import os
import os.path as osp
import re
import warnings
from typing import TYPE_CHECKING, Generic, Type

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
import numpy as np
import plotpy.io
from guidata.configtools import get_icon
from guidata.dataset import update_dataset
from guidata.io import JSONHandler
from guidata.qthelpers import add_actions, create_action, exec_dialog
from guidata.widgets.arrayeditor import ArrayEditor
from plotpy.plot import PlotDialog
from plotpy.tools import ActionTool
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import (
    getexistingdirectory,
    getopenfilename,
    getopenfilenames,
    getsavefilename,
)

from cdl import objectmodel
from cdl.adapters_plotpy.base import items_to_json
from cdl.adapters_plotpy.factories import create_adapter_from_object
from cdl.config import APP_NAME, Conf, _
from cdl.env import execenv
from cdl.gui import actionhandler, objectview
from cdl.gui.newobject import NewSignalParam
from cdl.gui.roieditor import TypeROIEditor
from cdl.objectmodel import get_uuid, set_uuid, short_id
from cdl.utils.qthelpers import (
    CallbackWorker,
    create_progress_bar,
    qt_long_callback,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)
from cdl.widgets.textimport import TextImportWizard
from sigima_ import (
    ImageObj,
    NewImageParam,
    ResultProperties,
    ResultShape,
    ShapeTypes,
    SignalObj,
    TypeObj,
    TypeROI,
    create_signal,
)
from sigima_.model.base import ROI_KEY

if TYPE_CHECKING:
    from typing import Callable

    from plotpy.items import CurveItem, LabelItem, MaskedImageItem

    from cdl.gui import ObjItf
    from cdl.gui.main import CDLMainWindow
    from cdl.gui.plothandler import ImagePlotHandler, SignalPlotHandler
    from cdl.gui.processor.image import ImageProcessor
    from cdl.gui.processor.signal import SignalProcessor
    from cdl.h5.native import NativeH5Reader, NativeH5Writer
    from sigima_.io.image import ImageIORegistry
    from sigima_.io.signal import SignalIORegistry


def is_plot_item_serializable(item: ShapeTypes) -> bool:
    """Return True if plot item is serializable"""
    try:
        plotpy.io.item_class_from_name(item.__class__.__name__)
        return True
    except AssertionError:
        return False


class ObjectProp(QW.QWidget):
    """Object handling panel properties"""

    def __init__(self, panel: BaseDataPanel, paramclass: SignalObj | ImageObj):
        super().__init__(panel)
        self.paramclass = paramclass
        self.properties = gdq.DataSetEditGroupBox(_("Properties"), paramclass)
        self.properties.SIG_APPLY_BUTTON_CLICKED.connect(panel.properties_changed)
        self.properties.setEnabled(False)

        self.add_prop_layout = QW.QHBoxLayout()
        playout: QW.QGridLayout = self.properties.edit.layout
        playout.addLayout(
            self.add_prop_layout, playout.rowCount() - 1, 0, 1, 1, QC.Qt.AlignLeft
        )

        self.param_label = QW.QLabel()
        self.param_label.setTextFormat(QC.Qt.RichText)
        self.param_label.setTextInteractionFlags(
            QC.Qt.TextBrowserInteraction | QC.Qt.TextSelectableByKeyboard
        )
        self.param_label.setAlignment(QC.Qt.AlignTop)
        param_scroll = QW.QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setWidget(self.param_label)

        child: QW.QTabWidget = None
        for child in self.properties.children():
            if isinstance(child, QW.QTabWidget):
                break
        child.addTab(param_scroll, _("Analysis parameters"))

        vlayout = QW.QVBoxLayout()
        vlayout.addWidget(self.properties)
        self.setLayout(vlayout)

    def add_button(self, button):
        """Add additional button on bottom of properties panel"""
        self.add_prop_layout.addWidget(button)

    def set_param_label(self, param: SignalObj | ImageObj):
        """Set computing parameters label"""
        text = ""
        for key, value in param.metadata.items():
            if key.endswith("Param") and isinstance(value, str):
                if text:
                    text += "<br><br>"
                lines = value.splitlines(False)
                lines[0] = f"<b>{lines[0]}</b>"
                text += "<br>".join(lines)
        self.param_label.setText(text)

    def update_properties_from(self, param: SignalObj | ImageObj | None = None):
        """Update properties from signal/image dataset"""
        self.properties.setDisabled(param is None)
        if param is None:
            param = self.paramclass()
        self.properties.dataset.set_defaults()
        update_dataset(self.properties.dataset, param)
        self.properties.get()
        self.set_param_label(param)
        self.properties.apply_button.setEnabled(False)


class AbstractPanelMeta(type(QW.QSplitter), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class AbstractPanel(QW.QSplitter, metaclass=AbstractPanelMeta):
    """Object defining DataLab panel interface,
    based on a vertical QSplitter widget

    A panel handle an object list (objects are signals, images, macros...).
    Each object must implement ``cdl.gui.ObjItf`` interface
    """

    H5_PREFIX = ""
    SIG_OBJECT_ADDED = QC.Signal()
    SIG_OBJECT_REMOVED = QC.Signal()

    @abc.abstractmethod
    def __init__(self, parent):
        super().__init__(QC.Qt.Vertical, parent)
        self.setObjectName(self.__class__.__name__[0].lower())
        # Check if the class implements __len__, __getitem__ and __iter__
        for method in ("__len__", "__getitem__", "__iter__"):
            if not hasattr(self, method):
                raise NotImplementedError(
                    f"Class {self.__class__.__name__} must implement method {method}"
                )

    # pylint: disable=unused-argument
    def get_serializable_name(self, obj: ObjItf) -> str:
        """Return serializable name of object"""
        title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
        name = f"{short_id(obj)}: {title}"
        return name

    def serialize_object_to_hdf5(self, obj: ObjItf, writer: NativeH5Writer) -> None:
        """Serialize object to HDF5 file"""
        with writer.group(self.get_serializable_name(obj)):
            obj.serialize(writer)

    def deserialize_object_from_hdf5(self, reader: NativeH5Reader, name: str) -> ObjItf:
        """Deserialize object from a HDF5 file"""
        with reader.group(name):
            obj = self.create_object()
            obj.deserialize(reader)
            set_uuid(obj)
        return obj

    @abc.abstractmethod
    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file"""

    @abc.abstractmethod
    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file"""

    @abc.abstractmethod
    def create_object(self) -> ObjItf:
        """Create and return object"""

    @abc.abstractmethod
    def add_object(self, obj: ObjItf) -> None:
        """Add object to panel"""

    @abc.abstractmethod
    def remove_all_objects(self):
        """Remove all objects"""
        self.SIG_OBJECT_REMOVED.emit()


@dataclasses.dataclass
class ResultData:
    """Result data associated to a shapetype"""

    results: list[ResultShape | ResultProperties] = None
    xlabels: list[str] = None
    ylabels: list[str] = None


def create_resultdata_dict(
    objs: list[SignalObj | ImageObj],
) -> dict[str, ResultData]:
    """Return result data dictionary

    Args:
        objs: List of objects

    Returns:
        Result data dictionary: keys are result categories, values are ResultData
    """
    rdatadict: dict[str, ResultData] = {}
    for obj in objs:
        for result in list(obj.iterate_resultshapes()) + list(
            obj.iterate_resultproperties()
        ):
            rdata = rdatadict.setdefault(result.category, ResultData([], None, []))
            rdata.results.append(result)
            rdata.xlabels = result.headers
            for i_row_res in range(result.array.shape[0]):
                ylabel = f"{result.title}({short_id(obj)})"
                i_roi = int(result.array[i_row_res, 0])
                if i_roi >= 0:
                    ylabel += f"|ROI{i_roi}"
                rdata.ylabels.append(ylabel)
    return rdatadict


class PasteMetadataParam(gds.DataSet):
    """Paste metadata parameters"""

    keep_roi = gds.BoolItem(_("Regions of interest"), default=True)
    keep_resultshapes = gds.BoolItem(_("Result shapes"), default=False).set_pos(col=1)
    keep_resultproperties = gds.BoolItem(_("Result properties"), default=False).set_pos(
        col=1
    )
    keep_other = gds.BoolItem(_("Other metadata"), default=True)


class BaseDataPanel(AbstractPanel, Generic[TypeObj, TypeROI, TypeROIEditor]):
    """Object handling the item list, the selected item properties and plot"""

    PANEL_STR = ""  # e.g. "Signal Panel"
    PANEL_STR_ID = ""  # e.g. "signal"
    PARAMCLASS: TypeObj = None  # Replaced in child object
    ANNOTATION_TOOLS = ()
    MINDIALOGSIZE = (800, 600)
    MAXDIALOGSIZE = 0.95  # % of DataLab's main window size
    # Replaced by the right class in child object:
    IO_REGISTRY: SignalIORegistry | ImageIORegistry | None = None
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_REFRESH_PLOT = QC.Signal(
        str, bool, bool, bool, bool
    )  # Connected to PlotHandler.refresh_plot
    ROIDIALOGOPTIONS = {}

    @staticmethod
    @abc.abstractmethod
    def get_roieditor_class() -> Type[TypeROIEditor]:
        """Return ROI editor class"""

    @abc.abstractmethod
    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.mainwindow: CDLMainWindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objmodel = objectmodel.ObjectModel()
        self.objview = objectview.ObjectView(self, self.objmodel)
        self.objview.SIG_IMPORT_FILES.connect(self.handle_dropped_files)
        self.objview.populate_tree()
        self.plothandler: SignalPlotHandler | ImagePlotHandler = None
        self.processor: SignalProcessor | ImageProcessor = None
        self.acthandler: actionhandler.BaseActionHandler = None
        self.__metadata_clipboard = {}
        self.context_menu = QW.QMenu()
        self.__separate_views: dict[QW.QDialog, TypeObj] = {}

    def closeEvent(self, event):
        """Reimplement QMainWindow method"""
        self.processor.close()
        super().closeEvent(event)

    # ------AbstractPanel interface-----------------------------------------------------
    def plot_item_parameters_changed(
        self, item: CurveItem | MaskedImageItem | LabelItem
    ) -> None:
        """Plot items changed: update metadata of all objects from plot items"""
        # Find the object corresponding to the plot item
        obj = self.plothandler.get_obj_from_item(item)
        if obj is not None:
            # Ensure that item's parameters are up-to-date:
            item.param.update_param(item)
            # Update object metadata from plot item parameters
            obj.update_metadata_from_plot_item(item)
            if obj is self.objview.get_current_object():
                self.objprop.update_properties_from(obj)
        self.plothandler.update_resultproperty_from_plot_item(item)

    def plot_item_moved(
        self,
        item: LabelItem,
        x0: float,  # pylint: disable=unused-argument
        y0: float,  # pylint: disable=unused-argument
        x1: float,  # pylint: disable=unused-argument
        y1: float,  # pylint: disable=unused-argument
    ) -> None:
        """Plot item moved: update metadata of all objects from plot items

        Args:
            item: Plot item
            x0: new x0 coordinate
            y0: new y0 coordinate
            x1: new x1 coordinate
            y1: new y1 coordinate
        """
        self.plothandler.update_resultproperty_from_plot_item(item)

    def serialize_object_to_hdf5(self, obj: TypeObj, writer: NativeH5Writer) -> None:
        """Serialize object to HDF5 file"""
        # Before serializing, update metadata from plot item parameters, in order to
        # save the latest visualization settings:
        try:
            item = self.plothandler[get_uuid(obj)]
            obj.update_metadata_from_plot_item(item)
        except KeyError:
            # Plot item has not been created yet (this happens when auto-refresh has
            # been disabled)
            pass
        super().serialize_object_to_hdf5(obj, writer)

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file"""
        with writer.group(self.H5_PREFIX):
            for group in self.objmodel.get_groups():
                with writer.group(self.get_serializable_name(group)):
                    with writer.group("title"):
                        writer.write_str(group.title)
                    for obj in group.get_objects():
                        self.serialize_object_to_hdf5(obj, writer)

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file"""
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                with reader.group(name):
                    group = self.add_group("")
                    with reader.group("title"):
                        group.title = reader.read_str()
                    for obj_name in reader.h5.get(f"{self.H5_PREFIX}/{name}", []):
                        obj = self.deserialize_object_from_hdf5(reader, obj_name)
                        self.add_object(obj, get_uuid(group), set_current=False)
                    self.selection_changed()

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self.objmodel)

    def __getitem__(self, nb: int) -> TypeObj:
        """Return object from its number (1 to N)"""
        return self.objmodel.get_object_from_number(nb)

    def __iter__(self):
        """Iterate over objects"""
        return iter(self.objmodel)

    def create_object(self) -> TypeObj:
        """Create object (signal or image)

        Returns:
            SignalObj or ImageObj object
        """
        return self.PARAMCLASS()  # pylint: disable=not-callable

    @qt_try_except()
    def add_object(
        self,
        obj: TypeObj,
        group_id: str | None = None,
        set_current: bool = True,
    ) -> None:
        """Add object

        Args:
            obj: SignalObj or ImageObj object
            group_id: group id to which the object belongs. If None or empty string,
             the object is added to the current group.
            set_current: if True, set the added object as current
        """
        if obj in self.objmodel:
            # Prevent adding the same object twice
            raise ValueError(
                f"Object {hex(id(obj))} already in panel. "
                f"The same object cannot be added twice: "
                f"please use a copy of the object."
            )
        if group_id is None or group_id == "":
            group_id = self.objview.get_current_group_id()
            if group_id is None:
                groups = self.objmodel.get_groups()
                if groups:
                    group_id = get_uuid(groups[0])
                else:
                    group_id = get_uuid(self.add_group(""))
        obj.check_data()
        self.objmodel.add_object(obj, group_id)

        # Block signals to avoid updating the plot (unnecessary refresh)
        self.objview.blockSignals(True)
        self.objview.add_object_item(obj, group_id, set_current=set_current)
        self.objview.blockSignals(False)

        # Emit signal to ensure that the data panel is shown in the main window and
        # that the plot is updated (trigger a refresh of the plot)
        self.SIG_OBJECT_ADDED.emit()

        self.objview.update_tree()

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        # iterate over a copy of self.__separate_views dict keys to avoid RuntimeError:
        # dictionary changed size during iteration
        for dlg in list(self.__separate_views):
            dlg.done(QW.QDialog.DialogCode.Rejected)
        self.objmodel.clear()
        self.plothandler.clear()
        self.objview.populate_tree()
        self.refresh_plot("selected", True, False)
        super().remove_all_objects()

    # ---- Signal/Image Panel API ------------------------------------------------------
    def setup_panel(self) -> None:
        """Setup panel"""
        self.acthandler.create_all_actions()
        self.processor.SIG_ADD_SHAPE.connect(self.plothandler.add_shapes)
        self.SIG_REFRESH_PLOT.connect(self.plothandler.refresh_plot)
        self.objview.SIG_SELECTION_CHANGED.connect(self.selection_changed)
        self.objview.SIG_ITEM_DOUBLECLICKED.connect(
            lambda oid: self.open_separate_view([oid])
        )
        self.objview.SIG_CONTEXT_MENU.connect(self.__popup_contextmenu)
        self.objprop.properties.SIG_APPLY_BUTTON_CLICKED.connect(
            self.properties_changed
        )
        self.addWidget(self.objview)
        self.addWidget(self.objprop)
        self.add_objprop_buttons()

    def refresh_plot(
        self,
        what: str,
        update_items: bool = True,
        force: bool = False,
        only_visible: bool = True,
        only_existing: bool = False,
    ) -> None:
        """Refresh plot.

        Args:
            what: string describing the objects to refresh.
             Valid values are "selected" (refresh the selected objects),
             "all" (refresh all objects), "existing" (refresh existing plot items),
             or an object uuid.
            update_items: if True, update the items.
             If False, only show the items (do not update them, except if the
             option "Use reference item LUT range" is enabled and more than one
             item is selected). Defaults to True.
            force: if True, force refresh even if auto refresh is disabled.
             Defaults to False.
            only_visible: if True, only refresh visible items. Defaults to True.
             Visible items are the ones that are not hidden by other items or the items
             except the first one if the option "Show first only" is enabled.
             This is useful for images, where the last image is the one that is shown.
             If False, all items are refreshed.
            only_existing: if True, only refresh existing items. Defaults to False.
             Existing items are the ones that have already been created and are
             associated to the object uuid. If False, create new items for the
             objects that do not have an item yet.

        Raises:
            ValueError: if `what` is not a valid value
        """
        if what not in ("selected", "all", "existing") and not isinstance(what, str):
            raise ValueError(f"Invalid value for 'what': {what}")
        self.SIG_REFRESH_PLOT.emit(
            what, update_items, force, only_visible, only_existing
        )

    def manual_refresh(self) -> None:
        """Manual refresh"""
        self.refresh_plot("selected", True, True)

    def get_category_actions(
        self, category: actionhandler.ActionCategory
    ) -> list[QW.QAction]:  # pragma: no cover
        """Return actions for category"""
        return self.acthandler.feature_actions.get(category, [])

    def get_context_menu(self) -> QW.QMenu:
        """Update and return context menu"""
        # Note: For now, this is completely unnecessary to clear context menu everytime,
        # but implementing it this way could be useful in the future in menu contents
        # should take into account current object selection
        self.context_menu.clear()
        actions = self.get_category_actions(actionhandler.ActionCategory.CONTEXT_MENU)
        add_actions(self.context_menu, actions)
        return self.context_menu

    def __popup_contextmenu(self, position: QC.QPoint) -> None:  # pragma: no cover
        """Popup context menu at position"""
        menu = self.get_context_menu()
        menu.popup(position)

    # ------Creating, adding, removing objects------------------------------------------
    def add_group(self, title: str, select: bool = False) -> objectmodel.ObjectGroup:
        """Add group

        Args:
            title: group title
            select: if True, select the group in the tree view. Defaults to False.

        Returns:
            Created group object
        """
        group = self.objmodel.add_group(title)
        self.objview.add_group_item(group)
        if select:
            self.objview.select_groups([group])
        return group

    def __duplicate_individual_obj(
        self, oid: str, new_group_id: str | None = None, set_current: bool = True
    ) -> None:
        """Duplicate individual object"""
        obj = self.objmodel[oid]
        if new_group_id is None:
            new_group_id = self.objmodel.get_object_group_id(obj)
        self.add_object(obj.copy(), group_id=new_group_id, set_current=set_current)

    def duplicate_object(self) -> None:
        """Duplication signal/image object"""
        if not self.mainwindow.confirm_memory_state():
            return
        # Duplicate individual objects (exclusive with respect to groups)
        for oid in self.objview.get_sel_object_uuids():
            self.__duplicate_individual_obj(oid, set_current=False)
        # Duplicate groups (exclusive with respect to individual objects)
        for group in self.objview.get_sel_groups():
            new_group = self.add_group(group.title)
            for oid in self.objmodel.get_group_object_ids(get_uuid(group)):
                self.__duplicate_individual_obj(
                    oid, get_uuid(new_group), set_current=False
                )
        self.selection_changed(update_items=True)

    def copy_metadata(self) -> None:
        """Copy object metadata"""
        obj = self.objview.get_sel_objects()[0]
        self.__metadata_clipboard = obj.metadata.copy()
        new_pref = short_id(obj) + "_"
        for key, value in obj.metadata.items():
            if ResultShape.match(key, value):
                mshape = ResultShape.from_metadata_entry(key, value)
                if not re.match(obj.PREFIX + r"[0-9]{3}[\s]*", mshape.title):
                    # Handling additional result (e.g. diameter)
                    for a_key, a_value in obj.metadata.items():
                        if isinstance(a_key, str) and a_key.startswith(mshape.title):
                            self.__metadata_clipboard.pop(a_key)
                            self.__metadata_clipboard[new_pref + a_key] = a_value
                    mshape.title = new_pref + mshape.title
                    # Handling result shape
                    self.__metadata_clipboard.pop(key)
                    self.__metadata_clipboard[mshape.key] = value

    def paste_metadata(self, param: PasteMetadataParam | None = None) -> None:
        """Paste metadata to selected object(s)"""
        if param is None:
            param = PasteMetadataParam(
                _("Paste metadata"),
                comment=_(
                    "Select what to keep from the clipboard.<br><br>"
                    "Result shapes and annotations, if kept, will be merged with "
                    "existing ones. <u>All other metadata will be replaced</u>."
                ),
            )
            if not param.edit(parent=self.parent()):
                return
        metadata = {}
        if param.keep_roi and ROI_KEY in self.__metadata_clipboard:
            metadata[ROI_KEY] = self.__metadata_clipboard[ROI_KEY]
        if param.keep_resultshapes:
            for key, value in self.__metadata_clipboard.items():
                if ResultShape.match(key, value):
                    metadata[key] = value
        if param.keep_resultproperties:
            for key, value in self.__metadata_clipboard.items():
                if ResultProperties.match(key, value):
                    metadata[key] = value
        if param.keep_other:
            for key, value in self.__metadata_clipboard.items():
                if (
                    not ResultShape.match(key, value)
                    and not ResultProperties.match(key, value)
                    and key not in (ROI_KEY,)
                ):
                    metadata[key] = value
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sorted(sel_objects, key=lambda obj: short_id(obj), reverse=True):
            obj.update_metadata_from(metadata)
        # We have to do a special refresh in order to force the plot handler to update
        # all plot items, even the ones that are not visible (otherwise, image masks
        # would not be updated after pasting the metadata: see issue #123)
        self.refresh_plot(
            "selected", update_items=True, only_visible=False, only_existing=True
        )

    def remove_object(self, force: bool = False) -> None:
        """Remove signal/image object

        Args:
            force: if True, remove object without confirmation. Defaults to False.
        """
        sel_groups = self.objview.get_sel_groups()
        if sel_groups and not force and not execenv.unattended:
            answer = QW.QMessageBox.warning(
                self,
                _("Delete group(s)"),
                _("Are you sure you want to delete the selected group(s)?"),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
            )
            if answer == QW.QMessageBox.No:
                return
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sorted(sel_objects, key=lambda obj: short_id(obj), reverse=True):
            dlg_list: list[QW.QDialog] = []
            for dlg, obj_i in self.__separate_views.items():
                if obj_i is obj:
                    dlg_list.append(dlg)
            for dlg in dlg_list:
                dlg.done(QW.QDialog.DialogCode.Rejected)
            self.plothandler.remove_item(get_uuid(obj))
            self.objview.remove_item(get_uuid(obj), refresh=False)
            self.objmodel.remove_object(obj)
        for group in sel_groups:
            self.objview.remove_item(get_uuid(group), refresh=False)
            self.objmodel.remove_group(group)
        self.objview.update_tree()
        self.selection_changed(update_items=True)
        self.SIG_OBJECT_REMOVED.emit()

    def delete_all_objects(self) -> None:  # pragma: no cover
        """Confirm before removing all objects"""
        if len(self) == 0:
            return
        answer = QW.QMessageBox.warning(
            self,
            _("Delete all"),
            _("Do you want to delete all objects (%s)?") % self.PANEL_STR,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
        )
        if answer == QW.QMessageBox.Yes:
            self.remove_all_objects()

    def delete_metadata(
        self, refresh_plot: bool = True, keep_roi: bool | None = None
    ) -> None:
        """Delete metadata of selected objects

        Args:
            refresh_plot: Refresh plot. Defaults to True.
            keep_roi: Keep regions of interest, if any. Defaults to None (ask user).
        """
        sel_objs = self.objview.get_sel_objects(include_groups=True)
        # Check if there are regions of interest first:
        roi_backup: dict[TypeObj, np.ndarray] = {}
        if any(obj.roi is not None for obj in sel_objs):
            if execenv.unattended and keep_roi is None:
                keep_roi = False
            elif keep_roi is None:
                answer = QW.QMessageBox.warning(
                    self,
                    _("Delete metadata"),
                    _(
                        "Some selected objects have regions of interest.<br>"
                        "Do you want to delete them as well?"
                    ),
                    QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Cancel,
                )
                if answer == QW.QMessageBox.Cancel:
                    return
                keep_roi = answer == QW.QMessageBox.No
            if keep_roi:
                for obj in sel_objs:
                    if obj.roi is not None:
                        roi_backup[obj] = obj.roi

        # Delete metadata:
        for index, obj in enumerate(sel_objs):
            obj.reset_metadata_to_defaults()
            if not keep_roi:
                obj.invalidate_maskdata_cache()
            if obj in roi_backup:
                obj.roi = roi_backup[obj]
            if index == 0:
                self.selection_changed()

        # When calling object `reset_metadata_to_defaults` method, we removed all
        # metadata application options, among them the object number which is used
        # to compute the short ID of the object.
        # So we have to reset the short IDs of all objects in the model to recalculate
        # the object numbers:
        self.objmodel.reset_short_ids()

        if refresh_plot:
            # We have to do a special refresh in order to force the plot handler to
            # update all plot items, even the ones that are not visible (otherwise,
            # image masks would remained visible after deleting the ROI for example:
            # see issue #122)
            self.refresh_plot(
                "selected", update_items=True, only_visible=False, only_existing=True
            )

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
            refresh_plot: refresh plot. Defaults to True.
        """
        for obj in self.objview.get_sel_objects(include_groups=True):
            create_adapter_from_object(obj).add_annotations_from_items(items)
        if refresh_plot:
            self.refresh_plot("selected", True, False)

    def update_metadata_view_settings(self) -> None:
        """Update metadata view settings"""
        def_dict = Conf.view.get_def_dict(self.__class__.__name__[:3].lower())
        for obj in self.objmodel:
            obj.set_metadata_options_defaults(def_dict, overwrite=True)
        self.refresh_plot("all", True, False)

    def copy_titles_to_clipboard(self) -> None:
        """Copy object titles to clipboard (for reproducibility)"""
        QW.QApplication.clipboard().setText(str(self.objview))

    def new_group(self) -> None:
        """Create a new group"""
        # Open a message box to enter the group name
        group_name, ok = QW.QInputDialog.getText(self, _("New group"), _("Group name:"))
        if ok:
            self.add_group(group_name)

    def rename_selected_object_or_group(self, new_name: str | None = None) -> None:
        """Rename selected object or group

        Args:
            new_name: new name (default: None, i.e. ask user)
        """
        sel_objects = self.objview.get_sel_objects(include_groups=False)
        sel_groups = self.objview.get_sel_groups()
        if (not sel_objects and not sel_groups) or len(sel_objects) + len(
            sel_groups
        ) > 1:
            # Won't happen in the application, but could happen in tests or using the
            # API directly
            raise ValueError("Select one object or group to rename")
        if sel_objects:
            obj = sel_objects[0]
            if new_name is None:
                new_name, ok = QW.QInputDialog.getText(
                    self,
                    _("Rename object"),
                    _("Object name:"),
                    QW.QLineEdit.Normal,
                    obj.title,
                )
                if not ok:
                    return
            obj.title = new_name
            self.objview.update_item(get_uuid(obj))
            self.objprop.update_properties_from(obj)
        elif sel_groups:
            group = sel_groups[0]
            if new_name is None:
                new_name, ok = QW.QInputDialog.getText(
                    self,
                    _("Rename group"),
                    _("Group name:"),
                    QW.QLineEdit.Normal,
                    group.title,
                )
                if not ok:
                    return
            group.title = new_name
            self.objview.update_item(get_uuid(group))

    @abc.abstractmethod
    def get_newparam_from_current(
        self, newparam: NewSignalParam | NewImageParam | None = None
    ) -> NewSignalParam | NewImageParam | None:
        """Get new object parameters from the current object.

        Args:
            newparam: new object parameters. If None, create a new one.

        Returns:
            New object parameters
        """

    @abc.abstractmethod
    def new_object(
        self,
        newparam: NewSignalParam | NewImageParam | None = None,
        addparam: gds.DataSet | None = None,
        edit: bool = True,
        add_to_panel: bool = True,
    ) -> TypeObj | None:
        """Create a new object (signal/image).

        Args:
            newparam: new object parameters
            addparam: additional parameters
            edit: Open a dialog box to edit parameters (default: True)
            add_to_panel: Add object to panel (default: True)

        Returns:
            New object
        """

    def set_current_object_title(self, title: str) -> None:
        """Set current object title"""
        obj = self.objview.get_current_object()
        obj.title = title
        self.objview.update_item(get_uuid(obj))

    def __load_from_file(
        self, filename: str, create_group: bool = True, add_objects: bool = True
    ) -> list[SignalObj] | list[ImageObj]:
        """Open objects from file (signal/image), add them to DataLab and return them.

        Args:
            filename: file name
            create_group: if True, create a new group if more than one object is loaded.
             Defaults to True. If False, all objects are added to the current group.
            add_objects: if True, add objects to the panel. Defaults to True.

        Returns:
            New object or list of new objects
        """
        worker = CallbackWorker(lambda worker: self.IO_REGISTRY.read(filename, worker))
        objs = qt_long_callback(self, _("Adding objects to workspace"), worker, True)
        group_id = None
        if len(objs) > 1 and create_group:
            # Create a new group if more than one object is loaded
            group_id = get_uuid(self.add_group(osp.basename(filename)))
        for obj in objs:
            if add_objects:
                self.add_object(obj, group_id=group_id, set_current=obj is objs[-1])
        self.selection_changed()
        return objs

    def __save_to_file(self, obj: TypeObj, filename: str) -> None:
        """Save object to file (signal/image).

        Args:
            obj: object
            filename: file name
        """
        self.IO_REGISTRY.write(filename, obj)

    def load_from_directory(self, directory: str | None = None) -> list[TypeObj]:
        """Open objects from directory (signals or images, depending on the panel),
        add them to DataLab and return them.
        If the directory is not specified, ask the user to select a directory.

        Args:
            directory: directory name

        Returns:
            list of new objects
        """
        if not self.mainwindow.confirm_memory_state():
            return []
        if directory is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                directory = getexistingdirectory(self, _("Open"), basedir)
        if not directory:
            return []
        folders = [
            path
            for path in glob.glob(osp.join(directory, "**"), recursive=True)
            if osp.isdir(path) and len(os.listdir(path)) > 0
        ]
        objs = []
        with create_progress_bar(
            self, _("Scanning directory"), max_=len(folders) - 1
        ) as progress:
            # Iterate over all subfolders in the directory:
            for i_path, path in enumerate(folders):
                progress.setValue(i_path + 1)
                if progress.wasCanceled():
                    break
                path = osp.normpath(path)
                fnames = [
                    osp.join(path, fname)
                    for fname in os.listdir(path)
                    if osp.isfile(osp.join(path, fname))
                ]
                new_objs = self.load_from_files(
                    fnames,
                    create_group=False,
                    add_objects=False,
                    ignore_errors=True,
                )
                if new_objs:
                    objs += new_objs
                    grp_name = osp.relpath(path, directory)
                    if grp_name == ".":
                        grp_name = osp.basename(path)
                    grp = self.add_group(grp_name)
                    for obj in new_objs:
                        self.add_object(obj, group_id=get_uuid(grp), set_current=False)
        return objs

    def load_from_files(
        self,
        filenames: list[str] | None = None,
        create_group: bool = False,
        add_objects: bool = True,
        ignore_errors: bool = False,
    ) -> list[TypeObj]:
        """Open objects from file (signals/images), add them to DataLab and return them.

        Args:
            filenames: File names
            create_group: if True, create a new group if more than one object is loaded
             for a single file. Defaults to False: all objects are added to the current
             group.
            add_objects: if True, add objects to the panel. Defaults to True.
            ignore_errors: if True, ignore errors when loading files. Defaults to False.

        Returns:
            list of new objects
        """
        if not self.mainwindow.confirm_memory_state():
            return []
        if filenames is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            filters = self.IO_REGISTRY.get_read_filters()
            with save_restore_stds():
                filenames, _filt = getopenfilenames(self, _("Open"), basedir, filters)
        objs = []
        for filename in filenames:
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                Conf.main.base_dir.set(filename)
                try:
                    objs += self.__load_from_file(
                        filename, create_group=create_group, add_objects=add_objects
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    if ignore_errors:
                        # Ignore unknown file types
                        pass
                    else:
                        raise exc
        return objs

    def save_to_files(self, filenames: list[str] | str | None = None) -> None:
        """Save selected objects to files (signal/image).

        Args:
            filenames: File names
        """
        objs = self.objview.get_sel_objects(include_groups=True)
        if filenames is None:  # pragma: no cover
            filenames = [None] * len(objs)
        assert len(filenames) == len(objs), (
            "Number of filenames must match number of objects"
        )
        for index, obj in enumerate(objs):
            filename = filenames[index]
            if filename is None:
                basedir = Conf.main.base_dir.get()
                filters = self.IO_REGISTRY.get_write_filters()
                with save_restore_stds():
                    filename, _filt = getsavefilename(
                        self, _("Save as"), basedir, filters
                    )
            if filename:
                with qt_try_loadsave_file(self.parent(), filename, "save"):
                    Conf.main.base_dir.set(filename)
                    self.__save_to_file(obj, filename)

    def handle_dropped_files(self, filenames: list[str] | None = None) -> None:
        """Handle dropped files

        Args:
            filenames: File names

        Returns:
            None
        """
        dirnames = [fname for fname in filenames if osp.isdir(fname)]
        h5_fnames = [fname for fname in filenames if fname.endswith(".h5")]
        other_fnames = sorted(list(set(filenames) - set(h5_fnames) - set(dirnames)))
        if dirnames:
            for dirname in dirnames:
                self.load_from_directory(dirname)
        if h5_fnames:
            self.mainwindow.open_h5_files(h5_fnames, import_all=True)
        if other_fnames:
            self.load_from_files(other_fnames)

    def exec_import_wizard(self) -> None:
        """Execute import wizard"""
        wizard = TextImportWizard(self.PANEL_STR_ID, parent=self.parent())
        if exec_dialog(wizard):
            objs = wizard.get_objs()
            if objs:
                with create_progress_bar(
                    self, _("Adding objects to workspace"), max_=len(objs) - 1
                ) as progress:
                    for idx, obj in enumerate(objs):
                        progress.setValue(idx)
                        if progress.wasCanceled():
                            break
                        self.add_object(obj)

    def import_metadata_from_file(self, filename: str | None = None) -> None:
        """Import metadata from file (JSON).

        Args:
            filename: File name
        """
        if filename is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getopenfilename(
                    self, _("Import metadata"), basedir, "*.json"
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                Conf.main.base_dir.set(filename)
                obj = self.objview.get_sel_objects(include_groups=True)[0]
                # Import object's metadata from file as JSON:
                handler = JSONHandler(filename)
                handler.load()
                obj.metadata = handler.get_json_dict()
            self.refresh_plot("selected", True, False)

    def export_metadata_from_file(self, filename: str | None = None) -> None:
        """Export metadata to file (JSON).

        Args:
            filename: File name
        """
        obj = self.objview.get_sel_objects(include_groups=True)[0]
        if filename is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Export metadata"), basedir, "*.json"
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                # Export object's metadata to file as JSON:
                handler = JSONHandler(filename)
                handler.set_json_dict(obj.metadata)
                handler.save()

    # ------Refreshing GUI--------------------------------------------------------------
    def selection_changed(self, update_items: bool = False) -> None:
        """Object selection changed: update object properties, refresh plot and update
        object view.

        Args:
            update_items: Update plot items (default: False)
        """
        selected_objects = self.objview.get_sel_objects(include_groups=True)
        selected_groups = self.objview.get_sel_groups()
        self.objprop.update_properties_from(self.objview.get_current_object())
        self.acthandler.selected_objects_changed(selected_groups, selected_objects)
        self.refresh_plot("selected", update_items, False)

    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: update object properties,
        refresh plot and update object view."""
        obj = self.objview.get_current_object()
        # if obj is not None:  # XXX: Is it necessary?
        obj.invalidate_maskdata_cache()
        update_dataset(obj, self.objprop.properties.dataset)
        self.objview.update_item(get_uuid(obj))
        self.refresh_plot("selected", True, False)

    # ------Plotting data in modal dialogs----------------------------------------------
    def add_plot_items_to_dialog(self, dlg: PlotDialog, oids: list[str]) -> None:
        """Add plot items to dialog

        Args:
            dlg: Dialog
            oids: Object IDs
        """
        objs = self.objmodel.get_objects(oids)
        plot = dlg.get_plot()
        with create_progress_bar(
            self, _("Creating plot items"), max_=len(objs)
        ) as progress:
            for index, obj in enumerate(objs):
                progress.setValue(index + 1)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    return None
                item = create_adapter_from_object(obj).make_item(
                    update_from=self.plothandler[get_uuid(obj)]
                )
                item.set_readonly(True)
                plot.add_item(item, z=0)
        plot.set_active_item(item)
        item.unselect()
        plot.replot()
        return dlg

    def open_separate_view(
        self, oids: list[str] | None = None, edit_annotations: bool = False
    ) -> PlotDialog | None:
        """
        Open separate view for visualizing selected objects

        Args:
            oids: Object IDs (default: None)
            edit_annotations: Edit annotations (default: False)

        Returns:
            Instance of PlotDialog
        """
        if oids is None:
            oids = self.objview.get_sel_object_uuids(include_groups=True)
        obj = self.objmodel[oids[-1]]  # last selected object

        if not all(oid in self.plothandler for oid in oids):
            # This happens for example when opening an already saved workspace with
            # multiple images, and if the user tries to view in a new window a group of
            # images without having selected any object yet. In this case, only the
            # last image is actually plotted (because if the other have the same size
            # and position, they are hidden), and the plot item of every other image is
            # not created yet. So we need to refresh the plot to create the plot item of
            # those images.
            self.plothandler.refresh_plot(
                "selected", update_items=True, force=True, only_visible=False
            )

        # Create a new dialog and add plot items to it
        dlg = self.create_new_dialog(
            title=obj.title if len(oids) == 1 else None,
            edit=True,
            name=f"{obj.PREFIX}_new_window",
        )
        if dlg is None:
            return None
        self.add_plot_items_to_dialog(dlg, oids)

        mgr = dlg.get_manager()
        toolbar = QW.QToolBar(_("Annotations"), self)
        dlg.button_layout.insertWidget(0, toolbar)
        mgr.add_toolbar(toolbar, id(toolbar))
        toolbar.setToolButtonStyle(QC.Qt.ToolButtonTextUnderIcon)
        for tool in self.ANNOTATION_TOOLS:
            mgr.add_tool(tool, toolbar_id=id(toolbar))

        def toggle_annotations(enabled: bool):
            """Toggle annotation tools / edit buttons visibility"""
            for widget in (dlg.button_box, toolbar, mgr.get_itemlist_panel()):
                if enabled:
                    widget.show()
                else:
                    widget.hide()

        edit_ann_action = create_action(
            dlg,
            _("Annotations"),
            toggled=toggle_annotations,
            icon=get_icon("annotations.svg"),
        )
        mgr.add_tool(ActionTool, edit_ann_action)
        default_toolbar = mgr.get_default_toolbar()
        action_btn = default_toolbar.widgetForAction(edit_ann_action)
        action_btn.setToolButtonStyle(QC.Qt.ToolButtonTextBesideIcon)
        plot = dlg.get_plot()
        for item in plot.items:
            item.set_selectable(False)
        for item in create_adapter_from_object(obj).iterate_shape_items(editable=True):
            plot.add_item(item)
        self.__separate_views[dlg] = obj
        toggle_annotations(edit_annotations)
        if len(oids) > 1:
            # If multiple objects are displayed, show the item list panel
            # (otherwise, it is hidden by default to lighten the dialog, except
            # if `edit_annotations` is True):
            plot.manager.get_itemlist_panel().show()
        if edit_annotations:
            edit_ann_action.setChecked(True)
        dlg.show()
        dlg.finished.connect(self.__separate_view_finished)
        return dlg

    def __separate_view_finished(self, result: int) -> None:
        """Separate view was closed

        Args:
            result: result
        """
        dlg: PlotDialog = self.sender()
        if result == QW.QDialog.DialogCode.Accepted:
            rw_items = []
            for item in dlg.get_plot().get_items():
                if not item.is_readonly() and is_plot_item_serializable(item):
                    rw_items.append(item)
            obj = self.__separate_views[dlg]
            obj.annotations = items_to_json(rw_items)
            self.selection_changed(update_items=True)
        self.__separate_views.pop(dlg)
        dlg.deleteLater()

    def create_new_dialog(
        self,
        edit: bool = False,
        toolbar: bool = True,
        title: str | None = None,
        name: str | None = None,
        options: dict | None = None,
    ) -> PlotDialog | None:
        """Create new pop-up signal/image plot dialog.

        Args:
            edit: Edit mode
            toolbar: Show toolbar
            title: Dialog title
            name: Dialog object name
            options: Plot options

        Returns:
            Plot dialog instance
        """
        plot_options = self.plothandler.get_current_plot_options()
        if options is not None:
            plot_options = plot_options.copy(options)

        # Resize the dialog so that it's at least MINDIALOGSIZE (absolute values),
        # and at most MAXDIALOGSIZE (% of the main window size):
        minwidth, minheight = self.MINDIALOGSIZE
        maxwidth = int(self.mainwindow.width() * self.MAXDIALOGSIZE)
        maxheight = int(self.mainwindow.height() * self.MAXDIALOGSIZE)
        size = min(minwidth, maxwidth), min(minheight, maxheight)

        # pylint: disable=not-callable
        dlg = PlotDialog(
            parent=self.parent(),
            title=APP_NAME if title is None else f"{title} - {APP_NAME}",
            edit=edit,
            options=plot_options,
            toolbar=toolbar,
            size=size,
        )
        dlg.setWindowIcon(get_icon("DataLab.svg"))
        dlg.setObjectName(name)
        return dlg

    def get_roi_editor_output(self, extract: bool) -> tuple[TypeROI, bool] | None:
        """Get ROI data (array) from specific dialog box.

        Args:
            extract: Extract ROI from data

        Returns:
            A tuple containing the ROI object and a boolean indicating whether the
            dialog was accepted or not.
        """
        roi_s = _("Regions of interest")
        options = self.ROIDIALOGOPTIONS
        obj = self.objview.get_sel_objects(include_groups=True)[-1]

        # Create a new dialog

        dlg = self.create_new_dialog(
            edit=True,
            toolbar=True,
            title=f"{roi_s} - {obj.title}",
            name=f"{obj.PREFIX}_roi_dialog",
            options=options,
        )
        if dlg is None:
            return None

        # Create ROI editor (and add it to the dialog)

        item = create_adapter_from_object(obj).make_item(
            update_from=self.plothandler[get_uuid(obj)]
        )

        # pylint: disable=not-callable
        roi_editor = self.get_roieditor_class()(dlg, obj, extract, item=item)

        dlg.button_layout.insertWidget(0, roi_editor)

        if exec_dialog(dlg):
            return roi_editor.get_roieditor_results()
        return None

    def get_objects_with_dialog(
        self,
        title: str,
        comment: str = "",
        nb_objects: int = 1,
        parent: QW.QWidget | None = None,
    ) -> TypeObj | None:
        """Get object with dialog box.

        Args:
            title: Dialog title
            comment: Optional dialog comment
            nb_objects: Number of objects to select
            parent: Parent widget
        Returns:
            Object(s) (signal(s) or image(s), or None if dialog was canceled)
        """
        parent = self if parent is None else parent
        dlg = objectview.GetObjectsDialog(parent, self, title, comment, nb_objects)
        if exec_dialog(dlg):
            return dlg.get_selected_objects()
        return None

    def __new_objprop_button(
        self, title: str, icon: str, tooltip: str, callback: Callable
    ) -> QW.QPushButton:
        """Create new object property button"""
        btn = QW.QPushButton(get_icon(icon), title, self)
        btn.setToolTip(tooltip)
        self.objprop.add_button(btn)
        btn.clicked.connect(callback)
        self.acthandler.add_action(
            btn,
            select_condition=actionhandler.SelectCond.at_least_one,
        )
        return btn

    def add_objprop_buttons(self) -> None:
        """Insert additional buttons in object properties panel"""
        self.__new_objprop_button(
            _("Results"),
            "show_results.svg",
            _("Show results obtained from previous analysis"),
            self.show_results,
        )
        self.__new_objprop_button(
            _("Annotations"),
            "annotations.svg",
            _("Open a dialog to edit annotations"),
            lambda: self.open_separate_view(edit_annotations=True),
        )

    def __show_no_result_warning(self):
        """Show no result warning"""
        msg = "<br>".join(
            [
                _("No result currently available for this object."),
                "",
                _(
                    "This feature leverages the results of previous analysis "
                    "performed on the selected object(s).<br><br>"
                    "To compute results, select one or more objects and choose "
                    "a feature in the <u>Analysis</u> menu."
                ),
            ]
        )
        QW.QMessageBox.information(self, APP_NAME, msg)

    def show_results(self) -> None:
        """Show results"""
        objs = self.objview.get_sel_objects(include_groups=True)
        rdatadict = create_resultdata_dict(objs)
        if rdatadict:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for category, rdata in rdatadict.items():
                    dlg = ArrayEditor(self.parent())
                    dlg.setup_and_check(
                        np.vstack([result.shown_array for result in rdata.results]),
                        _("Results") + f" ({category})",
                        readonly=True,
                        xlabels=rdata.xlabels,
                        ylabels=rdata.ylabels,
                    )
                    dlg.setObjectName(f"{objs[0].PREFIX}_results")
                    dlg.resize(750, 300)
                    exec_dialog(dlg)
        else:
            self.__show_no_result_warning()

    def __add_result_signal(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        title: str,
        xaxis: str,
        yaxis: str,
    ) -> None:
        """Add result signal"""
        xdata = np.array(x, dtype=float)
        ydata = np.array(y, dtype=float)

        obj = create_signal(
            title=f"{title}: {yaxis} = f({xaxis})",
            x=xdata,
            y=ydata,
            labels=[xaxis, yaxis],
        )
        self.mainwindow.signalpanel.add_object(obj)

    def plot_results(self) -> None:
        """Plot results"""
        objs = self.objview.get_sel_objects(include_groups=True)
        rdatadict = create_resultdata_dict(objs)
        if rdatadict:
            for category, rdata in rdatadict.items():
                xchoices = (("indices", _("Indices")),)
                for xlabel in rdata.xlabels:
                    xchoices += ((xlabel, xlabel),)
                ychoices = xchoices[1:]

                # Regrouping ResultShape results by their `title` attribute:
                grouped_results: dict[str, list[ResultShape]] = {}
                for result in rdata.results:
                    grouped_results.setdefault(result.title, []).append(result)

                # From here, results are already grouped by their `category` attribute,
                # and then by their `title` attribute. We can now plot them.
                #
                # Now, we have two common use cases:
                # 1. Plotting one curve per object (signal/image) and per `title`
                #    attribute, if each selected object contains a result object
                #    with multiple values (e.g. from a blob detection feature).
                # 2. Plotting one curve per `title` attribute, if each selected object
                #    contains a result object with a single value (e.g. from a FHWM
                #    feature) - in that case, we select only the first value of each
                #    result object.

                # The default kind of plot depends on the number of values in each
                # result and the number of selected objects:
                default_kind = (
                    "one_curve_per_object"
                    if any(result.array.shape[0] > 1 for result in rdata.results)
                    else "one_curve_per_title"
                )

                class PlotResultParam(gds.DataSet):
                    """Plot results parameters"""

                    kind = gds.ChoiceItem(
                        _("Plot kind"),
                        (
                            (
                                "one_curve_per_object",
                                _("One curve per object (or ROI) and per result title"),
                            ),
                            ("one_curve_per_title", _("One curve per result title")),
                        ),
                        default=default_kind,
                    )
                    xaxis = gds.ChoiceItem(_("X axis"), xchoices, default="indices")
                    yaxis = gds.ChoiceItem(
                        _("Y axis"), ychoices, default=ychoices[0][0]
                    )

                comment = (
                    _(
                        "Plot results obtained from previous analyses.<br><br>"
                        "This plot is based on results associated with '%s' prefix."
                    )
                    % category
                )
                param = PlotResultParam(_("Plot results"), comment=comment)
                if not param.edit(parent=self.parent()):
                    return

                i_yaxis = rdata.xlabels.index(param.yaxis)
                if param.kind == "one_curve_per_title":
                    # One curve per ROI (if any) and per result title
                    # ------------------------------------------------------------------
                    # Begin by checking if all results have the same number of ROIs:
                    # for simplicity, let's check the number of unique ROI indices.
                    all_roi_indexes = [
                        np.unique(result.array[:, 0]) for result in rdata.results
                    ]
                    # Check if all roi_indexes are the same:
                    if len(set(map(tuple, all_roi_indexes))) > 1:
                        QW.QMessageBox.warning(
                            self,
                            _("Plot results"),
                            _(
                                "All objects associated with results must have the "
                                "same regions of interest (ROIs) to plot results "
                                "together."
                            ),
                        )
                        return
                    for i_roi in all_roi_indexes[0]:
                        roi_suffix = f"|ROI{int(i_roi + 1)}" if i_roi >= 0 else ""
                        for title, results in grouped_results.items():  # title
                            x, y = [], []
                            for index, result in enumerate(results):
                                mask = result.array[:, 0] == i_roi
                                if param.xaxis == "indices":
                                    x.append(index)
                                else:
                                    i_xaxis = rdata.xlabels.index(param.xaxis)
                                    x.append(result.shown_array[mask, i_xaxis][0])
                                y.append(result.shown_array[mask, i_yaxis][0])
                            self.__add_result_signal(
                                x, y, f"{title}{roi_suffix}", param.xaxis, param.yaxis
                            )
                else:
                    # One curve per result title, per object and per ROI
                    # ------------------------------------------------------------------
                    for title, results in grouped_results.items():  # title
                        for index, result in enumerate(results):  # object
                            roi_idx = np.array(np.unique(result.array[:, 0]), dtype=int)
                            for i_roi in roi_idx:  # ROI
                                roi_suffix = (
                                    f"|ROI{int(i_roi + 1)}" if i_roi >= 0 else ""
                                )
                                mask = result.array[:, 0] == i_roi
                                if param.xaxis == "indices":
                                    x = np.arange(result.array.shape[0])[mask]
                                else:
                                    i_xaxis = rdata.xlabels.index(param.xaxis)
                                    x = result.shown_array[mask, i_xaxis]
                                y = result.shown_array[mask, i_yaxis]
                                shid = short_id(objs[index])
                                stitle = f"{title} ({shid}){roi_suffix}"
                                self.__add_result_signal(
                                    x, y, stitle, param.xaxis, param.yaxis
                                )

        else:
            self.__show_no_result_warning()

    def delete_results(self) -> None:
        """Delete results"""
        objs = self.objview.get_sel_objects(include_groups=True)
        rdatadict = create_resultdata_dict(objs)
        if rdatadict:
            answer = QW.QMessageBox.warning(
                self,
                _("Delete results"),
                _(
                    "Are you sure you want to delete all results "
                    "of the selected object(s)?"
                ),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
            )
            if answer == QW.QMessageBox.Yes:
                objs = self.objview.get_sel_objects(include_groups=True)
                for obj in objs:
                    obj.delete_results()
                self.refresh_plot("selected", True, False)
        else:
            self.__show_no_result_warning()

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add a label with object title on the associated plot

        Args:
            title: Label title. Defaults to None.
             If None, the title is the object title.
        """
        objs = self.objview.get_sel_objects(include_groups=True)
        for obj in objs:
            create_adapter_from_object(obj).add_label_with_title(title=title)
        self.refresh_plot("selected", True, False)
