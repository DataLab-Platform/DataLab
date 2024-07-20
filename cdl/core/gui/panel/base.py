# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Base panel objects (see parent package :mod:`cdl.core.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import dataclasses
import os.path as osp
import re
import warnings
from typing import TYPE_CHECKING

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
from qtpy.compat import getopenfilename, getopenfilenames, getsavefilename

import cdl.computation.base
from cdl.config import APP_NAME, Conf, _
from cdl.core.gui import actionhandler, objectmodel, objectview, roieditor
from cdl.core.model.base import ResultProperties, ResultShape, items_to_json
from cdl.core.model.signal import create_signal
from cdl.env import execenv
from cdl.utils.qthelpers import (
    CallbackWorker,
    create_progress_bar,
    qt_long_callback,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)
from cdl.widgets.textimport import TextImportWizard

if TYPE_CHECKING:
    from typing import Callable

    from plotpy.items import CurveItem, LabelItem, MaskedImageItem
    from plotpy.tools.base import GuiTool

    from cdl.core.gui import ObjItf
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.gui.plothandler import ImagePlotHandler, SignalPlotHandler
    from cdl.core.gui.processor.image import ImageProcessor
    from cdl.core.gui.processor.signal import SignalProcessor
    from cdl.core.io.image import ImageIORegistry
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer
    from cdl.core.io.signal import SignalIORegistry
    from cdl.core.model.base import ShapeTypes
    from cdl.core.model.image import ImageObj, NewImageParam
    from cdl.core.model.signal import NewSignalParam, SignalObj


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
        child.addTab(param_scroll, _("Computing parameters"))

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

    A panel handle an object list (objects are signals, images, macros, ...).
    Each object must implement ``cdl.core.gui.ObjItf`` interface
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
        name = f"{obj.short_id}: {title}"
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
            obj.regenerate_uuid()
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


def create_resultdata_dict(objs: list[SignalObj | ImageObj]) -> dict[str, ResultData]:
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
                ylabel = f"{result.title}({obj.short_id})"
                i_roi = int(result.array[i_row_res, 0])
                if i_roi >= 0:
                    ylabel += f"|ROI{i_roi}"
                rdata.ylabels.append(ylabel)
    return rdatadict


class BaseDataPanel(AbstractPanel):
    """Object handling the item list, the selected item properties and plot"""

    PANEL_STR = ""  # e.g. "Signal Panel"
    PANEL_STR_ID = ""  # e.g. "signal"
    PARAMCLASS: SignalObj | ImageObj = None  # Replaced in child object
    ANNOTATION_TOOLS = ()
    DIALOGSIZE = (800, 600)
    # Replaced by the right class in child object:
    IO_REGISTRY: SignalIORegistry | ImageIORegistry | None = None
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_REFRESH_PLOT = QC.Signal(str, bool)  # Connected to PlotHandler.refresh_plot
    ROIDIALOGOPTIONS = {}
    # Replaced in child object:
    ROIDIALOGCLASS: roieditor.SignalROIEditor | roieditor.ImageROIEditor | None = None

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
        self.__separate_views: dict[QW.QDialog, SignalObj | ImageObj] = {}

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

    def serialize_object_to_hdf5(
        self, obj: SignalObj | ImageObj, writer: NativeH5Writer
    ) -> None:
        """Serialize object to HDF5 file"""
        # Before serializing, update metadata from plot item parameters, in order to
        # save the latest visualization settings:
        try:
            item = self.plothandler[obj.uuid]
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
                        self.add_object(obj, group.uuid, set_current=False)
                    self.selection_changed()

    def __len__(self) -> int:
        """Return number of objects"""
        return len(self.objmodel)

    def __getitem__(self, nb: int) -> SignalObj | ImageObj:
        """Return object from its number (1 to N)"""
        return self.objmodel.get_object_from_number(nb)

    def __iter__(self):
        """Iterate over objects"""
        return iter(self.objmodel)

    def create_object(self) -> SignalObj | ImageObj:
        """Create object (signal or image)

        Returns:
            SignalObj or ImageObj object
        """
        return self.PARAMCLASS()  # pylint: disable=not-callable

    @qt_try_except()
    def add_object(
        self,
        obj: SignalObj | ImageObj,
        group_id: str | None = None,
        set_current: bool = True,
    ) -> None:
        """Add object

        Args:
            obj: SignalObj or ImageObj object
            group_id: group id
            set_current: if True, set the added object as current
        """
        if obj in self.objmodel:
            # Prevent adding the same object twice
            raise ValueError(
                f"Object {hex(id(obj))} already in panel. "
                f"The same object cannot be added twice: "
                f"please use a copy of the object."
            )
        if group_id is None:
            group_id = self.objview.get_current_group_id()
            if group_id is None:
                groups = self.objmodel.get_groups()
                if groups:
                    group_id = groups[0].uuid
                else:
                    group_id = self.add_group("").uuid
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
        self.SIG_REFRESH_PLOT.emit("selected", True)
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
    def add_group(self, title: str) -> objectmodel.ObjectGroup:
        """Add group"""
        group = self.objmodel.add_group(title)
        self.objview.add_group_item(group)
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
            for oid in self.objmodel.get_group_object_ids(group.uuid):
                self.__duplicate_individual_obj(oid, new_group.uuid, set_current=False)
        self.selection_changed(update_items=True)

    def copy_metadata(self) -> None:
        """Copy object metadata"""
        obj = self.objview.get_sel_objects()[0]
        self.__metadata_clipboard = obj.metadata.copy()
        new_pref = obj.short_id + "_"
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

    def paste_metadata(self) -> None:
        """Paste metadata to selected object(s)"""
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sorted(sel_objects, key=lambda obj: obj.short_id, reverse=True):
            obj.metadata.update(self.__metadata_clipboard)
        self.SIG_REFRESH_PLOT.emit("selected", True)

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
        for obj in sorted(sel_objects, key=lambda obj: obj.short_id, reverse=True):
            dlg_list: list[QW.QDialog] = []
            for dlg, obj_i in self.__separate_views.items():
                if obj_i is obj:
                    dlg_list.append(dlg)
            for dlg in dlg_list:
                dlg.done(QW.QDialog.DialogCode.Rejected)
            self.plothandler.remove_item(obj.uuid)
            self.objview.remove_item(obj.uuid, refresh=False)
            self.objmodel.remove_object(obj)
        for group in sel_groups:
            self.objview.remove_item(group.uuid, refresh=False)
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
        if execenv.unattended:
            keep_roi = False
        sel_objs = self.objview.get_sel_objects(include_groups=True)
        # Check if there are regions of interest first:
        roi_backup: dict[SignalObj | ImageObj, np.ndarray] = {}
        if any(obj.roi is not None for obj in sel_objs):
            if keep_roi is None:
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
            if obj in roi_backup:
                obj.roi = roi_backup[obj]
            if index == 0:
                self.selection_changed()
        if refresh_plot:
            self.SIG_REFRESH_PLOT.emit("selected", True)

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items: annotation plot items
            refresh_plot: refresh plot. Defaults to True.
        """
        for obj in self.objview.get_sel_objects(include_groups=True):
            obj.add_annotations_from_items(items)
        if refresh_plot:
            self.SIG_REFRESH_PLOT.emit("selected", True)

    def update_metadata_view_settings(self) -> None:
        """Update metadata view settings"""
        for obj in self.objmodel:
            obj.update_metadata_view_settings()
        self.SIG_REFRESH_PLOT.emit("all", True)

    def copy_titles_to_clipboard(self) -> None:
        """Copy object titles to clipboard (for reproducibility)"""
        QW.QApplication.clipboard().setText(str(self.objview))

    def new_group(self) -> None:
        """Create a new group"""
        # Open a message box to enter the group name
        group_name, ok = QW.QInputDialog.getText(self, _("New group"), _("Group name:"))
        if ok:
            self.add_group(group_name)

    def rename_group(self) -> None:
        """Rename a group"""
        # Open a message box to enter the group name
        group = self.objview.get_sel_groups()[0]
        group_name, ok = QW.QInputDialog.getText(
            self, _("Rename group"), _("Group name:"), QW.QLineEdit.Normal, group.title
        )
        if ok:
            group.title = group_name
            self.objview.update_item(group.uuid)

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
    ) -> SignalObj | ImageObj | None:
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
        self.objview.update_item(obj.uuid)

    def __load_from_file(self, filename: str) -> list[SignalObj] | list[ImageObj]:
        """Open objects from file (signal/image), add them to DataLab and return them.

        Args:
            filename: file name

        Returns:
            New object or list of new objects
        """
        worker = CallbackWorker(lambda worker: self.IO_REGISTRY.read(filename, worker))
        objs = qt_long_callback(self, _("Adding objects to workspace"), worker, True)
        group_id = None
        if len(objs) > 1:
            group_id = self.add_group(osp.basename(filename)).uuid
        for obj in objs:
            obj.metadata["source"] = filename
            self.add_object(obj, group_id=group_id, set_current=obj is objs[-1])
        self.selection_changed()
        return objs

    def __save_to_file(self, obj: SignalObj | ImageObj, filename: str) -> None:
        """Save object to file (signal/image).

        Args:
            obj: object
            filename: file name
        """
        self.IO_REGISTRY.write(filename, obj)

    def load_from_files(
        self, filenames: list[str] | None = None
    ) -> list[SignalObj | ImageObj]:
        """Open objects from file (signals/images), add them to DataLab and return them.

        Args:
            filenames: File names

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
                objs += self.__load_from_file(filename)
        return objs

    def save_to_files(self, filenames: list[str] | str | None = None) -> None:
        """Save selected objects to files (signal/image).

        Args:
            filenames: File names
        """
        objs = self.objview.get_sel_objects(include_groups=True)
        if filenames is None:  # pragma: no cover
            filenames = [None] * len(objs)
        assert len(filenames) == len(
            objs
        ), "Number of filenames must match number of objects"
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
        h5_fnames = [fname for fname in filenames if fname.endswith(".h5")]
        other_fnames = sorted(list(set(filenames) - set(h5_fnames)))
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
            self.SIG_REFRESH_PLOT.emit("selected", True)

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
        self.SIG_REFRESH_PLOT.emit("selected", update_items)

    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: update object properties,
        refresh plot and update object view."""
        obj = self.objview.get_current_object()
        update_dataset(obj, self.objprop.properties.dataset)
        self.objview.update_item(obj.uuid)
        self.SIG_REFRESH_PLOT.emit("selected", True)

    # ------Plotting data in modal dialogs----------------------------------------------
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
        title = _("Annotations")
        if oids is None:
            oids = self.objview.get_sel_object_uuids(include_groups=True)
        obj = self.objmodel[oids[0]]
        dlg = self.create_new_dialog(
            oids,
            edit=True,
            name="new_window",
            options={"show_itemlist": edit_annotations},
        )
        if dlg is None:
            return None
        width, height = self.DIALOGSIZE
        dlg.resize(width, height)
        mgr = dlg.get_manager()
        toolbar = QW.QToolBar(title, self)
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
        plot.unselect_all()
        for item in plot.items:
            item.set_selectable(False)
        for item in obj.iterate_shape_items(editable=True):
            plot.add_item(item)
        self.__separate_views[dlg] = obj
        toggle_annotations(edit_annotations)
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

    def manual_refresh(self) -> None:
        """Manual refresh"""
        self.plothandler.refresh_plot("selected", True, force=True)

    def create_new_dialog(
        self,
        oids: list[str],
        edit: bool = False,
        toolbar: bool = True,
        title: str | None = None,
        tools: list[GuiTool] | None = None,
        name: str | None = None,
        options: dict | None = None,
    ) -> PlotDialog | None:
        """Create new pop-up signal/image plot dialog.

        Args:
            oids: Object IDs
            edit: Edit mode
            toolbar: Show toolbar
            title: Dialog title
            tools: list of tools to add to the toolbar
            name: Dialog name
            options: Plot options

        Returns:
            QDialog instance
        """
        if title is not None or len(oids) == 1:
            if title is None:
                title = self.objview.get_sel_objects(include_groups=True)[0].title
            title = f"{title} - {APP_NAME}"
        else:
            title = APP_NAME

        plot_options = self.plothandler.get_current_plot_options()
        if options is not None:
            plot_options = plot_options.copy(options)

        # pylint: disable=not-callable
        dlg = PlotDialog(
            parent=self.parent(),
            title=title,
            edit=edit,
            options=plot_options,
            toolbar=toolbar,
        )
        dlg.setWindowIcon(get_icon("DataLab.svg"))
        if tools is not None:
            for tool in tools:
                dlg.get_manager().add_tool(tool)
        plot = dlg.get_plot()

        objs = self.objmodel.get_objects(oids)
        dlg.setObjectName(f"{objs[0].PREFIX}_{name}")

        with create_progress_bar(
            self, _("Creating plot items"), max_=len(objs)
        ) as progress:
            for index, obj in enumerate(objs):
                progress.setValue(index + 1)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    return None
                item = obj.make_item(update_from=self.plothandler[obj.uuid])
                item.set_readonly(True)
                plot.add_item(item, z=0)
        plot.set_active_item(item)
        plot.replot()
        return dlg

    def create_new_dialog_for_selection(
        self,
        title: str,
        name: str,
        options: dict[str, any] = None,
        toolbar: bool = False,
        tools: list[GuiTool] = None,
    ) -> tuple[PlotDialog | None, SignalObj | ImageObj]:
        """Create new pop-up dialog for the currently selected signal/image.

        Args:
            title: Dialog title
            name: Dialog name
            options: Plot options
            toolbar: Show toolbar
            tools: list of tools to add to the toolbar

        Returns:
            QDialog instance, selected object
        """
        obj = self.objview.get_sel_objects(include_groups=True)[0]
        dlg = self.create_new_dialog(
            [obj.uuid],
            edit=True,
            toolbar=toolbar,
            title=f"{title} - {obj.title}",
            tools=tools,
            name=name,
            options=options,
        )
        return dlg, obj

    def get_roi_dialog(
        self, extract: bool, singleobj: bool, add_roi: bool = False
    ) -> tuple[cdl.computation.base.ROIDataParam, bool] | None:
        """Get ROI data (array) from specific dialog box.

        Args:
            extract: Extract ROI from data
            singleobj: Single object
            add_roi: Add ROI immediately after opening the dialog (default: False)

        Returns:
            ROI data
        """
        roi_s = _("Regions of interest")
        options = self.ROIDIALOGOPTIONS
        dlg, obj = self.create_new_dialog_for_selection(roi_s, "roi_dialog", options)
        if dlg is None:
            return None
        plot = dlg.get_plot()
        plot.unselect_all()
        for item in plot.items:
            item.set_selectable(False)
        # pylint: disable=not-callable
        roi_editor: roieditor.SignalROIEditor | roieditor.ImageROIEditor = (
            self.ROIDIALOGCLASS(dlg, obj, extract, singleobj)
        )
        dlg.plot_layout.addWidget(roi_editor, 1, 0, 1, 1)
        if add_roi:
            roi_editor.add_roi()
        if exec_dialog(dlg):
            return roi_editor.get_roieditor_results()
        return None

    def get_object_with_dialog(
        self, title: str, parent: QW.QWidget | None = None
    ) -> SignalObj | ImageObj | None:
        """Get object with dialog box.

        Args:
            title: Dialog title
            parent: Parent widget

        Returns:
            Object (signal or image, or None if dialog was canceled)
        """
        parent = self if parent is None else parent
        dlg = objectview.GetObjectDialog(parent, self, title)
        if exec_dialog(dlg):
            obj_uuid = dlg.get_current_object_uuid()
            return self.objmodel[obj_uuid]
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
            _("Show results obtained from previous computations"),
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
                    "This feature leverages the results of previous computations "
                    "performed on the selected object(s).<br><br>"
                    "To compute results, select one or more objects and choose "
                    "a computing feature in the <u>Compute</u> menu."
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
                xchoices = (("indexes", _("Indexes")),)
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
                    xaxis = gds.ChoiceItem(_("X axis"), xchoices, default="indexes")
                    yaxis = gds.ChoiceItem(
                        _("Y axis"), ychoices, default=ychoices[0][0]
                    )

                comment = (
                    _(
                        "Plot results obtained from previous computations.<br><br>"
                        "This plot is based on results associated with '%s' prefix."
                    )
                    % category
                )
                param = PlotResultParam(_("Plot results"), comment=comment)
                if not param.edit(parent=self.parent()):
                    return

                i_yaxis = rdata.xlabels.index(param.yaxis)
                if param.kind == "one_curve_per_title":
                    # One curve per result title:
                    for title, results in grouped_results.items():  # title
                        x, y = [], []
                        for index, result in enumerate(results):  # object
                            if param.xaxis == "indexes":
                                x.append(index)
                            else:
                                i_xaxis = rdata.xlabels.index(param.xaxis)
                                x.append(result.shown_array[0][i_xaxis])
                            y.append(result.shown_array[0][i_yaxis])
                        self.__add_result_signal(x, y, title, param.xaxis, param.yaxis)
                else:
                    # One curve per result title, per object and per ROI:
                    for title, results in grouped_results.items():  # title
                        for index, result in enumerate(results):  # object
                            roi_idx = np.array(np.unique(result.array[:, 0]), dtype=int)
                            for i_roi in roi_idx:  # ROI
                                mask = result.array[:, 0] == i_roi
                                if param.xaxis == "indexes":
                                    x = np.arange(result.array.shape[0])[mask]
                                else:
                                    i_xaxis = rdata.xlabels.index(param.xaxis)
                                    x = result.shown_array[mask, i_xaxis]
                                y = result.shown_array[mask, i_yaxis]
                                stitle = f"{title} ({objs[index].short_id})"
                                if len(roi_idx) > 1:
                                    stitle += f"|ROI{i_roi}"
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
                self.SIG_REFRESH_PLOT.emit("selected", True)
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
            obj.add_label_with_title(title=title)
        self.SIG_REFRESH_PLOT.emit("selected", True)
