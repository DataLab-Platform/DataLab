# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Panel widgets (core.gui.panel)
--------------------------------------

Signal and Image Panel widgets relie on components:

  * `ObjectProp`: widget handling signal/image properties
  using a guidata DataSet

  * `core.gui.panel.objecthandler.ObjectHandler`: widget handling signal/image list

  * `core.gui.panel.actionhandler.SignalActionHandler` or `ImageActionHandler`:
  classes handling Qt actions

  * `core.gui.panel.plothandler.SignalPlotHandler` or `ImagePlotHandler`:
  classes handling guiqwt plot items

  * `core.gui.panel.processor.signal.SignalProcessor` or
  `core.gui.panel.processor.image.ImageProcessor`: classes handling computing features

  * `core.gui.panel.roieditor.SignalROIEditor` or `ImageROIEditor`:
  classes handling ROI editor widgets

.. autosummary::
    :toctree:

    ObjectProp
    BaseDataPanel

.. autoclass:: ObjectProp
    :members:

.. autoclass:: BaseDataPanel
    :members:
    :inherited-members:

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import dataclasses
import re
import warnings
from typing import TYPE_CHECKING

import guidata.dataset.qtwidgets as gdq
import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions
from guidata.utils import update_dataset
from guidata.widgets.arrayeditor import ArrayEditor
from guiqwt.tools import (
    HCursorTool,
    LabelTool,
    RectangleTool,
    SegmentTool,
    VCursorTool,
    XCursorTool,
)
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getopenfilenames, getsavefilename

from cdl.config import APP_NAME, Conf, _
from cdl.core.gui import actionhandler, objectmodel, objectview, roieditor
from cdl.core.io.base import IOAction
from cdl.core.model.base import MetadataItem, ResultShape
from cdl.utils.qthelpers import (
    create_progress_bar,
    exec_dialog,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)

if TYPE_CHECKING:  # pragma: no cover
    import guidata.dataset.datatypes as gdt
    from guiqwt.plot import CurveDialog, CurveWidget, ImageDialog, ImageWidget
    from guiqwt.tools import GuiTool

    from cdl.core.gui import ObjItf
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.gui.plothandler import BasePlotHandler
    from cdl.core.gui.processor.base import BaseProcessor
    from cdl.core.io.image import ImageIORegistry
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer
    from cdl.core.io.signal import SignalIORegistry
    from cdl.core.model.base import ShapeTypes
    from cdl.core.model.image import ImageObj, NewImageParam
    from cdl.core.model.signal import NewSignalParam, SignalObj

#  Registering MetadataItem edit widget
gdq.DataSetEditLayout.register(MetadataItem, gdq.ButtonWidget)


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

    @staticmethod
    def get_serializable_name(obj: ObjItf) -> str:
        """Return serializable name of object"""
        title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
        name = f"{obj.short_id}: {title}"
        return name

    def serialize_object_to_hdf5(
        self, obj: SignalObj | ImageObj, writer: NativeH5Writer
    ) -> None:
        """Serialize object to HDF5 file"""
        with writer.group(self.get_serializable_name(obj)):
            obj.serialize(writer)

    def deserialize_object_from_hdf5(
        self, reader: NativeH5Reader, name: str
    ) -> SignalObj | ImageObj:
        """Deserialize object from a HDF5 file"""
        with reader.group(name):
            obj = self.create_object()
            obj.deserialize(reader)
        return obj

    @abc.abstractmethod
    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file"""

    @abc.abstractmethod
    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file"""

    @property
    @abc.abstractmethod
    def object_number(self):
        """Return object number"""

    @abc.abstractmethod
    def create_object(self, title=None) -> ObjItf:
        """Create and return object

        Args:
            title: object title
        """

    @abc.abstractmethod
    def add_object(self, obj: ObjItf) -> None:
        """Add object to panel"""

    @abc.abstractmethod
    def remove_all_objects(self):
        """Remove all objects"""
        self.SIG_OBJECT_REMOVED.emit()


class BaseDataPanel(AbstractPanel):
    """Object handling the item list, the selected item properties and plot"""

    PANEL_STR = ""  # e.g. "Signal Panel"
    PARAMCLASS: SignalObj | ImageObj = None  # Replaced in child object
    DIALOGCLASS: CurveDialog | ImageDialog = None  # Idem
    ANNOTATION_TOOLS = (
        LabelTool,
        VCursorTool,
        HCursorTool,
        XCursorTool,
        SegmentTool,
        RectangleTool,
    )
    DIALOGSIZE = (800, 600)
    # Replaced by the right class in child object:
    IO_REGISTRY: SignalIORegistry | ImageIORegistry | None = None
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_UPDATE_PLOT_ITEM = QC.Signal(str)  # Update plot item associated to uuid
    SIG_UPDATE_PLOT_ITEMS = QC.Signal()  # Update plot items associated to selected objs
    ROIDIALOGOPTIONS = {}
    # Replaced in child object:
    ROIDIALOGCLASS: roieditor.SignalROIEditor | roieditor.ImageROIEditor | None = None

    @abc.abstractmethod
    def __init__(
        self, parent: QW.QWidget, plotwidget: CurveWidget | ImageWidget, toolbar
    ) -> None:
        super().__init__(parent)
        self.mainwindow: CDLMainWindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objmodel = objectmodel.ObjectModel()
        self.objview = objectview.ObjectView(self, self.objmodel)
        self.objview.SIG_IMPORT_FILES.connect(self.handle_dropped_files)
        self.objview.populate_tree()
        self.plothandler: BasePlotHandler = None
        self.processor: BaseProcessor = None
        self.acthandler: actionhandler.BaseActionHandler = None
        self.__metadata_clipboard = {}
        self.context_menu = QW.QMenu()
        self.__separate_views: dict[QW.QDialog, SignalObj | ImageObj] = {}

    def closeEvent(self, event):
        """Reimplement QMainWindow method"""
        self.processor.close()
        super().closeEvent(event)

    # ------AbstractPanel interface-----------------------------------------------------
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
                        group.set_title(reader.read_str())
                    for obj_name in reader.h5.get(f"{self.H5_PREFIX}/{name}", []):
                        obj = self.deserialize_object_from_hdf5(reader, obj_name)
                        self.add_object(obj, group.uuid, set_current=False)
                    self.selection_changed()

    @property
    def object_number(self) -> int:
        """Return number of objects in panel

        Returns:
            number of objects (signals or images)
        """
        return len(self.objmodel)

    def create_object(self, title: str | None = None) -> SignalObj | ImageObj:
        """Create object (signal or image)

        Args:
            title: object title

        Returns:
            SignalObj or ImageObj object
        """
        # TODO: [P2] Add default signal/image visualization settings
        # 1. Initialize here (at object creation) metadata with default settings
        #    (see guiqwt.styles.CurveParam and ImageObj for inspiration)
        # 2. Add a dialog box to edit default settings in main window
        #    (use a guidata dataset with only a selection of items from guiqwt.styles
        #     classes)
        # 3. Update all active objects when settings were changed
        # 4. Persist settings in .INI configuration file
        obj = self.PARAMCLASS(title=title)  # pylint: disable=not-callable
        obj.title = title
        return obj

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
        self.objview.add_object_item(obj, group_id, set_current=set_current)
        self.SIG_OBJECT_ADDED.emit()

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        # iterate over a copy of self.__separate_views dict keys to avoid RuntimeError:
        # dictionary changed size during iteration
        for dlg in list(self.__separate_views):
            dlg.done(QW.QDialog.DialogCode.Rejected)
        self.objmodel.clear()
        self.plothandler.clear()
        self.objview.populate_tree()
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        super().remove_all_objects()

    # ---- Signal/Image Panel API ------------------------------------------------------
    def setup_panel(self) -> None:
        """Setup panel"""
        self.acthandler.create_all_actions()
        self.processor.SIG_ADD_SHAPE.connect(self.plothandler.add_shapes)
        self.SIG_UPDATE_PLOT_ITEM.connect(self.plothandler.refresh_plot)
        self.SIG_UPDATE_PLOT_ITEMS.connect(self.plothandler.refresh_plot)
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
        self.add_results_button()

    def get_category_actions(
        self, category: actionhandler.ActionCategory
    ) -> list[QW.QAction]:  # pragma: no cover
        """Return actions for category"""
        return self.acthandler.feature_actions.get(category, [])

    def __popup_contextmenu(self, position: QC.QPoint) -> None:  # pragma: no cover
        """Popup context menu at position"""
        # Note: For now, this is completely unnecessary to clear context menu everytime,
        # but implementing it this way could be useful in the future in menu contents
        # should take into account current object selection
        self.context_menu.clear()
        actions = self.get_category_actions(actionhandler.ActionCategory.CONTEXT_MENU)
        add_actions(self.context_menu, actions)
        self.context_menu.popup(position)

    # ------Creating, adding, removing objects------------------------------------------
    def add_group(self, title: str) -> objectmodel.ObjectGroup:
        """Add group"""
        group = self.objmodel.add_group(title)
        self.objview.add_group_item(group)
        return group

    # TODO: [P2] New feature: move objects up/down
    # TODO: [P2] New feature: move objects to another group
    def __duplicate_individual_obj(
        self, oid: str, new_group_id: str | None = None, set_current: bool = True
    ) -> None:
        """Duplicate individual object"""
        obj = self.objmodel[oid]
        objcopy = self.create_object()
        objcopy.title = obj.title
        objcopy.copy_data_from(obj)
        if new_group_id is None:
            new_group_id = self.objmodel.get_object_group_id(obj)
        self.add_object(objcopy, group_id=new_group_id, set_current=set_current)

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
        self.selection_changed()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def copy_metadata(self) -> None:
        """Copy object metadata"""
        obj = self.objview.get_sel_objects()[0]
        self.__metadata_clipboard = obj.metadata.copy()
        new_pref = obj.short_id + "_"
        for key, value in obj.metadata.items():
            if ResultShape.match(key, value):
                mshape = ResultShape.from_metadata_entry(key, value)
                if not re.match(obj.PREFIX + r"[0-9]{3}[\s]*", mshape.label):
                    # Handling additional result (e.g. diameter)
                    for a_key, a_value in obj.metadata.items():
                        if isinstance(a_key, str) and a_key.startswith(mshape.label):
                            self.__metadata_clipboard.pop(a_key)
                            self.__metadata_clipboard[new_pref + a_key] = a_value
                    mshape.label = new_pref + mshape.label
                    # Handling result shape
                    self.__metadata_clipboard.pop(key)
                    self.__metadata_clipboard[mshape.key] = value

    def paste_metadata(self) -> None:
        """Paste metadata to selected object(s)"""
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sorted(sel_objects, key=lambda obj: obj.short_id, reverse=True):
            obj.metadata.update(self.__metadata_clipboard)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def remove_object(self) -> None:
        """Remove signal/image object"""
        sel_groups = self.objview.get_sel_groups()
        if sel_groups:
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
            for dlg, obj_i in self.__separate_views.items():
                if obj_i is obj:
                    dlg.done(QW.QDialog.DialogCode.Rejected)
            self.plothandler.remove_item(obj.uuid)
            self.objview.remove_item(obj.uuid, refresh=False)
            self.objmodel.remove_object(obj)
        for group in sel_groups:
            self.objview.remove_item(group.uuid, refresh=False)
            self.objmodel.remove_group(group)
        self.objview.update_tree()
        self.selection_changed()
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        self.SIG_OBJECT_REMOVED.emit()

    def delete_all_objects(self) -> None:  # pragma: no cover
        """Confirm before removing all objects"""
        if self.object_number == 0:
            return
        answer = QW.QMessageBox.warning(
            self,
            _("Delete all"),
            _("Do you want to delete all objects (%s)?") % self.PANEL_STR,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
        )
        if answer == QW.QMessageBox.Yes:
            self.remove_all_objects()

    def delete_metadata(self) -> None:
        """Delete object metadata"""
        for index, obj in enumerate(self.objview.get_sel_objects(include_groups=True)):
            obj.reset_metadata_to_defaults()
            if index == 0:
                self.selection_changed()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

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
            group.set_title(group_name)
            self.objview.update_item(group.uuid)

    @abc.abstractmethod
    def new_object(
        self,
        newparam: NewSignalParam | NewImageParam | None = None,
        addparam: gdt.DataSet | None = None,
        edit: bool = True,
    ) -> SignalObj | ImageObj | None:
        """Create a new object (signal/image).

        Args:
            newparam (DataSet): new object parameters
            addparam (DataSet): additional parameters
            edit (bool): Open a dialog box to edit parameters (default: True)

        Returns:
            New object
        """

    def open_object(
        self, filename: str
    ) -> SignalObj | ImageObj | list[SignalObj | ImageObj]:
        """Open object from file (signal/image), add it to DataLab and return it.

        Args:
            filename (str): file name

        Returns:
            New object or list of new objects
        """
        obj_or_objlist = self.IO_REGISTRY.read(filename)
        objs = obj_or_objlist if isinstance(obj_or_objlist, list) else [obj_or_objlist]
        for obj in objs:
            self.add_object(obj, set_current=obj is objs[-1])
        self.selection_changed()
        if len(objs) == 1:
            return objs[0]
        return objs

    def save_object(self, obj, filename: str | None = None) -> None:
        """Save object to file (signal/image)"""
        if filename is None:
            basedir = Conf.main.base_dir.get()
            filters = self.IO_REGISTRY.get_filters(IOAction.SAVE)
            with save_restore_stds():
                filename, _filt = getsavefilename(self, _("Save as"), basedir, filters)
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                self.IO_REGISTRY.write(filename, obj)

    def handle_dropped_files(self, filenames: list[str] | None = None) -> None:
        """Handle dropped files

        Args:
            filenames (list(str)): File names

        Returns:
            None
        """
        h5_fnames = [fname for fname in filenames if fname.endswith(".h5")]
        other_fnames = list(set(filenames) - set(h5_fnames))
        if h5_fnames:
            self.mainwindow.open_h5_files(h5_fnames, import_all=True)
        if other_fnames:
            self.open_objects(other_fnames)

    def open_objects(
        self, filenames: list[str] | None = None
    ) -> list[SignalObj | ImageObj]:
        """Open objects from file (signals/images), add them to DataLab and return them.

        Args:
            filenames (list(str)): File names

        Returns:
            list of new objects
        """
        if not self.mainwindow.confirm_memory_state():
            return []
        if filenames is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            filters = self.IO_REGISTRY.get_filters(IOAction.LOAD)
            with save_restore_stds():
                filenames, _filt = getopenfilenames(self, _("Open"), basedir, filters)
        objs = []
        for filename in filenames:
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                Conf.main.base_dir.set(filename)
                objs.append(self.open_object(filename))
        return objs

    def save_objects(self, filenames: list[str] | None = None) -> None:
        """Save selected objects to file (signal/image).

        Args:
            filenames (list(str)): File names

        Returns:
            None
        """
        objs = self.objview.get_sel_objects(include_groups=True)
        if filenames is None:  # pragma: no cover
            filenames = [None] * len(objs)
        assert len(filenames) == len(objs)
        for index, obj in enumerate(objs):
            filename = filenames[index]
            self.save_object(obj, filename)

    def import_metadata_from_file(self, filename: str | None = None) -> None:
        """Import metadata from file (JSON).

        Args:
            filename (str): File name

        Returns:
            None
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
                obj.import_metadata_from_file(filename)
            self.SIG_UPDATE_PLOT_ITEMS.emit()

    def export_metadata_from_file(self, filename: str | None = None) -> None:
        """Export metadata to file (JSON).

        Args:
            filename (str): File name

        Returns:
            None
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
                obj.export_metadata_to_file(filename)

    # ------Refreshing GUI--------------------------------------------------------------
    def selection_changed(self) -> None:
        """Object selection changed: update object properties, refresh plot and update
        object view.
        """
        selected_objects = self.objview.get_sel_objects(include_groups=True)
        selected_groups = self.objview.get_sel_groups()
        self.objprop.update_properties_from(self.objview.get_current_object())
        self.plothandler.refresh_plot(just_show=True)
        self.acthandler.selected_objects_changed(selected_groups, selected_objects)

    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: update object properties,
        refresh plot and update object view."""
        obj = self.objview.get_current_object()
        update_dataset(obj, self.objprop.properties.dataset)
        self.objview.update_item(obj.uuid)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    # ------Plotting data in modal dialogs----------------------------------------------
    def open_separate_view(self, oids: list[str] | None = None) -> QW.QDialog | None:
        """
        Open separate view for visualizing selected objects

        Args:
            oids (list(str)): Object IDs

        Returns:
            QDialog instance
        """
        title = _("Annotations")
        if oids is None:
            oids = self.objview.get_sel_object_uuids(include_groups=True)
        obj = self.objmodel[oids[0]]
        dlg = self.create_new_dialog(oids, edit=True, name="new_window")
        if dlg is None:
            return None
        width, height = self.DIALOGSIZE
        dlg.resize(width, height)
        dlg.get_itemlist_panel().show()
        toolbar = QW.QToolBar(title, self)
        dlg.button_layout.insertWidget(0, toolbar)
        # dlg.layout().insertWidget(1, toolbar)  # other possible location
        # dlg.plot_layout.addWidget(toolbar, 1, 0, 1, 1)  # other possible location
        dlg.add_toolbar(toolbar, id(toolbar))
        toolbar.setToolButtonStyle(QC.Qt.ToolButtonTextUnderIcon)
        for tool in self.ANNOTATION_TOOLS:
            dlg.add_tool(tool, toolbar_id=id(toolbar))
        plot = dlg.get_plot()
        plot.unselect_all()
        for item in plot.items:
            item.set_selectable(False)
        for item in obj.iterate_shape_items(editable=True):
            plot.add_item(item)
        self.__separate_views[dlg] = obj
        dlg.show()
        dlg.finished.connect(self.__separate_view_finished)
        return dlg

    def __separate_view_finished(self, result: int) -> None:
        """Separate view was closed"""
        dlg: CurveDialog | ImageDialog = self.sender()
        if result == QW.QDialog.DialogCode.Accepted:
            items = dlg.get_plot().get_items()
            rw_items = [item for item in items if not item.is_readonly()]
            if rw_items:
                obj = self.__separate_views[dlg]
                obj.set_annotations_from_items(rw_items)
                self.selection_changed()
                self.SIG_UPDATE_PLOT_ITEMS.emit()
        self.__separate_views.pop(dlg)
        dlg.deleteLater()

    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show annotations option"""
        Conf.view.show_label.set(state)
        for obj in self.objmodel:
            obj.metadata[obj.METADATA_LBL] = state
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def create_new_dialog(
        self,
        oids: list[str],
        edit: bool = False,
        toolbar: bool = True,
        title: str | None = None,
        tools: list[GuiTool] | None = None,
        name: str | None = None,
        options: dict | None = None,
    ) -> CurveDialog | ImageDialog | None:
        """Create new pop-up signal/image plot dialog.

        Args:
            oids (list(str)): Object IDs
            edit (bool): Edit mode
            toolbar (bool): Show toolbar
            title (str): Dialog title
            tools (list(GuiTool)): list of tools to add to the toolbar
            name (str): Dialog name
            options (dict): Plot options

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
            plot_options.update(options)

        # pylint: disable=not-callable
        dlg: CurveDialog | ImageDialog = self.DIALOGCLASS(
            parent=self,
            wintitle=title,
            edit=edit,
            options=plot_options,
            toolbar=toolbar,
        )
        dlg.setWindowIcon(get_icon("DataLab.svg"))
        if tools is not None:
            for tool in tools:
                dlg.add_tool(tool)
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
    ) -> tuple[QW.QDialog | None, SignalObj | ImageObj]:
        """Create new pop-up dialog for the currently selected signal/image.

        Args:
            title (str): Dialog title
            name (str): Dialog name
            options (dict): Plot options
            toolbar (bool): Show toolbar
            tools (list(GuiTool)): list of tools to add to the toolbar

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

    def get_roi_dialog(self, extract: bool, singleobj: bool) -> roieditor.ROIEditorData:
        """Get ROI data (array) from specific dialog box.

        Args:
            extract (bool): Extract ROI from data
            singleobj (bool): Single object

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
        roi_editor = self.ROIDIALOGCLASS(dlg, obj, extract, singleobj)
        dlg.plot_layout.addWidget(roi_editor, 1, 0, 1, 1)
        if exec_dialog(dlg):
            return roi_editor.get_data()
        return None

    def get_object_dialog(
        self, title: str, parent: QW.QWidget | None = None
    ) -> objectview.GetObjectDialog:
        """Get object dialog.

        Args:
            title (str): Dialog title
            parent (QWidget): Parent widget

        Returns:
            GetObjectDialog instance
        """
        parent = self if parent is None else parent
        dlg = objectview.GetObjectDialog(parent, self, title)
        if exec_dialog(dlg):
            return dlg.get_current_object()
        return None

    def add_results_button(self) -> None:
        """Add 'Show results' button"""
        btn = QW.QPushButton(get_icon("show_results.svg"), _("Show results"), self)
        btn.setToolTip(_("Show results obtained from previous computations"))
        self.objprop.add_button(btn)
        btn.clicked.connect(self.show_results)
        self.acthandler.add_action(
            btn,
            select_condition=actionhandler.SelectCond.at_least_one,
        )

    def show_results(self) -> None:
        """Show results"""

        @dataclasses.dataclass
        class ResultData:
            """Result data associated to a shapetype"""

            results: list[ResultShape] = None
            xlabels: list[str] = None
            ylabels: list[str] = None

        rdatadict: dict[ShapeTypes, ResultData] = {}
        objs = self.objview.get_sel_objects(include_groups=True)
        for obj in objs:
            for result in obj.iterate_resultshapes():
                rdata = rdatadict.setdefault(result.shapetype, ResultData([], None, []))
                title = f"{result.label}"
                rdata.results.append(result)
                rdata.xlabels = result.xlabels
                for _i_row_res in range(result.array.shape[0]):
                    ylabel = f"{obj.short_id}: {result.label}"
                    rdata.ylabels.append(ylabel)
        if rdatadict:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for rdata in rdatadict.values():
                    dlg = ArrayEditor(self.parent())
                    title = _("Results")
                    dlg.setup_and_check(
                        np.vstack([result.array for result in rdata.results]),
                        title,
                        readonly=True,
                        xlabels=rdata.xlabels,
                        ylabels=rdata.ylabels,
                    )
                    dlg.setObjectName(f"{objs[0].PREFIX}_results")
                    dlg.resize(750, 300)
                    exec_dialog(dlg)
        else:
            msg = "<br>".join(
                [
                    _("No result currently available for this object."),
                    "",
                    _(
                        "This feature shows result arrays as displayed after "
                        'calling one of the computing feature (see "Compute" menu).'
                    ),
                ]
            )
            QW.QMessageBox.information(self, APP_NAME, msg)
