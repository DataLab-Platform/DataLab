# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

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

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
import dataclasses
import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

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
    exec_dialog,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)

if TYPE_CHECKING:
    from guiqwt.plot import CurveDialog, ImageDialog
    from guiqwt.tools import GuiTool

    from cdl.core.gui import ObjItf
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.gui.plothandler import BasePlotHandler
    from cdl.core.gui.processor.base import BaseProcessor
    from cdl.core.io.base import BaseIORegistry
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer
    from cdl.core.model.base import ShapeTypes
    from cdl.core.model.image import ImageParam
    from cdl.core.model.signal import SignalParam

#  Registering MetadataItem edit widget
gdq.DataSetEditLayout.register(MetadataItem, gdq.ButtonWidget)


class ObjectProp(QW.QWidget):
    """Object handling panel properties"""

    def __init__(self, panel: BaseDataPanel, paramclass: SignalParam | ImageParam):
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

    def set_param_label(self, param: SignalParam | ImageParam):
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

    def update_properties_from(self, param: SignalParam | ImageParam = None):
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

    @property
    @abc.abstractmethod
    def object_number(self):
        """Return object number"""

    @abc.abstractmethod
    def create_object(self, title=None) -> ObjItf:
        """Create and return object

        :param str title: Title of the object
        """

    @abc.abstractmethod
    def add_object(self, obj: ObjItf) -> None:
        """Add object to panel"""

    @abc.abstractmethod
    def remove_all_objects(self):
        """Remove all objects"""
        self.SIG_OBJECT_REMOVED.emit()

    @staticmethod
    def get_serializable_name(obj: ObjItf) -> str:
        """Return serializable name of object"""
        title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
        name = f"{obj.short_id}: {title}"
        return name

    def serialize_object_to_hdf5(
        self, obj: SignalParam | ImageParam, writer: NativeH5Writer
    ) -> None:
        """Serialize object to HDF5 file"""
        with writer.group(self.get_serializable_name(obj)):
            obj.serialize(writer)

    # pylint: disable=unused-argument
    def deserialize_objects_from_hdf5(self, reader: NativeH5Reader, name: str) -> None:
        """Deserialize objects from a HDF5 file"""
        obj = self.create_object()
        obj.deserialize(reader)
        self.add_object(obj)
        QW.QApplication.processEvents()

    @abc.abstractmethod
    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file"""

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize whole panel from a HDF5 file"""
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                with reader.group(name):
                    self.deserialize_objects_from_hdf5(reader, name)


class BaseDataPanel(AbstractPanel):
    """Object handling the item list, the selected item properties and plot"""

    PANEL_STR = ""  # e.g. "Signal Panel"
    PARAMCLASS: SignalParam | ImageParam = None  # Replaced in child object
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
    IO_REGISTRY: BaseIORegistry = None  # Replaced by the right class in child object
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_UPDATE_PLOT_ITEM = QC.Signal(str)  # Update plot item associated to uuid
    SIG_UPDATE_PLOT_ITEMS = QC.Signal()  # Update plot items associated to selected objs
    ROIDIALOGOPTIONS = {}
    ROIDIALOGCLASS: roieditor.BaseROIEditor = None  # Replaced in child object

    @abc.abstractmethod
    def __init__(self, parent, plotwidget, toolbar):
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
        self.__separate_views: Dict[QW.QDialog, SignalParam | ImageParam] = {}

    # ------AbstractPanel interface-----------------------------------------------------
    @property
    def object_number(self) -> int:
        """Return object number"""
        return len(self.objmodel)

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        for dlg in self.__separate_views:
            dlg.done(QW.QDialog.DialogCode.Rejected)
        self.objmodel.clear()
        self.plothandler.clear()
        self.objview.populate_tree()
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        super().remove_all_objects()

    def create_object(self, title: Optional[str] = None) -> SignalParam | ImageParam:
        """Create object (signal or image)

        :param str title: Title of the object
        """
        # TODO: [P2] Add default signal/image visualization settings
        # 1. Initialize here (at object creation) metadata with default settings
        #    (see guiqwt.styles.CurveParam and ImageParam for inspiration)
        # 2. Add a dialog box to edit default settings in main window
        #    (use a guidata dataset with only a selection of items from guiqwt.styles
        #     classes)
        # 3. Update all active objects when settings were changed
        # 4. Persist settings in .INI configuration file
        obj = self.PARAMCLASS(title=title)
        obj.title = title
        return obj

    @qt_try_except()
    def add_object(self, obj: SignalParam | ImageParam, group_id: str = None):
        """Add object

        :param bool refresh: Refresh object list (e.g. listwidget for signals/images)"""
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
        self.objview.add_object_item(obj, group_id)
        self.SIG_OBJECT_ADDED.emit()

    def add_group(self, title: str) -> objectmodel.ObjectGroup:
        """Add group"""
        group = self.objmodel.add_group(title)
        self.objview.add_group_item(group)
        return group

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file"""
        with writer.group(self.H5_PREFIX):
            for group in self.objmodel.get_groups():
                with writer.group(self.get_serializable_name(group)):
                    with writer.group("title"):
                        writer.write_str(group.title)
                    for obj in group.get_objects():
                        self.serialize_object_to_hdf5(obj, writer)

    def deserialize_objects_from_hdf5(self, reader: NativeH5Reader, name: str) -> None:
        """Deserialize objects from a HDF5 file"""
        group = self.add_group("")
        with reader.group("title"):
            group.set_title(reader.read_str())
        for obj_name in reader.h5.get(f"{self.H5_PREFIX}/{name}", []):
            obj = self.create_object()
            with reader.group(obj_name):
                obj.deserialize(reader)
            self.add_object(obj, group.uuid)
            QW.QApplication.processEvents()

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
    ) -> List[QW.QAction]:  # pragma: no cover
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

    # TODO: [P2] New feature: move objects up/down
    def __duplicate_individual_object(self, oid: str, new_group_id: str = None) -> None:
        """Duplicate individual object"""
        obj = self.objmodel[oid]
        objcopy = self.create_object()
        objcopy.title = obj.title
        objcopy.copy_data_from(obj)
        if new_group_id is None:
            new_group_id = self.objmodel.get_object_group_id(obj)
        self.add_object(objcopy, group_id=new_group_id)

    def duplicate_object(self) -> None:
        """Duplication signal/image object"""
        if not self.mainwindow.confirm_memory_state():
            return
        # Duplicate individual objects (exclusive with respect to groups)
        for oid in self.objview.get_sel_object_uuids():
            self.__duplicate_individual_object(oid)
        # Duplicate groups (exclusive with respect to individual objects)
        for group in self.objview.get_sel_groups():
            new_group = self.add_group(group.title)
            for oid in self.objmodel.get_group_object_ids(group.uuid):
                self.__duplicate_individual_object(oid, new_group.uuid)
            self.objview.set_current_item_id(new_group.uuid)
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
            self.objview.remove_item(obj.uuid)
            self.objmodel.remove_object(obj)
        for group in sel_groups:
            self.objview.remove_item(group.uuid)
            self.objmodel.remove_group(group)
        self.objview.update_tree()
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
        self, newparam=None, addparam=None, edit=True
    ) -> SignalParam | ImageParam:
        """Create a new object (signal/image).

        :param guidata.dataset.DataSet newparam: new object parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        :return: New object"""

    def open_object(
        self, filename: str
    ) -> Union[SignalParam | ImageParam, List[SignalParam | ImageParam]]:
        """Open object from file (signal/image), add it to DataLab and return it.

        :param str filename: File name
        :return: Object or list of objects"""
        obj_or_objlist = self.IO_REGISTRY.read(filename)
        if not isinstance(obj_or_objlist, list):
            obj_or_objlist = [obj_or_objlist]
        for obj in obj_or_objlist:
            self.add_object(obj)
        if len(obj_or_objlist) == 1:
            return obj_or_objlist[0]
        return obj_or_objlist

    def save_object(self, obj, filename: str = None) -> None:
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

    def handle_dropped_files(self, filenames: List[str] = None) -> None:
        """Handle dropped files"""
        h5_fnames = [fname for fname in filenames if fname.endswith(".h5")]
        other_fnames = list(set(filenames) - set(h5_fnames))
        if h5_fnames:
            self.mainwindow.open_h5_files(h5_fnames, import_all=True)
        if other_fnames:
            self.open_objects(other_fnames)

    def open_objects(
        self, filenames: List[str] = None
    ) -> List[SignalParam | ImageParam]:
        """Open objects from file (signals/images), add them to DataLab and return them.

        :param list(str) filenames: File names
        :return: List of objects"""
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

    def save_objects(self, filenames: List[str] = None) -> None:
        """Save selected objects to file (signal/image).

        :param list(str) filenames: File names"""
        objs = self.objview.get_sel_objects(include_groups=True)
        if filenames is None:  # pragma: no cover
            filenames = [None] * len(objs)
        assert len(filenames) == len(objs)
        for index, obj in enumerate(objs):
            filename = filenames[index]
            self.save_object(obj, filename)

    def import_metadata_from_file(self, filename: str = None) -> None:
        """Import metadata from file (JSON)"""
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

    def export_metadata_from_file(self, filename: str = None) -> None:
        """Export metadata to file (JSON)"""
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
        """Signal list: selection changed"""
        selected_objects = self.objview.get_sel_objects(include_groups=True)
        selected_groups = self.objview.get_sel_groups()
        self.objprop.update_properties_from(self.objview.get_current_object())
        self.plothandler.refresh_plot(just_show=True)
        self.acthandler.selected_objects_changed(selected_groups, selected_objects)

    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: updating signal"""
        obj = self.objview.get_current_object()
        update_dataset(obj, self.objprop.properties.dataset)
        self.objview.update_item(obj.uuid)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    # ------Plotting data in modal dialogs----------------------------------------------
    def open_separate_view(self, oids: Optional[List[str]] = None) -> QW.QDialog:
        """
        Open separate view for visualizing selected objects

        :param list oids: List of object IDs to visualize (default: selected objects)
        :return: Dialog instance
        """
        title = _("Annotations")
        if oids is None:
            oids = self.objview.get_sel_object_uuids(include_groups=True)
        obj = self.objmodel[oids[0]]
        dlg = self.create_new_dialog(oids, edit=True, name="new_window")
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
        dlg = self.sender()
        if result == QW.QDialog.DialogCode.Accepted:
            items = dlg.get_plot().get_items()
            rw_items = [item for item in items if not item.is_readonly()]
            if rw_items:
                obj = self.__separate_views[dlg]
                obj.set_annotations_from_items(rw_items)
                self.selection_changed()
                self.SIG_UPDATE_PLOT_ITEMS.emit()

    def toggle_show_titles(self, state: bool) -> None:
        """Toggle show annotations option"""
        Conf.view.show_label.set(state)
        for obj in self.objmodel:
            obj.metadata[obj.METADATA_LBL] = state
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def create_new_dialog(
        self,
        oids: List[str],
        edit: bool = False,
        toolbar: bool = True,
        title: str = None,
        tools: List[GuiTool] = None,
        name: str = None,
        options: dict = None,
    ) -> CurveDialog | ImageDialog:
        """
        Create new pop-up signal/image plot dialog

        :param list oids: List of uuids for the objects to be shown in dialog
        :param bool edit: If True, show "OK" and "Cancel" buttons
        :param bool toolbar: If True, add toolbar
        :param str title: Title of the dialog box
        :param list tools: List of plot tools
        :param str name: Name of the widget (used as screenshot basename)
        :param dict options: Plot options
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
        for obj in objs:
            item = obj.make_item(update_from=self.plothandler[obj.uuid])
            item.set_readonly(True)
            plot.add_item(item, z=0)
        plot.set_active_item(item)
        plot.replot()
        return dlg

    def create_new_dialog_for_selection(
        self, title, name, options=None, toolbar=False, tools=None
    ):
        """
        Create new pop-up dialog for the currently selected signal/image

        :param str title: Title of the dialog box
        :param str name: Name of the widget (used as screenshot basename)
        :param dict options: Plot options
        :param list tools: List of plot tools
        :return: tuple (dialog, current_object)
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
        """Get ROI data (array) from specific dialog box"""
        roi_s = _("Regions of interest")
        options = self.ROIDIALOGOPTIONS
        dlg, obj = self.create_new_dialog_for_selection(roi_s, "roi_dialog", options)
        plot = dlg.get_plot()
        plot.unselect_all()
        for item in plot.items:
            item.set_selectable(False)
        roi_editor = self.ROIDIALOGCLASS(dlg, obj, extract, singleobj)
        dlg.plot_layout.addWidget(roi_editor, 1, 0, 1, 1)
        if exec_dialog(dlg):
            return roi_editor.get_data()
        return None

    def get_object_dialog(
        self, title: str, parent: Optional[QW.QWidget] = None
    ) -> objectview.GetObjectDialog:
        """Get object dialog"""
        parent = self if parent is None else parent
        dlg = objectview.GetObjectDialog(parent, self, title)
        if exec_dialog(dlg):
            return dlg.get_object()
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

            results: List[ResultShape] = None
            xlabels: List[str] = None
            ylabels: List[str] = None

        rdatadict: Dict[ShapeTypes, ResultData] = {}
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
