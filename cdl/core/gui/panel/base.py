# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab Panel widgets (core.gui.panel)

Signal and Image Panel widgets relie on components:

  * `ObjectProp`: widget handling signal/image properties
  using a guidata DataSet

  * `core.gui.panel.objectlist.ObjectList`: widget handling signal/image list

  * `core.gui.panel.actionhandler.SignalActionHandler` or `ImageActionHandler`:
  classes handling Qt actions

  * `core.gui.panel.plotitemlist.SignalItemList` or `ImageItemList`:
  classes handling guiqwt plot items

  * `core.gui.panel.processor.signal.SignalProcessor` or
  `core.gui.panel.processor.image.ImageProcessor`: classes handling computing features

  * `core.gui.panel.roieditor.SignalROIEditor` or `ImageROIEditor`:
  classes handling ROI editor widgets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations  # To be removed when dropping Python <=3.9 support

import abc
import dataclasses
import os
import re
import warnings
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

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
from cdl.core.gui import actionhandler, objectlist, roieditor
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

    from cdl.core.gui import ObjItf
    from cdl.core.gui.main import CDLMainWindow
    from cdl.core.gui.plotitemlist import BaseItemList
    from cdl.core.gui.processor.base import BaseProcessor
    from cdl.core.io.base import BaseIORegistry
    from cdl.core.io.native import NativeH5Reader, NativeH5Writer
    from cdl.core.model.base import ObjectItf, ShapeTypes
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
        hlayout = QW.QHBoxLayout()
        hlayout.addWidget(self.properties)
        param_layout = QW.QVBoxLayout()
        self.param_label = QW.QLabel()
        self.param_label.setTextFormat(QC.Qt.RichText)
        self.param_label.setTextInteractionFlags(
            QC.Qt.TextBrowserInteraction | QC.Qt.TextSelectableByKeyboard
        )
        self.param_label.setAlignment(QC.Qt.AlignTop)
        param_scroll = QW.QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setWidget(self.param_label)
        param_layout.addWidget(param_scroll)
        self.param_group = QW.QGroupBox(_("Computing parameters"))
        self.param_group.setLayout(param_layout)
        vlayout = QW.QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.param_group)
        self.setLayout(vlayout)

    def add_button(self, button):
        """Add additional button on bottom of properties panel"""
        self.add_prop_layout.addWidget(button)

    def set_param_label(self, param: ObjectItf):
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

    def update_properties_from(self, param: ObjectItf = None):
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
    """Object defining CobraDataLab panel interface,
    based on a vertical QSplitter widget

    A panel handle an object list (objects are signals, images, macros, ...).
    Each object must implement ``cdl.core.gui.ObjItf`` interface
    """

    PREFIX = ""  # e.g. "s"
    H5_PREFIX = ""
    SIG_OBJECT_ADDED = QC.Signal()
    SIG_OBJECT_REMOVED = QC.Signal()

    @abc.abstractmethod
    def __init__(self, parent):
        super().__init__(QC.Qt.Vertical, parent)
        self.setObjectName(self.PREFIX)

    @property
    @abc.abstractmethod
    def object_number(self):
        """Return object number"""

    @abc.abstractmethod
    def object_iterator(self) -> Iterator[ObjItf]:
        """Iterate over objects handled by panel"""

    @abc.abstractmethod
    def create_object(self, title=None) -> ObjItf:
        """Create and return object

        :param str title: Title of the object
        """

    @abc.abstractmethod
    def add_object(self, obj, refresh=True) -> ObjItf:
        """Add object

        :param bool refresh: Refresh object list (e.g. listwidget for signals/images)"""
        self.SIG_OBJECT_ADDED.emit()
        return obj

    @abc.abstractmethod
    def remove_all_objects(self):
        """Remove all objects"""
        self.SIG_OBJECT_REMOVED.emit()

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize objects to a HDF5 file"""
        with writer.group(self.H5_PREFIX):
            for idx, obj in enumerate(self.object_iterator()):
                title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
                name = f"{self.PREFIX}{idx:03d}: {title}"
                with writer.group(name):
                    obj.serialize(writer)

    def deserialize_from_hdf5(self, reader: NativeH5Reader) -> None:
        """Deserialize objects from a HDF5 file"""
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                obj = self.create_object()
                with reader.group(name):
                    obj.deserialize(reader)
                    self.add_object(obj)
                    QW.QApplication.processEvents()


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
    SIG_UPDATE_PLOT_ITEM = QC.Signal(int)  # Update plot item associated to row number
    SIG_UPDATE_PLOT_ITEMS = QC.Signal()  # Update plot items associated to selected rows
    ROIDIALOGOPTIONS = {}
    ROIDIALOGCLASS: roieditor.BaseROIEditor = None  # Replaced in child object

    @abc.abstractmethod
    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent)
        self.mainwindow: CDLMainWindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objlist = objectlist.ObjectList(self)
        self.objlist.SIG_IMPORT_FILES.connect(self.handle_dropped_files)
        self.itmlist: BaseItemList = None
        self.processor: BaseProcessor = None
        self.acthandler: actionhandler.BaseActionHandler = None
        self.__metadata_clipboard = {}
        self.context_menu = QW.QMenu()
        self.__separate_views: Dict[QW.QDialog, ObjectItf] = {}

    # ------AbstractPanel interface-----------------------------------------------------
    @property
    def object_number(self) -> int:
        """Return object number"""
        return len(self.objlist)

    def object_iterator(self) -> Iterator[ObjItf]:
        """Iterate over objects handled by panel"""
        return iter(self.objlist)

    def remove_all_objects(self) -> None:
        """Remove all objects"""
        for dlg in self.__separate_views:
            dlg.done(QW.QDialog.DialogCode.Rejected)
        self.objlist.remove_all()
        self.itmlist.remove_all()
        self.objlist.refresh_list(0)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        super().remove_all_objects()

    def create_object(self, title: Optional[str] = None) -> ObjItf:
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
    def add_object(self, obj: ObjectItf, refresh: bool = True) -> ObjItf:
        """Add object

        :param bool refresh: Refresh object list (e.g. listwidget for signals/images)"""
        obj.check_data()
        self.objlist.append(obj)
        self.itmlist.append(None)
        if refresh:
            self.objlist.refresh_list(-1)
        return super().add_object(obj, refresh=refresh)

    # ---- Signal/Image Panel API ------------------------------------------------------
    def setup_panel(self) -> None:
        """Setup panel"""
        self.acthandler.create_all_actions()
        self.processor.SIG_ADD_SHAPE.connect(self.itmlist.add_shapes)
        self.SIG_UPDATE_PLOT_ITEM.connect(self.itmlist.refresh_plot)
        self.SIG_UPDATE_PLOT_ITEMS.connect(self.itmlist.refresh_plot)
        self.objlist.itemSelectionChanged.connect(self.selection_changed)
        self.objlist.SIG_ITEM_DOUBLECLICKED.connect(
            lambda row: self.open_separate_view([row])
        )
        self.objlist.SIG_CONTEXT_MENU.connect(self.__popup_contextmenu)
        self.objprop.properties.SIG_APPLY_BUTTON_CLICKED.connect(
            self.properties_changed
        )
        self.addWidget(self.objlist)
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
    @qt_try_except()
    def insert_object(self, obj: ObjectItf, row: int, refresh: bool = True) -> None:
        """Insert signal/image object after row"""
        obj.check_data()
        self.objlist.insert(row, obj)
        self.itmlist.insert(row)
        if refresh:
            self.objlist.refresh_list(new_current_row=row + 1)
        self.SIG_OBJECT_ADDED.emit()

    def duplicate_object(self) -> None:
        """Duplication signal/image object"""
        if not self.mainwindow.confirm_memory_state():
            return
        rows = sorted(self.objlist.get_selected_rows())
        row = None
        for row in rows:
            obj = self.objlist[row]
            objcopy = self.create_object()
            objcopy.title = obj.title
            objcopy.copy_data_from(obj)
            self.add_object(objcopy, refresh=False)
        self.objlist.refresh_list(new_current_row=-1)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def copy_metadata(self) -> None:
        """Copy object metadata"""
        row = self.objlist.get_selected_rows()[0]
        obj = self.objlist[row]
        self.__metadata_clipboard = obj.metadata.copy()
        pfx = self.objlist.prefix
        new_pref = f"{pfx}{row:03d}_"
        for key, value in obj.metadata.items():
            if ResultShape.match(key, value):
                mshape = ResultShape.from_metadata_entry(key, value)
                if not re.match(pfx + r"[0-9]{3}[\s]*", mshape.label):
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
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        row = None
        for row in rows:
            obj = self.objlist[row]
            obj.metadata.update(self.__metadata_clipboard)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def remove_object(self) -> None:
        """Remove signal/image object"""
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        for row in rows:
            for dlg, obj in self.__separate_views.items():
                if obj is self.objlist[row]:
                    dlg.done(QW.QDialog.DialogCode.Rejected)
            del self.objlist[row]
            del self.itmlist[row]
        self.objlist.refresh_list(max(0, rows[-1] - 1))
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
        for index, row in enumerate(self.objlist.get_selected_rows()):
            self.objlist[row].reset_metadata_to_defaults()
            if index == 0:
                self.selection_changed()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def copy_titles_to_clipboard(self) -> None:
        """Copy object titles to clipboard (for reproducibility)"""
        text = os.linesep.join(
            [
                f"{self.PREFIX}{idx:03d}: {obj.title}"
                for idx, obj in enumerate(self.objlist)
            ]
        )
        QW.QApplication.clipboard().setText(text)

    @abc.abstractmethod
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new object (signal/image).

        :param guidata.dataset.DataSet newparam: new object parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """

    def open_object(self, filename: str) -> None:
        """Open object from file (signal/image)"""
        obj_or_objlist = self.IO_REGISTRY.read(filename)
        if not isinstance(obj_or_objlist, list):
            obj_or_objlist = [obj_or_objlist]
        for obj in obj_or_objlist:
            self.add_object(obj)

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

    def open_objects(self, filenames: List[str] = None) -> None:
        """Open objects from file (signals/images)"""
        if not self.mainwindow.confirm_memory_state():
            return
        if filenames is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            filters = self.IO_REGISTRY.get_filters(IOAction.LOAD)
            with save_restore_stds():
                filenames, _filt = getopenfilenames(self, _("Open"), basedir, filters)
        for filename in filenames:
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                Conf.main.base_dir.set(filename)
                self.open_object(filename)

    def save_objects(self, filenames: List[str] = None) -> None:
        """Save selected objects to file (signal/image)"""
        rows = self.objlist.get_selected_rows()
        if filenames is None:  # pragma: no cover
            filenames = [None] * len(rows)
        assert len(filenames) == len(rows)
        for index, row in enumerate(rows):
            filename = filenames[index]
            obj = self.objlist[row]
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
                row = self.objlist.get_selected_rows()[0]
                obj = self.objlist[row]
                obj.import_metadata_from_file(filename)
            self.SIG_UPDATE_PLOT_ITEMS.emit()

    def export_metadata_from_file(self, filename: str = None) -> None:
        """Export metadata to file (JSON)"""
        row = self.objlist.get_selected_rows()[0]
        obj = self.objlist[row]
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
        row = self.objlist.currentRow()
        sel_objs = self.objlist.get_sel_objects()
        if not sel_objs:
            row = -1
        self.objprop.update_properties_from(self.objlist[row] if row != -1 else None)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        self.acthandler.selection_rows_changed(sel_objs)

    def properties_changed(self) -> None:
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.objlist.currentRow()
        update_dataset(self.objlist[row], self.objprop.properties.dataset)
        self.objlist.refresh_list()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    # ------Plotting data in modal dialogs----------------------------------------------
    def open_separate_view(self, rows: Optional[List[int]] = None) -> QW.QDialog:
        """
        Open separate view for visualizing selected objects

        :param list rows: List of row indexes for the objects to be shown in dialog
        :return: Dialog instance
        """
        title = _("Annotations")
        if rows is None:
            rows = self.objlist.get_selected_rows()
        row = rows[0]
        obj = self.objlist[row]
        dlg = self.create_new_dialog(rows, edit=True, name="new_window")
        width, height = self.DIALOGSIZE
        dlg.resize(width, height)
        dlg.plot_widget.itemlist.setVisible(True)
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
        for obj in self.objlist:
            obj.metadata[obj.METADATA_LBL] = state
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def create_new_dialog(
        self,
        rows,
        edit=False,
        toolbar=True,
        title=None,
        tools=None,
        name=None,
        options=None,
    ):
        """
        Create new pop-up signal/image plot dialog

        :param list rows: List of row indexes for the objects to be shown in dialog
        :param bool edit: If True, show "OK" and "Cancel" buttons
        :param bool toolbar: If True, add toolbar
        :param str title: Title of the dialog box
        :param list tools: List of plot tools
        :param str name: Name of the widget (used as screenshot basename)
        :param dict options: Plot options
        """
        if title is not None or len(rows) == 1:
            if title is None:
                title = self.objlist.get_sel_object().title
            title = f"{title} - {APP_NAME}"
        else:
            title = APP_NAME
        plot_options = self.itmlist.get_current_plot_options()
        if options is not None:
            plot_options.update(options)
        dlg = self.DIALOGCLASS(
            parent=self,
            wintitle=title,
            edit=edit,
            options=plot_options,
            toolbar=toolbar,
        )
        dlg.setWindowIcon(get_icon("CobraDataLab.svg"))
        dlg.setObjectName(f"{self.PREFIX}_{name}")
        if tools is not None:
            for tool in tools:
                dlg.add_tool(tool)
        plot = dlg.get_plot()
        for row in rows:
            obj = self.objlist[row]
            item = obj.make_item(update_from=self.itmlist[row])
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
        row = self.objlist.get_selected_rows()[0]
        obj = self.objlist[row]
        dlg = self.create_new_dialog(
            [row],
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
        self, parent: QW.QWidget, title: str
    ) -> objectlist.GetObjectDialog:
        """Get object dialog"""
        dlg = objectlist.GetObjectDialog(parent, self, title)
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
        rows = self.objlist.get_selected_rows()

        @dataclasses.dataclass
        class ResultData:
            """Result data associated to a shapetype"""

            results: List[ResultShape] = None
            xlabels: List[str] = None
            ylabels: List[str] = None

        rdatadict: Dict[ShapeTypes, ResultData] = {}
        for idx, row in enumerate(rows):
            obj = self.objlist[row]
            for result in obj.iterate_resultshapes():
                rdata = rdatadict.setdefault(result.shapetype, ResultData([], None, []))
                title = f"{result.label}"
                rdata.results.append(result)
                rdata.xlabels = result.xlabels
                for _i_row_res in range(result.array.shape[0]):
                    ylabel = f"{self.PREFIX}{idx:03d}: {result.label}"
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
                    dlg.setObjectName(f"{self.PREFIX}_results")
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
