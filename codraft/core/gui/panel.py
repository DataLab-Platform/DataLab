# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Panel widgets (core.gui.panel)

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
import os.path as osp
import re
import warnings
from typing import TYPE_CHECKING, Iterator, List

import guidata.dataset.qtwidgets as gdq
import numpy as np
from guidata.config import CONF
from guidata.configtools import get_font, get_icon
from guidata.qthelpers import add_actions, create_action, is_dark_mode
from guidata.qtwidgets import DockableWidgetMixin
from guidata.utils import update_dataset
from guidata.widgets.arrayeditor import ArrayEditor
from guidata.widgets.console.shell import PythonShellWidget
from guiqwt.io import imread, imwrite, iohandler
from guiqwt.plot import CurveDialog, ImageDialog
from guiqwt.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
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

from codraft.config import APP_NAME, Conf, _
from codraft.core.gui import actionhandler, objectlist, plotitemlist, roieditor
from codraft.core.gui.macroeditor import Macro
from codraft.core.gui.processor.image import ImageProcessor
from codraft.core.gui.processor.signal import SignalProcessor
from codraft.core.io.signal import read_signal, write_signal
from codraft.core.model.base import MetadataItem, ResultShape
from codraft.core.model.image import (
    ImageDatatypes,
    ImageParam,
    create_image,
    create_image_from_param,
    new_image_param,
)
from codraft.core.model.signal import (
    SignalParam,
    create_signal_from_param,
    new_signal_param,
)
from codraft.utils.qthelpers import (
    exec_dialog,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)

if TYPE_CHECKING:
    from codraft.core.gui import ObjItf
    from codraft.core.model.base import ObjectItf

#  Registering MetadataItem edit widget
gdq.DataSetEditLayout.register(MetadataItem, gdq.ButtonWidget)


class ObjectProp(QW.QWidget):
    """Object handling panel properties"""

    def __init__(self, panel, paramclass):
        super().__init__(panel)
        self.paramclass = paramclass
        self.properties = gdq.DataSetEditGroupBox(_("Properties"), paramclass)
        self.properties.SIG_APPLY_BUTTON_CLICKED.connect(panel.properties_changed)
        self.properties.setEnabled(False)
        self.add_prop_layout = QW.QHBoxLayout()
        playout = self.properties.edit.layout
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
    """Object defining CodraFT panel interface, based on a vertical QSplitter widget

    A panel handle an object list (objects are signals, images, macros, ...).
    Each object must implement ``codraft.core.gui.ObjItf`` interface
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

    def serialize_to_hdf5(self, writer):
        """Serialize objects to a HDF5 file"""
        with writer.group(self.H5_PREFIX):
            for idx, obj in enumerate(self.object_iterator()):
                title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
                name = f"{self.PREFIX}{idx:03d}: {title}"
                with writer.group(name):
                    obj.serialize(writer)

    def deserialize_from_hdf5(self, reader):
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
    PARAMCLASS = SignalParam  # Replaced by the right class in child object
    DIALOGCLASS = CurveDialog  # Idem
    ANNOTATION_TOOLS = (
        LabelTool,
        VCursorTool,
        HCursorTool,
        XCursorTool,
        SegmentTool,
        RectangleTool,
    )
    DIALOGSIZE = (800, 600)
    OPEN_FILTERS = ""  # Qt file open dialog filters
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_UPDATE_PLOT_ITEM = QC.Signal(int)  # Update plot item associated to row number
    SIG_UPDATE_PLOT_ITEMS = QC.Signal()  # Update plot items associated to selected rows
    ROIDIALOGOPTIONS = {}
    ROIDIALOGCLASS = roieditor.BaseROIEditor  # Replaced in child object

    @abc.abstractmethod
    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent)
        self.mainwindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objlist = objectlist.ObjectList(self)
        self.objlist.SIG_IMPORT_FILES.connect(self.handle_dropped_files)
        self.itmlist = None
        self.processor = None
        self.acthandler: actionhandler.BaseActionHandler = None
        self.__metadata_clipboard = {}
        self.context_menu = QW.QMenu()
        self.__separate_views = {}

    # ------AbstractPanel interface-----------------------------------------------------
    @property
    def object_number(self):
        """Return object number"""
        return len(self.objlist)

    def object_iterator(self) -> Iterator[ObjItf]:
        """Iterate over objects handled by panel"""
        return iter(self.objlist)

    def remove_all_objects(self):
        """Remove all objects"""
        for dlg in self.__separate_views:
            dlg.done(QW.QDialog.DialogCode.Rejected)
        self.objlist.remove_all()
        self.itmlist.remove_all()
        self.objlist.refresh_list(0)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        super().remove_all_objects()

    def create_object(self, title=None):
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
    def add_object(self, obj, refresh=True) -> ObjItf:
        """Add object

        :param bool refresh: Refresh object list (e.g. listwidget for signals/images)"""
        obj.check_data()
        self.objlist.append(obj)
        self.itmlist.append(None)
        if refresh:
            self.objlist.refresh_list(-1)
        return super().add_object(obj, refresh=refresh)

    # ---- Signal/Image Panel API ------------------------------------------------------
    def setup_panel(self):
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

    def get_category_actions(self, category) -> List[QW.QAction]:  # pragma: no cover
        """Return actions for category"""
        return self.acthandler.feature_actions.get(category, [])

    def __popup_contextmenu(self, position: QC.QPoint):  # pragma: no cover
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
    def insert_object(self, obj, row, refresh=True):
        """Insert signal/image object after row"""
        obj.check_data()
        self.objlist.insert(row, obj)
        self.itmlist.insert(row)
        if refresh:
            self.objlist.refresh_list(new_current_row=row + 1)
        self.SIG_OBJECT_ADDED.emit()

    def duplicate_object(self):
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

    def copy_metadata(self):
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

    def paste_metadata(self):
        """Paste metadata to selected object(s)"""
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        row = None
        for row in rows:
            obj = self.objlist[row]
            obj.metadata.update(self.__metadata_clipboard)
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def remove_object(self):
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

    def delete_all_objects(self):  # pragma: no cover
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

    def delete_metadata(self):
        """Delete object metadata"""
        for index, row in enumerate(self.objlist.get_selected_rows()):
            self.objlist[row].reset_metadata_to_defaults()
            if index == 0:
                self.selection_changed()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    def copy_titles_to_clipboard(self):
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

    @abc.abstractmethod
    def open_object(self, filename: str) -> None:
        """Open object from file (signal/image)"""

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
            with save_restore_stds():
                filenames, _filter = getopenfilenames(
                    self, _("Open"), basedir, self.OPEN_FILTERS
                )
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

    @abc.abstractmethod
    def save_object(self, obj, filename: str = None) -> None:
        """Save object to file (signal/image)"""

    def import_metadata_from_file(self, filename: str = None):
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

    def export_metadata_from_file(self, filename: str = None):
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

    # ------Serializing/deserializing objects-------------------------------------------
    def serialize_to_hdf5(self, writer):
        """Serialize objects to a HDF5 file"""
        with writer.group(self.H5_PREFIX):
            for idx, obj in enumerate(self.objlist):
                title = re.sub("[^-a-zA-Z0-9_.() ]+", "", obj.title.replace("/", "_"))
                name = f"{self.PREFIX}{idx:03d}: {title}"
                with writer.group(name):
                    obj.serialize(writer)

    def deserialize_from_hdf5(self, reader):
        """Deserialize objects from a HDF5 file"""
        with reader.group(self.H5_PREFIX):
            for name in reader.h5.get(self.H5_PREFIX, []):
                obj = self.PARAMCLASS()
                with reader.group(name):
                    obj.deserialize(reader)
                    self.add_object(obj)
                    QW.QApplication.processEvents()

    # ------Refreshing GUI--------------------------------------------------------------
    def selection_changed(self):
        """Signal list: selection changed"""
        row = self.objlist.currentRow()
        sel_objs = self.objlist.get_sel_objects()
        if not sel_objs:
            row = -1
        self.objprop.update_properties_from(self.objlist[row] if row != -1 else None)
        self.SIG_UPDATE_PLOT_ITEMS.emit()
        self.acthandler.selection_rows_changed(sel_objs)

    def properties_changed(self):
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.objlist.currentRow()
        update_dataset(self.objlist[row], self.objprop.properties.dataset)
        self.objlist.refresh_list()
        self.SIG_UPDATE_PLOT_ITEMS.emit()

    # ------Plotting data in modal dialogs----------------------------------------------
    def open_separate_view(self, rows=None) -> QW.QDialog:
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

    def __separate_view_finished(self, result: int):
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

    def toggle_show_titles(self, state):
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
        dlg.setWindowIcon(get_icon("codraft.svg"))
        dlg.setObjectName(f"{self.PREFIX}_{name}")
        if tools is not None:
            for tool in tools:
                dlg.add_tool(tool)
        plot = dlg.get_plot()
        for row in rows:
            item = self.itmlist.make_item_from_existing(row)
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

    def add_results_button(self):
        """Add 'Show results' button"""
        btn = QW.QPushButton(get_icon("show_results.svg"), _("Show results"), self)
        btn.setToolTip(_("Show results obtained from previous computations"))
        self.objprop.add_button(btn)
        btn.clicked.connect(self.show_results)
        self.acthandler.add_action(
            btn,
            select_condition=actionhandler.SelectCond.at_least_one,
        )

    def show_results(self):
        """Show results"""
        rows = self.objlist.get_selected_rows()

        @dataclasses.dataclass
        class ResultData:
            """Result data associated to a shapetype"""

            results: List[ResultShape] = None
            xlabels: List[str] = None
            ylabels: List[str] = None

        rdatadict = {}
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


class SignalPanel(BaseDataPanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Signal objects"""

    PANEL_STR = _("Signal List")
    PARAMCLASS = SignalParam
    DIALOGCLASS = CurveDialog
    PREFIX = "s"
    OPEN_FILTERS = f'{_("Text files")} (*.txt *.csv)\n{_("NumPy arrays")} (*.npy)'
    H5_PREFIX = "CodraFT_Sig"
    ROIDIALOGCLASS = roieditor.SignalROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = plotitemlist.SignalItemList(self, self.objlist, plotwidget)
        self.processor = SignalProcessor(self, self.objlist, plotwidget)
        self.acthandler = actionhandler.SignalActionHandler(
            self, self.itmlist, self.processor, toolbar
        )

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new signal.

        :param codraft.core.model.signal.SignalNewParam newparam: new signal parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """
        if not self.mainwindow.confirm_memory_state():
            return
        curobj = self.objlist.get_sel_object(-1)
        if curobj is not None:
            newparam = newparam if newparam is not None else new_signal_param()
            newparam.size = len(curobj.data)
            newparam.xmin = curobj.x.min()
            newparam.xmax = curobj.x.max()
        signal = create_signal_from_param(
            newparam, addparam=addparam, edit=edit, parent=self
        )
        if signal is not None:
            self.add_object(signal)

    def open_object(self, filename: str) -> None:
        """Open object from file (signal/image)"""
        signal = read_signal(filename)
        self.add_object(signal)

    def save_object(self, obj, filename: str = None) -> None:
        """Save object to file (signal/image)"""
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getsavefilename(  # pylint: disable=duplicate-code
                    self, _("Save as"), basedir, self.OPEN_FILTERS
                )
        if filename:
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                write_signal(obj, filename)


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
    OPEN_FILTERS = iohandler.get_filters("load", dtype=None)
    H5_PREFIX = "CodraFT_Ima"
    ROIDIALOGOPTIONS = dict(show_itemlist=True, show_contrast=False)
    ROIDIALOGCLASS = roieditor.ImageROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = plotitemlist.ImageItemList(self, self.objlist, plotwidget)
        self.processor = ImageProcessor(self, self.objlist, plotwidget)
        self.acthandler = actionhandler.ImageActionHandler(
            self, self.itmlist, self.processor, toolbar
        )

    # ------Refreshing GUI--------------------------------------------------------------
    def properties_changed(self):
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.objlist.currentRow()
        self.objlist[row].invalidate_maskdata_cache()
        super().properties_changed()

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
        reducepath = osp.relpath(filename, osp.join(osp.dirname(filename), osp.pardir))
        if filename.lower().endswith(".sif") and len(data.shape) == 3:
            for idx in range(data.shape[0]):
                image = create_image(reducepath + "_Im" + str(idx), data[idx, ::])
                self.add_object(image)
        else:
            if data.ndim == 3:
                # Converting to grayscale
                data = data[..., :4].mean(axis=2)
            image = create_image(reducepath, data)
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
            kwargs = {}
            if osp.splitext(filename)[1].lower() == ".dcm":
                kwargs["template"] = obj.dicom_template
            with qt_try_loadsave_file(self.parent(), filename, "save"):
                Conf.main.base_dir.set(filename)
                imwrite(filename, obj.data, **kwargs)

    def toggle_show_contrast(self, state):
        """Toggle show contrast option"""
        Conf.view.show_contrast.set(state)
        self.SIG_UPDATE_PLOT_ITEMS.emit()


class MacroTabs(QW.QTabWidget):
    """Macro tabwidget"""

    SIG_CONTEXT_MENU = QC.Signal(QC.QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)

    def contextMenuEvent(self, event):  # pylint: disable=C0103
        """Override Qt method"""
        self.SIG_CONTEXT_MENU.emit(event.globalPos())


class MacroPanel(AbstractPanel, DockableWidgetMixin):
    """Macro manager widget"""

    LOCATION = QC.Qt.LeftDockWidgetArea

    PREFIX = "m"
    H5_PREFIX = "CodraFT_Mac"

    SIG_OBJECT_MODIFIED = QC.Signal()

    def __init__(self, parent: QW.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle(_("Macro manager"))
        self.setWindowIcon(get_icon("libre-gui-cogs.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.context_menu = QW.QMenu()

        self.console = PythonShellWidget(self)
        self.console.set_light_background(not is_dark_mode())
        self.console.setMaximumBlockCount(5000)
        font = get_font(CONF, "console")
        font.setPointSize(10)
        self.console.set_font(font)
        self.console.write(_("-***- Macro Console -***-"), prompt=True)

        self.tabwidget = MacroTabs(self)
        self.tabwidget.tabBarDoubleClicked.connect(self.rename_macro)
        self.tabwidget.tabCloseRequested.connect(self.remove_macro)
        self.tabwidget.currentChanged.connect(self.current_macro_changed)

        for widget in (self.tabwidget, self.console):
            self.addWidget(widget)

        self.run_action = None
        self.stop_action = None
        self.obj_actions = []  # Object-dependent actions
        self.__macros = {}

        # TODO: Add action "Import from file..."
        # TODO: Add action "Export to file..."

        self.setup_actions()

    # ------AbstractPanel interface-----------------------------------------------------
    @property
    def object_number(self):
        """Return object number"""
        return len(self.__macros)

    def object_iterator(self) -> Iterator[ObjItf]:
        """Iterate over objects handled by panel"""
        return iter(self.__macros.values())

    def remove_all_objects(self):
        """Remove all objects"""
        while self.tabwidget.count() > 0:
            self.tabwidget.removeTab(0)
        super().remove_all_objects()

    def create_object(self, title=None):
        """Create object

        :param str title: Title of the object
        """
        macro = Macro(self.console, title)
        macro.objectNameChanged.connect(self.macro_name_changed)
        macro.STARTED.connect(
            lambda orig_macro=macro: self.macro_state_changed(orig_macro, True)
        )
        macro.FINISHED.connect(
            lambda orig_macro=macro: self.macro_state_changed(orig_macro, False)
        )
        macro.MODIFIED.connect(self.macro_contents_changed)
        return macro

    def add_object(self, obj: Macro, refresh: bool = True) -> ObjItf:
        """Add object

        :param bool refresh: Refresh object list (e.g. listwidget for signals/images)"""
        self.tabwidget.addTab(obj.editor, obj.title)
        self.__macros[id(obj)] = obj
        return super().add_object(obj, refresh=refresh)

    # ---- Macro panel API -------------------------------------------------------------
    def setup_actions(self):
        """Setup macro menu actions"""
        self.run_action = create_action(
            self,
            _("Run macro"),
            icon=get_icon("libre-camera-flash-on.svg"),
            triggered=self.run_macro,
            shortcut="Ctrl+F5",
        )
        self.stop_action = create_action(
            self,
            _("Stop macro"),
            icon=get_icon("libre-camera-flash-off.svg"),
            triggered=self.stop_macro,
            shortcut="Shift+F5",
        )
        self.stop_action.setDisabled(True)
        add_act = create_action(
            self,
            _("New macro"),
            icon=get_icon("libre-gui-add.svg"),
            triggered=self.add_macro,
        )
        ren_act = create_action(
            self,
            _("Rename macro"),
            icon=get_icon("libre-gui-pencil.svg"),
            triggered=self.rename_macro,
        )
        exp_act = create_action(
            self,
            _("Export macro to file"),
            icon=get_icon("libre-gui-export.svg"),
            triggered=self.export_macro_to_file,
        )
        imp_act = create_action(
            self,
            _("Import macro from file"),
            icon=get_icon("libre-gui-import.svg"),
            triggered=self.import_macro_from_file,
        )
        rem_act = create_action(
            self,
            _("Remove macro"),
            icon=get_icon("libre-gui-action-delete.svg"),
            triggered=self.remove_macro,
        )
        actions = (
            self.run_action,
            self.stop_action,
            None,
            add_act,
            ren_act,
            exp_act,
            imp_act,
            None,
            rem_act,
        )
        self.obj_actions += [
            self.run_action,
            self.stop_action,
            ren_act,
            exp_act,
            rem_act,
        ]

        self.tabwidget.SIG_CONTEXT_MENU.connect(self.__popup_contextmenu)

        toolbar = QW.QToolBar(_("Macro editor toolbar"), self)
        menu_button = QW.QPushButton(get_icon("libre-gui-menu.svg"), "", self)
        menu_button.setFlat(True)
        menu_button.setMenu(self.context_menu)
        toolbar.addWidget(menu_button)
        self.tabwidget.setCornerWidget(toolbar)

        add_actions(toolbar, [self.run_action, self.stop_action, None])
        add_actions(self.context_menu, actions)

    def __popup_contextmenu(self, position: QC.QPoint):  # pragma: no cover
        """Popup context menu at position"""
        not_empty = self.tabwidget.count() > 0
        for action in self.obj_actions:
            action.setEnabled(not_empty)
        if not_empty:
            self.current_macro_changed()
        self.context_menu.popup(position)

    def get_macro(self, index: int = None) -> Macro:
        """Return macro at index (if index is None, return current macro)"""
        if index is None:
            index = self.tabwidget.currentIndex()
        for macro in self.__macros.values():
            if self.tabwidget.widget(index) is macro.editor:
                return macro
        return None

    def current_macro_changed(self, index: int = None):  # pylint: disable=W0613
        """Current macro has changed"""
        macro = self.get_macro()
        if macro is not None:
            state = macro.is_running()
            self.macro_state_changed(macro, state)

    def macro_contents_changed(self):
        """One of the macro contents has changed"""
        self.SIG_OBJECT_MODIFIED.emit()

    def run_macro(self):
        """Run current macro"""
        # XXX: Macros should be executed in a separate process: access to CodraFT
        # is provided by the 'remote_controlling' feature (see corresponding branch).
        # So, macros should be Python scripts similar to "remoteclient_test.py".
        # Connection to the XML-RPC server should be simplified (to a single line).
        #
        # Important: an environment variable must contained current CodraFT session
        # XML-RPC server port number (e.g. "CODRAFT_XMLRPC_PORT"). This allows to
        # handle the case of multiple instances of CodraFT running at the same time
        # (each macro will then control the right instance of CodraFT, the one from
        # which it was executed)
        macro = self.get_macro()
        macro.run()

    def stop_macro(self):
        """Stop current macro"""
        macro = self.get_macro()
        macro.kill()

    def macro_state_changed(self, orig_macro: Macro, state: bool):
        """Macro state has changed (True: started, False: stopped)"""
        macro = self.get_macro()
        if macro is orig_macro:
            self.run_action.setEnabled(not state)
            self.stop_action.setEnabled(state)

    def add_macro(self, name: str = None, rename: bool = True) -> Macro:
        """Add macro, optionally with name"""
        macro = self.create_object(name)
        self.add_object(macro)
        if rename:
            self.rename_macro()
        self.current_macro_changed()
        return macro

    def macro_name_changed(self, name):
        """Macro name has been changed"""
        index = self.indexOf(self.tabwidget)
        self.tabwidget.setTabText(index, name)

    def rename_macro(self, index: int = None):
        """Rename macro"""
        macro = self.get_macro(index)
        name, valid = QW.QInputDialog.getText(
            self,
            _("Rename"),
            _("New title:"),
            QW.QLineEdit.Normal,
            macro.objectName(),
        )
        if valid:
            macro.setObjectName(name)
            if index is not None:
                self.tabwidget.setCurrentIndex(index)

    def export_macro_to_file(self):
        """Export macro to file"""
        raise NotImplementedError

    def import_macro_from_file(self):
        """Import macro from file"""
        raise NotImplementedError

    def remove_macro(self, index: int = None):
        """Remove macro"""
        if index is None:
            index = self.tabwidget.currentIndex()
        txt = "<br>".join(
            [
                _(
                    "When closed, the macro is <u>permanently destroyed</u>, "
                    "unless it has been exported first."
                ),
                "",
                _("Do you want to continue?"),
            ]
        )
        btns = QW.QMessageBox.StandardButton.Yes | QW.QMessageBox.StandardButton.No
        choice = QW.QMessageBox.warning(self, self.windowTitle(), txt, btns)
        if choice == QW.QMessageBox.StandardButton.Yes:
            self.tabwidget.removeTab(index)
            self.SIG_OBJECT_REMOVED.emit()
