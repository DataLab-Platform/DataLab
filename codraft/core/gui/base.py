# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT Base GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc
import enum
import re
from typing import Callable, List

import guidata.dataset.qtwidgets as gdq
import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guidata.utils import update_dataset
from guiqwt.builder import make
from guiqwt.plot import CurveDialog
from guiqwt.styles import style_generator
from guiqwt.tools import HCursorTool, LabelTool, VCursorTool, XCursorTool
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getopenfilenames, getsavefilename

from codraft.config import APP_NAME, Conf, _
from codraft.core.model.base import MetadataItem
from codraft.core.model.signal import SignalParam
from codraft.utils.qthelpers import (
    exec_dialog,
    qt_try_loadsave_file,
    save_restore_stds,
)

#  Registering MetadataItem edit widget
gdq.DataSetEditLayout.register(MetadataItem, gdq.ButtonWidget)


class ActionCategory(enum.Enum):
    """Action categories"""

    FILE = enum.auto()
    EDIT = enum.auto()
    VIEW = enum.auto()
    OPERATION = enum.auto()
    PROCESSING = enum.auto()
    COMPUTING = enum.auto()


class BaseActionHandler(metaclass=abc.ABCMeta):
    """Object handling panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = ""  # e.g. "signal"

    def __init__(self, panel, objlist, itmlist, processor, toolbar):
        self.panel = panel
        self.objlist = objlist
        self.itmlist = itmlist
        self.processor = processor
        self.feature_actions = {}
        self.operation_end_actions = None
        # Object selection dependent actions
        self.actlist_1more = []
        self.actlist_2more = []
        self.actlist_1 = []
        self.actlist_2 = []
        if self.__class__ is not BaseActionHandler:
            self.create_all_actions(toolbar)

    def selection_rows_changed(self):
        """Number of selected rows has changed"""
        nbrows = len(self.objlist.get_selected_rows())
        for act in self.actlist_1more:
            act.setEnabled(nbrows >= 1)
        for act in self.actlist_2more:
            act.setEnabled(nbrows >= 2)
        for act in self.actlist_1:
            act.setEnabled(nbrows == 1)
        for act in self.actlist_2:
            act.setEnabled(nbrows == 2)

    def create_all_actions(self, toolbar):
        """Setup actions, menus, toolbar"""
        featact = self.feature_actions
        featact[ActionCategory.FILE] = file_act = self.create_file_actions()
        featact[ActionCategory.EDIT] = edit_act = self.create_edit_actions()
        featact[ActionCategory.VIEW] = view_act = self.create_view_actions()
        featact[ActionCategory.OPERATION] = self.create_operation_actions()
        featact[ActionCategory.PROCESSING] = self.create_processing_actions()
        featact[ActionCategory.COMPUTING] = self.create_computing_actions()
        add_actions(toolbar, file_act + [None] + edit_act + [None] + view_act)

    def cra(
        self, title, triggered=None, toggled=None, shortcut=None, icon=None, tip=None
    ):
        """Create action convenience method"""
        return create_action(self.panel, title, triggered, toggled, shortcut, icon, tip)

    def create_file_actions(self):
        """Create file actions"""
        new_act = self.cra(
            _("New %s...") % self.OBJECT_STR,
            icon=get_icon(f"new_{self.OBJECT_STR}.svg"),
            tip=_("Create new %s") % self.OBJECT_STR,
            triggered=self.panel.new_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.New),
        )
        open_act = self.cra(
            _("Open %s...") % self.OBJECT_STR,
            icon=get_icon("libre-gui-import.svg"),
            tip=_("Open %s") % self.OBJECT_STR,
            triggered=self.panel.open_objects,
            shortcut=QG.QKeySequence(QG.QKeySequence.Open),
        )
        save_act = self.cra(
            _("Save %s...") % self.OBJECT_STR,
            icon=get_icon("libre-gui-export.svg"),
            tip=_("Save selected %s") % self.OBJECT_STR,
            triggered=self.panel.save_objects,
            shortcut=QG.QKeySequence(QG.QKeySequence.Save),
        )
        importmd_act = self.cra(
            _("Import metadata into %s...") % self.OBJECT_STR,
            icon=get_icon("metadata_import.svg"),
            tip=_("Import metadata into %s") % self.OBJECT_STR,
            triggered=self.panel.import_metadata_from_file,
        )
        exportmd_act = self.cra(
            _("Export metadata from %s...") % self.OBJECT_STR,
            icon=get_icon("metadata_export.svg"),
            tip=_("Export selected %s metadata") % self.OBJECT_STR,
            triggered=self.panel.export_metadata_from_file,
        )
        self.actlist_1more += [save_act]
        self.actlist_1 += [importmd_act, exportmd_act]
        return [new_act, open_act, save_act, None, importmd_act, exportmd_act]

    def create_edit_actions(self):
        """Create edit actions"""
        dup_action = self.cra(
            _("Duplicate"),
            icon=get_icon("libre-gui-copy.svg"),
            triggered=self.panel.duplicate_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.Copy),
        )
        cpymeta_action = self.cra(
            _("Copy metadata"),
            icon=get_icon("metadata_copy.svg"),
            triggered=self.panel.copy_metadata,
        )
        pstmeta_action = self.cra(
            _("Paste metadata"),
            icon=get_icon("metadata_paste.svg"),
            triggered=self.panel.paste_metadata,
        )
        cleanup_action = self.cra(
            _("Clean up data view"),
            icon=get_icon("libre-tools-vacuum-cleaner.svg"),
            tip=_("Clean up data view before updating plotting panels"),
            toggled=self.itmlist.toggle_cleanup_dataview,
        )
        cleanup_action.setChecked(True)
        delm_action = self.cra(
            _("Delete object metadata"),
            icon=get_icon("metadata_delete.svg"),
            tip=_("Delete all that is contained in object metadata"),
            triggered=self.panel.delete_metadata,
        )
        delall_action = self.cra(
            _("Delete all"),
            shortcut="Shift+Ctrl+Suppr",
            triggered=self.panel.delete_all_objects,
        )
        del_action = self.cra(
            _("Remove"),
            icon=get_icon("libre-gui-trash.svg"),
            triggered=self.panel.remove_object,
            shortcut=QG.QKeySequence(QG.QKeySequence.Delete),
        )
        self.actlist_1more += [dup_action, del_action, delm_action, pstmeta_action]
        self.actlist_1 += [cpymeta_action]
        return [
            dup_action,
            del_action,
            delall_action,
            None,
            cpymeta_action,
            pstmeta_action,
            delm_action,
        ]

    def create_view_actions(self):
        """Create view actions"""
        view_action = self.cra(
            _("View in a new window"),
            icon=get_icon("libre-gui-binoculars.svg"),
            triggered=self.panel.open_separate_view,
        )
        showlabel_action = self.cra(
            _("Graphical object titles"),
            icon=get_icon("show_titles.svg"),
            tip=_("Show or hide ROI and other graphical object titles or subtitles"),
            toggled=self.panel.toggle_show_titles,
        )
        showlabel_action.setChecked(True)
        self.actlist_1more += [view_action]
        return [view_action, showlabel_action]

    def create_operation_actions(self):
        """Create operation actions"""
        proc = self.processor
        sum_action = self.cra(_("Sum"), proc.compute_sum)
        average_action = self.cra(_("Average"), proc.compute_average)
        diff_action = self.cra(_("Difference"), proc.compute_difference)
        prod_action = self.cra(_("Product"), proc.compute_product)
        div_action = self.cra(_("Division"), proc.compute_division)
        roi_action = self.cra(
            _("ROI extraction"),
            proc.extract_roi,
            icon=get_icon(f"{self.OBJECT_STR}_roi.svg"),
        )
        swapaxes_action = self.cra(_("Swap X/Y axes"), proc.swap_axes)
        abs_action = self.cra(_("Absolute value"), proc.compute_abs)
        log_action = self.cra("Log10(y)", proc.compute_log10)
        self.actlist_1more += [roi_action, swapaxes_action, abs_action, log_action]
        self.actlist_2more += [sum_action, average_action, prod_action]
        self.actlist_2 += [diff_action, div_action]
        self.operation_end_actions = [roi_action, swapaxes_action]
        return [
            sum_action,
            average_action,
            diff_action,
            prod_action,
            div_action,
            None,
            abs_action,
            log_action,
        ]

    def create_processing_actions(self):
        """Create processing actions"""
        proc = self.processor
        threshold_action = self.cra(_("Thresholding"), proc.compute_threshold)
        clip_action = self.cra(_("Clipping"), proc.compute_clip)
        lincal_action = self.cra(_("Linear calibration"), proc.calibrate)
        gauss_action = self.cra(_("Gaussian filter"), proc.compute_gaussian)
        movavg_action = self.cra(_("Moving average"), proc.compute_moving_average)
        movmed_action = self.cra(_("Moving median"), proc.compute_moving_median)
        wiener_action = self.cra(_("Wiener filter"), proc.compute_wiener)
        fft_action = self.cra(_("FFT"), proc.compute_fft)
        ifft_action = self.cra(_("Inverse FFT"), proc.compute_ifft)
        for act in (fft_action, ifft_action):
            act.setToolTip(_("Warning: only real part is plotted"))
        actions = [
            threshold_action,
            clip_action,
            lincal_action,
            gauss_action,
            movavg_action,
            movmed_action,
            wiener_action,
            fft_action,
            ifft_action,
        ]
        self.actlist_1more += actions
        return actions

    @abc.abstractmethod
    def create_computing_actions(self):
        """Create computing actions"""
        proc = self.processor
        defineroi_action = self.cra(
            _("Regions of interest..."),
            triggered=proc.edit_regions_of_interest,
            icon=get_icon("roi.svg"),
        )
        stats_action = self.cra(
            _("Statistics") + "...",
            triggered=proc.compute_stats,
            icon=get_icon("stats.svg"),
        )
        self.actlist_1 += [defineroi_action, stats_action]
        return [defineroi_action, None, stats_action]


class SimpleObjectList(QW.QListWidget):
    """Base object handling panel list widget, object (sig/ima) lists"""

    SIG_ITEM_DOUBLECLICKED = QC.Signal(int)

    def __init__(self, panel, parent=None):
        parent = panel if parent is None else parent
        super().__init__(parent)
        self.panel = panel
        self.prefix = panel.PREFIX
        self.setAlternatingRowColors(True)
        self._objects = []  # signals or images
        self.itemDoubleClicked.connect(self.item_double_clicked)

    def init_from(self, objlist):
        """Init from another SimpleObjectList, without making copies of objects"""
        self._objects = objlist.get_objects()
        self.refresh_list()
        self.setCurrentRow(objlist.currentRow())

    def get_objects(self):
        """Get all objects"""
        return self._objects

    def set_current_row(self, row, extend=False):
        """Set list widget current row"""
        if row < 0:
            row += self.count()
        if extend:
            command = QC.QItemSelectionModel.Select
        else:
            command = QC.QItemSelectionModel.ClearAndSelect
        self.setCurrentRow(row, command)

    def refresh_list(self, new_current_row=None):
        """
        Refresh object list

        :param new_current_row: New row (if None, new current row is unchanged)
        """
        row = self.currentRow()
        if new_current_row is not None:
            row = new_current_row
        self.clear()
        self.addItems(
            [
                f"{self.prefix}{idx:03d}: {obj.title}"
                for idx, obj in enumerate(self._objects)
            ]
        )
        if row < self.count():
            self.set_current_row(row)

    def item_double_clicked(self, listwidgetitem):
        """Item was double-clicked: open a pop-up plot dialog"""
        self.SIG_ITEM_DOUBLECLICKED.emit(self.row(listwidgetitem))


class GetObjectDialog(QW.QDialog):
    """Get object dialog box"""

    def __init__(self, parent, panel, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QW.QVBoxLayout())

        self.objlist = SimpleObjectList(panel, parent=parent)
        self.objlist.init_from(panel.objlist)
        self.objlist.SIG_ITEM_DOUBLECLICKED.connect(lambda row: self.accept())
        self.layout().addWidget(self.objlist)

        bbox = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        bbox.button(QW.QDialogButtonBox.Ok).setEnabled(self.objlist.count() > 0)
        self.layout().addSpacing(10)
        self.layout().addWidget(bbox)

    def get_object(self):
        """Return current object"""
        return self.objlist.get_objects()[self.objlist.currentRow()]


class ObjectList(SimpleObjectList):
    """Object handling panel list widget, object (sig/ima) lists"""

    def __init__(self, panel):
        super().__init__(panel)
        self.setSelectionMode(QW.QListWidget.ExtendedSelection)

    def __len__(self):
        """Return number of objects"""
        return len(self._objects)

    def __getitem__(self, row):
        """Return object at row"""
        return self._objects[row]

    def __setitem__(self, row, obj):
        """Set object at row"""
        self._objects[row] = obj

    def __fix_obj_titles(self, row: int, sign: int) -> None:
        """Fix all object titles before adding (sign==1) or removing (sign==-1)
        an object at row index"""
        pfx = self.prefix
        oname = f"{pfx}%03d"
        for obj in self:
            for match in re.finditer(pfx + "[0-9]{3}", obj.title):
                before = match.group()
                i_match = int(before[1:])
                if sign == -1 and i_match == row:
                    after = f"{pfx}xxx"
                elif (sign == -1 and i_match > row) or (sign == 1 and i_match >= row):
                    after = oname % (i_match + sign)
                else:
                    continue
                obj.title = obj.title.replace(before, after)

    def __delitem__(self, row):
        """Del object at row"""
        self.__fix_obj_titles(row, -1)
        self._objects.pop(row)

    def __iter__(self):
        """Return an iterator over objects"""
        yield from self._objects

    def get_sel_object(self, position=0):
        """
        Return currently selected object

        :param int position: Position in selection list (0 means first, -1 means last)
        :return: Current object or None if there is no selection
        """
        rows = self.get_selected_rows()
        if rows:
            return self[rows[position]]
        return None

    def get_sel_objects(self):
        """Return selected objects"""
        return [self[row] for row in self.get_selected_rows()]

    def append(self, obj):
        """Append object"""
        self._objects.append(obj)

    def insert(self, row, obj):
        """Insert object at row index"""
        self.__fix_obj_titles(row, 1)
        self._objects.insert(row, obj)

    def remove_all(self):
        """Remove all objects"""
        self._objects = []

    def select_rows(self, rows):
        """Select multiple list widget rows"""
        for index, row in enumerate(sorted(rows)):
            self.set_current_row(row, extend=index != 0)

    def select_all_rows(self):
        """Select all widget rows"""
        self.selectAll()

    def get_selected_rows(self):
        """Return selected rows"""
        return [index.row() for index in self.selectionModel().selectedRows()]


class BaseItemList:
    """Object handling plot items associated to objects (signals/images)"""

    def __init__(self, panel, objlist, plotwidget):
        self._enable_cleanup_dataview = True
        self.panel = panel
        self.objlist = objlist
        self.plotwidget = plotwidget
        self.plot = plotwidget.get_plot()
        self.__plotitems = []  # plot items associated to objects (sig/ima)
        self.__shapeitems = []

    def __len__(self):
        """Return number of items"""
        return len(self.__plotitems)

    def __getitem__(self, row):
        """Return item at row"""
        return self.__plotitems[row]

    def __setitem__(self, row, item):
        """Set item at row"""
        self.__plotitems[row] = item

    def __delitem__(self, row):
        """Del item at row"""
        item = self.__plotitems.pop(row)
        self.plot.del_item(item)

    def __iter__(self):
        """Return an iterator over items"""
        yield from self.__plotitems

    def append(self, item):
        """Append item"""
        self.__plotitems.append(item)

    def insert(self, row):
        """Insert object at row index"""
        self.__plotitems.insert(row, None)

    def add_item_to_plot(self, row):
        """Add plot item to plot"""
        item = self.objlist[row].make_item()
        item.set_readonly(True)
        if row < len(self):
            self[row] = item
        else:
            self.append(item)
        self.plot.add_item(item)
        return item

    def make_item_from_existing(self, row):
        """Make plot item from existing object/item at row"""
        return self.objlist[row].make_item(update_from=self[row])

    def update_item(self, row):
        """Update plot item associated to data"""
        self.objlist[row].update_item(self[row])

    def add_shapes(self, row):
        """Add geometric shape items associated to computed results and annotations"""
        obj = self.objlist[row]
        if obj.metadata:
            for item in obj.iterate_shape_items(editable=False):
                self.plot.add_item(item)
                self.__shapeitems.append(item)

    def remove_all(self):
        """Remove all plot items"""
        self.__plotitems = []
        self.plot.del_all_items()

    def remove_all_shape_items(self):
        """Remove all geometric shapes associated to result items"""
        if set(self.__shapeitems).issubset(set(self.plot.items)):
            self.plot.del_items(self.__shapeitems)
        self.__shapeitems = []

    def refresh_plot(self):
        """Refresh plot"""
        rows = self.objlist.get_selected_rows()
        self.remove_all_shape_items()
        if self._enable_cleanup_dataview and len(rows) == 1:
            self.cleanup_dataview()
        for item in self:
            if item is not None:
                item.hide()
        title_keys = ("title", "xlabel", "ylabel", "zlabel", "xunit", "yunit", "zunit")
        titles_dict = {}
        if rows:
            for i_row, row in enumerate(rows):
                for key in title_keys:
                    title = getattr(self.objlist[row], key, "")
                    value = titles_dict.get(key)
                    if value is None:
                        titles_dict[key] = title
                    elif value != title:
                        titles_dict[key] = ""
                item = self[row]
                if item is None:
                    item = self.add_item_to_plot(row)
                else:
                    if i_row == 0:
                        make.style = style_generator()
                    self.update_item(row)
                self.plot.set_item_visible(item, True, replot=False)
                self.plot.set_active_item(item)
                item.unselect()
                self.add_shapes(row)
            self.plot.replot()
        else:
            for key in title_keys:
                titles_dict[key] = ""
        tdict = titles_dict
        tdict["ylabel"] = (tdict["ylabel"], tdict.pop("zlabel"))
        tdict["yunit"] = (tdict["yunit"], tdict.pop("zunit"))
        self.plot.set_titles(**titles_dict)
        self.plot.do_autoscale()

    def toggle_cleanup_dataview(self, state):
        """Toggle clean up data view option"""
        self._enable_cleanup_dataview = state

    def cleanup_dataview(self):
        """Clean up data view"""
        for item in self.plot.items[:]:
            if item not in self:
                self.plot.del_item(item)

    def get_current_plot_options(self):
        """
        Return standard signal/image plot options

        :return: Dictionary containing plot arguments for CurveDialog/ImageDialog
        """
        return dict(
            xlabel=self.plot.get_axis_title("bottom"),
            ylabel=self.plot.get_axis_title("left"),
            xunit=self.plot.get_axis_unit("bottom"),
            yunit=self.plot.get_axis_unit("left"),
        )


class ObjectProp(QW.QWidget):
    """Object handling panel properties"""

    def __init__(self, panel, paramclass):
        super().__init__(panel)
        self.properties = gdq.DataSetEditGroupBox(_("Properties"), paramclass)
        self.properties.SIG_APPLY_BUTTON_CLICKED.connect(panel.properties_changed)
        self.properties.setEnabled(False)
        hlayout = QW.QHBoxLayout()
        hlayout.addWidget(self.properties)
        vlayout = QW.QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addStretch()
        self.setLayout(vlayout)


class BaseROIEditorMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseROIEditor(QW.QWidget, metaclass=BaseROIEditorMeta):
    """ROI Editor"""

    ICON_NAME = None

    def __init__(self, parent: QW.QDialog, roi_items: list, func: Callable):
        super().__init__(parent)
        self.plot = parent.get_plot()
        self.plot.SIG_ITEMS_CHANGED.connect(lambda _plot: self.update_roi_titles())
        self.plot.SIG_ITEM_REMOVED.connect(self.item_removed)
        self.roi_items = roi_items
        self.update_roi_titles()
        for roi_item in roi_items:
            self.plot.add_item(roi_item)
            self.plot.set_active_item(roi_item)
        self.new_roi_func = func
        self.add_btn = QW.QPushButton(
            get_icon(self.ICON_NAME), _("Add region of interest"), self
        )
        self.add_btn.clicked.connect(self.add_roi)
        layout = QW.QHBoxLayout()
        layout.addWidget(self.add_btn)
        layout.addStretch()
        self.setLayout(layout)

    def add_roi(self):
        """Add ROI"""
        self.plot.unselect_all()
        roi_item = self.new_roi_func()
        self.roi_items.append(roi_item)
        self.update_roi_titles()
        self.plot.add_item(roi_item)
        self.plot.set_active_item(roi_item)

    @abc.abstractmethod
    def update_roi_titles(self):
        """Update ROI annotation titles"""

    def item_removed(self, item):
        """Item was removed. Since all items are read-only except ROIs...
        this must be an ROI."""
        assert item in self.roi_items
        self.roi_items.remove(item)
        self.update_roi_titles()

    @staticmethod
    @abc.abstractmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""

    def get_roi_coords(self) -> list:
        """Return list of ROI plot coordinates"""
        coords = []
        for roi_item in self.roi_items:
            coords.append(list(self.get_roi_item_coords(roi_item)))
        return coords


class BasePanelMeta(type(QW.QSplitter), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BasePanel(QW.QSplitter, metaclass=BasePanelMeta):
    """Object handling the item list, the selected item properties and plot"""

    PANEL_STR = ""  # e.g. "Signal Panel"
    PARAMCLASS = SignalParam  # Replaced by the right class in child object
    DIALOGCLASS = CurveDialog  # Idem
    PLOT_TOOLS = (LabelTool, VCursorTool, HCursorTool, XCursorTool)
    PREFIX = ""  # e.g. "s"
    OPEN_FILTERS = ""  # Qt file open dialog filters
    H5_PREFIX = ""
    SIG_STATUS_MESSAGE = QC.Signal(str)  # emitted by "qt_try_except" decorator
    SIG_OBJECT_ADDED = QC.Signal()
    SIG_OBJECT_REMOVED = QC.Signal()
    SIG_REFRESH_PLOT = QC.Signal()
    ROIDIALOGOPTIONS = {}
    ROIDIALOGCLASS = BaseROIEditor

    @abc.abstractmethod
    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(QC.Qt.Vertical, parent)
        self.setObjectName(self.PREFIX)
        self.mainwindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objlist = ObjectList(self)
        self.itmlist = None
        self.processor = None
        self.acthandler = None
        self.__metadata_clipboard = {}

    def setup_panel(self):
        """Setup panel"""
        self.processor.SIG_ADD_SHAPE.connect(self.itmlist.add_shapes)
        self.SIG_REFRESH_PLOT.connect(self.itmlist.refresh_plot)
        self.objlist.itemSelectionChanged.connect(self.selection_changed)
        self.objlist.currentRowChanged.connect(self.current_item_changed)
        self.objlist.SIG_ITEM_DOUBLECLICKED.connect(
            lambda row: self.open_separate_view([row])
        )
        self.objprop.properties.SIG_APPLY_BUTTON_CLICKED.connect(
            self.properties_changed
        )
        self.addWidget(self.objlist)
        self.addWidget(self.objprop)

    def get_category_actions(self, category):
        """Return actions for category"""
        return self.acthandler.feature_actions[category]

    # ------Creating, adding, removing objects------------------------------------------
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

    def add_object(self, obj, refresh=True):
        """Add signal/image object"""
        self.objlist.append(obj)
        row = len(self.objlist) - 1
        item = self.itmlist.add_item_to_plot(row)
        if refresh:
            self.objlist.refresh_list(-1)
        self.SIG_OBJECT_ADDED.emit()
        return item

    # TODO: [P2] New feature: move objects up/down
    def insert_object(self, obj, row, refresh=True):
        """Insert signal/image object after row"""
        self.objlist.insert(row, obj)
        self.itmlist.insert(row)
        if refresh:
            self.objlist.refresh_list(new_current_row=row + 1)
        self.SIG_OBJECT_ADDED.emit()

    def duplicate_object(self):
        """Duplication signal/image object"""
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        row = None
        for row in rows:
            obj = self.objlist[row]
            objcopy = self.create_object()
            objcopy.title = obj.title
            objcopy.copy_data_from(obj)
            self.insert_object(objcopy, row=row + 1, refresh=False)
        self.objlist.refresh_list(new_current_row=row + 1)
        self.SIG_REFRESH_PLOT.emit()

    def copy_metadata(self):
        """Copy object metadata"""
        obj = self.objlist.get_sel_object()
        self.__metadata_clipboard = obj.metadata

    def paste_metadata(self):
        """Paste metadata to selected object(s)"""
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        row = None
        for row in rows:
            obj = self.objlist[row]
            obj.metadata.update(self.__metadata_clipboard)
        self.SIG_REFRESH_PLOT.emit()

    def remove_object(self):
        """Remove signal/image object"""
        rows = sorted(self.objlist.get_selected_rows(), reverse=True)
        for row in rows:
            del self.objlist[row]
            del self.itmlist[row]
        self.objlist.refresh_list(0)
        self.SIG_REFRESH_PLOT.emit()
        self.SIG_OBJECT_REMOVED.emit()

    def delete_all_objects(self):
        """Confirm before removing all objects"""
        if len(self.objlist) == 0:
            return
        answer = QW.QMessageBox.warning(
            self,
            _("Delete all"),
            _("Do you want to delete all objects from the %s?") % self.PANEL_STR,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
        )
        if answer == QW.QMessageBox.Yes:
            self.remove_all_objects()

    def remove_all_objects(self):
        """Remove all signal/image objects"""
        self.objlist.remove_all()
        self.itmlist.remove_all()
        self.objlist.refresh_list(0)
        self.SIG_REFRESH_PLOT.emit()
        self.SIG_OBJECT_REMOVED.emit()

    def delete_metadata(self):
        """Delete object metadata"""
        for row in self.objlist.get_selected_rows():
            self.objlist[row].metadata = {}
        self.SIG_REFRESH_PLOT.emit()

    @abc.abstractmethod
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new object (signal/image).

        :param guidata.dataset.DataSet newparam: new object parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """

    def open_objects(self, filenames: List[str] = None) -> None:
        """Open objects from file (signals/images)"""
        if not self.mainwindow.confirm_memory_state():
            return
        if filenames is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filenames, _filter = getopenfilenames(
                    self, _("Open"), basedir, self.OPEN_FILTERS
                )
        for filename in filenames:
            Conf.main.base_dir.set(filename)
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                self.open_object(filename)

    def save_objects(self, filenames: List[str] = None) -> None:
        """Save selected objects to file (signal/image)"""
        rows = self.objlist.get_selected_rows()
        if filenames is None:
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
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getopenfilename(
                    self, _("Import metadata"), basedir, "*.json"
                )
        if filename:
            Conf.main.base_dir.set(filename)
            row = self.objlist.get_selected_rows()[0]
            obj = self.objlist[row]
            with qt_try_loadsave_file(self.parent(), filename, "load"):
                obj.import_metadata_from_file(filename)
            self.SIG_REFRESH_PLOT.emit()

    def export_metadata_from_file(self, filename: str = None):
        """Export metadata to file (JSON)"""
        row = self.objlist.get_selected_rows()[0]
        obj = self.objlist[row]
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Export metadata"), basedir, "*.json"
                )
        if filename:
            Conf.main.base_dir.set(filename)
            with qt_try_loadsave_file(self.parent(), filename, "save"):
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
    def current_item_changed(self, row):
        """Current item changed"""
        if row != -1:
            update_dataset(self.objprop.properties.dataset, self.objlist[row])
            self.objprop.properties.get()

    def selection_changed(self):
        """Signal list: selection changed"""
        row = self.objlist.currentRow()
        self.objprop.properties.setDisabled(row == -1)
        self.SIG_REFRESH_PLOT.emit()
        self.acthandler.selection_rows_changed()

    def properties_changed(self):
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.objlist.currentRow()
        update_dataset(self.objlist[row], self.objprop.properties.dataset)
        self.objlist.refresh_list()
        self.SIG_REFRESH_PLOT.emit()

    # ------Plotting data in modal dialogs----------------------------------------------
    def open_separate_view(self, rows=None):
        """
        Open separate view for visualizing selected objects

        :param list rows: List of row indexes for the objects to be shown in dialog
        """
        if rows is None:
            rows = self.objlist.get_selected_rows()
        dlg = self.create_new_dialog(rows, tools=self.PLOT_TOOLS, name="new_window")
        dlg.plot_widget.itemlist.setVisible(True)
        exec_dialog(dlg)

    def toggle_show_titles(self, state):
        """Toggle show annotations option"""
        Conf.view.show_label.set(state)
        for obj in self.objlist:
            obj.metadata[obj.METADATA_LBL] = state
        self.SIG_REFRESH_PLOT.emit()

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

    def get_roi_dialog(self) -> np.ndarray:
        """Get ROI data (array) from specific dialog box"""
        roi_s = _("Regions of interest")
        dlg, obj = self.create_new_dialog_for_selection(
            roi_s, "roi_dialog", self.ROIDIALOGOPTIONS
        )
        fmt = obj.metadata.get(obj.METADATA_FMT, "%s")
        roi_items = list(obj.iterate_roi_items(fmt, True))
        plot = dlg.get_plot()
        plot.unselect_all()
        for item in plot.items:
            item.set_selectable(False)
        roi_editor = self.ROIDIALOGCLASS(
            dlg, roi_items, lambda: obj.new_roi_item(fmt, True, editable=True)
        )
        dlg.plot_layout.addWidget(roi_editor, 1, 0, 1, 1)
        if exec_dialog(dlg):
            coords = roi_editor.get_roi_coords()
            return obj.roi_coords_to_indexes(coords)
        return None

    def get_object_dialog(self, parent: QW.QWidget, title: str) -> GetObjectDialog:
        """Get object dialog"""
        dlg = GetObjectDialog(parent, self, title)
        if exec_dialog(dlg):
            return dlg.get_object()
        return None
