# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Base panel objects (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import glob
import os
import os.path as osp
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Generic, Literal, Type

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
import h5py
import numpy as np
import plotpy.io
from guidata.configtools import get_icon
from guidata.dataset import restore_dataset, update_dataset
from guidata.qthelpers import add_actions, create_action, exec_dialog
from plotpy.plot import PlotDialog
from plotpy.tools import ActionTool
from qtpy import QtCore as QC  # type: ignore[import]
from qtpy import QtWidgets as QW
from qtpy.compat import (
    getexistingdirectory,
    getopenfilename,
    getopenfilenames,
    getsavefilename,
)
from sigima.io import read_metadata, read_roi, write_metadata, write_roi
from sigima.io.base import get_file_extensions
from sigima.io.common.basename import format_basenames
from sigima.objects import (
    ImageObj,
    NewImageParam,
    SignalObj,
    TypeObj,
    TypeROI,
    create_image_from_param,
    create_signal,
    create_signal_from_param,
)
from sigima.objects.base import ROI_KEY
from sigima.params import SaveToDirectoryParam

from datalab import objectmodel
from datalab.adapters_metadata import (
    GeometryAdapter,
    ResultData,
    TableAdapter,
    create_resultdata_dict,
    show_resultdata,
)
from datalab.adapters_plotpy import create_adapter_from_object, items_to_json
from datalab.config import APP_NAME, Conf, _
from datalab.env import execenv
from datalab.gui import actionhandler, objectview
from datalab.gui.newobject import (
    CREATION_PARAMETERS_OPTION,
    NewSignalParam,
    extract_creation_parameters,
    insert_creation_parameters,
)
from datalab.gui.processor.base import (
    PROCESSING_PARAMETERS_OPTION,
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.gui.roieditor import TypeROIEditor
from datalab.objectmodel import ObjectGroup, get_short_id, get_uuid, set_uuid
from datalab.utils.qthelpers import (
    CallbackWorker,
    create_progress_bar,
    qt_long_callback,
    qt_try_except,
    qt_try_loadsave_file,
    save_restore_stds,
)
from datalab.widgets.textimport import TextImportWizard

if TYPE_CHECKING:
    from typing import Callable

    from plotpy.items import CurveItem, LabelItem, MaskedXYImageItem
    from sigima.io.image import ImageIORegistry
    from sigima.io.signal import SignalIORegistry

    from datalab.gui import ObjItf
    from datalab.gui.main import DLMainWindow
    from datalab.gui.plothandler import ImagePlotHandler, SignalPlotHandler
    from datalab.gui.processor.image import ImageProcessor
    from datalab.gui.processor.signal import SignalProcessor
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


# Metadata keys that should not be pasted when copying metadata between objects
METADATA_PASTE_EXCLUSIONS = {
    ROI_KEY,  # ROI has dedicated copy/paste operations
    "__uuid",  # Each object must have a unique identifier
    f"__{PROCESSING_PARAMETERS_OPTION}",  # Object-specific processing history
    f"__{CREATION_PARAMETERS_OPTION}",  # Object-specific creation parameters
}


def is_plot_item_serializable(item: Any) -> bool:
    """Return True if plot item is serializable"""
    try:
        plotpy.io.item_class_from_name(item.__class__.__name__)
        return True
    except AssertionError:
        return False


def is_hdf5_file(filename: str, check_content: bool = False) -> bool:
    """Return True if filename has an HDF5 extension or is an HDF5 file.

    Args:
        filename: Path to the file to check
        check_content: If True, also attempts to open the file to verify it's a
                      valid HDF5 file. If False, only checks the file extension.

    Returns:
        True if the file is (likely) an HDF5 file, False otherwise.
    """
    # First, check by extension (fast)
    has_hdf5_extension = filename.lower().endswith((".h5", ".hdf5", ".hdf", ".he5"))

    if not check_content:
        return has_hdf5_extension

    # If checking content, try to open as HDF5 file
    if has_hdf5_extension:
        return True  # Trust common HDF5 extensions

    # For other extensions, attempt to open the file to verify it's HDF5
    try:
        with h5py.File(filename, "r"):
            return True
    except (OSError, IOError, ValueError):
        # Not a valid HDF5 file
        return False


@dataclass
class ProcessingReport:
    """Report of processing operation

    Args:
        success: True if processing succeeded
        obj_uuid: UUID of the processed object
        message: Optional message (error or info)
    """

    success: bool
    obj_uuid: str | None = None
    message: str | None = None


class ObjectProp(QW.QTabWidget):
    """Object handling panel properties

    Args:
        panel: parent data panel
        objclass: class of the object handled by the panel (SignalObj or ImageObj)
    """

    def __init__(self, panel: BaseDataPanel, objclass: SignalObj | ImageObj) -> None:
        super().__init__(panel)
        self.setTabBarAutoHide(True)
        self.setTabPosition(QW.QTabWidget.West)

        self.panel = panel
        self.objclass = objclass

        # Object creation tab
        self.creation_param_editor: gdq.DataSetEditGroupBox | None = None
        self.current_creation_obj: SignalObj | ImageObj | None = None
        self.creation_scroll: QW.QScrollArea | None = None

        # Object processing tab
        self.processing_param_editor: gdq.DataSetEditGroupBox | None = None
        self.current_processing_obj: SignalObj | ImageObj | None = None
        self.processing_scroll: QW.QScrollArea | None = None

        # Properties tab
        self.properties = gdq.DataSetEditGroupBox("", objclass)
        self.properties.SIG_APPLY_BUTTON_CLICKED.connect(panel.properties_changed)
        self.properties.setEnabled(False)
        self.__original_values: dict[str, Any] = {}

        self.add_prop_layout = QW.QHBoxLayout()
        playout: QW.QGridLayout = self.properties.edit.layout
        playout.addLayout(
            self.add_prop_layout, playout.rowCount() - 1, 0, 1, 1, QC.Qt.AlignLeft
        )

        # Create Analysis and History widgets
        font = Conf.proc.small_mono_font.get_font()

        self.processing_history = QW.QTextEdit()
        self.processing_history.setReadOnly(True)
        self.processing_history.setFont(font)

        self.analysis_parameter = QW.QTextEdit()
        self.analysis_parameter.setReadOnly(True)
        self.analysis_parameter.setFont(font)

        self.addTab(self.processing_history, _("History"))
        self.addTab(self.analysis_parameter, _("Analysis"))
        self.addTab(self.properties, _("Properties"))

        self.processing_history.textChanged.connect(self._update_tab_visibility)
        self.analysis_parameter.textChanged.connect(self._update_tab_visibility)

    def _update_tab_visibility(self) -> None:
        """Update visibility of a tab based on its content."""
        # Save current tab to restore it after visibility changes
        current_index = self.currentIndex()
        current_widget = self.widget(current_index)

        for textedit in (self.processing_history, self.analysis_parameter):
            tab_index = self.indexOf(textedit)
            if tab_index >= 0:
                has_content = bool(textedit.toPlainText().strip())
                self.setTabVisible(tab_index, has_content)

        # Restore the previously selected tab if it's still visible
        # But only if we're not making History or Analysis visible
        # (they shouldn't steal focus)
        if current_widget is not None:
            # Don't restore if current widget was History or Analysis
            # that just became visible
            if current_widget not in (
                self.processing_history,
                self.analysis_parameter,
            ):
                restored_index = self.indexOf(current_widget)
                if restored_index >= 0 and self.isTabVisible(restored_index):
                    self.setCurrentIndex(restored_index)
            else:
                # Current widget was History or Analysis - select Properties instead
                properties_index = self.indexOf(self.properties)
                if properties_index >= 0:
                    self.setCurrentIndex(properties_index)

    def add_button(self, button: QW.QPushButton) -> None:
        """Add additional button on bottom of properties panel"""
        self.add_prop_layout.addWidget(button)

    def find_object_by_uuid(
        self, uuid: str
    ) -> SignalObj | ImageObj | ObjectGroup | None:
        """Find an object by UUID, searching across all panels if needed.

        This method first searches in the current panel, then in other panels
        for cross-panel computations (e.g., radial profile: ImageObj -> SignalObj).

        Args:
            uuid: UUID of the object to find

        Returns:
            The object if found, None otherwise
        """
        other_panel = self.panel.mainwindow.imagepanel
        if self.panel is other_panel:
            other_panel = self.panel.mainwindow.signalpanel
        for panel in (self.panel, other_panel):
            if panel is not None:
                try:
                    return panel.objmodel[uuid]
                except KeyError:
                    continue
        return None

    def display_analysis_parameter(self, obj: SignalObj | ImageObj) -> None:
        """Set analysis parameter label.

        Args:
            obj: Signal or Image object
        """
        text = ""
        for key, value in obj.metadata.items():
            if key.endswith("__param_html") and isinstance(value, str):
                if text:
                    text += "<br><br>"
                text += value
        self.analysis_parameter.setText(text)

    def _build_processing_history(self, obj: SignalObj | ImageObj) -> str:
        """Build processing history as a simple text list.

        Args:
            obj: Signal or Image object

        Returns:
            Processing history as text
        """
        history_items = []
        current_obj = obj
        max_depth = 20  # Prevent infinite loops

        # Walk backwards through processing chain, collecting items
        while current_obj is not None and len(history_items) < max_depth:
            proc_params = extract_processing_parameters(current_obj)

            if proc_params is None:
                # Check for creation parameters
                creation_params = extract_creation_parameters(current_obj)
                if creation_params is not None:
                    text = f"{_('Created')}: {creation_params.title}"
                    history_items.append(text)
                else:
                    history_items.append(_("Original object"))
                break

            # Add current processing step
            func_name = proc_params.func_name.replace("_", " ").title()
            history_items.append(func_name)

            # Try to find source object
            if proc_params.source_uuid:
                current_obj = self.find_object_by_uuid(proc_params.source_uuid)
                if current_obj is None:
                    history_items.append(_("(source deleted)"))
                    break
            elif proc_params.source_uuids:
                # Multiple sources (n-to-1 or 2-to-1 pattern)
                history_items.append(_("(multiple sources)"))
                break
            else:
                break

        if len(history_items) <= 1:
            return ""  # Shows the history tab only when there is some history

        # Reverse to show from oldest to newest, then add indentation
        history_items.reverse()
        history_lines = []
        for i, item in enumerate(history_items):
            indent = "  " * i
            history_lines.append(f"{indent}└─ {item}")

        return "\n".join(history_lines)

    def display_processing_history(self, obj: SignalObj | ImageObj) -> None:
        """Display processing history.

        Args:
            obj: Signal or Image object
        """
        history_text = self._build_processing_history(obj)
        self.processing_history.setText(history_text)

    def update_properties_from(self, obj: SignalObj | ImageObj | None = None) -> None:
        """Update properties from signal/image dataset

        Args:
            obj: Signal or Image object
        """
        self.properties.setDisabled(obj is None)
        if obj is None:
            obj = self.objclass()
        dataset: SignalObj | ImageObj = self.properties.dataset
        dataset.set_defaults()
        update_dataset(dataset, obj)
        self.properties.get()
        self.display_analysis_parameter(obj)
        self.display_processing_history(obj)
        self.properties.apply_button.setEnabled(False)

        # Store original values to detect which properties have changed
        # Using restore_dataset to convert the dataset to a dictionary
        self.__original_values = {}
        restore_dataset(dataset, self.__original_values)

        # Remove only Creation and Processing tabs (dynamic tabs)
        # Use widget references instead of text labels for reliable identification
        if self.creation_scroll is not None:
            index = self.indexOf(self.creation_scroll)
            if index >= 0:
                self.removeTab(index)
        if self.processing_scroll is not None:
            index = self.indexOf(self.processing_scroll)
            if index >= 0:
                self.removeTab(index)

        # Reset references for dynamic tabs
        self.creation_param_editor = None
        self.current_creation_obj = None
        self.creation_scroll = None
        self.processing_param_editor = None
        self.current_processing_obj = None
        self.processing_scroll = None

        # Setup Creation and Processing tabs (if applicable)
        if obj is not None:
            self.setup_creation_tab(obj)
            self.setup_processing_tab(obj)

        # Trigger visibility update for History and Analysis tabs
        # (will be called via textChanged signals, but we call explicitly
        # here to ensure initial state is correct)
        self._update_tab_visibility()

    def get_changed_properties(self) -> dict[str, Any]:
        """Get dictionary of properties that have changed from original values.

        Returns:
            Dictionary mapping property names to their new values, containing only
            the properties that were modified by the user.
        """
        dataset = self.properties.dataset
        changed = {}

        # Get current values as a dictionary
        current_values = {}
        restore_dataset(dataset, current_values)

        # Compare with original values
        for key, current_value in current_values.items():
            original_value = self.__original_values.get(key)
            # Check if value has changed
            if not self._values_equal(current_value, original_value):
                changed[key] = current_value
        return changed

    def update_original_values(self) -> None:
        """Update the stored original values to the current dataset values.

        This should be called after applying changes to reset the baseline
        for detecting future changes.
        """
        dataset = self.properties.dataset
        self.__original_values = {}
        restore_dataset(dataset, self.__original_values)

    @staticmethod
    def _values_equal(val1: Any, val2: Any) -> bool:
        """Compare two values, handling special cases like numpy arrays.

        Args:
            val1: first value
            val2: second value

        Returns:
            True if values are equal
        """
        # Handle numpy arrays
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not isinstance(val1, np.ndarray) or not isinstance(val2, np.ndarray):
                return False
            return np.array_equal(val1, val2)
        # Handle regular comparison
        return val1 == val2

    def setup_creation_tab(self, obj: SignalObj | ImageObj) -> bool:
        """Setup the Creation tab with parameter editor for interactive object creation.

        Args:
            obj: Signal or Image object

        Returns:
            True if Creation tab was set up, False otherwise
        """
        param = extract_creation_parameters(obj)
        if param is None:
            return False

        # Create parameter editor widget using the actual parameter class
        # (which is a subclass of NewSignalParam or NewImageParam)
        editor = gdq.DataSetEditGroupBox(_("Creation Parameters"), param.__class__)
        update_dataset(editor.dataset, param)
        editor.get()

        # Connect Apply button to recreation handler
        editor.SIG_APPLY_BUTTON_CLICKED.connect(self.apply_creation_parameters)
        editor.set_apply_button_state(False)

        # Store reference to be able to retrieve it later
        self.creation_param_editor = editor
        self.current_creation_obj = obj

        # Set the parameter editor as the scroll area widget
        # Creation tab is always at index 0 (before all other tabs)
        self.creation_scroll = QW.QScrollArea()
        self.creation_scroll.setWidgetResizable(True)
        self.creation_scroll.setWidget(editor)
        self.insertTab(0, self.creation_scroll, _("Creation"))
        self.setCurrentIndex(0)
        return True

    def apply_creation_parameters(self) -> None:
        """Apply creation parameters: recreate object with updated parameters."""
        editor = self.creation_param_editor
        if editor is None or self.current_creation_obj is None:
            return
        if isinstance(self.current_creation_obj, SignalObj):
            otext = _("Signal was modified in-place.")
        else:
            otext = _("Image was modified in-place.")
        text = f"⚠️ {otext} ⚠️ "
        text += _(
            "If computation were performed based on this object, "
            "they may need to be redone."
        )
        self.panel.SIG_STATUS_MESSAGE.emit(text, 20000)

        # Recreate object with new parameters
        # (serialization is done automatically in create_signal/image_from_param)
        param = editor.dataset
        try:
            if isinstance(self.current_creation_obj, SignalObj):
                new_obj = create_signal_from_param(param)
            else:  # ImageObj
                new_obj = create_image_from_param(param)
        except Exception as exc:  # pylint: disable=broad-except
            QW.QMessageBox.warning(
                self,
                _("Error"),
                _("Failed to recreate object with new parameters:\n%s") % str(exc),
            )
            return

        # Update the current object in-place
        obj_uuid = get_uuid(self.current_creation_obj)
        self.current_creation_obj.title = new_obj.title
        if isinstance(self.current_creation_obj, SignalObj):
            self.current_creation_obj.xydata = new_obj.xydata
        else:  # ImageObj
            self.current_creation_obj.data = new_obj.data
        # Update metadata with new creation parameters
        insert_creation_parameters(self.current_creation_obj, param)

        # Update the tree view item (to show new title if it changed)
        self.panel.objview.update_item(obj_uuid)

        # Refresh only the plot, not the entire panel
        # (avoid calling selection_changed which would trigger a full refresh
        # of the Properties tab and could cause recursion issues)
        self.panel.refresh_plot(obj_uuid, update_items=True, force=True)

        # Refresh the Creation tab with the new parameters
        # Use QTimer to defer this until after the current event is processed
        QC.QTimer.singleShot(
            0, lambda: self.setup_creation_tab(self.current_creation_obj)
        )

    def setup_processing_tab(
        self, obj: SignalObj | ImageObj, reset_params: bool = True
    ) -> bool:
        """Setup the Processing tab with parameter editor for re-processing.

        Args:
            obj: Signal or Image object
            reset_params: If True, call update_from_obj() to reset parameters from
                source object. If False, use parameters as stored in metadata.

        Returns:
            True if Processing tab was set up, False otherwise
        """
        # Extract processing parameters
        proc_params = extract_processing_parameters(obj)
        if proc_params is None:
            return False

        # Check if the pattern type is 1-to-1 (only interactive pattern)
        if proc_params.pattern != "1-to-1":
            return False

        # Store reference to be able to retrieve it later
        self.current_processing_obj = obj

        # Check if object has processing parameter
        param = proc_params.param
        if param is None:
            return False

        # Skip interactive processing for list of parameters
        # (e.g., ROI extraction, erase operations)
        if isinstance(param, list):
            return False

        # Eventually call the `update_from_obj` method to properly initialize
        # the parameter object from the current object state.
        # Only do this when reset_params is True (initial setup), not when
        # refreshing after user has modified parameters.
        if reset_params and hasattr(param, "update_from_obj"):
            # Warning: the `update_from_obj` method takes the input object as argument,
            # not the output object (`obj` is the processed object here):
            # Retrieve the input object from the source UUID
            if proc_params.source_uuid is not None:
                source_obj = self.find_object_by_uuid(proc_params.source_uuid)
                if source_obj is not None:
                    param.update_from_obj(source_obj)

        # Create parameter editor widget
        editor = gdq.DataSetEditGroupBox(_("Processing Parameters"), param.__class__)
        update_dataset(editor.dataset, param)
        editor.get()

        # Connect Apply button to reprocessing handler
        editor.SIG_APPLY_BUTTON_CLICKED.connect(self.apply_processing_parameters)
        editor.set_apply_button_state(False)

        # Store reference to be able to retrieve it later
        self.processing_param_editor = editor

        # Remove existing Processing tab if it exists
        if self.processing_scroll is not None:
            index = self.indexOf(self.processing_scroll)
            if index >= 0:
                self.removeTab(index)

        # Processing tab comes after Creation tab (if it exists)
        # Find the correct insertion index: after Creation (index 0) if it exists,
        # otherwise at index 0
        has_creation = (
            self.creation_scroll is not None and self.indexOf(self.creation_scroll) >= 0
        )
        insert_index = 1 if has_creation else 0

        # Create new processing scroll area and tab
        self.processing_scroll = QW.QScrollArea()
        self.processing_scroll.setWidgetResizable(True)
        self.processing_scroll.setWidget(editor)
        self.insertTab(insert_index, self.processing_scroll, _("Processing"))
        self.setCurrentIndex(insert_index)
        return True

    def apply_processing_parameters(
        self, obj: SignalObj | ImageObj | None = None, interactive: bool = True
    ) -> ProcessingReport:
        """Apply processing parameters: re-run processing with updated parameters.

        Args:
            obj: Signal or Image object to reprocess. If None, uses the current object.
            interactive: If True, show progress and error messages in the UI.

        Returns:
            ProcessingReport with success status, object UUID, and optional message.
        """
        report = ProcessingReport(success=False)
        editor = self.processing_param_editor
        obj = obj or self.current_processing_obj
        if obj is None:
            report.message = _("No processing object available.")
            return report

        report.obj_uuid = get_uuid(obj)

        # Extract processing parameters
        proc_params = extract_processing_parameters(obj)
        if proc_params is None:
            report.message = _("Processing metadata is incomplete.")
            if interactive:
                QW.QMessageBox.critical(self, _("Error"), report.message)
            return report

        # Check if source object still exists
        if proc_params.source_uuid is None:
            report.message = _(
                "Processing metadata is incomplete (missing source UUID)."
            )
            if interactive:
                QW.QMessageBox.critical(self, _("Error"), report.message)
            return report

        # Find source object
        source_obj = self.find_object_by_uuid(proc_params.source_uuid)
        if source_obj is None:
            report.message = _("Source object no longer exists.")
            if interactive:
                QW.QMessageBox.critical(
                    self,
                    _("Error"),
                    report.message
                    + "\n\n"
                    + _(
                        "The object that was used to create this processed object "
                        "has been deleted and cannot be used for reprocessing."
                    ),
                )
            return report

        # Get updated parameters from editor
        param = editor.dataset if editor is not None else proc_params.param

        # For cross-panel computations, we need to use the processor from the panel
        # that owns the source object (e.g., radial_profile is in ImageProcessor)
        processor = self.panel.processor
        if isinstance(source_obj, SignalObj):
            processor = self.panel.mainwindow.signalpanel.processor
        elif isinstance(source_obj, ImageObj):
            processor = self.panel.mainwindow.imagepanel.processor

        # Recompute using the dedicated method (with multiprocessing support)
        try:
            new_obj = processor.recompute_1_to_1(
                proc_params.func_name, source_obj, param
            )
        except Exception as exc:  # pylint: disable=broad-except
            report.message = _("Failed to reprocess object:\n%s") % str(exc)
            if interactive:
                QW.QMessageBox.warning(self, _("Error"), report.message)
            return report

        if new_obj is None:
            # User cancelled the operation
            report.message = _("Processing was cancelled.")

        else:
            report.success = True

            # Update the current object in-place with data from new object
            obj.title = new_obj.title
            if isinstance(obj, SignalObj):
                obj.xydata = new_obj.xydata
            else:  # ImageObj
                obj.data = new_obj.data

            # Update metadata with new processing parameters
            updated_proc_params = ProcessingParameters(
                func_name=proc_params.func_name,
                pattern=proc_params.pattern,
                param=param,
                source_uuid=proc_params.source_uuid,
            )
            insert_processing_parameters(obj, updated_proc_params)

            # Update the tree view item and refresh plot
            obj_uuid = get_uuid(obj)
            self.panel.objview.update_item(obj_uuid)
            self.panel.refresh_plot(obj_uuid, update_items=True, force=True)

            # Refresh the Processing tab with the new parameters
            # Don't reset parameters from source object - keep the user's values
            QC.QTimer.singleShot(
                0, lambda: self.setup_processing_tab(obj, reset_params=False)
            )

            if isinstance(obj, SignalObj):
                report.message = _("Signal was reprocessed.")
            else:
                report.message = _("Image was reprocessed.")
            self.panel.SIG_STATUS_MESSAGE.emit("✅ " + report.message, 5000)

        return report


class AbstractPanelMeta(type(QW.QSplitter), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class AbstractPanel(QW.QSplitter, metaclass=AbstractPanelMeta):
    """Object defining DataLab panel interface,
    based on a vertical QSplitter widget

    A panel handle an object list (objects are signals, images, macros...).
    Each object must implement ``datalab.gui.ObjItf`` interface
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
        name = f"{get_short_id(obj)}: {title}"
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
            if isinstance(obj, (SignalObj, ImageObj, ObjectGroup)):
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


class PasteMetadataParam(gds.DataSet):
    """Paste metadata parameters"""

    keep_roi = gds.BoolItem(_("Regions of interest"), default=True)
    keep_geometry = gds.BoolItem(_("Geometry results"), default=False).set_pos(col=1)
    keep_tables = gds.BoolItem(_("Table results"), default=False).set_pos(col=1)
    keep_other = gds.BoolItem(_("Other metadata"), default=True)


class NonModalInfoDialog(QW.QMessageBox):
    """Non-modal information message box with selectable text.

    This widget displays an information message in a message dialog box, allowing users
    to select and copy the text content.
    """

    def __init__(self, parent: QW.QWidget, title: str, text: str) -> None:
        """Create a non-modal information message box with selectable text.

        Args:
            parent: The parent widget.
            title: The title of the message box.
            text: The text to display in the message box.
        """
        super().__init__(parent)
        self.setIcon(QW.QMessageBox.Information)
        self.setWindowTitle(title)
        if re.search(r"<[a-zA-Z/][^>]*>", text):
            self.setTextFormat(QC.Qt.RichText)  # type: ignore[attr-defined]
            self.setTextInteractionFlags(
                QC.Qt.TextBrowserInteraction  # type: ignore[attr-defined]
            )
        else:
            self.setTextFormat(QC.Qt.PlainText)  # type: ignore[attr-defined]
            self.setTextInteractionFlags(
                QC.Qt.TextSelectableByMouse  # type: ignore[attr-defined]
                | QC.Qt.TextSelectableByKeyboard  # type: ignore[attr-defined]
            )
        self.setText(text)
        self.setStandardButtons(QW.QMessageBox.Close)
        self.setDefaultButton(QW.QMessageBox.Close)
        # ! Necessary only on non-Windows platforms
        self.setWindowFlags(QC.Qt.Window)  # type: ignore[attr-defined]
        self.setModal(False)


class SaveToDirectoryGUIParam(gds.DataSet, title=_("Save to directory")):
    """Save to directory parameters"""

    def __init__(
        self, objs: list[TypeObj] | None = None, extensions: list[str] | None = None
    ) -> None:
        super().__init__()
        self.__objs = objs or []
        self.__extensions = extensions or []

    def on_button_click(
        self: SaveToDirectoryGUIParam,
        _item: gds.ButtonItem,
        _value: None,
        parent: QW.QWidget,
    ) -> None:
        """Help button callback."""
        text = "<br>".join(
            [
                """Pattern accepts a Python format string. Standard Python format
                specifiers apply. Two extra modifiers are supported: 'upper' for
                uppercase and 'lower' for lowercase.""",
                "",
                "<b>Available placeholders:</b>",
                """
            <table border="1" cellspacing="0" cellpadding="4">
                <tr><th>Keyword</th><th>Description</th></tr>
                <tr><td>{title}</td><td>Title</td></tr>
                <tr><td>{index}</td><td>1-based index</td></tr>
                <tr><td>{count}</td><td>Total number of selected objects</td></tr>
                <tr><td>{xlabel}, {xunit}, {ylabel}, {yunit}</td>
                    <td>Axis information for signals</td></tr>
                <tr><td>{metadata[key]}</td><td>Specific metadata value<br>
                    <i>(direct {metadata} use is ignored)</i></td></tr>
            </table>
            """,
                "",
                "<b>Examples:</b>",
                """
            <table border="1" cellspacing="0" cellpadding="4">
                <tr><th>Pattern</th><th>Description</th></tr>
                <tr>
                    <td>{index:03d}</td>
                    <td>3-digit index with leading zeros</td>
                </tr>
                <tr>
                    <td>{title:20.20}</td>
                    <td>Title truncated to 20 characters</td>
                </tr>
                <tr>
                    <td>{title:20.20upper}</td>
                    <td>Title truncated to 20 characters, upper case</td>
                </tr>
                <tr>
                    <td>{title:20.20lower}</td>
                    <td>Title truncated to 20 characters, lower case</td>
                </tr>
            </table>
            """,
            ]
        )
        NonModalInfoDialog(parent, "Pattern help", text).show()

    def get_extension_choices(self, _item=None, _value=None):
        """Return list of available extensions for choice item."""
        return [("." + ext, "." + ext, None) for ext in self.__extensions]

    def build_filenames(self, objs: list[TypeObj] | None = None) -> list[str]:
        """Build filenames according to current parameters."""
        objs = objs or self.__objs
        extension = self.extension if self.extension is not None else ""
        filenames = format_basenames(objs, self.basename + extension)
        used: set[str] = set()  # Ensure all filenames are unique.
        for i, filename in enumerate(filenames):
            root, ext = osp.splitext(filename)
            filepath = osp.join(self.directory, filename)
            k = 1
            while (filename in used) or (not self.overwrite and osp.exists(filepath)):
                filename = f"{root}_{k}{ext}"
                filepath = osp.join(self.directory, filename)
                k += 1
            used.add(filename)
            filenames[i] = filename
        return filenames

    def generate_filepath_obj_pairs(
        self, objs: list[TypeObj]
    ) -> Generator[tuple[str, TypeObj], None, None]:
        """Iterate over (filepath, object) pairs to be saved."""
        for filename, obj in zip(self.build_filenames(objs), objs):
            yield osp.join(self.directory, filename), obj

    def update_preview(self, _item=None, _value=None) -> None:
        """Update preview."""
        try:
            filenames = self.build_filenames()
            preview_lines = []
            for i, (obj, filename) in enumerate(zip(self.__objs, filenames), start=1):
                # Try to get short ID if object has been added to panel
                try:
                    obj_id = get_short_id(obj)
                except (ValueError, KeyError):
                    # Fallback to simple index for objects not yet in panel
                    obj_id = str(i)
                preview_lines.append(f"{obj_id}: {filename}")
            self.preview = "\n".join(preview_lines)
        except (ValueError, KeyError, TypeError) as exc:
            # Handle formatting errors gracefully (e.g., incomplete format string)
            self.preview = f"Invalid pattern:{os.linesep}{exc}"

    directory = gds.DirectoryItem(_("Directory"), default=Conf.main.base_dir.get())

    basename = gds.StringItem(
        _("Basename pattern"),
        default="{title}",
        help=_("Python format string. See description for details."),
    ).set_prop("display", callback=update_preview)

    help = gds.ButtonItem(_("Help"), on_button_click, "MessageBoxInformation").set_pos(
        col=1
    )

    extension = gds.ChoiceItem(_("Extension"), get_extension_choices).set_prop(
        "display", callback=update_preview
    )

    overwrite = gds.BoolItem(
        _("Overwrite"), default=False, help=_("Overwrite existing files")
    ).set_pos(col=1)

    preview = gds.TextItem(
        _("Preview"), default=None, regexp=r"^(?!Invalid).*"
    ).set_prop("display", readonly=True)


class AddMetadataParam(
    gds.DataSet,
    title=_("Add metadata"),
    comment=_(
        "Add a new metadata item to the selected objects.<br><br>"
        "The metadata key will be the same for all objects, "
        "but the value can use a pattern to generate different values.<br>"
        "Click the <b>Help</b> button for details on the pattern syntax.<br>"
    ),
):
    """Add metadata parameters"""

    def __init__(self, objs: list[TypeObj] | None = None) -> None:
        super().__init__()
        self.__objs = objs or []

    def on_help_button_click(
        self: AddMetadataParam,
        _item: gds.ButtonItem,
        _value: None,
        parent: QW.QWidget,
    ) -> None:
        """Help button callback."""
        text = "<br>".join(
            [
                """Pattern accepts a Python format string. Standard Python format
                specifiers apply. Two extra modifiers are supported: 'upper' for
                uppercase and 'lower' for lowercase.""",
                "",
                "<b>Available placeholders:</b>",
                """
            <table border="1" cellspacing="0" cellpadding="4">
                <tr><th>Keyword</th><th>Description</th></tr>
                <tr><td>{title}</td><td>Title</td></tr>
                <tr><td>{index}</td><td>1-based index</td></tr>
                <tr><td>{count}</td><td>Total number of selected objects</td></tr>
                <tr><td>{xlabel}, {xunit}, {ylabel}, {yunit}</td>
                    <td>Axis information for signals</td></tr>
                <tr><td>{metadata[key]}</td><td>Specific metadata value<br>
                    <i>(direct {metadata} use is ignored)</i></td></tr>
            </table>
            """,
                "",
                "<b>Examples:</b>",
                """
            <table border="1" cellspacing="0" cellpadding="4">
                <tr><th>Pattern</th><th>Description</th></tr>
                <tr>
                    <td>{index:03d}</td>
                    <td>3-digit index with leading zeros</td>
                </tr>
                <tr>
                    <td>{title:20.20}</td>
                    <td>Title truncated to 20 characters</td>
                </tr>
                <tr>
                    <td>{title:20.20upper}</td>
                    <td>Title truncated to 20 characters, upper case</td>
                </tr>
                <tr>
                    <td>{title:20.20lower}</td>
                    <td>Title truncated to 20 characters, lower case</td>
                </tr>
            </table>
            """,
            ]
        )
        NonModalInfoDialog(parent, "Pattern help", text).show()

    def get_conversion_choices(self, _item=None, _value=None):
        """Return list of available conversion choices."""
        return [
            ("string", _("String"), None),
            ("float", _("Float"), None),
            ("int", _("Integer"), None),
            ("bool", _("Boolean"), None),
        ]

    def build_values(
        self, objs: list[TypeObj] | None = None
    ) -> list[str | float | int | bool]:
        """Build values according to current parameters."""
        objs = objs or self.__objs
        # Generate values using the pattern
        raw_values = format_basenames(objs, self.value_pattern)

        # Convert values according to the selected conversion type
        converted_values = []
        for value_str in raw_values:
            if self.conversion == "string":
                converted_values.append(value_str)
            elif self.conversion == "float":
                try:
                    converted_values.append(float(value_str))
                except ValueError:
                    # Keep as string if conversion fails
                    converted_values.append(value_str)
            elif self.conversion == "int":
                try:
                    converted_values.append(int(value_str))
                except ValueError:
                    # Keep as string if conversion fails
                    converted_values.append(value_str)
            elif self.conversion == "bool":
                # Convert to boolean: "true", "1", "yes" -> True, others -> False
                lower_val = value_str.lower()
                converted_values.append(lower_val in ("true", "1", "yes", "on"))

        return converted_values

    def update_preview(self, _item=None, _value=None) -> None:
        """Update preview."""
        try:
            values = self.build_values()
            preview_lines = []
            for i, (obj, value) in enumerate(zip(self.__objs, values), start=1):
                # Try to get short ID if object has been added to panel
                try:
                    obj_id = get_short_id(obj)
                except (ValueError, KeyError):
                    # Fallback to simple index for objects not yet in panel
                    obj_id = str(i)
                preview_lines.append(f"{obj_id}: {self.metadata_key} = {value!r}")
            self.preview = "\n".join(preview_lines)
        except (ValueError, KeyError, TypeError) as exc:
            # Handle formatting errors gracefully (e.g., incomplete format string)
            self.preview = f"Invalid pattern:{os.linesep}{exc}"

    metadata_key = gds.StringItem(
        _("Metadata key"),
        default="custom_key",
        notempty=True,
        regexp=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        help=_("The key name for the metadata item"),
    ).set_prop("display", callback=update_preview)

    value_pattern = gds.StringItem(
        _("Value pattern"),
        default="{index}",
        help=_("Python format string. See description for details."),
    ).set_prop("display", callback=update_preview)

    help = gds.ButtonItem(
        _("Help"), on_help_button_click, "MessageBoxInformation"
    ).set_pos(col=1)

    conversion = gds.ChoiceItem(
        _("Conversion"), get_conversion_choices, default="string"
    ).set_prop("display", callback=update_preview)

    preview = gds.TextItem(
        _("Preview"), default=None, regexp=r"^(?!Invalid).*"
    ).set_prop("display", readonly=True)


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
    SIG_STATUS_MESSAGE = QC.Signal(str, int)  # emitted by "qt_try_except" decorator
    SIG_REFRESH_PLOT = QC.Signal(
        str, bool, bool, bool, bool
    )  # Connected to PlotHandler.refresh_plot

    @staticmethod
    @abc.abstractmethod
    def get_roi_class() -> Type[TypeROI]:
        """Return ROI class"""

    @staticmethod
    @abc.abstractmethod
    def get_roieditor_class() -> Type[TypeROIEditor]:
        """Return ROI editor class"""

    @abc.abstractmethod
    def __init__(self, parent: QW.QWidget) -> None:
        super().__init__(parent)
        self.mainwindow: DLMainWindow = parent
        self.objprop = ObjectProp(self, self.PARAMCLASS)
        self.objmodel = objectmodel.ObjectModel()
        self.objview = objectview.ObjectView(self, self.objmodel)
        self.objview.SIG_IMPORT_FILES.connect(self.handle_dropped_files)
        self.objview.populate_tree()
        self.plothandler: SignalPlotHandler | ImagePlotHandler = None
        self.processor: SignalProcessor | ImageProcessor = None
        self.acthandler: actionhandler.BaseActionHandler = None
        self.__metadata_clipboard = {}
        self.__roi_clipboard: TypeROI | None = None
        self.context_menu = QW.QMenu()
        self.__separate_views: dict[QW.QDialog, TypeObj] = {}

    def closeEvent(self, event):
        """Reimplement QMainWindow method"""
        self.processor.close()
        super().closeEvent(event)

    # ------AbstractPanel interface-----------------------------------------------------
    def plot_item_parameters_changed(
        self, item: CurveItem | MaskedXYImageItem | LabelItem
    ) -> None:
        """Plot items changed: update metadata of all objects from plot items"""
        # Find the object corresponding to the plot item
        obj = self.plothandler.get_obj_from_item(item)
        if obj is not None:
            # Unselect the item in the plot so that we update the item parameters
            # in the right state (fix issue #184):
            item.unselect()
            # Ensure that item's parameters are up-to-date:
            item.param.update_param(item)
            # Update object metadata from plot item parameters
            create_adapter_from_object(obj).update_metadata_from_plot_item(item)
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
            create_adapter_from_object(obj).update_metadata_from_plot_item(item)
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
        # Update object properties panel to clear creation/processing tabs
        self.selection_changed()

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
             If False, only show the items (do not update them).
             Defaults to True.
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

        # Rename geometry results to avoid conflicts when pasting to same object type
        new_pref = get_short_id(obj) + "_"
        self._rename_results_in_clipboard(new_pref)

    def _rename_results_in_clipboard(self, prefix: str) -> None:
        """Rename geometry and table results in clipboard to avoid conflicts.

        Args:
            prefix: Prefix to add to result titles
        """
        for aclass in (GeometryAdapter, TableAdapter):
            result_keys = [
                k for k, v in self.__metadata_clipboard.items() if aclass.match(k, v)
            ]
            for dict_key in result_keys:
                try:
                    # Get the result data
                    result_data = self.__metadata_clipboard[dict_key]

                    # Update the title in the result data
                    if isinstance(result_data, dict) and "title" in result_data:
                        result_data = result_data.copy()  # Don't modify original
                        result_data["title"] = prefix + result_data["title"]

                        # Create new key with updated title
                        new_dict_key = dict_key.replace(
                            aclass.META_PREFIX, aclass.META_PREFIX + prefix, 1
                        )

                        # Remove old entry and add new one
                        del self.__metadata_clipboard[dict_key]
                        self.__metadata_clipboard[new_dict_key] = result_data

                except (KeyError, ValueError, IndexError, TypeError):
                    # If we can't process this result, leave it as is
                    continue

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
            if not param.edit(parent=self.parentWidget()):
                return
        metadata = {}
        if param.keep_roi and ROI_KEY in self.__metadata_clipboard:
            metadata[ROI_KEY] = self.__metadata_clipboard[ROI_KEY]
        if param.keep_geometry:
            for key, value in self.__metadata_clipboard.items():
                if GeometryAdapter.match(key, value):
                    metadata[key] = value
        if param.keep_tables:
            for key, value in self.__metadata_clipboard.items():
                if TableAdapter.match(key, value):
                    metadata[key] = value
        if param.keep_other:
            for key, value in self.__metadata_clipboard.items():
                if (
                    not GeometryAdapter.match(key, value)
                    and not TableAdapter.match(key, value)
                    and key not in METADATA_PASTE_EXCLUSIONS
                ):
                    metadata[key] = value
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sorted(sel_objects, key=get_short_id, reverse=True):
            obj.update_metadata_from(metadata)
        # We have to do a special refresh in order to force the plot handler to update
        # all plot items, even the ones that are not visible (otherwise, image masks
        # would not be updated after pasting the metadata: see issue #123)
        self.refresh_plot(
            "selected", update_items=True, only_visible=False, only_existing=True
        )

    def add_metadata(self, param: AddMetadataParam | None = None) -> None:
        """Add metadata item to selected object(s)

        Args:
            param: Add metadata parameters
        """
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        if not sel_objects:
            return

        if param is None:
            param = AddMetadataParam(sel_objects)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
                if not param.edit(parent=self.parentWidget(), wordwrap=False):
                    return

        # Build values for all selected objects
        values = param.build_values(sel_objects)

        # Add metadata to each object
        for obj, value in zip(sel_objects, values):
            obj.metadata[param.metadata_key] = value

        # Refresh the plot to update any changes
        self.refresh_plot(
            "selected", update_items=True, only_visible=False, only_existing=True
        )

    def copy_roi(self) -> None:
        """Copy regions of interest"""
        obj = self.objview.get_sel_objects()[0]
        self.__roi_clipboard = obj.roi.copy()

    def paste_roi(self) -> None:
        """Paste regions of interest"""
        sel_objects = self.objview.get_sel_objects(include_groups=True)
        for obj in sel_objects:
            if obj.roi is None:
                obj.roi = self.__roi_clipboard.copy()
            else:
                obj.roi = obj.roi.combine_with(self.__roi_clipboard)
        self.selection_changed(update_items=True)
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
        for obj in sorted(sel_objects, key=get_short_id, reverse=True):
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
                obj.mark_roi_as_changed()
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
        param: NewSignalParam | NewImageParam | None = None,
        edit: bool = False,
        add_to_panel: bool = True,
    ) -> TypeObj | None:
        """Create a new object (signal/image).

        Args:
            param: new object parameters
            edit: Open a dialog box to edit parameters (default: False).
             When False, the object is created with default parameters and creation
             parameters are stored in metadata for interactive editing.
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
        objs = qt_long_callback(self, _("Reading objects from file"), worker, True)
        group_id = None
        if len(objs) > 1 and create_group:
            # Create a new group if more than one object is loaded
            group_id = get_uuid(self.add_group(osp.basename(filename)))
        with create_progress_bar(
            self, _("Adding objects to workspace"), max_=len(objs) - 1
        ) as progress:
            for i_obj, obj in enumerate(objs):
                progress.setValue(i_obj + 1)
                if progress.wasCanceled():
                    break
                if add_objects:
                    set_uuid(obj)  # In case the object UUID was serialized in the file,
                    # we need to reset it to a new UUID to avoid conflicts
                    # (e.g. HDF5 file)
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
            with qt_try_loadsave_file(self.parentWidget(), filename, "load"):
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
                with qt_try_loadsave_file(self.parentWidget(), filename, "save"):
                    Conf.main.base_dir.set(filename)
                    self.__save_to_file(obj, filename)

    def save_to_directory(self, param: SaveToDirectoryParam | None = None) -> None:
        """Save signals or images to directory using a filename pattern.

        Opens a dialog to select the output directory, the basename pattern and the
        extension.

        Args:
            param: parameters.
        """
        objs = self.objview.get_sel_objects(include_groups=True)

        if param is None:
            extensions = get_file_extensions(self.IO_REGISTRY.get_write_filters())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
                guiparam = SaveToDirectoryGUIParam(objs, extensions)
                if not guiparam.edit(parent=self.parentWidget()):
                    return
            param = SaveToDirectoryParam()
            update_dataset(param, guiparam)

        Conf.main.base_dir.set(param.directory)

        with create_progress_bar(self, _("Saving..."), max_=len(objs)) as progress:
            for i, (path, obj) in enumerate(param.generate_filepath_obj_pairs(objs)):
                progress.setValue(i + 1)
                if progress.wasCanceled():
                    break
                with qt_try_loadsave_file(self.parentWidget(), path, "save"):
                    self.__save_to_file(obj, path)

    def handle_dropped_files(self, filenames: list[str] | None = None) -> None:
        """Handle dropped files

        Args:
            filenames: File names

        Returns:
            None
        """
        dirnames = [fname for fname in filenames if osp.isdir(fname)]
        h5_fnames = [
            fname for fname in filenames if is_hdf5_file(fname, check_content=True)
        ]
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
        wizard = TextImportWizard(self.PANEL_STR_ID, parent=self.parentWidget())
        if exec_dialog(wizard):
            objs = wizard.get_objs()
            if objs:
                with create_progress_bar(
                    self, _("Adding objects to workspace"), max_=len(objs) - 1
                ) as progress:
                    for idx, obj in enumerate(objs):
                        progress.setValue(idx)
                        QW.QApplication.processEvents()
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
                    self, _("Import metadata"), basedir, "*.dlabmeta"
                )
        if filename:
            with qt_try_loadsave_file(self.parentWidget(), filename, "load"):
                Conf.main.base_dir.set(filename)
                obj = self.objview.get_sel_objects(include_groups=True)[0]
                obj.metadata = read_metadata(filename)
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
                    self, _("Export metadata"), basedir, "*.dlabmeta"
                )
        if filename:
            with qt_try_loadsave_file(self.parentWidget(), filename, "save"):
                Conf.main.base_dir.set(filename)
                write_metadata(filename, obj.metadata)

    def import_roi_from_file(self, filename: str | None = None) -> None:
        """Import regions of interest from file (JSON).

        Args:
            filename: File name
        """
        if filename is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getopenfilename(
                    self, _("Import ROI"), basedir, "*.dlabroi"
                )
        if filename:
            with qt_try_loadsave_file(self.parentWidget(), filename, "load"):
                Conf.main.base_dir.set(filename)
                obj = self.objview.get_sel_objects(include_groups=True)[0]
                roi = read_roi(filename)
                if obj.roi is None:
                    obj.roi = roi
                else:
                    obj.roi = obj.roi.combine_with(roi)
            self.selection_changed(update_items=True)
            self.refresh_plot("selected", True, False)

    def export_roi_to_file(self, filename: str | None = None) -> None:
        """Export regions of interest to file (JSON).

        Args:
            filename: File name
        """
        obj = self.objview.get_sel_objects(include_groups=True)[0]
        assert obj.roi is not None
        if filename is None:  # pragma: no cover
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Export ROI"), basedir, "*.dlabroi"
                )
        if filename:
            with qt_try_loadsave_file(self.parentWidget(), filename, "save"):
                Conf.main.base_dir.set(filename)
                write_roi(filename, obj.roi)

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
        # Get only the properties that have changed from the original values
        changed_props = self.objprop.get_changed_properties()

        # Apply only the changed properties to all selected objects
        for obj in self.objview.get_sel_objects(include_groups=True):
            obj.mark_roi_as_changed()
            # Update only the changed properties instead of all properties
            update_dataset(obj, changed_props)
            self.objview.update_item(get_uuid(obj))
        # Refresh all selected items, including non-visible ones (only_visible=False)
        # This ensures that plot items are updated for all selected objects, even if
        # they are temporarily hidden behind other objects
        self.refresh_plot(
            "selected", update_items=True, force=False, only_visible=False
        )

        # Update the stored original values to reflect the new state
        # This ensures subsequent changes are compared against the current values
        self.objprop.update_original_values()

    def recompute_processing(self) -> None:
        """Recompute/rerun selected objects or group with stored processing parameters.

        This method handles both single objects and groups. For each object, it checks
        if it has 1-to-1 processing parameters that can be recomputed. Objects without
        recomputable parameters are skipped.
        """
        # Get selected objects (handles both individual selection and groups)
        objects = self.objview.get_sel_objects(include_groups=True)
        if not objects:
            return

        # Filter objects that have recomputable processing parameters
        recomputable_objects: list[SignalObj | ImageObj] = []
        for obj in objects:
            proc_params = extract_processing_parameters(obj)
            if proc_params is not None and proc_params.pattern == "1-to-1":
                recomputable_objects.append(obj)

        if not recomputable_objects:
            QW.QMessageBox.information(
                self,
                _("Recompute"),
                _(
                    "Selected object(s) do not have processing parameters "
                    "that can be recomputed."
                ),
            )
            return

        # Recompute each object
        with create_progress_bar(
            self, _("Recomputing objects"), max_=len(recomputable_objects)
        ) as progress:
            for index, obj in enumerate(recomputable_objects):
                progress.setValue(index + 1)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break

                # Temporarily set this object as current to use existing infrastructure
                self.objview.set_current_object(obj)
                report = self.objprop.apply_processing_parameters(
                    obj=obj, interactive=False
                )
                if not report.success:
                    failtxt = _("Failed to recompute object")
                    if index == len(recomputable_objects) - 1:
                        QW.QMessageBox.warning(
                            self,
                            _("Recompute"),
                            f"{failtxt} '{obj.title}':\n{report.message}",
                        )
                    else:
                        conttxt = _("Do you want to continue with the next object?")
                        answer = QW.QMessageBox.warning(
                            self,
                            _("Recompute"),
                            f"{failtxt} '{obj.title}':\n{report.message}\n\n{conttxt}",
                            QW.QMessageBox.Yes | QW.QMessageBox.No,
                        )
                        if answer == QW.QMessageBox.No:
                            break

    def select_source_objects(self) -> None:
        """Select source objects associated with the selected object's processing.

        This method retrieves the source object UUIDs from the selected object's
        processing parameters and selects them in the object view.
        """
        # Get the selected object (should be exactly one)
        objects = self.objview.get_sel_objects(include_groups=False)
        if len(objects) != 1:
            return

        obj = objects[0]

        # Extract processing parameters
        proc_params = extract_processing_parameters(obj)
        if proc_params is None:
            QW.QMessageBox.information(
                self,
                _("Select source objects"),
                _("Selected object does not have processing metadata."),
            )
            return

        # Get source UUIDs
        source_uuids = []
        if proc_params.source_uuid:
            source_uuids.append(proc_params.source_uuid)
        if proc_params.source_uuids:
            source_uuids.extend(proc_params.source_uuids)

        if not source_uuids:
            QW.QMessageBox.information(
                self,
                _("Select source objects"),
                _("Selected object does not have source object references."),
            )
            return

        # Check if source objects still exist
        existing_uuids = [
            uuid for uuid in source_uuids if uuid in self.objmodel.get_object_ids()
        ]
        if not existing_uuids:
            QW.QMessageBox.warning(
                self,
                _("Select source objects"),
                _("Source object(s) no longer exist."),
            )
            return

        # Select the existing source objects
        self.objview.clearSelection()
        for uuid in existing_uuids:
            self.objview.set_current_item_id(uuid, extend=True)

        # Show info if some sources are missing
        missing_count = len(source_uuids) - len(existing_uuids)
        if missing_count > 0:
            QW.QMessageBox.information(
                self,
                _("Select source objects"),
                _("Selected %d source object(s). %d source object(s) no longer exist.")
                % (len(existing_uuids), missing_count),
            )

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

    def get_dialog_size(self) -> tuple[int, int]:
        """Get dialog size (minimum and maximum)"""
        # Resize the dialog so that it's at least MINDIALOGSIZE (absolute values),
        # and at most MAXDIALOGSIZE (% of the main window size):
        minwidth, minheight = self.MINDIALOGSIZE
        maxwidth = int(self.mainwindow.width() * self.MAXDIALOGSIZE)
        maxheight = int(self.mainwindow.height() * self.MAXDIALOGSIZE)
        size = min(minwidth, maxwidth), min(minheight, maxheight)
        return size

    def create_new_dialog(
        self,
        edit: bool = False,
        toolbar: bool = True,
        title: str | None = None,
        name: str | None = None,
        options: dict[str, Any] | None = None,
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
        plot_options = self.plothandler.get_plot_options()
        if options is not None:
            plot_options = plot_options.copy(options)

        # pylint: disable=not-callable
        dlg = PlotDialog(
            parent=self.parentWidget(),
            title=APP_NAME if title is None else f"{title} - {APP_NAME}",
            options=plot_options,
            toolbar=toolbar,
            icon="DataLab.svg",
            edit=edit,
            size=self.get_dialog_size(),
        )
        dlg.setObjectName(name)
        return dlg

    def get_roi_editor_output(
        self, mode: Literal["apply", "extract", "define"] = "apply"
    ) -> tuple[TypeROI, bool] | None:
        """Get ROI data (array) from specific dialog box.

        Args:
            mode: Mode of operation, either "apply" (define ROI, then apply it to
             selected objects), "extract" (define ROI, then extract data from it),
             or "define" (define ROI without applying or extracting).

        Returns:
            A tuple containing the ROI object and a boolean indicating whether the
            dialog was accepted or not.
        """
        obj = self.objview.get_sel_objects(include_groups=True)[-1]
        item = create_adapter_from_object(obj).make_item(
            update_from=self.plothandler[get_uuid(obj)]
        )
        roi_editor_class = self.get_roieditor_class()  # pylint: disable=not-callable
        roi_editor = roi_editor_class(
            parent=self.parentWidget(),
            obj=obj,
            mode=mode,
            item=item,
            options=self.plothandler.get_plot_options(),
            size=self.get_dialog_size(),
        )
        if exec_dialog(roi_editor):
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
            for rdata in rdatadict.values():
                show_resultdata(self.parentWidget(), rdata, f"{objs[0].PREFIX}_results")
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

    def __plot_result(
        self, category: str, rdata: ResultData, objs: list[SignalObj | ImageObj]
    ) -> None:
        """Plot results for a specific category"""
        xchoices = (("indices", _("Indices")),)
        for xlabel in rdata.headers:
            # If this column data is not numeric, we skip it:
            if not isinstance(
                rdata.results[0].get_column_values(xlabel)[0], (int, float, np.number)
            ):
                continue
            xchoices += ((xlabel, xlabel),)
        ychoices = xchoices[1:]

        # Regrouping ResultShape results by their `title` attribute:
        grouped_results: dict[str, list[GeometryAdapter | TableAdapter]] = {}
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
            if any(len(result.to_dataframe()) > 1 for result in rdata.results)
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
            yaxis = gds.ChoiceItem(_("Y axis"), ychoices, default=ychoices[0][0])

        comment = (
            _(
                "Plot results obtained from previous analyses.<br><br>"
                "This plot is based on results associated with '%s' prefix."
            )
            % category
        )
        param = PlotResultParam(_("Plot results"), comment=comment)
        if not param.edit(parent=self.parentWidget()):
            return

        if param.kind == "one_curve_per_title":
            # One curve per ROI (if any) and per result title
            # ------------------------------------------------------------------
            # Begin by checking if all results have the same number of ROIs:
            # for simplicity, let's check the number of unique ROI indices.
            all_roi_indexes = [
                result.get_unique_roi_indices() for result in rdata.results
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
            obj = objs[0]
            for i_roi in all_roi_indexes[0]:
                roi_suffix = f"|ROI{int(i_roi + 1)}" if i_roi >= 0 else ""
                for title, results in grouped_results.items():  # title
                    x, y = [], []
                    for index, result in enumerate(results):
                        if param.xaxis == "indices":
                            x.append(index)
                        else:
                            x.append(result.get_column_values(param.xaxis, i_roi)[0])
                        y.append(result.get_column_values(param.yaxis, i_roi)[0])
                    if i_roi >= 0:
                        roi_suffix = f"|{obj.roi.get_single_roi_title(int(i_roi))}"
                    self.__add_result_signal(
                        x, y, f"{title}{roi_suffix}", param.xaxis, param.yaxis
                    )
        else:
            # One curve per result title, per object and per ROI
            # ------------------------------------------------------------------
            for title, results in grouped_results.items():  # title
                for index, result in enumerate(results):  # object
                    obj = objs[index]
                    roi_indices = result.get_unique_roi_indices()
                    for i_roi in roi_indices:  # ROI
                        roi_suffix = ""
                        if i_roi >= 0:
                            roi_suffix = f"|{obj.roi.get_single_roi_title(int(i_roi))}"
                        roi_data = result.get_roi_data(i_roi)
                        if param.xaxis == "indices":
                            x = list(range(len(roi_data)))
                        else:
                            x = roi_data[param.xaxis].values
                        y = roi_data[param.yaxis].values
                        shid = get_short_id(objs[index])
                        stitle = f"{title} ({shid}){roi_suffix}"
                        self.__add_result_signal(x, y, stitle, param.xaxis, param.yaxis)

    def plot_results(self) -> None:
        """Plot results"""
        objs = self.objview.get_sel_objects(include_groups=True)
        rdatadict = create_resultdata_dict(objs)
        if rdatadict:
            for category, rdata in rdatadict.items():
                self.__plot_result(category, rdata, objs)
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
                    # Remove all table and geometry results using adapter methods
                    TableAdapter.remove_all_from(obj)
                    GeometryAdapter.remove_all_from(obj)
                self.refresh_plot("selected", True, False)
        else:
            self.__show_no_result_warning()

    def add_label_with_title(
        self, title: str | None = None, ignore_msg: bool = True
    ) -> None:
        """Add a label with object title on the associated plot

        Args:
            title: Label title. Defaults to None.
             If None, the title is the object title.
            ignore_msg: If True, do not show the information message. Defaults to True.
             If False, show a message box to inform the user that the label has been
             added as an annotation, and that it can be edited or removed using the
             annotation editing window.
        """
        objs = self.objview.get_sel_objects(include_groups=True)
        for obj in objs:
            create_adapter_from_object(obj).add_label_with_title(title=title)
        if (
            not Conf.view.ignore_title_insertion_msg.get(False)
            and not ignore_msg
            and not execenv.unattended
        ):
            answer = QW.QMessageBox.information(
                self,
                _("Annotation added"),
                _(
                    "The label has been added as an annotation. "
                    "You can edit or remove it using the annotation editing window."
                    "<br><br>"
                    "Choosing to ignore this message will prevent it "
                    "from being displayed again."
                ),
                QW.QMessageBox.Ok | QW.QMessageBox.Ignore,
            )
            if answer == QW.QMessageBox.Ignore:
                Conf.view.ignore_title_insertion_msg.set(True)
        self.refresh_plot("selected", True, False)
