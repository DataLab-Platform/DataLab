# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal panel (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import guidata.dataset as gds
from guidata.config import ValidationMode, temporary_validation_mode
from plotpy.tools import (
    HCursorTool,
    HRangeTool,
    LabelTool,
    RectangleTool,
    SegmentTool,
    VCursorTool,
    XCursorTool,
)
from sigima.io.signal import SignalIORegistry
from sigima.objects import SignalObj, SignalROI

from datalab.adapters_plotpy import CURVESTYLES, LINE_STYLE_KEYS, LINE_WIDTH_KEY
from datalab.config import Conf, _
from datalab.gui import roieditor
from datalab.gui.actionhandler import SignalActionHandler
from datalab.gui.newobject import NewSignalParam, create_signal_gui
from datalab.gui.panel.base import BaseDataPanel
from datalab.gui.plothandler import SignalPlotHandler
from datalab.gui.processor.signal import SignalProcessor
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW

    from datalab.gui.docks import DockablePlotWidget
    from datalab.h5.native import NativeH5Reader

#: Line width stored by DataLab 1.3 or earlier for curves drawn with the default
#: line width, or with a line width clamped for curves with many points
LEGACY_AUTOMATIC_LINE_WIDTH = 1.0


class SignalPanel(BaseDataPanel[SignalObj, SignalROI, roieditor.SignalROIEditor]):
    """Object handling the item list, the selected item properties and plot,
    specialized for Signal objects"""

    PANEL_STR = _("Signal Panel")
    PANEL_STR_ID = "signal"
    PARAMCLASS = SignalObj

    # The following tools are used to create annotations on signals. The annotation
    # items are created using PlotPy's default settings. Those appearance settings
    # may be modified in the configuration (see `datalab.config`).
    ANNOTATION_TOOLS = (
        LabelTool,
        VCursorTool,
        HCursorTool,
        XCursorTool,
        SegmentTool,
        RectangleTool,
        HRangeTool,
    )

    IO_REGISTRY = SignalIORegistry
    H5_PREFIX = "DataLab_Sig"

    # pylint: disable=duplicate-code

    @staticmethod
    def get_roi_class() -> Type[SignalROI]:
        """Return ROI class"""
        return SignalROI

    @staticmethod
    def get_roieditor_class() -> Type[roieditor.SignalROIEditor]:
        """Return ROI editor class"""
        return roieditor.SignalROIEditor

    def __init__(
        self,
        parent: QW.QWidget,
        dockableplotwidget: DockablePlotWidget,
        panel_toolbar: QW.QToolBar,
    ) -> None:
        super().__init__(parent)
        self.plothandler = SignalPlotHandler(self, dockableplotwidget.plotwidget)
        self.processor = SignalProcessor(self, dockableplotwidget.plotwidget)
        view_toolbar = dockableplotwidget.toolbar
        self.acthandler = SignalActionHandler(self, panel_toolbar, view_toolbar)

    def deserialize_object_from_hdf5(
        self, reader: NativeH5Reader, name: str, reset_all: bool = False
    ) -> SignalObj:
        """Deserialize object from a HDF5 file

        DataLab 1.3 and earlier stored automatic curve styles in signal metadata:
        when loading a file saved by these versions, the colors and line styles of
        their automatic style cycle are removed, so that curves are given the
        current automatic styles. Other stored styles are kept. These versions
        also stored the line width of every curve: a line width of 1.0 (the
        default width, also used for curves with many points) is removed, so that
        curves are given the configured line width; other widths were chosen by
        the user and are kept.

        Args:
            reader: HDF5 reader
            name: Object name in HDF5 file
            reset_all: If True, preserve original UUIDs (workspace reload).
             If False, regenerate only UUIDs that conflict with existing
             objects (object import).
        """
        obj = super().deserialize_object_from_hdf5(reader, name, reset_all)
        metadata = obj.metadata
        if reader.has_legacy_curve_styles:
            if CURVESTYLES.is_legacy_automatic(
                *(metadata.get(key) for key in LINE_STYLE_KEYS)
            ):
                for key in LINE_STYLE_KEYS:
                    metadata.pop(key, None)
            if metadata.get(LINE_WIDTH_KEY) == LEGACY_AUTOMATIC_LINE_WIDTH:
                metadata.pop(LINE_WIDTH_KEY)
        return obj

    # ------Creating, adding, removing objects------------------------------------------
    def get_newparam_from_current(
        self, newparam: NewSignalParam | None = None, title: str | None = None
    ) -> NewSignalParam | None:
        """Get new object parameters from the current object.

        Args:
            newparam (guidata.dataset.DataSet): new object parameters.
             If None, create a new one.
            title: new object title. If None, use the current object title, or the
             default title.

        Returns:
            New object parameters
        """
        curobj: SignalObj = self.objview.get_current_object()
        if newparam is None:
            newparam = NewSignalParam()
        if title is not None:
            newparam.title = title
        if curobj is not None and Conf.use_signal_bounds.get(False):
            # Use current signal bounds for new signal:
            newparam.size = len(curobj.data)
            # try to set xmin/xmax from current signal data range
            with temporary_validation_mode(ValidationMode.STRICT):
                try:
                    newparam.xmin = curobj.x.min()
                except gds.DataItemValidationError:
                    pass
                try:
                    newparam.xmax = curobj.x.max()
                except gds.DataItemValidationError:
                    pass
        return newparam

    def new_object(
        self,
        param: NewSignalParam | None = None,
        edit: bool = False,
        add_to_panel: bool = True,
    ) -> SignalObj | None:
        """Create a new object (signal).

        Args:
            param (guidata.dataset.DataSet): new object parameters
            edit (bool): Open a dialog box to edit parameters (default: False).
             When False, the object is created with default parameters and creation
             parameters are stored in metadata for interactive editing.
            add_to_panel (bool): Add the new object to the panel (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        param = self.get_newparam_from_current(param)
        signal = create_signal_gui(param, edit=edit, parent=self.parentWidget())
        if signal is None:
            return None
        action = self.mainwindow.historypanel.add_ui_entry(
            _("New signal"),
            target="signalpanel",
            method_name="new_object",
            save_state=False,
            param=param,
            add_to_panel=add_to_panel,
        )
        if add_to_panel:
            self.add_object(signal)
            if action is not None:
                self.mainwindow.historypanel.register_action_outputs(
                    action, [get_uuid(signal)]
                )
        return signal

    # ------Plotting--------------------------------------------------------------------
    def toggle_anti_aliasing(self, state: bool) -> None:
        """Toggle anti-aliasing on/off

        Args:
            state: state of the anti-aliasing
        """
        self.plothandler.toggle_anti_aliasing(state)

    def reset_curve_styles(self) -> None:
        """Reset the curve styles of the selected signals

        Colors and line styles chosen by the user are removed from the metadata of
        the selected signals, whose curves are given automatic styles again.
        """
        CURVESTYLES.reset_styles()
        modified = False
        for obj in self.objview.get_sel_objects(include_groups=True):
            for key in LINE_STYLE_KEYS:
                if obj.metadata.pop(key, None) is not None:
                    modified = True
            item = self.plothandler.get(get_uuid(obj))
            if item is not None and not CURVESTYLES.has_automatic_style(item):
                CURVESTYLES.restore_automatic_style(item)
                item.param.update_item(item)
                modified = True
        self.refresh_plot("selected", update_items=False)
        if modified:
            self.SIG_OBJECT_MODIFIED.emit()
