# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Signal panel (see parent package :mod:`cdl.core.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

from plotpy.tools import (
    HCursorTool,
    HRangeTool,
    LabelTool,
    RectangleTool,
    SegmentTool,
    VCursorTool,
    XCursorTool,
)

from cdl.config import _
from cdl.core.gui import roieditor
from cdl.core.gui.actionhandler import SignalActionHandler
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.plothandler import SignalPlotHandler
from cdl.core.gui.processor.signal import SignalProcessor
from cdl.core.io.signal import SignalIORegistry
from cdl.core.model.signal import (
    CURVESTYLES,
    SignalObj,
    create_signal_from_param,
    new_signal_param,
)

if TYPE_CHECKING:
    import guidata.dataset as gds
    from plotpy.plot import PlotWidget
    from qtpy import QtWidgets as QW

    from cdl.core.model.signal import NewSignalParam


class SignalPanel(BaseDataPanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Signal objects"""

    PANEL_STR = _("Signal Panel")
    PANEL_STR_ID = "signal"
    PARAMCLASS = SignalObj
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
    ROIDIALOGCLASS = roieditor.SignalROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent: QW.QWidget, plotwidget: PlotWidget, toolbar) -> None:
        super().__init__(parent, plotwidget, toolbar)
        self.plothandler = SignalPlotHandler(self, plotwidget)
        self.processor = SignalProcessor(self, plotwidget)
        self.acthandler = SignalActionHandler(self, toolbar)

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
            newparam = new_signal_param(title=title)
        if curobj is not None:
            newparam.size = len(curobj.data)
            newparam.xmin = curobj.x.min()
            newparam.xmax = curobj.x.max()
        return newparam

    def new_object(
        self,
        newparam: NewSignalParam | None = None,
        addparam: gds.DataSet | None = None,
        edit: bool = True,
        add_to_panel: bool = True,
    ) -> SignalObj | None:
        """Create a new object (signal).

        Args:
            newparam (guidata.dataset.DataSet): new object parameters
            addparam (guidata.dataset.DataSet): additional parameters
            edit (bool): Open a dialog box to edit parameters (default: True)
            add_to_panel (bool): Add the new object to the panel (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        newparam = self.get_newparam_from_current(newparam)
        signal = create_signal_from_param(
            newparam, addparam=addparam, edit=edit, parent=self.parent()
        )
        if signal is None:
            return None
        if add_to_panel:
            self.add_object(signal)
        return signal

    # ------Plotting--------------------------------------------------------------------
    def toggle_anti_aliasing(self, state: bool) -> None:
        """Toggle anti-aliasing on/off

        Args:
            state: state of the anti-aliasing
        """
        self.plothandler.toggle_anti_aliasing(state)

    def reset_curve_styles(self) -> None:
        """Reset curve styles"""
        CURVESTYLES.reset_styles()
