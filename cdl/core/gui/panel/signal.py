# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""DataLab Signal Panel"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from guiqwt.plot import CurveDialog

from cdl.config import _
from cdl.core.gui import roieditor
from cdl.core.gui.actionhandler import SignalActionHandler
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.plothandler import SignalPlotHandler
from cdl.core.gui.processor.signal import SignalProcessor
from cdl.core.io.signal import SignalIORegistry
from cdl.core.model.signal import (
    SignalParam,
    create_signal_from_param,
    new_signal_param,
)


class SignalPanel(BaseDataPanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Signal objects"""

    PANEL_STR = _("Signal panel")
    PARAMCLASS = SignalParam
    DIALOGCLASS = CurveDialog
    IO_REGISTRY = SignalIORegistry
    H5_PREFIX = "DataLab_Sig"
    ROIDIALOGCLASS = roieditor.SignalROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.plothandler = SignalPlotHandler(self, plotwidget)
        self.processor = SignalProcessor(self, plotwidget)
        self.acthandler = SignalActionHandler(self, toolbar)

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(self, newparam=None, addparam=None, edit=True) -> SignalParam:
        """Create a new object (signal).

        Args:
            newparam (DataSet): new object parameters
            addparam (DataSet): additional parameters
            edit (bool): Open a dialog box to edit parameters (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        curobj: SignalParam = self.objview.get_current_object()
        if curobj is not None:
            newparam = newparam if newparam is not None else new_signal_param()
            newparam.size = len(curobj.data)
            newparam.xmin = curobj.x.min()
            newparam.xmax = curobj.x.max()
        signal = create_signal_from_param(
            newparam, addparam=addparam, edit=edit, parent=self
        )
        if signal is None:
            return None
        self.add_object(signal)
        return signal
