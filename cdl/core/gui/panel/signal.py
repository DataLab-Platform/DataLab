# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""CobraDataLab Signal Panel"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from guiqwt.plot import CurveDialog

from cdl.config import _
from cdl.core.gui import plotitemlist, roieditor
from cdl.core.gui.actionhandler import SignalActionHandler
from cdl.core.gui.panel.base import BaseDataPanel
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

    PANEL_STR = _("Signal List")
    PARAMCLASS = SignalParam
    DIALOGCLASS = CurveDialog
    PREFIX = "s"
    IO_REGISTRY = SignalIORegistry
    H5_PREFIX = "CDL_Sig"
    ROIDIALOGCLASS = roieditor.SignalROIEditor

    # pylint: disable=duplicate-code

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = plotitemlist.SignalItemList(self, self.objlist, plotwidget)
        self.processor = proc = SignalProcessor(self, self.objlist, plotwidget)
        self.acthandler = SignalActionHandler(self, self.itmlist, proc, toolbar)

    # ------Creating, adding, removing objects------------------------------------------
    def new_object(self, newparam=None, addparam=None, edit=True):
        """Create a new signal.
        :param cdl.core.model.signal.SignalNewParam newparam: new signal parameters
        :param guidata.dataset.datatypes.DataSet addparam: additional parameters
        :param bool edit: Open a dialog box to edit parameters (default: True)
        """
        if not self.mainwindow.confirm_memory_state():
            return
        curobj: SignalParam = self.objlist.get_sel_object(-1)
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
