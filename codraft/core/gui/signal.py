# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Signal GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import os.path as osp
import traceback
from typing import Callable

import numpy as np
from guidata.configtools import get_icon
from guiqwt.builder import make
from guiqwt.label import ObjectInfo
from guiqwt.plot import CurveDialog
from qtpy import QtWidgets as QW
from qtpy.compat import getsavefilename

from codraft.config import APP_NAME, Conf, _
from codraft.core.gui import base as guibase
from codraft.core.gui.processor.signal import SignalProcessor
from codraft.core.model.signal import (
    SignalParam,
    create_signal,
    create_signal_from_param,
    new_signal_param,
)
from codraft.utils.qthelpers import save_restore_stds
from codraft.widgets import fitdialog


class SignalActionHandler(guibase.BaseActionHandler):
    """Object handling signal panel GUI interactions: actions, menus, ..."""

    OBJECT_STR = _("signal")

    def create_operation_actions(self):
        """Create operation actions"""
        base_actions = super().create_operation_actions()
        proc = self.processor
        peakdetect_action = self.cra(
            _("Peak detection"),
            proc.detect_peaks,
            icon=get_icon("peak_detect.svg"),
        )
        self.actlist_1more += [peakdetect_action]
        roi_actions = self.operation_end_actions
        return base_actions + [None, peakdetect_action, None] + roi_actions

    def create_processing_actions(self):
        """Create processing actions"""
        base_actions = super().create_processing_actions()
        proc = self.processor
        normalize_action = self.cra(_("Normalize"), proc.normalize)
        deriv_action = self.cra(_("Derivative"), proc.compute_derivative)
        integ_action = self.cra(_("Integral"), proc.compute_integral)
        polyfit_action = self.cra(_("Polynomial fit"), proc.compute_polyfit)
        mgfit_action = self.cra(_("Multi-Gaussian fit"), proc.compute_multigaussianfit)

        def cra_fit(title, fitdlgfunc):
            """Create curve fitting action"""
            return self.cra(title, lambda: proc.compute_fit(title, fitdlgfunc))

        gaussfit_action = cra_fit(_("Gaussian fit"), fitdialog.gaussianfit)
        lorentzfit_action = cra_fit(_("Lorentzian fit"), fitdialog.lorentzianfit)
        voigtfit_action = cra_fit(_("Voigt fit"), fitdialog.voigtfit)
        actions1 = [normalize_action, deriv_action, integ_action]
        actions2 = [
            gaussfit_action,
            lorentzfit_action,
            voigtfit_action,
            polyfit_action,
            mgfit_action,
        ]
        self.actlist_1more += actions1 + actions2
        return actions1 + [None] + base_actions + [None] + actions2

    def create_computing_actions(self):
        """Create computing actions"""
        base_actions = super().create_computing_actions()
        proc = self.processor
        fwhm_action = self.cra(
            _("Full width at half-maximum"),
            triggered=proc.compute_fwhm,
            tip=_("Compute Full Width at Half-Maximum (FWHM)"),
        )
        fw1e2_action = self.cra(
            _("Full width at") + " 1/e²",
            triggered=proc.compute_fw1e2,
            tip=_("Compute Full Width at Maximum") + "/e²",
        )
        self.actlist_1more += [fwhm_action, fw1e2_action]
        return base_actions + [fwhm_action, fw1e2_action]


class ROIRangeInfo(ObjectInfo):
    """ObjectInfo for ROI selection"""

    def __init__(self, roi_items):
        self.roi_items = roi_items

    def get_text(self):
        textlist = []
        for index, roi_item in enumerate(self.roi_items):
            x0, x1 = roi_item.get_range()
            textlist.append(f"ROI{index:02d}: {x0} ≤ x ≤ {x1}")
        return "<br>".join(textlist)


class SignalROIEditor(guibase.BaseROIEditor):
    """Signal ROI Editor"""

    ICON_NAME = "signal_roi_new.svg"

    def __init__(self, parent: QW.QDialog, roi_items: list, func: Callable):
        info = ROIRangeInfo(roi_items)
        info_label = make.info_label("BL", info, title=_("Regions of interest"))
        parent.get_plot().add_item(info_label)
        self.info_label = info_label
        super().__init__(parent, roi_items, func)

    def update_roi_titles(self):
        """Update ROI annotation titles"""
        self.info_label.update_text()

    @staticmethod
    def get_roi_item_coords(roi_item):
        """Return ROI item coords"""
        return roi_item.get_range()


class SignalPanel(guibase.BasePanel):
    """Object handling the item list, the selected item properties and plot,
    specialized for Signal objects"""

    PANEL_STR = "Signal Panel"
    PARAMCLASS = SignalParam
    DIALOGCLASS = CurveDialog
    PREFIX = "s"
    OPEN_FILTERS = f'{_("Text files")} (*.txt *.csv)\n{_("NumPy arrays")} (*.npy)'
    H5_PREFIX = "CodraFT_Sig"
    ROIDIALOGCLASS = SignalROIEditor

    def __init__(self, parent, plotwidget, toolbar):
        super().__init__(parent, plotwidget, toolbar)
        self.itmlist = guibase.BaseItemList(self, self.objlist, plotwidget)
        self.processor = SignalProcessor(self, self.objlist)
        self.acthandler = SignalActionHandler(
            self, self.objlist, self.itmlist, self.processor, toolbar
        )
        self.setup_panel()

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
        if curobj is not None:  # pylint: disable=duplicate-code
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
        if osp.splitext(filename)[1] == ".npy":
            xydata = np.load(filename)
        else:
            for delimiter in ("\t", ",", " ", ";"):
                try:
                    xydata = np.loadtxt(filename, delimiter=delimiter)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError
        assert len(xydata.shape) in (1, 2), "Data not supported"
        signal = create_signal(osp.basename(filename))
        if len(xydata.shape) == 1:
            signal.set_xydata(np.arange(xydata.size), xydata)
        else:
            rows, cols = xydata.shape
            for colnb in (2, 3, 4):
                if cols == colnb and rows > colnb:
                    xydata = xydata.T
                    break
            if cols == 3:
                # x, y, dy
                xarr, yarr, dyarr = xydata
                signal.set_xydata(xarr, yarr, dx=None, dy=dyarr)
            else:
                signal.xydata = xydata
        self.add_object(signal)

    def save_object(self, obj, filename: str = None) -> None:
        """Save object to file (signal/image)"""
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filter = getsavefilename(  # pylint: disable=duplicate-code
                    self, _("Save as"), basedir, _("CSV files") + " (*.csv)"
                )
        if filename:
            Conf.main.base_dir.set(filename)
            try:
                np.savetxt(filename, obj.xydata, delimiter=",")
            except Exception as msg:  # pylint: disable=broad-except
                traceback.print_exc()
                QW.QMessageBox.critical(
                    self.parent(),
                    APP_NAME,
                    (_("%s could not be written:") % osp.basename(filename))
                    + "\n"
                    + str(msg),
                )
