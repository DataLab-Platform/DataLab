# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Base Processor GUI module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import abc
import warnings
from typing import Callable, Dict, List

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
from guidata.configtools import get_icon
from guidata.widgets.arrayeditor import ArrayEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from codraft import env
from codraft.config import _
from codraft.core.gui.objectlist import ObjectList
from codraft.core.gui.roieditor import ROIEditorData
from codraft.core.model.base import ResultShape
from codraft.utils import misc
from codraft.utils.qthelpers import (
    create_progress_bar,
    exec_dialog,
    qt_try_except,
)


class GaussianParam(gdt.DataSet):
    """Gaussian filter parameters"""

    sigma = gdi.FloatItem("σ", default=1.0)


class MovingAverageParam(gdt.DataSet):
    """Moving average parameters"""

    n = gdi.IntItem(_("Size of the moving window"), default=3, min=1)


class MovingMedianParam(gdt.DataSet):
    """Moving median parameters"""

    n = gdi.IntItem(_("Size of the moving window"), default=3, min=1, even=False)


class ThresholdParam(gdt.DataSet):
    """Threshold parameters"""

    value = gdi.FloatItem(_("Threshold"))


class ClipParam(gdt.DataSet):
    """Data clipping parameters"""

    value = gdi.FloatItem(_("Clipping value"))


class BaseProcessor(QC.QObject):
    """Object handling data processing: operations, processing, computing"""

    SIG_ADD_SHAPE = QC.Signal(int)
    EDIT_ROI_PARAMS = False

    def __init__(self, panel, objlist: ObjectList, plotwidget):
        super().__init__()
        self.panel = panel
        self.objlist = objlist
        self.plotwidget = plotwidget
        self.prefix = panel.PREFIX

    @qt_try_except()
    def compute_sum(self):
        """Compute sum"""
        rows = self.objlist.get_selected_rows()
        outobj = self.panel.create_object()
        outobj.title = "+".join([f"{self.prefix}{row:03d}" for row in rows])
        roilist = []
        for row in rows:
            obj = self.objlist[row]
            if obj.roi is not None:
                roilist.append(obj.roi)
            if outobj.data is None:
                outobj.copy_data_from(obj)
            else:
                outobj.data += np.array(obj.data, dtype=outobj.data.dtype)
                outobj.update_resultshapes_from(obj)
        if roilist:
            outobj.roi = np.vstack(roilist)
        self.panel.add_object(outobj)

    @qt_try_except()
    def compute_average(self):
        """Compute average"""
        rows = self.objlist.get_selected_rows()
        outobj = self.panel.create_object()
        title = ", ".join([f"{self.prefix}{row:03d}" for row in rows])
        outobj.title = f'{_("Average")}({title})'
        original_dtype = self.objlist.get_sel_object().data.dtype
        new_dtype = complex if misc.is_complex_dtype(original_dtype) else float
        for row in rows:
            obj = self.objlist[row]
            if outobj.data is None:
                outobj.copy_data_from(obj, dtype=new_dtype)
            else:
                outobj.data += np.array(obj.data, dtype=outobj.data.dtype)
                outobj.update_resultshapes_from(obj)
        outobj.data /= float(len(rows))
        if misc.is_integer_dtype(original_dtype):
            outobj.set_data_type(dtype=original_dtype)
        self.panel.add_object(outobj)

    @qt_try_except()
    def compute_product(self):
        """Compute product"""
        rows = self.objlist.get_selected_rows()
        outobj = self.panel.create_object()
        outobj.title = "*".join([f"{self.prefix}{row:03d}" for row in rows])
        for row in rows:
            obj = self.objlist[row]
            if outobj.data is None:
                outobj.copy_data_from(obj)
            else:
                outobj.data *= np.array(obj.data, dtype=outobj.data.dtype)
        self.panel.add_object(outobj)

    @qt_try_except()
    def compute_difference(self, quad: bool):
        """Compute (quadratic) difference"""
        rows = self.objlist.get_selected_rows()
        outobj = self.panel.create_object()
        outobj.title = "-".join([f"{self.prefix}{row:03d}" for row in rows])
        if quad:
            outobj.title = f"({outobj.title})/sqrt(2)"
        obj0, obj1 = self.objlist.get_sel_object(), self.objlist.get_sel_object(1)
        outobj.copy_data_from(obj0)
        outobj.data -= np.array(obj1.data, dtype=outobj.data.dtype)
        if quad:
            outobj.data = outobj.data / np.sqrt(2.0)
        if np.issubdtype(outobj.data.dtype, np.unsignedinteger):
            outobj.data[obj0.data < obj1.data] = 0
        self.panel.add_object(outobj)

    @qt_try_except()
    def compute_division(self):
        """Compute division"""
        rows = self.objlist.get_selected_rows()
        outobj = self.panel.create_object()
        outobj.title = "/".join([f"{self.prefix}{row:03d}" for row in rows])
        obj0, obj1 = self.objlist.get_sel_object(), self.objlist.get_sel_object(1)
        outobj.copy_data_from(obj0)
        outobj.data = outobj.data / np.array(obj1.data, dtype=outobj.data.dtype)
        self.panel.add_object(outobj)

    def _get_roieditordata(
        self, roidata: np.ndarray = None, singleobj: bool = None
    ) -> ROIEditorData:
        """Eventually open ROI Editing Dialog, and return ROI editor data"""
        # Expected behavior:
        # -----------------
        # * If roidata argument is not None, skip the ROI dialog
        # * If first selected obj has a ROI, use this ROI as default but open
        #   ROI Editor dialog anyway
        # * If multiple objs are selected, then apply the first obj ROI to all

        if roidata is None:
            roieditordata = self.edit_regions_of_interest(extract=True)
            if roieditordata is not None and roieditordata.roidata is None:
                # This only happens in unattended mode (forcing QDialog accept)
                return None
        else:
            roieditordata = ROIEditorData()
            roieditordata.roidata = roidata
            roieditordata.singleobj = singleobj
        return roieditordata

    @abc.abstractmethod
    def extract_roi(self, roidata: np.ndarray = None) -> None:
        """Extract Region Of Interest (ROI) from data"""

    @abc.abstractmethod
    def swap_axes(self):
        """Swap data axes"""

    @abc.abstractmethod
    def compute_abs(self):
        """Compute absolute value"""

    @abc.abstractmethod
    def compute_log10(self):
        """Compute Log10"""

    # ------Data Processing
    @abc.abstractmethod
    def apply_11_func(self, obj, orig, func, param, message):
        """Apply 11 function: 1 object in --> 1 object out"""

    def compute_11(
        self,
        name: str,
        func: Callable,
        param: gdt.DataSet = None,
        suffix: Callable = None,
        func_obj: Callable = None,
        edit: bool = True,
    ):
        """Compute 11 function: 1 object in --> 1 object out"""
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return
        self._compute_11_subroutine([name], func, [param], suffix, func_obj)

    def compute_1n(
        self,
        names: List,
        func: Callable,
        params: List = None,
        suffix: Callable = None,
        func_obj: Callable = None,
        edit: bool = True,
    ):
        """Compute 1n function: 1 object in --> n objects out"""
        if params is not None:
            group = gdt.DataSetGroup(params, title=_("Parameters"))
            if edit and not group.edit(parent=self.panel.parent()):
                return
        self._compute_11_subroutine(names, func, params, suffix, func_obj)

    def _compute_11_subroutine(
        self,
        names: List,
        func: Callable,
        params: List,
        suffix: Callable,
        func_obj: Callable,
    ):
        """Compute 11 subroutine: used by compute 11 and compute 1n methods"""
        rows = self.objlist.get_selected_rows()
        with create_progress_bar(
            self.panel, names[0], max_=len(rows) * len(params)
        ) as progress:
            for i_row, row in enumerate(rows):
                for i_param, (param, name) in enumerate(zip(params, names)):
                    progress.setValue(i_row * i_param)
                    progress.setLabelText(name)
                    QW.QApplication.processEvents()
                    if progress.wasCanceled():
                        break
                    orig = self.objlist[row]
                    obj = self.panel.create_object()
                    obj.title = f"{name}({self.prefix}{row:03d})"
                    if suffix is not None:
                        obj.title += "|" + suffix(param)
                    obj.copy_data_from(orig)
                    message = _("Computing:") + " " + obj.title
                    self.apply_11_func(obj, orig, func, param, message)
                    if func_obj is not None:
                        if param is None:
                            func_obj(obj)
                        else:
                            func_obj(obj, param)
                    self.panel.add_object(obj)

    @abc.abstractmethod
    def apply_10_func(self, orig, func, param, message) -> ResultShape:
        """Apply 10 function: 1 object in --> 0 object out (scalar result)"""

    def compute_10(
        self,
        name: str,
        func: Callable,
        param: gdt.DataSet = None,
        suffix: Callable = None,
        edit: bool = True,
    ) -> Dict[int, ResultShape]:
        """Compute 10 function: 1 object in --> 0 object out
        (the result of this method is stored in original object's metadata)"""
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return None
        rows = self.objlist.get_selected_rows()
        with create_progress_bar(self.panel, name, max_=len(rows)) as progress:
            results = {}
            xlabels = None
            ylabels = []
            title_suffix = "" if suffix is None else "|" + suffix(param)
            for idx, row in enumerate(rows):
                progress.setValue(idx)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                orig = self.objlist[row]
                title = f"{name}{title_suffix}"
                message = _("Computing:") + " " + title
                result = self.apply_10_func(orig, func, param, message)
                if result is None:
                    continue
                results[row] = result
                xlabels = result.xlabels
                self.SIG_ADD_SHAPE.emit(row)
                self.panel.current_item_changed(row)
                self.panel.SIG_REFRESH_PLOT.emit()
                for _i_row_res in range(result.array.shape[0]):
                    ylabel = f"{name}({self.prefix}{idx:03d}){title_suffix}"
                    ylabels.append(ylabel)
        if results:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dlg = ArrayEditor(self.panel.parent())
                title = _("Results")
                res = np.vstack([result.array for result in results.values()])
                dlg.setup_and_check(
                    res, title, readonly=True, xlabels=xlabels, ylabels=ylabels
                )
                dlg.setObjectName(f"{self.prefix}_results")
                dlg.resize(750, 300)
                exec_dialog(dlg)
        return results

    @abc.abstractmethod
    @qt_try_except()
    def calibrate(self, param=None) -> None:
        """Compute data linear calibration"""

    @staticmethod
    @abc.abstractmethod
    def func_gaussian_filter(x, y, p):
        """Compute gaussian filter"""

    @qt_try_except()
    def compute_gaussian(self, param: GaussianParam = None) -> None:
        """Compute gaussian filter"""
        edit = param is None
        if edit:
            param = GaussianParam(_("Gaussian filter"))
        func = self.func_gaussian_filter
        self.compute_11(
            "GaussianFilter",
            func,
            param,
            suffix=lambda p: f"σ={p.sigma:.3f} pixels",
            edit=edit,
        )

    @staticmethod
    @abc.abstractmethod
    def func_moving_average(x, y, p):
        """Moving average computing function"""

    @qt_try_except()
    def compute_moving_average(self, param: MovingAverageParam = None) -> None:
        """Compute moving average"""
        edit = param is None
        if edit:
            param = MovingAverageParam(_("Moving average"))
        func = self.func_moving_average
        self.compute_11("MovAvg", func, param, suffix=lambda p: f"n={p.n}", edit=edit)

    @staticmethod
    @abc.abstractmethod
    def func_moving_median(x, y, p):
        """Moving median computing function"""

    @qt_try_except()
    def compute_moving_median(self, param: MovingMedianParam = None) -> None:
        """Compute moving median"""
        edit = param is None
        if edit:
            param = MovingMedianParam(_("Moving median"))
        func = self.func_moving_median
        self.compute_11("MovMed", func, param, suffix=lambda p: f"n={p.n}", edit=edit)

    @abc.abstractmethod
    @qt_try_except()
    def compute_wiener(self):
        """Compute Wiener filter"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_fft(self):
        """Compute iFFT"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_ifft(self):
        """Compute FFT"""

    # ------Computing
    def edit_regions_of_interest(self, extract=False) -> ROIEditorData:
        """Define Region Of Interest (ROI) for computing functions"""
        roieditordata = self.panel.get_roi_dialog(extract=extract)
        if roieditordata is not None:
            row = self.objlist.get_selected_rows()[0]
            obj = self.objlist[row]
            roigroup = obj.roidata_to_params(roieditordata.roidata)
            if (
                env.execenv.unattended
                or roieditordata.roidata.size == 0
                or not self.EDIT_ROI_PARAMS
                or roigroup.edit(parent=self.panel)
            ):
                roidata = obj.params_to_roidata(roigroup)
                if roieditordata.modified:
                    # If ROI has been modified, save ROI (even in "extract mode")
                    obj.roi = roidata
                    self.SIG_ADD_SHAPE.emit(row)
                    self.panel.current_item_changed(row)
                    self.panel.SIG_REFRESH_PLOT.emit()
        return roieditordata

    @abc.abstractmethod
    def _get_stat_funcs(self):
        """Return statistics functions list"""

    @qt_try_except()
    def compute_stats(self):
        """Compute data statistics"""
        row = self.objlist.get_selected_rows()[0]
        obj = self.objlist.get_sel_object()
        stfuncs = self._get_stat_funcs()
        nbcal = len(stfuncs)
        roi_nb = 0 if obj.roi is None else obj.roi.shape[0]
        res = np.zeros((1 + roi_nb, nbcal))
        xlabels = [None] * nbcal
        obj_t = f"{self.prefix}{row:03d}"
        ylabels = [None] * (roi_nb + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with np.errstate(all="ignore"):
                for iroi, roi_index in enumerate([None] + list(range(roi_nb))):
                    for ical, (label, func) in enumerate(stfuncs):
                        xlabels[ical] = label
                        res[iroi, ical] = func(obj.get_data(roi_index=roi_index))
                    if roi_index is None:
                        ylabels[iroi] = obj_t
                    else:
                        ylabels[iroi] = f"{obj_t}|ROI{roi_index:02d}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dlg = ArrayEditor(self.panel.parent())
            title = _("Statistics")
            dlg.setup_and_check(
                res, title, readonly=True, xlabels=xlabels, ylabels=ylabels
            )
            dlg.setObjectName(f"{self.prefix}_stats")
            dlg.setWindowIcon(get_icon("stats.svg"))
            dlg.resize(750, 300)
            exec_dialog(dlg)
