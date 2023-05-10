# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Base Processor GUI module
---------------------------------

This module defines the base class for data processing GUIs.

.. autosummary::

    BaseProcessor

.. autoclass:: BaseProcessor
    :members:
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
from guidata.configtools import get_icon
from guidata.utils import update_dataset
from guidata.widgets.arrayeditor import ArrayEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl import env
from cdl.config import _
from cdl.core.gui.roieditor import ROIEditorData
from cdl.core.model.base import ResultShape
from cdl.utils import misc
from cdl.utils.qthelpers import create_progress_bar, exec_dialog, qt_try_except

if TYPE_CHECKING:
    from guiqwt.plot import CurveWidget, ImageWidget

    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel
    from cdl.core.model.image import ImageParam
    from cdl.core.model.signal import SignalParam

    Obj = Union[SignalParam, ImageParam]


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

    SIG_ADD_SHAPE = QC.Signal(str)
    EDIT_ROI_PARAMS = False
    PARAM_DEFAULTS: Dict[str, gdt.DataSet] = {}

    def __init__(
        self, panel: SignalPanel | ImagePanel, plotwidget: CurveWidget | ImageWidget
    ):
        super().__init__()
        self.panel = panel
        self.plotwidget = plotwidget

    def init_param(
        self,
        param: gdt.DataSet,
        paramclass: gdt.DataSet,
        title: str,
        comment: str = None,
    ) -> Tuple[bool, gdt.DataSet]:
        """Initialize processing parameters"""
        edit = param is None
        if edit:
            param = paramclass(title, comment)
            pdefaults = self.PARAM_DEFAULTS.get(paramclass.__name__)
            if pdefaults is not None:
                update_dataset(param, pdefaults)
            self.PARAM_DEFAULTS[paramclass.__name__] = param
        return edit, param

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
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        grps = self.panel.objview.get_sel_groups()
        new_gids = {}
        with create_progress_bar(
            self.panel, names[0], max_=len(objs) * len(params)
        ) as progress:
            for i_row, obj in enumerate(objs):
                for i_param, (param, name) in enumerate(zip(params, names)):
                    progress.setValue(i_row * i_param)
                    progress.setLabelText(name)
                    QW.QApplication.processEvents()
                    if progress.wasCanceled():
                        break
                    new_obj = self.panel.create_object()
                    new_obj.title = f"{name}({obj.short_id})"
                    if suffix is not None:
                        new_obj.title += "|" + suffix(param)
                    new_obj.copy_data_from(obj)
                    message = _("Computing:") + " " + new_obj.title
                    self.apply_11_func(new_obj, obj, func, param, message)
                    if func_obj is not None:
                        if param is None:
                            func_obj(new_obj, obj)
                        else:
                            func_obj(new_obj, obj, param)
                    new_gid = None
                    if grps:
                        # If groups are selected, then it means that there is no
                        # individual object selected: we work on groups only
                        old_gid = self.panel.objmodel.get_object_group_id(obj)
                        new_gid = new_gids.get(old_gid)
                        if new_gid is None:
                            # Create a new group for each selected group
                            old_g = self.panel.objmodel.get_group(old_gid)
                            new_g = self.panel.add_group(f"{name}({old_g.short_id})")
                            new_gids[old_gid] = new_gid = new_g.uuid
                    self.panel.add_object(new_obj, group_id=new_gid)
        # Select newly created groups, if any
        for group_id in new_gids.values():
            self.panel.objview.set_current_item_id(group_id, extend=True)

    def apply_10_func(self, obj, func, param, message) -> ResultShape:
        """Apply 10 function: 1 object in --> 0 object out (scalar result)"""

        # (self is used by @qt_try_except)
        # pylint: disable=unused-argument
        @qt_try_except(message)
        def apply_10_func_callback(self, obj, func, param):
            """Apply 10 function cb: 1 object in --> 0 object out (scalar result)"""
            if param is None:
                return func(obj)
            return func(obj, param)

        return apply_10_func_callback(self, obj, func, param)

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
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        with create_progress_bar(self.panel, name, max_=len(objs)) as progress:
            results = {}
            xlabels = None
            ylabels = []
            title_suffix = "" if suffix is None else "|" + suffix(param)
            for idx, obj in enumerate(objs):
                progress.setValue(idx)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                title = f"{name}{title_suffix}"
                message = _("Computing:") + " " + title
                result = self.apply_10_func(obj, func, param, message)
                if result is None:
                    continue
                results[obj.uuid] = result
                xlabels = result.xlabels
                self.SIG_ADD_SHAPE.emit(obj.uuid)
                self.panel.selection_changed()
                self.panel.SIG_UPDATE_PLOT_ITEM.emit(obj.uuid)
                for _i_row_res in range(result.array.shape[0]):
                    ylabel = f"{name}({obj.short_id}){title_suffix}"
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
                dlg.setObjectName(f"{objs[0].PREFIX}_results")
                dlg.resize(750, 300)
                exec_dialog(dlg)
        return results

    def compute_n1(
        self,
        name: str,
        func: Callable,
        param: gdt.DataSet = None,
        suffix: Callable = None,
        func_objs: Callable = None,
        edit: bool = True,
    ):
        """Compute n1 function: N(>=2) objects in --> 1 object out"""
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)

        # [new_objs dictionary] keys: old group id, values: new object
        new_objs: Dict[str, Obj] = {}
        # [old_dtypes dictionary] keys: old group id, values: old data type
        old_dtypes: Dict[str, np.dtype] = {}
        # [old_objs dictionary] keys: old group id, values: list of old objects
        old_objs: Dict[str, List[Obj]] = {}

        with create_progress_bar(self.panel, name, max_=len(objs)) as progress:
            for index, obj in enumerate(objs):
                progress.setValue(index)
                progress.setLabelText(name)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                old_gid = self.panel.objmodel.get_object_group_id(obj)
                new_obj = new_objs.get(old_gid)
                if new_obj is None:
                    old_dtypes[old_gid] = old_dtype = obj.data.dtype
                    new_objs[old_gid] = new_obj = self.panel.create_object()
                    old_objs[old_gid] = [obj]
                    if suffix is not None:
                        new_obj.title += "|" + suffix(param)
                    new_dtype = complex if misc.is_complex_dtype(old_dtype) else float
                    new_obj.copy_data_from(obj, dtype=new_dtype)
                else:
                    old_objs[old_gid].append(obj)
                    if param is None:
                        func(obj.data, new_obj.data)
                    else:
                        func(obj.data, new_obj.data, param)
                    new_obj.update_resultshapes_from(obj)
                if obj.roi is not None:
                    if new_obj.roi is None:
                        new_obj.roi = obj.roi.copy()
                    else:
                        new_obj.roi = np.vstack((new_obj.roi, obj.roi))

        grps = self.panel.objview.get_sel_groups()
        if grps:
            # (Group exclusive selection)
            # At least one group is selected: create a new group
            new_gname = f"{name}({','.join([grp.short_id for grp in grps])}"
            new_gid = self.panel.add_group(new_gname).uuid
        else:
            # (Object exclusive selection)
            # No group is selected: use each object's group
            new_gid = None

        for old_gid, new_obj in new_objs.items():
            if misc.is_integer_dtype(old_dtypes[old_gid]):
                new_obj.set_data_type(dtype=old_dtypes[old_gid])
            if func_objs is not None:
                func_objs(new_obj, old_objs[old_gid])
            short_ids = [obj.short_id for obj in old_objs[old_gid]]
            new_obj.title = f'{name}({", ".join(short_ids)})'
            group_id = new_gid if new_gid is not None else old_gid
            self.panel.add_object(new_obj, group_id=group_id)

        # Select newly created groups, if any
        if new_gid is not None:
            self.panel.objview.set_current_item_id(new_gid)

    def compute_n1n(
        self,
        name: str,
        obj2: Optional[Obj],
        obj2_name: str,
        func: Callable,
        param: gdt.DataSet = None,
        suffix: Callable = None,
        func_obj: Callable = None,
        edit: bool = True,
    ):
        """Compute n1n function: N(>=1) objects + 1 object in --> N objects out.

        Examples: subtract, divide"""
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return
        if obj2 is None:
            obj2 = self.panel.get_object_dialog(_("Select %s") % obj2_name)
            if obj2 is None:
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)
        with create_progress_bar(self.panel, name, max_=len(objs)) as progress:
            for index, obj in enumerate(objs):
                progress.setValue(index)
                progress.setLabelText(name)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                new_obj = self.panel.create_object()
                short_ids = (obj.short_id, obj2.short_id)
                new_obj.title = f'{name}({", ".join(short_ids)})'
                new_obj.copy_data_from(obj)
                if param is None:
                    new_obj.data = func(obj.data, obj2.data)
                else:
                    new_obj.data = func(obj.data, obj2.data, param)
                if func_obj is not None:
                    if param is None:
                        func_obj(new_obj, obj, obj2)
                    else:
                        func_obj(new_obj, obj, obj2, param)
                if suffix is not None:
                    new_obj.title += "|" + suffix(param)
                group_id = self.panel.objmodel.get_object_group_id(obj)
                self.panel.add_object(new_obj, group_id=group_id)

    # ------Data Operations-------------------------------------------------------------

    @staticmethod
    def __sum_func(in_i: np.ndarray, out: np.ndarray) -> None:
        """Compute sum of input data"""
        out += np.array(in_i, dtype=out.dtype)

    @qt_try_except()
    def compute_sum(self):
        """Compute sum"""
        self.compute_n1(_("Sum"), self.__sum_func)

    @qt_try_except()
    def compute_average(self):
        """Compute average"""

        def func_objs(new_obj: Obj, old_objs: List[Obj]) -> None:
            """Finalize average computation"""
            new_obj.data = new_obj.data / float(len(old_objs))

        self.compute_n1(_("Average"), self.__sum_func, func_objs=func_objs)

    @qt_try_except()
    def compute_product(self):
        """Compute product"""

        def prod_func(in_i: np.ndarray, out: np.ndarray) -> None:
            """Compute product of input data"""
            out *= np.array(in_i, dtype=out.dtype)

        self.compute_n1(_("Product"), prod_func)

    @qt_try_except()
    def compute_difference(
        self,
        obj2: Optional[Obj] = None,
        quadratic: Optional[bool] = None,
    ):
        """Compute (quadratic) difference"""

        def func(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
            """Function to be applied to each object"""
            data = data1 - np.array(data2, dtype=data1.dtype)
            if quadratic:
                data = data / np.sqrt(2.0)
            return data

        def func_obj(new_obj: Obj, obj: Obj, obj2: Obj) -> None:
            """Function to be applied to each object"""
            if np.issubdtype(new_obj.data.dtype, np.unsignedinteger):
                new_obj.data[obj.data < obj2.data] = 0

        name = _("QuadDiff") if quadratic else _("Diff")
        obj2_name = _("object to subtract")
        self.compute_n1n(name, obj2, obj2_name, func=func, func_obj=func_obj)

    @qt_try_except()
    def compute_division(self, obj2: Optional[Obj] = None):
        """Compute division"""
        self.compute_n1n(
            _("Div"),
            obj2,
            _("divider"),
            func=lambda d1, d2: d1 / np.array(d2, dtype=d1.dtype),
        )

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
            roieditordata = self.edit_regions_of_interest(
                extract=True, singleobj=singleobj
            )
            if roieditordata is not None and roieditordata.roidata is None:
                # This only happens in unattended mode (forcing QDialog accept)
                return None
        else:
            roieditordata = ROIEditorData(roidata=roidata, singleobj=singleobj)
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

    # ------Data Processing-------------------------------------------------------------

    @abc.abstractmethod
    @qt_try_except()
    def compute_calibration(self, param=None) -> None:
        """Compute data linear calibration"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_threshold(self, param: ThresholdParam = None) -> None:
        """Compute threshold clipping"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_clip(self, param: ClipParam = None) -> None:
        """Compute maximum data clipping"""

    @staticmethod
    @abc.abstractmethod
    def func_gaussian_filter(x, y, p):
        """Compute gaussian filter"""

    @qt_try_except()
    def compute_gaussian(self, param: GaussianParam = None) -> None:
        """Compute gaussian filter"""
        edit, param = self.init_param(param, GaussianParam, _("Gaussian filter"))
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
        edit, param = self.init_param(param, MovingAverageParam, _("Moving average"))
        func = self.func_moving_average
        self.compute_11("MovAvg", func, param, suffix=lambda p: f"n={p.n}", edit=edit)

    @staticmethod
    @abc.abstractmethod
    def func_moving_median(x, y, p):
        """Moving median computing function"""

    @qt_try_except()
    def compute_moving_median(self, param: MovingMedianParam = None) -> None:
        """Compute moving median"""
        edit, param = self.init_param(param, MovingMedianParam, _("Moving median"))
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

    # ------Computing-------------------------------------------------------------------

    def edit_regions_of_interest(
        self, extract: bool = False, singleobj: bool = None
    ) -> ROIEditorData:
        """Define Region Of Interest (ROI) for computing functions"""
        roieditordata = self.panel.get_roi_dialog(extract=extract, singleobj=singleobj)
        if roieditordata is not None:
            obj = self.panel.objview.get_sel_objects()[0]
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
                    self.SIG_ADD_SHAPE.emit(obj.uuid)
                    self.panel.selection_changed()
                    self.panel.SIG_UPDATE_PLOT_ITEMS.emit()
        return roieditordata

    def delete_regions_of_interest(self):
        """Delete Regions Of Interest"""
        for obj in self.panel.objview.get_sel_objects():
            if obj.roi is not None:
                obj.roi = None
                self.panel.selection_changed()
                self.panel.SIG_UPDATE_PLOT_ITEMS.emit()

    @abc.abstractmethod
    def _get_stat_funcs(self) -> List[Tuple[str, Callable[[np.ndarray], float]]]:
        """Return statistics functions list"""

    @qt_try_except()
    def compute_stats(self):
        """Compute data statistics"""
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        stfuncs = self._get_stat_funcs()
        xlabels = [label for label, _func in stfuncs]
        ylabels: List[str] = []
        stats: List[List[float]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with np.errstate(all="ignore"):
                for obj in objs:
                    roi_nb = 0 if obj.roi is None else obj.roi.shape[0]
                    for roi_index in [None] + list(range(roi_nb)):
                        stats_i = []
                        for _label, func in stfuncs:
                            stats_i.append(func(obj.get_data(roi_index=roi_index)))
                        stats.append(stats_i)
                        if roi_index is None:
                            ylabels.append(obj.short_id)
                        else:
                            ylabels.append(f"{obj.short_id}|ROI{roi_index:02d}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dlg = ArrayEditor(self.panel.parent())
            title = _("Statistics")
            dlg.setup_and_check(
                np.array(stats), title, readonly=True, xlabels=xlabels, ylabels=ylabels
            )
            dlg.setObjectName(f"{objs[0].PREFIX}_stats")
            dlg.setWindowIcon(get_icon("stats.svg"))
            dlg.resize(750, 300)
            exec_dialog(dlg)
