# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Base Processor GUI module
---------------------------------

This module defines the base class for data processing GUIs.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import multiprocessing
import time
import warnings
from collections.abc import Callable
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, Union

import guidata.dataset as gds
import numpy as np
from guidata.configtools import get_icon
from guidata.dataset import update_dataset
from guidata.qthelpers import exec_dialog
from guidata.widgets.arrayeditor import ArrayEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl import env
from cdl.config import Conf, _
from cdl.core.computation.base import ROIDataParam
from cdl.core.gui.processor.catcher import CompOut, wng_err_func
from cdl.core.model.base import ResultShape, ShapeTypes
from cdl.utils import misc
from cdl.utils.qthelpers import create_progress_bar, qt_try_except
from cdl.widgets.warningerror import show_warning_error

if TYPE_CHECKING:  # pragma: no cover
    from multiprocessing.pool import AsyncResult

    from plotpy.plot import PlotWidget

    from cdl.core.computation.base import (
        ClipParam,
        GaussianParam,
        MovingAverageParam,
        MovingMedianParam,
        ThresholdParam,
    )
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj

    Obj = Union[SignalObj, ImageObj]


# Enable multiprocessing support for Windows, with frozen executable (e.g. PyInstaller)
multiprocessing.freeze_support()


COMPUTATION_TIP = _(
    "DataLab relies on various libraries to perform the computation. During the "
    "computation, errors may occur because of the data (e.g. division by zero, "
    "unexpected data type, etc.) or because of the libraries (e.g. memory error, "
    "etc.). If you encounter an error, before reporting it, please ensure that "
    "the computation is correct, by checking the data and the parameters."
)


POOL: Pool = None


class Worker:
    """Multiprocessing worker, to run long-running tasks in a separate process"""

    def __init__(self) -> None:
        self.asyncresult: AsyncResult = None
        self.result: Any = None

    @staticmethod
    def create_pool() -> None:
        """Create multiprocessing pool"""
        global POOL  # pylint: disable=global-statement
        # Create a pool with one process
        POOL = Pool(processes=1)  # pylint: disable=not-callable,consider-using-with

    @staticmethod
    def terminate_pool(wait: bool = False) -> None:
        """Terminate multiprocessing pool.

        Args:
            wait (bool | None): wait for all tasks to finish. Defaults to False.
        """
        global POOL  # pylint: disable=global-statement
        if POOL is not None:
            if wait:
                # Close the pool properly (wait for all tasks to finish)
                POOL.close()
            else:
                # Terminate the pool and stop the timer
                POOL.terminate()
            POOL.join()
            POOL = None

    def restart_pool(self) -> None:
        """Terminate and recreate the pool"""
        # Terminate the process and stop the timer
        self.terminate_pool(wait=False)
        # Recreate the pool for the next computation
        self.create_pool()

    def run(self, func: Callable, args: tuple[Any]) -> None:
        """Run computation.

        Args:
            func (Callable): function to run
            args (tuple[Any]): arguments
        """
        global POOL  # pylint: disable=global-statement,global-variable-not-assigned
        assert POOL is not None
        self.asyncresult = POOL.apply_async(wng_err_func, (func, args))

    def close(self) -> None:
        """Close worker: close pool properly and wait for all tasks to finish"""
        # Close multiprocessing Pool properly, but only if no computation is running,
        # to avoid blocking the GUI at exit (so, when wait=True, we wait for the
        # task to finish before closing the pool but there is actually no task running,
        # so the pool is closed immediately but *properly*)
        self.terminate_pool(wait=self.asyncresult is None)

    def is_computation_finished(self) -> bool:
        """Return True if computation is finished.

        Returns:
            bool: True if computation is finished
        """
        return self.asyncresult.ready()

    def get_result(self) -> CompOut:
        """Return computation result.

        Returns:
            CompOut: computation result
        """
        self.result = self.asyncresult.get()
        self.asyncresult = None
        return self.result


class BaseProcessor(QC.QObject):
    """Object handling data processing: operations, processing, computing.

    Args:
        panel (SignalPanel | ImagePanel): panel
        plotwidget (CurveWidget | ImageWidget): plot widget
    """

    SIG_ADD_SHAPE = QC.Signal(str)
    EDIT_ROI_PARAMS = False
    PARAM_DEFAULTS: dict[str, gds.DataSet] = {}

    def __init__(self, panel: SignalPanel | ImagePanel, plotwidget: PlotWidget):
        super().__init__()
        self.panel = panel
        self.plotwidget = plotwidget
        self.worker: Worker | None = None
        self.set_process_isolation_enabled(Conf.main.process_isolation_enabled.get())

    def close(self):
        """Close processor properly"""
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def set_process_isolation_enabled(self, enabled: bool) -> None:
        """Set process isolation enabled.

        Args:
            enabled (bool): enabled
        """
        if enabled:
            if self.worker is None:
                self.worker = Worker()
                self.worker.create_pool()
        else:
            if self.worker is not None:
                self.worker.terminate_pool()
                self.worker = None

    def update_param_defaults(self, param: gds.DataSet) -> None:
        """Update parameter defaults.

        Args:
            param (gds.DataSet): parameter
        """
        key = param.__class__.__name__
        pdefaults = self.PARAM_DEFAULTS.get(key)
        if pdefaults is not None:
            update_dataset(param, pdefaults)
        self.PARAM_DEFAULTS[key] = param

    def init_param(
        self,
        param: gds.DataSet,
        paramclass: gds.DataSet,
        title: str,
        comment: str | None = None,
    ) -> tuple[bool, gds.DataSet]:
        """Initialize processing parameters.

        Args:
            param (gds.DataSet): parameter
            paramclass (gds.DataSet): parameter class
            title (str): title
            comment (str | None): comment

        Returns:
            tuple[bool, gds.DataSet]: edit, param
        """
        edit = param is None
        if edit:
            param = paramclass(title, comment)
            self.update_param_defaults(param)
        return edit, param

    def compute_11(
        self,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ):
        """Compute 11 function: 1 object in --> 1 object out.

        Args:
            func (Callable): function
            param (guidata.dataset.DataSet | None): parameter
            paramclass (guidata.dataset.DataSet | None): parameter class
            title (str | None): title
            comment (str | None): comment
            edit (bool | None): edit parameters
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return
        self._compute_11_subroutine([func], [param], title)

    def compute_1n(
        self,
        funcs: list[Callable] | Callable,
        params: list | None = None,
        title: str | None = None,
        edit: bool | None = None,
    ):
        """Compute 1n function: 1 object in --> n objects out.

        Args:
            funcs (list[Callable] | Callable): list of functions
            params (list | None): list of parameters
            title (str | None): title
            edit (bool | None): edit parameters
        """
        if params is None:
            assert not isinstance(funcs, Callable)
            params = [None] * len(funcs)
        else:
            group = gds.DataSetGroup(params, title=_("Parameters"))
            if edit and not group.edit(parent=self.panel.parent()):
                return
            if isinstance(funcs, Callable):
                funcs = [funcs] * len(params)
            else:
                assert len(funcs) == len(params)
        self._compute_11_subroutine(funcs, params, title)

    def handle_output(
        self, compout: CompOut, context: str
    ) -> SignalObj | ImageObj | np.ndarray | None:
        """Handle computation output: if error, display error message,
        if warning, display warning message.

        Args:
            compout (CompOut): computation output
            context (str): context (e.g. "Computing: Gaussian filter")

        Returns:
            SignalObj | ImageObj | np.ndarray | None: output object
            (None if error)
        """
        if compout.error_msg:
            show_warning_error(
                self.panel, "error", context, compout.error_msg, COMPUTATION_TIP
            )
            return None
        if compout.warning_msg:
            show_warning_error(self.panel, "warning", context, compout.warning_msg)
        return compout.result

    def __exec_func(
        self,
        func: Callable,
        args: tuple,
        progress: QW.QProgressDialog,
    ) -> CompOut | None:
        """Execute function, eventually in a separate process.

        Args:
            func (Callable): function to execute
            args (tuple): function arguments
            progress (QW.QProgressDialog): progress dialog

        Returns:
            CompOut | None: computation output
        """
        QW.QApplication.processEvents()
        if not progress.wasCanceled():
            if self.worker is None:
                return wng_err_func(func, args)
            self.worker.run(func, args)
            while not self.worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.1)
                if progress.wasCanceled():
                    self.worker.restart_pool()
                    break
            if self.worker.is_computation_finished():
                return self.worker.get_result()
        return None

    def _compute_11_subroutine(self, funcs: list[Callable], params: list, title: str):
        """Compute 11 subroutine: used by compute 11 and compute 1n methods.

        Args:
            funcs (list[Callable]): list of functions to execute
            params (list): list of parameters
            title (str): title of progress bar
        """
        assert len(funcs) == len(params)
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        grps = self.panel.objview.get_sel_groups()
        new_gids = {}
        with create_progress_bar(
            self.panel, title, max_=len(objs) * len(params)
        ) as progress:
            for i_row, obj in enumerate(objs):
                for i_param, (param, func) in enumerate(zip(params, funcs)):
                    name = func.__name__.replace("compute_", "")
                    i_title = f"{title} ({i_row + 1}/{len(objs)})"
                    progress.setLabelText(i_title)
                    pvalue = (i_row + 1) * (i_param + 1)
                    pvalue = 0 if pvalue == 1 else pvalue
                    progress.setValue(pvalue)
                    args = (obj,) if param is None else (obj, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(result, _("Computing: %s") % i_title)
                    if new_obj is None:
                        continue
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

    def compute_10(
        self,
        func: Callable,
        shapetype: ShapeTypes,
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> dict[int, ResultShape]:
        """Compute 10 function: 1 object in --> 0 object out
        (the result of this method is stored in original object's metadata).

        Args:
            func (Callable): function to execute
            shapetype (ShapeTypes): shape type
            param (guidata.dataset.DataSet | None | None): parameters.
             Defaults to None.
            paramclass (guidata.dataset.DataSet | None | None): parameters
             class. Defaults to None.
            title (str | None | None): title of progress bar.
             Defaults to None.
            comment (str | None | None): comment. Defaults to None.
            edit (bool | None | None): if True, edit parameters.
             Defaults to None.

        Returns:
            dict[int, ResultShape]: dictionary of results
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return None
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        current_obj = self.panel.objview.get_current_object()
        name = func.__name__.replace("compute_", "")
        title = name if title is None else title
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            results: dict[int, ResultShape] = {}
            xlabels = None
            ylabels = []
            for idx, obj in enumerate(objs):
                pvalue = idx + 1
                pvalue = 0 if pvalue == 1 else pvalue
                progress.setValue(pvalue)
                args = (obj,) if param is None else (obj, param)
                result = self.__exec_func(func, args, progress)
                if result is None:
                    break
                result_array = self.handle_output(result, _("Computing: %s") % title)
                if result_array is None:
                    continue
                resultshape = obj.add_resultshape(name, shapetype, result_array, param)
                results[obj.uuid] = resultshape
                xlabels = resultshape.xlabels
                if obj is current_obj:
                    self.panel.selection_changed(update_items=True)
                else:
                    self.panel.SIG_REFRESH_PLOT.emit(obj.uuid, True)
                for _i_row_res in range(resultshape.array.shape[0]):
                    ylabel = f"{name}({obj.short_id})"
                    ylabels.append(ylabel)
        if results:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dlg = ArrayEditor(self.panel.parent())
                title = _("Results")
                res = np.vstack([rshape.array for rshape in results.values()])
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
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        func_objs: Callable | None = None,
        edit: bool | None = None,
    ):
        """Compute n1 function: N(>=2) objects in --> 1 object out.

        Args:
            name (str): name of function
            func (Callable): function to execute
            param (guidata.dataset.DataSet | None | None): parameters.
             Defaults to None.
            paramclass (guidata.dataset.DataSet | None | None):
             parameters class. Defaults to None.
            title (str | None | None): title of progress bar.
             Defaults to None.
            comment (str | None | None): comment. Defaults to None.
            func_objs (Callable | None | None): function to execute on objects.
             Defaults to None.
            edit (bool | None | None): if True, edit parameters.
             Defaults to None.
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)

        # [new_objs dictionary] keys: old group id, values: new object
        dst_objs: dict[str, Obj] = {}
        # [src_dtypes dictionary] keys: old group id, values: old data type
        src_dtypes: dict[str, np.dtype] = {}
        # [src_objs dictionary] keys: old group id, values: list of old objects
        src_objs: dict[str, list[Obj]] = {}

        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            for index, src_obj in enumerate(objs):
                progress.setValue(index + 1)
                progress.setLabelText(title)
                src_gid = self.panel.objmodel.get_object_group_id(src_obj)
                dst_obj = dst_objs.get(src_gid)
                if dst_obj is None:
                    src_dtypes[src_gid] = src_dtype = src_obj.data.dtype
                    dst_dtype = complex if misc.is_complex_dtype(src_dtype) else float
                    dst_objs[src_gid] = dst_obj = src_obj.copy(dtype=dst_dtype)
                    src_objs[src_gid] = [src_obj]
                else:
                    src_objs[src_gid].append(src_obj)
                    if param is None:
                        args = (dst_obj, src_obj)
                    else:
                        args = (dst_obj, src_obj, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    dst_obj = self.handle_output(result, _("Calculating: %s") % title)
                    if dst_obj is None:
                        break
                    dst_objs[src_gid] = dst_obj
                    dst_obj.update_resultshapes_from(src_obj)
                if src_obj.roi is not None:
                    if dst_obj.roi is None:
                        dst_obj.roi = src_obj.roi.copy()
                    else:
                        dst_obj.roi = np.vstack((dst_obj.roi, src_obj.roi))

        grps = self.panel.objview.get_sel_groups()
        if grps:
            # (Group exclusive selection)
            # At least one group is selected: create a new group
            dst_gname = f"{name}({','.join([grp.short_id for grp in grps])})"
            dst_gid = self.panel.add_group(dst_gname).uuid
        else:
            # (Object exclusive selection)
            # No group is selected: use each object's group
            dst_gid = None

        for src_gid, dst_obj in dst_objs.items():
            if misc.is_integer_dtype(src_dtypes[src_gid]):
                dst_obj.set_data_type(dtype=src_dtypes[src_gid])
            if func_objs is not None:
                func_objs(dst_obj, src_objs[src_gid])
            short_ids = [obj.short_id for obj in src_objs[src_gid]]
            dst_obj.title = f'{name}({", ".join(short_ids)})'
            group_id = dst_gid if dst_gid is not None else src_gid
            self.panel.add_object(dst_obj, group_id=group_id)

        # Select newly created group, if any
        if dst_gid is not None:
            self.panel.objview.set_current_item_id(dst_gid)

    def compute_n1n(
        self,
        obj2: Obj | None,
        obj2_name: str,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ):
        """Compute n1n function: N(>=1) objects + 1 object in --> N objects out.

        Examples: subtract, divide

        Args:
            obj2 (Obj | None): second object
            obj2_name (str): name of second object
            func (Callable): function to execute
            param (guidata.dataset.DataSet | None | None): parameters.
             Defaults to None.
            paramclass (guidata.dataset.DataSet | None | None):
             parameters class. Defaults to None.
            title (str | None | None): title of progress bar.
             Defaults to None.
            comment (str | None | None): comment. Defaults to None.
            edit (bool | None | None): if True, edit parameters.
             Defaults to None.
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return
        if obj2 is None:
            obj2 = self.panel.get_object_with_dialog(_("Select %s") % obj2_name)
            if obj2 is None:
                return
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        # name = func.__name__.replace("compute_", "")
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            for index, obj in enumerate(objs):
                progress.setValue(index + 1)
                progress.setLabelText(title)
                args = (obj, obj2) if param is None else (obj, obj2, param)
                result = self.__exec_func(func, args, progress)
                if result is None:
                    break
                new_obj = self.handle_output(result, _("Calculating: %s") % title)
                if new_obj is None:
                    continue
                group_id = self.panel.objmodel.get_object_group_id(obj)
                self.panel.add_object(new_obj, group_id=group_id)

    # ------Data Operations-------------------------------------------------------------

    @abc.abstractmethod
    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_average(self) -> None:
        """Compute average"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_product(self) -> None:
        """Compute product"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_difference(self, obj2: Obj | None = None) -> None:
        """Compute difference"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_quadratic_difference(self, obj2: Obj | None = None) -> None:
        """Compute quadratic difference"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_division(self, obj2: Obj | None = None) -> None:
        """Compute division"""

    def _get_roidataparam(self, param: ROIDataParam | None = None) -> ROIDataParam:
        """Eventually open ROI Editing Dialog, and return ROI editor data.

        Args:
            param (ROIDataParam | None | None): ROI data parameters.
                Defaults to None.

        Returns:
            ROIDataParam: ROI data parameters.
        """
        # Expected behavior:
        # -----------------
        # * If param.roidata argument is not None, skip the ROI dialog
        # * If first selected obj has a ROI, use this ROI as default but open
        #   ROI Editor dialog anyway
        # * If multiple objs are selected, then apply the first obj ROI to all
        if param is None:
            param = ROIDataParam()
        if param.roidata is None:
            param = self.edit_regions_of_interest(
                extract=True, singleobj=param.singleobj
            )
            if param is not None and param.roidata is None:
                # This only happens in unattended mode (forcing QDialog accept)
                return None
        return param

    @abc.abstractmethod
    @qt_try_except()
    def compute_roi_extraction(self, param=None) -> None:
        """Extract Region Of Interest (ROI) from data"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_swap_axes(self) -> None:
        """Swap data axes"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_abs(self) -> None:
        """Compute absolute value"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10"""

    # ------Data Processing-------------------------------------------------------------

    @abc.abstractmethod
    @qt_try_except()
    def compute_calibration(self, param=None) -> None:
        """Compute data linear calibration"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_threshold(self, param: ThresholdParam | None = None) -> None:
        """Compute threshold clipping"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_clip(self, param: ClipParam | None = None) -> None:
        """Compute maximum data clipping"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_gaussian_filter(self, param: GaussianParam | None = None) -> None:
        """Compute gaussian filter"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_moving_average(self, param: MovingAverageParam | None = None) -> None:
        """Compute moving average"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_moving_median(self, param: MovingMedianParam | None = None) -> None:
        """Compute moving median"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_wiener(self) -> None:
        """Compute Wiener filter"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_fft(self) -> None:
        """Compute iFFT"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_ifft(self) -> None:
        """Compute FFT"""

    # ------Computing-------------------------------------------------------------------

    def edit_regions_of_interest(
        self, extract: bool = False, singleobj: bool | None = None
    ) -> ROIDataParam:
        """Define Region Of Interest (ROI) for computing functions.

        Args:
            extract (bool | None): If True, ROI is extracted from data.
                Defaults to False.
            singleobj (bool | None | None): If True, ROI is extracted from
                first selected object only. If False, ROI is extracted from
                all selected objects. If None, ROI is extracted from all
                selected objects only if they all have the same ROI.
                Defaults to None.

        Returns:
            ROIDataParam: ROI data parameters.
        """
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
                    self.panel.selection_changed(update_items=True)
        return roieditordata

    def delete_regions_of_interest(self) -> None:
        """Delete Regions Of Interest"""
        for obj in self.panel.objview.get_sel_objects():
            if obj.roi is not None:
                obj.roi = None
                self.panel.selection_changed(update_items=True)

    @abc.abstractmethod
    def _get_stat_funcs(self) -> list[tuple[str, Callable[[np.ndarray], float]]]:
        """Return statistics functions list"""

    @qt_try_except()
    def compute_stats(self) -> None:
        """Compute data statistics"""
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        stfuncs = self._get_stat_funcs()
        xlabels = [label for label, _func in stfuncs]
        ylabels: list[str] = []
        stats: list[list[float]] = []
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
