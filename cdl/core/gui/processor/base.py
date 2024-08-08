# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Base processor object (see parent package :mod:`cdl.core.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import abc
import multiprocessing
import time
import warnings
from collections.abc import Callable
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING, Any, Union

import guidata.dataset as gds
import numpy as np
from guidata.dataset import update_dataset
from guidata.qthelpers import exec_dialog
from guidata.widgets.arrayeditor import ArrayEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl import env
from cdl.algorithms.datatypes import is_complex_dtype
from cdl.computation.base import ROIDataParam
from cdl.config import Conf, _
from cdl.core.gui.processor.catcher import CompOut, wng_err_func
from cdl.core.model.base import ResultProperties, ResultShape
from cdl.utils.qthelpers import create_progress_bar, qt_try_except
from cdl.widgets.warningerror import show_warning_error

if TYPE_CHECKING:
    from multiprocessing.pool import AsyncResult

    from plotpy.plot import PlotWidget

    from cdl.computation.base import (
        ArithmeticParam,
        ClipParam,
        ConstantParam,
        GaussianParam,
        MovingAverageParam,
        MovingMedianParam,
        NormalizeParam,
    )
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj

    Obj = Union[SignalObj, ImageObj]


# Enable multiprocessing support for Windows, with frozen executable (e.g. PyInstaller)
multiprocessing.freeze_support()

# Set start method to 'spawn' for Linux (default is 'fork' which is not safe here
# because of the use of Qt and multithreading) - for other OS, the default is
# 'spawn' anyway
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # This exception is raised if the method is already set (this may happen because
    # this module is imported more than once, e.g. when running tests)
    pass


COMPUTATION_TIP = _(
    "DataLab relies on various libraries to perform the computation. During the "
    "computation, errors may occur because of the data (e.g. division by zero, "
    "unexpected data type, etc.) or because of the libraries (e.g. memory error, "
    "etc.). If you encounter an error, before reporting it, please ensure that "
    "the computation is correct, by checking the data and the parameters."
)


POOL: Pool | None = None


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
            wait: wait for all tasks to finish. Defaults to False.
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
            func: function to run
            args: arguments
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


def is_pairwise_mode() -> bool:
    """Return True if operation mode is pairwise.

    Returns:
        bool: True if operation mode is pairwise
    """
    state = Conf.proc.operation_mode.get() == "pairwise"
    return state


class BaseProcessor(QC.QObject):
    """Object handling data processing: operations, processing, analysis.

    Args:
        panel: panel
        plotwidget: plot widget
    """

    SIG_ADD_SHAPE = QC.Signal(str)
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
            enabled: enabled
        """
        if enabled:
            if self.worker is None:
                self.worker = Worker()
                self.worker.create_pool()
        else:
            if self.worker is not None:
                self.worker.terminate_pool()
                self.worker = None

    def has_param_defaults(self, paramclass: type[gds.DataSet]) -> bool:
        """Return True if parameter defaults are available.

        Args:
            paramclass: parameter class

        Returns:
            bool: True if parameter defaults are available
        """
        return paramclass.__name__ in self.PARAM_DEFAULTS

    def update_param_defaults(self, param: gds.DataSet) -> None:
        """Update parameter defaults.

        Args:
            param: parameters
        """
        key = param.__class__.__name__
        pdefaults = self.PARAM_DEFAULTS.get(key)
        if pdefaults is not None:
            update_dataset(param, pdefaults)
        self.PARAM_DEFAULTS[key] = param

    def init_param(
        self,
        param: gds.DataSet,
        paramclass: type[gds.DataSet],
        title: str,
        comment: str | None = None,
    ) -> tuple[bool, gds.DataSet]:
        """Initialize processing parameters.

        Args:
            param: parameter
            paramclass: parameter class
            title: title
            comment: comment

        Returns:
            Tuple (edit, param) where edit is True if parameters have been edited,
            False otherwise.
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
    ) -> None:
        """Compute 11 function: 1 object in → 1 object out.

        Args:
            func: function
            param: parameter
            paramclass: parameter class
            title: title
            comment: comment
            edit: edit parameters
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
    ) -> None:
        """Compute 1n function: 1 object in → n objects out.

        Args:
            funcs: list of functions
            params: list of parameters
            title: title
            edit: edit parameters
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
        self, compout: CompOut, context: str, progress: QW.QProgressDialog
    ) -> SignalObj | ImageObj | ResultShape | ResultProperties | None:
        """Handle computation output: if error, display error message,
        if warning, display warning message.

        Args:
            compout: computation output
            context: context (e.g. "Computing: Gaussian filter")
            progress: progress dialog

        Returns:
            Output object: a signal or image object, or a result shape object,
             or None if error
        """
        if compout.error_msg or compout.warning_msg:
            mindur = progress.minimumDuration()
            progress.setMinimumDuration(1000000)
            if compout.error_msg:
                show_warning_error(
                    self.panel, "error", context, compout.error_msg, COMPUTATION_TIP
                )
            if compout.warning_msg:
                show_warning_error(self.panel, "warning", context, compout.warning_msg)
            progress.setMinimumDuration(mindur)
            if compout.error_msg:
                return None
        return compout.result

    def __exec_func(
        self,
        func: Callable,
        args: tuple,
        progress: QW.QProgressDialog,
    ) -> CompOut | None:
        """Execute function, eventually in a separate process.

        Args:
            func: function to execute
            args: function arguments
            progress: progress dialog

        Returns:
            Computation output object or None if canceled
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

    def _compute_11_subroutine(
        self, funcs: list[Callable], params: list, title: str
    ) -> None:
        """Compute 11 subroutine: used by compute 11 and compute 1n methods.

        Args:
            funcs: list of functions to execute
            params: list of parameters
            title: title of progress bar
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
                    new_obj = self.handle_output(
                        result, _("Computing: %s") % i_title, progress
                    )
                    if new_obj is None:
                        continue

                    # Is new object a native object (i.e. a Signal object for a Signal
                    # Panel, or an Image object for an Image Panel) ?
                    # (example of non-native object use case: image profile extraction)
                    is_new_obj_native = isinstance(new_obj, self.panel.PARAMCLASS)

                    new_gid = None
                    if grps and is_new_obj_native:
                        # If groups are selected, then it means that there is no
                        # individual object selected: we work on groups only
                        old_gid = self.panel.objmodel.get_object_group_id(obj)
                        new_gid = new_gids.get(old_gid)
                        if new_gid is None:
                            # Create a new group for each selected group
                            old_g = self.panel.objmodel.get_group(old_gid)
                            new_g = self.panel.add_group(f"{name}({old_g.short_id})")
                            new_gids[old_gid] = new_gid = new_g.uuid
                    if is_new_obj_native:
                        self.panel.add_object(new_obj, group_id=new_gid)
                    else:
                        self.panel.mainwindow.add_object(new_obj)
        # Select newly created groups, if any
        for group_id in new_gids.values():
            self.panel.objview.set_current_item_id(group_id, extend=True)

    def compute_10(
        self,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> dict[str, ResultShape | ResultProperties]:
        """Compute 10 function: 1 object in → 0 object out
        (the result of this method is stored in original object's metadata).

        Args:
            func: function to execute
            param: parameters. Defaults to None.
            paramclass: parameters class. Defaults to None.
            title: title of progress bar. Defaults to None.
            comment: comment. Defaults to None.
            edit: if True, edit parameters. Defaults to None.

        Returns:
            Dictionary of results (keys: object uuid, values: ResultShape or
             ResultProperties objects)
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return None
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        current_obj = self.panel.objview.get_current_object()
        title = func.__name__.replace("compute_", "") if title is None else title
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            results: dict[str, ResultShape | ResultProperties] = {}
            xlabels = None
            ylabels = []
            for idx, obj in enumerate(objs):
                pvalue = idx + 1
                pvalue = 0 if pvalue == 1 else pvalue
                progress.setValue(pvalue)
                args = (obj,) if param is None else (obj, param)

                # Execute function
                compout = self.__exec_func(func, args, progress)
                if compout is None:
                    break
                result = self.handle_output(
                    compout, _("Computing: %s") % title, progress
                )
                if result is None:
                    continue

                # Add result shape to object's metadata
                result.add_to(obj)
                if param is not None:
                    obj.metadata[f"{result.title}Param"] = str(param)

                results[obj.uuid] = result
                xlabels = result.headers
                if obj is current_obj:
                    self.panel.selection_changed(update_items=True)
                else:
                    self.panel.SIG_REFRESH_PLOT.emit(obj.uuid, True)
                for i_row_res in range(result.array.shape[0]):
                    ylabel = f"{result.title}({obj.short_id})"
                    i_roi = int(result.array[i_row_res, 0])
                    if i_roi >= 0:
                        ylabel += f"|ROI{i_roi}"
                    ylabels.append(ylabel)
        if results:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dlg = ArrayEditor(self.panel.parent())
                title = _("Results")
                res = np.vstack([result.shown_array for result in results.values()])
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
    ) -> None:
        """Compute n1 function: N(>=2) objects in → 1 object out.

        Args:
            name: name of function
            func: function to execute
            param: parameters. Defaults to None.
            paramclass: parameters class. Defaults to None.
            title: title of progress bar. Defaults to None.
            comment: comment. Defaults to None.
            func_objs: function to execute on objects. Defaults to None.
            edit: if True, edit parameters. Defaults to None.
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)
        if param is not None:
            if edit and not param.edit(parent=self.panel.parent()):
                return

        objs = self.panel.objview.get_sel_objects(include_groups=True)

        pairwise = is_pairwise_mode()

        if pairwise:
            # In pairwise mode, we need to create a new object for each pair of objects
            src_gids = list(
                set([self.panel.objmodel.get_object_group_id(obj) for obj in objs])
            )
            if len(src_gids) == 1:
                # In pairwise mode, we need at least two objects from different groups
                return
            # [src_objs dictionary] keys: old group id, values: list of old objects
            src_objs: dict[str, list[Obj]] = {}
            for src_gid in src_gids:
                src_objs[src_gid] = [
                    obj
                    for obj in objs
                    if self.panel.objmodel.get_object_group_id(obj) == src_gid
                ]
            grps = [self.panel.objmodel.get_group(gid) for gid in src_gids]
            dst_gname = f"{name}({','.join([grp.short_id for grp in grps])})|pairwise"
            group_exclusive = len(self.panel.objview.get_sel_groups()) != 0
            if not group_exclusive:
                # This is not a group exclusive selection
                dst_gname += "[...]"
            dst_gid = self.panel.add_group(dst_gname).uuid
            n_pairs = len(src_objs[src_gids[0]])
            max_i_pair = min(n_pairs, max(len(src_objs[grp.uuid]) for grp in grps))
            with create_progress_bar(self.panel, title, max_=n_pairs) as progress:
                for i_pair, src_obj1 in enumerate(src_objs[src_gids[0]][:max_i_pair]):
                    src_obj1: SignalObj | ImageObj
                    progress.setValue(i_pair + 1)
                    progress.setLabelText(title)
                    src_dtype = src_obj1.data.dtype
                    dst_dtype = complex if is_complex_dtype(src_dtype) else float
                    dst_obj = src_obj1.copy(dtype=dst_dtype)
                    src_objs_pair = [src_obj1]
                    for src_gid in src_gids[1:]:
                        src_obj = src_objs[src_gid][i_pair]
                        src_objs_pair.append(src_obj)
                        if param is None:
                            args = (dst_obj, src_obj)
                        else:
                            args = (dst_obj, src_obj, param)
                        result = self.__exec_func(func, args, progress)
                        if result is None:
                            break
                        dst_obj = self.handle_output(
                            result, _("Calculating: %s") % title, progress
                        )
                        if dst_obj is None:
                            break
                        dst_obj.update_resultshapes_from(src_obj)
                        if src_obj.roi is not None:
                            if dst_obj.roi is None:
                                dst_obj.roi = src_obj.roi.copy()
                            else:
                                dst_obj.roi = np.vstack((dst_obj.roi, src_obj.roi))
                    if func_objs is not None:
                        func_objs(dst_obj, src_objs_pair)
                    short_ids = [obj.short_id for obj in src_objs_pair]
                    dst_obj.title = f'{name}({", ".join(short_ids)})'
                    self.panel.add_object(dst_obj, group_id=dst_gid)

        else:
            # In single operand mode, we create a single object for all selected objects

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
                        dst_dtype = complex if is_complex_dtype(src_dtype) else float
                        dst_objs[src_gid] = dst_obj = src_obj.copy(dtype=dst_dtype)
                        dst_obj.roi = None
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
                        dst_obj = self.handle_output(
                            result, _("Calculating: %s") % title, progress
                        )
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
        obj2: Obj | list[Obj] | None,
        obj2_name: str,
        func: Callable,
        param: gds.DataSet | None = None,
        paramclass: gds.DataSet | None = None,
        title: str | None = None,
        comment: str | None = None,
        edit: bool | None = None,
    ) -> None:
        """Compute n1n function: N(>=1) objects + 1 object in → N objects out.

        Examples: subtract, divide

        Args:
            obj2: second object (or list of objects in case of pairwise operation mode)
            obj2_name: name of second object
            func: function to execute
            param: parameters. Defaults to None.
            paramclass: parameters class. Defaults to None.
            title: title of progress bar. Defaults to None.
            comment: comment. Defaults to None.
            edit: if True, edit parameters. Defaults to None.
        """
        if (edit is None or param is None) and paramclass is not None:
            edit, param = self.init_param(param, paramclass, title, comment)

        objs = self.panel.objview.get_sel_objects(include_groups=True)

        pairwise = is_pairwise_mode()

        if obj2 is None:
            objs2 = []
        elif isinstance(obj2, list):
            objs2 = obj2
            assert pairwise
        else:
            objs2 = [obj2]

        if not objs2:
            nb_objects = len(objs) if pairwise else 1
            dlg_title = _("Select %s") % obj2_name
            if pairwise:
                dlg_comm = (
                    f"<u>Note:</u> {_('operation mode is <i>pairwise</i>: ')} "
                    f"{nb_objects} object(s) expected (i.e. as many as input objects)"
                )
            else:
                dlg_comm = _(
                    "<u>Note:</u> operation mode is <i>single operand</i>: "
                    "1 object expected"
                )
            objs2 = self.panel.get_objects_with_dialog(dlg_title, dlg_comm, nb_objects)
            if objs2 is None:
                return

        if pairwise:
            group_exclusive = len(self.panel.objview.get_sel_groups()) != 0
            src_gids = list(
                set([self.panel.objmodel.get_object_group_id(obj) for obj in objs])
            )
            src_objs: dict[str, list[Obj]] = {}
            for src_gid in src_gids:
                src_objs[src_gid] = [
                    obj
                    for obj in objs
                    if self.panel.objmodel.get_object_group_id(obj) == src_gid
                ]
            grps = [self.panel.objmodel.get_group(gid) for gid in src_gids]
            name = func.__name__.replace("compute_", "")
            n_pairs = len(src_objs[src_gids[0]])
            max_i_pair = min(n_pairs, max(len(src_objs[grp.uuid]) for grp in grps))
            grp2_id = self.panel.objmodel.get_object_group_id(objs2[0])
            grp2 = self.panel.objmodel.get_group(grp2_id)
            with create_progress_bar(self.panel, title, max_=len(src_gids)) as progress:
                for i_group, src_gid in enumerate(src_gids):
                    progress.setValue(i_group + 1)
                    progress.setLabelText(title)
                    if group_exclusive:
                        # This is a group exclusive selection
                        src_grp = self.panel.objmodel.get_group(src_gid)
                        grp_short_ids = [grp.short_id for grp in (src_grp, grp2)]
                        dst_gname = f"{name}({','.join(grp_short_ids)})|pairwise"
                    else:
                        dst_gname = f"{name}[...]"
                    dst_gid = self.panel.add_group(dst_gname).uuid
                    for i_pair in range(max_i_pair):
                        args = [src_objs[src_gids[i_group]][i_pair], objs2[i_pair]]
                        if param is not None:
                            args.append(param)
                        result = self.__exec_func(func, tuple(args), progress)
                        if result is None:
                            break
                        new_obj = self.handle_output(
                            result, _("Calculating: %s") % title, progress
                        )
                        if new_obj is None:
                            continue
                        self.panel.add_object(new_obj, group_id=dst_gid)

        else:
            obj2 = objs2[0]
            with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
                for index, obj in enumerate(objs):
                    progress.setValue(index + 1)
                    progress.setLabelText(title)
                    args = (obj, obj2) if param is None else (obj, obj2, param)
                    result = self.__exec_func(func, args, progress)
                    if result is None:
                        break
                    new_obj = self.handle_output(
                        result, _("Calculating: %s") % title, progress
                    )
                    if new_obj is None:
                        continue
                    group_id = self.panel.objmodel.get_object_group_id(obj)
                    self.panel.add_object(new_obj, group_id=group_id)

    # ------Data Operations-------------------------------------------------------------

    @abc.abstractmethod
    @qt_try_except()
    def compute_arithmetic(self, param: ArithmeticParam | None = None) -> None:
        """Compute arithmetic operation"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_sum(self) -> None:
        """Compute sum"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_normalize(self, param: NormalizeParam | None = None) -> None:
        """Normalize data"""

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
    def compute_difference(self, obj2: Obj | list[Obj] | None = None) -> None:
        """Compute difference"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_quadratic_difference(self, obj2: Obj | list[Obj] | None = None) -> None:
        """Compute quadratic difference"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_division(self, obj2: Obj | list[Obj] | None = None) -> None:
        """Compute division"""

    def _get_roidataparam(self, param: ROIDataParam | None = None) -> ROIDataParam:
        """Eventually open ROI Editing Dialog, and return ROI editor data.

        Args:
            param: ROI data parameters.
                Defaults to None.

        Returns:
            ROI data parameters.
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
    def compute_re(self) -> None:
        """Compute real part"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_im(self) -> None:
        """Compute imaginary part"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_astype(self) -> None:
        """Convert data type"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_log10(self) -> None:
        """Compute Log10"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_exp(self) -> None:
        """Compute exponential"""

    # ------Data Processing-------------------------------------------------------------

    @abc.abstractmethod
    @qt_try_except()
    def compute_calibration(self, param=None) -> None:
        """Compute data linear calibration"""

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

    @abc.abstractmethod
    @qt_try_except()
    def compute_addition_constant(self, param: ConstantParam) -> None:
        """Compute sum with a constant"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_difference_constant(self, param: ConstantParam) -> None:
        """Compute difference with a constant"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_product_constant(self, param: ConstantParam) -> None:
        """Compute product with a constant"""

    @abc.abstractmethod
    @qt_try_except()
    def compute_division_constant(self, param: ConstantParam) -> None:
        """Compute division by a constant"""

    # ------Analysis-------------------------------------------------------------------

    def edit_regions_of_interest(
        self,
        extract: bool = False,
        singleobj: bool | None = None,
        add_roi: bool = False,
    ) -> ROIDataParam | None:
        """Define Region Of Interest (ROI).

        Args:
            extract: If True, ROI is extracted from data. Defaults to False.
            singleobj: If True, ROI is extracted from first selected object only.
             If False, ROI is extracted from all selected objects. If None, ROI is
             extracted from all selected objects only if they all have the same ROI.
             Defaults to None.
            add_roi: If True, add ROI to data immediately after opening the ROI editor.
             Defaults to False.

        Returns:
            ROI data parameters or None if ROI dialog has been canceled.
        """
        results = self.panel.get_roi_editor_output(
            extract=extract, singleobj=singleobj, add_roi=add_roi
        )
        if results is None:
            return None
        roieditordata, modified = results
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        roigroup = obj.roidata_to_params(roieditordata.roidata)
        if (
            env.execenv.unattended
            or roieditordata.roidata.size == 0
            or roigroup.edit(parent=self.panel.parent())
        ):
            roidata = obj.params_to_roidata(roigroup)
            if modified:
                roieditordata.roidata = roidata
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
    @qt_try_except()
    def compute_stats(self) -> dict[str, ResultShape]:
        """Compute data statistics"""
