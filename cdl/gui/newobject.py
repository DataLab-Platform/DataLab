# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New object creation GUI
=======================

This module provides a graphical user interface (GUI) for creating new objects
in the DataLab environment. It allows users to create new signals and images
interactively.

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Union

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import exec_dialog
from plotpy.builder import make
from plotpy.plot import PlotDialog
from plotpy.tools import EditPointTool
from qtpy import QtWidgets as QW

from cdl.config import _
from sigima_.model import NormalRandomParam, UniformRandomParam
from sigima_.model.base import Choices
from sigima_.model.image import (
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageTypes,
    NewImageParam,
    get_next_image_number,
)
from sigima_.model.image import create_image_from_param as create_image_headless
from sigima_.model.signal import (
    DEFAULT_TITLE,
    ExponentialParam,
    GaussLorentzVoigtParam,
    PeriodicParam,
    PolyParam,
    PulseParam,
    SignalObj,
    SignalTypes,
    StepParam,
    create_signal,
    get_next_signal_number,
)
from sigima_.model.signal import NewSignalParam as OrigNewSignalParam
from sigima_.model.signal import create_signal_from_param as create_signal_headless


class ExtendedSignalTypes(Choices):
    """Signal types"""

    ZEROS = SignalTypes.ZEROS
    GAUSS = SignalTypes.GAUSS
    LORENTZ = SignalTypes.LORENTZ
    VOIGT = SignalTypes.VOIGT
    UNIFORMRANDOM = SignalTypes.UNIFORMRANDOM
    NORMALRANDOM = SignalTypes.NORMALRANDOM
    SINUS = SignalTypes.SINUS
    COSINUS = SignalTypes.COSINUS
    SAWTOOTH = SignalTypes.SAWTOOTH
    TRIANGLE = SignalTypes.TRIANGLE
    SQUARE = SignalTypes.SQUARE
    SINC = SignalTypes.SINC
    STEP = SignalTypes.STEP
    EXPONENTIAL = SignalTypes.EXPONENTIAL
    PULSE = SignalTypes.PULSE
    POLYNOMIAL = SignalTypes.POLYNOMIAL
    #: Experimental function
    EXPERIMENTAL = _("experimental")


class ExperSignalParam(gds.DataSet):
    """Parameters for experimental signal"""

    size = gds.IntItem("Size", default=5).set_prop("display", hide=True)
    xyarray = gds.FloatArrayItem(
        "XY Values",
        format="%g",
    )
    xmin = gds.FloatItem("Min", default=0).set_prop("display", hide=True)
    xmax = gds.FloatItem("Max", default=1).set_prop("display", hide=True)

    def edit_curve(self, *args) -> None:  # pylint: disable=unused-argument
        """Edit experimental curve"""
        win: PlotDialog = make.dialog(
            wintitle=_("Select one point then press OK to accept"),
            edit=True,
            type="curve",
        )
        edit_tool = win.manager.add_tool(
            EditPointTool, title=_("Edit experimental curve")
        )
        edit_tool.activate()
        plot = win.manager.get_plot()
        x, y = self.xyarray[:, 0], self.xyarray[:, 1]
        curve = make.mcurve(x, y, "-+")
        plot.add_item(curve)
        plot.set_active_item(curve)

        insert_btn = QW.QPushButton(_("Insert point"), win)
        insert_btn.clicked.connect(edit_tool.trigger_insert_point_at_selection)
        win.button_layout.insertWidget(0, insert_btn)

        exec_dialog(win)

        new_x, new_y = curve.get_data()
        self.xmax = new_x.max()
        self.xmin = new_x.min()
        self.size = new_x.size
        self.xyarray = np.vstack((new_x, new_y)).T

    btn_curve_edit = gds.ButtonItem(
        "Edit curve", callback=edit_curve, icon="signal.svg"
    )

    def setup_array(
        self,
        size: int | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> None:
        """Setup the xyarray from size, xmin and xmax (use the current values is not
        provided)

        Args:
            size: xyarray size (default: None)
            xmin: X min (default: None)
            xmax: X max (default: None)
        """
        self.size = size or self.size
        self.xmin = xmin or self.xmin
        self.xmax = xmax or self.xmax
        x_arr = np.linspace(self.xmin, self.xmax, self.size)  # type: ignore
        self.xyarray = np.vstack((x_arr, x_arr)).T


class NewSignalParam(OrigNewSignalParam):
    """New signal dataset"""

    stype = gds.ChoiceItem(_("Type"), ExtendedSignalTypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_signal_type")
    )


def create_signal_gui(
    base_param: NewSignalParam | None = None,
    extra_param: gds.DataSet | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> SignalObj | None:
    """Create a new Signal object from GUI

    Args:
        base_param: base signal parameters (NewSignalParam)
        extra_param: additional parameters (e.g., GaussLorentzVoigtParam,
         PeriodicParam, etc.)
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Signal object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if base_param is None:
        base_param = NewSignalParam()
        if not base_param.edit(parent=parent):
            return None
    elif edit:
        if not base_param.edit(parent=parent):
            return None

    incr_sig_nb = not base_param.title
    if incr_sig_nb:
        base_param.title = f"{base_param.title} {get_next_signal_number():d}"

    if base_param.stype == ExtendedSignalTypes.EXPERIMENTAL:
        p2 = ExperSignalParam("Experimental points")
        p2.setup_array(size=base_param.size, xmin=base_param.xmin, xmax=base_param.xmax)
        if edit and not p2.edit(parent=parent):
            return None
        signal = create_signal(base_param.title)
        signal.xydata = p2.xyarray.T
        if signal.title == DEFAULT_TITLE:
            signal.title = f"experimental(npts={p2.size})"
        return signal

    param_classes = {
        ExtendedSignalTypes.UNIFORMRANDOM: UniformRandomParam,
        ExtendedSignalTypes.NORMALRANDOM: NormalRandomParam,
        ExtendedSignalTypes.GAUSS: GaussLorentzVoigtParam,
        ExtendedSignalTypes.LORENTZ: GaussLorentzVoigtParam,
        ExtendedSignalTypes.VOIGT: GaussLorentzVoigtParam,
        ExtendedSignalTypes.SINUS: PeriodicParam,
        ExtendedSignalTypes.COSINUS: PeriodicParam,
        ExtendedSignalTypes.SAWTOOTH: PeriodicParam,
        ExtendedSignalTypes.TRIANGLE: PeriodicParam,
        ExtendedSignalTypes.SQUARE: PeriodicParam,
        ExtendedSignalTypes.SINC: PeriodicParam,
        ExtendedSignalTypes.STEP: StepParam,
        ExtendedSignalTypes.EXPONENTIAL: ExponentialParam,
        ExtendedSignalTypes.PULSE: PulseParam,
        ExtendedSignalTypes.POLYNOMIAL: PolyParam,
    }
    if base_param.stype in param_classes:
        if extra_param is None:
            extra_param = param_classes[base_param.stype]()
        if edit and not extra_param.edit(parent=parent):
            return None

    try:
        signal = create_signal_headless(base_param, extra_param)
    except Exception as exc:
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError("Error creating signal: {exc}") from exc
        return None

    return signal


def create_image_gui(
    base_param: NewImageParam | None = None,
    extra_param: gds.DataSet | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> ImageObj | None:
    """Create a new Image object from GUI

    Args:
        base_param: base image parameters (NewImageParam)
        extra_param: additional parameters (e.g., Gauss2DParam, etc.)
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Image object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if base_param is None:
        base_param = NewImageParam()
        if not base_param.edit(parent=parent):
            return None
    elif edit:
        if not base_param.edit(parent=parent):
            return None

    if base_param.height is None:
        base_param.height = 500
    if base_param.width is None:
        base_param.width = 500
    if base_param.dtype is None:
        base_param.dtype = ImageDatatypes.UINT16

    incr_img_nb = not base_param.title
    if incr_img_nb:
        base_param.title = f"{base_param.title} {get_next_image_number()}"

    ep = extra_param
    if base_param.itype == ImageTypes.GAUSS:
        if ep is None:
            ep = Gauss2DParam("2D-gaussian image")
        if ep.a is None:
            try:
                ep.a = np.iinfo(base_param.dtype.value).max / 2.0
            except ValueError:
                ep.a = 10.0
        if edit:
            if not ep.edit(parent=parent):
                return None
        extra_param = ep

    elif base_param.itype in (ImageTypes.UNIFORMRANDOM, ImageTypes.NORMALRANDOM):
        param_classes = {
            ImageTypes.UNIFORMRANDOM: UniformRandomParam,
            ImageTypes.NORMALRANDOM: NormalRandomParam,
        }
        if ep is None:
            ep = param_classes[base_param.itype]("Image - " + base_param.itype.value)
            ep: Union[UniformRandomParam | NormalRandomParam]
            ep.set_from_datatype(base_param.dtype.value)
        if edit and not ep.edit(parent=parent):
            return None

    try:
        image = create_image_headless(base_param, ep)
    except Exception as exc:
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError("Error creating image: {exc}") from exc
        return None

    return image
