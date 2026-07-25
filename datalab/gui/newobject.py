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

import json

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import exec_dialog
from plotpy.builder import make
from plotpy.plot import PlotDialog
from plotpy.tools import EditPointTool
from qtpy import QtWidgets as QW
from sigima.objects import (
    PEAK_PARAMETERIZATION,
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    NewImageParam,
    NewSignalParam,
    SignalObj,
    convert_legacy_peak_creation_params,
    create_signal,
    validate_peak_creation_params,
)
from sigima.objects import CustomSignalParam as OrigCustomSignalParam
from sigima.objects import create_image_from_param as create_image_headless
from sigima.objects import create_signal_from_param as create_signal_headless
from sigima.objects.base import BaseProcParam
from sigima.objects.signal import (
    DEFAULT_TITLE as SIGNAL_DEFAULT_TITLE,
)
from sigima.objects.signal import (
    BaseGaussLorentzVoigtParam,
)

from datalab.config import _

CREATION_PARAMETERS_OPTION = "creation_parameters"
LEGACY_CREATION_PARAMETERS_OPTION = "creation_param_json"
CREATION_PARAMETERS_FORMAT_VERSION = 1


def _decode_dataset_json(dataset_json: str) -> dict[str, object]:
    """Decode a DataSet JSON payload without instantiating its class."""
    try:
        payload = json.loads(dataset_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid creation parameter JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Creation parameter JSON must contain an object")
    return payload


def insert_creation_parameters(obj: SignalObj | ImageObj, param: gds.DataSet) -> None:
    """Insert creation parameters into object metadata.

    Args:
        param: creation parameters
    """
    dataset_json = gds.dataset_to_json(param)
    raw_params = _decode_dataset_json(dataset_json)
    envelope: dict[str, object] = {
        "format_version": CREATION_PARAMETERS_FORMAT_VERSION,
        "dataset_json": dataset_json,
    }
    if isinstance(param, BaseGaussLorentzVoigtParam):
        validate_peak_creation_params(raw_params)
        envelope["peak_parameterization"] = PEAK_PARAMETERIZATION
    obj.set_metadata_option(CREATION_PARAMETERS_OPTION, envelope)
    obj.metadata.pop(f"__{LEGACY_CREATION_PARAMETERS_OPTION}", None)


def extract_creation_parameters(obj: SignalObj | ImageObj) -> gds.DataSet | None:
    """Extract creation parameters from object metadata.

    Returns:
        Creation parameters or None if not found
    """
    options = obj.get_metadata_options()
    has_current = CREATION_PARAMETERS_OPTION in options
    has_legacy = LEGACY_CREATION_PARAMETERS_OPTION in options
    if has_current and has_legacy:
        raise ValueError("Conflicting creation parameter formats")
    if not has_current and not has_legacy:
        return None

    if has_current:
        envelope = options[CREATION_PARAMETERS_OPTION]
        if not isinstance(envelope, dict):
            raise ValueError("Creation parameters must use a versioned envelope")
        version = envelope.get("format_version")
        if version != CREATION_PARAMETERS_FORMAT_VERSION:
            raise ValueError(f"Unsupported creation parameter format: {version!r}")
        dataset_json = envelope.get("dataset_json")
        if not isinstance(dataset_json, str):
            raise ValueError("Creation parameter envelope has no dataset_json")
        raw_params = _decode_dataset_json(dataset_json)
        is_peak = raw_params.get("class_name") in {
            "GaussParam",
            "LorentzParam",
            "VoigtParam",
        }
        parameterization = envelope.get("peak_parameterization")
        if is_peak:
            if parameterization != PEAK_PARAMETERIZATION:
                raise ValueError(
                    f"Unsupported peak parameterization: {parameterization!r}"
                )
            validate_peak_creation_params(raw_params)
        elif parameterization is not None:
            raise ValueError(
                "Peak parameterization set on non-peak creation parameters"
            )
        return gds.json_to_dataset(dataset_json)

    dataset_json = options[LEGACY_CREATION_PARAMETERS_OPTION]
    if not isinstance(dataset_json, str):
        raise ValueError("Legacy creation parameters must contain DataSet JSON")
    raw_params = _decode_dataset_json(dataset_json)
    if raw_params.get("class_name") in {"GaussParam", "LorentzParam", "VoigtParam"}:
        validate_peak_creation_params(raw_params)
    return gds.json_to_dataset(dataset_json)


def convert_legacy_creation_parameters(
    obj: SignalObj | ImageObj,
) -> gds.DataSet:
    """Explicitly convert legacy peak creation metadata to version 2.

    The object data is not regenerated; only its reusable creation parameters
    are converted and stored under the new metadata option.

    Args:
        obj: Object carrying historical creation metadata.

    Returns:
        Converted peak creation parameters.
    """
    options = obj.get_metadata_options()
    if CREATION_PARAMETERS_OPTION in options:
        raise ValueError("Current creation parameters already exist")
    dataset_json = options.get(LEGACY_CREATION_PARAMETERS_OPTION)
    if not isinstance(dataset_json, str):
        raise ValueError("Object has no legacy creation parameters")
    raw_params = _decode_dataset_json(dataset_json)
    converted = convert_legacy_peak_creation_params(raw_params)
    param = gds.json_to_dataset(json.dumps(converted))
    insert_creation_parameters(obj, param)
    return param


class CustomSignalParam(OrigCustomSignalParam):
    """Parameters for custom signal (e.g. manually defined experimental data)"""

    def edit_curve(self, *args) -> None:  # pylint: disable=unused-argument
        """Edit custom curve"""
        win: PlotDialog = make.dialog(
            wintitle=_("Select one point then press OK to accept"),
            edit=True,
            type="curve",
        )
        edit_tool = win.manager.add_tool(
            EditPointTool, title=_("Edit curve interactively")
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


def create_signal_gui(
    param: NewSignalParam | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> SignalObj | None:
    """Create a new Signal object from GUI

    Args:
        param: base signal parameters (NewSignalParam)
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Signal object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if param is None:
        param = NewSignalParam()
        edit = True  # Default to editing if no parameters provided

    # CustomSignalParam requires edit mode to initialize the xyarray.
    # Without this, if edit=False (the default in new_object), the setup_array
    # call would be skipped, leaving xyarray as None, which would cause an
    # AttributeError when trying to access param.xyarray.T later.
    if isinstance(param, OrigCustomSignalParam):
        edit = True

    if isinstance(param, OrigCustomSignalParam) and edit:
        p_init = NewSignalParam(_("Custom signal"))
        p_init.size = 10  # Set smaller default size for initial input
        if not p_init.edit(parent=parent):
            return None
        param.setup_array(size=p_init.size, xmin=p_init.xmin, xmax=p_init.xmax)

    if edit:
        if not param.edit(parent=parent):
            return None

    if isinstance(param, OrigCustomSignalParam):
        signal = create_signal(param.title)
        signal.xydata = param.xyarray.T
        if signal.title == SIGNAL_DEFAULT_TITLE:
            signal.title = f"custom(npts={param.size})"
        return signal

    try:
        signal = create_signal_headless(param)
    except Exception as exc:  # pylint: disable=broad-except
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError(f"Error creating signal: {exc}") from exc
        signal = None

    # Insert creation parameters into metadata, only if `param` is an instance of a
    # class deriving from `NewSignalParam` (not an instance of `NewSignalParam` itself):
    # pylint: disable=unidiomatic-typecheck
    if isinstance(param, NewSignalParam) and type(param) is not NewSignalParam:
        insert_creation_parameters(signal, param)
    return signal


def create_image_gui(
    param: NewImageParam | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> ImageObj | None:
    """Create a new Image object from GUI

    Args:
        param: image parameters
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Image object or None if canceled

    Raises:
        ValueError: if base_param is None and edit is False
    """
    if param is None:
        param = NewImageParam()
        edit = True  # Default to editing if no parameters provided

    if param.height is None:
        param.height = 500
    if param.width is None:
        param.width = 500
    if param.dtype is None:
        param.dtype = ImageDatatypes.UINT16
    dtype: ImageDatatypes = param.dtype
    numpy_dtype = dtype.to_numpy_dtype()
    if isinstance(param, Gauss2DParam):
        if param.a is None:
            try:
                param.a = np.iinfo(numpy_dtype).max / 2.0
            except ValueError:
                param.a = 10.0
    elif isinstance(param, BaseProcParam):
        param.set_from_datatype(numpy_dtype)

    if edit:
        if not param.edit(parent=parent):
            return None

    try:
        image = create_image_headless(param)
    except Exception as exc:  # pylint: disable=broad-except
        if parent is not None:
            QW.QMessageBox.warning(parent, _("Error"), str(exc))
        else:
            raise ValueError(f"Error creating image: {exc}") from exc
        return None

    # Insert creation parameters into metadata, only if `param` is an instance of a
    # class deriving from `NewImageParam` (not an instance of `NewImageParam` itself):
    # pylint: disable=unidiomatic-typecheck
    if isinstance(param, NewImageParam) and type(param) is not NewImageParam:
        insert_creation_parameters(image, param)
    return image
