# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Dependency-neutral object creation services."""

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np
from sigima.objects import (
    CustomSignalParam,
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    NewImageParam,
    NewSignalParam,
    SignalObj,
    create_signal,
)
from sigima.objects import create_image_from_param as create_image_headless
from sigima.objects import create_signal_from_param as create_signal_headless
from sigima.objects.base import BaseProcParam
from sigima.objects.signal import DEFAULT_TITLE as SIGNAL_DEFAULT_TITLE

from datalab.config import _

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW

CREATION_PARAMETERS_OPTION = "creation_param_json"


def insert_creation_parameters(obj: SignalObj | ImageObj, param: gds.DataSet) -> None:
    """Insert creation parameters into object metadata.

    Args:
        obj: Object receiving the serialized parameters.
        param: Creation parameters.
    """
    obj.set_metadata_option(CREATION_PARAMETERS_OPTION, gds.dataset_to_json(param))


def extract_creation_parameters(obj: SignalObj | ImageObj) -> gds.DataSet | None:
    """Extract creation parameters from object metadata.

    Args:
        obj: Object containing serialized creation parameters.

    Returns:
        Creation parameters or None if not found.
    """
    try:
        param_json = obj.get_metadata_option(CREATION_PARAMETERS_OPTION)
    except ValueError:
        return None
    return gds.json_to_dataset(param_json)


def create_signal_from_param(param: NewSignalParam) -> SignalObj:
    """Create a signal from initialized parameters."""
    if isinstance(param, CustomSignalParam):
        signal = create_signal(param.title)
        signal.xydata = param.xyarray.T
        if signal.title == SIGNAL_DEFAULT_TITLE:
            signal.title = f"custom(npts={param.size})"
        return signal
    signal = create_signal_headless(param)
    if param.__class__ is not NewSignalParam:
        insert_creation_parameters(signal, param)
    return signal


def prepare_signal_parameters(
    param: NewSignalParam | None,
    edit: bool,
    parent: QW.QWidget | None = None,
) -> NewSignalParam | None:
    """Initialize and optionally edit signal creation parameters."""
    if param is None:
        param = NewSignalParam()
        edit = True
    if isinstance(param, CustomSignalParam):
        edit = True
    if isinstance(param, CustomSignalParam) and edit:
        initial = NewSignalParam(_("Custom signal"))
        initial.size = 10
        if not initial.edit(parent=parent):
            return None
        param.setup_array(size=initial.size, xmin=initial.xmin, xmax=initial.xmax)
    if edit and not param.edit(parent=parent):
        return None
    return param


def initialize_image_parameters(param: NewImageParam) -> None:
    """Fill image creation defaults required by the editor and constructor."""
    if param.height is None:
        param.height = 500
    if param.width is None:
        param.width = 500
    if param.dtype is None:
        param.dtype = ImageDatatypes.UINT16
    numpy_dtype = param.dtype.to_numpy_dtype()
    if isinstance(param, Gauss2DParam):
        if param.a is None:
            try:
                param.a = np.iinfo(numpy_dtype).max / 2.0
            except ValueError:
                param.a = 10.0
    elif isinstance(param, BaseProcParam):
        param.set_from_datatype(numpy_dtype)


def create_image_from_param(param: NewImageParam) -> ImageObj:
    """Create an image from initialized parameters."""
    initialize_image_parameters(param)
    image = create_image_headless(param)
    if param.__class__ is not NewImageParam:
        insert_creation_parameters(image, param)
    return image
