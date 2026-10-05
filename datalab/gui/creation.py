# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Dependency-neutral object creation services."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np
from sigima.objects import (
    PEAK_PARAMETERIZATION,
    CustomSignalParam,
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

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW

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
        obj: Object receiving the serialized parameters.
        param: Creation parameters.
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

    Args:
        obj: Object containing serialized creation parameters.

    Returns:
        Creation parameters or None if not found.
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
