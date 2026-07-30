# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Creation parameter metadata schema and compatibility tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from guidata.dataset import dataset_to_json
from sigima.objects import (
    CREATION_PARAMS_VERSION,
    PEAK_PARAMETERIZATION,
    GaussParam,
    SineParam,
    create_signal,
)
from sigima.tools.signal.pulse import GaussianModel, LegacyPeakParameterizationError

from datalab.gui.newobject import (
    CREATION_PARAMETERS_FORMAT_VERSION,
    CREATION_PARAMETERS_OPTION,
    LEGACY_CREATION_PARAMETERS_OPTION,
    convert_legacy_creation_parameters,
    extract_creation_parameters,
    insert_creation_parameters,
)


def _legacy_gaussian_json(
    *, amplitude: float = -2.5, sigma: float = 0.7, mu: float = 0.3, y0: float = 0.75
) -> str:
    """Return a historical area-based Gaussian DataSet payload."""
    return json.dumps(
        {
            "class_module": GaussParam.__module__,
            "class_name": GaussParam.__name__,
            "a": GaussianModel.area_from_amplitude(amplitude, sigma),
            "sigma": sigma,
            "mu": mu,
            "y0": y0,
        }
    )


def test_creation_parameters_use_versioned_envelope() -> None:
    """New peak creation metadata uses only the v2 option and envelope."""
    obj = create_signal("Gaussian")
    param = GaussParam.create(amplitude=-2.5, sigma=0.7, mu=0.3, y0=0.75)
    obj.set_metadata_option(LEGACY_CREATION_PARAMETERS_OPTION, "obsolete")

    insert_creation_parameters(obj, param)

    options = obj.get_metadata_options()
    assert LEGACY_CREATION_PARAMETERS_OPTION not in options
    envelope = options[CREATION_PARAMETERS_OPTION]
    assert envelope["format_version"] == CREATION_PARAMETERS_FORMAT_VERSION
    assert envelope["peak_parameterization"] == PEAK_PARAMETERIZATION
    payload = json.loads(envelope["dataset_json"])
    assert payload["creation_params_version"] == CREATION_PARAMS_VERSION
    restored = extract_creation_parameters(obj)
    assert isinstance(restored, GaussParam)
    # pylint: disable-next=no-member  # astroid cannot resolve the `create` TypeVar
    assert restored.amplitude == pytest.approx(param.amplitude)


def test_legacy_non_peak_creation_parameters_remain_readable() -> None:
    """Unchanged v1 DataSets can still be reopened through the legacy option."""
    obj = create_signal("Sine")
    param = SineParam.create(a=3.0, freq=2.0)
    obj.set_metadata_option(LEGACY_CREATION_PARAMETERS_OPTION, dataset_to_json(param))

    restored = extract_creation_parameters(obj)

    assert isinstance(restored, SineParam)
    assert restored.a == pytest.approx(param.a)
    assert restored.freq == pytest.approx(param.freq)


def test_legacy_peak_creation_parameters_require_explicit_conversion() -> None:
    """A v1 peak is refused until conversion, which leaves X/Y untouched."""
    obj = create_signal(
        "Legacy Gaussian", np.linspace(-5.0, 5.0, 100), np.arange(100.0)
    )
    original_x, original_y = (array.copy() for array in obj.xydata)
    obj.set_metadata_option(LEGACY_CREATION_PARAMETERS_OPTION, _legacy_gaussian_json())

    with pytest.raises(
        LegacyPeakParameterizationError,
        match="convert_legacy_peak_creation_params",
    ):
        extract_creation_parameters(obj)

    converted = convert_legacy_creation_parameters(obj)
    assert isinstance(converted, GaussParam)
    assert converted.amplitude == pytest.approx(-2.5)
    np.testing.assert_array_equal(obj.x, original_x)
    np.testing.assert_array_equal(obj.y, original_y)
    assert LEGACY_CREATION_PARAMETERS_OPTION not in obj.get_metadata_options()
    assert isinstance(extract_creation_parameters(obj), GaussParam)


def test_creation_parameter_formats_never_fall_back_implicitly() -> None:
    """Future versions and simultaneous v1/v2 options are rejected."""
    obj = create_signal("Future")
    obj.set_metadata_option(
        CREATION_PARAMETERS_OPTION,
        {
            "format_version": CREATION_PARAMETERS_FORMAT_VERSION + 1,
            "dataset_json": "{}",
        },
    )
    with pytest.raises(ValueError, match="Unsupported creation parameter format"):
        extract_creation_parameters(obj)

    obj.set_metadata_option(LEGACY_CREATION_PARAMETERS_OPTION, _legacy_gaussian_json())
    with pytest.raises(ValueError, match="Conflicting creation parameter formats"):
        extract_creation_parameters(obj)
