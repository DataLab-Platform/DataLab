# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Parameter sets stored in workspaces never choose the code DataLab runs

Workspaces store parameter sets as JSON naming their class. These tests check
that decoding them neither imports a module named by a file nor calls anything
but an already known parameter class (issue #407), and that legitimate
parameters, including plugin ones and those saved by earlier versions, are read
as before.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import types
from pathlib import Path

import guidata.dataset as gds
import numpy as np
import pytest
import sigima.objects as sio
from sigima.objects import GaussParam
from sigima.objects.scalar import GeometryResult, KindShape
from sigima.params import NormalizeParam

import datalab
from datalab.adapters_metadata import GeometryAdapter
from datalab.gui.newobject import (
    CREATION_PARAMETERS_OPTION,
    extract_creation_parameters,
    insert_creation_parameters,
)
from datalab.gui.processor.base import ProcessingParameters
from datalab.history.core import copy_history_value, decode_kwargs
from datalab.utils.datasetjson import ParameterClassError, decode_dataset_json

FAKE_MODULE = "dl_untrusted_test_module"
PLUGIN_MODULE = "dl_fake_plugin_module"


class PluginParam(gds.DataSet):
    """Parameter set of a plugin module"""

    value = gds.FloatItem("Value", default=1.0)


def _crafted(module_name: str, class_name: str) -> str:
    """Return normalization parameters whose class fields were edited."""
    payload = json.loads(gds.dataset_to_json(NormalizeParam()))
    payload.update(class_module=module_name, class_name=class_name)
    return json.dumps(payload)


@pytest.fixture(name="calls")
def fixture_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Register a loaded module whose callables record their calls."""
    calls: list[str] = []

    def boom() -> object:
        calls.append("boom")
        return object()

    class NotAParameterSet:  # pylint: disable=too-few-public-methods
        """Class with a side effect when instantiated"""

        def __init__(self) -> None:
            calls.append("NotAParameterSet")

    module = types.ModuleType(FAKE_MODULE)
    module.boom = boom
    module.NotAParameterSet = NotAParameterSet
    monkeypatch.setitem(sys.modules, FAKE_MODULE, module)
    return calls


@pytest.fixture(name="plugin_module")
def fixture_plugin_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Register a loaded third-party module defining a parameter set."""
    module = types.ModuleType(PLUGIN_MODULE)
    module.PluginParam = PluginParam
    monkeypatch.setitem(sys.modules, PLUGIN_MODULE, module)
    monkeypatch.setattr(PluginParam, "__module__", PLUGIN_MODULE)
    return module


@pytest.mark.parametrize(
    "param",
    [
        NormalizeParam.create(method="amplitude"),
        GaussParam.create(amplitude=-2.5, sigma=0.7, mu=0.3, y0=0.75),
    ],
)
def test_legitimate_parameters_are_decoded_as_by_guidata(param: gds.DataSet) -> None:
    """Parameters of Sigima classes are decoded exactly as guidata decodes them."""
    text = gds.dataset_to_json(param)
    decoded = decode_dataset_json(text)
    assert type(decoded) is type(param)
    assert gds.dataset_to_json(decoded) == gds.dataset_to_json(
        gds.json_to_dataset(text)
    )


def test_fields_are_read_tolerantly_as_before() -> None:
    """Unknown fields are ignored and missing ones keep their default values."""
    payload = json.loads(gds.dataset_to_json(NormalizeParam.create(method="area")))
    payload["field_of_a_future_version"] = 42
    decoded = decode_dataset_json(json.dumps(payload))
    assert decoded.method == NormalizeParam.create(method="area").method
    del payload["method"]
    decoded = decode_dataset_json(json.dumps(payload))
    assert decoded.method == NormalizeParam().method


def test_plugin_parameters_are_decoded(plugin_module: types.ModuleType) -> None:
    """A parameter class of a loaded plugin module stays usable."""
    param = PluginParam()
    param.value = 3.5
    text = gds.dataset_to_json(param)
    assert json.loads(text)["class_module"] == plugin_module.__name__
    decoded = decode_dataset_json(text)
    assert isinstance(decoded, PluginParam)
    assert decoded.value == 3.5


@pytest.mark.parametrize("name", ["boom", "NotAParameterSet"])
def test_loaded_callables_are_never_called(calls: list[str], name: str) -> None:
    """A loaded function or class that is not a parameter set is refused."""
    with pytest.raises(ParameterClassError, match="not a parameter set class"):
        decode_dataset_json(_crafted(FAKE_MODULE, name))
    assert not calls


def test_issue_reproduction_creates_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproduction of issue #407 no longer calls ``tempfile.mkdtemp``."""
    calls: list[str] = []
    monkeypatch.setattr(tempfile, "mkdtemp", lambda: calls.append("mkdtemp"))
    with pytest.raises(ParameterClassError):
        decode_dataset_json(_crafted("tempfile", "mkdtemp"))
    assert not calls


def test_untrusted_modules_are_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module that DataLab has not loaded is not imported for a file."""
    monkeypatch.delitem(sys.modules, "this", raising=False)
    with pytest.raises(ParameterClassError, match="which DataLab does not use"):
        decode_dataset_json(_crafted("this", "d"))
    assert "this" not in sys.modules


@pytest.mark.parametrize(
    ("module_name", "message"),
    [
        ("datalab_lookalike_module.params", "which DataLab does not use"),
        ("datalab.__main__", "which DataLab does not use"),
        ("datalab.tests.not_a_module", "which DataLab does not use"),
        ("sigima._not_a_module", "which DataLab does not use"),
        ("sigima.not_a_module_for_tests", "missing module"),
    ],
)
def test_only_public_modules_of_datalab_packages_may_be_imported(
    module_name: str, message: str
) -> None:
    """Only public modules of DataLab's own packages are looked for on import."""
    with pytest.raises(ParameterClassError, match=message):
        decode_dataset_json(_crafted(module_name, "NormalizeParam"))


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"class_name": "NormalizeParam"}),
        json.dumps({"class_module": 1, "class_name": "NormalizeParam"}),
        json.dumps({"class_module": "sigima.params", "class_name": "os.path"}),
        json.dumps({"class_module": "sigima.params", "class_name": "__class__"}),
        json.dumps(["sigima.params", "NormalizeParam"]),
    ],
)
def test_malformed_class_fields_are_refused(text: str) -> None:
    """Missing, non-text or non-class names are refused."""
    with pytest.raises(ParameterClassError):
        decode_dataset_json(text)


def test_processing_parameters_refuse_crafted_classes(calls: list[str]) -> None:
    """Processing parameters stored with an object (Processing tab, History)."""
    stored = ProcessingParameters(func_name="normalize", pattern="1-to-1")
    with pytest.warns(UserWarning):
        stored.set_param_from_json(_crafted(FAKE_MODULE, "boom"))
    assert stored.param is None
    assert not calls
    stored.set_param_from_json(gds.dataset_to_json(NormalizeParam()))
    assert isinstance(stored.param, NormalizeParam)


def test_analysis_parameters_refuse_crafted_classes(calls: list[str]) -> None:
    """Parameters of analysis results shown in the Properties panel."""
    obj = sio.create_signal("signal", np.arange(4.0), np.arange(4.0))
    result = GeometryResult(
        title="fwhm",
        func_name="fwhm",
        kind=KindShape.SEGMENT,
        coords=np.array([[0.0, 0.0, 1.0, 0.0]]),
        roi_indices=None,
        attrs={"param_json": _crafted(FAKE_MODULE, "boom")},
    )
    GeometryAdapter(result).add_to(obj)
    (adapter,) = GeometryAdapter.iterate_from_obj(obj)
    with pytest.warns(UserWarning):
        assert adapter.get_param() is None
    assert not calls


def test_creation_parameters_refuse_crafted_classes(calls: list[str]) -> None:
    """Creation parameters shown in the Creation tab."""
    obj = sio.create_signal("signal", np.arange(4.0), np.arange(4.0))
    insert_creation_parameters(obj, sio.SineParam())
    envelope = obj.get_metadata_options()[CREATION_PARAMETERS_OPTION]
    envelope["dataset_json"] = _crafted(FAKE_MODULE, "boom")
    obj.set_metadata_option(CREATION_PARAMETERS_OPTION, envelope)
    with pytest.raises(ValueError):
        extract_creation_parameters(obj)
    assert not calls


def test_history_arguments_refuse_crafted_classes(calls: list[str]) -> None:
    """Parameters of History actions read from a workspace or a .dlhist file."""
    crafted = _crafted(FAKE_MODULE, "boom")
    with pytest.warns(UserWarning):
        decoded = decode_kwargs(
            {
                "param": {"__dataset_json__": crafted},
                "params": {"__dataset_list_json__": [crafted]},
            }
        )
    assert decoded == {"param": None, "params": []}
    assert not calls


def test_history_copies_keep_their_parameters() -> None:
    """Duplicating a History action still copies its parameter sets."""
    param = NormalizeParam.create(method="area")
    copied = copy_history_value(param)
    assert copied is not param
    assert gds.dataset_to_json(copied) == gds.dataset_to_json(param)


def test_no_permissive_decoding_left_in_datalab() -> None:
    """DataLab decodes stored parameter sets only through the safe decoder."""
    package = Path(datalab.__file__).parent
    pattern = re.compile(r"\bjson_to_dataset\b")
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if "tests" not in path.relative_to(package).parts
        and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders
