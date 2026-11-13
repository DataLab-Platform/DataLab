# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Computing feature identity unit tests."""

from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from sigima.objects import SignalObj, create_signal
from sigima.proc.decorator import computation_function

from datalab.config import Conf
from datalab.gui.processor.base import (
    BaseProcessor,
    ComputingFeature,
    ProcessingParameters,
    extract_processing_parameters,
)
from datalab.tests import datalab_test_app_context

PLUGIN_ORIGIN = {
    "plugin_class": "IdentityTestPlugin",
    "module": "datalab_identity_test_plugin",
    "directory": None,
    "version": None,
}


@computation_function()
def normalize(src: SignalObj) -> SignalObj:
    """Plugin computation sharing a built-in feature identifier."""
    return src.copy()


@computation_function()
def layered_offset(src: SignalObj) -> SignalObj:
    """Computation of a layered plugin, outside the module of its plugin class."""
    dst = src.copy()
    dst.y = dst.y + 1.0
    return dst


LAYERED_MODULE = "layered_plugin.workflow.computations"
for _function in (layered_offset, getattr(layered_offset, "__wrapped__", None)):
    if _function is not None:
        _function.__module__ = LAYERED_MODULE


class LayeredPlugin:
    """Plugin class of a layered plugin project."""

    info = SimpleNamespace(version="1.2.3")


LayeredPlugin.__module__ = "layered_plugin.adapters.desktop"


def test_layered_plugin_feature_origin_and_workspace(tmp_path) -> None:
    """A feature of a layered plugin has the origin of its owner and can be saved.

    Its function module is not under the module of the plugin class: the origin
    used to be guessed as anonymous, with None values which made workspaces
    impossible to save (issue #543).
    """
    plugin = LayeredPlugin()
    owner = "org.example.layered"
    with (
        Conf.process_isolation_enabled.context(False),
        datalab_test_app_context(console=False) as win,
        patch("datalab.plugins.PluginRegistry.get_plugin", return_value=plugin),
        patch("datalab.plugins.PluginRegistry.get_plugins", return_value=[plugin]),
    ):
        panel = win.signalpanel
        processor = panel.processor
        processor.add_feature(
            ComputingFeature("1_to_1", function=layered_offset, title="Offset"),
            owner=owner,
        )
        feature_id = f"{owner}:layered_offset"
        origin = processor.get_feature(feature_id).plugin_origin
        assert origin["plugin_class"] == "LayeredPlugin"
        assert origin["module"] == LAYERED_MODULE
        assert origin["version"] == "1.2.3"
        assert None not in origin.values()

        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        panel.add_object(source)
        panel.objview.select_objects([source])
        processor.run_feature(feature_id)
        result = panel[len(panel)]
        assert extract_processing_parameters(result).plugin_origin == origin
        filename = str(tmp_path / "layered.h5")
        win.save_to_h5_file(filename)
        win.load_h5_workspace([filename], reset_all=True)
        reopened = extract_processing_parameters(panel[len(panel)])
        assert reopened.func_name == feature_id
        assert reopened.plugin_origin == origin


def test_processing_parameters_drop_unknown_origin_values() -> None:
    """Origins recorded with None values can still be stored in workspaces."""
    parameters = ProcessingParameters(
        "plugin:feature",
        "1-to-1",
        plugin_origin={"plugin_class": None, "module": "plugin", "version": None},
    )
    assert parameters.to_dict()["plugin_origin"] == {"module": "plugin"}


def test_builtin_feature_ids_are_unique(monkeypatch) -> None:
    """Built-in registrations never reuse an identifier within a panel."""
    registrations: list[tuple[str, str]] = []
    add_feature = BaseProcessor.add_feature

    def record_feature(
        self: BaseProcessor, feature: ComputingFeature, owner: str | None = None
    ) -> None:
        add_feature(self, feature, owner)
        if feature.plugin_origin is None and feature.owner_plugin_id is None:
            registrations.append((self.panel.PANEL_STR_ID, feature.feature_id))

    monkeypatch.setattr(BaseProcessor, "add_feature", record_feature)
    with datalab_test_app_context(console=False):
        pass
    assert registrations
    counts = Counter(registrations)
    assert [key for key, count in counts.items() if count > 1] == []


def test_get_feature_prefers_builtin_or_matching_plugin() -> None:
    """Identifier lookup does not depend on registration order."""
    with datalab_test_app_context(console=False) as win:
        processor = win.signalpanel.processor
        plugin_feature = ComputingFeature(
            "1_to_1",
            function=normalize,
            title="Plugin normalization",
            plugin_origin=dict(PLUGIN_ORIGIN),
        )
        processor.computing_registry = {
            plugin_feature.feature_id: plugin_feature,
            **processor.computing_registry,
        }

        assert processor.get_feature("normalize").function is sips.normalize
        feature = processor.get_feature("normalize", plugin_origin=PLUGIN_ORIGIN)
        assert feature is plugin_feature
        assert processor.get_feature(normalize) is plugin_feature
        assert (
            processor.get_feature_id(normalize)
            == "datalab_identity_test_plugin:normalize"
        )


def test_compute_1_to_1_honours_registered_preview_flag(monkeypatch) -> None:
    """Direct 1-to-1 calls respect the registered preview capability."""
    allowed_values: list[bool] = []

    def cancel_edit(draft, function, sources, parent, allowed, results, **kwargs):
        allowed_values.append(allowed)
        return False

    monkeypatch.setattr(
        "datalab.widgets.processingpreview.edit_processing_parameters", cancel_edit
    )
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        processor = panel.processor
        feature = processor.get_feature(sips.moving_average)
        processor.remove_feature(feature.feature_id)
        processor.add_feature(dataclasses.replace(feature, preview_enabled=False))
        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        panel.add_object(source)
        panel.objview.select_objects([source])
        processor.compute_1_to_1(
            sips.moving_average,
            paramclass=sigima.params.MovingAverageParam,
            edit=True,
        )
    assert allowed_values == [False]
