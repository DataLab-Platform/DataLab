# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Computing feature identity unit tests."""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from sigima.objects import SignalObj, create_signal
from sigima.proc.decorator import computation_function

from datalab.gui.processor.base import BaseProcessor, ComputingFeature
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


def test_builtin_feature_ids_are_unique(monkeypatch) -> None:
    """Built-in registrations never reuse an identifier within a panel."""
    registrations: list[tuple[str, str]] = []
    add_feature = BaseProcessor.add_feature

    def record_feature(self: BaseProcessor, feature: ComputingFeature) -> None:
        add_feature(self, feature)
        if feature.plugin_origin is None:
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
            normalize: plugin_feature,
            **processor.computing_registry,
        }

        assert processor.get_feature("normalize").function is sips.normalize
        feature = processor.get_feature("normalize", plugin_origin=PLUGIN_ORIGIN)
        assert feature is plugin_feature
        assert processor.get_feature(normalize) is plugin_feature
        assert processor.get_feature_id(normalize) == "normalize"


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
