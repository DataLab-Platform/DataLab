# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Single-file plugins created by ``datalab-plugin create``, used in DataLab."""

# guitest: show

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from qtpy import QtWidgets as QW
from sigima.objects import create_signal

from datalab.config import Conf
from datalab.gui.actionhandler import ActionCategory
from datalab.gui.main import DLMainWindow
from datalab.plugins import PluginRegistry
from datalab.plugins.generator import PluginProject, create_single_file_plugin
from datalab.tests import datalab_test_app_context
from datalab.tests.features.plugins.plugin_test_dataset import temporary_plugin_dir

PLUGIN_ID = "org.example.generated-single-file"
PLUGIN_NAME = "Generated Single File"
FEATURE_ID = f"{PLUGIN_ID}:sample-processing"
X = np.linspace(-1.0, 1.0, 21)


@contextmanager
def enabled_plugins_context():
    """Temporarily enable all plugins."""
    with Conf.plugins_enabled.context(True), Conf.plugins_enabled_list.context(None):
        yield


def get_plugin_menu_actions(win: DLMainWindow) -> list[str]:
    """Return the actions of the plugin submenu in the signal panel."""
    for item in win.signalpanel.get_category_actions(ActionCategory.PLUGINS):
        if isinstance(item, QW.QMenu) and item.title() == PLUGIN_NAME:
            # Qt menus populate their actions lazily
            item.aboutToShow.emit()
            return [action.text() for action in item.actions()]
    raise AssertionError(f"{PLUGIN_NAME!r} menu not found in the Plugins menu")


def run_sample_processing(win: DLMainWindow) -> np.ndarray:
    """Run the sample processing on a signal with negative values."""
    win.signalpanel.add_object(create_signal("Ramp", X, X.copy()))
    count = len(win.signalpanel)
    win.signalpanel.processor.run_feature(FEATURE_ID)
    assert len(win.signalpanel) == count + 1
    return win.signalpanel[len(win.signalpanel)].y


@enabled_plugins_context()
def test_single_file_plugin_in_datalab() -> None:
    """A generated plugin adds a working processing to the Plugins menu."""
    with temporary_plugin_dir() as plugin_dir:
        create_single_file_plugin(
            PluginProject(
                destination=Path(plugin_dir),
                name=PLUGIN_NAME,
                package="datalab_test_plugin_generated_single_file",
                plugin_id=PLUGIN_ID,
                description="Plugin created by datalab-plugin create",
            )
        )
        expected = np.maximum(X, 0.0)

        with datalab_test_app_context(console=False) as win:
            assert PluginRegistry.get_plugin(PLUGIN_ID) is not None
            assert get_plugin_menu_actions(win) == ["Sample processing"]
            assert np.array_equal(run_sample_processing(win), expected)

            win.reload_plugins()
            QW.QApplication.processEvents()

            registry = list(win.signalpanel.processor.computing_registry)
            assert registry.count(FEATURE_ID) == 1
            assert get_plugin_menu_actions(win) == ["Sample processing"]
            assert np.array_equal(run_sample_processing(win), expected)


if __name__ == "__main__":
    test_single_file_plugin_in_datalab()
