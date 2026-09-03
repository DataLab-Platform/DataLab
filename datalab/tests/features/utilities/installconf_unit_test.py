# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dependencies viewer test
"""

# guitest: show

import os
import shutil
from pathlib import Path

import pytest
from guidata.qthelpers import qt_app_context

from datalab.widgets import instconfviewer
from datalab.widgets.instconfviewer import (
    InstallConfigViewerWindow,
    exec_datalab_installconfig_dialog,
)

CONFIG_FIXTURE = (
    Path(__file__).parents[3] / "data" / "tests" / "config" / "DataLab_v1.ini"
)


@pytest.fixture(name="config_filename")
def fixture_config_filename(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the viewer at a bundled configuration file.

    Keeps the tests independent from the developer's own user configuration,
    which may be absent, outdated or customized.
    """
    path = tmp_path / "DataLab_v1_typed.ini"
    shutil.copyfile(CONFIG_FIXTURE, path)
    monkeypatch.setattr(instconfviewer, "get_config_filename", lambda: str(path))
    return str(path)


def test_dep_viewer(config_filename: str):
    """Test dep viewer window"""
    del config_filename
    with qt_app_context():
        exec_datalab_installconfig_dialog()


def test_user_config_tab_can_show_config_in_folder(
    monkeypatch: pytest.MonkeyPatch, config_filename: str
):
    """User configuration tab exposes a show-in-folder action."""
    calls: list[str] = []

    def _show_in_folder(path: str) -> bool:
        calls.append(path)
        return True

    monkeypatch.setattr(instconfviewer, "_show_in_folder", _show_in_folder)

    with qt_app_context():
        window = InstallConfigViewerWindow()
        widget = window.tabs.widget(1)

        assert widget.show_in_folder_button is not None
        assert widget.show_in_folder_button.text() == "Show in folder"

        widget.show_in_folder_button.click()

        assert calls == [os.path.abspath(config_filename)]


if __name__ == "__main__":
    with qt_app_context():
        exec_datalab_installconfig_dialog()
