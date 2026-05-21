# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dependencies viewer test
"""

# guitest: show

import os

import pytest
from guidata.qthelpers import qt_app_context

from datalab.config import Conf
from datalab.widgets import instconfviewer
from datalab.widgets.instconfviewer import (
    InstallConfigViewerWindow,
    exec_datalab_installconfig_dialog,
)


def test_dep_viewer():
    """Test dep viewer window"""
    with qt_app_context():
        exec_datalab_installconfig_dialog()


def test_user_config_tab_can_show_config_in_folder(monkeypatch: pytest.MonkeyPatch):
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

        assert calls == [os.path.abspath(Conf.get_filename())]


if __name__ == "__main__":
    test_dep_viewer()
