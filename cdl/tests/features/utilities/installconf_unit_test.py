# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dependencies viewer test
"""

# guitest: show

from __future__ import annotations

from guidata.qthelpers import qt_app_context

import cdl.utils.qthelpers as qth
from cdl.widgets import instconfviewer


def test_dep_viewer(screenshots: bool = False) -> None:
    """Test dep viewer window"""
    with qt_app_context():
        instconfviewer.exec_cdl_installconfig_dialog()
        if screenshots:
            dlg = instconfviewer.InstallConfigViewerWindow()
            dlg.show()
            qth.grab_save_window(dlg, dlg.objectName())


if __name__ == "__main__":
    test_dep_viewer(screenshots=True)
