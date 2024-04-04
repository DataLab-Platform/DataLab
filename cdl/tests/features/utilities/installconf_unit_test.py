# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Dependencies viewer test
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from cdl.widgets.instconfviewer import exec_cdl_installconfig_dialog


def test_dep_viewer():
    """Test dep viewer window"""
    with qt_app_context():
        exec_cdl_installconfig_dialog()


if __name__ == "__main__":
    test_dep_viewer()
