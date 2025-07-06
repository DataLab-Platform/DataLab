# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Log viewer test
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from datalab.widgets.logviewer import exec_datalab_logviewer_dialog


def test_logviewer_dialog():
    """Test log viewer window"""
    with qt_app_context():
        exec_datalab_logviewer_dialog()


if __name__ == "__main__":
    test_logviewer_dialog()
