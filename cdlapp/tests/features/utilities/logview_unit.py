# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Log viewer test
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from cdlapp.widgets.logviewer import exec_cdl_logviewer_dialog


def test_log_viewer():
    """Test log viewer window"""
    with qt_app_context():
        exec_cdl_logviewer_dialog()


if __name__ == "__main__":
    test_log_viewer()
