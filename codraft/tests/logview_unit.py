# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Log viewer test
"""

from codraft.utils.qthelpers import qt_app_context
from codraft.widgets.logviewer import exec_codraft_logviewer_dialog

SHOW = True  # Show test in GUI-based test launcher


def test_log_viewer():
    """Test log viewer window"""
    with qt_app_context():
        exec_codraft_logviewer_dialog()


if __name__ == "__main__":
    test_log_viewer()
