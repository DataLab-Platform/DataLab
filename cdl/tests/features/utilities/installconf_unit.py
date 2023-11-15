# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

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
