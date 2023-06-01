# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab settings test
"""

from cdl.core.gui.settings import edit_settings
from cdl.utils.qthelpers import qt_app_context

SHOW = True  # Show test in GUI-based test launcher


def test_edit_settings():
    """Test edit settings"""
    with qt_app_context():
        edit_settings()


if __name__ == "__main__":
    test_edit_settings()
