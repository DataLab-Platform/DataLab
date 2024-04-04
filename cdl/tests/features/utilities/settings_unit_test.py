# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab settings test
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from cdl.core.gui.settings import edit_settings
from cdl.env import execenv


def test_edit_settings():
    """Test edit settings"""
    with qt_app_context():
        changed = edit_settings(None)
        execenv.print(changed)


if __name__ == "__main__":
    test_edit_settings()
