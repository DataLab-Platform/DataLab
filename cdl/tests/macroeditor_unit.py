# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Macro editor test
"""

from cdl.core.gui.panel import MacroPanel
from cdl.utils.qthelpers import qt_app_context

SHOW = True  # Show test in GUI-based test launcher


def test_macro_editor():
    """Test dep viewer window"""
    with qt_app_context(exec_loop=True):
        widget = MacroPanel()
        widget.resize(800, 600)
        widget.show()


if __name__ == "__main__":
    test_macro_editor()
