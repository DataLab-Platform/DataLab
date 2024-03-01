# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Macro editor test
"""

# guitest: show

from guidata.qthelpers import qt_app_context

from cdl.core.gui.panel import macro


def test_macro_editor():
    """Test dep viewer window"""
    with qt_app_context(exec_loop=True):
        widget = macro.MacroPanel()
        widget.resize(800, 600)
        widget.show()


if __name__ == "__main__":
    test_macro_editor()
