# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro Panel application test
(essentially for the screenshot...)
"""

# guitest: show

from cdl import config
from cdl.core.gui.main import CDLMainWindow
from cdl.utils import qthelpers as qth


def test_macro(screenshots: bool = False) -> None:
    """Run image tools test scenario"""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    with qth.cdl_app_context(exec_loop=True):
        win = CDLMainWindow()
        win.show()
        win.set_current_panel("macro")
        win.macropanel.add_macro()
        if screenshots:
            qth.grab_save_window(win.macropanel, "macro_panel")


if __name__ == "__main__":
    test_macro()
