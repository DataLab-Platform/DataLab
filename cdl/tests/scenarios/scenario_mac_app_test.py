# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro-commands test scenario
----------------------------

Testing all the macro-commands features.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import os.path as osp
import time

from cdl.gui.macroeditor import Macro
from cdl.gui.main import CDLMainWindow
from cdl.tests import cdltest_app_context
from cdl.utils.tests import WorkdirRestoringTempDir


def add_macro_sample(win: CDLMainWindow, index: int) -> Macro:
    """Add a macro sample to the macro panel

    Args:
        win: CDLMainWindow
        index: index of the macro sample to add
    """
    macro = win.macropanel.add_macro()
    macro.title = f"Test macro {index}"
    samples = [
        "import time; [print(f'{i}:{time.sleep(.1)}') for i in range(50)]",
        "print('Hello world')",
    ]
    assert index < len(samples), f"index={index} is out of range"
    macro.set_code(samples[index])
    return macro


def test_scenario_macro() -> None:
    """Example of high-level test scenario with HDF5 file"""
    with WorkdirRestoringTempDir() as tmpdir:
        with cdltest_app_context(console=False) as win:
            win.set_current_panel("macro")
            add_macro_sample(win, 0)
            win.macropanel.run_macro()
            time.sleep(1)
            win.macropanel.stop_macro()
            code2 = add_macro_sample(win, 1).get_code()
            fname = osp.join(tmpdir, "test.macro")
            win.macropanel.export_macro_to_file(2, fname)
            win.macropanel.remove_macro()
            macro_nb = win.macropanel.import_macro_from_file(fname)
            macro2 = win.macropanel.get_macro(macro_nb)
            assert macro2.get_code() == code2, "Macro code is not the same"


if __name__ == "__main__":
    test_scenario_macro()
