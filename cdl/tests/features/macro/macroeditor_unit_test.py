# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Macro Panel unit tests
----------------------

The objective of this test is to check all the functionalities of the Macro Panel
widget, by calling all its methods and checking the results.

Some methods are not tested here, as they are tested in remote control tests
(see cdl/tests/features/control/remoteclient_app.py):
- `run_macro`
- `stop_macro`

All other methods should be tested here.
"""

# guitest: show

import os.path as osp

from guidata.qthelpers import qt_app_context

from cdl.env import execenv
from cdl.gui.macroeditor import Macro
from cdl.gui.panel import macro
from cdl.utils import tests
from cdl.utils.tests import get_temporary_directory


def get_macro_example_path() -> str:
    """Return macro example path"""
    path = get_temporary_directory()
    contents = """
# Simple DataLab macro example

import numpy as np

from cdl.proxy import RemoteProxy

proxy = RemoteProxy()

z = np.random.rand(20, 20)
proxy.add_image("toto", z)
proxy.compute_fft()

print("All done! :)")
"""
    filename = osp.join(path, "macro_example.py")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(contents)
    return filename


def test_macro_editor():
    """Test dep viewer window"""
    with qt_app_context(exec_loop=True):
        widget = macro.MacroPanel()
        widget.resize(800, 600)
        widget.show()

        # Create a new macro
        new_macro = widget.add_macro()
        assert new_macro is widget.get_macro()

        # Check out the macro title, serializable name, number, ...
        execenv.print("Macro title:", new_macro.title)
        execenv.print("Serializable name:", widget.get_serializable_name(new_macro))
        nb1 = widget.get_number_from_macro(new_macro)
        nb2 = widget.get_number_from_title(new_macro.title)
        assert nb1 == nb2 and nb1 == 1
        execenv.print("Macro number:", nb1)
        titles = widget.get_macro_titles()
        assert titles[0] == new_macro.title and len(titles) == 1
        new_title = "New title"
        widget.rename_macro(1, new_title)
        assert widget.get_macro_titles()[0] == new_title

        with tests.CDLTemporaryDirectory() as tmpdir:
            fname = osp.join(tmpdir, "macro.py")
            widget.export_macro_to_file(1, fname)
            widget.import_macro_from_file(fname)
            imported_macro: Macro = widget.get_macro(2)
            assert imported_macro.title == new_macro.title
            assert imported_macro.get_code() == new_macro.get_code()
            widget.rename_macro(1, "Other title")
            widget.remove_macro(1)
            assert len(widget.get_macro_titles()) == 1
            assert widget.get_macro_titles()[0] == imported_macro.title

        # Remove all macros
        widget.remove_all_objects()
        assert len(widget.get_macro_titles()) == 0

        # Load a macro from file
        macro_path = get_macro_example_path()
        widget.import_macro_from_file(macro_path)
        assert len(widget.get_macro_titles()) == 1
        assert widget.get_macro_titles()[0] == osp.basename(macro_path)


if __name__ == "__main__":
    test_macro_editor()
