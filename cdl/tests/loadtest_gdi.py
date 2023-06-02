# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
GDI objects loading test
========================

This test is specific to Windows. It is not relevant for other platforms.

This test aims at checking that all GDI objects are released when the
widget are closed (i.e. when the Python object is garbage collected).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import ctypes
import os

from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests.data import create_test_image1
from cdl.tests.newobject_unit import iterate_image_creation
from cdl.tests.scenario_sig_app import test_common_operations

SHOW = os.name == "nt"  # Show test in GUI-based test launcher


if SHOW:
    from ctypes import WinDLL

    def get_gdi_count() -> int:
        """Get the number of GDI objects for the current process.

        This function uses the Windows API to get the count of GDI (Graphical
        Device Interface) objects used by the current process.

        Note: This function will only work on Windows.

        Returns:
            int: The count of GDI objects for the current process.
        """
        # Constants
        GR_GDIOBJECTS: int = 0

        # Load the User32 DLL
        user32: WinDLL = ctypes.windll.user32

        # Get the current process id
        pid: int = os.getpid()

        # Get handle of the process
        handle: int = ctypes.windll.kernel32.OpenProcess(1040, 0, pid)

        # Call the function and get the GDI count
        gdi_count: int = user32.GetGuiResources(handle, GR_GDIOBJECTS)

        # Close the handle
        ctypes.windll.kernel32.CloseHandle(handle)

        return gdi_count

else:

    def get_gdi_count() -> int:
        """Dumb function that always returns 0."""
        return 0


def test_various_image_features(win: CDLMainWindow):
    """Run image related tests."""
    win.switch_to_panel("image")
    panel = win.imagepanel
    data_size = 150
    for image in iterate_image_creation(data_size, non_zero=True, verbose=False):
        panel.add_object(create_test_image1(data_size))
        panel.add_object(image)
        test_common_operations(panel)
        panel.remove_all_objects()
        break


def test_gdi_count(win: CDLMainWindow) -> int | None:
    """Test the GDI count.

    This function will create a DataSetGroup, show the widget, and then
    close the widget. It will then check that the GDI count has not
    increased.

    Raises:
        AssertionError: If the GDI count has increased.

    Returns:
        int: The GDI count after creating the widget.
    """
    # Get the GDI count before creating the widget
    gdi_count_before: int = get_gdi_count()

    execenv.print(f"   GDI count: {gdi_count_before} --> ", end="")

    # Create widgets during the test
    test_various_image_features(win)

    # Get the GDI count after creating the widget
    gdi_count_after: int = get_gdi_count()

    execenv.print(gdi_count_after)

    # # Check that the GDI count has not increased
    # assert gdi_count_before == gdi_count_after
    # assert gdi_count_before == gdi_count_after_close

    return gdi_count_after


def load_test():
    """Load test."""
    execenv.unattended = True
    with cdl_app_context() as win:
        gdi_count = []
        for iteration in range(4):
            execenv.print(f"Test iteration: {iteration}")
            count = test_gdi_count(win)
            if count is None:
                execenv.print("Test aborted")
                break
            gdi_count.append(count)
            if iteration > 0:
                increase = gdi_count[-1] - gdi_count[-2]
                assert increase <= 0, "GDI count should not increase (memory leak)"
                increase_pct = increase / gdi_count[0] * 100
                execenv.print(
                    f"   GDI count increase: {increase:+d} ({increase_pct:.2f}%)"
                )


if __name__ == "__main__":
    load_test()
