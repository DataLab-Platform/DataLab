# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Miscellaneous application test
------------------------------

Whenever we have a test that does not fit in any of the other test files,
we put it here...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cdl.obj
import cdl.param
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import (
    create_2dstep_image,
    create_paracetamol_signal,
    get_test_fnames,
)

if TYPE_CHECKING:
    from cdl.core.gui.main import CDLMainWindow


def __print_test_result(title: str, result: Any = None) -> None:
    """Print a test result"""
    execenv.print(f"Testing {title}{'' if result is None else ':'}")
    if result is not None:
        execenv.print(str(result))
        execenv.print("")


def __misc_unit_function(win: CDLMainWindow) -> None:
    """Run miscellaneous unit tests

    Args:
        win: CDLMainWindow instance
    """
    panel = win.signalpanel
    objview = panel.objview

    sig = create_paracetamol_signal()
    panel.add_object(sig)
    panel.processor.compute_derivative()
    panel.processor.compute_moving_average(cdl.param.MovingAverageParam.create(n=5))

    __print_test_result("`SimpleObjectTree.__str__` method", objview)

    # Updated metadata view settings:
    __print_test_result("Updated metadata view settings")
    panel.update_metadata_view_settings()

    # Double click on the first signal item in object view:
    __print_test_result("Double click on the first signal item in object view")
    tree_item = objview.currentItem()
    objview.itemDoubleClicked.emit(tree_item, 0)

    # Open context menu on current item:
    __print_test_result("Open context menu on current item")
    tree_item_pos = objview.mapToGlobal(objview.visualItemRect(tree_item).center())
    objview.SIG_CONTEXT_MENU.emit(tree_item_pos)

    # Plot item parameters changed:
    __print_test_result("Plot item parameters changed")
    objview.select_objects([sig.uuid])
    item = panel.plothandler[sig.uuid]
    panel.plot_item_parameters_changed(item)

    # Duplicate group:
    __print_test_result("Duplicate group")
    objview.select_groups([1])
    panel.duplicate_object()

    # Delete group:
    __print_test_result("Delete group")
    objview.select_groups([2])
    panel.remove_object()

    # Exec import wizard:
    __print_test_result("Exec signal import wizard")
    win.signalpanel.exec_import_wizard()
    __print_test_result("Exec image import wizard")
    win.imagepanel.exec_import_wizard()

    # Properties changed:
    __print_test_result("Properties changed")
    objview.select_objects([sig.uuid])
    panel.properties_changed()

    # Get object titles:
    __print_test_result("Get object titles")
    execenv.print(win.get_object_titles())

    # Get object titles with info:
    __print_test_result("Get group titles with object infos")
    execenv.print(win.get_group_titles_with_object_infos())

    # Pop up tab menu:
    __print_test_result("Pop up tab menu")
    win.tabmenu.popup(win.mapToGlobal(win.tabmenu.pos()))

    # Repopulate panel trees:
    __print_test_result("Repopulate panel trees")
    win.repopulate_panel_trees()

    # Browse HDF5 files:
    __print_test_result("Browse HDF5 files")
    win.browse_h5_files([], False)

    # Open object
    __print_test_result("Open object")
    fname = get_test_fnames("*.csv")[0]
    win.open_object(fname)

    # Open objects from signal panel
    __print_test_result("Open objects from signal panel")
    win.signalpanel.open_objects(get_test_fnames("curve_formats/*.*"))

    # Get version
    __print_test_result("Get version")
    execenv.print(win.get_version())

    # Add signal
    __print_test_result("Add signal")
    win.add_signal(
        sig.title, sig.x, sig.y, sig.xunit, sig.yunit, sig.xlabel, sig.ylabel
    )

    # Add image
    __print_test_result("Add image")
    ima = create_2dstep_image()
    win.add_image(
        ima.title,
        ima.data,
        ima.xunit,
        ima.yunit,
        ima.zunit,
        ima.xlabel,
        ima.ylabel,
        ima.zlabel,
    )

    # Signal and Image ROI extraction test: test adding a default ROI
    __print_test_result("Adding a default ROI to signal and image")
    for panel in (win.signalpanel, win.imagepanel):
        panel.processor.edit_regions_of_interest(add_roi=True)

    # Close application
    __print_test_result("Close application")
    win.close_application()


def test_misc_app() -> None:
    """Run misc. application test scenario"""
    with cdltest_app_context(console=False) as win:
        execenv.print("Miscellaneous application test")
        execenv.print("==============================")
        execenv.print("")

        # We don't need to refresh the GUI during those tests (that's faster!)
        with win.context_no_refresh():
            __misc_unit_function(win)


if __name__ == "__main__":
    test_misc_app()
