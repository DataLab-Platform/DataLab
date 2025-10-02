# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Miscellaneous application test
------------------------------

Whenever we have a test that does not fit in any of the other test files,
we put it here...
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING, Any

import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import (
    create_2dstep_image,
    create_paracetamol_signal,
    get_test_fnames,
)

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


def __print_test_result(title: str, result: Any = None) -> None:
    """Print a test result"""
    execenv.print(f"Testing {title}{'' if result is None else ':'}")
    if result is not None:
        execenv.print(str(result))
        execenv.print("")


def __misc_unit_function(win: DLMainWindow) -> None:
    """Run miscellaneous unit tests

    Args:
        win: DLMainWindow instance
    """
    panel = win.signalpanel
    objview = panel.objview

    sig = create_paracetamol_signal()
    panel.add_object(sig)
    panel.processor.run_feature(sips.derivative)
    panel.processor.run_feature(
        sips.moving_average, sigima.params.MovingAverageParam.create(n=5)
    )

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
    objview.select_objects([get_uuid(sig)])
    item = panel.plothandler[get_uuid(sig)]
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
    objview.select_objects([get_uuid(sig)])
    panel.properties_changed()

    # Get object titles:
    __print_test_result("Get object titles")
    execenv.print(win.get_object_titles())

    # Get object titles with info:
    __print_test_result("Get group titles with object info")
    execenv.print(win.get_group_titles_with_object_info())

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
    win.load_from_files([fname])

    # Open directory
    __print_test_result("Open directory (signals)")
    path = osp.dirname(get_test_fnames("curve_formats/*.*")[0])
    win.signalpanel.load_from_directory(path)

    # Select all signals and save them to a temporary directory
    __print_test_result("Select first group and save signals to a temporary directory")
    win.signalpanel.objview.select_groups([1])
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        param = sigima.params.SaveToDirectoryParam.create(
            directory=tmpdir, basename="{title}_test", extension=".csv", overwrite=True
        )
        win.signalpanel.save_to_directory(param)

    # Open objects from signal panel
    __print_test_result("Open objects from signal panel")
    fnames = [f for f in get_test_fnames("curve_formats/*.*") if not f.endswith(".mca")]
    win.signalpanel.load_from_files(fnames)

    # Get version
    __print_test_result("Get version")
    execenv.print(win.get_version())

    # Add signal
    __print_test_result("Add signal")
    win.add_signal(
        sig.title, sig.x, sig.y, sig.xunit, sig.yunit, sig.xlabel, sig.ylabel
    )
    gp2 = win.signalpanel.add_group("group2")
    win.add_signal(
        sig.title,
        sig.x,
        sig.y,
        sig.xunit,
        sig.yunit,
        sig.xlabel,
        sig.ylabel,
        group_id=get_uuid(gp2),
        set_current=False,
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
    gp3 = win.imagepanel.add_group("group3")
    win.add_image(
        ima.title,
        ima.data,
        ima.xunit,
        ima.yunit,
        ima.zunit,
        ima.xlabel,
        ima.ylabel,
        ima.zlabel,
        group_id=get_uuid(gp3),
        set_current=False,
    )

    # Close application
    __print_test_result("Close application")
    win.close_application()


def test_misc_app() -> None:
    """Run misc. application test scenario"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Miscellaneous application test")
        execenv.print("==============================")
        execenv.print("")

        # We don't need to refresh the GUI during those tests (that's faster!)
        with win.context_no_refresh():
            __misc_unit_function(win)


if __name__ == "__main__":
    test_misc_app()
