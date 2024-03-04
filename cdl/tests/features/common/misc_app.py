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

from typing import Any

import cdl.obj
import cdl.param
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_paracetamol_signal


def __print_test_result(title: str, result: Any = None) -> None:
    """Print a test result"""
    execenv.print(f"Testing {title}{'' if result is None else ':'}")
    if result is not None:
        execenv.print(str(result))
        execenv.print("")


def test_misc_app() -> None:
    """Run misc. application test scenario"""
    with cdltest_app_context(console=False) as win:
        execenv.print("Miscellaneous application test")
        execenv.print("==============================")
        execenv.print("")

        # We don't need to refresh the GUI during those tests (that's faster!)
        win.toggle_auto_refresh(False)

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


if __name__ == "__main__":
    test_misc_app()
