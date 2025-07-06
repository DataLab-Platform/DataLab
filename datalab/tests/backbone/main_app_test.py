# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Application test for main window
--------------------------------

Testing the features of the main window of the application that are not
covered by other tests.
"""

# guitest: show

import os

import sigima.computation.signal as sigima_signal
import sigima.param
from sigima.tests.data import create_paracetamol_signal

from datalab.env import execenv
from datalab.objectmodel import get_short_id, get_uuid
from datalab.tests import cdltest_app_context


def test_main_app():
    """Main window test"""
    with cdltest_app_context(console=False) as win:
        # Switch from panel to panel
        for panelname in ("macro", "image", "signal"):
            win.set_current_panel(panelname)
        # Switch to an unknown panel
        try:
            win.set_current_panel("unknown_panel")
            raise RuntimeError("Unknown panel should have raised an exception")
        except ValueError:
            pass

        panel = win.signalpanel

        # Create new groups
        grp1 = panel.add_group("Group 1")
        panel.add_group("Group 2")
        # Add group using different levels of the API
        panel.add_group("Group 3", select=True)
        panel.remove_object(force=True)
        win.add_group("Group 4", select=True)
        panel.remove_object(force=True)
        win.add_group("Group 5", panel="signal", select=True)
        panel.remove_object(force=True)
        # Rename group
        panel.objview.select_groups([2])
        panel.rename_selected_object_or_group("Group xxx")
        panel.remove_object(force=True)

        # Add signals to signal panel
        sig1 = create_paracetamol_signal(500)
        panel.add_object(sig1)
        panel.processor.run_feature(sigima_signal.derivative)
        panel.processor.run_feature(sigima_signal.wiener)

        # Get object titles
        titles = win.get_object_titles()
        execenv.print(f"Object titles:{os.linesep}{titles}")

        # Get object uuids
        uuids = win.get_object_uuids()
        uuids2 = win.get_object_uuids(group=1)
        uuids3 = win.get_object_uuids(group="Group 1")
        uuids4 = win.get_object_uuids(group=get_uuid(grp1))
        assert uuids == uuids2 == uuids3 == uuids4, "Group UUIDs should be the same"
        execenv.print(f"Object uuids:{os.linesep}{uuids}")

        # Testing `get_object`
        execenv.print("*** Testing `get_object` ***")
        # Get object from title
        obj = win.get_object(titles[-1])
        execenv.print(f"  Object (from title) '{get_short_id(obj)}':{os.linesep}{obj}")
        # Get object
        obj = win.get_object(1)
        execenv.print(
            f"  Object (from number)  '{get_short_id(obj)}':{os.linesep}{obj}"
        )
        # Get object by uuid
        obj = win.get_object(uuids[-1])
        execenv.print(f"  Object (from uuid)  '{get_short_id(obj)}':{os.linesep}{obj}")

        # Testing dict-like interface of main window:
        execenv.print("*** Testing dict-like interface of proxy ***")
        # Get object from title
        obj = win[titles[-1]]
        execenv.print(f"  Object (from title) '{get_short_id(obj)}':{os.linesep}{obj}")
        # Get object
        obj = win[1]
        execenv.print(
            f"  Object (from number)  '{get_short_id(obj)}':{os.linesep}{obj}"
        )
        # Get object by uuid
        obj = win[uuids[-1]]
        execenv.print(f"  Object (from uuid)  '{get_short_id(obj)}':{os.linesep}{obj}")

        # Use "calc" method with parameters
        param = sigima.param.MovingMedianParam.create(n=5)
        win.calc("compute_moving_median", param)
        # Use "calc" method without parameters
        win.calc("compute_integral")
        # Use "calc" and choose an unknown computation method
        try:
            win.calc("unknown_method")
            raise RuntimeError("Unknown method should have raised an exception")
        except ValueError:
            pass

        # Force application menus to pop-up
        for menu in (
            win.file_menu,
            win.edit_menu,
            win.operation_menu,
            win.processing_menu,
            win.analysis_menu,
            win.view_menu,
            win.help_menu,
        ):
            menu.popup(menu.pos())
        win.file_menu.popup(win.mapToGlobal(win.file_menu.pos()))

        # Open settings dialog
        win.settings_action.trigger()


if __name__ == "__main__":
    test_main_app()
