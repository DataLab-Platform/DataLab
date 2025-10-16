# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab settings test
"""

# guitest: show

import guidata.dataset as gds
from guidata.dataset.qtwidgets import DataSetGroupEditDialog
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.settings import create_dataset_dict, edit_settings
from datalab.utils import qthelpers as qth


def test_edit_settings():
    """Test edit settings"""
    with qt_app_context():
        changed = edit_settings(None)
        execenv.print(changed)


def capture_settings_screenshots():
    """Capture screenshots of each settings tab"""
    with qt_app_context(exec_loop=False):
        paramdict = create_dataset_dict()
        params = gds.DataSetGroup(paramdict.values(), title=_("Settings"))
        names = list(paramdict.keys())

        # Create the dialog manually so we can access the tab widget
        dialog = DataSetGroupEditDialog(
            instance=params, icon="", parent=None, apply=None, wordwrap=True, size=None
        )

        # Find the QTabWidget in the dialog
        tab_widget = dialog.findChild(QW.QTabWidget)

        if tab_widget is not None:
            # Take a screenshot of each tab
            for i in range(tab_widget.count()):
                tab_widget.setCurrentIndex(i)
                dialog.show()
                qth.grab_save_window(
                    dialog, f"settings_{names[i]}", add_timestamp=False
                )

        # Don't execute the dialog, just close it
        dialog.close()


if __name__ == "__main__":
    capture_settings_screenshots()
