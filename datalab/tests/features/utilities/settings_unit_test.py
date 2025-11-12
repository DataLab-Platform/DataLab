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
    """Capture screenshots of each settings tab

    Note: Screenshot filenames are language-independent (use English names)
    to ensure consistency across different language builds.
    """
    with qt_app_context(exec_loop=False):
        paramdict = create_dataset_dict()
        params = gds.DataSetGroup(paramdict.values(), title=_("Settings"))
        names = list(paramdict.keys())

        # Define fixed sub-tab names for View settings (language-independent)
        # These correspond to the order of BeginGroup items in ViewSettings
        view_subtab_names = ["common", "signals", "images", "results"]

        # Create the dialog manually so we can access the tab widget
        dialog = DataSetGroupEditDialog(
            instance=params, icon="", parent=None, apply=None, wordwrap=True, size=None
        )

        # Find the main QTabWidget in the dialog
        main_tab_widget = dialog.findChild(QW.QTabWidget)

        if main_tab_widget is not None:
            # Take a screenshot of each main tab
            for i in range(main_tab_widget.count()):
                main_tab_widget.setCurrentIndex(i)
                dialog.show()
                tab_name = names[i]

                # Check if this tab contains a nested tab widget (like View settings)
                current_widget = main_tab_widget.currentWidget()
                nested_tab_widgets = current_widget.findChildren(QW.QTabWidget)

                if nested_tab_widgets:
                    # Handle nested tabs (e.g., Common, Signals, Images, Results)
                    nested_tab_widget = nested_tab_widgets[0]
                    for j in range(nested_tab_widget.count()):
                        nested_tab_widget.setCurrentIndex(j)
                        # Use predefined names instead of translated tab text
                        if tab_name == "view" and j < len(view_subtab_names):
                            nested_tab_name = view_subtab_names[j]
                        else:
                            # Fallback for any other potential nested tabs
                            nested_tab_name = f"tab{j}"
                        qth.grab_save_window(
                            dialog,
                            f"settings_{tab_name}_{nested_tab_name}",
                            add_timestamp=False,
                        )
                else:
                    # No nested tabs, just take a screenshot of the main tab
                    qth.grab_save_window(
                        dialog, f"settings_{tab_name}", add_timestamp=False
                    )

        # Don't execute the dialog, just close it
        dialog.close()


if __name__ == "__main__":
    capture_settings_screenshots()
