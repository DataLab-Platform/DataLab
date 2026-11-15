# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History Panel application test
(essentially for the screenshot...)

Records a representative sequence of UI and computation actions and grabs
a screenshot of the history panel, used in the documentation
(:ref:`historypanel`).
"""

# guitest: show

from __future__ import annotations

import os
from pathlib import Path

import sigima.objects
import sigima.proc.signal as sips
from guidata.qthelpers import add_actions
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab import config
from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_history_panel(screenshots: bool = False) -> None:
    """Record a representative session and grab the History Panel screenshot."""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    config.Conf.history_new_session_behavior.set("no")
    config.Conf.welcome_on_startup.set(False)
    with datalab_test_app_context(
        console=False, exec_loop=not screenshots, history=True
    ) as win:
        history = win.historypanel
        history.toggle_record_mode(True)

        panel = win.signalpanel

        # [New Voigt, New Lorentzian, New Lorentzian]
        panel.new_object(param=sigima.objects.VoigtParam(), edit=False)
        panel.new_object(param=sigima.objects.LorentzParam(), edit=False)
        panel.new_object(param=sigima.objects.LorentzParam(), edit=False)

        # Remove the third signal
        panel.objview.select_objects([3])
        panel.remove_object(force=True)

        # New Gaussian
        panel.new_object(param=sigima.objects.GaussParam(), edit=False)

        # Average of the 3 remaining signals
        panel.objview.select_objects([1, 2, 3])
        panel.processor.run_feature(sips.average)

        # Add Gaussian noise to the average
        noise_param = sigima.objects.NormalDistributionParam()
        noise_param.sigma = 0.05
        panel.objview.select_objects([4])
        panel.processor.run_feature(sips.add_gaussian_noise, noise_param)

        # Gaussian fit
        panel.processor.run_feature(sips.gaussian_fit)

        # Make sure the History Panel dock is raised over the Macro Panel
        win.docks[history].raise_()

        if screenshots:
            from datalab.tests.scenarios.tutorial_screenshots import save_screenshot

            win.move(0, 0)
            session = history.tree.topLevelItem(0)
            history.tree.setCurrentItem(session.child(session.childCount() - 1))
            action = history.ui.actions["provenance"]
            assert action.isEnabled()
            menu = QW.QMenu(history)
            add_actions(menu, history.ui.menu_actions)
            menu.popup(history.mapToGlobal(QC.QPoint(230, 120)))
            menu.setActiveAction(action)
            QW.QApplication.processEvents()
            menu.move(history.mapToGlobal(QC.QPoint(230, 120)))
            language = "fr" if os.environ.get("LANG", "").startswith("fr") else "en"
            try:
                save_screenshot(
                    history,
                    Path(execenv.screenshot_path) / f"history_panel.{language}.png",
                    (menu,),
                )
            finally:
                menu.close()


if __name__ == "__main__":
    test_history_panel()
