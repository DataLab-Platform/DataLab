# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History Panel application test
(essentially for the screenshot...)

Records a representative sequence of UI and computation actions and grabs
a screenshot of the history panel, used in the documentation
(:ref:`historypanel`).
"""

# guitest: show

import sigima.objects
import sigima.proc.signal as sips

from datalab import config
from datalab.tests import datalab_test_app_context
from datalab.utils import qthelpers as qth


def test_history_panel(screenshots: bool = False) -> None:
    """Record a representative session and grab the History Panel screenshot."""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    with datalab_test_app_context(console=False, exec_loop=not screenshots) as win:
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
            qth.grab_save_window(history, "history_panel", add_timestamp=False)


if __name__ == "__main__":
    test_history_panel()
