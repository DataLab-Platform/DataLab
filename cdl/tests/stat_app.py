# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Statistics test

Testing the following:
  - Create a signal
  - Compute statistics on signal and show results
  - Create an image
  - Compute statistics on image and show results
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np

from cdl.config import _
from cdl.core.model import image as imod
from cdl.core.model import signal as smod
from cdl.tests import cdl_app_context, take_plotwidget_screenshot

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run statistics unit test scenario"""
    with cdl_app_context() as win:
        # === Signal statistics test ===
        panel = win.signalpanel
        snew = smod.new_signal_param(_("Gaussian"), stype=smod.SignalTypes.GAUSS)
        addparam = smod.GaussLorentzVoigtParam()
        sig = smod.create_signal_from_param(snew, addparam=addparam, edit=False)
        panel.add_object(sig)
        panel.processor.compute_stats()
        sig.roi = np.array([[len(sig.x) // 2, len(sig.x) - 1]], int)
        take_plotwidget_screenshot(panel, "stat_test")
        panel.processor.compute_stats()
        # === Image statistics test ===
        panel = win.imagepanel
        inew = imod.new_image_param(_("Raw data (2D-Gaussian)"), imod.ImageTypes.GAUSS)
        addparam = imod.Gauss2DParam()
        ima = imod.create_image_from_param(inew, addparam=addparam, edit=False)
        dy, dx = ima.size
        ima.roi = np.array(
            [
                [dx // 2, 0, dx, dy],
                [0, 0, dx // 3, dy // 3],
                [dx // 2, dy // 2, dx, dy],
            ],
            int,
        )
        panel.add_object(ima)
        take_plotwidget_screenshot(panel, "stat_test")
        panel.processor.compute_stats()


if __name__ == "__main__":
    test()
