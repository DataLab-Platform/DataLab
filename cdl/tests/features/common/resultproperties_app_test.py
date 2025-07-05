# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result properties application test
"""

# guitest: show

from __future__ import annotations

import numpy as np
from sigima.tests import data as test_data

from cdl.tests import cdltest_app_context


def create_image_with_resultproperties():
    """Create test image with result properties"""
    image = test_data.create_multigauss_image()
    for prop in test_data.create_resultproperties():
        prop.add_to(image)
    return image


def test_resultproperties():
    """Result properties application test"""
    obj1 = test_data.create_sincos_image()
    obj2 = create_image_with_resultproperties()
    with cdltest_app_context(console=False) as win:
        panel = win.signalpanel
        noiseparam = test_data.GaussianNoiseParam()
        for sigma in np.linspace(0.0, 0.5, 11):
            noiseparam.sigma = sigma
            sig = test_data.create_noisy_signal(noiseparam=noiseparam)
            panel.add_object(sig)
            panel.processor.run_feature("dynamic_parameters")
            panel.processor.run_feature("contrast")
        panel.objview.selectAll()
        panel.show_results()
        panel.plot_results()
        win.set_current_panel("image")
        panel = win.imagepanel
        for obj in (obj1, obj2):
            panel.add_object(obj)
        panel.show_results()
        panel.plot_results()


if __name__ == "__main__":
    test_resultproperties()
