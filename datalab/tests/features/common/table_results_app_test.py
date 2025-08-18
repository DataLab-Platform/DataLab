# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result properties application test
"""

# guitest: show

from __future__ import annotations

import numpy as np
from sigima.tests import data as test_data

from datalab.adapters_metadata.table_adapter import TableAdapter
from datalab.tests import datalab_test_app_context


def create_image_with_table_results():
    """Create test image with table results"""
    image = test_data.create_multigaussian_image()
    for table in test_data.generate_table_results():
        TableAdapter(table).add_to(image)
    return image


def test_table_results():
    """Result properties application test"""
    obj1 = test_data.create_sincos_image()
    obj2 = create_image_with_table_results()
    with datalab_test_app_context() as win:
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
    test_table_results()
