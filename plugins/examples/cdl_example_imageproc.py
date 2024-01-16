# -*- coding: utf-8 -*-

"""
Extract blobs (plugin example)
==============================

This is a simple example of a DataLab image processing plugin.

It adds a new menu entry in "Plugins" menu, with a sub-menu "Extract blobs (example)".
This sub-menu contains two actions, "Preprocess image" and "Detect blobs".

.. note::

    This plugin is not installed by default. To install it, copy this file to
    your DataLab plugins directory (see `DataLab documentation
    <https://codra-ingenierie-informatique.github.io/DataLab/en/features/general/plugins.html>`_).
"""

import numpy as np
import skimage.draw

import cdl.obj
import cdl.param
import cdl.plugins


class ExtractBlobs(cdl.plugins.PluginBase):
    """DataLab Example Plugin"""

    PLUGIN_INFO = cdl.plugins.PluginInfo(
        name="Extract blobs (example)",
        version="1.0.0",
        description="This is an example plugin",
    )

    def generate_test_image(self) -> None:
        """Generate test image"""
        # Create a NumPy array:
        arr = np.random.normal(10000, 1000, (2048, 2048))
        for _ in range(10):
            row = np.random.randint(0, arr.shape[0])
            col = np.random.randint(0, arr.shape[1])
            rr, cc = skimage.draw.disk((row, col), 40, shape=arr.shape)
            arr[rr, cc] -= np.random.randint(5000, 6000)
        icenter = arr.shape[0] // 2
        rr, cc = skimage.draw.disk((icenter, icenter), 200, shape=arr.shape)
        arr[rr, cc] -= np.random.randint(5000, 8000)
        data = np.clip(arr, 0, 65535).astype(np.uint16)

        # Create a new image object and add it to the image panel
        image = cdl.obj.create_image("Test image", data, units=("mm", "mm", "lsb"))
        self.proxy.add_object(image)

    def preprocess(self) -> None:
        """Preprocess image"""
        panel = self.imagepanel
        param = cdl.param.BinningParam.create(binning_x=2, binning_y=2)
        panel.processor.compute_binning(param)
        panel.processor.compute_moving_median(cdl.param.MovingMedianParam.create(n=5))

    def detect_blobs(self) -> None:
        """Detect circular blobs"""
        panel = self.imagepanel
        param = cdl.param.BlobOpenCVParam()
        param.filter_by_color = False
        param.min_area = 600.0
        param.max_area = 6000.0
        param.filter_by_circularity = True
        param.min_circularity = 0.8
        param.max_circularity = 1.0
        panel.processor.compute_blob_opencv(param)

    def create_actions(self) -> None:
        """Create actions"""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            acth.new_action(
                "Generate test image",
                triggered=self.generate_test_image,
                select_condition="always",
            )
            acth.new_action("Preprocess image", triggered=self.preprocess)
            acth.new_action("Detect circular blobs", triggered=self.detect_blobs)
