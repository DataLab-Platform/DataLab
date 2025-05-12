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
    <https://datalab-platform.com/en/features/general/plugins.html>`_).
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
        newparam = self.edit_new_image_parameters(
            title="Test image", hide_image_dtype=True, shape=(2048, 2048)
        )
        if newparam is not None:
            # Create a NumPy array:
            shape = (newparam.height, newparam.width)
            arr = np.random.normal(10000, 1000, shape)
            for _ in range(10):
                row = np.random.randint(0, shape[0])
                col = np.random.randint(0, shape[1])
                rr, cc = skimage.draw.disk((row, col), min(shape) // 50, shape=shape)
                arr[rr, cc] -= np.random.randint(5000, 6000)
            center = (shape[0] // 2,) * 2
            rr, cc = skimage.draw.disk(center, min(shape) // 10, shape=shape)
            arr[rr, cc] -= np.random.randint(5000, 8000)
            data = np.clip(arr, 0, 65535).astype(np.uint16)

            # Create a new image object and add it to the image panel
            obj = cdl.obj.create_image(newparam.title, data, units=("mm", "mm", "lsb"))
            self.proxy.add_object(obj)

    def preprocess(self) -> None:
        """Preprocess image"""
        panel = self.imagepanel
        param = cdl.param.BinningParam.create(sx=2, sy=2)
        panel.processor.compute("binning", param)
        panel.processor.compute(
            "moving_median", cdl.param.MovingMedianParam.create(n=5)
        )

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
        panel.processor.compute("blob_opencv", param)

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
