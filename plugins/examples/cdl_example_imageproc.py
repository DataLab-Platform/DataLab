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
    <https://cdlapp.readthedocs.io/en/latest/features/general/plugins.html>`_).
"""

import cdl.param
import cdl.plugins


class ExtractBlobs(cdl.plugins.PluginBase):
    """DataLab Example Plugin"""

    PLUGIN_INFO = cdl.plugins.PluginInfo(
        name="Extract blobs (example)",
        version="1.0.0",
        description="This is an example plugin",
    )

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
        param.filter_by_circularity = True
        param.min_circularity = 0.8
        param.max_circularity = 1.0
        panel.processor.compute_blob_opencv(param)

    def create_actions(self) -> None:
        """Create actions"""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            acth.new_action("Preprocess image", triggered=self.preprocess)
            acth.new_action("Detect circular blobs", triggered=self.detect_blobs)
