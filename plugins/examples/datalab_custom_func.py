# -*- coding: utf-8 -*-

"""
Custom denoising filter plugin
==============================

This is a simple example of a DataLab image processing plugin.

It is part of the DataLab custom function tutorial.

.. note::

    This plugin is not installed by default. To install it, copy this file to
    your DataLab plugins directory (see `DataLab documentation
    <https://datalab-platform.com/en/features/advanced/plugins.html>`_).
"""

import numpy as np
import scipy.ndimage as spi
import sigima.proc.image as sipi

import datalab.plugins


def weighted_average_denoise(data: np.ndarray) -> np.ndarray:
    """Apply a custom denoising filter to an image.

    This filter averages the pixels in a 5x5 neighborhood, but gives less weight
    to pixels that significantly differ from the central pixel.
    """

    def filter_func(values: np.ndarray) -> float:
        """Filter function"""
        central_pixel = values[len(values) // 2]
        differences = np.abs(values - central_pixel)
        weights = np.exp(-differences / np.mean(differences))
        return np.average(values, weights=weights)

    return spi.generic_filter(data, filter_func, size=5)


class CustomFilters(datalab.plugins.PluginBase):
    """DataLab Custom Filters Plugin"""

    FEATURE_ID = "org.datalab.examples.custom-filters.weighted-average-denoise"
    PLUGIN_INFO = datalab.plugins.PluginInfo(
        id="org.datalab.examples.custom-filters",
        name="My custom filters",
        version="1.0.0",
        description="This is an example plugin",
    )

    def register_computations(self) -> None:
        """Register computations owned by this plugin."""
        wrapped_func = sipi.Wrap1to1Func(weighted_average_denoise)
        self.imagepanel.processor.register_1_to_1(
            wrapped_func,
            "Weighted average denoise",
            feature_id=self.FEATURE_ID,
            owner_plugin_id=self.plugin_id,
        )

    def create_actions(self) -> None:
        """Create actions"""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            acth.action_for(self.FEATURE_ID)
