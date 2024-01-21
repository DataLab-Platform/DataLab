# -*- coding: utf-8 -*-

"""
Custom denoising filter plugin
==============================

This is a simple example of a DataLab image processing plugin.

It is part of the DataLab custom function tutorial.

.. note::

    This plugin is not installed by default. To install it, copy this file to
    your DataLab plugins directory (see `DataLab documentation
    <https://codra-ingenierie-informatique.github.io/DataLab/en/features/general/plugins.html>`_).
"""

import numpy as np
import scipy.ndimage as spi

import cdl.core.computation.image as cpi
import cdl.obj
import cdl.param
import cdl.plugins


def weighted_average_denoise(values: np.ndarray) -> float:
    """Apply a custom denoising filter to an image.

    This filter averages the pixels in a 5x5 neighborhood, but gives less weight
    to pixels that significantly differ from the central pixel.
    """
    central_pixel = values[len(values) // 2]
    differences = np.abs(values - central_pixel)
    weights = np.exp(-differences / np.mean(differences))
    return np.average(values, weights=weights)


def compute_weighted_average_denoise(src: cdl.obj.ImageObj) -> cdl.obj.ImageObj:
    """Compute Weighted average denoise

    This function is a wrapper around the ``weighted_average_denoise`` function,
    allowing to process an image object (instead of a NumPy array).
    It is required to use the function in DataLab.

    Args:
        src (ImageObj): input image object
    Returns:
        ImageObj: output image object
    """
    dst = cpi.dst_11(src, weighted_average_denoise.__name__)
    dst.data = spi.generic_filter(src.data, weighted_average_denoise, size=5)
    return dst


class CustomFilters(cdl.plugins.PluginBase):
    """DataLab Custom Filters Plugin"""

    PLUGIN_INFO = cdl.plugins.PluginInfo(
        name="My custom filters",
        version="1.0.0",
        description="This is an example plugin",
    )

    def apply_func(self, func, title) -> None:
        """Apply custom function"""
        self.imagepanel.processor.compute_11(func, title=title)

    def create_actions(self) -> None:
        """Create actions"""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            for name, func in (
                ("Weighted average denoise", compute_weighted_average_denoise),
            ):
                acth.new_action(name, triggered=lambda: self.apply_func(func, name))
