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
    <https://datalab-platform.com/en/features/advanced/plugins.html>`_).

Usage
-----

1. Copy this file to your DataLab plugins directory
2. Restart DataLab or use "Plugins > Reload plugins"
3. Use "Generate test image" to create a test image with blobs
4. Apply "Preprocess image" to prepare the image
5. Use "Detect circular blobs" to find blobs

Key Concepts
------------

**Using the Processor:**

Plugins can run DataLab processing operations programmatically:

- ``panel.processor.run_feature(operation_name, param)``
  Runs a processing operation on the selected object(s)

**Common Operations:**

- Binning: ``panel.processor.run_feature("binning",
  BinningParam.create(sx=2, sy=2))``
- Filtering: ``panel.processor.run_feature("moving_median",
  MovingMedianParam.create(n=5))``
- Detection: ``panel.processor.run_feature("blob_opencv",
  BlobOpenCVParam(...))``

**Creating Objects:**

Use ``sigima.objects`` to create signals or images:

- ``create_signal(title, x, y)``: Create signal
- ``create_image(title, data, units=None)``: Create image

Then add to DataLab: ``self.proxy.add_object(obj)``

**Submenus for Organization:**

Group related actions in submenus for better UI organization:

- Test data generation → Direct action
- Processing pipeline → Submenu with multiple steps

See Also
--------

- DataLab plugin documentation: https://datalab-platform.com/en/features/advanced/plugins.html
- Sigima computation library: https://sigima.readthedocs.io/
- ``datalab_example_dialogs.py``: Dialog methods
"""

import numpy as np
import sigima.objects
import sigima.params
import skimage.draw

import datalab.plugins


class ExtractBlobs(datalab.plugins.PluginBase):
    """DataLab Example Plugin"""

    PLUGIN_INFO = datalab.plugins.PluginInfo(
        name="Extract blobs (example)",
        version="1.0.0",
        description="This is an example plugin",
    )

    def generate_test_image(self) -> None:
        """Generate test image with circular blobs

        Creates a synthetic image with:
        - Gaussian noise background
        - 10 random small blobs
        - 1 large central blob
        """
        newparam = self.edit_new_image_parameters(
            title="Test image", hide_dtype=True, shape=(2048, 2048)
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
            obj = sigima.objects.create_image(
                newparam.title, data, units=("mm", "mm", "lsb")
            )
            self.proxy.add_object(obj)

    def preprocess(self) -> None:
        """Preprocess image

        Apply processing pipeline:
        1. Binning (2x2) - reduce size and noise
        2. Moving median filter (5x5) - smooth image

        Note: processor.run_feature() operates on currently selected image(s)
        """
        panel = self.imagepanel
        param = sigima.params.BinningParam.create(sx=2, sy=2)
        panel.processor.run_feature("binning", param)
        panel.processor.run_feature(
            "moving_median", sigima.params.MovingMedianParam.create(n=5)
        )

    def detect_blobs(self) -> None:
        """Detect circular blobs using OpenCV detector

        Detection criteria:
        - Area: 600-6000 pixels
        - Circularity: 0.8-1.0 (nearly circular)
        - Color filtering: disabled

        Results are shown as ROIs (Regions of Interest) on the image.
        """
        panel = self.imagepanel
        param = sigima.params.BlobOpenCVParam()
        param.filter_by_color = False
        param.min_area = 600.0
        param.max_area = 6000.0
        param.filter_by_circularity = True
        param.min_circularity = 0.8
        param.max_circularity = 1.0
        panel.processor.run_feature("blob_opencv", param)

    def create_actions(self) -> None:
        """Create actions

        Menu structure:
        - Generate test image (always enabled)
        - Processing Pipeline (submenu)
          - Preprocess image (enabled when image selected)
          - Detect circular blobs (enabled when image selected)
        """
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            # Always enabled action (doesn't require image selection)
            acth.new_action(
                "Generate test image",
                triggered=self.generate_test_image,
                select_condition="always",
            )

            # Submenu for related processing steps
            # Actions use default select_condition (requires ≥1 image selected)
            with acth.new_menu("Processing Pipeline"):
                acth.new_action("Preprocess image", triggered=self.preprocess)
                acth.new_action("Detect circular blobs", triggered=self.detect_blobs)
