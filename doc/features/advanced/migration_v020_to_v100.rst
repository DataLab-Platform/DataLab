.. _migration_v020_to_v100:

Migrating from DataLab v0.20 to v1.0
====================================

.. meta::
    :description: Guide for migrating DataLab plugins from v0.20 to v1.0
    :keywords: DataLab, plugin, migration, v0.20, v1.0, Sigima, API changes

.. warning::

    **Critical compatibility notice:**

    DataLab v1.0 introduces **major breaking changes** that are **not backward compatible** with v0.20.

    - **Plugins** developed for v0.20 **will not work** and must be updated
    - **API changes** require code modifications for all custom integrations

    Please read this guide carefully before upgrading to v1.0.

DataLab v1.0 introduces significant architectural changes compared to v0.20. The most important change is the **externalization of signal and image processing features** into a separate library called **Sigima**.

This architectural change improves modularity, testability, and enables the reuse of processing functions in other projects. However, it requires updates to existing plugins to adapt to the new API.

This guide describes the steps to migrate plugins from DataLab v0.20 to v1.0.

.. contents::
   :local:
   :depth: 2

What changed in v1.0?
---------------------

Major architectural changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most significant change in DataLab v1.0 is the creation of the **Sigima library**, which now contains:

- **Signal and image objects** (``SignalObj``, ``ImageObj``, ROI classes)
- **Processing parameters** (all ``*Param`` classes)
- **Processing functions** (``sigima.proc.signal`` and ``sigima.proc.image`` modules)
- **Algorithms** for signal and image processing, or data type conversion, coordinates transformation, ... (``sigima.tools`` module)
- **I/O functions** (reading and writing signals/images)
- **Test data utilities** (test signal and image generation)
- **Result objects** (``TableResult`` and ``GeometryResult``, replacing ``ResultProperties`` and ``ResultShape``)

What remains in DataLab:

- **GUI components** (panels, widgets, plot integration)
- **Plugin system** (``datalab.plugins`` module)
- **Application logic** (main window, configuration, project management)
- **Result metadata adapters** (``TableAdapter``, ``GeometryAdapter`` for storing results in object metadata)

Package renaming
^^^^^^^^^^^^^^^^

The DataLab package was renamed from ``cdl`` to ``datalab`` for clarity:

.. code-block:: python

   # v0.20
   import cdl.obj
   import cdl.param
   from cdl.plugins import PluginBase

   # v1.0
   import sigima.objects
   import sigima.params
   from datalab.plugins import PluginBase

New result objects
^^^^^^^^^^^^^^^^^^

DataLab v1.0 introduces two new immutable result types:

- **TableResult**: Replaces ``ResultProperties`` for tabular scalar results (e.g., statistics, measurements)
- **GeometryResult**: Replaces ``ResultShape`` for geometric results (e.g., detected features, contours)

These new result types are computation-oriented and free of application-specific logic (e.g., Qt, metadata). All metadata-related behaviors have been migrated to the DataLab application layer using **adapter classes** (``TableAdapter``, ``GeometryAdapter``).

Unified processor interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The processor methods have been unified and renamed:

- Most specific ``compute_*`` methods now use the generic ``run_feature()`` method
- Naming conventions updated: ``compute_11`` → ``compute_1_to_1``, ``compute_10`` → ``compute_1_to_0``, etc.

Updating the imports
--------------------

Import path changes
^^^^^^^^^^^^^^^^^^^

The following table gives the equivalence between DataLab v0.20 and v1.0 imports.

The table below shows the mapping between DataLab v0.20 and v1.0 API. For most entries, only the import statement needs to be updated.

.. csv-table:: DataLab v0.20 to v1.0 Compatibility Table
    :file: v020_to_v100.csv
    :header-rows: 1
    :widths: 50, 50
    :delim: ;

Migrating plugin code
---------------------

Example 1: Simple processing plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's how a simple custom filter plugin needs to be updated:

**DataLab v0.20:**

.. code-block:: python

   import numpy as np
   import scipy.ndimage as spi

   import cdl.computation.image as cpi
   import cdl.obj
   import cdl.param
   import cdl.plugins


   def weighted_average_denoise(data: np.ndarray) -> np.ndarray:
       """Apply a custom denoising filter to an image."""
       def filter_func(values: np.ndarray) -> float:
           central_pixel = values[len(values) // 2]
           differences = np.abs(values - central_pixel)
           weights = np.exp(-differences / np.mean(differences))
           return np.average(values, weights=weights)
       return spi.generic_filter(data, filter_func, size=5)


   class CustomFilters(cdl.plugins.PluginBase):
       """DataLab Custom Filters Plugin"""

       PLUGIN_INFO = cdl.plugins.PluginInfo(
           name="My custom filters",
           version="1.0.0",
           description="This is an example plugin",
       )

       def create_actions(self) -> None:
           """Create actions"""
           acth = self.imagepanel.acthandler
           proc = self.imagepanel.processor
           with acth.new_menu(self.PLUGIN_INFO.name):
               for name, func in (("Weighted average denoise", weighted_average_denoise),):
                   # Wrap function to handle ImageObj objects
                   wrapped_func = cpi.Wrap11Func(func)
                   acth.new_action(
                       name, triggered=lambda: proc.compute_11(wrapped_func, title=name)
                   )

**DataLab v1.0:**

.. code-block:: python

   import numpy as np
   import scipy.ndimage as spi
   import sigima.proc.image as sipi

   import datalab.plugins


   def weighted_average_denoise(data: np.ndarray) -> np.ndarray:
       """Apply a custom denoising filter to an image."""
       def filter_func(values: np.ndarray) -> float:
           central_pixel = values[len(values) // 2]
           differences = np.abs(values - central_pixel)
           weights = np.exp(-differences / np.mean(differences))
           return np.average(values, weights=weights)
       return spi.generic_filter(data, filter_func, size=5)


   class CustomFilters(datalab.plugins.PluginBase):
       """DataLab Custom Filters Plugin"""

       PLUGIN_INFO = datalab.plugins.PluginInfo(
           name="My custom filters",
           version="1.0.0",
           description="This is an example plugin",
       )

       def create_actions(self) -> None:
           """Create actions"""
           acth = self.imagepanel.acthandler
           proc = self.imagepanel.processor
           with acth.new_menu(self.PLUGIN_INFO.name):
               for name, func in (("Weighted average denoise", weighted_average_denoise),):
                   # Wrap function to handle ImageObj objects
                   wrapped_func = sipi.Wrap1to1Func(func)
                   acth.new_action(
                       name,
                       triggered=lambda: proc.compute_1_to_1(wrapped_func, title=name),
                   )

**Key changes:**

1. ``cdl.computation.image`` → ``sigima.proc.image``
2. ``cdl.plugins`` → ``datalab.plugins``
3. ``Wrap11Func`` → ``Wrap1to1Func``
4. ``compute_11`` → ``compute_1_to_1``

Example 2: Plugin using built-in features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DataLab v0.20:**

.. code-block:: python

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

       def preprocess(self) -> None:
           """Preprocess image"""
           panel = self.imagepanel
           param = cdl.param.BinningParam.create(sx=2, sy=2)
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
               acth.new_action("Preprocess image", triggered=self.preprocess)
               acth.new_action("Detect circular blobs", triggered=self.detect_blobs)

**DataLab v1.0:**

.. code-block:: python

   import sigima.objects
   import sigima.params

   import datalab.plugins


   class ExtractBlobs(datalab.plugins.PluginBase):
       """DataLab Example Plugin"""

       PLUGIN_INFO = datalab.plugins.PluginInfo(
           name="Extract blobs (example)",
           version="1.0.0",
           description="This is an example plugin",
       )

       def preprocess(self) -> None:
           """Preprocess image"""
           panel = self.imagepanel
           param = sigima.params.BinningParam.create(sx=2, sy=2)
           panel.processor.run_feature("binning", param)
           panel.processor.run_feature(
               "moving_median", sigima.params.MovingMedianParam.create(n=5)
           )

       def detect_blobs(self) -> None:
           """Detect circular blobs"""
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
           """Create actions"""
           acth = self.imagepanel.acthandler
           with acth.new_menu(self.PLUGIN_INFO.name):
               acth.new_action("Preprocess image", triggered=self.preprocess)
               acth.new_action("Detect circular blobs", triggered=self.detect_blobs)

**Key changes:**

1. ``cdl.param`` → ``sigima.params``
2. ``cdl.plugins`` → ``datalab.plugins``
3. ``compute_binning(param)`` → ``run_feature("binning", param)``
4. ``compute_moving_median(param)`` → ``run_feature("moving_median", param)``
5. ``compute_blob_opencv(param)`` → ``run_feature("blob_opencv", param)``

Example 3: Plugin creating objects and handling results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**DataLab v0.20:**

.. code-block:: python

   import numpy as np
   import cdl.obj as dlo
   import cdl.tests.data as test_data
   from cdl.computation import image as cpima
   from cdl.config import _
   from cdl.plugins import PluginBase, PluginInfo


   def add_noise_to_image(src: dlo.ImageObj, p: dlo.NormalRandomParam) -> dlo.ImageObj:
       """Add gaussian noise to image"""
       dst = cpima.dst_11(src, "add_gaussian_noise", f"mu={p.mu},sigma={p.sigma}")
       test_data.add_gaussian_noise_to_image(dst, p)
       return dst


   class PluginTestData(PluginBase):
       """DataLab Test Data Plugin"""

       PLUGIN_INFO = PluginInfo(
           name=_("Test data"),
           version="1.0.0",
           description=_("Testing DataLab functionalities"),
       )

       def add_noise_to_image(self) -> None:
           """Add noise to image"""
           self.imagepanel.processor.compute_11(
               add_noise_to_image,
               paramclass=dlo.NormalRandomParam,
               title=_("Add noise"),
           )

       def create_noisygauss_image(self) -> None:
           """Create 2D noisy gauss image"""
           newparam = self.edit_new_image_parameters(
               hide_image_height=True, hide_image_type=True
           )
           if newparam is not None:
               obj = test_data.create_noisygauss_image(newparam, add_annotations=False)
               self.proxy.add_object(obj)

       def create_actions(self) -> None:
           """Create actions"""
           iah = self.imagepanel.acthandler
           with iah.new_menu(_("Test data")):
               iah.new_action(_("Add noise to image"), triggered=self.add_noise_to_image)
               iah.new_action(
                   _("Create 2D noisy gauss image"),
                   triggered=self.create_noisygauss_image,
                   select_condition="always",
               )

**DataLab v1.0:**

.. code-block:: python

   import numpy as np
   import sigima.objects
   import sigima.tests.data as test_data
   from sigima.proc import image as sipi
   from datalab.config import _
   from datalab.plugins import PluginBase, PluginInfo


   def add_noise_to_image(
       src: sigima.objects.ImageObj, p: sigima.objects.NormalDistribution2DParam
   ) -> sigima.objects.ImageObj:
       """Add gaussian noise to image"""
       dst = sipi.dst_1to1(src, "add_gaussian_noise", f"mu={p.mu},sigma={p.sigma}")
       test_data.add_gaussian_noise_to_image(dst, p)
       return dst


   class PluginTestData(PluginBase):
       """DataLab Test Data Plugin"""

       PLUGIN_INFO = PluginInfo(
           name=_("Test data"),
           version="1.0.0",
           description=_("Testing DataLab functionalities"),
       )

       def add_noise_to_image(self) -> None:
           """Add noise to image"""
           self.imagepanel.processor.compute_1_to_1(
               add_noise_to_image,
               paramclass=sigima.objects.NormalDistribution2DParam,
               title=_("Add noise"),
           )

       def create_noisy_gaussian_image(self) -> None:
           """Create 2D noisy gauss image"""
           newparam = self.edit_new_image_parameters(
               hide_height=True, hide_type=True
           )
           if newparam is not None:
               obj = test_data.create_noisy_gaussian_image(newparam, add_annotations=False)
               self.proxy.add_object(obj)

       def create_actions(self) -> None:
           """Create actions"""
           iah = self.imagepanel.acthandler
           with iah.new_menu(_("Test data")):
               iah.new_action(_("Add noise to image"), triggered=self.add_noise_to_image)
               iah.new_action(
                   _("Create 2D noisy gaussian image"),
                   triggered=self.create_noisy_gaussian_image,
                   select_condition="always",
               )

**Key changes:**

1. ``cdl.obj`` → ``sigima.objects``
2. ``cdl.tests.data`` → ``sigima.tests.data``
3. ``cdl.computation.image`` → ``sigima.proc.image``
4. ``dst_11`` → ``dst_1to1``
5. ``compute_11`` → ``compute_1_to_1``
6. ``dlo.NormalRandomParam`` → ``sigima.objects.NormalDistribution2DParam``
7. ``create_noisygauss_image`` → ``create_noisy_gaussian_image``
8. ``hide_image_height`` → ``hide_height``, ``hide_image_type`` → ``hide_type``

Working with result objects
---------------------------

Changes to result objects
^^^^^^^^^^^^^^^^^^^^^^^^^

The result object architecture has been completely redesigned in v1.0:

**v0.20 Result Objects:**

- ``ResultProperties``: For tabular results (statistics, etc.)
- ``ResultShape``: For geometric results (detected features, etc.)
- These objects contained Qt-specific code and metadata handling logic
- Methods like ``add_to(obj)`` and ``from_metadata_entry()`` were part of the result class

**v1.0 Result Objects:**

- ``TableResult``: Immutable, computation-oriented table results
- ``GeometryResult``: Immutable, computation-oriented geometry results
- No Qt or metadata dependencies
- Metadata handling moved to DataLab adapter classes

Using TableResult and GeometryResult
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In v1.0, when your plugin receives results from ``compute_1_to_0`` operations, you'll work with ``TableResult`` or ``GeometryResult`` objects.

**Example 1: Computing signal dynamic parameters (TableResult)**

This example shows how to compute dynamic parameters (ENOB, SINAD, THD, etc.) and work with the resulting ``TableResult``:

.. code-block:: python

   import sigima.params
   import sigima.proc.signal
   from datalab.adapters_metadata import TableAdapter

   # Get the current signal from the panel
   panel = self.signalpanel
   obj = panel.objview.get_current_object()

   # Compute dynamic parameters
   param = sigima.params.DynamicParam.create(full_scale=1.0)
   table = sigima.proc.signal.dynamic_parameters(obj, param)

   # Access specific values from the table
   enob = table["enob"][0]  # Effective Number of Bits
   sinad = table["sinad"][0]  # Signal-to-Noise and Distortion Ratio
   thd = table["thd"][0]  # Total Harmonic Distortion

   # Convert to dictionary for easy access
   result_dict = table.as_dict()
   print(f"ENOB: {result_dict['enob']}")
   print(f"SINAD: {result_dict['sinad']} dB")

   # Store result in object metadata using adapter
   adapter = TableAdapter(table)
   adapter.add_to(obj)

   # Convert to pandas DataFrame for further processing
   df = table.to_dataframe()
   print(df)

**Example 2: Detecting image peaks (GeometryResult)**

This example shows how to detect peaks in an image and work with the resulting ``GeometryResult``:

.. code-block:: python

   import sigima.params
   import sigima.proc.image
   from datalab.adapters_metadata import GeometryAdapter

   # Get the current image from the panel
   panel = self.imagepanel
   obj = panel.objview.get_current_object()

   # Configure peak detection parameters
   param = sigima.params.Peak2DDetectionParam.create(
       size=50,  # Neighborhood size
       threshold=0.5,  # Relative threshold
       create_rois=True  # Create ROI for each peak
   )

   # Compute peak detection
   geometry = sigima.proc.image.peak_detection(obj, param)

   # Access geometry data
   coords = geometry.coords  # numpy array of shape (n_peaks, 2)
   n_peaks = len(coords)
   print(f"Detected {n_peaks} peaks")

   # Iterate over detected peaks
   for i, (x, y) in enumerate(coords):
       print(f"Peak {i+1}: x={x:.2f}, y={y:.2f}")

   # Check geometry kind
   from sigima.objects import KindShape
   assert geometry.kind == KindShape.POINT

   # Store geometry in object metadata using adapter
   adapter = GeometryAdapter(geometry)
   adapter.add_to(obj)

**Example 3: Computing bandwidth (GeometryResult with segments)**

This example shows how to compute signal bandwidth, which returns a segment geometry:

.. code-block:: python

   import sigima.proc.signal
   from datalab.adapters_metadata import GeometryAdapter

   # Get the current signal from the panel
   panel = self.signalpanel
   obj = panel.objview.get_current_object()

   # Compute -3dB bandwidth
   geometry = sigima.proc.signal.bandwidth_3db(obj)

   # Access segment coordinates [x0, y0, x1, y1]
   x0, y0, x1, y1 = geometry.coords[0]
   print(f"Bandwidth segment: ({x0}, {y0}) to ({x1}, {y1})")

   # Calculate bandwidth length
   bandwidth = geometry.segments_lengths()[0]
   print(f"Bandwidth@-3dB: {bandwidth:.2f}")

   # Store in metadata
   adapter = GeometryAdapter(geometry)
   adapter.add_to(obj)

Retrieving results from metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To retrieve results stored in object metadata:

.. code-block:: python

   from datalab.adapters_metadata import TableAdapter, GeometryAdapter

   obj = ...  # Some signal or image object

   # Iterate over all table results
   for adapter in TableAdapter.iterate_from_obj(obj):
       result = adapter.result  # TableResult instance
       title = adapter.title
       df = result.to_dataframe()

   # Iterate over all geometry results
   for adapter in GeometryAdapter.iterate_from_obj(obj):
       result = adapter.result  # GeometryResult instance
       title = adapter.title
       coords = result.coords

Processor method changes
------------------------

The processor interface has been unified in v1.0. Most specific ``compute_*`` methods now use the generic ``run_feature()`` method.

Using run_feature()
^^^^^^^^^^^^^^^^^^^

The ``run_feature()`` method is a unified interface for running processing features:

.. code-block:: python

   # v1.0 - Generic interface
   proc = panel.processor

   # Simple features (no parameters)
   proc.run_feature("normalize")
   proc.run_feature("fft")

   # Features with parameters
   param = sigima.params.GaussianParam.create(sigma=2.0)
   proc.run_feature("gaussian_filter", param)

   # Features with direct function reference
   import sigima.proc.image as sipi
   proc.run_feature(sipi.normalize)
   proc.run_feature(sipi.gaussian_filter, param)

Method renaming
^^^^^^^^^^^^^^^

Several processor methods have been renamed for consistency:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v0.20
     - v1.0
   * - ``compute_11(func, ...)``
     - ``compute_1_to_1(func, ...)``
   * - ``compute_10(func, ...)``
     - ``compute_1_to_0(func, ...)``
   * - ``compute_n1(func, ...)``
     - ``compute_n_to_1(func, ...)``
   * - ``compute_1n(func, ...)``
     - ``compute_1_to_n(func, ...)``
   * - ``compute_21(func, ...)``
     - ``compute_2_to_1(func, ...)``

Built-in features using run_feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most built-in processing features that previously had dedicated ``compute_*`` methods now use ``run_feature()``:

.. code-block:: python

   # v0.20
   proc.compute_binning(param)
   proc.compute_moving_median(param)
   proc.compute_blob_opencv(param)
   proc.compute_gaussian_filter(param)

   # v1.0
   proc.run_feature("binning", param)
   proc.run_feature("moving_median", param)
   proc.run_feature("blob_opencv", param)
   proc.run_feature("gaussian_filter", param)

Some complex features still have dedicated methods:

.. code-block:: python

   # These still use dedicated methods in v1.0
   proc.compute_roi_extraction(roi)
   proc.compute_multigaussianfit()
   proc.compute_peak_detection(param)

Testing your plugin
-------------------

After migration, thoroughly test your plugin:

1. **Import checks**: Verify all imports work correctly
2. **Parameter creation**: Check that parameter classes are correctly instantiated
3. **Object creation**: Test signal/image object creation functions
4. **Processing operations**: Verify all processing features work as expected
5. **Result handling**: Check that results are correctly generated and stored
6. **GUI integration**: Test that menus and actions appear correctly

Common issues and solutions
---------------------------

Import errors
^^^^^^^^^^^^^

**Problem**: ``ModuleNotFoundError: No module named 'cdl'``

**Solution**: Update all imports from ``cdl.*`` to either ``datalab.*`` or ``sigima.*`` depending on the module.

Attribute errors on processor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: ``AttributeError: 'ImageProcessor' object has no attribute 'compute_binning'``

**Solution**: Replace ``compute_*`` method calls with ``run_feature()`` calls:

.. code-block:: python

   # Wrong (v0.20 style)
   proc.compute_binning(param)

   # Correct (v1.0 style)
   proc.run_feature("binning", param)

Result object incompatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: ``AttributeError: 'TableResult' object has no attribute 'add_to'``

**Solution**: Use adapter classes for metadata operations:

.. code-block:: python

   # Wrong (v0.20 style)
   result.add_to(obj)

   # Correct (v1.0 style)
   from datalab.adapters_metadata import TableAdapter
   TableAdapter(result).add_to(obj)

Missing parameters
^^^^^^^^^^^^^^^^^^

**Problem**: ``ImportError: cannot import name 'SomeParam' from 'datalab'``

**Solution**: Import parameters from ``sigima.params`` instead of ``cdl.param``:

.. code-block:: python

   # Wrong
   from cdl.param import GaussianParam

   # Correct
   from sigima.params import GaussianParam

Additional resources
--------------------

- `DataLab v1.0 documentation <https://datalab-platform.com/>`_
- `Sigima documentation <https://sigima.readthedocs.io/en/latest/>`_
- `Plugin examples <https://github.com/DataLab-Platform/DataLab/tree/main/plugins/examples>`_
- :doc:`DataLab API reference <api>`
- `Sigima API reference <https://sigima.readthedocs.io/en/latest/api/index.html>`_

Getting help
------------

If you encounter issues during migration:

1. Check the `GitHub issue tracker <https://github.com/DataLab-Platform/DataLab/issues>`_
2. Consult the built-in plugin examples in the ``datalab/plugins`` directory
