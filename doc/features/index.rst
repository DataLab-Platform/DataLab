Features
========

.. meta::
    :description: DataLab features
    :keywords: DataLab, features, validation, signal processing, image processing

The following sections describe the features of DataLab, the open-source scientific
data analysis and visualization platform.

.. note::

    Before jumping into the details of other parts of the documentation, it is
    recommended to start with the :ref:`overview` page, which describes the
    basic concepts of DataLab.

.. seealso::

    For a synthetic overview of the features of DataLab, please refer to the
    :ref:`key_features` page.

.. _validation:

Overview & Common features
--------------------------

.. figure:: /images/shots/i_blob_detection_flower.png

    DataLab main window

.. toctree::
   :maxdepth: 1
   :caption: Overview & Common features:

   common/overview
   common/settings
   common/h5browser

.. raw:: latex

    \newpage

Signal processing
-----------------

This section describes the features specific to the signal processing panel. The
signal processing panel is the default panel when DataLab is started.

.. figure:: /images/shots/s_beautiful.png

    DataLab main window: Signal processing view

.. toctree::
   :maxdepth: 1
   :caption: Signal processing:

   signal/menu_file
   signal/menu_create
   signal/menu_edit
   signal/menu_roi
   signal/menu_operations
   signal/menu_processing
   signal/menu_analysis
   signal/menu_view

.. raw:: latex

    \newpage

Image processing
----------------

This section describes the features specific to the image processing panel. The
image processing panel can be selected by clicking on the "Images" tab at the
bottom-right of the DataLab main window.

.. figure:: /images/shots/i_beautiful.png

    DataLab main window: Image processing view

.. toctree::
   :maxdepth: 1
   :caption: Image processing:

   image/menu_file
   image/menu_create
   image/menu_edit
   image/menu_roi
   image/menu_operations
   image/menu_processing
   image/menu_analysis
   image/menu_view
   image/2d_peak_detection
   image/contour_detection

Validation
----------

DataLab is a platform for scientific data analysis and visualization. It may be used
in a variety of scientific disciplines, including biology, physics, and astronomy. The
common ground for all these disciplines is the need to validate the results of
computational analysis against ground-truth data. This is a critical step in the
scientific method, and it is essential for reproducibility and trust in the results.
This is what we call **technical validation**.

DataLab is also used in industrial applications, where the validation of the results
is essential for guaranteeing the quality of the process but a wider range of validation
is required to ensure that a maximum of use cases are covered from a functional point
of view. This is what we call **functional validation**.

Thus, DataLab validation may be categorized into two types:

- **Technical validation**: ensures that the results of computational analysis
  are accurate and reliable.

- **Functional validation**: checks that the software behaves as expected in a
  variety of use cases (classic automated unit test suite).

.. toctree::
   :maxdepth: 1
   :caption: Validation:

   validation/technical
   validation/functional
   validation/status

Advanced features
-----------------

This section describes the advanced features of DataLab.

.. toctree::
   :maxdepth: 1
   :caption: Advanced features:

   advanced/plugins
   advanced/migration_v020_to_v100
   advanced/macros
   advanced/remote
   advanced/model
   advanced/logviewer
   advanced/instviewer
   advanced/command
   advanced/api
