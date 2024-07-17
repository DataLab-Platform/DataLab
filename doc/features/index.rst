Features
========

.. meta::
    :description: DataLab features
    :keywords: DataLab, features, validation, signal processing, image processing

The following sections describe the features of DataLab, the open-source scientific
data analysis and visualization platform.

First, the **validation** section explains why and how DataLab is validated - that is
another singular and fundamental feature of DataLab.

Then, the **general features** section describes the general features of DataLab, which
concern both the signal and image processing panels.

The **signal processing** and **image processing** sections describe the features
specific to the signal and image processing panels, respectively.

.. note::

    Before jumping into the details of other parts of the documentation, it is
    recommended to start with the :ref:`workspace` page, which describes the
    basic concepts of DataLab.

.. seealso::

    For a synthetic overview of the features of DataLab, please refer to the
    :ref:`key_features` page.

.. _validation:

Validation
----------

DataLab is a platform for scientific data analysis and visualization. It may be used
in a variety of scientific disciplines, including biology, physics, and astronomy. The
common ground for all these disciplines is the need to validate the results of
computational analysis against ground-truth data. This is a critical step in the
scientific method, and it is essential for reproducibility and trust in the results.
This is what we call **scientific validation**.

DataLab is also used in industrial applications, where the validation of the results
is essential for guaranteeing the quality of the process but a wider range of validation
is required to ensure that a maximum of use cases are covered from a functional point
of view. This is what we call **functional validation**.

Thus, DataLab validation may be categorized into two types:

- **Scientific validation**: ensures that the results of computational analysis
  are accurate and reliable.

- **Functional validation**: checks that the software behaves as expected in a
  variety of use cases (classic automated unit test suite).

.. toctree::
   :maxdepth: 1
   :caption: Validation:

   validation/scientific
   validation/functional
   validation/status

General features
----------------

This section describes the general features of DataLab, which concern both the
signal and image processing panels.

.. figure:: /images/DataLab-Screenshot.png

    DataLab main window

.. toctree::
   :maxdepth: 1
   :caption: General features:

   general/workspace
   general/h5browser
   general/macros
   general/remote
   general/model
   general/plugins
   general/logviewer
   general/instviewer
   general/settings
   general/command

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
   signal/menu_operations
   signal/menu_processing
   signal/menu_computing
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
   image/menu_operations
   image/menu_processing
   image/menu_computing
   image/menu_view
   image/2d_peak_detection
   image/contour_detection
