.. _validation_status:

Validation Status of DataLab
============================

.. meta::
    :description: Validation in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, scientific, data, analysis, validation, ground-truth, analytical

Functional validation
---------------------

In DataLab, functional validation is based on a classic test strategy (see
:ref:`functional_validation`).

Test coverage is around 90%, with more than 200 tests.

Scientific validation
---------------------

This paragraph provides the validation status of compute functions in DataLab (this is
what we call scientific validation, see :ref:`scientific_validation`).

.. note:: This is a work in progress: the tables below are updated continuously as new
    functions are validated or test code is adapted (the tables are generated from the
    test code). Some functions are already validated but do not appear in the list
    below yet, while others are still in the validation process.

.. warning:: The validation status must not be confused with the test coverage. The
    validation status indicates whether the function has been validated against
    ground-truth data or analytical models. The test coverage indicates the percentage
    of the code that is executed by the test suite, but it does not necessarily take
    into account the correctness of the results (DataLab's test coverage is around 90%).

.. csv-table:: Validation Statistics
   :file: ../../validation_statistics.csv
   :header: Category, Signal, Image, Total

Signal Compute Functions
^^^^^^^^^^^^^^^^^^^^^^^^

The table below shows the validation status of signal compute functions in DataLab.
It is automatically generated from the source code.

.. csv-table:: Validation status of signal compute functions
   :file: ../../validation_status_signal.csv
   :header: Compute function, Description, Test function
   :widths: 30, 40, 30

Image Compute Functions
^^^^^^^^^^^^^^^^^^^^^^^

The table below shows the validation status of image compute functions in DataLab.
It is automatically generated from the source code.

.. csv-table:: Validation status of image compute functions
    :file: ../../validation_status_image.csv
    :header: Compute function, Description, Test function
    :widths: 30, 40, 30
