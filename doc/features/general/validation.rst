.. _validation:

Validation
==========

.. meta::
    :description: Validation in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, scientific, data, analysis, validation, ground-truth, analytical

DataLab is a platform for scientific data analysis and visualization. It may be used
in a variety of scientific disciplines, including biology, physics, and astronomy. The
common ground for all these disciplines is the need to validate the results of
computational analysis against ground-truth data. This is a critical step in the
scientific method, and it is essential for reproducibility and trust in the results.

DataLab validation is based on two key concepts: **ground-truth data** and **analytical
validation**.

Ground-truth data
-----------------

Ground-truth data is data that is known to be correct. It is used to validate the
results of computational analysis.

In DataLab, ground-truth data may be obtained from a variety of sources, including:

- Experimental data
- Simulated data
- Synthetic data
- Data from a trusted source

Analytical validation
---------------------

Analytical validation is the process of comparing the results of computational analysis
to ground-truth data. This is done to ensure that the results are accurate and reliable.

In DataLab, analytical validation is implemented using a variety of techniques, including:

- Cross-validation with an analytical model (from a trusted source, e.g. `SciPy <https://www.scipy.org/>`_ or `NumPy <https://numpy.org/>`)
- Statistical analysis
- Visual inspection
- Expert review

Validation tests
----------------

How to execute tests
~~~~~~~~~~~~~~~~~~~~

In DataLab, validation tests are disseminated in the test suite of the project, but
they can also be executed separately using the command line interface.

.. seealso:: See paragraph :ref:`run_validation_tests` for more information on how to run validation tests.

Validation Status of DataLab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paragraph provides the validation status of compute functions in the DataLab project.

.. note:: This is a work in progress: the validation status of compute functions is
    being updated continuously. Some functions are already validated but do not appear
    in the list below yet, while others are still in the validation process.

.. csv-table:: Validation Statistics
   :file: ../../validation_statistics.csv
   :header-rows: 1

.. csv-table:: Validation status of compute functions
   :file: ../../validation_status.csv
   :header-rows: 1
   :widths: 20, 40, 40
