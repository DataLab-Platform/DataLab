.. _scientific_validation:

Technical Validation
=====================

.. meta::
    :description: Validation in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, scientific, data, analysis, validation, ground-truth, analytical

DataLab technical validation is based on two key concepts:
**ground-truth data** and **analytical validation**.

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

- Cross-validation with an analytical model (from a trusted source,
  e.g. `SciPy <https://www.scipy.org/>`_ or `NumPy <https://numpy.org/>`_)
- Statistical analysis
- Visual inspection
- Expert review

Scope
-----

The scope of technical validation in DataLab includes all compute functions that
operate on DataLab's signal and image objects (i.e. :class:`cdl.obj.SignalObj` and
:class:`cdl.obj.ImageObj`).

This includes functions for (all functions are named ``compute_<function_name>``):

- Signal processing (:mod:`cdl.computation.signal`)
- Image processing (:mod:`cdl.computation.image`)

Implementation
--------------

The tests are implemented using the `pytest <https://docs.pytest.org/en/latest/>`_
framework.

When writing a new technical validation test, the following rules should be followed
regarding the test function:

- The test function should be named:

  - ``test_signal_<function_name>`` for signal compute functions
  - ``test_image_<function_name>`` for image compute functions

.. note::

    The ``signal`` or ``image`` prefix is used to indicate the type of object that the
    function operates on. It may be omitted if the function operates exclusively on
    one type of object (e.g. ``test_adjust_gamma`` is the test function for the
    ``compute_adjust_gamma`` function, which operates on images).

- The test function should be marked with the ``@pytest.mark.validation`` decorator.

Following those rules ensures that:

- The tests are easily identified as technical validation tests.

- The tests can be executed separately using the command line interface
  (see :ref:`run_scientific_validation_tests`).

- The tests are automatically discovered for synthetizing the validation status of
  the compute functions (see :ref:`validation_status`).

Executing tests
---------------

In DataLab, technical validation tests are disseminated in the test suite of the
project, but they can also be executed separately using the command line interface.

.. seealso::

    See paragraph :ref:`run_scientific_validation_tests` for more information on
    how to run technical validation tests.