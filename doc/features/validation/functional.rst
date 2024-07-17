.. _functional_validation:

Functional validation
=====================

.. meta::
    :description: Validation in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, scientific, data, analysis

Strategy
--------

DataLab functional validation is based a classic test strategy, with a strong emphasis
on automated testing. Apart from one or two manual tests (e.g. load test), all tests
are automated (more than 99% of the tests are automated).

Writing tests follows the TDD (Test-Driven Development) principle:

- When *a new feature is developed*, the developer writes the tests first. The tests
  are then executed to ensure that they fail. The developer then implements the feature,
  and the tests are executed again to ensure that they pass.

- When *a bug is reported*, the developer writes a test that reproduces the bug.
  The test is executed to ensure that it fails. The developer then fixes the bug,
  and the test is executed again to ensure that it passes.

Depending on the abstraction level, unit tests and/or application tests are written.
When writing both types of tests, the developer starts with the unit tests and then
writes the application tests.

Types of tests
--------------

The functional validation of DataLab is based on two main types of tests:

- **Unit tests** (test scripts named ``*_unit_test.py``): Test individual functions or
  methods. All unit tests are automated.

- **Application tests** (test scripts named ``*_app_test.py``): Test the interaction
  between components (integration tests), or the application as a whole.
  All application tests are automated.

Implementation
--------------

The tests are implemented using the `pytest <https://docs.pytest.org/en/latest/>`_
framework. Many existing tests may be derived from to create new tests.

Executing tests
---------------

To execute the tests, the developer uses the command line interface.
See section :ref:`run_functional_validation_tests` for more information on how to run
functional validation tests.