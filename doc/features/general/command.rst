.. _ref-to-command-line-features:

Command line features
=====================

.. meta::
    :description: DataLab command line features
    :keywords: command line, datalab, run, test, demo

Run DataLab
-----------

To run DataLab from the command line, type the following::

    $ datalab

To show help on command line usage, simply run::

    $ datalab --help
    usage: app.py [-h] [-b path] [-v] [--unattended] [--screenshot] [--delay DELAY] [--xmlrpcport PORT]
                  [--verbose {quiet,minimal,normal}]
                  [h5]

    Run DataLab

    positional arguments:
    h5                    HDF5 file names (separated by ';'), optionally with dataset name (separated by ',')

    options:
    -h, --help            show this help message and exit
    -b path, --h5browser path
                            path to open with HDF5 browser
    -v, --version         show DataLab version
    --reset               reset DataLab configuration
    --unattended          non-interactive mode
    --accept_dialogs      accept dialogs in unattended mode
    --screenshot          automatic screenshots
    --delay DELAY         delay (ms) before quitting application in unattended mode
    --xmlrpcport XMLRPCPORT
                            XML-RPC port number
    --verbose {quiet,normal,debug}
                            verbosity level: for debugging/testing purpose

Open HDF5 file at startup
-------------------------

To open HDF5 files, or even import only a specified HDF5 dataset, use the following::

    $ datalab /path/to/file1.h5
    $ datalab /path/to/file1.h5,/path/to/dataset1
    $ datalab /path/to/file1.h5,/path/to/dataset1;/path/to/file2.h5,/path/to/dataset2

Open HDF5 browser at startup
----------------------------

To open the HDF5 browser at startup, use one of the following commands::

    $ datalab -b /path/to/file1.h5
    $ datalab --h5browser /path/to/file1.h5

Run DataLab demo
----------------

To execute DataLab demo, run the following::

    $ datalab-demo

.. _run_scientific_validation_tests:

Run technical validation tests
-------------------------------

.. note:: Technical validation tests are directly included in individual unit tests
    and are disseminated throughout the code. The test functions including validation
    tests are marked with the ``@pytest.mark.validation`` decorator.

To execute DataLab technical validation tests, run the following::

    $ pytest -m validation

.. seealso:: See section :ref:`validation` for more information on DataLab's validation strategy.

.. _run_functional_validation_tests:

Run complete test suite
------------------------

To execute all DataLab unit tests, simply run::

    $ pytest

Run interactive tests
---------------------

To execute DataLab interactive tests, run the following::

    $ datalab-tests

.. image:: /images/interactive_tests.png
