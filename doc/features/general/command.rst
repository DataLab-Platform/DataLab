.. _ref-to-command-line-features:

Command line features
=====================

.. meta::
    :description: DataLab command line features
    :keywords: command line, cdl, datalab, run, test, demo

Run DataLab
-----------

To run DataLab from the command line, type the following::

    $ cdl

To show help on command line usage, simply run::

    $ cdl --help
    usage: app.py [-h] [-b path] [-v] [--unattended] [--screenshot] [--delay DELAY] [--xmlrpcport PORT]
                  [--verbose {quiet,minimal,normal}]
                  [h5]

    Run DataLab

    positional arguments:
      h5                    HDF5 file names (separated by ';'), optionally with dataset name (separated by ',')

    optional arguments:
      -h, --help            show this help message and exit
      -b path, --h5browser path
                            path to open with HDF5 browser
      -v, --version         show DataLab version
      --unattended          non-interactive mode
      --screenshot          automatic screenshots
      --delay DELAY         delay (seconds) before quitting application in unattended mode
      --xmlrpcport XMLRPCPORT
                            XML-RPC port number
      --verbose {quiet,minimal,normal}
                            verbosity level: for debugging/testing purpose

Open HDF5 file at startup
-------------------------

To open HDF5 files, or even import only a specified HDF5 dataset, use the following::

    $ cdl /path/to/file1.h5
    $ cdl /path/to/file1.h5,/path/to/dataset1
    $ cdl /path/to/file1.h5,/path/to/dataset1;/path/to/file2.h5,/path/to/dataset2

Open HDF5 browser at startup
----------------------------

To open the HDF5 browser at startup, use one of the following commands::

    $ cdl -b /path/to/file1.h5
    $ cdl --h5browser /path/to/file1.h5

Run DataLab demo
---------------------

To execute DataLab demo, run the following::

    $ cdl-demo

Run unit tests
--------------

.. note::

    This test suite is based on `guidata.guitest` discovery mechanism.
    It is not compatible with `pytest` because most of the high level tests
    have to be executed in a separate process (e.g. scenario tests will fail
    if executed in the same process as other tests).

To execute all DataLab unit tests, simply run::

    $ cdl-alltests

    ================================================================================
    🚀 DataLab v0.9.0 automatic unit tests 🌌
    ================================================================================

    🔥 DataLab characteristics/environment:
    Configuration version: 1.0.0
    Path: C:\Dev\Projets\DataLab\cdl
    Frozen: False
    Debug: False

    🔥 DataLab configuration:
    Process isolation: enabled
    RPC server: enabled
    Console: enabled
    Available memory threshold: 500 MB
    Ignored dependencies: disabled
    Processing:
        Extract all ROIs in a single signal or image
        FFT shift: enabled

    🔥 Test parameters:
    ⚡ Selected 51 tests (51 total available)
    ⚡ Test data path:
        C:\Dev\Projets\DataLab\cdl\data\tests
    ⚡ Environment:
        CDL_DATA=C:\Dev\Projets\DataLab_data\
        PYTHONPATH=.
        DEBUG=

    Please wait while test scripts are executed (a few minutes).
    Only error messages will be printed out (no message = test OK).

    ===[01/51]=== 🍺 Running test [tests\annotations_app.py]
    ===[02/51]=== 🍺 Running test [tests\annotations_unit.py]
    ===[03/51]=== 🍺 Running test [tests\auto_app.py]
    ===[04/51]=== 🍺 Running test [tests\basic1_app.py]
    ===[05/51]=== 🍺 Running test [tests\basic2_app.py]
    ===[06/51]=== 🍺 Running test [tests\basic3_app.py]

Run interactive tests
---------------------

To execute DataLab interactive tests, run the following::

    $ cdl-tests

.. image:: /images/interactive_tests.png
