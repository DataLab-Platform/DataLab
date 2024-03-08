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

    $ pytest

Run interactive tests
---------------------

To execute DataLab interactive tests, run the following::

    $ cdl-tests

.. image:: /images/interactive_tests.png
