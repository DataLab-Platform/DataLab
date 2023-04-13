Command line features
=====================

Run CobraDataLab
----------------

To run CobraDataLab from the command line, type the following::

    $ cdl

To show help on command line usage, simply run::

    $ cdl --help
    usage: app.py [-h] [-b path] [-v] [--mode {unattended,screenshot}] [--delay DELAY] [--verbose {quiet,minimal,normal}]
                  [h5]

    Run CobraDataLab

    positional arguments:
      h5                    HDF5 file names (separated by ';'), optionally with dataset name (separated by ',')

    optional arguments:
      -h, --help            show this help message and exit
      -b path, --h5browser path
                            path to open with HDF5 browser
      -v, --version         show CobraDataLab version
      --mode {unattended,screenshot}
                            unattended: non-interactive test mode ; screenshot: unattended mode, with automatic
                            screenshots
      --delay DELAY         delay (seconds) before quitting application in unattended mode
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

Run CobraDataLab demo
---------------------

To execute CobraDataLab demo, run the following::

    $ cdl-demo

Run unit tests
--------------

To execute all CobraDataLab unit tests, simply run::

    $ cdl-alltests
    *** CobraDataLab automatic unit tests ***

    Test parameters:
    Selected 39 tests (39 total available)
    Test data path:
        C:\Dev\Projets\CobraDataLab\cdl\data\tests
    Environment:
        DATA_CDL=C:\Dev\Projets\CDL_data\
        PYTHONPATH=.
        DEBUG=

    Please wait while test scripts are executed (a few minutes).
    Only error messages will be printed out (no message = test OK).

    ===[01/39]=== 🍺 Running test [tests\annotations_app.py]
    ===[02/39]=== 🍺 Running test [tests\annotations_unit.py]
    ===[03/39]=== 🍺 Running test [tests\auto_app.py]
    ===[04/39]=== 🍺 Running test [tests\basic1_app.py]
    ===[05/39]=== 🍺 Running test [tests\basic2_app.py]
    ===[06/39]=== 🍺 Running test [tests\basic3_app.py]

Run interactive tests
---------------------

To execute CobraDataLab interactive tests, run the following::

    $ cdl-tests

.. image:: /images/interactive_tests.png
