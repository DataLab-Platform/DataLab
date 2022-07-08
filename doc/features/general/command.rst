Command line features
=====================

Run CodraFT
-----------

To run CodraFT from the command line, type the following::

    $ codraft

To show help on command line usage, simply run::

    $ codraft --help
    usage: app.py [-h] [-b path] [-v] [--mode {unattended,screenshot}] [--delay DELAY] [--verbose {quiet,minimal,normal}]
                  [h5]

    Run CodraFT

    positional arguments:
      h5                    HDF5 file names (separated by ';'), optionally with dataset name (separated by ',')

    optional arguments:
      -h, --help            show this help message and exit
      -b path, --h5browser path
                            path to open with HDF5 browser
      -v, --version         show CodraFT version
      --mode {unattended,screenshot}
                            unattended: non-interactive test mode ; screenshot: unattended mode, with automatic
                            screenshots
      --delay DELAY         delay (seconds) before quitting application in unattended mode
      --verbose {quiet,minimal,normal}
                            verbosity level: for debugging/testing purpose

Open HDF5 file at startup
-------------------------

To open HDF5 files, or even import only a specified HDF5 dataset, use the following::

    $ codraft /path/to/file1.h5
    $ codraft /path/to/file1.h5,/path/to/dataset1
    $ codraft /path/to/file1.h5,/path/to/dataset1;/path/to/file2.h5,/path/to/dataset2

Open HDF5 browser at startup
----------------------------

To open the HDF5 browser at startup, use one of the following commands::

    $ codraft -b /path/to/file1.h5
    $ codraft --h5browser /path/to/file1.h5

Run CodraFT demo
----------------

To execute CodraFT demo, run the following::

    $ codraft-demo

Run unit tests
--------------

To execute all CodraFT unit tests, simply run::

    $ codraft-alltests
    *** CodraFT automatic unit tests ***

    Test parameters:
    Selected 39 tests (39 total available)
    Test data path:
        C:\Dev\Projets\CodraFT\codraft\data\tests
    Environment:
        DATA_CODRAFT=C:\Dev\Projets\CodraFT_data\
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

To execute CodraFT interactive tests, run the following::

    $ codraft-tests

.. image:: /images/interactive_tests.png
