Installation
============

Dependencies
------------

CodraFT requirements are the following *(Note: the Windows installer package
already include all those required libraries as well as Python itself)*:

.. list-table::
    :header-rows: 1
    :widths: 20, 15, 65

    * - Name
      - Version (min.)
      - Comment
    * - Python language
      - 3.7
      - Python 3.8 is the reference for production release
    * - PyQt
      - 5.15
      - Should work with PySide2/PySide6/PyQt6 as well
    * - QtPy
      - 1.9
      -
    * - h5py
      - 3.0
      -
    * - psutil
      - 5.5
      -
    * - guidata
      - 2.2
      -
    * - guiqwt
      - 4.1
      -
    * - NumPy
      - 1.21
      -
    * - SciPy
      - 1.7
      -
    * - scikit-image
      - 0.18
      -

How to install
--------------

From the installer:
^^^^^^^^^^^^^^^^^^^

CodraFT is available as a stand-alone application, which does not require
any Python distribution to be installed.
Just run the installer and you're good to go!

The installer package is available in the `Releases`_ section.

.. _Releases: https://github.com/CODRA-Ingenierie-Informatique/CodraFT/releases


From the source package::
^^^^^^^^^^^^^^^^^^^^^^^^^

    $ python setup.py install

From the wheel package::
^^^^^^^^^^^^^^^^^^^^^^^^

    $ pip install --upgrade --no-deps codraft-1.7.0-py2.py3-none-any.whl
