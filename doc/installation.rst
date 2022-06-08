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
      - 4.2
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

Windows installer:
^^^^^^^^^^^^^^^^^^

CodraFT is available as a stand-alone application for Windows,
which does not require any Python distribution to be installed.
Just run the installer and you're good to go!

.. image:: /images/shots/windows_installer.png

The installer package is available in the `Releases`_ section.
It supports automatic uninstall and upgrade feature (no need to uninstall
CodraFT before runinng the installer of another version of the application).

.. _Releases: https://github.com/CODRA-Ingenierie-Informatique/CodraFT/releases


Wheel package:
^^^^^^^^^^^^^^

On any operating system, using pip and the Wheel package is the easiest way to
install CodraFT on an existing Python distribution:

.. code-block:: console

    $ pip install --upgrade codraft-2.0.2-py2.py3-none-any.whl


Source package:
^^^^^^^^^^^^^^^

Installing CodraFT directly from the source package is straigthforward:

.. code-block:: console

    $ python setup.py install
