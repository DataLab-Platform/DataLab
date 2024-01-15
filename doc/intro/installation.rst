Installation
============

How to install
--------------

DataLab is available in several forms:

-   As a Python package, which can be installed on GNU/Linux, macOS and
    Windows using the package manager ``pip``.

-   As a stand-alone application for Windows, which does not require any
    Python distribution to be installed. Just run the installer and
    you're good to go!

-   As a source package, which can be installed on GNU/Linux, macOS and
    Windows using ``pip`` or manually.

Package manager
^^^^^^^^^^^^^^^

.. note::

    Supported platforms: GNU/Linux, macOS, Windows

DataLab's package ``cdl`` is available on the Python Package Index (PyPI)
on the following URL: https://pypi.python.org/pypi/cdl.

Installing DataLab from PyPI is as simple as running this command:

.. code-block:: console

    $ pip install cdl

.. note::

    If you already have a previous version of DataLab installed, you can
    upgrade it by running the same command with the ``--upgrade`` option:

    .. code-block:: console

        $ pip install --upgrade cdl

All-in-one installer
^^^^^^^^^^^^^^^^^^^^

.. note::

    Supported platforms: Windows

DataLab is available as a stand-alone application for Windows,
which does not require any Python distribution to be installed.
Just run the installer and you're good to go!

.. image:: /images/shots/windows_installer.png

The installer package is available in the `Releases`_ section.
It supports automatic uninstall and upgrade feature (no need to uninstall
DataLab before runinng the installer of another version of the
application).

.. _Releases: https://github.com/Codra-Ingenierie-Informatique/DataLab/releases

Wheel package
^^^^^^^^^^^^^

.. note::

    Supported platforms: GNU/Linux, macOS, Windows

On any operating system, using pip and the Wheel package is the easiest way to
install DataLab on an existing Python distribution:

.. code-block:: console

    $ pip install --upgrade DataLab-0.11.1-py2.py3-none-any.whl


Source package
^^^^^^^^^^^^^^

.. note::

    Supported platforms: GNU/Linux, macOS, Windows

Installing DataLab directly from the source package may be done using ``pip``:

.. code-block:: console

    $ pip install --upgrade cdl-0.11.1.tar.gz

Or, if you prefer, you can install it manually by running the following command
from the root directory of the source package:

.. code-block:: console

    $ pip install --upgrade .

Finally, you can also build your own Wheel package and install it using ``pip``,
by running the following command from the root directory of the source package
(this requires the ``build`` and ``wheel`` packages to be installed):

.. code-block:: console

    $ pip install build wheel  # Install build and wheel packages (if needed)
    $ python -m build  # Build the wheel package
    $ pip install --upgrade dist/cdl-0.11.1-py2.py3-none-any.whl  # Install the wheel package

Dependencies
------------

.. note::

    The DataLab all-in-one installer already include all those required
    libraries as well as Python itself.

.. include:: ../requirements.rst

.. note::

    Python 3.11 and PyQt5 are the reference for production release
