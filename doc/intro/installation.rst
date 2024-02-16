.. _installation:

Installation
============

.. meta::
    :description: How to install DataLab, the open-source data analysis and visualization platform
    :keywords: DataLab, installation, install, pip, wheel, source, Windows, Linux, macOS

This section provides information on how to install DataLab on your system.
Once installed, you can start DataLab by running the ``cdl`` command in a terminal,
or by clicking on the DataLab shortcut in the Start menu (on Windows).

.. seealso::

    For more details on how to execute DataLab and its command-line options,
    see :ref:`ref-to-command-line-features`.

How to install
--------------

DataLab is available in several forms:

-   As a Python package, which can be installed using the :ref:`install_pip`.

-   :bdg-info-line:`Windows` As a stand-alone application, which does not require any
    Python distribution to be installed. Just run the :ref:`install_aioinstaller`
    and you're good to go!

-   As a precompiled :ref:`install_wheel`, which can be installed using ``pip``.

-   As a :ref:`install_source`, which can be installed using ``pip`` or manually.

.. seealso::

    Impatient to try the next version of DataLab? You can also install the
    latest development version of DataLab from the master branch of the
    Git repository. See :ref:`install_development` for more information.

.. _install_pip:

Package manager ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

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

.. _install_aioinstaller:

All-in-one installer
^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`Windows`

DataLab is available as a stand-alone application for Windows,
which does not require any Python distribution to be installed.
Just run the installer and you're good to go!

.. figure:: /images/shots/windows_installer.png

    DataLab all-in-one installer for Windows

The installer package is available in the `Releases`_ section.
It supports automatic uninstall and upgrade feature (no need to uninstall
DataLab before runinng the installer of another version of the
application).

.. warning::

    DataLab Windows installer is available for Windows 8, 10 and 11 (main release,
    based on Python 3.11) and also for Windows 7 SP1 (Python 3.8 based release, see
    file ending with ``-Win7.exe``).

    :octicon:`alert;1em;sd-text-warning` On Windows 7 SP1, before running DataLab
    (or any other Python 3 application), you must install Microsoft Update `KB2533623`
    (`Windows6.1-KB2533623-x64.msu`) and also may need to install
    `Microsoft Visual C++ 2015-2022 Redistribuable package <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_.

.. _Releases: https://github.com/DataLab-Platform/DataLab/releases

.. _install_wheel:

Wheel package
^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

On any operating system, using pip and the Wheel package is the easiest way to
install DataLab on an existing Python distribution:

.. code-block:: console

    $ pip install --upgrade DataLab-0.11.1-py2.py3-none-any.whl

.. _install_source:

Source package
^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

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

.. _install_development:

Development version
^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

If you want to try the latest development version of DataLab, you can install
it directly from the master branch of the Git repository.

The first time you install DataLab from the Git repository, enter the following
command:

.. code-block:: console

    $ pip install git+https://github.com/DataLab-Platform/DataLab.git

Then, if at some point you want to upgrade to the latest version of DataLab,
just run the same command with options to force the reinstall of the package
without handling dependencies (because it would reinstall all dependencies):

.. code-block:: console

    $ pip install --force-reinstall --no-deps git+https://github.com/DataLab-Platform/DataLab.git

.. note::

    If dependencies have changed, you may need to execute the same command as above,
    but without the ``--no-deps`` option.

Dependencies
------------

.. note::

    The DataLab all-in-one installer already include all those required
    libraries as well as Python itself.

.. include:: ../requirements.rst

.. note::

    Python 3.11 and PyQt5 are the reference for production release
