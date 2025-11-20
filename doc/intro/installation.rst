.. _installation:

Installation
============

.. meta::
    :description: How to install DataLab, the open-source data analysis and visualization platform
    :keywords: DataLab, installation, install, pip, wheel, source, Windows, Linux, macOS

.. only:: html and not latex

    .. dropdown:: Quick install on Windows
        :animate: fade-in
        :icon: zap

        Direct download link for the latest version of DataLab:

        - |download_link1|

.. warning::

    **Important notice for users upgrading from DataLab v0.20 or earlier:**

    DataLab v1.0 introduces **breaking changes** that are **not backward compatible** with v0.20.

    - **Plugins** developed for v0.20 **must be updated** to work with v1.0
    - **API changes** affect custom code integrations

    For detailed migration information, see the :ref:`migration guide <migration_v020_to_v100>`.

This section provides information on how to install DataLab on your system.
Once installed, you can start DataLab by running the ``datalab`` command in a terminal,
or by clicking on the DataLab shortcut in the Start menu (on Windows).

.. seealso::

    For more details on how to execute DataLab and its command-line options,
    see :ref:`ref-to-command-line-features`.

    For installation on systems without internet access, see the
    :ref:`offline installation guide <installation_offline>`.

How to install
--------------

DataLab is available in several forms:

-   As a :ref:`install_conda`.

-   As a Python package, which can be installed using the :ref:`install_pip`.

-   :bdg-info-line:`Windows` As a stand-alone application, which does not require any
    Python distribution to be installed. Just run the :ref:`install_aioinstaller`
    and you're good to go!

-   :bdg-info-line:`Windows` Within a ready-to-use :ref:`install_winpython`, based on
    `WinPython <https://winpython.github.io/>`_.

-   As a precompiled :ref:`install_wheel`, which can be installed using ``pip``.

-   As a :ref:`install_source`, which can be installed using ``pip`` or manually.

.. seealso::

    Impatient to try the next version of DataLab? You can also install the
    latest development version of DataLab from the master branch of the
    Git repository. See :ref:`install_development` for more information.

.. _install_conda:

Conda package
^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

To install ``datalab`` package from the `conda-forge` channel (https://anaconda.org/conda-forge/datalab), run the following command:

.. code-block:: console

    $ conda install conda-forge::datalab

.. _install_pip:

Package manager ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

DataLab's package ``datalab-platform`` is available on the Python Package Index (PyPI)
on the following URL: https://pypi.python.org/pypi/datalab-platform.

Installing DataLab from PyPI with Qt is as simple as running this command
(you may need to use ``pip3`` instead of ``pip`` on some systems):

.. code-block:: console

    $ pip install datalab-platform[qt]

Or, if you prefer, you can install DataLab without the Qt library (not recommended):

.. code-block:: console

    $ pip install datalab-platform

.. note::

    If you already have a previous version of DataLab installed, you can
    upgrade it by running the same command with the ``--upgrade`` option:

    .. code-block:: console

        $ pip install --upgrade datalab-platform[qt]

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

    DataLab Windows installer is available for Windows 7 SP1, 8, 10 and 11.

    :octicon:`alert;1em;sd-text-warning` On Windows 7 SP1, before running DataLab
    (or any other Python 3 application), you must install Microsoft Update `KB2533623`
    (`Windows6.1-KB2533623-x64.msu`) and also may need to install
    `Microsoft Visual C++ 2015-2022 Redistribuable package <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_.

.. _Releases: https://github.com/DataLab-Platform/DataLab/releases

.. _install_winpython:

Python distribution
^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`Windows`

DataLab is also available within a ready-to-use Python distribution, based on
`WinPython <https://winpython.github.io/>`_. This distribution is called
`DataLab-WinPython <https://github.com/DataLab-Platform/DataLab-WinPython?tab=readme-ov-file#datalab-winpython>`_
and is available in the `DataLab-WinPython Releases <https://github.com/DataLab-Platform/DataLab-WinPython/releases>`_
section.

.. figure:: /images/logos/DataLab-WinPython.png

    DataLab-WinPython is a ready-to-use Python distribution including the DataLab platform.

The main difference with the all-in-one installer is that you can use the Python
distribution for other purposes than running DataLab, and you may also extend it
with additional packages. On the downside, it is also *much bigger* than the
all-in-one installer because it includes a full Python distribution.

.. note::

    DataLab-WinPython includes `Spyder <https://www.spyder-ide.org/>`_,
    a powerful IDE for scientific programming in Python,
    as well as `Jupyter Notebook <https://jupyter.org/>`_ for interactive computing.

.. figure:: /images/shots/wpcp.png

    DataLab-WinPython Control Panel

.. warning::

    Whereas the all-in-one installer provides a monolithic package that guarantees
    the compatibility of all its components because it cannot be modified by the user,
    the WinPython distribution is more flexible and thus can be broken by a bad
    manipulation of the Python distribution by the user. This should be taken into
    account when choosing the installation method.

.. _install_wheel:

Wheel package
^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

On any operating system, using pip and the Wheel package is the easiest way to
install DataLab on an existing Python distribution:

.. code-block:: console

    $ pip install --upgrade DataLab-1.0.1-py2.py3-none-any.whl

.. _install_source:

Source package
^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

Installing DataLab directly from the source package may be done using ``pip``:

.. code-block:: console

    $ pip install --upgrade datalab-1.0.1.tar.gz

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
    $ pip install --upgrade dist/datalab-1.0.1-py2.py3-none-any.whl  # Install the wheel package

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
