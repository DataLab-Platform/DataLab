.. _installation_offline:

Offline Installation
====================

.. meta::
    :description: Offline installation guide for DataLab on systems without internet access
    :keywords: DataLab, offline, installation, Python, pip, wheel, no internet, Windows, Linux, macOS

This guide explains how to install DataLab on a system **without internet access**.

.. note::

    This page focuses on **offline installation using pip** (Python package manager).
    For other offline installation methods, see the :ref:`installation_offline_alternatives`
    section below.

Offline installation via pip
----------------------------

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

This method works on **any operating system** with **Python 3.9 to 3.14**.
It assumes you have an online machine available to download packages, and a separate
offline machine where DataLab will be installed.

Requirements
------------

Offline machine
^^^^^^^^^^^^^^^

- GNU/Linux, Windows, or macOS
- Python 3.9 to 3.14 installed with the **same major version** as the online machine
- No assumptions regarding installed packages

.. important::

    **Version compatibility is critical:**

    - ✅ **Allowed:** Python 3.10.2 ↔ Python 3.10.11 (same major version)
    - ❌ **Not allowed:** Python 3.10.x ↔ Python 3.11.x (different major versions)

    SciPy, NumPy, and scikit-image are highly dependent on the Python major version.
    Both machines must use the same major Python version for reliable offline installation.

To check your Python version, run:

.. code-block:: console

    python --version

Online machine
^^^^^^^^^^^^^^

- Windows, Linux, or macOS
- Python with the **same major version** as the offline machine
- ``pip`` installed

Preparing packages (online machine)
-----------------------------------

Download all required **Wheel packages** for offline installation.

Create a folder for packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Windows

        .. code-block:: console

            mkdir C:\datalab-offline
            cd C:\datalab-offline

    .. tab-item:: GNU/Linux / macOS

        .. code-block:: console

            mkdir ~/datalab-offline
            cd ~/datalab-offline

Download DataLab and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Windows

        .. code-block:: console

            pip download datalab-platform --dest C:\datalab-offline

    .. tab-item:: GNU/Linux / macOS

        .. code-block:: console

            pip download datalab-platform --dest ~/datalab-offline

This command downloads:

- ``datalab-platform``
- ``sigima``
- ``plotpy``
- ``guidata``
- ``pythonqwt``
- ``numpy``, ``scipy``, ``scikit-image``, and other dependencies

Expected folder structure
^^^^^^^^^^^^^^^^^^^^^^^^^

After downloading, your folder should contain files similar to:

.. code-block:: text

    datalab-offline/
    ├─ datalab_platform‑1.x.x‑py3‑none‑any.whl
    ├─ sigima‑1.x.x‑py3‑none‑any.whl
    ├─ plotpy‑x.x.x‑py3‑none‑any.whl
    ├─ guidata‑3.x.x‑py3‑none‑any.whl
    ├─ pythonqwt‑x.x.x‑py3‑none‑any.whl
    ├─ numpy‑...‑cp310‑cp310‑[platform].whl
    ├─ scipy‑...‑cp310‑cp310‑[platform].whl
    ├─ scikit_image‑...‑cp310‑cp310‑[platform].whl
    └─ ...

.. note::

    The ``[platform]`` suffix in the wheel filenames will vary depending on your
    system: ``win_amd64`` (Windows), ``manylinux`` (Linux), ``macosx`` (macOS).

Transfer to offline machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transfer the ``datalab-offline`` folder to your offline machine using:

- USB drive
- Network share
- Any other approved method in your organization

Offline installation
--------------------

On the offline machine, navigate to the folder and install DataLab.

Navigate to the folder
^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Windows

        .. code-block:: console

            cd C:\datalab-offline

    .. tab-item:: GNU/Linux / macOS

        .. code-block:: console

            cd ~/datalab-offline

Install DataLab offline
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    python -m pip install --no-index --find-links . datalab-platform

The ``--no-index`` option tells pip not to connect to the internet, and
``--find-links .`` tells pip to look for packages in the current directory.

Verification
------------

To verify that DataLab is installed correctly, run:

.. code-block:: console

    python -m datalab.app

This should launch the DataLab application.

Optional: Virtual environment
-----------------------------

It is recommended to install DataLab in a virtual environment to avoid conflicts
with other Python packages.

Create and activate a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Windows

        .. code-block:: console

            python -m venv C:\venvs\datalab
            C:\venvs\datalab\Scripts\activate

    .. tab-item:: GNU/Linux / macOS

        .. code-block:: console

            python -m venv ~/venvs/datalab
            source ~/venvs/datalab/bin/activate

Install DataLab in the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Windows

        .. code-block:: console

            python -m pip install --no-index --find-links C:\datalab-offline datalab-platform

    .. tab-item:: GNU/Linux / macOS

        .. code-block:: console

            python -m pip install --no-index --find-links ~/datalab-offline datalab-platform

Launch DataLab
^^^^^^^^^^^^^^

.. code-block:: console

    python -m datalab.app

WinPython for offline usage
---------------------------

:octicon:`info;1em;sd-text-info` :bdg-info-line:`Windows`

`WinPython <https://winpython.github.io/>`_ is particularly well-suited for offline
installations as it provides a portable, self-contained Python environment.

Prepare on online machine
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download WinPython with the **same major Python version** as your target system
2. Install WinPython
3. Open the *WinPython Command Prompt*
4. Download packages:

.. code-block:: console

    pip download datalab-platform --dest C:\datalab-offline

Install on offline machine
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Transfer WinPython and the ``datalab-offline`` folder to the offline machine
2. Open the *WinPython Command Prompt*
3. Install DataLab:

.. code-block:: console

    cd C:\datalab-offline
    python -m pip install --no-index --find-links . datalab-platform

Benefits of WinPython
^^^^^^^^^^^^^^^^^^^^^

- **Portable:** No installation required, can run from USB drive
- **Self-contained:** All dependencies included
- **Ideal for restricted environments:** No admin rights needed
- **Multiple versions:** Can have multiple Python versions side by side

Quick summary
-------------

The complete workflow in four steps:

1. **On online machine:** Download packages

   .. code-block:: console

       pip download datalab-platform --dest /path/to/datalab-offline

2. **Transfer:** Copy the ``datalab-offline`` folder to the offline machine

3. **On offline machine:** Install offline

   .. code-block:: console

       python -m pip install --no-index --find-links /path/to/datalab-offline datalab-platform

4. **Verify:** Launch DataLab

   .. code-block:: console

       python -m datalab.app

.. _installation_offline_alternatives:

Other offline installation methods
----------------------------------

While this guide focuses on pip-based offline installation, DataLab can also be
installed offline using other methods:

Stand-alone installer (Windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`Windows` **Recommended for offline installation**

The **all-in-one installer** is the simplest way to install DataLab offline on Windows.
It does not require any Python distribution or internet connection on the target machine.

**Advantages:**

- No Python installation required
- Single executable file
- Self-contained with all dependencies
- No internet connection needed during installation
- Works on air-gapped systems

**How to use:**

1. On a machine with internet access, download the installer from the
   `DataLab Releases <https://github.com/DataLab-Platform/DataLab/releases>`_ page
2. Transfer the installer executable to the offline machine (USB drive, network share, etc.)
3. Run the installer on the offline machine

For more information, see :ref:`install_aioinstaller`.

WinPython distribution (Windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`Windows`

`DataLab-WinPython <https://github.com/DataLab-Platform/DataLab-WinPython>`_ is a
portable Python distribution that includes DataLab and all its dependencies.

**Advantages:**

- Portable (can run from USB drive)
- No installation required
- Includes full Python environment
- Can be extended with additional packages
- Ideal for restricted environments

**How to use:**

1. Download DataLab-WinPython from the
   `releases page <https://github.com/DataLab-Platform/DataLab-WinPython/releases>`_
2. Transfer to the offline machine
3. Extract and run

For more information, see :ref:`install_winpython`.

Conda offline installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

Conda packages can also be installed offline using a similar approach to pip:

1. **On online machine:** Download the conda package and dependencies:

   .. code-block:: console

       conda create -n datalab-offline --download-only --json conda-forge::datalab

   Or use ``conda list --explicit`` to create a specification file.

2. **Transfer** the downloaded packages to the offline machine

3. **On offline machine:** Install from the local packages:

   .. code-block:: console

       conda create -n datalab --offline -c file:///path/to/packages datalab

For more information about offline conda installation, see the
`Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages-offline>`_.

For standard online installation, see :ref:`install_conda`.

.. seealso::

    For standard online installation methods, see :ref:`installation`.
