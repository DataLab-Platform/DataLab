.. _dependencies:

About Dependencies
==================

Dependencies are an important part of any software project, and they can significantly affect the development process. In this section, we will discuss the dependencies used in our project, how to manage them, and best practices for keeping them up to date.

Managing Dependencies
---------------------

Dependencies in DataLab Project are defined in the `pyproject.toml` file, as officially recommended by the Python packaging authority (see `Writing your pyproject.toml <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>`_).

The `pyproject.toml` file contains many sections, but the most relevant for dependencies are:

- `[project.dependencies]`: This section lists the direct dependencies of the application. These are the packages that the application needs to run.
- `[project.optional-dependencies]`: This section lists optional dependencies that can be installed to enable additional features. These dependencies are not required for the core functionality of the application but can enhance its capabilities.

Among the optional dependencies, we have:

- `qt`: This group contains dependencies needed for the Qt-based graphical user interface (currently, PyQt5).
- `opencv`: This group contains dependencies needed for computer vision tasks requiring OpenCV.
- `dev`: This group contains dependencies needed for development and testing (linters, formatters, etc.).
- `doc`: This group contains dependencies needed for building the documentation (Sphinx, etc.).
- `test`: This group contains dependencies needed for running the tests (pytest, etc.).

Deploying Dependencies
----------------------

Production
^^^^^^^^^^

In production, we recommend using the package manager that is most suitable for your environment and that you are most comfortable with. All package managers (e.g., `pip`, `conda`) are directly or indirectly using the information from the `pyproject.toml` file to install the dependencies.

See :ref:`installation` for more information on how to install DataLab and its dependencies.

Development
^^^^^^^^^^^

In development, you may also use the `requirements.txt` file to make it easier to install the dependencies in a virtual environment or container.

The `requirements.txt` file is generated from the `pyproject.toml` file using `toml-to-requirements` tool. This file is useful for quickly installing the dependencies in a virtual environment or container.

Adding New Dependencies
-----------------------

When adding new dependencies to the project, please follow these rules:

1. If it is a direct dependency, that is a package that the application needs to run, add it to the `[project.dependencies]` section in the `pyproject.toml` file, and specify the version range that is compatible with the application, in order to avoid breaking changes.

2. If it is an optional dependency, that is a package that can be installed to enable additional features, add it to the appropriate section in the `[project.optional-dependencies]` section in the `pyproject.toml` file.

  - For the `dev` dependencies, unless it's absolutely necessary, do not specify a version range, as it may limit the ability to use the latest version of the package.

  - For the other optional dependencies, specify the version range that is compatible with the application, in order to avoid breaking changes.

.. note::

    In other words, except for the `dev` dependencies, always specify a version range that is compatible with the application.