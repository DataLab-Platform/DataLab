.. _ref-to-instviewer:

Installation and Configuration
==============================

The **Installation and Configuration** dialog box is a diagnostic tool available from DataLab’s graphical interface. It provides a structured overview of the current installation, user-specific settings, and available plugins and I/O features.

This tool is primarily designed for troubleshooting and verification purposes — whether you are a user trying to understand how DataLab is set up on your system, or a developer validating your plugin integration.

To open the dialog box, go to the "?" menu in the main DataLab window and select **Installation and configuration**. This will open a dialog window that displays all relevant information about your DataLab installation.

The dialog window is divided into **three tabs**, each covering a specific aspect of the environment.

1. Installation Configuration
-----------------------------

.. figure:: /images/shots/instviewer.png

    Installation and configuration (see "?" menu), tab 1

This tab displays system-level information about the DataLab installation, including:

- Application version;
- Platform and operating system;
- Whether DataLab is running in frozen (standalone) mode or as a standard Python package;
- Internal paths used for data storage and plugin discovery;
- Python interpreter details (version, architecture, paths);
- Location of the `datalab` package.

This section is especially useful for debugging path-related issues or confirming the installation context (e.g., standalone vs. pip-installed).

2. User Configuration
---------------------

.. figure:: /images/shots/instviewer2.png

    Installation and configuration (see "?" menu), tab 2

This tab lists user-specific configuration details, through the contents of DataLab's configuration file.

It helps users verify their personalized settings and ensures that DataLab is reading the correct directories for user-defined plugins or preferences.

3. Plugins and I/O Features
---------------------------

.. figure:: /images/shots/instviewer3.png

    Installation and configuration (see "?" menu), tab 3

This tab lists all plugins and I/O handlers currently available in the application.

This view is ideal for verifying that all expected plugins and I/O features are detected, loaded, and functional.

Usage Tips
----------

- You can copy text from any of the tabs for diagnostic or support purposes.
- If a plugin or I/O feature does not appear, check that its filename starts with `datalab_` and that it is located in a recognized plugin directory (see the *User Configuration* tab).

.. note::

  Reporting unexpected behavior or any other bug on `GitHub Issues`_ will be greatly appreciated, especially if the contents of this viewer are attached to the report (as well as log files, see :ref:`ref-to-logviewer`).

.. _GitHub Issues: https://github.com/DataLab-Platform/DataLab/issues/new/choose
