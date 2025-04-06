.. _about_plugins:

Plugins
=======

.. meta::
    :description: Plugin system for DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, plugin, processing, input/output, HDF5, file format, data analysis, visualization, scientific, open-source, platform

DataLab is a modular application. It is possible to add new features to DataLab
by writing plugins. A plugin is a Python module that is loaded at startup by
DataLab. A plugin may add new features to DataLab, or modify existing features.

The plugin system currently supports the following features:

- Processing features: add new processing tasks to the DataLab processing
  system, including specific graphical user interfaces.
- Input/output features: add new file formats to the DataLab file I/O system.
- HDF5 features: add new HDF5 file formats to the DataLab HDF5 I/O system.

What is a plugin?
^^^^^^^^^^^^^^^^^

A plugin is a Python module that is loaded at startup by DataLab. A plugin may
add new features to DataLab, or modify existing features.

A plugin is a Python module which file name starts with ``cdl_``, and which
contains a class derived from the :class:`cdl.plugins.PluginBase` class.
The name of the class is not important, as long as it is derived from
:class:`cdl.plugins.PluginBase` and has a ``PLUGIN_INFO`` attribute that is an
instance of the :class:`cdl.plugins.PluginInfo` class. The ``PLUGIN_INFO`` attribute
is used by DataLab to retrieve information about the plugin.

.. note::

    DataLab's plugin discovery mechanism will only load plugins that are defined
    in Python files which names start with ``cdl_`` (e.g. ``cdl_myplugin.py``).

Where to put a plugin?
^^^^^^^^^^^^^^^^^^^^^^

As plugins are Python modules, they can be put anywhere in the Python path of
the DataLab installation.

Special additional locations are available for plugins:

- The `plugins` directory in the user configuration folder
  (e.g. `C:/Users/JohnDoe/.DataLab/plugins` on Windows
  or `~/.DataLab/plugins` on Linux).

- The `plugins` directory in the same folder as the `DataLab` executable
  in case of a standalone installation.

- The `plugins` directory in the `cdl` package in case for internal plugins
  only (i.e. it is not recommended to put your own plugins there).

How to develop a plugin?
^^^^^^^^^^^^^^^^^^^^^^^^

To develop a plugin, you may start by deriving from one of the example plugins
(see below) and modify it to suit your needs.

If you want to code a plugin in your usual Python development environment (preferably
with an IDE like `Spyder <https://www.spyder-ide.org/>`_) and take advantage of the
code completion, you can add the `cdl` package to your Python path.

This can be done:

- By installing DataLab in your Python environment (using one of the following methods:
  :ref:`install_conda`, :ref:`install_pip`, :ref:`install_wheel`, or :ref:`install_source`),

- Or by adding the `cdl` package to your Python path manually:

  - Download the DataLab source code from the `PyPI page <https://pypi.org/project/cdl/>`_,
  - Unzip the source code to a folder on your computer,
  - Add the `cdl` package to your Python path (e.g. by using the *PYTHONPATH Manager* in Spyder).

.. note::

    Even if you have installed the `cdl` package properly in your Python environment,
    you won't be able to run the DataLab application from your development environment
    to test your plugin. You will need to run DataLab from the command line or from
    the shortcut created during the installation.

Example: processing plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a simple example of a plugin that adds a new features to DataLab.

.. literalinclude:: ../../../cdl/plugins/cdl_testdata.py

Example: input/output plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a simple example of a plugin that adds a new file formats to DataLab.

.. literalinclude:: ../../../cdl/plugins/cdl_imageformats.py

Other examples
^^^^^^^^^^^^^^

Other examples of plugins can be found in the `plugins/examples` directory of
the DataLab source code (explore `here on GitHub <https://github.com/DataLab-Platform/DataLab/tree/main/plugins/examples>`_).

Public API
^^^^^^^^^^

.. automodule:: cdl.plugins
    :members: PluginInfo, PluginBase, FormatInfo, ImageFormatBase, ClassicsImageFormat, SignalFormatBase
