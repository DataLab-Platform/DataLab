.. _about_plugins:

Plugins
=======

.. meta::
    :description: Plugin system for DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, plugin, processing, input/output, HDF5, file format, data analysis, visualization, scientific, open-source, platform

DataLab supports a robust plugin architecture, allowing users to extend the application’s features without modifying its core. Plugins can introduce new processing tools, data import/export formats, or custom GUI elements — all seamlessly integrated into the platform.

What is a plugin?
-----------------

A plugin is a Python module that is automatically loaded by DataLab at startup. It can define new features or modify existing ones.

To be recognized as a plugin, the file must:

- Be a Python module whose name **starts with** ``cdl_`` (e.g. ``cdl_myplugin.py``),
- Contain a class that **inherits from** :class:`cdl.plugins.PluginBase`,
- Include a class attribute named ``PLUGIN_INFO``, which must be an instance of :class:`cdl.plugins.PluginInfo`.

This `PLUGIN_INFO` object is used by DataLab to retrieve metadata such as the plugin name, type, and menu integration.

.. note::

   Only Python files whose names start with ``cdl_`` will be scanned for plugins.

DataLab supports three categories of plugins, each with its own purpose and registration mechanism:

- **Processing and visualization plugins**
  Add custom actions for signal or image processing. These may include new computation functions, data visualization tools, or interactive dialogs. Integrated into a dedicated submenu of the “Plugins” menu.

- **Input/Output plugins**
  Define new file formats (read and/or write) handled transparently by DataLab's I/O framework. These plugins extend compatibility with custom or third-party data formats.

- **HDF5 plugins**
  Special plugins that support HDF5 files with domain-specific tree structures. These allow DataLab to interpret signals or images organized in non-standard ways.

Where to put a plugin?
----------------------

Plugins are automatically discovered at startup from multiple locations:

- The user plugin directory:
  Typically `~/.DataLab/plugins` on Linux/macOS or `C:/Users/YourName/.DataLab/plugins` on Windows.

- A custom plugin directory:
  Configurable in DataLab's preferences.

- The standalone distribution directory:
  If using a frozen (standalone) build, the `plugins` folder located next to the executable is scanned.

- The internal `cdl/plugins` folder (not recommended for user plugins):
  This location is reserved for built-in or bundled plugins and should not be modified manually.

How to develop a plugin?
------------------------

The recommended approach to developing a plugin is to derive from an existing example and adapt it to your needs. You can explore the source code in the `cdl/plugins` folder or refer to community-contributed examples.

To develop in your usual Python environment (e.g., with an IDE like `Spyder <https://www.spyder-ide.org/>`_), you can:

1. **Install DataLab in your Python environment**, using one of the following methods:

   - :ref:`install_conda`
   - :ref:`install_pip`
   - :ref:`install_wheel`
   - :ref:`install_source`

2. **Or add the `cdl` package manually to your Python path**:

   - Download the source from the `PyPI page <https://pypi.org/project/cdl/>`_,
   - Unzip the archive,
   - Add the `cdl` directory to your PYTHONPATH (e.g., using the *PYTHONPATH Manager* in Spyder).

.. note::

   Even if you’ve installed `cdl` in your environment, you cannot run the full DataLab application directly from an IDE. You must launch DataLab via the command line or using the installer-created shortcut to properly test your plugin.

Example: processing plugin
--------------------------

Here is a minimal example of a plugin that prints a message when activated:

.. literalinclude:: ../../../cdl/plugins/cdl_testdata.py

Example: input/output plugin
----------------------------

Here is a simple example of a plugin that adds a new file formats to DataLab.

.. literalinclude:: ../../../cdl/plugins/cdl_imageformats.py

Other examples
--------------

Other examples of plugins can be found in the `plugins/examples` directory of the DataLab source code (explore `here on GitHub <https://github.com/DataLab-Platform/DataLab/tree/main/plugins/examples>`_).

Public API
----------

.. automodule:: cdl.plugins
    :members: PluginInfo, PluginBase, FormatInfo, ImageFormatBase, ClassicsImageFormat, SignalFormatBase
