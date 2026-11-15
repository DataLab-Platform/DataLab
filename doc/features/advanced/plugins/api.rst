.. _plugins_api:

Public API
==========

.. meta::
    :description: Public API of the DataLab plugin system
    :keywords: DataLab, plugin, API, PluginBase, PluginInfo, recipe, tool, instrument

This page is the reference of the classes and functions available to plugins. See :ref:`plugins_developing` for their use, and :ref:`plugins_applications` for the declarations of application plugins.

.. automodule:: datalab.plugins
    :members: PluginInfo, PluginCapability, PluginBase, FormatInfo, ImageFormatBase, ClassicsImageFormat, SignalFormatBase

``datalab.plugins.PLUGIN_ENTRY_POINT_GROUP`` is the entry-point group of DataLab Desktop plugin classes (``"datalab.plugins"``); DataLab-Web plugin classes use the ``datalab.web_plugins`` group (see :ref:`plugins_web`). The other names exported by :mod:`datalab.plugins`, such as ``PluginRegistry`` or ``discover_plugins``, are used by DataLab to load and manage plugins, and are not described here.

.. automodule:: datalab.plugins.recipes
  :members:

.. automodule:: datalab.plugins.recipe_binding
  :members:

.. automodule:: datalab.plugins.examples
  :members:

.. automodule:: datalab.plugins.tiles
  :members:

.. automodule:: datalab.plugins.tools
  :members:

.. automodule:: datalab.plugins.instruments
  :members:

.. automodule:: datalab.plugins.resources
  :members:

.. automodule:: datalab.gui.plugins.recipe_runner
  :members:
