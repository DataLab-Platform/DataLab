.. _plugins_using:

Installing and managing plugins
===============================

.. meta::
    :description: Install DataLab plugins from the plugin catalog or from a file, and manage them in the plugin configuration dialog
    :keywords: DataLab, plugin, install, plugin catalog, plugin configuration, wheel, plugin directory

This page describes where DataLab finds plugins, and how to install, enable, update and uninstall them. Plugins are offered by the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_; to distribute plugins within an organization, see :ref:`plugins_deploying`.

.. _plugin_locations:

Where to put a plugin?
----------------------

Plugins are automatically discovered at startup from multiple locations:

- An installed Python package may declare its plugin class through the
  standard ``datalab.plugins`` entry-point group:

  .. code-block:: toml

    [project.entry-points."datalab.plugins"]
    my-plugin = "my_plugin.plugin:MyPlugin"

  The target must be a :class:`datalab.plugins.PluginBase` subclass. This is
  the recommended distribution mechanism for plugins installed with ``pip``.

- The user plugin directory:
  Typically `~/.DataLab_v1/plugins` on Linux/macOS or
  `C:/Users/YourName/.DataLab_v1/plugins` on Windows.

- Additional plugin directories:
  Any number of extra directories may be declared from
  **Plugins > Configure plugins...**, in the **Advanced** tab (see
  :ref:`plugin_search_paths`). They are saved in the DataLab configuration and
  scanned at every startup.

- The standalone distribution directory:
  If using a frozen (standalone) build, the `plugins` folder located next to the executable is scanned.

- Installed plugins:
  Wheels and modules installed from **Plugins > Configure plugins...**, either from a file in the **From file** tab (see :ref:`plugin_install_from_file`) or from a plugin catalog in the **Catalog** tab (see :ref:`plugin_catalog`).

- The internal `datalab/plugins/builtin` folder (not recommended for user plugins):
  This location is reserved for built-in or bundled plugins and should not be modified manually.

- Additional directories listed in the ``DATALAB_PLUGINS`` environment variable:
  One or more directories may be specified, separated by the OS path separator
  (``;`` on Windows, ``:`` on Linux/macOS), following the same convention as
  ``PYTHONPATH``. All listed directories are appended to the plugin search path
  at startup; non-existent paths are silently skipped (a warning is written to
  the log file). Examples:

  .. code-block:: bash

     # Linux/macOS
     export DATALAB_PLUGINS="/opt/my_plugins:/home/alice/datalab_plugins"

  .. code-block:: bat

     :: Windows
     set DATALAB_PLUGINS=C:\my_plugins;D:\shared\datalab_plugins

  Changes to this variable are only taken into account at DataLab startup.

A plugin directory may also be the folder of a plugin project, such as one created by ``datalab-plugin create --project``: DataLab then also searches its ``src`` subfolder, which holds the ``datalab_*`` folder of the plugin. The Python packages the plugin depends on must be available in DataLab's environment.

Managing plugins in DataLab
---------------------------

The **Plugins** menu provides two dedicated actions:

- **Configure plugins...**
  Opens the plugin configuration dialog, organized in four tabs: **Installed plugins**, **Catalog**, **From file** and **Advanced**. Hovering over a tab title describes its content. The dialog reopens with the size it had when last closed.

- **Reload plugins**
  Reloads plugin modules from disk without restarting DataLab.

When plugins are globally enabled, a single left click on the **Plugins:** label in the status bar also opens the same configuration dialog. The icon is not clickable; the shortcut remains available when no plugins are loaded or some plugins failed to load.

Installed plugins tab
~~~~~~~~~~~~~~~~~~~~~

This tab lists all discovered plugins with their icon, name, version, description and
file path, as well as the plugins that failed to import. For each entry:

- an individual checkbox enables or disables the plugin,
- **Open file** opens the plugin source in your editor,
- **Show in folder** reveals the plugin file in the system file manager.

An **Enable all plugins** checkbox and a **Filter** combo box (*All plugins*,
*Enabled plugins*, *Disabled plugins*, *Plugins with errors*) help navigating a
large plugin collection. A **Last loaded: ...** indicator shows when plugins
were last loaded during the current session, and the **Apply and reload
plugins** button applies the changes without leaving the dialog.

.. _plugin_catalog:

Catalog tab
~~~~~~~~~~~

This tab lists the plugins of the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ and of the local plugin catalogs (see :ref:`plugin_local_catalogs`), with their description, license and source code link. Plugins maintained by the DataLab team are marked **Official**; the others are **Community** plugins, maintained by their authors, or **Local catalog** plugins, provided by your organization (the tooltip of such a plugin shows the folder of its catalog). A search field filters the list.

**Install** copies or downloads the wheel or single-file module available for Desktop and the running Python version, checks its size and SHA-256 digest, then installs it as described in :ref:`plugin_install_from_file`, after confirmation. Versioned releases may offer **Update to...**. For a module without a comparable version, changed content offers **Replace...** instead; an identical digest is already installed. Plugins installed with ``pip`` or supplied by unmanaged source directories are identified but are not modified.

The catalogs are read only when the tab is shown or refreshed, and again when the local catalog folders change. A catalog that cannot be read, for example a network share that is not connected, is reported below the plugin count without hiding the plugins of the other catalogs. The address of the DataLab plugin catalog is the ``plugins_catalog_url`` option of the DataLab configuration file: it may point to a private copy of the catalog, or be emptied to disable it, for example on computers without Internet access. For a catalog stored in a folder, including a folder shared on the network, use a local plugin catalog instead.

.. _plugin_install_from_file:

From file tab
~~~~~~~~~~~~~

This tab installs a plugin shared as a file, without any Python tool: this is the way to add plugins to the standalone version of DataLab. **Install from file...** accepts two kinds of files:

- a wheel (``.whl``) built from a plugin project, for example with ``python -m build``: it must be pure Python (``py3-none-any``), declare its plugin class in the ``datalab.plugins`` entry-point group, and have its requirements provided by the host or the adjacent dependency wheelhouse (see :ref:`plugin_dependencies`);
- a single Python module named ``datalab_<name>.py``.

Before installing, DataLab checks the file without running it. For a wheel, the confirmation shows the distribution, version, entry points and requirements from its metadata. For a single-file module, it shows its import name, file, size, SHA-256 and any statically readable or explicit catalog metadata. An unknown version stays unknown and is displayed as a revision; no class or dependency list is guessed. A plugin runs with the same rights as DataLab: install only plugins from authors you trust.

Installed plugins are kept in the ``installed_plugins`` folder of the DataLab configuration directory (e.g. ``~/.DataLab_v1/installed_plugins``) and listed in the tab, where **Uninstall** removes them. Modules can be reloaded after installation or replacement; a newly installed plugin is enabled. Additional dependencies and new versions of already loaded wheels are only activated at the next start of DataLab.

When reloading plugins, DataLab performs the following steps:

1. Unregister currently active plugins and their owned computations,
2. Clear plugin actions from signal and image panels,
3. Re-discover and reload plugin modules,
4. Re-register enabled plugins,
5. Re-register owned computations,
6. Recreate plugin actions and refresh menus.

This workflow allows iterative plugin development while DataLab is running.

.. note::

  Plugin enable/disable state is persisted in DataLab settings. Disabled plugins remain listed in the configuration dialog and can be re-enabled later. The global third-party plugins setting in Preferences is also applied immediately: disabling it removes plugin actions and disables the status indicator, which displays **Plugins: -**, while enabling it reloads plugins automatically. The **Plugins > Configure plugins...** menu action remains accessible when the status indicator is disabled.

.. _plugin_search_paths:

Advanced tab
~~~~~~~~~~~~

This tab lists every directory scanned at startup, in two groups:

- **Default plugin directories** (read-only): the user plugin directory, the
  bundled `datalab/plugins/builtin` folder, the standalone distribution folder when
  applicable, and the directories declared through the ``DATALAB_PLUGINS``
  environment variable (identified by a *(from DATALAB_PLUGINS)* suffix).

- **Additional plugin directories**: your own directories, added with the
  **Add** button and managed with the *Edit directory* and *Remove directory*
  buttons. Unlike the environment variable, these directories are saved in the
  DataLab configuration and therefore persist across sessions.

It also lists the **Local plugin catalogs**: folders, possibly shared on the network, whose plugins are offered in the **Catalog** tab (see :ref:`plugin_local_catalogs`). They are managed with the same buttons, and the folders declared through the ``DATALAB_PLUGINS_CATALOG`` environment variable come first, as read-only entries. Changing these folders does not require reloading plugins.

The tab also holds a **Compatibility warnings** option to hide warnings for
incompatible DataLab v0.20 plugins.
