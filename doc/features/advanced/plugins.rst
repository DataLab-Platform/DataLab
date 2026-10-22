.. _about_plugins:

Plugins
=======

.. meta::
    :description: Plugin system for DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, plugin, processing, input/output, HDF5, file format, data analysis, visualization, scientific, open-source, platform

DataLab supports a robust plugin architecture, allowing users to extend the application’s features without modifying its core. Plugins can introduce new processing tools, data import/export formats, or custom GUI elements — all seamlessly integrated into the platform.

What is a plugin?
-----------------

A plugin is Python code that is automatically loaded by DataLab at startup. It can define new features or modify existing ones.

A plugin defines a class that **inherits from** :class:`datalab.plugins.PluginBase`, includes a class attribute named ``PLUGIN_INFO`` (an instance of :class:`datalab.plugins.PluginInfo`) and implements the ``create_actions`` method. DataLab finds this class in a plugin directory (see :ref:`plugin_locations`), in one of two forms:

- **A single file** whose name **starts with** ``datalab_`` (e.g. ``datalab_myplugin.py``) and which contains the class. This is the simplest form, recommended to get started: ``datalab-plugin create`` creates such a file, ready to use (see :ref:`plugin_generator`).
- **A folder** whose name starts with ``datalab_`` and which holds an ``__init__.py`` file (a Python package, e.g. ``datalab_myplugin/``). DataLab loads the folder, then its ``adapters/desktop.py`` file when it exists: this is where projects created by ``datalab-plugin create --project`` define the class (see :ref:`plugin_project`). Otherwise, the class is defined in ``__init__.py``.

Nothing needs to be installed: the file or folder is loaded as it is.

Plugins distributed as installed Python packages, such as the wheels of the plugin catalog, declare their ``PluginBase`` subclass through the ``datalab.plugins`` entry-point group, without relying on the ``datalab_`` prefix. A plugin found both ways, for example a project that is installed and whose folder is also a plugin directory, is loaded only once.

The ``PLUGIN_INFO`` object must define a unique, namespaced ``id`` that remains
stable when the display name or implementation class changes. DataLab uses this
ID for registration and persisted enablement settings. Plugins without an ID
remain supported through a module-and-class fallback for backward compatibility,
but new plugins should not rely on that fallback.

.. code-block:: python

  from datalab.plugins import PluginCapability, PluginInfo

  PLUGIN_INFO = PluginInfo(
    id="org.example.my-plugin",
    name="My Plugin",
    version="1.0.0",
    capabilities=(
      PluginCapability.APPLICATION,
      PluginCapability.PROCESSING,
    ),
  )

``capabilities`` declares how DataLab may present and consume a plugin. Supported
values are ``PROCESSING``, ``IO``, ``VISUALIZATION`` and ``APPLICATION``. A
plugin may combine several values; domain applications typically declare
``APPLICATION`` together with ``PROCESSING``. The plugin configuration dialog
shows declared capabilities in a stable order. Existing plugins that omit the
field remain valid and are shown without capability labels.

.. note::

   Only files and folders whose names start with ``datalab_`` are scanned for plugins.

DataLab supports three categories of plugins, each with its own purpose and registration mechanism:

- **Processing and visualization plugins**
  Add custom actions for signal or image processing. These may include new computation functions, data visualization tools, or interactive dialogs. Integrated into a dedicated submenu of the “Plugins” menu.

- **Input/Output plugins**
  Define new file formats (read and/or write) handled transparently by DataLab's I/O framework. These plugins extend compatibility with custom or third-party data formats.

- **HDF5 plugins**
  Special plugins that support HDF5 files with domain-specific tree structures. These allow DataLab to interpret signals or images organized in non-standard ways.

.. _plugin_forms:

Source organization and distribution
-----------------------------------

**Source organization**, **distribution format** and **functional capabilities** are independent concepts. A single-file plugin and a plugin organized as a Python package use the same :class:`datalab.plugins.PluginBase` API; neither form determines whether it provides processing, input/output, visualization or application features.

- **Single-file plugin:** one ``datalab_<name>.py`` module, easy to edit and share without a packaging project. This historical form is also the default output of ``datalab-plugin create``; it is not deprecated.
- **Package or project:** Python modules and, optionally, resources, tests and packaging metadata. ``datalab-plugin create --project`` offers one layered organization, not a mandatory architecture for all plugins.
- **Wheel distribution:** an installable ``.whl`` archive with standardized metadata and plugin entry points. A wheel can contain a single module or a larger package; it is a distribution format, not a functional plugin category.

At present, a single-file plugin can be loaded from a plugin directory or installed with **Install from file...**, but online and local catalogs accept only wheels. Publishing an existing single-module implementation as a wheel does not require reorganizing it into the generated layered project, although packaging metadata must be supplied.

A directly loaded source file remains in its development directory. A file installed through DataLab is copied to its managed user storage. A plugin is not a macro: a plugin registers persistent application features, whereas a macro is a script run to automate a task. Supporting the historical single-file form does not mean that obsolete APIs from DataLab v0.20 are automatically supported.

Live preview compatibility
--------------------------

.. note::

  Standard parameterized functions registered with ``register_1_to_1`` can
  participate in live preview without a custom renderer. They must return a
  ``SignalObj`` or ``ImageObj``, be importable in a spawned process, and accept
  serializable source/parameter copies. Preview uses a private process and does
  not publish results to the workspace. It is not a sandbox for filesystem,
  network or other external side effects. Register functions that must not run
  speculatively with ``preview_enabled=False``. Serialization or computation
  failures are reported in the preview; the normal OK path remains available.
  Unknown custom ``DataSet.edit`` implementations and alternate guidata
  backends are not replaced.

  The dedicated preview process may be reused between dialogs. Plugin functions
  must not assume a fresh Python interpreter for each preview: process-local
  module state may persist until a preview is cancelled while running, plugins
  are reloaded, or DataLab exits.

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
  **Plugins > Configure plugins...**, in the **Plugin settings** tab (see
  :ref:`plugin_search_paths`). They are saved in the DataLab configuration and
  scanned at every startup.

- The standalone distribution directory:
  If using a frozen (standalone) build, the `plugins` folder located next to the executable is scanned.

- Installed plugins:
  Wheels and modules installed from **Plugins > Configure plugins...**, either from a file in the **Install plugins** tab (see :ref:`plugin_install_from_file`) or from a plugin catalog in the **Available plugins** tab (see :ref:`plugin_catalog`).

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
  Opens the plugin configuration dialog, organized in four tabs: **Enable/disable plugins**, **Plugin settings**, **Available plugins** and **Install plugins**.

- **Reload plugins**
  Reloads plugin modules from disk without restarting DataLab.

Enable/disable plugins
~~~~~~~~~~~~~~~~~~~~~~

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

.. _plugin_search_paths:

Plugin settings
~~~~~~~~~~~~~~~

This tab lists every directory scanned at startup, in two groups:

- **Default plugin directories** (read-only): the user plugin directory, the
  bundled `datalab/plugins/builtin` folder, the standalone distribution folder when
  applicable, and the directories declared through the ``DATALAB_PLUGINS``
  environment variable (identified by a *(from DATALAB_PLUGINS)* suffix).

- **Additional plugin directories**: your own directories, added with the
  **Add** button and managed with the *Edit directory* and *Remove directory*
  buttons. Unlike the environment variable, these directories are saved in the
  DataLab configuration and therefore persist across sessions.

It also lists the **Local plugin catalogs**: folders, possibly shared on the network, whose plugins are offered in the **Available plugins** tab (see :ref:`plugin_local_catalogs`). They are managed with the same buttons, and the folders declared through the ``DATALAB_PLUGINS_CATALOG`` environment variable come first, as read-only entries. Changing these folders does not require reloading plugins.

The tab also holds a **Compatibility warnings** option to hide warnings for
incompatible DataLab v0.20 plugins.

.. _plugin_catalog:

Available plugins
~~~~~~~~~~~~~~~~~

This tab lists the plugins of the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ and of the local plugin catalogs (see :ref:`plugin_local_catalogs`), with their description, license and source code link. Plugins maintained by the DataLab team are marked **Official**; the others are **Community** plugins, maintained by their authors, or **Local catalog** plugins, provided by your organization (the tooltip of such a plugin shows the folder of its catalog). A search field filters the list.

**Install** downloads the newest release available for DataLab desktop and for the running Python version, checks that its size and SHA-256 digest match the catalog, then installs it as described in :ref:`plugin_install_from_file`, after the same confirmation. When the catalog offers a newer version of a plugin installed this way, the button becomes **Update to...**. Plugins installed with ``pip`` in the Python environment are shown as such and are not modified.

The catalogs are read only when the tab is shown or refreshed, and again when the local catalog folders change. A catalog that cannot be read, for example a network share that is not connected, is reported below the plugin count without hiding the plugins of the other catalogs. The address of the DataLab plugin catalog is the ``plugins_catalog_url`` option of the DataLab configuration file: it may point to a private copy of the catalog, or be emptied to disable it, for example on computers without Internet access. For a catalog stored in a folder, including a folder shared on the network, use a local plugin catalog instead.

.. _plugin_local_catalogs:

Local plugin catalogs
^^^^^^^^^^^^^^^^^^^^^

A local plugin catalog is a folder, typically shared on the network (for example ``\\fileserver\datalab\plugins``), from which an organization distributes its own plugins, or the versions of public plugins it has approved, without publishing them. Local catalogs complement the DataLab plugin catalog and take precedence over it: when several catalogs list the same plugin (same distribution name or plugin ID), only the first one is shown, so that users install the version chosen by the organization. If the DataLab plugin catalog withdraws a plugin hidden this way, the tab reports it.

Local catalogs are declared:

- in the **Plugin settings** tab (see :ref:`plugin_search_paths`): these folders are saved in the DataLab configuration (``plugins_catalog_path_list`` option);
- through the ``DATALAB_PLUGINS_CATALOG`` environment variable, for example to configure all the computers of an organization: folders are separated by the OS path separator (``;`` on Windows, ``:`` on Linux/macOS) and take precedence over the folders of the configuration. Unlike ``DATALAB_PLUGINS``, folders unavailable when DataLab starts are not ignored: they are reported in the **Available plugins** tab, and used as soon as they become available and the catalogs are refreshed.

.. code-block:: bat

   :: Windows
   set DATALAB_PLUGINS_CATALOG=\\fileserver\datalab\plugins;D:\approved_plugins

A local catalog folder holds either:

- **plugin wheels only**: DataLab lists every pure-Python wheel of the folder (subfolders are not scanned) declaring a ``datalab.plugins`` entry point, and groups the versions of each plugin. Other files are ignored and reported. The plugin name is its distribution name; its description, license, keywords and source code link come from the wheel metadata. Copying a new version to the folder is enough to offer it as an update;
- **or a** ``catalog.json`` **file**, in the format of the DataLab plugin catalog, whose release URLs are relative to the folder: only the plugins it lists are shown. It may be a copy of the DataLab plugin catalog with its wheels, or a catalog built from the wheels of the folder with ``datalab-plugin catalog build``, then edited to give a display name to the plugins, deprecate or revoke them (``status`` and ``status_reason`` fields), or withdraw a release (``yanked`` field).

Whatever the catalog, DataLab checks the size and SHA-256 digest of the wheel before installing it, then inspects it without running it and asks for confirmation; the confirmation dialog shows the catalog folder as the source of the plugin. Being listed in a local catalog does not mean that a plugin was audited: the organization managing the folder is responsible for its content, and write access to the folder should be restricted accordingly.

The ``datalab-plugin`` command maintains the ``catalog.json`` file of a folder:

.. code-block:: bat

   :: Write catalog.json, listing the plugin wheels of the folder
   datalab-plugin catalog build \\fileserver\datalab\plugins

   :: Check catalog.json against the plugin wheels of the folder
   datalab-plugin catalog check \\fileserver\datalab\plugins

``build`` checks the plugin wheels (pure Python, plugin entry point, requirements available from the host or the catalog wheelhouse, supported Python version) and does not write the file if a wheel is refused. It keeps the fields edited in an existing ``catalog.json`` file (plugin ID, name, tier, status and its reason, repository, documentation, capabilities and keywords, and the reasons why releases are withdrawn), and revoked plugins remain listed. The file is replaced atomically, so that users may read the catalog meanwhile. ``check`` reports wheels that are missing, changed, refused or not listed, including the dependency artifacts. Run ``build`` again after adding wheels. Both commands exit with a non-zero status when they find a problem.

.. _plugin_dependencies:

Additional dependencies and offline installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Desktop plugins may require packages not shipped with DataLab. Keep the plugin itself as a pure-Python wheel, and supply the additional packages, including all their transitive dependencies, as ordinary wheels in a ``dependencies`` subfolder of the local catalog. These dependency wheels may contain native libraries, provided they match the Python version, ABI and platform of the host. They do not need a DataLab plugin entry point.

On a computer with Internet access, the authoring command prepares the wheelhouse and writes the catalog:

.. code-block:: bat

   datalab-plugin catalog prepare D:\approved_plugins

The command downloads wheels for the current host without installing them or changing its packages. It uses DataLab's declared runtime dependencies as the reference, so that unrelated plugins already installed on the author's computer are not mistaken for packages shipped with DataLab. Prepare wheels on each target environment when different Python versions or platforms need different artifacts, then copy the complete folder to the offline computer or network share. ``build`` and ``check`` themselves do not download missing packages.

For example, a Windows CPython 3.11 catalog may contain:

.. code-block:: text

   approved_plugins\
       catalog.json
       datalab_tango-0.1.0-py3-none-any.whl
       dependencies\
           pytango-10.3.1-cp311-cp311-win_amd64.whl
           docstring_parser-0.18.0-py3-none-any.whl

The **Available plugins** tab copies the verified dependency artifacts of the chosen release. For **Install from file...**, place the ``dependencies`` folder next to the plugin wheel. DataLab resolves the installation using only these local files, with no package-index fallback and no compilation. Before confirmation it displays the additional packages, versions, sizes, source folder and SHA-256 digests.

Packages already provided by the Python environment or by DataLab are never replaced, upgraded or downgraded. Additional packages are shared by plugins in the same Python runtime. An incompatible installed version, missing transitive wheel or unsupported native wheel causes an explicit refusal; no partial installation is activated. Catalog metadata checks do not replace this resolution against the actual host.

An installation adding dependencies is activated after restarting DataLab. A newly installed plugin is then enabled, including when an explicit enablement list is configured. Uninstalling the plugin keeps the shared dependencies. Different Python/ABI/platform runtimes use separate dependency directories; dependencies incompatible with a changed host are reported rather than repaired over the network.

There is no environment per plugin: plugins requiring contradictory versions cannot coexist. Wheels requiring external system libraries or drivers still need those prerequisites installed separately. Standalone dependencies are intended for library imports, not for supplying an external Python interpreter or promising that their command-line scripts can be run independently. The Web installer remains unchanged.

On Windows, native library paths must also fit the DLL loader's path limit. DataLab uses short generation directory names and refuses an installation whose DLL paths would still be too long, before publishing it; use a shorter configuration directory in that case.

.. _plugin_offline_deployment:

Offline deployment, step by step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This procedure is self-contained: the disconnected computer does not need a package index, a Python development environment, or access to the public plugin website. A local network share may be used without Internet access; a USB drive or an ordinary local folder works as well. Preparation takes place on a connected computer, installation on the disconnected computer.

.. important::

   A plugin catalog is a **source of installation files**, not a plugin search directory. Add it under **Local plugin catalogs**, not **Additional plugin directories**. ``DATALAB_PLUGINS_CATALOG`` supplies catalogs; ``DATALAB_PLUGINS`` supplies code loaded directly. Installing copies the approved files into the user's DataLab configuration directory, so the source folder or network share is not needed to run the installed plugin afterwards.

1. Prepare a complete catalog on a connected computer
"""""""""""""""""""""""""""""""""""""""""""""""""""

The provider needs DataLab installed in a Python environment, with the ``datalab-plugin`` command available. The standalone application does not supply this authoring command; its users only need the prepared files. Use a reference environment matching the target's DataLab version, Python version, operating system, architecture and host package versions. **? > Installation and configuration...** identifies the target installation. A Windows 64-bit CPython 3.11 wheel is not interchangeable with a Linux, 32-bit or CPython 3.12 wheel.

Create a folder, place the plugin's ``.whl`` file at its root, then run:

.. code-block:: powershell

   datalab-plugin catalog prepare D:\approved_plugins
   datalab-plugin catalog check D:\approved_plugins

``prepare`` needs Internet access: it resolves the missing requirements against the reference host, downloads their wheels, including transitive dependencies, into ``dependencies``, and writes ``catalog.json``. It does not install these packages in the reference host. No source distribution is compiled. If it reports a missing wheel or a conflict, fix the reference catalog before transferring it.

Keep all files produced, including the dependency wheels. Do not copy just ``catalog.json`` or just the plugin wheel. If serving several target environments, prepare their wheels on matching reference environments, combine the needed files, then run ``datalab-plugin catalog build D:\approved_plugins`` followed by ``check``. A pure-Python wheel may also restrict the supported Python versions in its metadata.

``check`` checks the catalog files, their integrity and metadata. It is not a substitute for an installation test on the actual target: complete dependency resolution happens at installation time. Before wider deployment, test the complete folder on an equivalent installation with Internet access unavailable.

2. Transfer the folder, preserving its layout
"""""""""""""""""""""""""""""""""""""""""""

Copy the complete folder to a USB drive, local disk or network share accessible from the target computer. Preserve relative paths and file names. A mirror of the public catalog may store files under ``wheels\<digest>\...`` instead of ``dependencies``: preserve those directories too, because ``catalog.json`` references them. The JSON file alone is not a mirror, and DataLab does not automatically export an online catalog for offline use.

For a strictly offline JSON catalog, the installation ``url`` of every release and dependency artifact must resolve to an existing local file. An absolute ``https://`` URL is still downloaded from its server even if the catalog itself is local; placing a wheel beside that JSON does not override its URL. Informational links such as ``repository``, ``documentation`` and ``source_url`` do not replace installation files. Catalogs generated by ``build`` use relative file URLs; published mirrors must retain the matching relative directory tree.

The user needs read access to this source, and write access to their own configuration and temporary directories. Administrator privileges are not required to install a plugin this way. Limit write access to the shared catalog to its maintainers; a SHA-256 check detects changed files but does not certify that a plugin or dependency is safe.

3. Configure the disconnected computer
"""""""""""""""""""""""""""""""""""""

Open **Plugins > Configure plugins... > Plugin settings**. Under **Local plugin catalogs**, click **Add**, select the catalog root (the folder containing ``catalog.json`` or the plugin wheels), and apply the configuration. Do not select the ``dependencies`` subfolder. Open **Available plugins** and click **Refresh** if needed.

Administrators may instead set the following environment variable before launching DataLab:

.. code-block:: powershell

   $env:DATALAB_PLUGINS_CATALOG = 'D:\approved_plugins;\\fileserver\datalab\plugins'

This example applies to DataLab started from the same PowerShell window; a shortcut launched separately does not inherit that window's environment. An organization may also deploy the variable through its usual workstation configuration. Environment-variable folders come before folders saved in DataLab's settings; both come before the online catalog. The first matching plugin ID or distribution wins: releases from different sources are not merged.

.. warning::

   Adding a local catalog does **not** disable the online catalog. If its URL is left configured, DataLab still tries to read it when refreshing, which can cause a network wait and an error on a disconnected computer. The other catalogs remain usable after that error.

To disable that lookup, close DataLab, back up its configuration file, and set ``plugins_catalog_url`` to an empty value in its existing ``[main]`` section. With the standard Windows user profile, DataLab 1.3 and later use ``%USERPROFILE%\.DataLab_v1\DataLab_v1_typed.ini`` (on Linux/macOS, normally ``~/.DataLab_v1/DataLab_v1_typed.ini``). Custom profiles may use another base folder. Do not edit the legacy ``DataLab_v1.ini`` file instead.

.. code-block:: ini

   [main]
   plugins_catalog_url =

Preserve the rest of the file. Do not delete this option to disable it: an absent option uses the default public URL again. Restart DataLab after editing. A private HTTPS URL still requires access to that server; for a strictly file-based offline installation, use a local catalog folder.

4. Install, confirm and restart when requested
""""""""""""""""""""""""""""""""""""""""""""

In **Available plugins**, find the plugin and click **Install**. DataLab chooses the newest release offered for Desktop and the running Python version, checks the plugin file and listed compatible dependency artifacts against their sizes and SHA-256 digests, then plans any additional installation using the local files only. Selection of a release does not guarantee that all its dependencies can be installed: this is checked against the actual host before confirmation.

Read the confirmation: it gives the plugin identity, classes, requirements and fingerprint, and lists the additional packages, their versions, sizes, fingerprints and source folder when needed. Cancel leaves the installation unchanged. If approved, DataLab installs into its private user storage; it does not modify the environment's existing package versions or load code from the shared catalog at every startup.

Restart DataLab if dependencies were added or if replacing an already loaded wheel. Otherwise, DataLab may offer a plugin reload. After restarting, check **Enable/disable plugins** for the plugin's state and any import error, then try one of its actions. The source catalog may be disconnected after successful installation. For a one-off file installation, **Install plugins > Install from file...** uses the same checks; keep ``dependencies`` next to the plugin wheel.

5. Maintain and troubleshoot the offline catalog
"""""""""""""""""""""""""""""""""""""""""""""""

When ``catalog.json`` exists, it is authoritative: DataLab lists and copies only its declared releases and dependency artifacts. For a catalog managed from plugin wheels at its root, run ``build`` and ``check`` again on the preparation computer after adding or changing a plugin **or dependency** wheel, then redistribute the complete folder. Simply copying a new wheel beside an existing JSON file does not update that file. ``prepare`` may require Internet; ``build`` and ``check`` do not fetch missing packages.

Do not run ``build`` on an unchanged mirror whose plugin wheels live only under ``wheels\<digest>\...``: the command scans the catalog root, not that recursive tree, and would remove unrepresented entries from the generated catalog. Update such a mirror by transferring a complete, consistent published catalog and its files. To curate it with the local authoring commands instead, first arrange the plugin wheels at the root and the additional packages in ``dependencies``.

Users refresh **Available plugins** to discover updates, then explicitly click **Update to...**. Updates are not installed automatically. Plugins installed with ``pip`` in the Python environment are identified but not updated by this dialog. Shared additional dependencies are retained when a plugin is uninstalled, and their existing versions remain protected on subsequent installations.

If an installation fails, read the error in the plugin dialog before changing files:

- **Catalog not found or inaccessible:** check the selected root, drive letter, USB drive, network connection and read permissions. A missing share is not an Internet problem. Refresh once it is available.
- **Size or SHA-256 mismatch:** the copied files differ from the catalog. Ask the provider for a complete, consistent copy or regenerate the catalog from the intended wheels; do not bypass the check.
- **No matching wheel or missing requirement:** the wheelhouse may lack a transitive dependency or the wheel for this Python/ABI/platform. Ask the provider to run ``prepare`` on a matching reference environment. The installer never falls back to PyPI.
- **Incompatible installed version:** the host or another plugin already provides a conflicting package version. Ask for a compatible release or reference installation; do not force an upgrade or downgrade to bypass the refusal.
- **Windows DLL path too long:** use a shorter configuration path. **DLL import or missing system library:** check the native package's external prerequisites with the provider; DataLab installs neither drivers nor system packages.
- **Installed but not active:** restart when requested, check the global and individual plugin enablement settings, and consult the import error displayed in **Enable/disable plugins**. Reloading alone does not replace an imported native library.

For local help, use **? > PDF documentation**, then search for *Offline deployment, step by step* or ``DATALAB_PLUGINS_CATALOG``. This PDF is a local file and does not need Internet access. **Online documentation**, the welcome page's documentation link, and external source-code/documentation links open websites instead. The PDF menu action appears only when a local PDF is present; DataLab tries the system language first, then English. If it is absent or unreadable, ask the administrator for the PDF shipped with the matching DataLab version rather than relying on an online link.

.. _plugin_install_from_file:

Install plugins
~~~~~~~~~~~~~~~

This tab installs a plugin shared as a file, without any Python tool: this is the way to add plugins to the standalone version of DataLab. **Install from file...** accepts two kinds of files:

- a wheel (``.whl``) built from a plugin project, for example with ``python -m build``: it must be pure Python (``py3-none-any``), declare its plugin class in the ``datalab.plugins`` entry-point group, and have its requirements provided by the host or the adjacent dependency wheelhouse (see :ref:`plugin_dependencies`);
- a single Python module named ``datalab_<name>.py``.

Before installing, DataLab checks the file without running it. For a wheel, the confirmation shows the distribution, version, entry points and requirements from its metadata. For a single-file module, it shows the module name, file, size and SHA-256 digest, without claiming to know its version, classes or dependencies. A plugin runs with the same rights as DataLab: install only plugins from authors you trust.

Installed plugins are kept in the ``installed_plugins`` folder of the DataLab configuration directory (e.g. ``~/.DataLab_v1/installed_plugins``) and listed in the tab, where **Uninstall** removes them. Installations not adding dependencies retain the plugin reload workflow; a newly installed plugin is enabled. Additional dependencies and new versions of already loaded plugins are only activated at the next start of DataLab.

When reloading plugins, DataLab performs the following steps:

1. Unregister currently active plugins and their owned computations,
2. Clear plugin actions from signal and image panels,
3. Re-discover and reload plugin modules,
4. Re-register enabled plugins,
5. Re-register owned computations,
6. Recreate plugin actions and refresh menus.

This workflow allows iterative plugin development while DataLab is running.

.. note::

  Plugin enable/disable state is persisted in DataLab settings. Disabled plugins remain listed in the configuration dialog and can be re-enabled later. The global third-party plugins setting in Preferences is also applied immediately: disabling it removes plugin actions and greys out the Plugins menu and status indicator, while enabling it reloads plugins automatically.

Hot-reload workflow for plugin development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hot-reload feature is designed to accelerate the plugin development cycle.
Here is the recommended workflow:

1. Start DataLab normally.
2. Create or edit your plugin file (e.g. ``datalab_myplugin.py``) in one of the
   plugin directories (e.g. ``~/.DataLab_v1/plugins``).
3. In DataLab, use **Plugins > Reload plugins** to pick up your changes instantly.
4. Test your plugin actions directly in the running application.
5. Iterate: edit the file, reload, test — without restarting DataLab.

To selectively enable or disable specific plugins during development, use
**Plugins > Configure plugins...** and its **Apply and reload plugins** button,
so that changes take effect without leaving the dialog.

Plugin API helpers
------------------

Plugins inheriting from :class:`datalab.plugins.PluginBase` have direct access to useful helpers:

- ``self.signalpanel`` and ``self.imagepanel``: access to panel APIs and action handlers,
- ``self.proxy``: a :class:`datalab.control.proxy.LocalProxy` instance for object creation and processing,
- ``show_warning``, ``show_error``, ``show_info``, ``ask_yesno``: convenience dialog methods,
- ``edit_new_signal_parameters`` and ``edit_new_image_parameters``: helpers for object parameter dialogs.

These helpers simplify plugin code and keep it consistent with DataLab behavior.

Processing plugins should override ``register_computations()`` and register each
feature with a stable ``feature_id`` of the form ``<plugin_id>:<local_id>`` and
``owner_plugin_id=self.plugin_id``. DataLab calls this hook after the signal and
image panels exist. Owned features are removed automatically when the plugin is
disabled, reloaded, or uninstalled. ``create_actions()`` may then reference the
registered feature by its stable ID.

A registered feature runs a Sigima computation function, decorated with :func:`sigima.proc.decorator.computation_function`. For a 1-to-1 feature without parameters, the simplest is to wrap a function processing NumPy arrays with :class:`sigima.proc.signal.Wrap1to1Func` (signal Y values) or :class:`sigima.proc.image.Wrap1to1Func` (image data), as in the processing plugin example (see :ref:`plugin_example_processing`). DataLab refuses other functions when they are registered.

Headless recipes
----------------

Plugins may expose versioned scientific workflows through the class-level
``RECIPES`` tuple. Each :class:`datalab.plugins.recipes.RecipeDescriptor` declares a
stable ID namespaced by the plugin ID, typed input slots, an optional guidata
``DataSet`` parameter class, and a headless callable:

Recipe descriptors remain owned by the current plugin class and are read
through :meth:`datalab.plugins.PluginBase.get_recipes`; they are not copied into
a separate mutable registry. Reloading the plugin class therefore replaces its
recipe declarations together with its implementation.

.. code-block:: python

  from datalab.plugins import PluginBase, PluginInfo
  from datalab.plugins.recipes import (
    RecipeDescriptor,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeOutcome,
  )

  def run_quick_check(inputs, parameters, context):
    source = inputs["source"][0]
    context.raise_if_cancelled()
    output = source.copy()
    context.report_progress(1.0, "Quick check complete")
    return RecipeOutcome(
      objects=(RecipeObjectOutput("checked-signal", output),),
    )

  class MyPlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(
      id="org.example.my-plugin",
      name="My Plugin",
      version="1.0.0",
    )
    RECIPES = (
      RecipeDescriptor(
        recipe_id="org.example.my-plugin:quick-check",
        plugin_version="1.0.0",
        title="Quick check",
        version="1.0.0",
        inputs=(RecipeInputSlot("source", "signal", "one"),),
        run=run_quick_check,
      ),
    )

    def create_actions(self):
      pass

The descriptor's ``plugin_version`` must match ``PLUGIN_INFO.version``. Keeping
the plugin and recipe versions explicit lets execution records identify both
the installed implementation and the independently versioned workflow.

The recipe callable receives a mapping from slot IDs to tuples of Sigima
``SignalObj`` or ``ImageObj`` instances, the parameter ``DataSet`` instance (or
``None``), and a :class:`datalab.plugins.recipes.RecipeExecutionContext`. The context
provides technology-neutral progress and cancellation callbacks and has no Qt
dependency.

When ``parameter_class`` is ``None``, the recipe declares no parameters and the
callable receives ``None``. Otherwise, consumers must provide an instance of the
declared ``DataSet`` subclass. The recipe runner enforces this rule before
execution.

A :class:`datalab.plugins.recipes.RecipeOutcome` contains named object outputs,
structured diagnostics, and optional scalar results. A ``TableResult`` or
``GeometryResult`` is wrapped in :class:`datalab.plugins.recipes.RecipeResultOutput` and
uses ``anchor_id`` to reference the named object output that will own it. The
contract validates these references without assigning DataLab workspace UUIDs.
Workspace mutation and atomic commit are responsibilities of the recipe runner,
not of the recipe callable.

Describing recipe inputs
~~~~~~~~~~~~~~~~~~~~~~~~

DataLab shows users what each recipe expects and whether the current selection suits it. Describe every input slot accordingly:

.. code-block:: python

  from datalab.plugins.recipes import RecipeInputSlot, RecipeMetadataRequirement

  FLAT_FRAMES = RecipeInputSlot(
    "flat_frames",
    "image",
    "many",
    title="Flat frames",
    description="Uniformly illuminated frames, two or more per exposure level",
    min_count=4,
    metadata=(
      RecipeMetadataRequirement(
        "org.example.my-plugin.exposure_time_s", "Exposure time in seconds"
      ),
      RecipeMetadataRequirement(
        "org.example.my-plugin.frame_role", "dark or flat", required=False
      ),
    ),
  )

``min_count`` is the minimum number of objects bound to a ``many`` slot. Required metadata keys must be present on every bound object; optional keys are hints shown to the user. The recipe runner enforces both rules before the recipe callable is called.

Two optional hooks of :class:`datalab.plugins.recipes.RecipeDescriptor` refine these declarations:

- ``suggest_bindings(candidates)`` returns a mapping from slot IDs to some of the candidate objects, for example to split frames by role from their metadata. Without it, compatible candidates go to the only slot accepting their type, and slots sharing a type are left for the user to assign.
- ``check_inputs(inputs, parameters)`` returns a list of :class:`datalab.plugins.recipes.RecipeDiagnostic` values for fast, recipe-specific checks such as shapes, counts per level, or units. It must not compute results. Error diagnostics block the run and warnings are shown before it. An exception raised by the hook is reported as an ``input-check-failed`` error.

:func:`datalab.plugins.recipe_binding.assess_recipe_inputs` combines the slot declarations and both hooks. It returns a :class:`datalab.plugins.recipe_binding.RecipeReadiness` with a status (``no_input``, ``needs_assignment``, ``not_ready``, ``warnings`` or ``ready``), the proposed bindings, structured issues that hosts translate, and the recipe diagnostics. The module has no Qt dependency, so DataLab-Web uses the same code.

Running a recipe on Desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most plugins do not call the runner directly. :meth:`datalab.plugins.PluginBase.start_recipe` runs a recipe on the current selection with DataLab's generic launcher. It assesses the selection, opens an input assignment dialog only when the bindings are ambiguous or invalid, edits the declared parameters, and then calls the runner. A plugin menu action therefore only needs:

.. code-block:: python

  def run_quick_check(self):
    return self.start_recipe("org.example.my-plugin:quick-check")

The **Run on selection...** buttons of the **Applications** catalog use the same launcher. A plugin that needs a dedicated interface for one recipe may map its recipe ID to a plugin method name in the class-level ``RECIPE_LAUNCHERS`` mapping; the catalog then calls this method, without arguments, instead of the generic launcher.

Use :class:`datalab.gui.plugins.recipe_runner.RecipeRunner` from the GUI thread to run a recipe on explicit inputs. It validates inputs and parameters, executes the headless callable, and commits its outcome to the DataLab workspace:

.. code-block:: python

  from datalab.gui.plugins.recipe_runner import RecipeRunner

  descriptor = self.get_recipes()[0]
  outcome = RecipeRunner(self.main).run(
    descriptor,
    inputs={"source": (source_signal,)},
  )

The runner rejects missing, extra, mistyped, or incorrectly sized input slots
before recipe code is called. It also enforces ``min_count``, required metadata and the error diagnostics of ``check_inputs``, validates the optional ``DataSet``
instance and checks cancellation before execution, after execution, and
immediately before commit. Recipe code must remain headless and must not mutate
the workspace itself.

Only a validated :class:`datalab.plugins.recipes.RecipeOutcome` reaches the commit
phase. The Desktop runner creates one group per output panel, using the recipe
title by default, then adds all signal and image outputs. Scalar result IDs are
persisted as ``<recipe-id>:<result-id>`` function names on their named anchor
objects, so several tables or geometries may coexist without metadata-key
collisions. A failure during insertion removes every object and group created
by that invocation and restores the previous workspace modified state and
current panel.

Local execution provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~

After successful headless execution, the Desktop runner stores the same
:class:`datalab.plugins.recipes.RecipeRunRecord` in every output object's metadata.
Its versioned, JSON-compatible payload contains a shared run UUID, plugin and
recipe IDs and versions, resolved parameters as JSON, named input and output
UUIDs, DataLab and Sigima versions, the completed status, and UTC start and
finish timestamps.

The metadata survives JSON export and HDF5 workspace round-trips. The shared
run UUID links outputs across the Signal and Image panels without adding group
metadata or a global workflow history. Failed or cancelled invocations do not
commit outputs and therefore leave no run record in the workspace.

Packaged examples
~~~~~~~~~~~~~~~~~

Plugins may expose native DataLab workspaces through the class-level
``EXAMPLES`` tuple. Each :class:`datalab.plugins.examples.PluginExample` uses a
``package:relative/path`` resource instead of a development filesystem path:

.. code-block:: python

  from datalab.plugins.examples import PluginExample

  class MyPlugin(PluginBase):
    EXAMPLES = (
      PluginExample(
        id="quickstart",
        title="Quick start",
        description="Small deterministic workspace",
        resource="datalab_my_plugin:examples/quickstart.h5",
        recipe_ids=("org.example.my-plugin:quick-check",),
        expected_checks=("summary-table",),
      ),
    )

    def create_actions(self):
      pass

The package must include the resource in its wheel. ``resolve()`` returns an
``importlib.resources`` traversable that also works for packages imported from
a ZIP archive. For APIs requiring a filesystem path, use ``as_file()`` as a
context manager and do not retain the returned path after the context exits.

:meth:`datalab.plugins.PluginBase.get_examples` validates unique local IDs and
recipe references. A registered plugin may call ``open_example("quickstart")``
to materialize and load a native DataLab HDF5 workspace. Opening clears the
current workspace by default; pass ``reset_all=False`` to merge it instead.

``recipe_ids`` lists the recipes an example is designed for. One example may serve several recipes, for instance a dark ramp analyzed by two methods. A plugin may also generate an example in memory by overriding ``materialize_example()`` to return a :class:`datalab.plugins.examples.PluginExampleData`; its ``parameter_values`` mapping gives, for each recipe ID, the parameter values suited to the generated objects.

The **Applications** catalog presents each recipe as a method with its expected inputs, a live status for the current selection, a **Run on selection...** button, and the examples designed for it. **Try with this example** calls :meth:`datalab.plugins.PluginBase.try_example`: it opens the example, prefills the recipe parameters with the example values, and runs the recipe once the user accepts them. Examples without ``recipe_ids`` are listed as datasets, which only open.

Plugin tools
~~~~~~~~~~~~

A recipe covers a headless analysis run by DataLab. Any other interaction owned by an application plugin, such as a wizard, an interactive editor, a data preparation step or a simulated instrument, is a tool. Declare tools in the class-level ``TOOLS`` tuple of :class:`datalab.plugins.tools.PluginTool` values:

.. code-block:: python

  from datalab.plugins.recipes import RecipeObjectType
  from datalab.plugins.tools import PluginTool, ToolSelection

  class MyPlugin(PluginBase):
    TOOLS = (
      PluginTool(
        id="annotate",
        title="Annotate frames...",
        launcher="annotate_frames",
        description="Set the acquisition metadata of the selected frames",
        object_type=RecipeObjectType.IMAGE,
        selection=ToolSelection.AT_LEAST_ONE,
      ),
    )

    def annotate_frames(self):
      ...

DataLab lists each tool in two places: in the **Tools** section of the **Applications** catalog, and in the plugin's submenu of the **Plugins** menu, after the plugin's own actions. ``object_type`` gives the panel whose menu shows the tool; with ``None``, both panels show it. ``selection`` states which selection the tool needs: ``NONE`` (the default), ``EXACTLY_ONE``, ``AT_LEAST_ONE`` or ``AT_LEAST_TWO`` objects of ``object_type``. When the selection does not match, DataLab disables the menu entry and the catalog button, and the button tooltip tells what to select. ``icon`` takes a ``package:path`` resource or a DataLab icon file name; by default, the tool uses the plugin icon.

The launcher is called without arguments. :meth:`datalab.plugins.PluginBase.get_tools` validates unique local IDs and the tool methods, and rejects tools declared by plugins without the ``APPLICATION`` capability. :meth:`datalab.plugins.PluginBase.assess_tool` returns the reason why a tool cannot be opened for the current selection, or ``None``.

Instruments
^^^^^^^^^^^

A tool may name an ``instrument`` method instead of a ``launcher``. This method returns a :class:`datalab.plugins.instruments.PluginInstrument`, and DataLab opens a window for it: a live view on the left, the instrument settings on the right, a **Live** button that refreshes the view continuously, and an **Acquire** button. The plugin writes no user interface code:

.. code-block:: python

  import guidata.dataset as gds
  from datalab.plugins.instruments import (
    InstrumentAcquisition,
    InstrumentFrame,
    PluginInstrument,
  )

  class SourceSettings(gds.DataSet):
    amplitude = gds.FloatItem("Amplitude", default=1.0, unit="V")
    count = gds.IntItem("Signals per acquisition", default=10, min=1)

  class Source(PluginInstrument):
    live_interval_ms = 200

    def __init__(self):
      super().__init__(SourceSettings())

    def preview(self):
      signal = self.make_signal()
      return InstrumentFrame([signal], summary="...", value_range=(-2.0, 2.0))

    def acquire(self):
      signals = [self.make_signal() for _ in range(self.settings.count)]
      return InstrumentAcquisition("Source - acquisition 001", signals)

  class MyPlugin(PluginBase):
    TOOLS = (
      PluginTool(
        id="source",
        title="Signal source...",
        instrument="signal_source",
        object_type=RecipeObjectType.SIGNAL,
      ),
    )

    def signal_source(self):
      return Source()

DataLab edits :attr:`~datalab.plugins.instruments.PluginInstrument.settings` in place, with the active state, groups and tabs of the guidata DataSet, then calls :meth:`~datalab.plugins.instruments.PluginInstrument.preview` after each change and, in live mode, every ``live_interval_ms`` milliseconds. An :class:`~datalab.plugins.instruments.InstrumentFrame` holds signals drawn together or a single image, a short summary shown below the view, and an optional fixed Y range (signals) or color range (image). :meth:`~datalab.plugins.instruments.PluginInstrument.acquire` returns an :class:`~datalab.plugins.instruments.InstrumentAcquisition`: DataLab adds its objects to a new group of the matching panel and selects them, so that a recipe can run on them at once. Both methods raise ``ValueError`` with a user-facing message when the settings cannot be used; DataLab shows the message in the window.

DataLab creates the instrument once per tool and keeps it until the plugins are reloaded, so the settings remain from one opening to the next.

DataLab-Web supports tools too: it lists them in the same places, opens instruments in the same window, and calls launchers in the plugin's Pyodide host.

Welcome page tiles
------------------

Every active plugin declaring the ``APPLICATION`` capability adds a tile to the **Applications** section, at the top of the :ref:`welcome_page`. By default, this tile is built from the plugin's ``PluginInfo`` name, description and icon. Clicking this tile opens the plugin page of the **Applications** catalog.

The plugin icon is declared with ``PluginInfo.icon``, either as a ``package:relative/path`` resource (SVG or bitmap image shipped in the plugin wheel) or as the file name of a DataLab icon. It is shown on the default tile, in the **Applications** catalog and in the plugin configuration dialog; plugins without icon use a generic plugin icon. To offer other entry points, for example to open an example directly, declare the class-level ``WELCOME_TILES`` tuple of :class:`datalab.plugins.tiles.WelcomeTile` values:

.. code-block:: python

  from datalab.plugins.tiles import WelcomeTile

  class MyPlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(
      id="org.example.my-plugin",
      name="My Plugin",
      icon="datalab_my_plugin:icons/my_plugin.svg",
      capabilities=(PluginCapability.APPLICATION,),
    )
    WELCOME_TILES = (
      WelcomeTile(id="application", title="My Plugin"),
      WelcomeTile(
        id="quickstart",
        title="Open quick start",
        description="Open the packaged quick start workspace",
        icon="datalab_my_plugin:icons/quickstart.svg",
        launcher="open_quickstart",
      ),
    )

    def open_quickstart(self):
      self.launch_example("quickstart")

    def create_actions(self):
      pass

Declared tiles replace the default tile. The first one is the main tile of the plugin; the others are shown next to it when the **Applications** section has room for them, and otherwise become actions of its menu, opened from its "…" button or by right-clicking it. A tile without ``launcher`` opens the plugin page of the **Applications** catalog; otherwise, the named plugin method is called without arguments. A tile without ``icon`` uses the plugin icon. :meth:`datalab.plugins.PluginBase.get_welcome_tiles` validates unique local IDs and launcher methods, and rejects tiles declared by plugins without the ``APPLICATION`` capability. The welcome page refreshes its tiles when plugins are loaded, reloaded or disabled, and reports errors raised by a launcher without closing DataLab.

The user decides which applications are shown and in which order: the welcome page shows a limited number of tile rows, pinned applications first, then recently used ones, then the others by name (see :ref:`welcome_page`). A plugin therefore cannot rely on its tile being visible: its features must remain available from the **Applications** catalog.

.. _plugin_generator:

Creating a plugin with ``datalab-plugin create``
-------------------------------------------------

The ``datalab-plugin`` command is installed with DataLab as a Python package; the standalone version of DataLab does not provide it. Run it without options to create a single-file plugin:

.. code-block:: bash

  datalab-plugin create

The command only asks for the plugin name and its description, then writes a ``datalab_<name>.py`` file to the user plugin directory (see :ref:`plugin_locations`). Select **Plugins > Reload plugins** in DataLab, or restart DataLab: the plugin adds its submenu to the **Plugins** menu, with a sample processing of the selected signals. To implement your own processing, edit the ``sample_processing`` function of the file, which receives the Y values of a signal as a NumPy array and returns new ones, then reload the plugins.

Options adapt the generated file:

- ``--object-kind image``: the sample processing applies to images, and its function receives the image data.
- ``--capability``: without the ``processing`` capability, the submenu holds an action showing a message, to replace with your own.
- A destination folder, given as argument: it must then be added to the plugin directories (see :ref:`plugin_search_paths`).
- ``--package`` and ``--plugin-id``: the module name and the plugin ID are otherwise derived from the plugin name. The module name must start with ``datalab_``.

An existing file is never overwritten. To share the plugin, send its file: other users install it with **Plugins > Configure plugins... > Install plugins** (see :ref:`plugin_install_from_file`).

The catalogs currently require a wheel, not a particular source architecture (see :ref:`plugin_forms`). To develop a larger plugin with packaging and tests already configured, create a layered project with ``--project`` instead (see :ref:`plugin_project`).

.. _plugin_project:

Creating a layered plugin project
----------------------------------

With ``--project``, the installed ``datalab-plugin`` command creates a small,
installable project that follows the stable plugin and feature ownership
contracts. Run it without other options for an interactive setup:

.. code-block:: bash

  datalab-plugin create --project

The command asks for the display name and the GitHub account (user or organization) that will host the project, and offers defaults for the Python package, reverse-domain plugin ID, description, and destination. The default plugin ID is ``io.github.<account>.<name>``; ``org.datalab.`` IDs are reserved for the DataLab-Platform organization, and ``org.example.`` is used when no account is given. For scripts or reproducible setup, pass the values explicitly:

.. code-block:: bash

  datalab-plugin create --project datalab-camera-characterization \
    --name "Camera Characterization" \
    --github-account DataLab-Platform \
    --package datalab_camera_characterization \
    --plugin-id org.datalab.camera-characterization \
    --description "Characterize scientific cameras" \
    --capability application \
    --capability processing \
    --object-kind image

The generated ``src``-layout project applies the architecture validated by the
Camera pilot:

.. code-block:: text

  package root (host-independent identity)
  ├── core
  ├── workflow
  └── adapters
      ├── desktop.py
      └── web.py

``core`` holds host-independent domain behavior, ``workflow`` may use DataLab's
headless recipe contracts, and ``adapters`` owns host integration. The
``datalab.plugins`` entry point targets the Desktop adapter; Web support starts
explicitly as ``unsupported``. A generated AST test prevents ``core`` and
``workflow`` from importing GUI, browser, or adapter modules.

The project also includes a stable :class:`datalab.plugins.PluginInfo`, an owned
sample processing when the ``processing`` capability is selected, Ruff
settings, architecture and contribution documentation, a changelog, a README,
and a BSD-3-Clause license. The command refuses to overwrite an existing
destination.

The template remains deliberately small. Add domain recipes, DataSet
parameters, packaged examples, and specialized CI only when the plugin needs
them.

To try the plugin, add the generated directory as a plugin directory (see :ref:`plugin_search_paths`) and reload the plugins: DataLab loads the plugin class from ``adapters/desktop.py`` without installing the project. To validate the project, install it with its test dependencies and run its checks from the generated directory:

.. code-block:: bash

  python -m pip install -e ".[test]"
  python -m pytest
  python -m ruff check .

The generated ``pyproject.toml`` requires the current DataLab major version (for example ``datalab-platform >= 1.4, < 2``) and declares the project URLs when the GitHub account is known. The generated ``.github/workflows/release.yml`` publishes the plugin without any PyPI account: pushing a tag such as ``v0.1.0`` builds the wheel, checks that the tag matches the package version, attaches the wheel to a GitHub release, and prints the entry to submit to the `DataLab plugin catalog <https://github.com/DataLab-Platform/plugins>`_, including the SHA-256 digest of the wheel. Users then install the wheel with **Plugins > Configure plugins... > Install plugins** (see :ref:`plugin_install_from_file`). Listing the plugin in the catalog is optional; once listed, it can also be installed from the **Available plugins** tab (see :ref:`plugin_catalog`).

How to develop a plugin?
------------------------

The recommended approach to developing a plugin is to derive from an existing example and adapt it to your needs. You can explore the source code in the `datalab/plugins/builtin` folder or refer to community-contributed examples.

.. note::

   Most of DataLab's signal and image processing functionalities have been externalized into a dedicated library called **Sigima** (`https://sigima.readthedocs.io/en/latest/ <https://sigima.readthedocs.io/en/latest/>`_). When developing DataLab plugins, you will typically import and use many Sigima functions and features to perform data processing, analysis, and visualization tasks. Sigima provides a comprehensive set of tools for scientific data manipulation that can be leveraged directly in your plugins.

To develop in your usual Python environment (e.g., with an IDE like `Spyder <https://www.spyder-ide.org/>`_), you can:

1. **Install DataLab in your Python environment**, using one of the following methods:

   - :ref:`install_conda`
   - :ref:`install_pip`
   - :ref:`install_wheel`
   - :ref:`install_source`

2. **Or add the `datalab` package manually to your Python path**:

   - Download the source from the `PyPI page <https://pypi.org/project/datalab-platform/>`_,
   - Unzip the archive,
   - Add the `datalab` directory to your PYTHONPATH (e.g., using the *PYTHONPATH Manager* in Spyder).

.. note::

   Even if you’ve installed `datalab` in your environment, you cannot run the full DataLab application directly from an IDE. You must launch DataLab via the command line or using the installer-created shortcut to properly test your plugin.

.. _plugin_example_processing:

Example: processing plugin
--------------------------

This example registers a processing feature with stable identity and ownership,
then creates an action which dispatches it through the processor registry:

.. literalinclude:: ../../../plugins/examples/datalab_custom_func.py

Example: input/output plugin
----------------------------

Here is a simple example of a plugin that adds new file formats to DataLab.

.. literalinclude:: ../../../datalab/plugins/builtin/datalab_imageformats.py

Example templates used by the test suite
----------------------------------------

DataLab also provides plugin templates used by integration tests in
``datalab/tests/features/plugins/templates``. They are useful as development references for:

- basic valid plugin structure,
- nested plugin menus,
- plugins with dialog actions,
- plugins with many actions,
- plugins with long descriptions.

The corresponding feature tests are located in
``datalab/tests/features/plugins/plugins_app_test.py`` and cover plugin lifecycle,
hot-reload behavior, error handling, duplicate names, and configuration filtering.

Other examples
--------------

Other examples of plugins can be found in the `plugins/examples` directory of the DataLab source code (explore `here on GitHub <https://github.com/DataLab-Platform/DataLab/tree/main/plugins/examples>`_).

Plugins and DataLab-Web
-----------------------

Plugins are largely portable between the desktop application and :ref:`DataLab-Web
<ecosystem>`, the browser-native edition of the platform. The same :class:`datalab.plugins.PluginBase`
subclass can run in both, because the plugin API is shared. A few constraints apply to the
browser runtime, however:

- **Parameter dialogs must be opened asynchronously.** In the browser, the synchronous
  ``param.edit(self.main)`` call cannot block the event loop; use
  ``await param.edit_async(self.main)`` instead. A plugin written this way still works
  unchanged on the desktop.
- **No Qt-only graphical interfaces.** Plugins that embed custom Qt widgets or rely on
  PlotPy interactive tools are desktop-only; their graphical parts have no equivalent in the
  browser.
- **Execution happens inside the browser** (WebAssembly), with no native file-system access
  beyond the in-memory file system.

As a result, a plugin that relies on a custom graphical user interface may not be fully compatible with DataLab-Web. The `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ indicates which of its plugins also support DataLab-Web. Refer to the DataLab-Web documentation for the practical guide to loading plugins in the browser.

Migrating from v0.20 to v1.0
----------------------------

If you have existing plugins written for DataLab v0.20, please refer to the :ref:`migration guide <migration_v020_to_v100>` for detailed instructions on updating your plugins to work with DataLab v1.0.

Public API
----------

.. automodule:: datalab.plugins
    :members: PluginInfo, PluginBase, FormatInfo, ImageFormatBase, ClassicsImageFormat, SignalFormatBase

.. automodule:: datalab.plugins.recipes
  :members:

.. automodule:: datalab.plugins.examples
  :members:

.. automodule:: datalab.plugins.tiles
  :members:

.. automodule:: datalab.plugins.tools
  :members:

.. automodule:: datalab.plugins.instruments
  :members:

.. automodule:: datalab.gui.plugins.recipe_runner
  :members:
