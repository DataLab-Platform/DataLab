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

To be recognized by the historical module scan, a local plugin file must:

- Be a Python module whose name **starts with** ``datalab_`` (e.g. ``datalab_myplugin.py``),
- Contain a class that **inherits from** :class:`datalab.plugins.PluginBase`,
- Include a class attribute named ``PLUGIN_INFO``, which must be an instance of :class:`datalab.plugins.PluginInfo`,
- Implement the ``create_actions`` method.

Plugins distributed as installed Python packages may instead expose their
``PluginBase`` subclass through the ``datalab.plugins`` entry-point group,
without relying on the module-name prefix.

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

   Only Python files whose names start with ``datalab_`` will be scanned for plugins.

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

- An installed Python package may declare its plugin class through the
  standard ``datalab.plugins`` entry-point group:

  .. code-block:: toml

    [project.entry-points."datalab.plugins"]
    my-plugin = "my_plugin.plugin:MyPlugin"

  The target must be a :class:`datalab.plugins.PluginBase` subclass. This is
  the recommended distribution mechanism for plugins installed with ``pip``.

- The user plugin directory:
  Typically `~/.DataLab/plugins` on Linux/macOS or `C:/Users/YourName/.DataLab/plugins` on Windows.

- A custom plugin directory:
  Configurable in DataLab's preferences.

- The standalone distribution directory:
  If using a frozen (standalone) build, the `plugins` folder located next to the executable is scanned.

- The internal `datalab/plugins` folder (not recommended for user plugins):
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

Managing plugins in DataLab
---------------------------

The **Plugins** menu provides two dedicated actions:

- **Configure plugins...**
  Opens the plugin configuration dialog where you can enable or disable plugins individually.
  After saving changes, DataLab can reload plugins immediately without restarting the application.

- **Reload plugins**
  Reloads plugin modules from disk without restarting DataLab.

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
   plugin directories (e.g. ``~/.DataLab/plugins``).
3. In DataLab, use **Plugins > Reload plugins** to pick up your changes instantly.
4. Test your plugin actions directly in the running application.
5. Iterate: edit the file, reload, test — without restarting DataLab.

To selectively enable or disable specific plugins during development, use
**Plugins > Configure plugins...**. The dialog lists all discovered plugins
with their name, version, description, and file path. Toggling a plugin
takes effect immediately after closing the dialog.

Plugin API helpers
------------------

Plugins inheriting from :class:`datalab.plugins.PluginBase` have direct access to useful helpers:

- ``self.signalpanel`` and ``self.imagepanel``: access to panel APIs and action handlers,
- ``self.proxy``: a :class:`datalab.control.proxy.LocalProxy` instance for object creation and processing,
- ``show_warning``, ``show_error``, ``show_info``, ``ask_yesno``: convenience dialog methods,
- ``edit_new_signal_parameters`` and ``edit_new_image_parameters``: helpers for object parameter dialogs.

These helpers simplify plugin code and keep it consistent with DataLab behavior.

Processing plugins should override ``register_computations()`` and register each
feature with a stable, namespaced ``feature_id`` and
``owner_plugin_id=self.plugin_id``. DataLab calls this hook after the signal and
image panels exist. Owned features are removed automatically when the plugin is
disabled, reloaded, or uninstalled. ``create_actions()`` may then reference the
registered feature by its stable ID.

Headless recipes
----------------

Plugins may expose versioned scientific workflows through the class-level
``RECIPES`` tuple. Each :class:`datalab.recipes.RecipeDescriptor` declares a
stable ID namespaced by the plugin ID, typed input slots, an optional guidata
``DataSet`` parameter class, and a headless callable:

Recipe descriptors remain owned by the current plugin class and are read
through :meth:`datalab.plugins.PluginBase.get_recipes`; they are not copied into
a separate mutable registry. Reloading the plugin class therefore replaces its
recipe declarations together with its implementation.

.. code-block:: python

  from datalab.plugins import PluginBase, PluginInfo
  from datalab.recipes import (
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
    )
    RECIPES = (
      RecipeDescriptor(
        recipe_id="org.example.my-plugin:quick-check",
        title="Quick check",
        version="1.0.0",
        inputs=(RecipeInputSlot("source", "signal", "one"),),
        run=run_quick_check,
      ),
    )

    def create_actions(self):
      pass

The recipe callable receives a mapping from slot IDs to tuples of Sigima
``SignalObj`` or ``ImageObj`` instances, the parameter ``DataSet`` instance (or
``None``), and a :class:`datalab.recipes.RecipeExecutionContext`. The context
provides technology-neutral progress and cancellation callbacks and has no Qt
dependency.

When ``parameter_class`` is ``None``, the recipe declares no parameters and the
callable receives ``None``. Otherwise, consumers must provide an instance of the
declared ``DataSet`` subclass. The recipe runner enforces this rule before
execution.

A :class:`datalab.recipes.RecipeOutcome` contains named object outputs,
structured diagnostics, and optional scalar results. A ``TableResult`` or
``GeometryResult`` is wrapped in :class:`datalab.recipes.RecipeResultOutput` and
uses ``anchor_id`` to reference the named object output that will own it. The
contract validates these references without assigning DataLab workspace UUIDs.
Workspace mutation and atomic commit are responsibilities of the recipe runner,
not of the recipe callable.

Running a recipe on Desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`datalab.gui.recipe_runner.RecipeRunner` from the GUI thread to
validate inputs and parameters, execute the headless callable, and commit its
outcome to the DataLab workspace:

.. code-block:: python

  from datalab.gui.recipe_runner import RecipeRunner

  descriptor = self.get_recipes()[0]
  outcome = RecipeRunner(self.main).run(
    descriptor,
    inputs={"source": (source_signal,)},
  )

The runner rejects missing, extra, mistyped, or incorrectly sized input slots
before recipe code is called. It also validates the optional ``DataSet``
instance and checks cancellation before execution, after execution, and
immediately before commit. Recipe code must remain headless and must not mutate
the workspace itself.

Only a validated :class:`datalab.recipes.RecipeOutcome` reaches the commit
phase. The Desktop runner creates one group per output panel, using the recipe
title by default, then adds all signal and image outputs. Scalar result IDs are
persisted as ``<recipe-id>:<result-id>`` function names on their named anchor
objects, so several tables or geometries may coexist without metadata-key
collisions. A failure during insertion removes every object and group created
by that invocation and restores the previous workspace modified state and
current panel.

How to develop a plugin?
------------------------

The recommended approach to developing a plugin is to derive from an existing example and adapt it to your needs. You can explore the source code in the `datalab/plugins` folder or refer to community-contributed examples.

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

Example: processing plugin
--------------------------

This example registers a processing feature with stable identity and ownership,
then creates an action which dispatches it through the processor registry:

.. literalinclude:: ../../../plugins/examples/datalab_custom_func.py

Example: input/output plugin
----------------------------

Here is a simple example of a plugin that adds new file formats to DataLab.

.. literalinclude:: ../../../datalab/plugins/datalab_imageformats.py

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

As a result, a plugin that relies on a custom graphical user interface may not be fully
compatible with DataLab-Web. A curated set of web-compatible plugins will be provided
separately; in the meantime, refer to the DataLab-Web documentation for the practical guide
to loading plugins in the browser.

Migrating from v0.20 to v1.0
----------------------------

If you have existing plugins written for DataLab v0.20, please refer to the :ref:`migration guide <migration_v020_to_v100>` for detailed instructions on updating your plugins to work with DataLab v1.0.

Public API
----------

.. automodule:: datalab.plugins
    :members: PluginInfo, PluginBase, FormatInfo, ImageFormatBase, ClassicsImageFormat, SignalFormatBase

.. automodule:: datalab.recipes
  :members:

.. automodule:: datalab.gui.recipe_runner
  :members:
