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
  Wheels and modules installed from **Plugins > Configure plugins...**, either from a file in the **Install plugins** tab (see :ref:`plugin_install_from_file`) or from the plugin catalog in the **Available plugins** tab (see :ref:`plugin_catalog`).

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

The tab also holds a **Compatibility warnings** option to hide warnings for
incompatible DataLab v0.20 plugins.

.. _plugin_catalog:

Available plugins
~~~~~~~~~~~~~~~~~

This tab lists the plugins of the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_, with their description, license and source code link. Plugins maintained by the DataLab team are marked **Official**; the others are **Community** plugins, maintained by their authors. A search field filters the list.

**Install** downloads the newest release available for DataLab desktop, checks that its size and SHA-256 digest match the catalog, then installs it as described in :ref:`plugin_install_from_file`, after the same confirmation. When the catalog offers a newer version of a plugin installed this way, the button becomes **Update to...**. Plugins installed with ``pip`` in the Python environment are shown as such and are not modified.

The catalog is downloaded only when the tab is shown or refreshed. Its address is the ``plugins_catalog_url`` option of the DataLab configuration file: it may point to a private copy of the catalog, including a ``file:`` URL for a folder shared on the network, or be emptied to disable the catalog.

.. _plugin_install_from_file:

Install plugins
~~~~~~~~~~~~~~~

This tab installs a plugin shared as a file, without any Python tool: this is the way to add plugins to the standalone version of DataLab. **Install from file...** accepts two kinds of files:

- a wheel (``.whl``) built from a plugin project, for example with ``python -m build``: it must be pure Python (``py3-none-any``), declare its plugin class in the ``datalab.plugins`` entry-point group, and depend only on packages already provided by DataLab;
- a single Python module named ``datalab_<name>.py``.

Before installing, DataLab checks the file without running it and shows its name, version, plugin classes, dependencies and SHA-256 digest. A plugin runs with the same rights as DataLab: install only plugins from authors you trust.

Installed plugins are kept in the ``installed_plugins`` folder of the DataLab configuration directory (e.g. ``~/.DataLab_v1/installed_plugins``) and listed in the tab, where **Uninstall** removes them. After installing or uninstalling, DataLab offers to reload plugins; a newly installed plugin is enabled. A new version of a plugin already loaded is only used at the next start of DataLab.

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

Creating a layered plugin project
----------------------------------

The installed ``datalab-plugin`` command creates a small, installable project
that follows the stable plugin and feature ownership contracts. Run it without
options for an interactive setup:

.. code-block:: bash

  datalab-plugin create

The command asks for the display name and the GitHub account (user or organization) that will host the project, and offers defaults for the Python package, reverse-domain plugin ID, description, and destination. The default plugin ID is ``io.github.<account>.<name>``; ``org.datalab.`` IDs are reserved for the DataLab-Platform organization, and ``org.example.`` is used when no account is given. For scripts or reproducible setup, pass the values explicitly:

.. code-block:: bash

  datalab-plugin create datalab-camera-characterization \
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
them. From the generated directory, install and validate the project with:

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
