.. _plugins_applications:

Application plugins
===================

.. meta::
    :description: Application plugins for DataLab: headless recipes, packaged examples, tools, instruments and welcome page tiles
    :keywords: DataLab, plugin, application, recipe, workflow, example, tool, instrument, acquisition, welcome page

An application plugin, declaring the ``APPLICATION`` capability, provides complete workflows, such as the characterization of a camera or of pulsed signals, or the analysis of X-ray radiographs: its methods are presented in the :ref:`applications_panel`, and its tiles on the :ref:`welcome_page`. This page describes what a plugin may declare for this purpose: headless recipes and packaged examples, which any plugin may provide, then tools, instruments and welcome page tiles, which require the ``APPLICATION`` capability. The `Camera & Detector Characterization <https://github.com/DataLab-Platform/datalab-camera-characterization>`_, `Pulse & Transient Characterization <https://github.com/DataLab-Platform/datalab-pulse-characterization>`_ and `X-ray Radiography & NDT <https://github.com/DataLab-Platform/datalab-radiography>`_ plugins of the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ are complete examples.

This page describes DataLab Desktop. DataLab-Web runs the headless recipes, examples and tools declared by a distinct plugin class, but it does not support the Desktop methods launching recipes and examples, nor welcome page tiles (see :ref:`plugins_web`).

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

DataLab validates the declarations of a plugin when loading it: recipes, recipe
launchers, examples, tools and welcome tiles, described below. A plugin with an
invalid declaration, for example a recipe whose ``plugin_version`` no longer
matches the plugin version, is not loaded: its error is shown in the internal
console and written to the log file, and the plugin is listed with this error in
the **Installed plugins** tab of **Configure plugins...** (*Plugins with errors*
filter). The other plugins are loaded normally.

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

The **Run...** buttons of the :ref:`applications_panel` use the same launcher. A plugin that needs a dedicated interface for one recipe may map its recipe ID to a plugin method name in the class-level ``RECIPE_LAUNCHERS`` mapping; the panel then calls this method, without arguments, instead of the generic launcher.

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
-----------------

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

The :ref:`applications_panel` presents each recipe as a method with a live status for the current selection and a **Run...** button; its details give the expected inputs and the examples designed for it. **Try with this example** calls :meth:`datalab.plugins.PluginBase.try_example`: it opens the example, prefills the recipe parameters with the example values, and runs the recipe once the user accepts them. Examples without ``recipe_ids`` are listed as datasets, which only open.

Plugin tools
------------

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

DataLab lists each tool in two places: in the **Tools** section of the :ref:`applications_panel`, and in the plugin's submenu of the **Plugins** menu, after the plugin's own actions. ``object_type`` gives the panel whose menu shows the tool; with ``None``, both panels show it. ``selection`` states which selection the tool needs: ``NONE`` (the default), ``EXACTLY_ONE``, ``AT_LEAST_ONE`` or ``AT_LEAST_TWO`` objects of ``object_type``. When the selection does not match, DataLab disables the menu entry and the button of the Applications panel, and the button tooltip tells what to select. ``icon`` takes a ``package:path`` resource, a ``module:filename`` image registered with guidata (e.g. ``datalab:delete.svg``) or a DataLab icon file name; by default, the tool uses the plugin icon.

The launcher is called without arguments. :meth:`datalab.plugins.PluginBase.get_tools` validates unique local IDs and the tool methods, and rejects tools declared by plugins without the ``APPLICATION`` capability. :meth:`datalab.plugins.PluginBase.assess_tool` returns the reason why a tool cannot be opened for the current selection, or ``None``.

.. _plugins_instruments:

Instruments
~~~~~~~~~~~

A tool may name an ``instrument`` method instead of a ``launcher``. This method returns a :class:`datalab.plugins.instruments.PluginInstrument`, and DataLab opens a window for it: a live view on the left, the instrument settings on the right, a **Live** button that refreshes the view continuously, a **New group** option and an **Acquire** button. The plugin writes no user interface code:

.. code-block:: python

  import guidata.dataset as gds
  import numpy as np
  from sigima.objects import create_signal

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

    def make_signal(self):
      x = np.linspace(0.0, 1.0, 500)
      y = self.settings.amplitude * np.sin(2 * np.pi * 5 * x)
      return create_signal("Source", x, y + np.random.normal(0.0, 0.05, x.size))

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

DataLab edits :attr:`~datalab.plugins.instruments.PluginInstrument.settings` in place, with the active state, groups and tabs of the guidata DataSet, then calls :meth:`~datalab.plugins.instruments.PluginInstrument.preview` after each change and, in live mode, every ``live_interval_ms`` milliseconds. An :class:`~datalab.plugins.instruments.InstrumentFrame` holds signals drawn together or a single image, a short summary shown below the view, and an optional fixed Y range (signals) or color range (image). :meth:`~datalab.plugins.instruments.PluginInstrument.acquire` returns an :class:`~datalab.plugins.instruments.InstrumentAcquisition`: DataLab adds its objects to a new group of the matching panel, titled after the acquisition, and selects them, so that a recipe can run on them at once. Both methods raise ``ValueError`` with a user-facing message when the settings cannot be used; DataLab shows the message in the window.

When the user turns off the **New group** option of the window, DataLab adds the acquired objects to the current group of the panel instead: as they are selected, the next acquisitions go to the same group, until the user selects another group. The option and the current group are taken when **Acquire** is clicked: changing them while the acquisition is running does not change its destination, and a new group is created if the group was removed meanwhile. A new group is still created when the panel has no current group, for example when it is empty. The :attr:`~datalab.plugins.instruments.PluginInstrument.new_group` class attribute sets the default state of the option (``True`` by default): an instrument whose acquisitions are usually single objects, such as a device attribute read on demand, may set it to ``False``. The choice of the user, once made, is remembered for each instrument and prevails over this default.

DataLab creates the instrument once per tool and keeps it until the plugins are reloaded, so the settings remain from one opening to the next.

DataLab Desktop calls the instrument outside the user interface thread, one call at a time, even when its window is closed and opened again while a call is running, and changes its settings only between two calls: a slow instrument, such as a device waiting for its data, does not freeze DataLab. The optional :meth:`~datalab.plugins.instruments.PluginInstrument.open` and :meth:`~datalab.plugins.instruments.PluginInstrument.close` methods are called when the window opens and closes, including when plugins are reloaded or DataLab quits, for example to connect to a device and to release it. If ``open`` raises an error, its message stays in the window and the instrument is not called: the next refresh or acquisition, for example after a settings change, calls ``open`` again first. ``close`` is called even if ``open`` failed, and must release only what is still held. Data received by the instrument in its own threads, such as device events, is kept by the instrument for its next ``preview`` or ``acquire`` call.

Instrument connectors should attach an ``experimental_source`` metadata entry to acquired objects before returning them; DataLab does not fill it automatically. Use Sigima's ``make_experimental_source`` and ``set_experimental_source`` helpers to record the device address, attribute, selection and timestamps as appropriate. This identifies the measurement origin, not a reproducible acquisition. See the `Sigima experimental data guide <https://sigima.readthedocs.io/en/latest/user_guide/overview.html#experimental-data-origin>`_ for an example and the portable metadata format.

DataLab-Web supports tools too: it lists them in the same places, opens instruments in the same window, and calls launchers in the plugin's Pyodide host.

Welcome page tiles
------------------

Every active plugin declaring the ``APPLICATION`` capability adds a tile to the **Applications** section, at the top of the :ref:`welcome_page`. By default, this tile is built from the plugin's ``PluginInfo`` name, description and icon. Clicking this tile shows the plugin page in the :ref:`applications_panel`.

The plugin icon is declared with ``PluginInfo.icon``, either as a ``package:relative/path`` resource (SVG or bitmap image shipped in the plugin wheel), as a ``module:filename`` image registered by a module with guidata (e.g. ``datalab:delete.svg``), or as the file name of a DataLab icon. It is shown on the default tile, in the **Applications** panel and in the plugin configuration dialog; plugins without icon use a generic plugin icon. To offer other entry points, for example to open an example directly, declare the class-level ``WELCOME_TILES`` tuple of :class:`datalab.plugins.tiles.WelcomeTile` values:

.. code-block:: python

  from datalab.plugins import PluginBase, PluginCapability, PluginInfo
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

Declared tiles replace the default tile. The first one is the main tile of the plugin; the others are shown next to it when the **Applications** section has room for them, and otherwise become actions of its menu, opened from its "…" button or by right-clicking it. A tile without ``launcher`` shows the plugin page in the **Applications** panel; otherwise, the named plugin method is called without arguments. A tile without ``icon`` uses the plugin icon. :meth:`datalab.plugins.PluginBase.get_welcome_tiles` validates unique local IDs and launcher methods, and rejects tiles declared by plugins without the ``APPLICATION`` capability. The welcome page refreshes its tiles when plugins are loaded, reloaded or disabled, and reports errors raised by a launcher without closing DataLab.

The user decides which applications are shown and in which order: the welcome page shows a limited number of tile rows, pinned applications first, then recently used ones, then the others by name (see :ref:`welcome_page`). A plugin therefore cannot rely on its tile being visible: its features must remain available from the **Applications** panel.

Welcome page tiles are specific to DataLab Desktop: DataLab-Web does not provide the ``datalab.plugins.tiles`` module, so the DataLab-Web class of a plugin cannot declare them.
