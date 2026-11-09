.. _plugins_developing:

Developing plugins
==================

.. meta::
    :description: Develop DataLab plugins: quick start, plugin generator, plugin API, examples and publication in the plugin catalog
    :keywords: DataLab, plugin, development, datalab-plugin, generator, PluginBase, hot-reload, plugin catalog, Sigima

This page explains how to write a plugin, from a single Python file to an installable project, and how to publish it in the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_. For a step-by-step walkthrough, see the tutorial :ref:`tutorial_plugin_catalog`.

.. _plugin_quickstart:

Your first plugin in five minutes
---------------------------------

The ``datalab-plugin`` command, installed with DataLab as a Python package, creates a working plugin in a single file:

1. Install DataLab in a Python environment, for example with ``pip install datalab-platform`` (see :ref:`installation`). The standalone version of DataLab does not provide the ``datalab-plugin`` command.

2. Create the plugin, giving its name and a one-line description when asked:

   .. code-block:: bash

     datalab-plugin create

   The command writes a ``datalab_<name>.py`` file to the user plugin directory.

3. Start DataLab, or select **Plugins > Reload plugins** if it is already running. The **Plugins** menu now holds a submenu named after the plugin, with a sample processing of the selected signals.

4. Open the file in your editor and change the ``sample_processing`` function, which receives the Y values of a signal as a NumPy array and returns new ones. Select **Plugins > Reload plugins** to try your changes, without restarting DataLab.

Then, you may:

- share the file with your colleagues, who install it with **Plugins > Configure plugins... > From file** (see :ref:`plugin_install_from_file`);
- publish it in the DataLab plugin catalog (see :ref:`plugin_publishing`);
- for a larger plugin, with packaging and tests, create a layered project (see :ref:`plugin_project`).

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
------------------------------------

**Source organization**, **distribution format** and **functional capabilities** are independent concepts. A single-file plugin and a plugin organized as a Python package use the same :class:`datalab.plugins.PluginBase` API; neither form determines whether it provides processing, input/output, visualization or application features.

- **Single-file plugin:** one ``datalab_<name>.py`` module, easy to edit and share without a packaging project. This historical form is also the default output of ``datalab-plugin create``; it is not deprecated.
- **Package or project:** Python modules and, optionally, resources, tests and packaging metadata. ``datalab-plugin create --project`` offers one layered organization, not a mandatory architecture for all plugins.
- **Wheel distribution:** an installable ``.whl`` archive with standardized metadata and plugin entry points. A wheel can contain a single module or a larger package; it is a distribution format, not a functional plugin category.

Both single-file modules and wheels can be installed with **Install from file...** or distributed through online and local catalogs. The catalogs do not require the layered project generated by ``--project``. Packaging a single module as a wheel remains possible, but is optional for Desktop.

.. list-table:: Development forms and distribution
   :header-rows: 1
   :widths: 28 36 36

   * - Development form
     - Direct source loading
     - Managed distribution
   * - Single Python file
     - ``datalab_<name>.py`` in a plugin directory
     - Raw ``.py`` file, or an optional wheel
   * - Python package or project
     - Package folder, including a project's ``src`` layout
     - Wheel containing the package and its resources

Both managed formats work in local and online Desktop catalogs. The historical single-file form is not a separate functional plugin category, and its source does not need to be changed merely to list it in a catalog. DataLab-Web support is not implied by a Desktop catalog entry.

A directly loaded source file remains in its development directory. A file installed through DataLab is copied to its managed user storage. A plugin is not a macro: a plugin registers persistent application features, whereas a macro is a script run to automate a task. Supporting the historical single-file form does not mean that obsolete APIs from DataLab v0.20 are automatically supported.

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

An existing file is never overwritten. To share the plugin, send its file: other users install it with **Plugins > Configure plugins... > From file** (see :ref:`plugin_install_from_file`).

Both the raw single-file plugin and a wheel can be published in a Desktop catalog (see :ref:`plugin_module_catalog`). To develop a larger plugin with packaging and tests already configured, create a layered project with ``--project`` instead (see :ref:`plugin_project`).

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

The generated ``pyproject.toml`` requires Python 3.10 or later and the current DataLab major version (for example ``datalab-platform >= 1.4, < 2``), and declares the project URLs when the GitHub account is known. The generated ``.github/workflows/release.yml`` publishes the plugin without any PyPI account: pushing a tag such as ``v0.1.0`` checks that the tag matches the version of ``pyproject.toml`` and of the package, runs the tests, builds the wheel, publishes it in a GitHub release with its SHA-256 checksum and its build provenance attestation, and prints the entry to submit to the `DataLab plugin catalog <https://github.com/DataLab-Platform/plugins>`_, including the SHA-256 digest of the wheel. A tag such as ``v0.2.0-rc1`` (version ``0.2.0rc1``) publishes a release candidate as a GitHub pre-release, without catalog entry. Users then install the wheel with **Plugins > Configure plugins... > From file** (see :ref:`plugin_install_from_file`). Listing the plugin in the catalog is optional; once listed, it can also be installed from the **Catalog** tab (see :ref:`plugin_catalog`).

Hot-reload workflow for plugin development
------------------------------------------

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

How to develop a plugin?
------------------------

The recommended approach to developing a plugin is to derive from an existing example and adapt it to your needs. You can explore the source code in the `datalab/plugins/builtin` folder, the examples below or the plugins of the DataLab plugin catalog (see :ref:`plugin_catalog_examples`).

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

.. _plugin_catalog_examples:

Plugins of the catalog
----------------------

The plugins of the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ are open-source and show complete, tested implementations:

- `Camera & Detector Characterization <https://github.com/DataLab-Platform/datalab-camera-characterization>`_ and `Pulse & Transient Characterization <https://github.com/DataLab-Platform/datalab-pulse-characterization>`_: application plugins with headless recipes, packaged examples, simulators and welcome page tiles (see :ref:`plugins_applications`), for DataLab Desktop and DataLab-Web. Their ``core``, ``workflow`` and ``adapters`` layers are those generated by ``datalab-plugin create --project`` (see :ref:`plugin_project`).
- `Synchrotron Beamline Data <https://github.com/DataLab-Platform/datalab-synchrotron>`_: a demonstration plugin adding file formats (SPEC and XDI files) and a beamline data analysis.
- `TANGO Instruments <https://github.com/DataLab-Platform/datalab-tango>`_: a demonstration plugin acquiring the attributes of TANGO devices in an instrument window, and requiring a package that users install in DataLab's Python environment.

.. _plugin_example_processing:

Example: processing plugin
--------------------------

This example registers a processing feature with stable identity and ownership,
then creates an action which dispatches it through the processor registry:

.. literalinclude:: ../../../../plugins/examples/datalab_custom_func.py

Example: input/output plugin
----------------------------

Here is a simple example of a plugin that adds new file formats to DataLab.

.. literalinclude:: ../../../../datalab/plugins/builtin/datalab_imageformats.py

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

.. _plugin_publishing:

Publishing a plugin in the catalog
----------------------------------

Any author may list a plugin in the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_, so that DataLab users find it and install it from the **Catalog** tab (see :ref:`plugin_catalog`). No PyPI account is needed:

1. Host the source code in a public GitHub repository, under an OSI-approved open-source license.
2. Attach the plugin file to a GitHub release: the unchanged ``datalab_<name>.py`` file of a single-file plugin, or the wheel of a plugin project, which the release workflow generated by ``datalab-plugin create --project`` builds and publishes.
3. Open a pull request adding one YAML file, which describes the plugin and its release, to the `catalog repository <https://github.com/DataLab-Platform/plugins>`_. Its automatic checks download the plugin file, verify its SHA-256 digest and inspect it without running it.

Once the pull request is merged, the plugin appears on the catalog page and in the **Catalog** tab of DataLab, marked **Community**: the **Official** mark is reserved for the plugins of the DataLab team. The tutorial :ref:`tutorial_plugin_catalog` details each step, and the `contribution guide <https://github.com/DataLab-Platform/plugins/blob/main/CONTRIBUTING.md>`_ gives the complete rules.

.. _plugins_web:

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
