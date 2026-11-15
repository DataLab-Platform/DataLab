.. _plugins_deploying:

Deploying plugins in an organization
====================================

.. meta::
    :description: Distribute DataLab plugins within an organization with local plugin catalogs, including on computers without Internet access
    :keywords: DataLab, plugin, local plugin catalog, offline installation, network share, dependencies, deployment

An organization may distribute its own plugins, or the versions of public plugins it has approved, from a folder shared on the network or copied to computers without Internet access. This page describes local plugin catalogs, plugins requiring additional packages and offline deployment. Users install these plugins from the **Catalog** tab, like those of the DataLab plugin catalog (see :ref:`plugin_catalog`).

.. _plugin_local_catalogs:

Local plugin catalogs
---------------------

A local plugin catalog is a folder, typically shared on the network (for example ``\\fileserver\datalab\plugins``), from which an organization distributes its own plugins, or the versions of public plugins it has approved, without publishing them. Local catalogs complement the DataLab plugin catalog and take precedence over it: when several catalogs list the same plugin (same distribution name or plugin ID), only the first one is shown, so that users install the version chosen by the organization. If the DataLab plugin catalog withdraws a plugin hidden this way, the tab reports it.

Local catalogs are declared:

- in the **Advanced** tab (see :ref:`plugin_search_paths`): these folders are saved in the DataLab configuration (``plugins_catalog_path_list`` option);
- through the ``DATALAB_PLUGINS_CATALOG`` environment variable, for example to configure all the computers of an organization: folders are separated by the OS path separator (``;`` on Windows, ``:`` on Linux/macOS) and take precedence over the folders of the configuration. Unlike ``DATALAB_PLUGINS``, folders unavailable when DataLab starts are not ignored: they are reported in the **Catalog** tab, and used as soon as they become available and the catalogs are refreshed.

.. code-block:: bat

   :: Windows
   set DATALAB_PLUGINS_CATALOG=\\fileserver\datalab\plugins;D:\approved_plugins

A local catalog folder holds either:

- **plugin files without an index**: DataLab scans pure-Python wheels declaring a ``datalab.plugins`` entry point and source files named ``datalab_<name>.py`` at the folder root, without scanning subfolders. Wheels are grouped by distribution and version; each module is one installable file, even if it declares several plugin classes. Module metadata is read statically where possible; unknown names use the module name, and unknown versions remain empty and are tracked by digest. Invalid artifacts are reported. Copying a changed module or a new wheel version is enough to offer it;
- **or a** ``catalog.json`` **file**, in the format of the DataLab plugin catalog, whose release URLs are relative to the folder: only the plugins it lists are shown. It may be a copy of the DataLab plugin catalog with its wheels, or a catalog built from the wheels of the folder with ``datalab-plugin catalog build``, then edited to give a display name to the plugins, deprecate or revoke them (``status`` and ``status_reason`` fields), or withdraw a release (``yanked`` field).

Whatever the catalog, DataLab checks the size and SHA-256 digest of the artifact before installing it, inspects it without running it and asks for confirmation; the dialog shows the catalog folder as the source. A module is checked for a valid importable file name, Python encoding and syntax, with a 4 MiB size limit. Inspection does not import its code or certify its behavior. The organization managing a shared catalog is responsible for its content and should restrict write access.

The ``datalab-plugin`` command maintains the ``catalog.json`` file of a folder:

.. code-block:: bat

   :: Write catalog.json, listing the plugin files of the folder
   datalab-plugin catalog build \\fileserver\datalab\plugins

   :: Check catalog.json against the plugin files of the folder
   datalab-plugin catalog check \\fileserver\datalab\plugins

``build`` checks wheels and source modules without executing them and does not write the index if an artifact is refused. Wheels retain their pure-Python, entry-point and dependency checks; modules follow the rules below. Curated metadata, withdrawal reasons and revoked entries are preserved. The JSON file is replaced atomically. ``check`` reports missing, changed, refused or unlisted plugin files and dependency artifacts. Run ``build`` again after changing the root files. Both commands exit with a non-zero status when they find a problem.

.. _plugin_module_catalog:

Single-file modules in a catalog
--------------------------------

For a simple offline catalog, put the unchanged ``datalab_<name>.py`` file at its root and add that folder under **Local plugin catalogs**. No ``pyproject.toml``, wheel or Internet access is needed. To generate an index that an administrator can curate, run ``datalab-plugin catalog build <folder>``, then ``check``. Mixed folders may contain both modules and wheels:

.. code-block:: text

   approved_plugins\
       datalab_spectroscopy.py
       datalab_other_plugin-1.0-py3-none-any.whl
       catalog.json
       dependencies\
           additional_wheel_dependency-1.0-py3-none-any.whl

The generated JSON uses ``kind: "module"`` and the real import name in ``module``; it does not give the script a fictitious Python distribution. Its release contains ``filename``, a relative ``url``, ``size``, ``sha256``, ``targets: ["desktop"]`` and ``version`` (possibly an empty string). ``kind`` is omitted or ``"wheel"`` for existing wheel entries. Add an optional ``requires_python`` constraint to a module release when its compatibility is known.

An administrator may edit the display name, summary, license and other descriptive fields in the JSON, and supply a catalog identity and version when they are not statically available in the source. ``build`` preserves these fields. A known source identity or version must agree with the catalog. For dynamic metadata or several plugin classes, the catalog describes the complete module without executing it. Its catalog identity does not rewrite the runtime ``PluginInfo`` or require editing a historical file.

If a module has no usable version, leave ``version`` empty: DataLab compares SHA-256 fingerprints instead. The confirmation and installed-files list distinguish a known version from a revision. A changed unversioned file, or different contents with the same declared version, requires explicit **Replace...** consent; it is not labelled as a newer version. Existing modules installed manually through DataLab are recognized too. Files in development directories are not overwritten; resolve a name conflict or remove the unmanaged copy before using the managed installation.

Single-file modules use packages already available to DataLab. Their ``import`` statements are not interpreted as package requirements, no dependencies are downloaded, and a ``dependencies`` subfolder is not installed for them. ``prepare`` includes modules in the catalog but performs dependency downloads only for wheels. Imports that cannot be checked statically may still fail when the plugin is loaded, and are then shown in the **Installed plugins** tab.

For the public catalog, publish the unchanged file as a named GitHub release asset and provide its version, license, description and SHA-256 in the YAML entry. The source is not required to expose those values statically. An entry has this structure (replace the example repository, file and fingerprint with the actual release information):

.. code-block:: yaml

   id: io.github.alice.spectroscopy
   name: Spectroscopy tools
   kind: module
   summary: A single-file Desktop plugin
   license: BSD-3-Clause
   repository: https://github.com/alice/spectroscopy
   capabilities: [processing]
   releases:
     - version: "1.0.0"
       filename: datalab_spectroscopy.py
       sha256: <SHA-256 of the released file>
       requires_python: ">=3.10"

The public catalog verifies and mirrors the script under ``modules/<sha256>/...``. To use that published catalog offline, copy its JSON and complete relative tree, including both ``modules`` and ``wheels`` when present. Do not rebuild an unchanged mirror whose artifacts are in these subdirectories: the local authoring commands scan the root. Source URLs, revision checks, withdrawals and local-catalog priority follow the same rules for both formats. Module entries target Desktop only, not DataLab-Web or PyPI.

.. _plugin_dependencies:

Additional dependencies and offline installation
------------------------------------------------

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

The **Catalog** tab copies the verified dependency artifacts of the chosen release. For **Install from file...**, place the ``dependencies`` folder next to the plugin wheel. DataLab resolves the installation using only these local files, with no package-index fallback and no compilation. Before confirmation it displays the additional packages, versions, sizes, source folder and SHA-256 digests.

Packages already provided by the Python environment or by DataLab are never replaced, upgraded or downgraded. Additional packages are shared by plugins in the same Python runtime. An incompatible installed version, missing transitive wheel or unsupported native wheel causes an explicit refusal; no partial installation is activated. Catalog metadata checks do not replace this resolution against the actual host.

A catalog release may instead list packages that users install themselves in DataLab's Python environment (``environment_requirements``), for example a large native library installed with ``pip`` or ``conda``: DataLab never downloads them. The **Catalog** tab shows these requirements and offers to install the plugin only once the Python environment satisfies them. The standalone application has no such environment: these plugins are shown as not available there, and can only be installed from a file with the wheels of the missing packages in a ``dependencies`` folder. When a wheel requires packages that are missing and no ``dependencies`` folder is available, DataLab names the missing packages and installs nothing.

An installation adding dependencies is activated after restarting DataLab. A newly installed plugin is then enabled, including when an explicit enablement list is configured. Uninstalling the plugin keeps the shared dependencies. Different Python/ABI/platform runtimes use separate dependency directories; dependencies incompatible with a changed host are reported rather than repaired over the network.

There is no environment per plugin: plugins requiring contradictory versions cannot coexist. Wheels requiring external system libraries or drivers still need those prerequisites installed separately. Standalone dependencies are intended for library imports, not for supplying an external Python interpreter or promising that their command-line scripts can be run independently. The Web installer remains unchanged.

On Windows, native library paths must also fit the DLL loader's path limit. DataLab uses short generation directory names and refuses an installation whose DLL paths would still be too long, before publishing it; use a shorter configuration directory in that case.

.. _plugin_offline_deployment:

Offline deployment, step by step
--------------------------------

This procedure is self-contained: the disconnected computer does not need a package index, a Python development environment, or access to the public plugin website. A local network share, USB drive or ordinary local folder works. Preparing missing wheel dependencies requires a connected reference computer; a catalog containing only source modules can be assembled entirely offline (see :ref:`plugin_module_catalog`).

.. important::

   A plugin catalog is a **source of installation files**, not a plugin search directory. Add it under **Local plugin catalogs**, not **Additional plugin directories**. ``DATALAB_PLUGINS_CATALOG`` supplies catalogs; ``DATALAB_PLUGINS`` supplies code loaded directly. Installing copies the approved files into the user's DataLab configuration directory, so the source folder or network share is not needed to run the installed plugin afterwards.

1. Prepare a complete catalog on a connected computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The provider needs DataLab installed in a Python environment, with the ``datalab-plugin`` command available. The standalone application does not supply this authoring command; its users only need the prepared files. Use a reference environment matching the target's DataLab version, Python version, operating system, architecture and host package versions. **? > Installation and configuration...** identifies the target installation. A Windows 64-bit CPython 3.11 wheel is not interchangeable with a Linux, 32-bit or CPython 3.12 wheel.

Create a folder, place the plugin's ``.whl`` or ``datalab_*.py`` file at its root, then run:

.. code-block:: powershell

   datalab-plugin catalog prepare D:\approved_plugins
   datalab-plugin catalog check D:\approved_plugins

For wheels, ``prepare`` resolves missing requirements against the reference host, downloads their wheels and transitive dependencies into ``dependencies``, then writes ``catalog.json``. It does not install them in the reference host or compile source distributions. Modules are included without dependency downloads. Use ``build`` for a purely file-based catalog; if preparation reports a missing wheel or a conflict, fix the catalog before transferring it.

Keep all files produced, including the dependency wheels. Do not copy just ``catalog.json`` or just the plugin wheel. If serving several target environments, prepare their wheels on matching reference environments, combine the needed files, then run ``datalab-plugin catalog build D:\approved_plugins`` followed by ``check``. A pure-Python wheel may also restrict the supported Python versions in its metadata.

``check`` checks the catalog files, their integrity and metadata. It is not a substitute for an installation test on the actual target: complete dependency resolution happens at installation time. Before wider deployment, test the complete folder on an equivalent installation with Internet access unavailable.

2. Transfer the folder, preserving its layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy the complete folder to a USB drive, local disk or network share accessible from the target computer. Preserve relative paths and file names. A public mirror may store artifacts under ``wheels\<digest>\...`` and ``modules\<digest>\...``: preserve those directories too, because ``catalog.json`` references them. The JSON file alone is not a mirror, and DataLab does not automatically export an online catalog for offline use.

For a strictly offline JSON catalog, the installation ``url`` of every release and dependency artifact must resolve to an existing local file. An absolute ``https://`` URL is still downloaded from its server even if the catalog itself is local; placing a wheel beside that JSON does not override its URL. Informational links such as ``repository``, ``documentation`` and ``source_url`` do not replace installation files. Catalogs generated by ``build`` use relative file URLs; published mirrors must retain the matching relative directory tree.

The user needs read access to this source, and write access to their own configuration and temporary directories. Administrator privileges are not required to install a plugin this way. Limit write access to the shared catalog to its maintainers; a SHA-256 check detects changed files but does not certify that a plugin or dependency is safe.

3. Configure the disconnected computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open **Plugins > Configure plugins... > Advanced**. Under **Local plugin catalogs**, click **Add**, select the catalog root (the folder containing ``catalog.json`` or the plugin wheels), and apply the configuration. Do not select the ``dependencies`` subfolder. Open the **Catalog** tab and click **Refresh** if needed.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the **Catalog** tab, find the plugin and click **Install**. DataLab chooses the newest release offered for Desktop and the running Python version, checks the plugin file and listed compatible dependency artifacts against their sizes and SHA-256 digests, then plans any additional installation using the local files only. Selection of a release does not guarantee that all its dependencies can be installed: this is checked against the actual host before confirmation.

Read the confirmation: it gives the plugin identity, classes, requirements and fingerprint, and lists the additional packages, their versions, sizes, fingerprints and source folder when needed. Cancel leaves the installation unchanged. If approved, DataLab installs into its private user storage; it does not modify the environment's existing package versions or load code from the shared catalog at every startup.

Restart DataLab if dependencies were added or if replacing an already loaded wheel. Otherwise, DataLab may offer a plugin reload. After restarting, check the **Installed plugins** tab for the plugin's state and any import error, then try one of its actions. The source catalog may be disconnected after successful installation. For a one-off file installation, **From file > Install from file...** uses the same checks; keep ``dependencies`` next to the plugin wheel.

5. Maintain and troubleshoot the offline catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``catalog.json`` exists, it is authoritative: DataLab lists and copies only its declared releases and dependency artifacts. For a catalog managed from files at its root, run ``build`` and ``check`` again after adding or changing a module, plugin wheel or dependency wheel, then redistribute the complete folder. Copying a new file beside an existing JSON does not update that index. ``prepare`` may require Internet for wheel dependencies; ``build`` and ``check`` never fetch missing packages.

Do not run ``build`` on an unchanged mirror whose plugins live only under ``wheels\<digest>\...`` or ``modules\<digest>\...``: it scans the catalog root, not the recursive tree, and would remove unrepresented entries. Update a mirror by transferring a complete, consistent published catalog and its files. To manage it with the local authoring commands instead, first arrange plugin wheels and modules at the root and additional packages in ``dependencies``.

Users refresh the **Catalog** tab to discover updates, then explicitly click **Update to...**. Updates are not installed automatically. Plugins installed with ``pip`` in the Python environment are identified but not updated by this dialog. Shared additional dependencies are retained when a plugin is uninstalled, and their existing versions remain protected on subsequent installations.

If an installation fails, read the error in the plugin dialog before changing files:

- **Catalog not found or inaccessible:** check the selected root, drive letter, USB drive, network connection and read permissions. A missing share is not an Internet problem. Refresh once it is available.
- **Size or SHA-256 mismatch:** the copied files differ from the catalog. Ask the provider for a complete, consistent copy or regenerate the catalog from the intended wheels; do not bypass the check.
- **No matching wheel or missing requirement:** the wheelhouse may lack a transitive dependency or the wheel for this Python/ABI/platform. Ask the provider to run ``prepare`` on a matching reference environment. The installer never falls back to PyPI.
- **Incompatible installed version:** the host or another plugin already provides a conflicting package version. Ask for a compatible release or reference installation; do not force an upgrade or downgrade to bypass the refusal.
- **Windows DLL path too long:** use a shorter configuration path. **DLL import or missing system library:** check the native package's external prerequisites with the provider; DataLab installs neither drivers nor system packages.
- **Installed but not active:** restart when requested, check the global and individual plugin enablement settings, and consult the import error displayed in the **Installed plugins** tab. Reloading alone does not replace an imported native library.

For local help, use **? > PDF documentation**, then search for *Offline deployment, step by step* or ``DATALAB_PLUGINS_CATALOG``. This PDF is a local file and does not need Internet access. **Online documentation**, the welcome page's documentation link, and external source-code/documentation links open websites instead. The PDF menu action appears only when a local PDF is present; DataLab tries the system language first, then English. If it is absent or unreadable, ask the administrator for the PDF shipped with the matching DataLab version rather than relying on an online link.
