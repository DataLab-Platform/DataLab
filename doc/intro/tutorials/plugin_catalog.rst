.. _tutorial_plugin_catalog:

:octicon:`book;1em;sd-text-info` From an idea to the plugin catalog
===================================================================

.. meta::
    :description: Tutorial showing how to create a DataLab plugin in a single Python file and publish it in the DataLab plugin catalog
    :keywords: DataLab, plugin, tutorial, datalab-plugin, plugin catalog, GitHub release, open-source, contribution

This tutorial shows how to turn an idea into a plugin available to all DataLab users: we create a plugin in a single Python file, try it in DataLab, then publish it in the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_, from which users install it in a few clicks. No packaging project and no PyPI account are needed.

As an example, we write a "Spectrum Tools" plugin, whose processing sets the negative values of a signal to zero. Replace it with your own idea: a processing specific to your field, a file format, an analysis...

What you need
-------------

- DataLab installed as a Python package, for example with ``pip install datalab-platform`` (see :ref:`installation`): the ``datalab-plugin`` command used below is not provided by the standalone version of DataLab;
- a `GitHub <https://github.com/>`_ account, to publish the source code of the plugin and to propose it to the catalog.

Step 1: create the plugin
-------------------------

Create the plugin with the ``datalab-plugin`` command. The ``--github-account`` option derives the plugin ID from your GitHub account, here ``alice``, as expected by the catalog:

.. code-block:: bash

    datalab-plugin create --name "Spectrum Tools" --description "Spectral processing tools for DataLab" --github-account alice

The command writes the ``datalab_spectrum_tools.py`` file to the user plugin directory (see :ref:`plugin_locations`) and prints its location. The beginning of the file declares the identity of the plugin:

.. code-block:: python

    PLUGIN_DESCRIPTION = "Spectral processing tools for DataLab"
    PLUGIN_ID = "io.github.alice.spectrum-tools"
    PLUGIN_NAME = "Spectrum Tools"

Its version, ``0.1.0``, is given by the ``PLUGIN_INFO`` attribute of the plugin class. Keep the plugin ID unchanged once the plugin is published: DataLab and the catalog use it to identify the plugin.

Step 2: try and develop the plugin
----------------------------------

Start DataLab, or select **Plugins > Reload plugins** if it is already running. In the signal panel, the **Plugins** menu now holds a **Spectrum Tools** submenu, with a **Sample processing** action. Create a signal with **Create > Sine**, keep it selected, and run this action: it adds a new signal, whose negative values are set to zero.

Open the file in your editor. The processing is the ``sample_processing`` function, which receives the Y values of the selected signal as a NumPy array and returns new ones:

.. code-block:: python

    def sample_processing(y: np.ndarray) -> np.ndarray:
        """Return new Y values computed from those of the selected signal.

        Edit this function to implement your own processing.
        """
        return np.maximum(y, 0.0)

Implement your own processing, give the action a meaningful title in the ``register_computations`` method, then select **Plugins > Reload plugins** to try your changes, without restarting DataLab. To go further (processing parameters, images, file formats...), see :ref:`plugins_developing`, and take the plugins of the catalog as examples (see :ref:`plugin_catalog_examples`).

Step 3: share the file with your colleagues
-------------------------------------------

The plugin can already be used by others: send them the file. They install it with **Plugins > Configure plugins... > From file** (see :ref:`plugin_install_from_file`), including in the standalone version of DataLab.

Step 4: publish the source code on GitHub
-----------------------------------------

Create a public GitHub repository, for example ``datalab-spectrum-tools``, holding:

- the ``datalab_spectrum_tools.py`` file;
- a ``LICENSE`` file, with an OSI-approved open-source license, such as BSD-3-Clause or MIT (the `contribution guide <https://github.com/DataLab-Platform/plugins/blob/main/CONTRIBUTING.md>`_ of the catalog lists the accepted licenses);
- a ``README.md`` file, describing what the plugin does and how to use it.

Then create a release of the repository, tagged ``v0.1.0`` after the version of the plugin, and attach the unchanged ``datalab_spectrum_tools.py`` file to it. With the `GitHub CLI <https://cli.github.com/>`_, from the folder of the repository:

.. code-block:: bash

    gh release create v0.1.0 datalab_spectrum_tools.py --title "Spectrum Tools 0.1.0" --notes "First release"

Keep the same file name for all versions: do not add the version number to it.

Step 5: propose the plugin to the catalog
-----------------------------------------

Fork the `catalog repository <https://github.com/DataLab-Platform/plugins>`_ and add a ``plugins/io.github.alice.spectrum-tools.yaml`` file, named after the plugin ID:

.. code-block:: yaml

    id: io.github.alice.spectrum-tools
    name: Spectrum Tools
    kind: module
    summary: Spectral processing tools for DataLab
    license: BSD-3-Clause
    repository: https://github.com/alice/datalab-spectrum-tools
    capabilities: [processing]
    keywords: [spectroscopy, spectrum]
    releases:
      - version: "0.1.0"
        filename: datalab_spectrum_tools.py
        sha256: <SHA-256 digest of the released file>
        requires_python: ">=3.10"

``kind: module`` means that the plugin is a single Python file: the catalog offers such plugins to DataLab Desktop. The ID and version must agree with those of the file; the catalog shows the name given in the YAML file. The ``sha256`` field gives the digest of the file attached to the release; you may compute it with ``sha256sum datalab_spectrum_tools.py`` on Linux and macOS, or with ``Get-FileHash datalab_spectrum_tools.py`` in Windows PowerShell (in lowercase). If you leave it out, the first run of the automatic checks fails and prints the line to add.

Then open a pull request. Its automatic checks download the file from the release, verify its digest and inspect it without running it: file name, Python syntax, plugin identity and version, license. The maintainers of the catalog then check that the plugin matches its description; this review is not a security audit.

Step 6: your plugin is in the catalog
-------------------------------------

Once the pull request is merged, the catalog is published again: the plugin appears on the `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ page, marked **Community**, and in the **Catalog** tab of the **Plugins > Configure plugins...** dialog, from which DataLab users install it with one click (see :ref:`plugin_catalog`).

To publish a new version, increase the version in the file, create a new release (for example ``v0.2.0``) holding the file under the same name, then add this version, with its digest, at the end of the ``releases`` list of the YAML file, in a new pull request. The **Catalog** tab then offers users to update the plugin.

.. admonition:: Larger plugins
    :class: tip

    For a plugin made of several modules, with resources, tests or dependencies, create a layered project with ``datalab-plugin create --project`` (see :ref:`plugin_project`). Its release workflow builds the wheel of the plugin when you push a version tag, attaches it to a GitHub release with its SHA-256 digest, and prints the entry to submit to the catalog. A wheel may also declare a distinct DataLab-Web plugin class in the ``datalab.web_plugins`` entry-point group, so that the plugin is offered to the users of both editions (see :ref:`plugins_web`).
