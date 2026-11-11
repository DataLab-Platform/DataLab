.. _about_plugins:

Plugins
=======

.. meta::
    :description: Plugin system for DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, plugin, processing, input/output, HDF5, file format, data analysis, visualization, scientific, open-source, platform

DataLab supports a robust plugin architecture, allowing users to extend the application’s features without modifying its core. Plugins can introduce new processing tools, data import/export formats, or custom GUI elements — all seamlessly integrated into the platform.

A plugin is Python code loaded by DataLab: a single ``datalab_<name>.py`` file, or a package distributed as a wheel. Application plugins, for example for the characterization of cameras or of pulsed signals, add complete workflows to the **Applications** panel and to the welcome page.

.. _plugin_catalog_overview:

Find plugins in the catalog
---------------------------

The `DataLab plugin catalog <https://datalab-platform.com/plugins/>`_ lists the plugins maintained by the DataLab team (**Official**) and by their authors (**Community**). Each release is checked automatically before it is published: pure Python plugin code, open-source license, and dependencies provided by DataLab or, for DataLab Desktop, supplied as additional wheels or installed by users in DataLab's Python environment (see :ref:`plugin_dependencies`).

To install a plugin of the catalog, open **Plugins > Configure plugins...** in DataLab and select the **Catalog** tab (see :ref:`plugin_catalog`). A plugin downloaded from the catalog page may also be installed from the **From file** tab, including in the standalone version of DataLab. The catalog indicates which plugins also run in DataLab-Web (see :ref:`plugins_web`).

.. only:: html and not latex

    .. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: :octicon:`package;1em;sd-text-info`  Browse the plugin catalog
            :link: https://datalab-platform.com/plugins/

            Official and community plugins, ready to install

        .. grid-item-card:: :octicon:`rocket;1em;sd-text-info`  Create your own plugin
            :link: tutorial_plugin_catalog
            :link-type: ref

            From an idea to the plugin catalog, step by step

Your plugin could be the next one in the catalog: writing a plugin may take a single Python file, and publishing it takes a GitHub repository and one pull request, without any PyPI account. See the tutorial :ref:`tutorial_plugin_catalog`.

Plugin documentation
--------------------

.. only:: html and not latex

    .. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: :octicon:`plug;1em;sd-text-info`  Installing and managing plugins
            :link: plugins/using
            :link-type: doc

            Plugin locations, installation from the catalog or from a file, plugin configuration

        .. grid-item-card:: :octicon:`organization;1em;sd-text-info`  Deploying plugins in an organization
            :link: plugins/deploying
            :link-type: doc

            Local plugin catalogs, additional dependencies and offline deployment

        .. grid-item-card:: :octicon:`code;1em;sd-text-info`  Developing plugins
            :link: plugins/developing
            :link-type: doc

            Quick start, plugin generator, plugin API, examples and publication in the catalog

        .. grid-item-card:: :octicon:`apps;1em;sd-text-info`  Application plugins
            :link: plugins/applications
            :link-type: doc

            Headless recipes, packaged examples, tools, instruments and welcome page tiles

        .. grid-item-card:: :octicon:`book;1em;sd-text-info`  Public API
            :link: plugins/api
            :link-type: doc

            Reference of the classes and functions available to plugins

.. only:: latex and not html

    The plugin documentation is organized as follows:

    - :ref:`plugins_using`: plugin locations, installation from the catalog or from a file, plugin configuration;
    - :ref:`plugins_deploying`: local plugin catalogs, additional dependencies and offline deployment;
    - :ref:`plugins_developing`: quick start, plugin generator, plugin API, examples and publication in the catalog;
    - :ref:`plugins_applications`: headless recipes, packaged examples, tools, instruments and welcome page tiles;
    - :ref:`plugins_api`: reference of the classes and functions available to plugins.

.. toctree::
    :maxdepth: 1
    :hidden:

    plugins/using
    plugins/deploying
    plugins/developing
    plugins/applications
    plugins/api
