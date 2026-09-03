.. _api:

API
===

The public Application Programming Interface (API) of DataLab offers a set of functions to access the DataLab features.

.. note::

    For more details about the `sigima` package, please refer to the `Sigima documentation <https://sigima.readthedocs.io/en/latest/>`_. The Sigima API is especially useful to write scripts that can be run in DataLab (see :ref:`about_macros`) or to develop plugins (see :ref:`about_plugins`).

.. note::

    A significant part of the DataLab graphical layer is provided by the `sigimax` package: generic main window, configuration system, dockable plot widgets, HDF5 workspace and browser, log viewer and scientific dialogs. Those classes are documented in the `SigimaX documentation <https://sigimax.readthedocs.io/>`_, and are the ones to look at when building a Qt scientific application on the same foundation (see :ref:`ecosystem`).

.. toctree::
   :maxdepth: 2
   :caption: Public features:

   proxy

.. toctree::
   :maxdepth: 1
   :caption: Internal features:

   api/index
   api/main
   api/panel
   api/actionhandler
   api/objectmodel
   api/objectview
   api/plothandler
   api/roieditor
   api/processor
   api/docks
   api/h5io
