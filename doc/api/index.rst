.. _api:

API
===

The public Application Programming Interface (API) of DataLab offers a set of
functions to access the DataLab features. Those functions are available in various
submodules of the `cdl` package. The following table lists the available submodules
and their purpose:

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`sigima_.algorithms`
      - Algorithms for data analysis, which operates on NumPy arrays

    * - :mod:`sigima_.param`
      - Convenience module to access the DataLab sets of parameters (instances of :class:`guidata.dataset.DataSet` objects)

    * - :mod:`sigima_.obj`
      - Convenience module to access the DataLab objects (:class:`sigima_.obj.SignalObj` or :class:`sigima_.obj.ImageObj`) and related functions

    * - :mod:`sigima_.computation`
      - Computation functions, which operate on DataLab objects (:class:`sigima_.obj.SignalObj` or :class:`sigima_.obj.ImageObj`)

    * - :mod:`cdl.proxy`
      - Proxy objects to access the DataLab interface from a Python script or a remote application

.. toctree::
   :maxdepth: 2
   :caption: Public features:

   algorithms
   param
   obj
   computation
   proxy

.. toctree::
   :maxdepth: 1
   :caption: Internal features:

   gui/index
   gui/main
   gui/panel
   gui/actionhandler
   gui/objectview
   gui/plothandler
   gui/roieditor
   gui/processor
   gui/docks
   gui/h5io
