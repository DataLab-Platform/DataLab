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

    * - :mod:`cdl.algorithms`
      - Algorithms for data analysis, which operates on NumPy arrays

    * - :mod:`cdl.param`
      - Convenience module to access the DataLab sets of parameters (instances of :class:`guidata.dataset.DataSet` objects)

    * - :mod:`cdl.obj`
      - Convenience module to access the DataLab objects (:class:`cdl.obj.SignalObj` or :class:`cdl.obj.ImageObj`) and related functions

    * - :mod:`cdl.core.computation`
      - Computation functions, which operate on DataLab objects (:class:`cdl.obj.SignalObj` or :class:`cdl.obj.ImageObj`)

    * - :mod:`cdl.proxy`
      - Proxy objects to access the DataLab interface from a Python script or a remote application

.. toctree::
   :maxdepth: 2
   :caption: Public features:

   algorithms
   param
   obj
   core.computation
   proxy

.. toctree::
   :maxdepth: 1
   :caption: Internal features:

   core.gui/index
   core.gui/main
   core.gui/panel
   core.gui/actionhandler
   core.gui/objectview
   core.gui/plothandler
   core.gui/roieditor
   core.gui/processor
   core.gui/docks
   core.gui/h5io
