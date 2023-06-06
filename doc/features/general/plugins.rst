Plugins
=======

DataLab is a modular application. It is possible to add new features to DataLab
by writing plugins. A plugin is a Python module that is loaded at startup by
DataLab. A plugin may add new features to DataLab, or modify existing features.

The plugin system currently supports the following features:

- Processing features: add new processing tasks to the DataLab processing
  system, including specific graphical user interfaces.
- Input/output features: add new file formats to the DataLab file I/O system.
- HDF5 features: add new HDF5 file formats to the DataLab HDF5 I/O system.

Public API
^^^^^^^^^^

.. automodule:: cdl.plugins
