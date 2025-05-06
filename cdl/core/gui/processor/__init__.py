"""
Processor
=========

The :mod:`cdl.core.gui.processor` package provides the **processor objects**
for signals and images.

Processor objects are the bridge between the computation modules
(in :mod:`cdl.computation`) and the GUI modules (in :mod:`cdl.core.gui`).
They are used to call the computation functions and to update the GUI from inside
the data panel objects.

When implementing a processing feature in DataLab, the steps are usually the following:

- Add an action in the :mod:`cdl.core.gui.actionhandler` module to trigger the
  processing feature from the GUI (e.g. a menu item or a toolbar button).

- Implement the computation function in the :mod:`cdl.computation` module
  (that would eventually call the algorithm from the :mod:`cdl.algorithms` module).

- Implement the processor object method in this package to call the computation
  function and eventually update the GUI.

The processor objects are organized in submodules according to their purpose.

The following submodules are available:

- :mod:`cdl.core.gui.processor.base`: Common processing features
- :mod:`cdl.core.gui.processor.signal`: Signal processing features
- :mod:`cdl.core.gui.processor.image`: Image processing features

Common features
---------------

.. automodule:: cdl.core.gui.processor.base

Signal processing features
--------------------------

.. automodule:: cdl.core.gui.processor.signal

Image processing features
-------------------------

.. automodule:: cdl.core.gui.processor.image
"""
