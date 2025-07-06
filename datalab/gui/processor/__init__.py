"""
Processor
=========

The :mod:`datalab.gui.processor` package provides the **processor objects**
for signals and images.

Processor objects are the bridge between the computation modules
(in :mod:`sigima.computation`) and the GUI modules (in :mod:`datalab.gui`).
They are used to call the computation functions and to update the GUI from inside
the data panel objects.

When implementing a processing feature in DataLab, the steps are usually the following:

- Add an action in the :mod:`datalab.gui.actionhandler` module to trigger the
  processing feature from the GUI (e.g. a menu item or a toolbar button).

- Implement the computation function in the :mod:`sigima.computation` module
  (that would eventually call the algorithm from the :mod:`sigima.algorithms` module).

- Implement the processor object method in this package to call the computation
  function and eventually update the GUI.

The processor objects are organized in submodules according to their purpose:

- :mod:`datalab.gui.processor.base`: Common processing features
- :mod:`datalab.gui.processor.signal`: Signal processing features
- :mod:`datalab.gui.processor.image`: Image processing features

Generic processing types
-------------------------

To support consistent processing workflows, the :class:`BaseProcessor` class defines
five generic processing methods, each corresponding to a fundamental input/output
pattern. These methods are tightly integrated with the GUI logic: the input objects
are taken from the current selection in the active panel (Signal or Image), and the
output objects are automatically appended to the same panel.

Descriptions:

- ``compute_1_to_1``: Applies an independent transformation to each selected object.
- ``compute_1_to_0``: Runs an analysis or measurement producing metadata or scalar data.
- ``compute_1_to_n``: Produces multiple output objects from a single input
  (e.g. ROI extraction).
- ``compute_n_to_1``: Aggregates multiple objects into one (e.g. sum, average);
  supports pairwise mode.
- ``compute_2_to_1``: Applies a binary operation with a second operand
  (object or constant); supports pairwise mode.

.. list-table::
    :header-rows: 1
    :align: left

    * - Method name
      - Signature
      - Multi-selection behavior

    * - ``compute_1_to_1``
      - 1 object ➝ 1 object
      - k ➝ k

    * - ``compute_1_to_0``
      - 1 object ➝ no object
      - k ➝ 0

    * - ``compute_1_to_n``
      - 1 object ➝ n objects
      - k ➝ k·n

    * - ``compute_n_to_1``
      - n objects ➝ 1 object
      - n ➝ 1<br>n ➝ n (pairwise mode)

    * - ``compute_2_to_1``
      - 1 object + 1 operand ➝ 1 object
      - k + 1 ➝ k<br>n + n ➝ n (pairwise mode)

These methods are for internal or advanced use (e.g. plugin or macro authors) and
will evolve without backward compatibility guarantees.

Future developments (such as a visual pipeline editor) may require generalizing
this model to support additional sources and destinations beyond the current
panel-based selection/output logic.
"""  # noqa: E501
