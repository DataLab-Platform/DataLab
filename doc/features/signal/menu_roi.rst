.. _sig-menu-roi:

Regions Of Interest (ROI)
=========================

This section describes how to manipulate Regions Of Interest (ROIs) for signals in DataLab.

.. figure:: /images/shots/s_roi.png

    Screenshot of the "ROI" menu.

The "ROI" menu allows you to manage Regions Of Interest (ROIs) associated with the current signal.

.. seealso::

    For more information about signal metadata, see the :ref:`sig-menu-edit` section.

The Regions Of Interest (ROI) are signal areas that are defined by the user to perform specific operations, processing, or analysis on them.

ROI are taken into account almost in all computing features in DataLab:

- The "Operations" menu features are done only on the ROI if one is defined (except if the operation changes the number of points, like interpolation or resampling).

- The "Processing" menu actions are performed only on the ROI if one is defined (except if the destination signal data type is different from the source's, like in the Fourier analysis features).

- The "Analysis" menu actions are done only on the ROI if one is defined.

.. note::

    ROI are stored as metadata, and thus attached to signal.

The "ROI" menu allows you to:

- "Edit regions of interest" |roi|: open a dialog box to manage ROI associated with the selected signal (add, remove, move, resize, etc.). The ROI definition dialog is exactly the same as ROI extraction (see below): the ROI is defined by moving the position and adjusting the width of an horizontal range.

.. figure:: /images/shots/s_roi_editor.png

    A signal with an ROI.

- "Remove regions of interest" |roi_delete|: remove all defined ROI for the selected signals.

- "Extract regions of interest" |signal_roi|: extract the defined ROI from the selected signals. This will create a new signal for each ROI (or a single signal, if the "Extract all ROIs into a single signal" option is selected in the dialog), with the same metadata as the original signal, but with the data corresponding to the ROI only. The new signals will be added to the current workspace.

.. figure:: /images/shots/s_roi_dialog.png

    ROI extraction dialog: the ROI is defined by moving the position and adjusting the width of an horizontal range.

.. |roi| image:: ../../../datalab/data/icons/roi/roi.svg
    :width: 24px
    :height: 24px
    :class: dark-light

.. |roi_delete| image:: ../../../datalab/data/icons/roi/roi_delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light

.. |signal_roi| image:: ../../../datalab/data/icons/roi/signal_roi.svg
    :width: 24px
    :height: 24px
    :class: dark-light