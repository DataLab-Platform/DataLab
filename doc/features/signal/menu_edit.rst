.. _sig-menu-edit:

Manipulate metadata
===================

This section describes how to manipulate metadata in DataLab.

.. figure:: /images/shots/s_edit.png

    Screenshot of the "Edit" menu.

The "Edit" menu allows you to perform classic editing operations on the current signal
or group of signals (create/rename group, move up/down, delete signal/group of signals,
etc.).

It also allows you to manipulate metadata associated with the current signal.

Copy/paste metadata
-------------------

As metadata contains useful information about the signal, it can be copied and pasted
from one signal to another by selecting the "Copy metadata" |metadata_copy| and
"Paste metadata" |metadata_paste| actions in the "Edit" menu.

.. |metadata_copy| image:: ../../../cdl/data/icons/edit/metadata_copy.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |metadata_paste| image:: ../../../cdl/data/icons/edit/metadata_paste.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

This feature allows you to tranfer those information from one signal to another:

- :ref:`Regions Of Interest (ROIs) <sig-roi>`: that is a very efficient way to reuse
  the same ROI on different signals and easily compare the results of the analysis
  on those signals
- Analyze results, such as peak positions or FHWM intervals (the relevance of
  transferring such information depends on the context and is up to the user to decide)
- Any other information that you may have added to the metadata of a signal

.. note::

    Copying metadata from a signal to another will overwrite the metadata of the
    destination signal (for the metadata keys that are common to both signals)
    or simply add the metadata keys that are not present in the destination signal.

Import/export metadata
----------------------

Metadata can also be imported and exported from/to a JSON file using the "Import
metadata" |metadata_import| and "Export metadata" |metadata_export| actions in the
"Edit" menu. This is exactly the same as the copy/paste metadata feature (see above
for more details on the use cases of this feature), but it allows you to save the
metadata to a file and then import it back later.

.. |metadata_import| image:: ../../../cdl/data/icons/edit/metadata_import.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |metadata_export| image:: ../../../cdl/data/icons/edit/metadata_export.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Delete metadata
---------------

When deleting metadata using the "Delete metadata" |metadata_delete| action in the
"Edit" menu, you will be prompted to confirm the deletion of Region of Interests (ROIs)
if they are present in the metadata. After this eventual confirmation, the metadata
will be deleted, meaning that analysis results, ROIs, and any other information
associated with the signal will be lost.

.. |metadata_delete| image:: ../../../cdl/data/icons/edit/metadata_delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Signal titles
-------------

Signal titles may be considered as metadata from a user point of view, even if they
are not stored in the metadata of the signal (but in an attribute of the signal object).

The "Edit" menu allows you to:

- "Add object title to plot": this action will add a label on top of the signal
  with its title.

- "Copy titles to clipboard" |copy_titles|: this action will copy the titles of the
  selected signals to the clipboard, which might be useful to paste them in a text
  editor or in a spreadsheet.

  Example of the content of the clipboard:

  .. code-block:: text

    g001:
        s001: lorentz(a=1,sigma=1,mu=0,ymin=0)
        s002: derivative(s001)
        s003: wiener(s002)
    g002: derivative(g001)
        s004: derivative(s001)
        s005: derivative(s002)
        s006: derivative(s003)
    g003: fft(g002)
        s007: fft(s004)
        s008: fft(s005)
        s009: fft(s006)

.. |copy_titles| image:: ../../../cdl/data/icons/edit/copy_titles.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. _sig-roi:

Regions Of Interest (ROI)
-------------------------

The Regions Of Interest (ROI) are signal areas that are defined by the user to
perform specific operations, processing, or analysis on them.

ROI are taken into account almost in all computing features in DataLab:

- The "Operations" menu features are done only on the ROI if one is defined (except
  if the operation changes the number of points, like interpolation or resampling).

- The "Processing" menu actions are performed only on the ROI if one is defined (except
  if the destination signal data type is different from the source's, like in the
  Fourier analysis features).

- The "Analysis" menu actions are done only on the ROI if one is defined.

.. note::

    ROI are stored as metadata, and thus attached to signal.

The "Edit" menu allows you to:

- "Edit regions of interest" |roi|: open a dialog box to manage ROI associated with
  the selected signal (add, remove, move, resize, etc.). The ROI definition dialog is
  exactly the same as ROI extraction (see below): the ROI is defined by moving the
  position and adjusting the width of an horizontal range.

.. figure:: /images/shots/s_roi_signal.png

    A signal with an ROI.

- "Remove regions of interest" |roi_delete|: remove all defined ROI for the selected
  signals.

.. |roi| image:: ../../../cdl/data/icons/edit/roi.svg
    :width: 24px
    :height: 24px
    :class: dark-light

.. |roi_delete| image:: ../../../cdl/data/icons/edit/roi_delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light
