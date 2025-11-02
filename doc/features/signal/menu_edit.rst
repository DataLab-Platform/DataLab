.. _sig-menu-edit:

Manipulate metadata and annotations
===================================

This section describes how to manipulate metadata and annotations in DataLab.

Signal metadata contains various information about the signal or its representation,
such as view settings, Regions Of Interest (ROIs), processing chain history,
analysis results, and any other information that you may have added to the metadata
of a signal (or that comes from the signal file itself).

.. figure:: /images/shots/s_edit.png

    Screenshot of the "Edit" menu.

The "Edit" menu allows you to perform classic editing operations on the current signal
or group of signals (create/rename group, move up/down, delete signal/group of signals,
etc.).

As detailed below, it also allows you to:

- Navigate and utilize the processing chain history through actions like "Recompute"
  and "Select source objects".
- Manipulate metadata and annotations associated with the current signal, thanks to the
  "Metadata" and "Annotations" submenus which provide the following features.

Recompute
---------

The "Recompute" |recompute| action allows you to recompute the selected signal(s) using
their original processing parameters. This is useful when you want to re-execute the
processing chain that was used to create a signal, for example after modifying global
settings or dependencies.

.. |recompute| image:: ../../../datalab/data/icons/edit/recompute.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. note::

    This action is only available for signals that were created through processing
    operations and have stored processing parameters.

Select source objects
---------------------

The "Select source objects" |goto_source| action allows you to select the source
object(s) that were used to create the currently selected signal. This helps trace
back the processing history and understand which original signals were used as input
for the current result.

.. |goto_source| image:: ../../../datalab/data/icons/edit/goto_source.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. note::

    This action is only available when exactly one signal is selected and that signal
    has source object references.

Metadata
--------

.. figure:: /images/shots/s_edit_metadata.png

    Screenshot of the "Metadata" submenu.

Copy/paste metadata
^^^^^^^^^^^^^^^^^^^

As metadata contains useful information about the signal, it can be copied and pasted
from one signal to another by selecting the "Copy metadata" |metadata_copy| and
"Paste metadata" |metadata_paste| actions in the "Edit" menu.

.. |metadata_copy| image:: ../../../datalab/data/icons/edit/metadata_copy.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |metadata_paste| image:: ../../../datalab/data/icons/edit/metadata_paste.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

This feature allows you to tranfer those information from one signal to another:

- :ref:`Regions Of Interest (ROIs) <sig-menu-roi>`: that is a very efficient way to reuse
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
^^^^^^^^^^^^^^^^^^^^^^

Metadata can also be imported and exported from/to a JSON file using the "Import
metadata" |metadata_import| and "Export metadata" |metadata_export| actions in the
"Edit" menu. This is exactly the same as the copy/paste metadata feature (see above
for more details on the use cases of this feature), but it allows you to save the
metadata to a file and then import it back later.

.. |metadata_import| image:: ../../../datalab/data/icons/edit/metadata_import.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |metadata_export| image:: ../../../datalab/data/icons/edit/metadata_export.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Delete metadata
^^^^^^^^^^^^^^^

When deleting metadata using the "Delete metadata" |metadata_delete| action in the
"Edit" menu, you will be prompted to confirm the deletion of Region of Interests (ROIs)
if they are present in the metadata. After this eventual confirmation, the metadata
will be deleted, meaning that analysis results, ROIs, and any other information
associated with the signal will be lost.

.. |metadata_delete| image:: ../../../datalab/data/icons/edit/metadata_delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Add metadata
------------

The "Add metadata" |metadata_add| action allows you to add custom metadata items to
one or more selected signals. This is useful for tagging signals with experiment IDs,
sample names, processing steps, or any other custom information.

.. |metadata_add| image:: ../../../datalab/data/icons/edit/metadata_add.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. figure:: /images/shots/s_add_metadata.png

    Add metadata dialog.

When you select "Add metadata..." from the Edit menu, a dialog appears where you can:

- **Metadata key**: Enter the name of the metadata field to add
- **Value pattern**: Define a pattern for the metadata value using Python format strings
- **Conversion**: Choose how to store the value (string, float, integer, or boolean)
- **Preview**: See how the metadata will be added to each selected signal

The value pattern supports the following placeholders:

- ``{title}``: Signal title
- ``{index}``: 1-based index of the signal in the selection
- ``{count}``: Total number of selected signals
- ``{xlabel}``, ``{xunit}``, ``{ylabel}``, ``{yunit}``: Axis labels and units
- ``{metadata[key]}``: Access existing metadata values

You can also use format modifiers:

- ``{title:upper}``: Convert to uppercase
- ``{title:lower}``: Convert to lowercase
- ``{index:03d}``: Format as 3-digit number with leading zeros

**Examples:**

- Add experiment ID: key=``experiment_id``, pattern=``EXP_{index:03d}``, conversion=string
  → Creates metadata like ``experiment_id="EXP_001"``

- Add sample temperature: key=``temperature``, pattern=``{metadata[temp]}``, conversion=float
  → Copies temperature from existing metadata and converts to float

- Mark processed signals: key=``is_processed``, pattern=``true``, conversion=bool
  → Sets ``is_processed=True`` for all selected signals

Annotations
-----------

Annotations are visual elements that can be added to signals to highlight specific
features, mark regions of interest, or add explanatory notes. DataLab provides a
dedicated submenu in the "Edit" menu for managing annotations.

.. figure:: /images/shots/s_edit_annotations.png

    Screenshot of the "Annotations" submenu.

Copy/paste annotations
^^^^^^^^^^^^^^^^^^^^^^

Annotations can be copied from one signal and pasted to one or more other signals
using the "Copy annotations" |annotations_copy| and "Paste annotations" |annotations_paste|
actions. This is useful when you want to apply the same visual markers across multiple
signals.

.. |annotations_copy| image:: ../../../datalab/data/icons/edit/annotations_copy.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |annotations_paste| image:: ../../../datalab/data/icons/edit/annotations_paste.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

The "Paste annotations" action is only enabled when there are annotations in the
clipboard (i.e., after using "Copy annotations").

Edit annotations
^^^^^^^^^^^^^^^^

The "Edit annotations" |annotations_edit| action opens a dialog where you can view,
add, modify, or remove annotations from the current signal. This provides a visual
way to manage all annotations on a signal.

.. |annotations_edit| image:: ../../../datalab/data/icons/edit/annotations_edit.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Import/export annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotations can be saved to and loaded from JSON files (.dlabann extension) using
the "Import annotations" |annotations_import| and "Export annotations" |annotations_export|
actions. This allows you to:

- Save annotation sets for later reuse
- Share annotations with colleagues
- Archive annotations separately from signal data
- Apply the same annotations to different signals across sessions

.. |annotations_import| image:: ../../../datalab/data/icons/edit/annotations_import.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |annotations_export| image:: ../../../datalab/data/icons/edit/annotations_export.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

The "Export annotations" action is only available when the selected signal has
annotations.

Delete annotations
^^^^^^^^^^^^^^^^^^

The "Delete annotations" |annotations_delete| action removes all annotations from the
selected signal(s). This action is only enabled when the selected signal(s) have
annotations.

.. |annotations_delete| image:: ../../../datalab/data/icons/edit/annotations_delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. note::

    Annotations are stored separately from metadata and analysis results. Deleting
    annotations does not affect ROIs or other metadata items.

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

.. |copy_titles| image:: ../../../datalab/data/icons/edit/copy_titles.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link
