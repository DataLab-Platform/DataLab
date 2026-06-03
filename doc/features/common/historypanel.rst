.. _historypanel:

History Panel
=============

.. meta::
    :description: History Panel in DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, history, record, replay, session, scientific, data, analysis, visualization, platform

Overview
--------

The "History Panel" records the sequence of actions performed by the user on
signals and images, organized into **sessions**. Each session is a chronological
list of either:

- **UI actions** (creating a new signal, removing selected objects, saving the
  workspace to HDF5, ...), or
- **computations** (FFT, average, Gaussian fit, ...) dispatched through the
  Sigima processor.

A recorded session can be:

- **Replayed** in validation mode, without adding new signal/image outputs to
  the workspace;
- **Duplicated and applied**, to create an explicit comparison branch with new
  outputs in the signal/image panels;
- **Restored to a given selection state** without re-executing anything, to
  quickly jump back to a previous working context;
- **Saved to a standalone history file** (``.dlhist``) or **embedded in the
  workspace** when saving to HDF5, so that the full processing chain travels
  with the data.

.. figure:: ../../images/shots/history_panel.png
   :align: center
   :alt: History Panel

   The History Panel after recording a representative session: create three
   signals (Voigt, Lorentzian, Lorentzian), remove one of them, create a
   Gaussian signal, compute the average, add Gaussian noise to the result
   and run a Gaussian fit.

Toolbar
-------

The toolbar at the top of the panel exposes the following actions:

.. |record| image:: ../../../datalab/data/icons/record.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |open_history| image:: ../../../datalab/data/icons/io/fileopen_h5.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |save_history| image:: ../../../datalab/data/icons/io/filesave_h5.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |replay| image:: ../../../datalab/data/icons/replay.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |restore_selection| image:: ../../../datalab/data/icons/restore_selection.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |edit_mode| image:: ../../../datalab/data/icons/edit_mode.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |duplicate| image:: ../../../datalab/data/icons/edit/duplicate.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |step_prev| image:: ../../../datalab/data/icons/libre-gui-arrow-left.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |step_next| image:: ../../../datalab/data/icons/libre-gui-arrow-right.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |delete| image:: ../../../datalab/data/icons/edit/delete.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |generate_macro| image:: ../../../datalab/data/icons/console.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |remove_incompatible| image:: ../../../datalab/data/icons/edit/delete_all.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

- |record| **Record mode**: toggle the recording of new actions. When off, no
  new entry is added to the history (existing sessions are preserved).
- |open_history| **Open history file**: load recorded sessions from a standalone
  ``.dlhist`` file.
- |save_history| **Save history file**: save the current recorded sessions to a
  standalone ``.dlhist`` file.
- |replay| **Replay**: validate/replay the selected action (or the whole
  session if a session row is selected) without changing the current workspace
  selection beforehand and without adding new outputs to the signal/image
  panels.
- |restore_selection| **Restore selection**: only re-select the objects that
  were selected when the action was originally executed; no computation is
  re-run.
- |edit_mode| **Edit mode**: when on, replaying a computation opens the
  parameters dialog so the user can tweak the parameters before re-running.
  When replaying a *whole session*, the parameter dialogs open in a
  **read-only** mode — all fields are shown with their recorded values but
  cannot be edited.
- |duplicate| **Duplicate**: copy the selected action or session into a new
  history session. The copied parameters are independent from the original
  record.
- |generate_macro| **Generate macro**: generate a Python macro script from the
  selected actions (or all actions if nothing is selected). The generated script
  is copied to the clipboard.
- |remove_incompatible| **Remove incompatible**: remove all actions whose
  workspace state is no longer compatible with the current workspace. A
  confirmation dialog shows how many actions will be removed.
- |delete| **Delete**: remove the selected actions or sessions from the
  history.
- |step_prev| **Previous step**: select the preceding action in the current
  session (keyboard shortcut: :kbd:`Ctrl+Left`).
- |step_next| **Next step**: select the following action in the current
  session (keyboard shortcut: :kbd:`Ctrl+Right`).

.. note::

   Double-clicking on an action row in the tree is equivalent to **Replay**.

Tree view
---------

The tree view organizes recorded actions into expandable sessions:

- Each top-level row is a **session**, automatically created when recording is
  enabled and a new application context is started.
- Each child row is an **action**, with its title, date/time and a description
  summarising the parameters (for computations) or the call (for UI actions).

The selection of one or several rows drives which actions are targeted by the
toolbar buttons.

Actions that are not compatible with the current workspace state (for example
because a referenced object identifier no longer exists, or because its data
shape changed) are shown with a disabled foreground and an explanatory tooltip.
They cannot be replayed until the workspace matches the recorded state again.

Workspace state display
-----------------------

Below the action tree, a split-view widget shows the **workspace state**
captured at the time of the selected action:

- **Left table**: lists the signals that were selected, with their data shape.
- **Right table**: lists the images that were selected, with their dimensions.

This information helps the user understand the context in which each action
was originally executed and diagnose compatibility issues when replaying
sessions on a different workspace.

Session replay across workspaces
--------------------------------

A full session can be replayed on a workspace that no longer contains the
objects originally referenced by the recorded actions -- typically after
loading a saved session into a fresh workspace. In that case, the panel
**remaps the recorded object identifiers** to the newly-created ones on the
fly:

- UI actions creating new objects (e.g. *New signal*) enqueue the freshly
  created identifiers;
- subsequent computations claim the identifiers they need from that queue,
  in the same order as the original recording;
- UI actions removing objects keep the queue in sync with the live workspace
  contents, so chained creation/removal sequences replay correctly.

This makes it possible, for instance, to record a full processing chain on
one dataset, save it, then re-apply the exact same chain on a different but
structurally identical input.

Persistence
-----------

The history can be persisted in two complementary ways:

- **Embedded in the workspace**: when the workspace is saved to HDF5
  (``File > Save to HDF5 file``), the History Panel content is automatically
  saved alongside the signals and images. Reloading the workspace restores
  the recorded sessions.
- **Standalone history file** (``.dlhist``): the file embeds both the
  recorded sessions **and** all signal/image objects referenced by those
  sessions. This makes the file fully self-contained:

  - Opening a ``.dlhist`` into an **empty workspace** loads sessions and
    objects directly, restoring the workspace to its recorded state.
  - Opening a ``.dlhist`` into a **non-empty workspace** creates new
    signal/image groups for the imported objects (with remapped identifiers
    to avoid collisions) and appends new history sessions that reference
    those fresh identifiers.

.. warning::

   Replaying a session that depends on external files (e.g. opening a
   dataset from disk) will only succeed if those files are still available at
   the same locations as when the session was recorded.

Chain reconnection on deletion
-------------------------------

When a result object is deleted from the **signal or image panel** (not
from the History Panel tree), and that object was produced by a recorded
processing step, the History Panel automatically reconnects the processing
chain:

- All downstream steps that consumed the deleted object are rewired to use
  the source of the deleted step as their new input.
- For ``2_to_1`` operations (e.g. *difference*), the first source is used
  for reconnection.
- If no valid source can be determined (e.g. the source itself was already
  deleted), a warning is displayed listing the unreconnectable operations,
  but the deletion is allowed to proceed.

This behaviour mirrors removing a link from a chain: the adjacent links
reconnect to preserve the processing flow.

.. note::

   Reconnection is only triggered by deletions initiated from the
   signal/image panels. Deleting an action directly from the History Panel
   tree removes it and all subsequent actions in that session.

Auto-recompute
--------------

.. note::

   When a result object is selected in the signal/image panel and it has
   processing parameters (i.e. was produced by a 1-to-1 computation), a
   **Processing** tab appears in the Properties panel. Checking
   **Auto-recompute on edit** in that tab will re-run the computation
   automatically 300 ms after any parameter modification.
