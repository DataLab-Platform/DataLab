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
- **computations** (FFT, average, Gaussian fit, ...) dispatched by the DataLab
  processors to Sigima.

A recorded session can be:

- **Replayed** silently or **step by step**, with an opportunity to edit
  available parameters. New outputs from computation actions are not added to
  the data panels. Recorded UI actions are invoked through their own methods
  and may reproduce their side effects, including creating, importing, or
  duplicating objects, unless an action has a specific replay guard;
- **Duplicated** as independent processing chains in new history sessions,
  with the required signal/image objects cloned as part of the operation;
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

Recording and session lifecycle
-------------------------------

Actions are recorded only while **Record mode** is enabled. Turning record
mode off preserves existing sessions but does not add new entries.

The Signals and Images panels each have their own active session. New actions
are added to the active session of the data panel they concern, so switching
between signals and images does not mix their recording contexts.

When a new object is created or a file is loaded into a populated active
session, a configurable policy determines whether DataLab asks, starts a new
session, or continues the current one. Plugin-created objects use separate
policies. An explicit plugin multi-load scope supplies one durable session
policy for the whole batch. With ordinary **Ask** behavior, repeated prompts
for synchronous additions to the same panel are debounced during the current
Qt event-loop turn.

These options are available under
``File > Settings > Processing > History sessions``. See
:ref:`history-session-settings` for the complete labels and default values.

Toolbar
-------

The toolbar at the top of the panel exposes the following actions:

.. |record| image:: ../../../datalab/data/icons/record.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |new_session| image:: ../../../datalab/data/icons/libre-gui-add.svg
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

.. |step_by_step| image:: ../../../datalab/data/icons/edit_mode.svg
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

.. |remove_incompatible| image:: ../../../datalab/data/icons/edit/delete_all.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

- |record| **Record mode**: toggle the recording of new actions. When off, no
  new entry is added to the history (existing sessions are preserved).
- |new_session| **New session**: start a new active history session for the
  current data panel.
- |open_history| **Open history file**: load recorded sessions from a standalone
  ``.dlhist`` file.
- |save_history| **Save history file**: save the current recorded sessions to a
  standalone ``.dlhist`` file.
- |step_prev| **Previous step**: select the preceding action in the current
  session (keyboard shortcut: :kbd:`Ctrl+Left`).
- |step_next| **Next step**: select the following action in the current
  session (keyboard shortcut: :kbd:`Ctrl+Right`).
- |replay| **Replay**: selecting an action replays that action directly;
  selecting a session replays the whole session. Compute actions restore their
  captured input selection when the translated object identifiers are
  compatible and resolvable; UI actions are replayed without restoring their
  recorded workspace selection. New objects returned by computation actions
  are not added to the data panels. Recorded UI actions may reproduce side
  effects, including creating, importing, or duplicating objects, unless an
  action has a specific replay guard.
- |step_by_step| **Step-by-step**: replay the selection one step at a time.
  Parameters may be reviewed and edited when the action supports it; supported
  actions and their dependent branches are then updated or recomputed in
  place.
- |duplicate| **Duplicate**: duplicate the processing chain containing each
  selected action, or the processing chains in each selected session. DataLab
  clones the required objects and creates independent history sessions.
- |remove_incompatible| **Remove incompatible**: remove all actions whose
  workspace state is no longer compatible with the current workspace. A
  confirmation dialog shows how many actions will be removed.
- |delete| **Delete**: remove the selected actions or sessions from the
  history. Removing an intermediate action splices it out and preserves its
  downstream steps as an independent chain.

.. note::

  Double-clicking a tree item invokes **Replay** for the current selection. It
  uses the same action/session and compute/UI selection semantics documented
  above.

Tree view
---------

The tree view organizes recorded actions into expandable sessions:

- Each top-level row is a **session** associated with the Signals or Images
  panel. Sessions may be started when recording is enabled, with **New
  session**, or according to the configured session policy.
- Each child row is an **action**, with its title, date/time and a description
  summarising its parameters or resolved call when available. A UI action whose
  call cannot be resolved may have an empty description.

The selection of one or several rows determines which entries are targeted by
the toolbar and context-menu commands. The context menu exposes the same
commands as the toolbar.

When an action row is selected, its result object is selected in the
corresponding data panel when available; otherwise, its existing input objects
are selected. DataLab then switches to that data panel.

While Record mode is enabled, selecting a session row makes that session active
for its data panel.

Actions that are not compatible with the current workspace state (for example
because a referenced object identifier no longer exists, or because its data
array shape changed) are shown with a disabled foreground and an explanatory
tooltip.
They cannot be replayed until the workspace matches the recorded state again.

Workspace state display
-----------------------

Below the action tree, a split-view widget shows the **workspace state**
captured at the time of the selected action:

- **Left table**: lists the signals that were selected, with their array shape.
- **Right table**: lists the images that were selected, with their dimensions.

This information helps the user understand the context in which each action
was originally executed and diagnose compatibility issues when replaying
the current selection.

Persistence
-----------

The history can be persisted in two complementary ways:

- **Embedded in the workspace**: when the workspace is saved to HDF5
  (``File > Save to HDF5 file``), the History Panel content is automatically
  saved alongside the signals and images. Reloading the workspace restores
  the recorded sessions.
- **Standalone history file** (``.dlhist``): the file embeds both the
  recorded sessions **and** all objects currently present in both the Signals
  and Images panels, whether or not an action references them. This makes the
  file fully self-contained:

  - Opening a ``.dlhist`` into a **pristine workspace** (with no data objects
    and no existing history sessions) restores the saved objects and sessions
    directly.
  - If the workspace is **already in use** (it contains any data object or
    history session), DataLab imports the objects into new signal/image groups,
    remaps their identifiers to avoid collisions, and appends imported history
    sessions that reference those fresh identifiers.

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

  Reconnection is only triggered by deletions initiated from the signal/image
  panels. Deleting an action directly from the History Panel tree behaves
  differently: the selected action is spliced out instead of truncating the
  session. If downstream steps depend on it, DataLab preserves them as an
  independent chain by cloning the required intermediate object and reconnecting
  those steps to the clone. Deleting a session removes that complete session.

Auto-recompute
--------------

.. note::

   When a result object is selected in the signal/image panel and it has
   processing parameters (i.e. was produced by a 1-to-1 computation), a
   **Processing** tab appears in the Properties panel. Checking
   **Auto-recompute on edit** in that tab will re-run the computation
   automatically 300 ms after any parameter modification.
