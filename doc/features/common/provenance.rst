.. _provenance:

Provenance of results
=====================

.. meta::
    :description: Provenance of results in DataLab: consultation, data check, verification by recomputation, applying a processing again, link with the History panel
    :keywords: DataLab, provenance, reproducibility, processing, verification, workspace, HDF5, history

DataLab records how each result was obtained: the processing or analysis applied,
its parameters, the objects it used with their exact content, and the software
versions. This record is saved with the HDF5 workspace and read back when the
workspace is reopened, by DataLab or by DataLab-Web.

This page describes how to consult that record for one object, how to check and
verify it, how to apply a recorded processing again, and how it relates to the
History panel. To exchange a workspace with the provenance of its processing,
see :ref:`capsules`.

Consulting the provenance of an object
--------------------------------------

Select an object and open the **Provenance** tab of the Properties panel. When the
workspace holds a provenance record for this object, the tab shows its
provenance:

- a summary: the recorded execution that produced the current content, whether
  the data still have to be checked, and any limitation (e.g. data altered in the
  file, or a state observed without recorded production);
- the **Production** of the object, its **Recorded analyses**, its **Earlier
  recorded states** and, for workspaces saved by older versions, the **Records
  of an old workspace** and the **Metadata history (not verified)**.

Each recorded step is shown as a card: the operation, how and when it was
recorded and by which software (DataLab Desktop or DataLab Web), its
parameters, its inputs and results with their status, and links acting on what
they are next to. Links marked with a triangle (▸) show a hidden part of the view
and become ▾ to hide it again; an earlier step is shown framed inside the card
of the step that used it:

- **Show recorded step** (under an input) shows how that input was itself
  produced, step by step, without computing anything. The state actually used
  is shown, even if the input object was modified or reprocessed since: for
  example, if ``S2`` was computed from ``S1`` filtered with sigma = 2 and ``S1``
  was then filtered again with sigma = 5, the recorded step of the input of
  ``S2`` shows sigma = 2. **Hide recorded step** hides it again;
- **Go to object** (under an input or a result) selects, in the signal or image
  panel, the object that holds it today (it may have changed since the
  computation);
- **Verify this step** recomputes the step separately and compares every result
  with the recorded one: *Exact*, *Within tolerance*, *Different* or *Not
  verified* (with the reason, e.g. an original input missing). The workspace is
  not changed;
- **History action** selects the History action whose last execution recorded
  the step, when there is one;
- **Show details** shows the context of the step (options, X alignment,
  metadata read) and the software versions.

Consulting reads the provenance record only: it never computes, never changes
the workspace and never adds a History action.

The buttons below the cards act on the object:

- **Check data** compares the object, the direct inputs and results of its
  production and the earlier steps shown with their recorded states, without
  computing. The steps shown and the details opened stay so. A deleted input or
  result is reported as missing at once, without checking (nothing has to be
  read to know it);
- **Verify computation** verifies the production of the object, like **Verify
  this step**.
States and messages
^^^^^^^^^^^^^^^^^^^

The provenance never invents what is unknown. The main messages are:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Message
      - Meaning
    * - Data to be checked
      - Nothing was compared yet: use **Check data**
    * - This input has changed since the computation
      - The input object no longer holds the state used by the computation; the
        recorded state is still described
    * - Original input missing
      - The input object was deleted (or its identifier is now held by another
        object, see :ref:`provenance_append`)
    * - The result was modified since the computation
      - The result was edited after it was computed: it is reported as altered
        rather than trusted
    * - The data of the object are not available
      - The data cannot be read to be compared (they are not checked)
    * - The state cannot be checked (no fingerprint)
      - The record does not allow a comparison
    * - Observed state, without recorded production
      - The object exists in the record, but how it was produced is unknown
        (e.g. data loaded from a file)

Applying a processing again
---------------------------

When the latest recorded state of an object was produced by a processing of
DataLab, the **Processing** tab of the Properties panel restores that call: its
operation, its parameters and every input, in order and with its role (source,
operand, or each aggregated object). This applies to processing combining
several objects (e.g. difference of two signals, average of several images) as
well as to 1-to-1 processing, and to workspaces reopened from a file, saved by
DataLab or by DataLab-Web.

- With parameters, edit them and click **Apply**;
- Without parameters, click **Reapply**.

The result is updated in place with the recorded inputs, never with the current
selection, and the new execution is recorded. When the processing cannot be
applied again, the tab says why (e.g. an input object no longer exists, the
recorded parameters cannot be used, or the operation is not available in this
version of DataLab).

When no provenance is recorded for an object, the processing metadata stored
with it are used as before.

Link with the History panel
---------------------------

When the History panel records a session, each action keeps the identifiers of
the provenance activities recorded by its last successful execution. Select an
action and use **Show execution provenance** (History panel menu or toolbar) to
display them, one tab per result (or per analysed object). If the parameters of
the action were edited without executing it again, the dialog says that the
provenance describes the last successful execution.

The History action is the recipe, the provenance records each execution:

- applying a processing again from the **Processing** tab, or an analysis from
  the **Analysis** tab, is recorded in the provenance of the result;
- in the History panel edit mode, the edited parameters also become those of
  the History action, which is linked to this new execution. For a command
  producing several results (e.g. several filters applied at once), only the
  parameters of the edited result change in the action, and replaying the
  action applies them to that result of every object it processed;
- outside the edit mode, the History action is left unchanged: replaying it
  applies its own parameters again.

These identifiers are stored in the optional ``provenance_activity_ids`` field of
History actions (History action schema version 3). They are never copied with an
action, and actions saved by earlier versions simply have none. Deleting a
History session does not change the provenance record.

.. _provenance_append:

Appending a workspace
---------------------

When a workspace file is opened without clearing the current workspace, its data
are added but **its earlier provenance is not imported**: the provenance record
of the current workspace is kept unchanged, and later processing of the added
objects is recorded as usual. An added object never inherits the provenance of
another object, even when it holds the identifier of an object that was deleted
from the current workspace: such an object is neither shown nor used as the
original input or result of a recorded step, including when verifying it.

DataLab states this limitation before anything changes:

- when DataLab asks whether to clear the current workspace, the files are chosen
  first and, if one of them holds provenance, the question says so;
- otherwise, a single warning offers to **Add without provenance** or to
  **Cancel**.

To keep the provenance of a file, open it replacing the current workspace. When
workspaces are loaded from a script with
:meth:`~datalab.control.proxy.LocalProxy.load_h5_workspace`, no question is
asked and the notice is logged.

What the provenance does not do
-------------------------------

- **It does not keep every earlier array.** Earlier states are described by
  their fingerprints; their data remain available only while an object holds
  them. When an input was modified or deleted after a computation, the step is
  still described, but it cannot be recomputed from that input: its
  verification is *Not verified*, with the reason. To keep a result verifiable,
  keep the objects it was computed from before saving.
- **It does not clean up the workspace.** Trials and intermediate objects stay
  in the workspace, and their recorded steps stay in the provenance record,
  until you remove them; deleting an object never rewrites the record.
- **It does not make everything verifiable.** Changes made outside recorded
  processing (data edited or imported, operations unknown to the computation
  library, such as some plugins) appear as observed states or opaque
  operations: they are described, never replayed.
- **It is not a proof of scientific validity.** *Exact* or *Within tolerance*
  says that the recorded computation is reproduced under the rule declared for
  that operation, not that the processing was the right one.
- **It does not identify the author.** Fingerprints show that the content was
  not altered; they do not authenticate who produced it, and capsules are not
  signed.
