.. _provenance:

Provenance of results
=====================

.. meta::
    :description: Provenance of results in DataLab: consultation, data check, verification by recomputation, applying a processing again, link with the History panel
    :keywords: DataLab, provenance, reproducibility, processing, verification, workspace, HDF5, history

For supported processing paths, DataLab records how a result was obtained: the processing or analysis applied,
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
the workspace and never adds a History action. The view is bounded and long
values are abbreviated: a truncated view does not mean that records were removed.

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

Automatic provenance capture does not depend on the History panel's **Record**
mode. Recording a recipe and recording a scientific execution are different
operations.

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
  them. Deleting or replacing an input's numerical data can prevent replay
  even after saving and reopening. For example, saving a filtered signal after
  deleting its source saves the description of the source, not its arrays.
  Keep the necessary inputs and the recorded results before saving.
- **ROI-only changes are a limited exception.** If an input's numerical data
  are unchanged, its recorded ROI can be restored on a copy for verification,
  including when a detection added ROI. This does not reconstruct missing
  values. A missing or altered output is still not a valid comparison reference:
  restoring an input's ROI does not repair the reference.
- **Analysis values are not all archived.** A later analysis can replace a
  result under the same metadata key. The earlier artifact stays described as
  ``replaced``, without necessarily retaining its old value.
- **It does not clean up the workspace.** Trials and intermediate objects stay
  until you remove them. Their recorded steps remain in the provenance record:
  deleting an object never purges its activities or sensitive information.
- **It does not make everything verifiable.** Changes made outside recorded
  processing may be observed later without a known production. An operation
  without a contract can be recorded as opaque, but is never replayed. Old
  metadata records have unknown input states and possibly incomplete parameters
  or environments; they are not qualified retroactively or executed from their
  Python names.
- **It is not a proof of scientific validity.** *Exact* or *Within tolerance*
  says that the recorded computation is reproduced under the rule declared for
  that operation, not that the processing was the right one.
- **It does not identify the author.** Fingerprints show that the content was
  not altered; they do not authenticate who produced it, and capsules are not
  signed.

Capture scope and verification conditions
-----------------------------------------

Ordinary and grouped processing, analyses, signal/image conversions and ROI
extraction use the supported capture paths. **Processing**, **Recompute** and
replaying a supported computation from the History panel record a new execution;
verification itself works on copies and records no new execution. An accepted
preview can become a committed result, but a discarded preview, a failed
calculation or a cancellation is not a successful execution. A successful
detection finding nothing is a normal recorded result.

Importing or creating an object does not record how its physical acquisition
or earlier conversions were performed. Editing data, ROI or metadata directly
does not by itself record a scientific call. Interactive Desktop fits and
creating a signal from a markers table are currently not recorded scientific
calls, even when an editable History action exists.

Plugins, macros, scripts and remote API clients are captured when they use the
supported processors. Code that creates or modifies objects directly can bypass
them. Capture does not block processing: if it fails, a result may exist without
an activity; the failure is logged and counted, not silently repaired.

A replay needs a local operation contract qualified for the recorded version,
decodable parameters, applicable context, available input data and valid output
references. The signal and image phase spectra are identified but not qualified.
Unknown operations, versions, codecs or policies produce an explicit refusal:
the reader never downloads code to guess their meaning.

Sigima increases the version of an operation contract whenever a change of the
computation changes its result, whatever the version of DataLab. Activities
recorded with an earlier version of the contract are then refused explicitly
(*unsupported contract*) instead of being replayed with a different result, and
their processing cannot be applied again as recorded: compute the result again
to record it with the current version. The same applies to the versioned rules
of the execution context, such as the selection of image pixels by regions of
interest. See `Operation contracts and versions
<https://sigima.readthedocs.io/en/latest/user_guide/contracts.html>`_ in the
Sigima documentation.

Scientific fingerprints cover numerical data, types, coordinates and units;
ROI are checked separately. Titles, display settings and other metadata are not
automatically covered. Only declared dependencies such as global options,
X alignment or metadata read by the computation are reproduced. For example,
changing fit parameters needed by ``evaluate_fit`` can refuse replay even if
the signal values are unchanged.

See the `common rules <https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/rules.md>`_
and `capture and interoperability scope <https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/interoperability.md>`_
for the conditions and the paths covered. Exchange, privacy and revision limits
are described in :ref:`capsules`.
