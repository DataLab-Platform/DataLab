.. _capsules:

Workspace capsules
==================

.. meta::
    :description: Workspace capsules in DataLab: export a workspace with the provenance of its processing, inspect a capsule, replay its activities, compare revisions, write reviews and exchange them through a registry
    :keywords: DataLab, capsule, provenance, reproducibility, revision, review, registry, RO-Crate, DataLab-Web

A **capsule** (``.dlcapsule`` file) holds a workspace and a description of the
provenance of its processing (see :ref:`provenance`), so that someone else can
check it and replay it, with DataLab or with DataLab-Web. Capsules are created,
read and checked by `DataLab-Capsule
<https://github.com/DataLab-Platform/DataLab-Capsule>`_: opening or inspecting a
capsule never executes anything it contains.

Capsules are handled from **File > Workspace capsule...**, a dialog with the
same tabs as in DataLab-Web. Three results are always shown apart:

- the **integrity** of a capsule, computed from its files (nothing is
  recomputed);
- the **supplied documents** (verification reports and reviews): they are
  checked against the capsule, but their verdicts are what their authors claim;
- the **local replay** of an activity of the open workspace, recomputed by
  DataLab and compared with its stored result.

Export
------

Exports the current workspace as a capsule:

- **Name** and **Description** of the capsule;
- **Revision of ...**: shown when the workspace was opened from a capsule, also
  after it was saved as a workspace file and opened again. When
  checked, the new capsule names that capsule as its parent revision. The
  attachments of the parent are not copied: the dialog lists them, so that they
  can be selected again if they still apply;
- **Reference the workspace instead of including it**: the capsule then holds
  the address (absolute IRI) where the workspace will be published, with its
  size and SHA-256, and the workspace is saved as a separate HDF5 file, to be
  published at that address;
- **Attachments**: light files (protocol, notes, small tables, 16 MiB at most
  each) declared in the capsule with their size and fingerprint. They are never
  opened or executed.

**Export capsule...** asks where to save the capsule (and, for a referenced
workspace, the workspace file).

.. warning::

    Review the data, metadata, titles, trials, timestamps and attachments before
    sharing. Nothing is anonymized or redacted automatically. Deleting an object
    does not guarantee that its traces disappear from the provenance record.

The export carries the current workspace, its record and the chosen attachments,
not every external resource used upstream. It does not automatically select the
minimal analysis needed for one result: trials and intermediate objects may
travel with it. Copying only a result to another workspace does not make that
workspace a complete verifiable analysis. Prepare a reduced exchange by
deliberately keeping or rebuilding the useful inputs and checking what remains
verifiable, not by editing the record by hand.

The software licence does not license the data. The manifest contains a workspace
licence declaration; when none is supplied, it uses the neutral statement
"No licence was specified for this workspace."

Inspect
-------

**Choose a capsule...** checks a capsule without opening it and shows:

- its summary: name, description, fingerprint, parent revision, whether the
  workspace is included or referenced, licence, and its activities (which ones
  can be replayed);
- its **Attachments**, which are checked against the manifest and saved only
  when you click **Save...**;
- its **Integrity**: archive, manifest, workspace and stored states. For a
  referenced workspace, **Select the referenced workspace...** supplies the
  file, which must match the size and SHA-256 declared in the capsule;
- the **Supplied documents** added with **Add a verification report...** and
  **Add a review...**, each checked against the capsule.

**Write a review** creates a review of this revision: a separate JSON file
(reviewer, statement, optional verdict *endorsed*, *disputed* or *comment*,
cited verification reports). The capsule itself is not changed.

**Open as workspace** replaces the current workspace by the workspace of the
capsule (after confirmation when the workspace is not empty). It is available
only for an intact capsule. A capsule exported later from this workspace is
proposed as a revision of it. Opening a capsule is recorded in the History
panel.

For a referenced capsule, obtain the HDF5 file separately: its address is not
downloaded automatically, and its size and fingerprint must match. A registry
does not deliver that workspace on your behalf.

Local replay
------------

Lists the recorded activities of the open workspace. **Replay** recomputes an
activity separately and compares every result with the stored one: *Exact*,
*Within tolerance*, *Different* or *Not verified* (with the reason). The
workspace is not changed. **Save report...** saves the verification report as a
JSON file, to be supplied with the capsule or cited by a review.

Revisions
---------

Lists the revisions exported, inspected or retrieved from a registry on this
computer (with their parent revision), and compares any two of them:
scientific, presentation, storage and integrity, environment and attachment
changes, with a scientific conclusion that never claims equality beyond what
the capsules record. The list is kept in the DataLab configuration folder (100
revisions at most); it is a convenience, never a source of truth: the capsule
files are.

A capsule is immutable and names at most one direct parent. If you open A and
export B, the next export still proposes A as its parent, not B; saving and
reopening the workspace preserves that origin. Inspecting another file does not
change it. Adding a workspace keeps the current record and parent without
merging provenance (see :ref:`provenance_append`).

Two exports can have different fingerprints without a new calculation: HDF5
storage, dates or presentation can change. ``none_detected`` means no scientific
change detected in the recorded activities and states, not equality of everything
outside the record. Changing an attached protocol is listed under attachments;
the reader, not the comparison, judges its scientific importance. There is no
automatic merge of concurrent work, collaborative editing or conflict resolution.

Registry
--------

A registry (optional, served by DataLab-Capsule) publishes capsule manifests,
their attachments and reviews, never workspaces. Everything it returns is
checked by DataLab.

- **Registry address** and **Publishing token**: the address is saved in the
  configuration; the token is kept for the current session only;
- **List revisions** lists the published revisions; **Retrieve** checks a
  revision and its reviews, adds it to the **Revisions** tab and offers its
  stored attachments, checked against the manifest;
- once a registry is set, the **Inspect** tab offers **Publish to the
  registry** (manifest, attachments and valid supplied reviews) and **Publish
  the review**.

The reference registry has no accounts, strong author identity, quotas, rate
limiting, delete API or fine-grained permissions. Its publishing token is not
proof of scientific authorship. See the existing `deployment and trust guidance
<https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/security.md>`_.
Reviews and reports target one revision; a checked author's assertion does not
replace local replay or implement a complete approval workflow.

From scripts
------------

Capsules can be exported and opened from scripts and macros (see
:ref:`ref-to-remote-control`), with
:meth:`~datalab.control.proxy.RemoteProxy.export_capsule` and
:meth:`~datalab.control.proxy.RemoteProxy.open_capsule`:

.. code-block:: python

    fingerprint = proxy.export_capsule(
        "chain.dlcapsule", name="chain", attachments=["protocol.txt"]
    )
    proxy.open_capsule("chain.dlcapsule")

The ``workspace-capsule`` command line of DataLab-Capsule also validates,
inspects, compares, verifies and publishes capsules.

Its ``verify`` command performs no Sigima calculation. Without h5py, workspace
checking is limited to size and SHA-256 (``integrity_only``), not individual
HDF5 states. A referenced workspace not supplied further limits the checks.

Limits
------

DataLab checks the size of a capsule, and of the workspace it references, before
reading it, and never writes a capsule it would refuse to open. These limits are
those of DataLab-Capsule (a workspace of 4 GiB at most): they are accepted
sizes, not a tested capacity.

What was measured, on a laptop of 2026 (32 GiB of RAM):

- inspecting or opening a 32 MiB capsule from the dialog takes 0.2 to 0.5 s,
  during which DataLab does not respond (wait cursor); exporting it takes
  0.25 s. Larger capsules take longer, roughly in proportion; the library
  verifies a 234 MiB capsule in less than a second;
- opening a capsule needs about its size in memory, in addition to the
  workspace it replaces;
- recording the provenance of a processing costs the fingerprints of its
  inputs and results: nothing measurable for a small signal, +29 ms for a
  2048×2048 image filtered in 450 ms;
- consulting the provenance of a result takes less than a millisecond, even
  with 1000 recorded steps; listing 1000 steps in **Local replay** takes about
  half a second.

Each processing gets slightly slower as a panel holds more objects, with or
without provenance (about 0.1 s in an empty panel, 0.3 s with 1000 objects).
DataLab-Web limits capsules to 256 MiB.

.. list-table:: Accepted file policies, not guaranteed capacities
    :header-rows: 1

    * - Consumer
      - Workspace / manifest
    * - Desktop and library
      - 4 GiB / 64 MiB; the archive has a separate total-size limit
    * - Web
      - 256 MiB / 16 MiB; archive also limited to 256 MiB
    * - Reference registry
      - No workspace; manifest 8 MiB, review 1 MiB

Each capsule accepts at most 8 attachments of 16 MiB each. Memory can hold
several representations of the archive and HDF5 while exporting; a valid file
may exceed another edition's policy. The ledger also has JSON volume and
structural bounds, even without a fixed activity-count limit. View truncation
does not remove records. See the `technical limits and measured capacity
<https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/performance.md>`_.

Local replay verifies one activity on copies; it does not reconstruct an entire
chain, restore all missing states or provide a workflow for new datasets.
Environment versions are described, not reinstalled: a capsule is not an OS
with all its dependencies, and a successful comparison is not universal
reproducibility between machines.

Fingerprints detect changes but do not authenticate authors; capsules and
reviews are unsigned and their fingerprints can be recomputed after editing.
Import never runs received macros, notebooks, attachments or Python names;
explicit replay uses locally registered, qualified operations. The prescribed
profile accepts native supported workspaces, not arbitrary ZIP/HDF5 files:
ZIP entries are uncompressed (independent of internal HDF5 compression), and
links and encrypted archives are refused. Format checks do not replace a
sandbox: HDF5 parsing and dependencies remain trusted code on the Desktop.

Profile 0.1 is experimental and unknown versions are refused. RO-Crate/PROV
metadata do not guarantee that any third-party reader can restore or replay
DataLab. See the `common rules
<https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/rules.md>`_
and `security analysis
<https://github.com/DataLab-Platform/DataLab-Capsule/blob/develop/doc/security.md>`_.
Desktop and the CLI can work offline once installed; the Web downloads its
runtime at startup and some optional packages on first use. Computations remain
local; neither a registry nor attachments certify the physical acquisition.
