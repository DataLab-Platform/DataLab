.. _gitworkflow:

Git Workflow
============

This document describes the Git workflow used in the DataLab project,
based on a ``main`` branch, a ``develop`` branch, and feature-specific branches.
It also defines how bug fixes are managed.

.. note::

      This workflow is a simplified version of the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_.
      It has been adapted to suit the needs of the DataLab project at the current stage of development.
      In the near future, we may consider adopting a more complex workflow, e.g. by adding release branches.

Branching Model
---------------

Main Branches
^^^^^^^^^^^^^

- ``main``: Represents the stable, production-ready version of the project.
- ``develop``: Prepares the next release, i.e. a patch release (e.g., 1.0.1) as long as it only contains fixes since the last release, or a minor or major release (e.g., 1.1.0) once new features are merged.

Feature Branches
^^^^^^^^^^^^^^^^

- ``feature/feature_name``: Used for the development of new features.

  - Created from ``develop``.
  - Merged back into ``develop`` once completed.
  - Deleted after merging.

Release Branch
^^^^^^^^^^^^^^

- ``release``: Represents the current maintenance line for patch releases, when ``develop`` already contains changes for the next minor or major release.

  - Created from ``main`` only when a patch release is needed while ``develop`` contains changes that must not be released yet.
  - Accepts ``fix/xxx`` and ``hotfix/xxx`` branch merges.
  - Merged back into ``main`` for each patch release tag (e.g., 1.0.1, 1.0.2).
  - Merged into ``develop`` after each patch release, so that its fixes reach ``develop`` without cherry-picking.
  - Deleted when the next minor or major version is released.

.. note::

      The ``release`` branch is not versioned (no ``release/1.0.x``), and it does not
      always exist. Without ``release`` branch, ``develop`` prepares the next patch
      release: fixes and patch releases are made from ``develop``. When a new minor
      version is released (e.g., 1.2.0), the ``release`` branch is deleted; a new one
      is created from the fresh tag only if a patch for 1.2.1 is needed while
      ``develop`` already contains changes for 1.3.0.

Bug Fix Branches
^^^^^^^^^^^^^^^^

- ``fix/xxx``: Used for general bug fixes that are not urgent.

  - Created from ``develop``, or from ``release`` when this branch exists and the fix must be in the next patch release.
  - Merged back into the originating branch once completed.
  - Deleted after merging.

- ``hotfix/xxx``: Used for urgent production-critical fixes.

  - Created from ``release`` (if exists) or ``develop`` (if no ``release`` branch).
  - Merged back into ``release`` or ``develop``.
  - A fix merged into ``release`` reaches ``develop`` when ``release`` is merged into ``develop``.
  - Deleted after merging.

.. note::

      Hotfixes (high-priority fixes) will be integrated in the next patch release
      (X.Y.Z -> Z+1). Fixes (low-priority fixes) will be integrated in the next
      release prepared by ``develop``: a patch release, or a feature release
      (X.Y -> Y+1) once ``develop`` contains new features.


Documentation Branches
----------------------

When working on documentation that is not related to source code
(e.g. training materials, user guides), branches should be named
using the ``doc/`` prefix.

Examples:

- ``doc/training-materials``
- ``doc/user-guide``

This naming convention improves clarity by clearly separating
documentation efforts from code-related development (features, fixes, etc.).


Workflow for New Features
-------------------------

1. Create a new feature branch from ``develop``:

   .. code-block:: sh

         git checkout develop
         git checkout -b feature/feature_name

2. Develop the feature and commit changes.

3. Merge the feature branch back into ``develop``:

   .. code-block:: sh

         git checkout develop
         git merge --no-ff feature/feature_name

4. Delete the feature branch:

   .. code-block:: sh

         git branch -d feature/feature_name

.. warning::

      Do not leave feature branches unmerged for too long.
      Regularly rebase them on ``develop`` to minimize conflicts.

Workflow for Regular Bug Fixes
------------------------------

For the next release prepared by ``develop`` (target: ``develop``):

1. Create a bug fix branch from ``develop``:

   .. code-block:: sh

         git checkout develop
         git checkout -b fix/bug_description

2. Apply the fix and commit changes.

3. Merge the fix branch back into ``develop``:

   .. code-block:: sh

         git checkout develop
         git merge --no-ff fix/bug_description

4. Delete the fix branch:

   .. code-block:: sh

         git branch -d fix/bug_description

For the next patch release, when a ``release`` branch exists (target: ``release``):

1. Create a bug fix branch from ``release``:

   .. code-block:: sh

         git checkout release
         git checkout -b fix/bug_description

2. Apply the fix and commit changes.

3. Merge the fix branch back into ``release``:

   .. code-block:: sh

         git checkout release
         git merge --no-ff fix/bug_description

4. Delete the fix branch:

   .. code-block:: sh

         git branch -d fix/bug_description

.. warning::

      Do not create a ``fix/xxx`` branch from a ``feature/feature_name`` branch.
      Always branch from ``develop`` or ``release`` to ensure fixes are correctly propagated.

      .. code-block:: sh

            # Incorrect:
            git checkout feature/feature_name
            git checkout -b fix/wrong_branch

      .. code-block:: sh

            # Correct:
            git checkout develop
            git checkout -b fix/correct_branch

Workflow for Critical Hotfixes
------------------------------

1. Create a hotfix branch from ``release`` (or ``develop`` if no ``release`` branch exists):

   .. code-block:: sh

         git checkout release  # or: git checkout develop
         git checkout -b hotfix/critical_bug

2. Apply the fix and commit changes.

3. Merge the fix back into ``release`` (or ``develop``):

   .. code-block:: sh

         git checkout release  # or: git checkout develop
         git merge --no-ff hotfix/critical_bug

4. If the fix was merged into ``release``, merge ``release`` into ``develop``, at the latest after the patch release:

   .. code-block:: sh

         git checkout develop
         git merge --no-ff release

5. Delete the hotfix branch:

   .. code-block:: sh

         git branch -d hotfix/critical_bug

.. warning::

      Do not merge ``fix/xxx`` or ``hotfix/xxx`` directly into ``main`` without following the workflow.
      Ensure ``release`` is merged into ``develop`` to avoid losing fixes in future releases.

Workflow for Patch Releases
----------------------------

When ready to release a patch version (e.g., 1.0.1, 1.0.2):

If no ``release`` branch exists, ``develop`` only contains fixes since the last release: release it with the workflow for minor/major releases below, using the patch version number. Otherwise, release the ``release`` branch:

1. Ensure the ``release`` branch contains all desired fixes.

2. Merge ``release`` into ``main``:

   .. code-block:: sh

         git checkout main
         git merge --no-ff release

3. Tag the release on ``main``:

   .. code-block:: sh

         git tag -a v1.0.1 -m "Release version 1.0.1"
         git push origin main --tags

4. Pushing the tag starts the **Release** workflow, which checks the tag, runs the tests and builds the release files, then creates a draft GitHub release. Review the draft release and publish it: this uploads the package to PyPI and deploys the documentation. Never create a GitHub release by hand (see ``RELEASING.md`` at the root of the repository).

5. Merge ``release`` into ``develop``, and keep the ``release`` branch for additional patches in the same minor version series:

   .. code-block:: sh

         git checkout develop
         git merge --no-ff release

Workflow for Minor/Major Releases
----------------------------------

When ready to release a new minor or major version (e.g., 1.1.0, 2.0.0):

1. Merge ``develop`` into ``main``:

   .. code-block:: sh

         git checkout main
         git merge --no-ff develop

2. Tag the release on ``main``:

   .. code-block:: sh

         git tag -a v1.1.0 -m "Release version 1.1.0"
         git push origin main --tags

3. Pushing the tag starts the **Release** workflow, which checks the tag, runs the tests and builds the release files, then creates a draft GitHub release. Review the draft release and publish it: this uploads the package to PyPI and deploys the documentation. Never create a GitHub release by hand (see ``RELEASING.md`` at the root of the repository).

4. Delete the old ``release`` branch (if exists):

   .. code-block:: sh

         git branch -d release
         git push origin --delete release

5. Create a new ``release`` branch from ``main`` only when a patch (e.g., 1.1.1) is needed while ``develop`` already contains changes for the next minor or major release:

   .. code-block:: sh

         git checkout main
         git checkout -b release
         git push -u origin release

Best Practices
--------------

- Regularly **rebase feature branches** on ``develop`` to stay up to date:

  .. code-block:: sh

        git checkout feature/feature_name
        git rebase develop

- Avoid long-lived branches to minimize merge conflicts.

- Merge ``release`` into ``develop`` after each patch release: fixes reach ``develop`` without cherry-picking.

- While ``develop`` prepares a patch release (no ``release`` branch), keep changes for the next minor or major release in their branches until the patch release, or create the ``release`` branch first.

- Clearly differentiate between ``fix/xxx`` (non-urgent fixes) and ``hotfix/xxx`` (critical production fixes).

- When creating the ``release`` branch, update release notes to indicate which version it targets (e.g., add a comment in the merge commit: "Create release branch for v1.0.x maintenance").

- When it exists, the ``release`` branch represents the current maintenance line. To know which version it targets, check the most recent tag on ``main`` or the release notes.

Takeaway
--------

This workflow ensures a structured yet flexible development process while keeping
``main`` stable and ``develop`` always updated with the latest changes.

When needed, the ``release`` branch provides a dedicated maintenance line for patch releases,
allowing ``develop`` to proceed with new features without interference. This solves
the problem of coordinating unreleased changes across the PlotPyStack ecosystem
(DataLab, SigimaX, Sigima, PlotPy, guidata, PythonQwt) by providing a stable branch
for CI testing during maintenance cycles.
