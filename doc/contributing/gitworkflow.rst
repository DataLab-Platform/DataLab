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
- ``develop``: Used for ongoing development and integration of new features.

Feature Branches
^^^^^^^^^^^^^^^^

- ``feature/feature_name``: Used for the development of new features.

  - Created from ``develop``.
  - Merged back into ``develop`` once completed.
  - Deleted after merging.

Release Branch
^^^^^^^^^^^^^^

- ``release``: Represents the current maintenance line for patch releases.

  - Created from ``main`` after a stable release when the first patch is needed.
  - Accepts ``fix/xxx`` and ``hotfix/xxx`` branch merges.
  - Merged back into ``main`` for each patch release tag (e.g., 1.0.1, 1.0.2).
  - Reset or recreated when starting a new minor/major release cycle.

.. note::

      The ``release`` branch is not versioned (no ``release/1.0.x``). It always
      represents "the current maintenance line." When a new minor version is released
      (e.g., 1.2.0), the old ``release`` branch is deleted and a new one is created
      from the fresh tag when the first patch for 1.2.1 is needed.

Bug Fix Branches
^^^^^^^^^^^^^^^^

- ``fix/xxx``: Used for general bug fixes that are not urgent.

  - Created from ``develop`` (for next feature release) or ``release`` (for patch release).
  - Merged back into the originating branch once completed.
  - Deleted after merging.

- ``hotfix/xxx``: Used for urgent production-critical fixes.

  - Created from ``release`` (if exists) or ``main`` (if no ``release`` branch yet).
  - Merged back into ``release`` or ``main``.
  - The fix is then cherry-picked into ``develop``.
  - Deleted after merging.

.. note::

      Hotfixes (high-priority fixes) will be integrated in the next maintenance
      release (X.Y.Z -> Z+1), while fixes (low-priority fixes) will be integrated
      in the next feature release (X.Y -> Y+1).


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
         git checkout -b develop/feature_name

2. Develop the feature and commit changes.

3. Merge the feature branch back into ``develop``:

   .. code-block:: sh

         git checkout develop
         git merge --no-ff develop/feature_name

4. Delete the feature branch:

   .. code-block:: sh

         git branch -d develop/feature_name

.. warning::

      Do not leave feature branches unmerged for too long.
      Regularly rebase them on ``develop`` to minimize conflicts.

Workflow for Regular Bug Fixes
------------------------------

For next feature release (target: ``develop``):

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

For current maintenance release (target: ``release``):

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

      Do not create a ``fix/xxx`` branch from a ``develop/feature_name`` branch.
      Always branch from ``develop`` or ``release`` to ensure fixes are correctly propagated.

      .. code-block:: sh

            # Incorrect:
            git checkout develop/feature_name
            git checkout -b fix/wrong_branch

      .. code-block:: sh

            # Correct:
            git checkout develop
            git checkout -b fix/correct_branch

Workflow for Critical Hotfixes
------------------------------

1. Create a hotfix branch from ``release`` (or ``main`` if no ``release`` branch exists):

   .. code-block:: sh

         git checkout release  # or: git checkout main
         git checkout -b hotfix/critical_bug

2. Apply the fix and commit changes.

3. Merge the fix back into ``release`` (or ``main``):

   .. code-block:: sh

         git checkout release  # or: git checkout main
         git merge --no-ff hotfix/critical_bug

4. Cherry-pick the fix into ``develop``:

   .. code-block:: sh

         git checkout develop
         git cherry-pick <commit_hash>

5. Delete the hotfix branch:

   .. code-block:: sh

         git branch -d hotfix/critical_bug

.. warning::

      Do not merge ``fix/xxx`` or ``hotfix/xxx`` directly into ``main`` without following the workflow.
      Ensure hotfixes are cherry-picked into ``develop`` to avoid losing fixes in future releases.

Workflow for Patch Releases
----------------------------

When ready to release a patch version (e.g., 1.0.1, 1.0.2):

1. Ensure the ``release`` branch contains all desired fixes.

2. Merge ``release`` into ``main``:

   .. code-block:: sh

         git checkout main
         git merge --no-ff release

3. Tag the release on ``main``:

   .. code-block:: sh

         git tag -a v1.0.1 -m "Release version 1.0.1"
         git push origin main --tags

4. Keep the ``release`` branch for additional patches in the same minor version series.

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

3. Delete the old ``release`` branch (if exists):

   .. code-block:: sh

         git branch -d release
         git push origin --delete release

4. Create a new ``release`` branch from ``main`` when the first patch for 1.1.1 is needed:

   .. code-block:: sh

         git checkout main
         git checkout -b release
         git push -u origin release

Best Practices
--------------

- Regularly **rebase feature branches** on ``develop`` to stay up to date:

  .. code-block:: sh

        git checkout develop/feature_name
        git rebase develop

- Avoid long-lived branches to minimize merge conflicts.

- Ensure bug fixes in ``release`` or ``main`` are **always cherry-picked** to ``develop``.

- Clearly differentiate between ``fix/xxx`` (non-urgent fixes) and ``hotfix/xxx`` (critical production fixes).

- When creating the ``release`` branch, update release notes to indicate which version it targets (e.g., add a comment in the merge commit: "Create release branch for v1.0.x maintenance").

- The ``release`` branch always represents the current maintenance line. To know which version it targets, check the most recent tag on ``main`` or the release notes.

Takeaway
--------

This workflow ensures a structured yet flexible development process while keeping
``main`` stable and ``develop`` always updated with the latest changes.

The ``release`` branch provides a dedicated maintenance line for patch releases,
allowing ``develop`` to proceed with new features without interference. This solves
the problem of coordinating unreleased changes across the PlotPyStack ecosystem
(DataLab, Sigima, PlotPy, guidata, PythonQwt) by providing a stable branch for
CI testing during maintenance cycles.
