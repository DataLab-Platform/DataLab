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

Bug Fix Branches
^^^^^^^^^^^^^^^^

- ``fix/xxx``: Used for general bug fixes that are not urgent.

  - Created from ``develop``.
  - Merged back into ``develop`` once completed.
  - Deleted after merging.

- ``hotfix/xxx``: Used for urgent production-critical fixes.

  - Created from ``main``.
  - Merged back into ``main``.
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

.. warning::

      Do not create a ``fix/xxx`` branch from a ``develop/feature_name`` branch.
      Always branch from ``develop`` to ensure fixes are correctly propagated.

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

1. Create a hotfix branch from ``main``:

   .. code-block:: sh

         git checkout main
         git checkout -b hotfix/critical_bug

2. Apply the fix and commit changes.

3. Merge the fix back into ``main``:

   .. code-block:: sh

         git checkout main
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

Best Practices
--------------

- Regularly **rebase feature branches** on ``develop`` to stay up to date:

  .. code-block:: sh

        git checkout develop/feature_name
        git rebase develop

- Avoid long-lived branches to minimize merge conflicts.

- Ensure bug fixes in ``main`` are **always cherry-picked** to ``develop``.

- Clearly differentiate between ``fix/xxx`` (non-urgent fixes) and ``hotfix/xxx`` (critical production fixes).

Takeaway
--------

This workflow ensures a structured yet flexible development process while keeping
``main`` stable and ``develop`` always updated with the latest changes.

It also ensures that bug fixes are correctly managed and propagated across branches.
