---
name: issue-pr-fusion
description: "Deliver a change end to end in the DataLab-Platform and PlotPyStack repositories: detailed plan for approval, GitHub issue, dedicated branch and PR per repository, implementation with tests, release notes and translations, manual validation with a test scenario, conventional commit with the 'Assisted-by: <model>' trailer, push, CI monitoring, merge once CI is green, issue closing and local sync. Use when asked for an issue, a dedicated branch and a PR, to commit and push, to watch the CI, to merge when green or to sync locally (issue, dedicated branch, PR, watch the CI, merge, sync locally)."
argument-hint: "<model name and version> [options: no issue | no manual validation | no merge | one PR per issue | base <branch> ...]"
---

# Issue, PR and Merge Workflow

Deliver an agreed change (fix, enhancement, documentation) from plan to merge with the same procedure in every repository of the workspace. Typical invocation, after an analysis that chose the solution(s) to implement: `/issue-pr-fusion Claude Opus 5.5`.

Talk to the user in their language. Write issues, PRs, commit messages and issue comments in English.

## Inputs

- **Model** (required): the first argument, i.e. the exact commercial name and version of the AI model (e.g. `Claude Opus 5.5`, `GPT-6.1 Sol`). Use it verbatim in the `Assisted-by: <model>` trailer of every commit and PR body. If it is missing, ask for it: never infer it from `git log`, never write `Assisted-by: GitHub Copilot`, never add `Co-authored-by: Copilot`.
- **Subject**: the rest of the invocation text or, when absent, the solution(s) chosen earlier in the conversation. Ask when the scope (which points, which repositories) is ambiguous.
- **Options** (free text, English or French), overriding the defaults:

| Option | Effect |
| --- | --- |
| `no issue` | No issue, e.g. fix of a change not released yet (then no release note entry either) |
| `issue #N` / `owner/repo#N` | Reuse an existing issue instead of creating one |
| `one PR per issue` / `one PR per fix` | One issue, branch and PR per fix instead of one for the whole change |
| `one commit per issue` | Several issues in one branch: one commit each, in the given order |
| `no manual validation` | Skip the manual validation stop (step 5) |
| `no merge` | Stop after the PR is created and its CI is green |
| `no CI` | Stop right after the push and the PR creation |
| `base <branch>` | Override the base branch |
| `direct` | Commit and push on the base branch: no issue, no PR |
| `milestone <name>`, `label <name>` | Set them on the issues and PRs |
| `issue comment` | After the merge, explain the chosen approach in an issue comment |

Defaults: one issue, one branch and one PR per repository; stops for plan approval and manual validation; merge once CI is green; issues closed; local repositories synchronized.

## Rules

- Create nothing on GitHub (issue, pushed branch, PR) before the plan is approved.
- Never merge with failing or pending checks. Never push to a base branch, except with the `direct` option or for the guidata `develop` synchronization. Force-push only the dedicated branch, with `--force-with-lease`. Never create tags or GitHub releases: releases follow each repository's `RELEASING.md`.
- Stop and ask on: a delicate merge conflict, a CI failure whose fix leaves the approved scope, a needed change of system settings (e.g. Windows light or dark mode), any destructive action not in the plan.
- Follow each repository's instructions (`.github/copilot-instructions.md`, `CONTRIBUTING.md`, `RELEASING.md`), e.g. the release notes guidelines of DataLab or the `contract_version` rule of Sigima.
- Do not silently fix defects found on the way that are outside the scope: propose a separate issue (in the plan or in the report) and list it in the "Out of scope" section of the PR.
- Keep temporary files (issue and PR bodies, commit messages, scripts) in the session folder `files/`, never in a repository, and delete them at the end.
- With several repositories or issues, track the steps in the SQL `todos` table.

## Procedure

Per-repository specifics (base branches, release notes, CI, environment) are in [repositories.md](./references/repositories.md). Issue, commit and PR body templates are in [templates.md](./references/templates.md).

### 1. Preflight (read-only)

For each repository involved:

```powershell
git -C <repo> fetch origin --prune
git -C <repo> status -sb
git -C <repo> log --oneline -3 origin/<base>
gh pr list --repo <owner/repo> --state merged --limit 8 --json number,baseRefName,headRefName
gh pr list --repo <owner/repo> --state open --json number,title,headRefName,baseRefName
gh api repos/<owner/repo>/milestones --jq ".[].title"
gh label list --repo <owner/repo> --json name --jq ".[].name"
```

- Local changes: when they belong to the change (e.g. "move the changes to a dedicated branch"), carry them to the new branch; otherwise keep them out of every commit.
- Released or not: find the commit that introduced the defect or the feature and run `git tag --contains <commit>`. A fix of a change not released yet needs neither an issue nor a release note entry: propose this in the plan.
- Check the open PRs and branches that could conflict with the change.

### 2. Plan: stop for approval

Present a detailed plan, then wait for explicit approval (with `ask_user` when available):

- Per repository: issue to create (English title, labels, milestone) or issue to reuse; branch name and base; commits (`type(scope): subject`, order); files to change; tests, including a regression test failing before the fix; documentation, release notes and translations; requirement changes.
- Cross-repository changes: identical branch names and merge order (step 9).
- Outline of the manual test scenario (detailed in step 5).
- Out-of-scope findings and the issues proposed for them.
- Open design questions, each with a recommendation.

Revise the plan until it is approved. The approval covers the rest of the workflow: only the stops below interrupt it.

### 3. Issue and Branch

- Write the issue to `files/issue_<repo>.md` (see [templates.md](./references/templates.md)), then:

  ```powershell
  gh issue create --repo <owner/repo> --title "<title>" --body-file <file> --label <bug|enhancement|documentation> [--milestone "<milestone>"]
  ```

  If `gh issue create` fails, post a JSON payload (`title`, `body`, `labels`) with `gh api repos/<owner/repo>/issues --method POST --input <json> --jq .html_url`.
- Name the branch `<type>/<issue>-<slug>`, e.g. `fix/510-float-field-display`, `feature/479-plugin-catalog`, `docs/…`, `chore/…`. Use the same name in every repository of the change: DataLab CI clones the Sigima, SigimaX and DataLab-Capsule branches with the same name.

  ```powershell
  git -C <repo> switch -c <branch> origin/<base> --no-track
  ```

  To carry uncommitted changes: `git stash push -m <name>`, switch, then `git stash pop`.
- Never base a branch on another feature branch unless asked: PR CI only runs for PRs targeting `develop`, `main` or `master` (except in Sigima), and deleting a base branch closes the PRs stacked on it.

### 4. Implementation and Automated Validation

- Implement according to the repository's instructions. Add a regression test and prove it fails without the fix (stash the fix, run the test, restore the fix). Run the targeted tests first, then broader ones as needed, plus Ruff on the changed files (commands in [repositories.md](./references/repositories.md)).
- Release notes, for user-visible changes of released code: a user-focused entry in the current release notes file, with the issue link, `(fixes [Issue #N](<url>))` for a bug or `(closes [Issue #N](<url>))` for a feature; its French translation goes in the same commit when the repository has release notes catalogs. Update the UI translations when `_()` strings change, the documentation when behavior changes, and the `Requirements` section and generated requirement files when a minimum version changes.
- Review `git status --short` and `git diff --stat`: no stray file (`build/`, scratch files, `.pyc`).

### 5. Manual Validation: stop (default)

Unless `no manual validation`, give the user a complete test scenario and wait for the result (with `ask_user` when available):

- Setup: branch to use and exact launch command (for DataLab: `DataLab.bat`, or the VS Code "Run DataLab" configuration).
- Numbered steps, each with its expected result, including edge cases.
- Regression checks of neighboring features.
- Variants when relevant: light and dark themes, Qt bindings, standalone version.
- What to report back.

If the user reports a problem, fix it, run the automated validation again and submit the scenario again.

### 6. Commit

- Write the message to `files/commit_<repo>.txt` (template in [templates.md](./references/templates.md)) and commit with `git commit -F <file>`. Subject: conventional `type(scope): summary` (`fix`, `feat`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`). Body: why and what. Then the issue reference (`Closes #N`, or `owner/repo#N` across repositories) and, as the last line and only AI trailer, `Assisted-by: <model>`.
- Stage explicit paths. With several issues in one branch, make one commit per issue in the approved order.
- Check with `git log -1 --format=%B`: right trailer, no `Co-authored-by`.

### 7. Push and PR

```powershell
git -C <repo> push -u origin <branch>
gh pr create --repo <owner/repo> --base <base> --head <branch> --title "<type(scope): summary>" --body-file <file> [--milestone "<milestone>"] [--label <label>]
```

The PR body (English, template in [templates.md](./references/templates.md)) starts with `Closes #N`, then gives the summary, the changes, the validation (automated and manual), the out-of-scope issues, the release note statement and the related PRs of other repositories (`owner/repo#N`); it ends with `Assisted-by: <model>`.

### 8. CI Monitoring

```powershell
Start-Sleep -Seconds 20; gh pr checks <n> --repo <owner/repo> --watch --interval 30
```

- For long pipelines, watch in the background and progress on other repositories meanwhile.
- No check after a minute: look at `gh run list --repo <owner/repo> --branch <branch>`, and start the workflows that need a manual start with `gh workflow run <workflow> --repo <owner/repo> --ref <branch>`.
- On failure, read `gh run view <id> --repo <owner/repo> --log-failed`:
  - Defect within the scope: fix it, run the tests, commit (or amend and `git push --force-with-lease` while nobody else reviews the branch), push and watch again.
  - Flaky or unrelated failure (runner, network, known flaky test): `gh run rerun <id> --repo <owner/repo> --failed` once, and say so in the report. If it persists, stop and report.
- Green means every check passes or is skipped: confirm with `gh pr view <n> --repo <owner/repo> --json mergeStateStatus,statusCheckRollup`.

### 9. Merge (unless `no merge`)

- Merge dependencies first: PythonQwt, guidata, PlotPy, Sigima, SigimaX, DataLab-Capsule, then DataLab, then DataLab-Web, DataLab-Kernel and plugins.
- Before deleting the branch, check that no PR is based on it: `gh pr list --repo <owner/repo> --base <branch>` must be empty.

  ```powershell
  gh pr merge <n> --repo <owner/repo> --merge --delete-branch
  ```

  Use a merge commit (no squash): the history keeps the `Merge pull request #N` commits.
- guidata fix merged into `master`: synchronize `develop` afterwards (see [repositories.md](./references/repositories.md)).

### 10. Issue Closing

- Check `gh issue view <N> --repo <owner/repo> --json state`. A PR merged into a branch other than the default one (e.g. `develop` when the default is `main`) does not close its issues: close them with `gh issue close <N> --repo <owner/repo> --reason completed --comment "Fixed by #<PR> (merged into develop)."`.
- With the `issue comment` option, or for an issue reported by an external user, explain the chosen approach in an English comment.

### 11. Local Synchronization and Cleanup

```powershell
git -C <repo> fetch origin --prune
git -C <repo> switch <base>
git -C <repo> merge --ff-only origin/<base>
git -C <repo> branch -d <branch>
```

Remove the worktrees and session temporary files. Check that `git status -sb` is clean and up to date in every repository involved.

### 12. Report

Summarize, in the user's language, with one table row per repository: issue, branch, PR, CI result, merge commit and issue state. Add the out-of-scope issues created and what is left to do (e.g. a requirement bump after an upstream release).
