---
name: doc-coherence-audit
description: "Audit the consistency of the documentation of DataLab, Sigima, SigimaX and DataLab-Web with each other and with their code bases, read-only: automated checks (Sphinx builds, French catalogs, links, dependency matrix), semantic checks by perimeter with introspection and execution of the examples, cross-project checks, verification of the high-severity findings, comparison with the previous audit, consolidated report and proposed follow-up issues. Use when asked to audit, check or review the documentation consistency, to replay or refresh the documentation audit, or to find outdated documentation (audit de cohérence de la documentation, rejouer l'audit, documentation obsolète)."
argument-hint: "[full | incremental since <tag|commit|date> | projects <list> | perimeters <P1,P4,...> | no network]"
---

# Documentation Consistency Audit

Check, without modifying any repository, that the documentation of DataLab, Sigima, SigimaX and DataLab-Web is consistent with each other and with the code. The result is a consolidated report of verified findings and a proposed follow-up; the fixes themselves go through the `issue-pr-fusion` skill, after the user has chosen them.

Talk to the user and write the report in their language. Quote documentation and code verbatim, in their original language.

## Inputs

| Option | Effect |
| --- | --- |
| `full` (default) | Every perimeter of [perimeters.md](./references/perimeters.md), every file in full |
| `incremental since <ref>` | Perimeters reduced to the documentation changed since `<ref>` (tag, commit or date) and to the pages describing the code changed since then; the automated and cross-project checks still run in full |
| `projects <list>` | Restrict to some projects (e.g. `Sigima SigimaX`); the cross-project checks then only compare these projects |
| `perimeters <list>` | Restrict to some perimeters (e.g. `P4,P9`) |
| `no network` | Skip the URL checks and the PyPI/conda-forge queries |

Optional projects (`DataLab-Capsule`, `DataLab-Kernel`, `DataLab-Plugins`, the plugin repositories) are audited only when named: add a perimeter for each one, built like those of [perimeters.md](./references/perimeters.md).

## Rules

- **Read-only.** Never modify, stage or commit a file of an audited repository, never switch their branches, never create anything on GitHub. Record `git status --porcelain` of every repository at the start and check it is identical at the end. Builds and scratch files go to `%TEMP%\docaudit`, deleted at the end. If a tool writes into a repository (e.g. `scripts/print_datalab_menus.py`), restore the file with `git restore` only if it was clean before the run.
- **Verified findings only.** A finding cites the documentation (`path:line` and quotation) and the code fact (`path:line`, introspected value or command output). Unproven suspicions go to a "To be confirmed" list. Do not trust the previous report: recheck each of its findings.
- **Audit the current state.** Audit the local branches the user works on (usually `develop`, after `git fetch`); report the branch, commit and version of each repository. When a local branch is behind its upstream or has local changes, say so and ask whether to audit it as is.
- Preserve unrelated local changes and untracked files of the repositories: never clean them.
- With several perimeters, track them in the SQL `todos` table.

## Procedure

### 1. Preflight (read-only)

For each project (folders `C:\Dev\Projets\DataLab`, `Sigima`, `SigimaX`, `DataLab-Web`), after `New-Item -ItemType Directory -Force "$env:TEMP\docaudit"`:

```powershell
git -C <repo> fetch origin --prune
git -C <repo> status -sb
git -C <repo> status --porcelain > "$env:TEMP\docaudit\status_<repo>_before.txt"
git -C <repo> log -1 --format="%h %ci %s"
git -C <repo> describe --tags --always
```

- Environment: DataLab and Sigima have a `.venv` and an `.env`; SigimaX uses DataLab's interpreter with `$env:PYTHONPATH = ".;..\guidata;..\PythonQwt;..\plotpy;..\Sigima"` (see `issue-pr-fusion` [repositories.md](../issue-pr-fusion/references/repositories.md)). Always set `$env:PYTHONIOENCODING = "utf-8"` and, for Qt code, `$env:QT_QPA_PLATFORM = "offscreen"`.
- Previous audit: the most recent `doc-audit-*.md` in `$HOME\.copilot\doc-audits\`. Load its findings to classify each new finding as fixed, still present or new.
- Known issues: list the open documentation issues to link the findings already reported instead of proposing duplicates: `gh issue list --repo <owner/repo> --state open --label documentation --limit 100 --json number,title`.
- Incremental mode: `git -C <repo> diff --name-only <ref>..HEAD` (with a date, find the commit with `git rev-list -1 --before=<date> HEAD`), then map the changed files to the perimeters.

Tell the user the scope, mode, branches and expected duration, then continue without waiting unless something is ambiguous (local changes, branch behind).

### 2. Automated Checks

Run them in the background while the perimeters are audited (step 3).

- **Sphinx builds**, English and French, strict, for DataLab, Sigima and SigimaX (from the repository folder):

  ```powershell
  python scripts\run_with_env.py python -m sphinx build doc "$env:TEMP\docaudit\<repo>_en" -b html -W --keep-going -D language=en
  python scripts\run_with_env.py python -m sphinx build doc "$env:TEMP\docaudit\<repo>_fr" -b html -W --keep-going -D language=fr
  ```

  Report every warning. Check `doc/conf.py` for `suppress_warnings` entries hiding real errors (they were used to hide docutils errors).
- **French catalogs**: extract the messages, then update a copy of the catalogs, never `doc/locale` itself:

  ```powershell
  python scripts\run_with_env.py python -m sphinx build doc "$env:TEMP\docaudit\<repo>_gettext" -b gettext
  Copy-Item doc\locale "$env:TEMP\docaudit\<repo>_locale" -Recurse
  python scripts\run_with_env.py python -m sphinx_intl update -d "$env:TEMP\docaudit\<repo>_locale" -p "$env:TEMP\docaudit\<repo>_gettext" -l fr --no-obsolete -w 0
  python scripts\run_with_env.py python -m sphinx_intl stat -d "$env:TEMP\docaudit\<repo>_locale" -l fr
  ```

  Report the catalogs with untranslated or fuzzy entries (empty `msgstr` for code or references only are expected), and the French pages whose meaning diverges from the English ones on the sampled findings. For DataLab-Web: `npm run i18n:check`.
- **Links**: [check_links.py](./scripts/check_links.py) checks the platform URLs (GET) and the relative Markdown links of the files tracked by git:

  ```powershell
  python .github\skills\doc-coherence-audit\scripts\check_links.py C:\Dev\Projets\DataLab C:\Dev\Projets\Sigima C:\Dev\Projets\SigimaX C:\Dev\Projets\DataLab-Web --workers 16
  ```

  Classify each broken link: dead link, page not deployed yet (target present in the `develop` sources, e.g. a new outreach page), historical link in old release notes (low severity, redirection possible).
- **Dependency matrix**: build a table of the minimum versions (Python, Sigima, SigimaX, DataLab-Capsule, guidata, PlotPy, NumPy, SciPy, Pyodide, Node.js) found in every `pyproject.toml`, `requirements.txt`, `doc/requirements.rst`, README, Copilot instructions, release notes `Requirements` sections, DataLab-Web `*-dependency.json`, `package.json` and `requirements-dev.txt`. Every divergence is a finding. Generated requirement files should match `python -m guidata.utils.genreqs` output (run it in a scratch copy, not in the repository).
- **Published packages** (unless `no network`): check the installation commands against PyPI (`https://pypi.org/pypi/<name>/json`) and conda-forge (`https://api.anaconda.org/package/conda-forge/<name>`).

### 3. Semantic Checks by Perimeter

Launch one `general-purpose` subagent per perimeter of [perimeters.md](./references/perimeters.md), in parallel, with the prompt template of [report-template.md](./references/report-template.md). Give each one the paths, branches, environment, scratch folder, the previous findings of its perimeter and, in incremental mode, its file list. While they run, do the automated checks; do not redo their work.

When a subagent returns findings without code facts or skips mandatory checks, send it a follow-up with `write_agent` rather than launching another one.

### 4. Cross-Project Checks

With the facts reported by the subagents and the automated checks, run the checks of [cross-project.md](./references/cross-project.md) yourself. Cite both projects in each finding.

### 5. Verification

- Re-verify yourself **every high-severity finding** and a sample of at least five medium ones (reread the documentation line and reproduce the code fact). Downgrade or drop the findings that do not hold, and count them in the report.
- Check the "To be confirmed" items that are cheap to settle.
- Check that every repository's `git status --porcelain` is identical to the one recorded in step 1.

### 6. Consolidation

- Deduplicate (same defect reported by several perimeters: keep one finding, cite all locations) and renumber by perimeter.
- Classify against the previous audit (fixed, still present, new) and link the open issues already covering a finding.
- Group the findings into follow-up proposals: one group per repository and topic, small enough for one PR; separate "documentation fix", "code fix" and "decision needed" (the documentation or the code may be the reference: say which one you recommend and why).

### 7. Report and Handover

- Write the report with the structure of [report-template.md](./references/report-template.md) to `files/doc-audit-<YYYY-MM-DD>.md` in the session folder, and archive a copy in `$HOME\.copilot\doc-audits\` (outside the repositories: the next audit compares with it).
- Present to the user: the summary table, the priorities (high-severity findings and code defects), the delta with the previous audit and the proposed follow-up groups. Do not create issues: propose them, and let the user choose the groups to deliver with `/issue-pr-fusion <model>`.
- If the repository tree changed (new pages, perimeters to split) or a new class of cross-project defect was found, propose the corresponding update of this skill's references.

### 8. Cleanup

Delete `%TEMP%\docaudit` and the session temporary files other than the report. Confirm that the repositories are in their initial state.
