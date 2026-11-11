# Repository Specifics

Conventions observed in November 2026. Before relying on a row, check it against `gh pr list --repo <owner/repo> --state merged --limit 10 --json baseRefName,headRefName` and the repository's `RELEASING.md` and `CONTRIBUTING.md`.

## Branches, Release Notes and Issues

Always pass `--repo <owner/repo>` to `gh`.

| Folder | GitHub | Fix base | Feature base | Default branch | Release notes |
| --- | --- | --- | --- | --- | --- |
| `DataLab` | `DataLab-Platform/DataLab` | `develop` | `develop` | `main` | `doc/release_notes/release_X.YY.md` + French `.po` |
| `Sigima` | `DataLab-Platform/Sigima` | `develop` | `develop` | `main` | `doc/release_notes/release_X.YY.md` + French `.po` |
| `SigimaX` | `DataLab-Platform/SigimaX` | `develop` | `develop` | `main` | `doc/release_notes/release_X.YY.md` + French `.po` |
| `DataLab-Capsule` | `DataLab-Platform/DataLab-Capsule` | `develop` | `develop` | `main` | `CHANGELOG.md` |
| `DataLab-Web` | `DataLab-Platform/web` | `develop` | `develop` | `main` | `CHANGELOG.md` |
| `DataLab-Kernel` | `DataLab-Platform/DataLab-Kernel` | `develop` | `develop` | `main` | `doc/release_notes/release_X.YY.md` |
| `guidata` | `PlotPyStack/guidata` | `master` | `develop` | `master` | `doc/release_notes/release_X.YY.md` |
| `PlotPy` | `PlotPyStack/PlotPy` | `develop` | `develop` | `master` | `doc/release_notes/release_X.YY.md` |
| `PythonQwt` | `PlotPyStack/PythonQwt` | `master` | `master` | `master` | `CHANGELOG.md` |
| `DataLab-Plugins` | `DataLab-Platform/plugins` | `main` | `main` | `main` | none |
| `datalab-tango`, `datalab-synchrotron`, `datalab-camera-characterization`, `datalab-pulse-characterization` | `DataLab-Platform/<folder>` | `develop` | `develop` | `develop` | `CHANGELOG.md` |
| `datalab-dynamic-demo` | `DataLab-Platform/datalab-dynamic-demo` | `main` | `main` | `main` | `CHANGELOG.md` |

- When the PR base differs from the default branch (e.g. `develop` vs `main`), `Closes #N` does not close the issue at merge time: close it manually (step 10).
- Milestone names vary between repositories (`v1.4.1`, `V1.2.0`, ...) and some stay open after the release: list the open ones and pick the next release, or ask.
- Labels: `bug` for fixes, `enhancement` for features and improvements, `documentation` for documentation-only changes.

## CI

- DataLab, SigimaX, guidata and PlotPy test workflows run on pull requests targeting `develop`, `main`/`master` or `release` only. Sigima's `test.yml` runs on every pull request, stacked ones included. DataLab-Capsule tests also run on pushes to `feature/**` and `fix/**`.
- DataLab CI (PRs targeting `develop`) clones Sigima, SigimaX and DataLab-Capsule at the branch named like the PR head branch, falling back to `develop`, and always clones guidata and PlotPy at `develop`. Hence:
  - Use identical branch names for a change spanning these repositories.
  - A guidata or PlotPy change must be in their `develop` branch before a DataLab PR relies on it.
- guidata fix merged into `master`: when `develop` is an ancestor of `master` (`git merge-base --is-ancestor origin/develop origin/master`), fast-forward it with `git push origin origin/master:refs/heads/develop`; otherwise, ask (merge PR from `master` into `develop`).
- The standalone build (PyInstaller, MSI) only runs in DataLab's `release-dryrun.yml` (manual) and `release.yml` (tags): PR CI does not test it. Start `release-dryrun.yml` only when the change affects packaging and the user agrees.
- External checks (e.g. Read the Docs builds in SigimaX PRs) count for the green status like workflow checks.

## Environment (Windows, PowerShell)

- `python` alone may resolve to the Microsoft Store stub: call the interpreter of the repository's `.venv` explicitly. SigimaX has neither `.venv` nor `.env`: use `C:\Dev\Projets\DataLab\.venv\Scripts\python.exe` with `$env:PYTHONPATH = ".;..\guidata;..\PythonQwt;..\plotpy;..\Sigima"`.
- `scripts/run_with_env.py` loads the repository's `.env` (sibling projects in `PYTHONPATH`) and prints an emoji at startup: set `$env:PYTHONIOENCODING = "utf-8"` first.

  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  .venv\Scripts\python.exe scripts\run_with_env.py python -m pytest <tests> -q
  .venv\Scripts\python.exe scripts\run_with_env.py python -m ruff format <files>
  .venv\Scripts\python.exe scripts\run_with_env.py python -m ruff check <files>
  ```

- A git worktree outside `C:\Dev\Projets` needs its own `.env` with absolute paths, otherwise imports silently use stale site-packages.
- Write issue bodies, PR bodies and commit messages to UTF-8 files (without BOM) in the session folder and pass them with `--body-file` or `git commit -F`: inline multi-line strings break on PowerShell quoting.
- DataLab-Web is a TypeScript/React and Pyodide project: follow its `CONTRIBUTING.md` and `doc/testing-strategy.md` for tests (npm scripts, Playwright).

## DataLab Release Notes and Translations

- Follow the "Release Notes Guidelines" of `.github/copilot-instructions.md`: user impact, not implementation; requirement changes in the `Requirements` section.
- French translation of the release notes, in the same commit as the English entry:

  ```powershell
  .venv\Scripts\python.exe scripts\run_with_env.py python -m sphinx build doc build/gettext -b gettext -W
  .venv\Scripts\python.exe scripts\run_with_env.py python -m sphinx_intl update -d doc/locale -p build/gettext -l fr --no-obsolete -w 0
  .venv\Scripts\python.exe scripts\run_with_env.py python -m guidata.utils.translations cleanup-doc --directory .
  ```

  Then fill in the new `msgstr` entries and commit only the catalogs of the changed pages. `build/gettext` keeps the `.pot` files of other branches: delete the `.po` files recreated for pages absent from the current branch.
- UI translations (`_()` strings): the `guidata.utils.translations scan` command also scans an ignored `build/lib/datalab` copy and misses `_()` calls inside f-strings. Remove `build/lib` first, and check the removed `msgid` entries before committing `datalab.po`.
