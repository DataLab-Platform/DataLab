# Decisions on Previous Findings

Findings of previous audits that the maintainers have decided on. Read this file in the preflight (step 1):

- **Accepted**: do not report the finding again, unless one of the facts it cites has changed (then report it as new, and say which fact changed). Count the accepted findings found again in the report, without detailing them.
- **Tracked**: link the open issue instead of proposing a new one; report the finding again only if the issue was closed while the defect is still present.

When the user decides not to fix a finding of an audit, or opens an issue to track it, add it here (in English, with the date and the facts that would reopen it) in the same change as the other follow-ups of the audit.

## Accepted (no action)

| Topic | Decision (date) | Facts that would reopen it |
| --- | --- | --- |
| DataLab Web API: requests from localhost are accepted without token by default (`webapi_localhost_no_token`, `default=True` in `datalab/config/options.py`) | Accepted risk (2026-11-15): the server only runs when the user starts it, it only accepts this bypass for local requests, and the Web API documentation and the setting tooltip warn about it (issue #522). | The default changes, the bypass also applies to non-local requests, or the documentation warning is removed. |
| DataLab CI `build_latest` jobs (`test_pyqt5.yml`, `test_pyqt6.yml`) clone the dependencies at `github.ref_name`, then at their default branch, unlike the `install-dependencies` action of the ordinary `build` jobs | Intended behavior (2026-11-15): these scheduled or manual jobs test the latest dependency versions. | The jobs run on pull requests, or the ordinary `build` jobs stop using the `install-dependencies` action. |
| Python 2 remote client (`doc/remotecontrol_py27.py`): automatic port discovery uses the configuration folder of the legacy guidata, which differs from the current one on macOS and on Linux with `XDG_CONFIG_HOME` | No further action (2026-11-15): the remote control documentation tells to pass the port explicitly. | The documented workaround is removed. |
| Broken links of old release notes (`doc/release_notes/release_0.09.md`, `release_0.11.md`), reported as 404 by `check_links.py` | Kept as historical references (2026-11-15). | Broken links in release notes of version 1.0 or later. |
| Wording of the DataLab home page and use-case pages about DataLab-Web: "the full feature set", "your data never leaves your machine" (`doc/index.rst`, `doc/contributing/roadmap.md`, `doc/use_cases/*.rst`) | Kept by editorial decision (2026-11-15): the ecosystem page and the AI Assistant documentation describe the limits (curated processing subset, optional remote AI provider). | These detailed explanations are removed or contradict the code. |
| DataLab AI Assistant: **Test connection** contacts the configured provider (`GET /models`) before any conversation | Documented behavior (2026-11-15): explicit user action, no workspace data sent. | The test sends workspace data or runs without user action. |

## Tracked by open issues

| Topic | Issue |
| --- | --- |
| DataLab-Web still pins Sigima 1.4.0 | DataLab-Platform/web#121 (milestone v0.12.0) |
| DataLab-Web: **Replace special values** missing from the processing catalog | DataLab-Platform/web#122 (milestone v0.12.0) |
| DataLab-Web: saving a DataLab Desktop workspace drops its History without warning | DataLab-Platform/web#123 |
| DataLab-Web: saving or exporting a large disk-mode workspace loads every array into memory | DataLab-Platform/web#124 |
| DataLab-Web: reloading plugins reimports the staged copy of local plugin files, not the edited file | DataLab-Platform/web#125 |
| Sigima: zero-crossing FWHM and pulse amplitude biased on smooth peaks, depending on the baseline level | DataLab-Platform/Sigima#138 |
| Sigima: no replay comparison rule for the phase spectrum contracts | DataLab-Platform/Sigima#139 (milestone v1.5.0) |
| DataLab Web API: object replacement seen by observers as an addition then a removal | DataLab-Platform/DataLab#573 |
| DataLab: no console output for `datalab --help` on Windows; error without installation hint when no Qt binding is installed | DataLab-Platform/DataLab#574 |
| DataLab: outdated `pip_list.txt` at the repository root | DataLab-Platform/DataLab#575 |
