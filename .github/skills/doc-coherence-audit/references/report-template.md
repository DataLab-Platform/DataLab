# Subagent Prompt and Report Templates

## Severity Scale

| Severity | Meaning | Examples |
| --- | --- | --- |
| High | The user is misled into a wrong result, a failure or a security risk; or the audit reveals a code defect | Broken command or example, documented feature absent from the menus, misleading security default, wrong algorithm description |
| Medium | Wrong but recoverable information | Wrong menu path or label, wrong default value, stale version, missing documentation of an existing feature, dead link to the platform |
| Low | Cosmetic or historical | Typo in a label, ambiguous wording, outdated screenshot with correct content, dead link in old release notes |

A finding is **verified** only when it cites both sides: the documentation location (`path:line` and quotation) and the code fact (`path:line`, introspected value, command output). Without the code fact, it is a question for the "To be confirmed" list, not a finding.

## Subagent Prompt (one per perimeter)

Fill the `<...>` fields from [perimeters.md](./perimeters.md) and the preflight table, then launch the subagents in parallel (`general-purpose` agent type: they must run Python).

```text
Documentation/code consistency audit, perimeter <Pn. title>. Read-only: do not modify, stage or commit anything in any repository; if a tool writes into a repository (e.g. scripts/print_datalab_menus.py), restore the file with git afterwards, only if it was clean before.

Repositories (Windows, PowerShell): <repo: path, branch, commit> ...
Environment: <python interpreter and PYTHONPATH from the preflight>; set $env:PYTHONIOENCODING="utf-8" and $env:QT_QPA_PLATFORM="offscreen"; scratch folder <%TEMP%\docaudit\scratch_pn>, to delete at the end.
<Optional, incremental mode: only the files changed since <ref> (list), plus the pages describing the changed code (list).>
<Optional: findings of the previous audit for this perimeter, to check whether each one is fixed, still present or changed.>

Documentation to read in full: <list>
Code to confront: <list>
Mandatory checks: <list>

For each factual statement of the documentation (menu path, label, default value, signature, command, file name, version, limit, behavior), check it against the code, preferably by introspection or execution. Run the examples and snippets that can run headless.

Report, in <user language>, Markdown without hard wrapping:
1. Findings: table "ID | Severity (High/Medium/Low) | Documentation (path:line, quotation) | Code fact (path:line or command and output) | Finding | Suggested fix (doc or code)". ID format <Pn>-<number>.
2. Coverage: what was checked without anomaly, with counts (e.g. "43 proxy methods").
3. Not verifiable: statements that could not be checked, and why.
4. Facts useful to other perimeters (versions, limits, protocols, names) for the cross-project checks.
Stop when every listed file is read and every check is done; do not fix anything.
```

## Report Structure

Write the report in the user's language, without hard wrapping, to `files/doc-audit-<YYYY-MM-DD>.md` in the session folder:

```markdown
# Documentation consistency audit: <projects>

Date: <date>. Audited branches and commits: <repo `branch` (`commit`, version)>, ... Mode: <full | incremental since <ref>>. Previous audit: <file or "none">.

## 1. Scope and method
- Automated checks: <builds, translations, links, dependencies>, with their commands.
- Semantic checks: <perimeters, subagents, introspection>.
- Read-only: `git status` identical before and after in every repository.

## 2. Summary
| Perimeter | High | Medium | Low |
| --- | --- | --- | --- |
| ... | | | |
| **Total** | | | |

Delta with the previous audit: <fixed / still present / new findings>, when a previous report exists.

## 3. What is consistent
<Highlights of the coverage sections.>

## 4. Priorities
### 4.1 Code defects revealed by the audit
### 4.2 Security and misleading behavior
### 4.3 Broken examples and commands
### 4.4 Outdated tutorials and screenshots

## 5. Cross-project inconsistencies
| # | Topic | Projects and files | Finding |

## 6. Automated check findings
| Severity | Location | Finding |

## 7. Proposed follow-up
| Group | Repository | Findings (IDs) | Proposed issue title | Type (doc / code / decision) |

## Appendix: detailed reports per perimeter
<One section per perimeter: findings table and coverage, after deduplication.>
```
