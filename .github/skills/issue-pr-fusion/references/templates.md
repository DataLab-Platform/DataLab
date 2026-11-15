# Templates

All texts are in English. Replace the `<...>` placeholders and drop the sections that do not apply.

## Issue Body

Title: the symptom from the user's point of view (bug) or the capability (enhancement), e.g. "Truncated range fields and noisy slider values in dataset editors".

```markdown
## Description

<What the user sees or needs, in which context. For an enhancement: the benefits.>

## Steps to reproduce

1. <Step>
2. <Step>

## Expected behavior

<What should happen.>

## Cause

<Root cause when known, with file and function names.>

## Proposed change

<Approach agreed in the plan; related issues or repositories (owner/repo#N).>
```

## Commit Message

Wrap the body at 72 columns (Git convention). `Assisted-by` is the last line and the only AI trailer.

```text
<type>(<scope>): <summary of the user-visible change>

<Why: symptom and cause.> <What: the change.>

- <Detail of a change>
- Tests, documentation, release notes and French translation

Closes #<N>

Assisted-by: <model>
```

## PR Body

Title: the conventional commit subject (single commit) or a summary in the same form.

```markdown
Closes #<N>

## Summary

<Symptom or need, and cause, in a few sentences.>

## Changes

- **Fix**: <change>
- **Regression test**: `<test name>` (`<path>`), failing before the fix with `<message>`
- **Documentation**: <pages>; release notes and French translation

## Validation

- Automated: <tests run and results>; Ruff clean on the changed files
- Manual: <scenario validated by the user>

## Out of scope

- <owner/repo#N: defect found on the way, not fixed here>

<Release note statement, e.g. "No release note: the regression was introduced after the last release.">

Related: <owner/repo#N (PR of the same change in another repository)>

Assisted-by: <model>
```

## Issue Closing Comment

When the merge did not close the issue (PR base other than the default branch):

```text
Fixed by #<PR> (merged into develop).
```

## Approach Comment

With the `issue comment` option or for an external reporter, after the merge:

```markdown
Thanks for the report. This is fixed by #<PR>, which will be part of <next version>.

<Cause in one or two sentences.> <Chosen approach and why, alternatives discarded.> <Any user-visible change of behavior.>
```
