---
description: "Use when updating gettext translation catalogs, French Sphinx documentation, or reStructuredText icon markup."
applyTo: "**/*.po,doc/**/*.rst"
---

# Translation catalogs and documentation icons

- Leave `msgstr` empty for strings that do not need translation: technical
  identifiers (for example `roi_copy`), icon-only strings, and platform badges
  whose text is unchanged. Do not copy `msgid` into `msgstr` just for completeness.
- For translatable text containing `:octicon:`, check the original reST context:
  - Inside a `.. grid-item-card::` directive, especially its title, translate
    the text **without the icon**.
    Do not retain or reintroduce `:octicon:` roles or generated SVG in `msgstr`.
  - Outside `.. grid-item-card::` directives (headings, paragraphs, lists, dropdowns,
    etc.), translate the text **and preserve the icon role exactly**, including
    its name, arguments, and position.
- Sphinx may extract a card title as generated `<svg ...>...</svg>` followed by
  its text rather than as a literal `:octicon:` role. Identify the card from the
  reST source, not from the PO spelling alone. Keep `msgid` unchanged.
- Never remove icons from all translations indiscriminately. Existing catalogs
  may contain exceptions to this convention; verify them against the sources.
- After updating catalogs, compile them and build the French documentation.
  Check one translated card title and one translated heading or paragraph:
  the card translation is text-only; the non-card translation retains its icon.
