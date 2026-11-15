---
description: "Use when updating gettext translation catalogs, French Sphinx documentation, or reStructuredText icon markup."
applyTo: "**/*.po,doc/**/*.rst"
---

# Translation catalogs and documentation icons

- Leave `msgstr` empty for strings that do not need translation: technical
  identifiers (for example `roi_copy`), icon-only strings, and platform badges
  whose text is unchanged. Do not copy `msgid` into `msgstr` just for completeness.
- For translatable text containing `:octicon:`, check the original reST context:
  - Inside a `.. grid-item-card::` title, use the `|card-icon-NAME|` substitutions defined in `doc/_card_icons.txt` and include that file in the page. These keep gettext keys stable across sphinx-design versions. Preserve the substitution and its position in `msgstr`, including its `|` delimiters: gettext extracts `card-icon-NAME` without the delimiters, e.g. `msgid "card-icon-pulse  Spectroscopy"` becomes `msgstr "|card-icon-pulse|  Spectroscopie"`. Leave `msgstr` empty for untranslated titles.
  - Outside `.. grid-item-card::` directives (headings, paragraphs, lists, dropdowns,
    etc.), translate the text **and preserve the icon role exactly**, including
    its name, arguments, and position.
- Direct `:octicon:` roles in card titles produce version-dependent gettext keys: generated SVG with sphinx-design 0.7, plain text with 0.8. Use the card substitutions instead. Never copy generated SVG into `msgstr`: Sphinx renders it as escaped text. When migrating a title, update its `msgid` to the freshly extracted key.
- Never remove icons from all translations indiscriminately. Existing catalogs
  may contain exceptions to this convention; verify them against the sources.
- After updating catalogs, compile them and build the French documentation. Check one translated card title and one translated heading or paragraph: both translations retain their icons, with no escaped SVG or literal role markup in the rendered output.
