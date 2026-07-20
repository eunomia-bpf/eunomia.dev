---
name: medium-publisher
description: Prepare eunomia.dev Markdown articles for Medium drafts, imports, and faithful long-form syndication. Use when asked to paste, import, QA, or record Medium posts for blog, tutorial, project, paper, release, or GitHub content. Preserves the English source title and body exactly apart from mechanical rendering fixes, with optional canonical settings, browser QA, safe stop-before-publish behavior, follow-up, and media ledger updates.
---

# Medium Publisher

Prepare Medium drafts from canonical eunomia.dev content and stop before final
publishing unless the user explicitly confirms it.

## Inputs

- Source Markdown path or source/canonical eunomia.dev URL.
- Source title, optional source subtitle, publication, tags, cover, or optional
  canonical URL.
- Optional GitHub link, paper link, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`.

## Platform Entry Points

- New story: <https://medium.com/new-story>
- Import story: <https://medium.com/p/import>
- Stories: <https://medium.com/me/stories/drafts>
- Notifications: <https://medium.com/me/notifications>

Use a browser surface with the logged-in session when UI work is required.

## References

Load `references/platform-preferences.md` when choosing canonical syndication
settings, light Medium metadata/rendering adaptations, title/subtitle/cover,
tags, browser QA, promotion balance, or follow-up. Do not load broad strategy
drafts for routine publishing unless the user asks for campaign or
content-platform planning.

## Browser-Only Platform Boundary

Do not directly access Medium APIs, internal endpoints, or background HTTP
interfaces. All drafting, QA, screenshots, and ledger evidence must come from
normal browser interactions. Use the Medium web import/editor UI and visible
submit buttons for publication; Medium publish APIs are not part of the default
workflow.

## Draft Preparation

1. Read the canonical English source and record the source URL, GitHub links,
   and paper links when known.
2. Preserve the source title exactly and keep the body substantively unchanged.
3. Make only mechanical adaptations: remove site front matter or a duplicate
   H1, convert heading levels, repair image URLs/uploads, and preserve readable
   code, tables, formulas, embeds, and links. Set Medium tags, publication,
   cover, and other metadata without changing the article.
4. Preserve GitHub, project, paper, and source links already in the article. Add
   body text or links only after the same source change or an explicit user
   request.
5. If the source is not suitable for Medium, skip it or fix the source first.
   Rewrite, translate, shorten, expand, reorder, or split it only when the user
   explicitly asks for that specific publication.

## Draft Archive

Before opening the Medium editor, write or update the Medium draft record under
`draft/media/YYYY-MM-DD/<source-slug>/medium.md` using the local date. For
canonical imports, this file may reference the source Markdown body instead of
duplicating it, but it must record the exact source title, optional source
subtitle, canonical
relationship when configured, GitHub/paper links, tags, source/project note if
useful, media choices, and QA state. For long-form posts, finish the
Medium-specific artifact or import checklist locally before opening Medium; use
the web editor/import UI for import, settings, preview, and QA rather than
structural repair.

## Browser QA

Before stopping for user confirmation, verify:

- canonical/import relationship is correct when configured
- the Medium body has not drifted from the canonical article except for
  necessary formatting/link/tag edits
- title matches the source exactly; subtitle is unchanged from the source or
  omitted unless the user requested one
- code blocks, images, embeds, links, and headings render cleanly
- tags are relevant and not spammy
- no confidential or unreleased claims appear
- the visible final `Publish` action has not been clicked

Before confirmed publishing, scroll through the full imported/editor story in
the browser preview or editor surface. After confirmed publishing, open the
public Medium URL and scroll through the rendered story from top to bottom
before updating the ledger. Verify images actually load, title/subtitle are not
polluted by site suffixes, headings do not include empty artifacts, tables have
survived or have readable fallbacks, code blocks are not mangled by language
detection labels, canonical settings when configured, source/project links work,
and mobile/narrow rendering is usable when practical. If the public page exposes
a formatting issue, edit the published story through the web UI and repeat the
public-page check.

Medium import is allowed to preserve the canonical body, but it is not safe to
trust blindly. Specifically check whether imported titles carried the source
site suffix such as `| eunomia`, whether Markdown tables were flattened into
loose paragraphs, whether image captions are empty placeholders, and whether
code block language labels appeared as prose. If Medium cannot preserve a table
cleanly, replace that table with a readable list or compact prose fallback in
the web editor before publishing.

## Follow-Up

After confirmed publish and public-page QA, capture the Medium URL. Monitor
responses, highlights, private notes, and publication feedback only when the
user asks or follow-up was part of the task. Draft replies that add context
rather than sell.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, canonical URL, Medium URL, date, tags/publication, media, and
follow-up notes.

Before final completion, add any Medium-specific issue encountered during this
session to this skill or `references/platform-preferences.md`, then record the
public-page QA result in the draft record.
