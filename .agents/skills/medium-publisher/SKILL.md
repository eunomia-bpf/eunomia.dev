---
name: medium-publisher
description: Prepare eunomia.dev Markdown articles for Medium drafts, imports, and canonical syndication. Use when asked to adapt, paste, import, QA, or record Medium posts for blog, tutorial, project, paper, release, or GitHub content. Defaults long-form posts to English canonical syndication rather than full rewrites, with canonical URL checks, title/subtitle/cover QA, safe stop-before-publish behavior, response/private-note follow-up, and media ledger updates.
---

# Medium Publisher

Prepare Medium drafts from canonical eunomia.dev content and stop before final
publishing unless the user explicitly confirms it.

## Inputs

- Source Markdown path or canonical eunomia.dev URL.
- Intended title, subtitle, publication, tags, cover, or canonical URL.
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

1. Read the canonical English source and confirm the canonical URL.
2. Default to import/syndication that preserves the article body.
3. Make only light adaptations: remove site front matter, micro-tune title or
   subtitle without changing the promise, fix images/headings/code/links, add
   relevant tags, and prepare alt text or image credits.
4. Add or preserve GitHub, project, and paper links as evidence or extended
   reading, not as the point of the story.
5. Rewrite the body only when the user asks, the source is not suitable for
   Medium, or a concrete quality problem blocks publication.

## Draft Archive

Before opening the Medium editor, write or update the Medium draft record under
`draft/media/YYYY-MM-DD/<source-slug>/medium.md` using the local date. For
canonical imports, this file may reference the source Markdown body instead of
duplicating it, but it must record the exact Medium title/subtitle, canonical
URL, GitHub/paper links, tags, source note, media choices, and QA state.

## Browser QA

Before stopping for user confirmation, verify:

- canonical link/import relationship is correct when syndicating
- the Medium body has not drifted from the canonical article except for
  necessary formatting/link/tag edits
- title, subtitle, and cover accurately represent the story
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
detection labels, canonical/source links work, and mobile/narrow rendering is
usable when practical. If the public page exposes a formatting issue, edit the
published story through the web UI and repeat the public-page check.

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
