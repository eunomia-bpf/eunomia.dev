---
name: devto-publisher
description: Prepare eunomia.dev Markdown articles for DEV Community drafts and canonical syndication. Use when asked to adapt, paste, import, QA, or record DEV.to posts for blog, tutorial, project, paper, release, or GitHub content. Defaults long-form posts to English canonical syndication rather than full rewrites, with Markdown/frontmatter conversion, canonical_url checks, safe stop-before-publish behavior, comment follow-up, and media ledger updates.
---

# DEV.to Publisher

Prepare DEV Community drafts from canonical eunomia.dev content and stop before
final publishing unless the user explicitly confirms it.

## Inputs

- Source Markdown path or canonical eunomia.dev URL.
- Intended title, tags, series, cover image, or canonical URL.
- Optional GitHub link, paper link, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`.

## Platform Entry Points

- New post: <https://dev.to/new>
- Dashboard: <https://dev.to/dashboard>
- Editor guide: <https://dev.to/p/editor_guide>
- Notifications: <https://dev.to/notifications>

Use a browser surface with the logged-in session when UI work is required.

## References

Load `references/platform-preferences.md` when choosing canonical syndication
settings, light DEV metadata/rendering adaptations, frontmatter, tags, series
shape, browser QA, promotion balance, or follow-up. Do not load broad strategy
drafts for routine publishing unless the user asks for campaign or
content-platform planning.

## Browser-Only Platform Boundary

Do not directly access DEV APIs, internal endpoints, or background HTTP
interfaces. All drafting, QA, screenshots, and ledger evidence must come from
normal browser interactions.

## Draft Preparation

1. Read the canonical English source and confirm the canonical URL.
2. Convert frontmatter to DEV/Jekyll-style fields when useful, including
   `canonical_url`.
3. Use H2 as the highest body heading because the post title is the H1.
4. Preserve the article body by default; only fix headings, code fences,
   images, links, tags, and DEV-specific Markdown rendering.
5. Use GitHub/project links as implementation sources and next steps.
6. Rewrite the body only when the user asks, the source is not developer
   relevant enough for DEV, or a concrete quality problem blocks publication.

## Draft Archive

Before opening the DEV editor, write or update the DEV draft record under
`draft/media/YYYY-MM-DD/<source-slug>/devto.md` using the local date. Include
the exact title, description, tags, `canonical_url`, source body path or
paste-ready body, GitHub/paper links, series/cover choices, and QA state.

## Browser QA

Before stopping for user confirmation, verify:

- title, description, tags, cover, and canonical URL are correct
- the DEV body has not drifted from the canonical article except for necessary
  Markdown/frontmatter/rendering edits
- headings start at H2 and code fences have language labels
- images render and have descriptions where supported
- links and embeds resolve
- preview is readable and self-contained
- the visible final publish action has not been clicked

## Follow-Up

After confirmed publish, capture the DEV URL. Monitor comments and
notifications only when the user asks or follow-up was part of the task. Draft
answers with reproducible details and move long-lived issues to GitHub.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, canonical URL, DEV URL, date, tags/series, media, and follow-up
notes.
