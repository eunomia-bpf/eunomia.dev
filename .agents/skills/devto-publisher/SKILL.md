---
name: devto-publisher
description: Prepare or publish eunomia.dev Markdown articles on DEV Community through faithful long-form syndication. Use when asked to paste, import, publish, QA, or record DEV.to posts for blog, tutorial, project, paper, release, or GitHub content. Preserves the English source title and body apart from mechanical Markdown/rendering fixes, with optional canonical_url settings, publication authorization, browser QA, follow-up, and media ledger updates.
---

# DEV.to Publisher

Prepare and publish DEV Community articles from canonical eunomia.dev content
through the DEV API. A request to publish or a queue item marked `排队`
authorizes creation; do not ask for another confirmation. Stop only for a draft
or preview task.

## Inputs

- Source Markdown path or source/canonical eunomia.dev URL.
- Source title, tags, series metadata, cover image, or optional canonical URL.
- Optional GitHub link, paper link, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`.

## Platform Entry Points

- Account preflight: `GET https://dev.to/api/users/me`
- Duplicate check: `GET https://dev.to/api/articles/me/all`
- Create article: `POST https://dev.to/api/articles`
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

## API-First Platform Boundary

Publish through the documented DEV API by default, using `DEV_TO_API_KEY` from
the local `.env` or an approved secret-backed publisher. Never print, record,
or commit the token. Verify the authenticated account, query the author's full
article list for an exact-title duplicate, then create the article with
`POST /api/articles`. Do not use hidden or internal DEV endpoints.

The visible web editor is a repair surface for behavior the API cannot correct,
not the default submission path. Every API-created article still requires a
normal visible-browser check of the complete public page.

## Draft Preparation

1. Read the canonical English source and record the source URL, GitHub links,
   and paper links when known.
2. Convert frontmatter to DEV/Jekyll-style fields when useful. Include
   `canonical_url` when known and convenient, but do not add a visible body
   source link just to satisfy a checklist.
3. Use H2 as the highest body heading because the post title is the H1.
4. Preserve the source title exactly and keep the body substantively unchanged.
   Only fix frontmatter, heading levels, code fences, image URLs/uploads, tables,
   formulas, embeds, links, tags, and DEV-specific rendering.
5. Preserve GitHub/project/paper links already in the source. Add body text or
   links only after the same source change or an explicit user request.
6. If the source is not suitable for DEV, skip it or fix the source first.
   Rewrite, translate, shorten, expand, reorder, or split it only when the user
   explicitly asks for that specific publication.
7. Strip local YAML front matter and internal HTML comments from the API
   `body_markdown`. Send the exact title separately, keep H2 as the highest body
   heading, include up to four accepted DEV tags, and set `canonical_url` only
   when it is useful and available; it is optional.

## Draft Archive

Before sending the DEV API request, write or update the DEV draft record under
`draft/media/YYYY-MM-DD/<source-slug>/devto.md` using the local date. Include
the exact title, description, tags, optional `canonical_url`, source body path
or paste-ready body, GitHub/paper links, series/cover choices, and QA state.
For long-form posts, finish this API-ready Markdown artifact locally before
publishing. Use the web editor only for supported metadata changes or repairs
that remain necessary after creation.

## Browser QA

Before stopping for user confirmation or sending the API request, verify:

- title matches the source exactly; description, tags, cover, and optional
  canonical URL accurately reflect the unchanged source
- the DEV body has not drifted from the canonical article except for necessary
  Markdown/frontmatter/rendering edits
- headings start at H2 and code fences have language labels
- images render and have descriptions where supported
- links and embeds resolve
- preview is readable and self-contained
- the API request has not been sent

Before confirmed publishing, inspect the complete local upload artifact. After
confirmed publishing, open the public DEV URL and inspect the rendered post
from top to bottom before updating the ledger. Verify
canonical field when configured, source/project notes when present, title, tags,
image loading, H2/H3 hierarchy, tables, code fences, link targets, embeds, and
narrow rendering when practical. If the public page reveals duplicated source
notes, wrong tags, broken images, heading artifacts, or mangled code blocks, edit
the published post through the web UI and repeat the public-page check.

DEV tags must be verified from the selected-tag chips after editing. Do not
assume a desired tag exists or was accepted just because it was typed into the
tag box; if the editor rejects a tag, choose a supported nearby tag and record
the fallback. When `canonical_url` is set, DEV already displays its own
"Originally published" notice, so avoid adding a duplicate manual source note
at the end unless the user explicitly asks for one.

For images, verify the exact final URL that will appear in the DEV Markdown.
Do not assume a relative `imgs/...` path becomes
`https://eunomia.dev/<article>/imgs/...`; that guessed path can 404 even when
the canonical article renders locally. Check each external image URL with a
browser or HEAD request before saving. If the eunomia.dev URL is not directly
200, use the actual rendered image URL, a stable GitHub raw URL for public repo
images, or upload the image through the DEV web editor, then re-check the
public page after lazy loading.

## Follow-Up

After confirmed publish and public-page QA, capture the DEV URL. Monitor
comments and notifications only when the user asks or follow-up was part of the
task. Draft answers with reproducible details and move long-lived issues to
GitHub.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, canonical URL, DEV URL, date, tags/series, media, and follow-up
notes.

Before final completion, add any DEV-specific issue encountered during this
session to this skill or `references/platform-preferences.md`, then record the
public-page QA result in the draft record.
