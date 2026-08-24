---
name: medium-publisher
description: Prepare or publish eunomia.dev Markdown articles on Medium through faithful long-form syndication. Use when asked to paste, import, publish, QA, or record Medium posts for blog, tutorial, project, paper, release, or GitHub content. Preserves the English source title and body exactly apart from mechanical rendering fixes, with optional canonical settings, publication authorization, browser QA, follow-up, and media ledger updates.
---

# Medium Publisher

Prepare and publish Medium stories from canonical eunomia.dev content through
the Medium API. A request to publish or a queue item marked `排队` authorizes
creation; do not ask for another confirmation. Stop only for a draft or preview
task.

## Inputs

- Source Markdown path or source/canonical eunomia.dev URL.
- Source title, optional source subtitle, publication, tags, cover, or optional
  canonical URL.
- Optional GitHub link, paper link, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`.

## Platform Entry Points

- Account preflight: `GET https://api.medium.com/v1/me`
- Create post: `POST https://api.medium.com/v1/users/{authorId}/posts`
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

## API-First Platform Boundary

Publish through the documented Medium API by default, using `MEDIUM_API_KEY`
from the local `.env` or an approved secret-backed publisher. Never print,
record, or commit the token. Verify the authenticated account with `/v1/me`,
check the public profile for an exact-title duplicate, and create the story with
`POST /v1/users/{authorId}/posts`.

The Medium API is archived and unsupported, so treat every response as
untrusted until the public story has been checked. Do not use private or hidden
Medium endpoints. Use the visible web editor only for repairs the API cannot
perform, then repeat public-page QA.

If a creation request times out or returns an unknown result, check the public
profile for the exact title before retrying. Make at most one retry; if its
result is also unknown, stop, leave the queue item unfinished, and record the
exception instead of risking duplicate stories.

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
6. Include the visible article H1 inside the API `content`; the API `title`
   field controls listing and SEO metadata but does not render the story title.
   If Medium rejects otherwise valid Markdown with parser error `2012`, convert
   the prepared artifact to semantic HTML and publish with
   `contentFormat: "html"`. Keep no more than three tags because Medium ignores
   additional tags.

## Draft Archive

Before sending the Medium API request, write or update the Medium draft record under
`draft/media/YYYY-MM-DD/<source-slug>/medium.md` using the local date. For
canonical imports, this file may reference the source Markdown body instead of
duplicating it, but it must record the exact source title, optional source
subtitle, canonical
relationship when configured, GitHub/paper links, tags, source/project note if
useful, media choices, and QA state. For long-form posts, finish the
Medium-specific artifact locally before publishing. Use the web editor only for
supported metadata changes or repairs that remain necessary after creation.

## Browser QA

Before stopping for user confirmation or sending the API request, verify:

- canonical/import relationship is correct when configured
- the Medium body has not drifted from the canonical article except for
  necessary formatting/link/tag edits
- title matches the source exactly; subtitle is unchanged from the source or
  omitted unless the user requested one
- code blocks, images, embeds, links, and headings render cleanly
- tags are relevant and not spammy
- no confidential or unreleased claims appear
- the API request has not been sent

Before confirmed publishing, inspect the complete local upload artifact. After
confirmed publishing, open the public Medium URL and scroll through the rendered
story from top to bottom before updating the ledger. Verify images actually
load, title/subtitle are not
polluted by site suffixes, headings do not include empty artifacts, tables have
survived or have readable fallbacks, code blocks are not mangled by language
detection labels, canonical settings when configured, source/project links work,
and mobile/narrow rendering is usable when practical. If the public page exposes
a formatting issue, edit the published story through the web UI and repeat the
public-page check.

Medium API conversion or import is allowed to preserve the canonical body, but
it is not safe to trust blindly. Specifically check whether titles carried the
source site suffix such as `| eunomia`, whether Markdown tables were flattened into
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
public-page QA result in the draft record. In particular, check that API-created
stories visibly contain their H1 and that HTML conversion preserved images,
headings, code blocks, links, and readable table fallbacks.
