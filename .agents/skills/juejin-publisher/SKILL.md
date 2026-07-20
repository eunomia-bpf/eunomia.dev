---
name: juejin-publisher
description: Prepare eunomia.dev Markdown articles for Juejin publishing. Use when asked to create, paste, QA, or record a Juejin draft or article from repository Markdown. Preserves a syndicated long-form source title and body apart from mechanical Markdown/rendering fixes, with browser-editor workflow, category and tag selection, safe stop-before-publish behavior, and media ledger updates.
---

# Juejin Publisher

Prepare a reviewed Juejin draft from the canonical repository article, choose
appropriate technical categories and tags, and stop before final publishing
unless the user explicitly confirms it.

## Inputs

- Source Markdown path or topic.
- Intended title, language, and audience, if specified.
- Optional cover, category, tags, source URL, GitHub link, or paper link.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`
entries before asking the user.

## Platform Entry Points

- Editor: <https://juejin.cn/editor/drafts/new>
- Observed profile: <https://juejin.cn/user/4288563097635144>
- Observed article list: <https://juejin.cn/user/4288563097635144/posts>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing Juejin-native framing,
category/tags, tutorial-vs-series shape, or promotion/link balance.

## Browser-Only Platform Boundary

Do not directly access Juejin APIs, internal endpoints, or background HTTP
interfaces under any circumstances. All verification, drafting, QA, screenshots,
and ledger evidence must come from normal browser interactions that a regular
logged-in user can perform: navigating pages, scrolling profile/article lists,
clicking visible controls, reading rendered page content, using the editor UI,
and capturing screenshots.

## Draft Preparation

1. Read the canonical Chinese source and extract title, summary, tags, images,
   code blocks, source URL for the ledger, GitHub links, and paper links.
2. Build a Juejin copy in canonical syndication mode:
   - remove YAML front matter
   - preserve the article body by default
   - convert relative images to checked public URLs or prepare editor upload
   - ensure code fences have language labels
   - preserve the source title exactly
   - preserve links already present in the source
3. Keep the opening, section order, claims, examples, and conclusion unchanged.
   If the source needs a content fix, update it first or skip syndication.
   Rewrite, translate, shorten, expand, reorder, or split only when the user
   explicitly asks for that specific publication.

## Draft Archive

Before opening the Juejin editor, write or update the Juejin draft record under
`draft/media/YYYY-MM-DD/<source-slug>/juejin.md` using the local date. For
unchanged Chinese canonical syndication, the file may reference the source body
instead of duplicating it, but it must record the exact title, source URL for
the ledger if known, GitHub/paper links, category/tags, source/project note if
useful, media choices, and QA state.

## Editor Workflow

1. Open <https://juejin.cn/editor/drafts/new>.
2. Fill the title field, observed as `输入文章标题...`.
3. For long-form posts, finish the Juejin-specific Markdown artifact locally
   before opening the editor. Paste or import that final artifact; do not use
   the platform editor for large rewrites or link-heavy tail-note repair.
4. Use `预览` to scan headings, images, links, code blocks, and table layout.
5. Click `发布` only to inspect publish settings when needed.
6. Choose category and tags carefully:
   - eBPF tutorials: `后端`, `Linux`, `开源`, `云原生`, `架构`
   - AI agent or runtime posts: `人工智能`, `AIGC`, `后端`, `架构`, `安全`
   - GPU observability posts: `人工智能`, `后端`, `架构`, `Linux`, `性能优化`
7. Stop before final `确定并发布` unless the user explicitly authorized publishing
   the current item.

For images, verify the exact final URL used in Markdown before publishing. Do
not assume `imgs/...` can be converted by guessing an eunomia.dev article path;
that path may return 404. Use actual rendered image URLs, stable GitHub raw URLs
for public repository images, or upload images through the Juejin web editor,
then verify the preview and public page.

## Content Strategy

Juejin-native short posts and new articles can use immediately useful technical
framing. This guidance does not apply to syndicated long-form content. For an
existing Chinese long-form eunomia.dev post, preserve the source title exactly
and keep the body substantively unchanged. Only fix Markdown/rendering and set
category, tags, cover, and summary metadata. Do not split a source article into
a series by default.

Optimize for the maintainer's personal technical account brand and practical
developer trust, not only for search ranking or traffic back to eunomia.dev.
Preserve GitHub, tutorial, docs, or paper links already in the source. A visible
eunomia.dev canonical/source note is optional and is not added to the body by
default.

## Safety Boundary

Do not automate:

- final `确定并发布` without explicit user authorization for the current item
- direct Juejin API access, internal endpoint reads, or browser-hidden data fetches
- sign-in, phone verification, or CAPTCHA
- `去签到`, likes, follows, comments, reposts, or private messages
- account settings or monetization settings
- deleting drafts

## Ledger Update

After a confirmed publish, update `.github/publisher/media/published.md` with
title, source path, Juejin URL, date, category, tags, and formatting fixes.
Remove or update the matching row in `.github/publisher/media/not-published.md`.

Before closing the publishing task, run a platform-lessons pass. Add any new,
reproducible editor failure and its verified workaround to this skill or its
references so the next publish does not repeat it.

Keep screenshots and observed UI notes under `.github/publisher/media/`.
