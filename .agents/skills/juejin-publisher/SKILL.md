---
name: juejin-publisher
description: Prepare eunomia.dev Markdown articles for Juejin publishing. Use when asked to create, review, paste, QA, or record a Juejin draft or article for content from docs/blog/posts, docs/blogs, docs/tutorials, or other repository Markdown sources. Defaults long-form posts to Chinese canonical syndication rather than full rewrites, with browser-editor workflow, Juejin Markdown preview checks, category and tag selection, safe stop-before-publish behavior, and media ledger updates.
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
   - micro-tune the title or intro only when needed for the same developer
     promise
   - add a short GitHub/project/paper note near the end only when useful
3. Rewrite the body only when the user asks, the source is English-only, or a
   concrete quality problem blocks publication.

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
7. Stop before final `确定并发布`.

For images, verify the exact final URL used in Markdown before publishing. Do
not assume `imgs/...` can be converted by guessing an eunomia.dev article path;
that path may return 404. Use actual rendered image URLs, stable GitHub raw URLs
for public repository images, or upload images through the Juejin web editor,
then verify the preview and public page.

## Content Strategy

Juejin readers reward immediately useful technical framing. Put the practical
payoff in the title or first paragraph, use diagrams and command output only
when they advance the tutorial, and prefer one article per concrete technique.
For existing Chinese long-form eunomia.dev posts, preserve the canonical body
and adapt only title, images, Markdown rendering, category, tags, and useful
source/project links. For large docs, split into a series and link the full
tutorial only when that helps the reader.

Optimize for the maintainer's personal technical account brand and practical
developer trust, not only for search ranking or traffic back to eunomia.dev.
Keep GitHub, tutorial, docs, or paper links as sources or extended reading, but
make the Juejin version stand alone as a useful platform-native article. A
visible eunomia.dev canonical/source note is optional.

## Safety Boundary

Do not automate:

- final `确定并发布`
- direct Juejin API access, internal endpoint reads, or browser-hidden data fetches
- sign-in, phone verification, or CAPTCHA
- `去签到`, likes, follows, comments, reposts, or private messages
- account settings or monetization settings
- deleting drafts

## Ledger Update

After a confirmed publish, update `.github/publisher/media/published.md` with
title, source path, Juejin URL, date, category, tags, and formatting fixes.
Remove or update the matching row in `.github/publisher/media/not-published.md`.

Keep screenshots and observed UI notes under `.github/publisher/media/`.
