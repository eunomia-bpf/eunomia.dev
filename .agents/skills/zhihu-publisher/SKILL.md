---
name: zhihu-publisher
description: Prepare eunomia.dev Markdown articles for Zhihu publishing. Use when asked to create, import, paste, QA, or record a Zhihu draft or article from repository Markdown. Preserves a syndicated long-form source title and body apart from mechanical rendering fixes, with browser-editor workflow, Zhihu formatting checks, safe stop-before-publish behavior, and media ledger updates.
---

# Zhihu Publisher

Prepare a reviewed Zhihu draft from the canonical repository article. Stop at
the editor, preview, or publish-settings page unless the user explicitly
confirms final publishing.

## Inputs

- Source Markdown path or topic.
- Intended language and title, if the user specifies them.
- Optional target column, cover, tags, source URL, GitHub link, or paper link.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, and recent `docs/blog/posts/`
entries before asking the user.

## Platform Entry Points

- Editor: <https://zhuanlan.zhihu.com/write>
- Creator center: <https://www.zhihu.com/creator>
- Observed profile: <https://www.zhihu.com/people/yun-wei-64-11>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing Zhihu-native framing,
title, opening, "想法" format, or promotion/link balance. Load
`references/ai-works.md` for Zhihu AI Works fields, campaign notes, project
material, and reusable copy.

## Browser-Only Platform Boundary

Do not directly access Zhihu APIs, internal endpoints, or background HTTP
interfaces under any circumstances. All verification, drafting, QA, screenshots,
and ledger evidence must come from normal browser interactions that a regular
logged-in user can perform: navigating pages, scrolling profile/article lists,
clicking visible controls, reading rendered page content, using the editor UI,
and capturing screenshots.

## Draft Preparation

1. Read the canonical Chinese source and extract title, summary, tags, images,
   code blocks, source URL for the ledger, GitHub links, and paper links.
2. Build a Zhihu copy in canonical syndication mode:
   - remove YAML front matter
   - preserve the article body by default
   - preserve the source title and first paragraph exactly
   - convert relative images to checked public URLs or prepare manual image
     upload
   - convert tables, formulas, Mermaid, footnotes, or complex HTML only when the
     target editor cannot render them faithfully
3. Preserve links already present in the source. A visible eunomia.dev
   original/canonical note is optional and is not added by default.
4. Keep the opening, section order, claims, examples, and conclusion unchanged.
   If the source needs a content fix, update it first or skip syndication.
   Rewrite, translate, shorten, expand, reorder, or split only when the user
   explicitly asks for that specific publication.

## Draft Archive

Before opening the Zhihu editor, write or update the Zhihu draft record under
`draft/media/YYYY-MM-DD/<source-slug>/zhihu.md` using the local date. For
unchanged Chinese canonical syndication, the file may reference the source body
instead of duplicating it, but it must record the exact title, source URL for
the ledger if known, GitHub/paper links, column/tags if known, source/project
note if useful, media choices, and QA state.

## Editor Workflow

1. Open <https://zhuanlan.zhihu.com/write>.
2. Fill the title field, observed as `请输入标题（最多 100 个字）`.
3. For long-form posts, prefer a local `.md` or `.docx` artifact prepared before
   opening the editor. Use Zhihu's visible document import path when available
   (toolbar more/menu import); paste only for short text or when import is not
   available.
4. Avoid post-paste surgical repair in the Zhihu editor. If a long draft needs
   structural fixes, regenerate the local artifact and re-import into a fresh
   draft when practical.
5. Verify visually:
   - heading hierarchy is preserved
   - code blocks are readable
   - tables do not collapse
   - images render
   - links point to intended GitHub, docs, project, or paper sources
   - Treat visible public-page DOM inspection as sufficient QA. If
     `Page.captureScreenshot` times out, do not retry it; verify the title,
     structure, code blocks, tables, image loading, and links through the
     visible DOM/page instead.
6. In the publish/settings flow, choose a relevant question when Zhihu offers a
   question selector. Verify the selected question is topical before final
   publish; do not leave an unrelated default question selected.
7. Set cover and column inclusion only when the user requested them or the
   choice is obvious from prior posts.
8. Stop at preview, the visible `发布` button, or the publish-settings page
   unless the user explicitly authorized publishing the current item.

## Content Strategy

Zhihu works best when the article reads like an explanatory essay rather than a
release note. For existing Chinese long-form eunomia.dev posts, preserve the
source title exactly and keep the body substantively unchanged. Platform work is
limited to rendering, cover, column, tags, question selection, and preview
quality. If the source is not already a usable Chinese essay, skip it or improve
the source before syndication.

Optimize for the maintainer's personal technical account brand: clear taste,
useful explanation, research and engineering credibility, and discussion from
readers who care about eBPF systems, AI agents, and runtime observability. Keep
GitHub, docs, project, or paper links as sources or extended reading. Use an
eunomia.dev link only when it helps the reader, not as a default canonical
requirement.

## Safety Boundary

Do not automate:

- final `发布` without explicit user authorization for the current item
- direct Zhihu API access, internal endpoint reads, or browser-hidden data fetches
- column submission unless the user names the exact column
- deleting drafts or changing account settings
- security, privacy, phone-verification, or CAPTCHA prompts
- likes, comments, follows, reposts, or private messages

## Ledger Update

After a confirmed publish, update `.github/publisher/media/published.md` with
title, source path, Zhihu URL, date, tags or column, and formatting fixes. Remove
or update the matching row in `.github/publisher/media/not-published.md`.

Before closing the publishing task, run a platform-lessons pass. Add any new,
reproducible editor failure and its verified workaround to this skill or its
references so the next publish does not repeat it.

Keep screenshots and observed UI notes under `.github/publisher/media/`.
