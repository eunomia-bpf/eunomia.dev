---
name: zhihu-publisher
description: Prepare eunomia.dev Markdown articles for Zhihu publishing. Use when asked to create, review, paste, QA, or record a Zhihu draft or article for content from docs/blog/posts, docs/blogs, docs/tutorials, or other repository Markdown sources. Defaults long-form posts to Chinese canonical syndication rather than full rewrites, with browser-editor workflow, Zhihu formatting checks, safe stop-before-publish behavior, and media ledger updates.
---

# Zhihu Publisher

Prepare a reviewed Zhihu draft from the canonical repository article. Stop at
the editor, preview, or publish-settings page unless the user explicitly
confirms final publishing.

## Inputs

- Source Markdown path or topic.
- Intended language and title, if the user specifies them.
- Optional target column, cover, tags, or canonical eunomia.dev URL.

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
   code blocks, and canonical URL.
2. Build a Zhihu copy in canonical syndication mode:
   - remove YAML front matter
   - preserve the article body by default
   - micro-tune the Chinese title only if it strengthens the same reader promise
   - keep the first paragraph as the hook unless it has a concrete problem
   - convert relative images to public `https://eunomia.dev/...` URLs or
     prepare manual image upload
   - simplify tables, formulas, Mermaid, footnotes, and complex HTML before
     paste or import
3. Add a short canonical-source/GitHub/paper note near the end when appropriate.
4. Rewrite the body only when the user asks, the source is English-only, or a
   concrete quality problem blocks publication.

## Draft Archive

Before opening the Zhihu editor, write or update the Zhihu draft record under
`draft/media/YYYY-MM-DD/<source-slug>/zhihu.md` using the local date. For
unchanged Chinese canonical syndication, the file may reference the source body
instead of duplicating it, but it must record the exact title, canonical URL,
GitHub/paper links, column/tags if known, source note, media choices, and QA
state.

## Editor Workflow

1. Open <https://zhuanlan.zhihu.com/write>.
2. Fill the title field, observed as `请输入标题（最多 100 个字）`.
3. Paste or import the article body into `请输入正文`.
4. Verify visually:
   - heading hierarchy is preserved
   - code blocks are readable
   - tables do not collapse
   - images render
   - links point to the intended canonical sources
5. Set cover and column inclusion only when the user requested them or the
   choice is obvious from prior posts.
6. Stop at preview, the visible `发布` button, or the publish-settings page.

## Content Strategy

Zhihu works best when the article reads like an explanatory essay rather than a
release note. For existing Chinese long-form eunomia.dev posts, preserve the
canonical body and use platform adaptation for title, images, links, cover,
column, and preview quality. Rewrite only when the source is not already a
usable Chinese essay. For long tutorials, publish the conceptual article on
Zhihu and link to the complete code or tutorial.

Optimize for the maintainer's personal technical account brand: clear taste,
useful explanation, research and engineering credibility, and discussion from
readers who care about eBPF systems, AI agents, and runtime observability. Keep
the eunomia.dev, GitHub, or paper link as source or extended reading rather than
making the article feel like outbound traffic acquisition.

## Safety Boundary

Do not automate:

- final `发布`
- direct Zhihu API access, internal endpoint reads, or browser-hidden data fetches
- column submission unless the user names the exact column
- deleting drafts or changing account settings
- security, privacy, phone-verification, or CAPTCHA prompts
- likes, comments, follows, reposts, or private messages

## Ledger Update

After a confirmed publish, update `.github/publisher/media/published.md` with
title, source path, Zhihu URL, date, tags or column, and formatting fixes. Remove
or update the matching row in `.github/publisher/media/not-published.md`.

Keep screenshots and observed UI notes under `.github/publisher/media/`.
