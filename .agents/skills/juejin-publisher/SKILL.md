---
name: juejin-publisher
description: Prepare or publish eunomia.dev Markdown articles on Juejin. Use when asked to create, paste, publish, QA, or record a Juejin draft or article from repository Markdown. Preserves a syndicated long-form source title and body apart from mechanical Markdown/rendering fixes, with browser-editor workflow, category and tag selection, publication authorization, and media ledger updates.
---

# Juejin Publisher

Prepare or publish a reviewed Juejin article from the canonical repository
source and choose appropriate technical categories and tags. A request to
publish or a queue item marked `排队` authorizes the final action; do not ask
again at the last button. Stop only for a draft or preview task.

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
For the maintained `yunwei37` account, prefer the existing Chrome `Yunwei`
profile, which carries the verified Juejin login. A fresh in-app browser session
may be logged out; do not treat that state as evidence that the account itself
is unavailable. Confirm the avatar and creator controls on the visible page
before proceeding.
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

When resuming an interrupted publish, first inspect the normal visible authored
profile for the exact title and source identity. The final submission may have
succeeded before the parent task stopped. If the article is already public, do
not reopen a new draft or submit again; verify the public page and reconcile the
artifact, ledger, queue, and snapshots instead.

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
7. Complete `确定并发布` when the task requests publication or the queue item
   is marked `排队`; do not ask for duplicate confirmation. Stop at preview only
   for a draft or preview task.

For images, verify the exact final URL used in Markdown before publishing. Do
not assume `imgs/...` can be converted by guessing an eunomia.dev article path;
that path may return 404. A public GitHub raw URL can also produce
`转存失败，建议直接上传图片文件` in Juejin. When that happens, upload the
source image through the visible editor, use the resulting Juejin-hosted URL in
the local publishing copy, and confirm that the failure marker is gone and the
rendered image has non-zero dimensions. Verify the same image again on the
public page.

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

- final `确定并发布` when the task is limited to a draft or preview
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
