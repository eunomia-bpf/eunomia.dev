---
name: x-publisher
description: Prepare eunomia.dev public content for X/Twitter posts, threads, and X Articles. Use when asked to draft, adapt, paste, QA, or record X posts for blog, tutorial, project, paper, release, or GitHub updates. Covers concise platform-native copy, media and link checks, safe stop-before-post behavior, reply/mention/DM follow-up, and media ledger updates.
---

# X Publisher

Prepare platform-native X drafts from eunomia.dev content and stop before any
externally visible posting action unless the user explicitly confirms it.

## Inputs

- Source Markdown path, project, repo URL, paper, or topic.
- Intended format when specified: single post, thread, quote post, or X Article.
- Optional image/video, target link, posting account, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, recent `docs/blog/posts/`, and
relevant GitHub repos before asking the user.

## Platform Entry Points

- Post composer: <https://x.com/compose/post>
- Article composer: <https://x.com/compose/articles>
- Home/feed: <https://x.com/home>
- Notifications: <https://x.com/notifications>
- Search: <https://x.com/search>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing X-native framing,
thread shape, link/media treatment, promotion balance, or reply/mention
follow-up. Do not load broad strategy drafts for routine posting unless the user
asks for campaign or content-platform planning.

## Browser-Only Platform Boundary

Do not directly access X APIs, internal endpoints, or background HTTP
interfaces. All drafting, QA, screenshots, and ledger evidence must come from
normal browser interactions.

## Draft Preparation

1. Read the source and extract the core claim, evidence, link, media, and
   follow-up target.
2. Choose the format:
   - single post for the default blog-share case: one concrete idea plus a
     canonical/GitHub/paper link
   - thread for a short argument or multi-step explanation
   - X Article only when the user explicitly asks for long-form on X
3. For an X Article that syndicates an existing long-form source, preserve the
   source title exactly and keep the body substantively unchanged. Only make
   mechanical rendering fixes and set platform metadata. A rewrite requires an
   explicit request for that article.
4. Build short posts and threads around one useful idea before any CTA.
5. Prefer one specific source link: GitHub, eunomia.dev, paper, docs, or demo.
6. Prepare media alt text when the UI allows it.

## Draft Archive

Before opening the X composer, write or update the X draft under
`draft/media/YYYY-MM-DD/<source-slug>/x.md` using the local date. Include the
full paste-ready post or thread, character counts, primary link, media/alt text,
account target when known, and QA state.

## Browser QA

Before stopping for user confirmation, verify:

- text is not truncated unexpectedly and the thread order is correct
- link cards resolve to the intended URL
- uploaded media renders and has useful alt text where supported
- hashtags are absent or minimal and genuinely useful
- no confidential or unreleased claims are present
- the visible final `Post` / `Publish` action has not been clicked

## Follow-Up

After a confirmed publish, capture the post URL, then check notifications or
search results only when the user asks or when follow-up was part of the task.
Prioritize substantive replies to technical questions, corrections, and
implementation interest. Do not send DMs, like, repost, follow, or quote-post
without explicit user instruction.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, title or hook, X URL, date, link target, media used, and follow-up
notes. Remove or update the matching row in
`.github/publisher/media/not-published.md`.
