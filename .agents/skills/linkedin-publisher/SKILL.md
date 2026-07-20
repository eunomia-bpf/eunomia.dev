---
name: linkedin-publisher
description: Prepare eunomia.dev content for LinkedIn posts, articles, carousels, and professional updates. Use when asked to draft, adapt, paste, QA, or record LinkedIn content for blog, tutorial, project, paper, release, consulting, research, or GitHub updates. Covers professional positioning, reader targeting, browser QA, safe stop-before-post behavior, comment/DM follow-up, and media ledger updates.
---

# LinkedIn Publisher

Prepare professional LinkedIn drafts that build technical credibility and stop
before final posting unless the user explicitly confirms it.

## Inputs

- Source Markdown path, project, repo URL, paper, or topic.
- Intended audience, format, and posting account when specified.
- Optional media, canonical link, GitHub link, hashtags, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, recent `docs/blog/posts/`, and
relevant GitHub repos before asking the user.

## Platform Entry Points

- Feed/composer: <https://www.linkedin.com/feed/>
- Articles/newsletters: <https://www.linkedin.com/pulse/new/>
- Notifications: <https://www.linkedin.com/notifications/>
- Messaging: <https://www.linkedin.com/messaging/>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing LinkedIn-native framing,
professional positioning, first-lines hook, media/link treatment, promotion
balance, or follow-up. Do not load broad strategy drafts for routine posting
unless the user asks for campaign or content-platform planning.

## Browser-Only Platform Boundary

Do not directly access LinkedIn APIs, internal endpoints, or background HTTP
interfaces. All drafting, QA, screenshots, and ledger evidence must come from
normal browser interactions.

## Draft Preparation

1. Read the source and identify the professional lesson, target reader, proof,
   and single next step.
2. Choose the format:
   - feed post for the default blog-share case: a concise professional lesson
     plus a canonical/GitHub/paper link
   - document/carousel plan for a step-by-step visual explanation
   - article only when the user explicitly asks for long-form on LinkedIn
3. For a LinkedIn article that syndicates an existing long-form source, preserve
   the source title exactly and keep the body substantively unchanged. Only make
   mechanical rendering fixes and set platform metadata. A rewrite requires an
   explicit request for that article.
4. Open short feed posts with the result, tension, or professional lesson.
5. Use the project or GitHub link as evidence after the reader-facing value is
   clear.

## Draft Archive

Before opening the LinkedIn composer, write or update the LinkedIn draft under
`draft/media/YYYY-MM-DD/<source-slug>/linkedin.md` using the local date. Include
the full paste-ready post or exact long-form source reference,
first-visible-lines check, primary
link, optional first-comment links, media/alt text, visibility target when
known, and QA state.

## Browser QA

Before stopping for user confirmation, verify:

- first visible lines make the value clear before "see more"
- link card, media, or document preview renders correctly
- audience/visibility setting is intentional
- mentions and hashtags are relevant and minimal
- no confidential customer, partner, or roadmap information appears
- the visible final `Post` / `Publish` action has not been clicked

## Follow-Up

After a confirmed publish, capture the URL and monitor comments, reactions, and
messages only when the user asks or follow-up was part of the task. Draft
substantive replies to technical questions and move implementation work to
GitHub when appropriate. Do not send connection requests, DMs, likes, reposts,
or comments without explicit instruction.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, LinkedIn URL, date, format, media, link target, and follow-up notes.
