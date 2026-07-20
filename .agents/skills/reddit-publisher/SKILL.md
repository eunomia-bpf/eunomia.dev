---
name: reddit-publisher
description: Prepare eunomia.dev content for Reddit submissions and comments. Use when asked to draft, adapt, paste, QA, or record Reddit posts for technical communities, project discussions, blog links, tutorials, or GitHub updates. Covers subreddit rule checks, community-first positioning, self-promotion safety, browser QA, stop-before-post behavior, comment/DM/mod follow-up, and media ledger updates.
---

# Reddit Publisher

Prepare Reddit drafts that fit the target subreddit and stop before posting
unless the user explicitly confirms it.

## Inputs

- Source Markdown path, project, repo URL, paper, or topic.
- Target subreddit if specified.
- Optional post type, flair, title, link, self-text, or follow-up plan.

If the target subreddit is missing, inspect the topic and suggest likely
communities, but do not post until subreddit rules are checked in the browser.

## Platform Entry Points

- Submit: <https://www.reddit.com/submit>
- Search: <https://www.reddit.com/search/>
- User notifications: <https://www.reddit.com/notifications>
- Messages: <https://www.reddit.com/message/messages>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing subreddit fit, title,
self-promotion safety, comment strategy, browser QA, promotion balance, or
follow-up behavior. Do not load broad strategy drafts for routine posting unless
the user asks for campaign or content-platform planning.

## Browser-Only Platform Boundary

Do not directly access Reddit APIs, internal endpoints, or background HTTP
interfaces. All drafting, QA, screenshots, rule checks, and ledger evidence must
come from normal browser interactions.

## Draft Preparation

1. Check subreddit relevance before writing.
2. Search for duplicate or recent related discussions.
3. Read subreddit rules, pinned posts, flair requirements, and self-promotion
   norms.
4. Draft for the community: question, lesson, reproducible artifact, or source
   link with transparent affiliation.
5. Keep the project link contextual and useful.

## Draft Archive

Before opening the Reddit composer, write or update the Reddit draft under
`draft/media/YYYY-MM-DD/<source-slug>/reddit.md` using the local date. Include
the target subreddit, post type, title, link/self-text, flair, affiliation note,
duplicate/rule check state, and full paste-ready comment or post body.

## Browser QA

Before stopping for user confirmation, verify:

- target subreddit and flair are correct
- title is factual and not editorialized
- duplicate search did not reveal an active same-topic thread
- Markdown preview renders links, code, and lists correctly
- affiliation/self-promotion disclosure is present when needed
- the visible final `Post` action has not been clicked

## Follow-Up

After confirmed posting, capture the permalink. Monitor comments, mod messages,
and DMs only when the user asks or follow-up was part of the task. Reply with
substance, disclose affiliation, and move durable bug reports to GitHub. Do not
vote, cross-post, brigade, DM, or message moderators without explicit
instruction.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, subreddit, Reddit URL, date, flair, link target, and follow-up
notes.
