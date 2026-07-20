---
name: eunomia-social-radar
description: Monitor the continuing public performance and conversation around Eunomia blogs, reports, projects, and platform posts. Use when Codex needs to inspect published-content metrics, verify that pages and previews still work, find reposts or citations, read comments and replies, follow active discussions, identify response or repost opportunities, compare observations over time, and return actionable findings to eunomia-content-patrol. This skill uses normal visible browser interaction for social platforms and does not authorize a final reply, repost, or other engagement action by itself.
---

# Eunomia Social Radar

Follow what happens after publication. Observe each relevant published item over
time, understand the surrounding conversation, and return only findings that can
improve distribution, responses, or future editorial work.

## Required Context

Read these before inspecting platforms:

- `CLAUDE.md`
- `.agents/README.md`
- today's media workspace and run log, if present:
  `draft/media/YYYY-MM-DD/` and `draft/media/YYYY-MM-DD/run-log.md`
- `.github/publisher/media/README.md`
- `.github/publisher/media/published.md`
- `.github/publisher/media/not-published.md`
- relevant `.github/publisher/media/platforms/*.json`
- matching publisher skill for any platform being inspected

Use those records to identify the canonical article, every known platform post,
publication time, account, and earlier observations.

## Observation Scope

Prioritize:

- newly published blogs and their platform adaptations
- posts with active or unresolved discussion
- posts whose earlier observation requested a follow-up checkpoint
- older evergreen posts that receive a new citation, repost, or traffic signal

Do not treat the radar as an endless feed-reading exercise. Start from known
published items, then follow their visible echoes and adjacent conversations.

## Workflow

### 1. Verify The Published Surface

Confirm that the canonical page and known platform posts remain publicly
reachable and that title, media preview, links, and visible formatting are
intact. Report a broken or incorrect artifact immediately; do not silently edit
or republish it.

### 2. Observe Performance

Collect only metrics visible to the normal account or public page, such as
views, impressions, reactions, comments, replies, reposts, quotes, saves, clicks,
or follower changes. Record the observation time and distinguish unavailable
metrics from zero activity.

Compare a post with its own earlier checkpoints and, when useful, with similar
posts on the same account and platform. Do not compare raw counts across
platforms as if they measured the same behavior. Avoid strong conclusions from
small or early samples.

### 3. Find External Echoes

Search for the canonical URL, exact title, distinctive phrases, project name,
and author/account mentions. Look for reposts, quote posts, backlinks, citations,
forum submissions, newsletters, and discussions that may not link the article
directly.

Search engines can discover leads. Inspect the original public page before
relying on a mention, quotation, or discussion context.

### 4. Read The Conversation

Read substantive comments, replies, quote context, objections, questions, and
maintainer discussion. Separate:

- factual correction or broken-link reports needing prompt attention
- genuine technical questions worth answering
- disagreement that could improve or challenge the article
- useful practitioner evidence for future research
- praise or low-information reactions that need no response
- spam, abuse, confidential requests, or promotional bait to ignore

Do not optimize for replying to everything. A thoughtful unanswered technical
question matters more than response volume.

### 5. Recommend The Next Action

For each actionable finding, recommend one of: reply, clarify the source,
correct the canonical article, quote/repost with context, prepare a follow-up,
capture a future research question, or take no action. Include the public URL,
why the action matters, and any factual support a response needs.

Use the matching publisher skill to prepare platform-native reply or repost
copy. When `eunomia-content-patrol` routes the action as part of today's tasks,
complete the final interaction; the daily task already supplies the required
authorization. In a standalone run, act when the user authorizes the action.

## Browser Boundary

Use only normal visible browser interaction for LinkedIn, Xiaohongshu, Zhihu,
Juejin, X, Reddit, Medium, DEV, Hacker News, Lobsters, and similar platforms.
Do not use platform APIs, hidden endpoints, background requests, or scraping
datasets. Search indexing can discover a lead but cannot replace reading the
original visible post before acting on it.

Do not send DMs, connection requests, follows, likes, votes, or account-setting
changes. Do not expose private analytics, customer information, or unpublished
strategy in a response draft or log.

## Output

Return a compact result to `eunomia-content-patrol` containing:

- items and platforms inspected, with observation time
- meaningful metric changes, without dumping every visible number
- reposts, citations, discussions, comments, or replies worth attention
- recommended actions and whether they are due in today's patrol
- the next useful checkpoint

When a separate observation record is useful, write one compact entry in
`draft/media/YYYY-MM-DD/run-log.md`. Do not create a monthly daily-log file or a
standalone daily radar file unless the user explicitly requests a public or
shareable analysis. Update platform ledgers only after a real action or when the
existing ledger schema explicitly stores observation status.

Load `references/observation-guide.md` when comparing performance or deciding
whether a conversation warrants action.
