---
name: hackernews-publisher
description: Prepare Hacker News submissions and follow-up drafts for eunomia.dev content, GitHub projects, papers, demos, and Show HN style launches. Use when asked to submit, adapt, QA, or record HN posts. Covers original-source selection, title normalization, curiosity-first positioning, browser duplicate checks, safe stop-before-submit behavior, comment follow-up, and media ledger updates.
---

# Hacker News Publisher

Prepare Hacker News submissions and stop before final submission unless the user
explicitly confirms it.

## Inputs

- Source URL, GitHub repo, paper, demo, blog post, or topic.
- Submission type when specified: regular link, Show HN, Ask HN, or comment.
- Optional title, text, or follow-up plan.

If the source URL is missing, inspect recent `docs/blog/posts/`, relevant
GitHub repos, and canonical eunomia.dev URLs before asking the user.

## Platform Entry Points

- Submit: <https://news.ycombinator.com/submit>
- Show HN: <https://news.ycombinator.com/show>
- New: <https://news.ycombinator.com/newest>
- Search: <https://hn.algolia.com/>

Use a browser surface with the logged-in session when UI work is required.

## References

Load `references/platform-preferences.md` when choosing HN fit, title, source
URL, Show HN eligibility, browser checks, promotion balance, or comment
follow-up. Do not load broad strategy drafts for routine submissions unless the
user asks for campaign or content-platform planning.

## Browser-Only Platform Boundary

Do not use unofficial submission automation, internal endpoints, or vote-related
actions. All checks and submission prep must come from normal browser
interactions.

## Draft Preparation

1. Decide whether HN is appropriate for the item.
2. Prefer the original source: GitHub repo, paper, demo, or canonical article.
3. Use the original title unless it is misleading, linkbait, too promotional, or
   needs a required tag such as `[pdf]`.
4. Use Show HN only for something readers can try.
5. Do not write generated comments for direct posting; draft comment options for
   user review and ownership.

## Draft Archive

Before opening the HN submit page or comment box, write or update the HN draft
under `draft/media/YYYY-MM-DD/<source-slug>/hackernews.md` using the local date.
Include submission type, title, URL, duplicate-check state, Show HN eligibility,
and any full paste-ready comment options for user review.

## Browser QA

Before stopping for user confirmation, verify:

- the same URL/topic is not already active
- title is neutral, not all-caps, and does not include the site name
- URL is the original/canonical source
- Show HN title is only used when the artifact is tryable
- the visible final `submit` action has not been clicked

## Follow-Up

After confirmed submission, capture the HN item URL. Monitor comments only when
the user asks or follow-up was part of the task. Draft concise, human-reviewed
technical replies. Do not solicit upvotes or comments, and do not ask others to
submit or promote the link.

## Ledger Update

After confirmed submission, update `.github/publisher/media/published.md` with
source path or URL, HN URL, date, submission type, title, and follow-up notes.
