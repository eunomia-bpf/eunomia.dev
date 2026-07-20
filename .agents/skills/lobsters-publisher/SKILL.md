---
name: lobsters-publisher
description: Prepare Lobsters submissions and follow-up drafts for eunomia.dev content, GitHub projects, papers, demos, and computing-focused technical posts. Use when asked to submit, adapt, QA, or record Lobsters posts. Covers tag selection, computing-topic fit, self-promotion safety, browser duplicate checks, safe stop-before-submit behavior, comment/private-message follow-up, and media ledger updates.
---

# Lobsters Publisher

Prepare Lobsters submissions that fit the computing-focused community and stop
before final submission unless the user explicitly confirms it.

## Inputs

- Source URL, GitHub repo, paper, demo, blog post, or topic.
- Optional tags, title, text, or follow-up plan.

If the source URL is missing, inspect recent `docs/blog/posts/`, relevant
GitHub repos, and canonical eunomia.dev URLs before asking the user.

## Platform Entry Points

- New story: <https://lobste.rs/stories/new>
- About/guidelines: <https://lobste.rs/about>
- Recent: <https://lobste.rs/recent>
- Search: <https://lobste.rs/search>

Use a browser surface with the logged-in session when UI work is required.

## References

Load `references/platform-preferences.md` when choosing Lobsters fit, tags,
title, source URL, self-promotion safety, browser checks, promotion balance, or
follow-up. Do not load broad strategy drafts for routine submissions unless the
user asks for campaign or content-platform planning.

## Browser-Only Platform Boundary

Do not use unofficial submission automation, internal endpoints, or vote-related
actions. All checks and submission prep must come from normal browser
interactions.

## Draft Preparation

1. Check whether the item is narrowly about computing.
2. Prefer the original/canonical source and remove tracking parameters.
3. Choose predefined tags that accurately match the story.
4. Keep the title factual and unpromotional.
5. Disclose affiliation in comments when relevant.

## Draft Archive

Before opening the Lobsters submit page or comment box, write or update the
Lobsters draft under `draft/media/YYYY-MM-DD/<source-slug>/lobsters.md` using
the local date. Include title, URL, tags, duplicate-check state, affiliation
note, and any full paste-ready comment options.

## Browser QA

Before stopping for user confirmation, verify:

- title, URL, and tags are correct
- duplicate/recent same-topic searches are clean
- the account is allowed to submit the domain and tags
- tracking parameters are absent
- the visible final submit action has not been clicked

## Follow-Up

After confirmed submission, capture the story URL. Monitor comments and private
messages only when the user asks or follow-up was part of the task. Draft
substantive replies and move durable implementation work to GitHub.

## Ledger Update

After confirmed submission, update `.github/publisher/media/published.md` with
source path or URL, Lobsters URL, date, tags, title, and follow-up notes.
