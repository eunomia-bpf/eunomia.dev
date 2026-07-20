---
name: content-launch-planner
description: Plan contribution-first cross-platform launches for eunomia.dev content, GitHub projects, papers, tutorials, demos, and public updates. Use when asked where/when/whether to publish, whether to do all-platform posting, whether others already posted, how to adapt one artifact into a platform matrix, how to prepare Product Hunt or community launches, or how to coordinate Zhihu/Juejin/X/LinkedIn/Reddit/Hacker News/Lobsters/Medium/DEV publisher skills without posting yet.
---

# Content Launch Planner

Plan the upstream publishing decision before any platform-specific drafting or
browser work. This skill decides whether to publish, which platforms fit, what
each platform should say, what assets are missing, and which publisher skill
should execute each item.

## References

Load references according to the task:

- Always load `references/source-and-research.md`,
  `references/platform-matrix.md`, `references/quality-gates.md`, and
  `references/output-template.md`.
- Load `references/launch-assets.md` for project, release, demo, paper,
  tutorial, or multi-post campaign planning.
- Load `references/producthunt.md` only when Product Hunt is proposed, visible
  in the browser, or plausibly useful for a tryable product/tool artifact.
- Load `references/external-patterns.md` when updating this skill or explaining
  which external skill patterns were absorbed.

## Workflow

1. Build a strategic source brief first: user pain, search/community intent,
   current alternatives, unique public evidence, brand pillar, reader outcome,
   and only then repository evidence, GitHub links, drafts, papers, screenshots,
   and published/not-published ledgers.
2. Check for duplicates and current discussion before recommending a community
   submission or another post about the same artifact.
3. Classify the launch tier and reader intent: contribution-first technical
   education, project/tool launch, paper/research discussion, tutorial, release
   note, or follow-up.
4. For long-form blog/tutorial/paper explainers with canonical English and/or
   Chinese sources, default to canonical syndication: Medium/DEV use the
   English version, Zhihu/Juejin use the Chinese version, and the body is not
   rewritten unless the user asks or the source itself has a problem.
5. Select platforms by fit, not by desire to be everywhere. All-platform plans
   must still produce platform-native angles and may recommend "skip" for weak
   surfaces.
6. Prepare a per-platform brief: target reader, surface, angle, hook, proof,
   link, media, CTA, risks, and follow-up.
7. Before browser/editor work, route platform-specific copy drafts to
   `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md`, using the local date.
   For long-form canonical syndication that preserves the article body, the
   draft may reference the canonical source file instead of duplicating it, but
   it must record the exact title, links, tags/categories, source note, media,
   and QA state. Short posts, comments, and replies should include the full
   paste-ready copy.
8. Hand off execution to the matching publisher skill with the required browser
   QA state. Do not paste into a platform or publish from this skill unless the
   user explicitly asks to move from planning into execution.
9. Include a follow-up plan for comments, private messages, GitHub issues,
   corrections, and retrospective notes.

## Boundaries

- Keep public content 80% contribution and 20% promotion. Project mentions are
  evidence, implementation artifacts, or next steps after useful explanation.
- Do not recommend full rewrites for long-form syndicated posts by default.
  Limit changes to title micro-tuning, canonical/GitHub/paper links, images,
  code blocks, tags, and a low-key project/source note.
- Preserve the maintainer posture: consulting, research, and helping solve hard
  technical problems, not selling a product.
- Do not include private strategy, customer claims, fundraising, pricing,
  partner/customer details, unreleased roadmaps, or unverifiable numbers.
- Social and media platforms are browser-first. Do not use hidden platform APIs,
  internal endpoints, background requests, or automatic posting tools. Medium
  and DEV publishing must use the normal web editor/import UI and visible submit
  buttons, not publish APIs.
- Stop at a plan, draft, editor, or confirmation screen unless the user clearly
  confirms final publishing.

## Publisher Handoff

Use the platform-specific publisher skill after the plan is accepted:

- `zhihu-publisher` for Zhihu articles, answers, ideas, and AI Works.
- `juejin-publisher` for Juejin developer articles and practical notes.
- `x-publisher` for short X posts, threads, replies, quotes, and articles only
  when explicitly useful.
- `linkedin-publisher` for short LinkedIn feed posts, articles/carousels only
  when explicitly useful, and comments.
- `reddit-publisher` for subreddit submissions and comments.
- `hackernews-publisher` for HN titles, Show HN, Ask HN, and comments.
- `lobsters-publisher` for Lobsters stories, tags, and comments.
- `medium-publisher` for Medium long-form syndication or essays.
- `devto-publisher` for DEV tutorials, series, and comments.

If no publisher skill exists for a recommended channel, output the plan only and
say what a future publisher skill would need to handle.
