---
name: content-launch-planner
description: Plan contribution-first cross-platform launches for eunomia.dev content, GitHub projects, papers, tutorials, demos, and public updates. Use when asked where/when/whether to publish, whether to do all-platform posting, whether others already posted, how to adapt one artifact into a platform matrix, how to prepare Product Hunt or community launches, or how to coordinate Zhihu/Juejin/X/LinkedIn/Xiaohongshu/Reddit/Hacker News/Lobsters/Medium/DEV publisher skills without posting yet.
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
4. For long-form blog/tutorial/paper explainers, default to faithful
   syndication: Medium/DEV use the existing English source and Zhihu/Juejin use
   the existing Chinese source. Preserve the selected source title exactly and
   keep the body substantively unchanged. Do not rewrite, shorten, expand,
   reorder, split, localize, or add platform-native openings, examples, or
   conclusions. Only mechanical rendering fixes and platform metadata are
   allowed. If the source needs a content fix, update the source first and then
   syndicate that corrected version. If no suitable source exists in the target
   language, stop and get explicit translation or rewrite authorization.
   Visible canonical/source links are optional on every platform.
5. Select platforms by fit, not by desire to be everywhere. All-platform plans
   must still produce platform-native angles and may recommend "skip" for weak
   surfaces.
6. Prepare a per-platform brief: target reader, surface, angle, hook, proof,
   link, media, CTA, risks, and follow-up.
7. Before browser/editor work, route platform-specific copy drafts to
   `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md`, using the local date.
   For long-form syndication that preserves the article body, the
   draft may reference the canonical source file instead of duplicating it, but
   it must record the exact source title, links, tags/categories, media, and QA
   state. Short posts, comments, and replies should include the full paste-ready
   copy.
8. For long-form posts on every platform, finish the platform-specific artifact
   locally before touching the platform editor whenever possible. Generate a
   temporary or `draft/media/YYYY-MM-DD/<source-slug>/` upload/import artifact
   with the exact source title, duplicate body H1 removed, image URLs or
   uploaded images, necessary table/code/formula rendering fallbacks,
   tags/categories, and links already present in the source.
   Use the platform editor for import/upload, settings, and QA, not for writing
   or structural repair.
9. Hand off execution to the matching publisher skill with the required browser
   QA state. Do not paste into a platform or publish from this skill unless the
   user explicitly asks to move from planning into execution.
10. Include a follow-up plan for comments, private messages, GitHub issues,
   corrections, and retrospective notes.

## Boundaries

- Keep public content 80% contribution and 20% promotion. Project mentions are
  evidence, implementation artifacts, or next steps after useful explanation.
- Do not make substantive changes to syndicated long-form posts. Preserve the
  source title, opening, section order, claims, examples, and conclusion. Limit
  platform changes to front matter, duplicate H1 removal, heading-level
  conversion, image upload/URL repair, code/table/formula rendering, and
  platform metadata. Add body text or links only when the user explicitly asks
  or after making the same change in the source article.
- Do not force visible canonical links on any platform. If a platform provides a
  dedicated canonical field, treat it as a hygiene setting when convenient; do
  not edit the article body just to add or repair a canonical/source link.
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
- `xiaohongshu-publisher` for Xiaohongshu visual notes, carousel outlines,
  title/cover adaptation, and Chinese short-form technical explainers.
- `reddit-publisher` for subreddit submissions and comments.
- `hackernews-publisher` for HN titles, Show HN, Ask HN, and comments.
- `lobsters-publisher` for Lobsters stories, tags, and comments.
- `medium-publisher` for Medium long-form syndication or essays.
- `devto-publisher` for DEV tutorials, series, and comments.

If no publisher skill exists for a recommended channel, output the plan only and
say what a future publisher skill would need to handle.
