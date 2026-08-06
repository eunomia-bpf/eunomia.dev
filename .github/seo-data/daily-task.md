# Daily technical SEO subtask

`DAILY_TASK.md` is the only scheduled entrypoint and the authoritative daily
operating contract. An external scheduler must not target this file directly.
Use this file only when the daily operation evaluates or selects technical SEO
and GEO work.

## Objective

Use measured search, acquisition, repository, and live-site evidence to maintain
and improve eunomia.dev's technical discoverability, indexability,
retrievability, and citation quality. Do not create a site edit merely because a
daily run occurred.

## Required context

Read:

- `DAILY_TASK.md`;
- `CLAUDE.md`;
- every Markdown file under `.github/seo-data/`;
- the newest record under `.github/seo-data/daily/`, when present;
- `.agents/skills/seo-geo/SKILL.md`;
- the pinned collection, change, and validation skills under
  `.github/seo-skills`.

The root daily task controls branching, pull requests, merging, deployment,
verification, records, and completion. Do not duplicate those rules here.

## Technical analysis

Use every enabled source in `site.md` and the windows defined there. Missing,
stale, partial, disabled, or inaccessible sources are unavailable, not zero.
Public repository and live-site evidence may supplement analytics but does not
replace source-native search or acquisition metrics.

Evaluate at least:

- crawl access, robots, sitemap coverage, status codes, redirects, and broken
  internal links;
- canonical URLs, duplicate routes, indexability, pagination, and URL stability;
- English and Chinese `hreflang` pairs and language navigation;
- titles, descriptions, headings, structured data, Open Graph, and semantic
  page identity;
- internal linking, orphan pages, hub structure, repository/paper/tutorial
  pathways, and citation-ready source presentation;
- rendering, performance, accessibility signals relevant to discovery, and the
  production deployment state;
- important query, page, landing-page, referrer, and outbound movements when
  Search Console or analytics coverage exists.

Record the evidence and interpretation in the day's shared record under
`.github/seo-data/daily/`; do not create a second SEO-only run log.

## Change gate

A technical change is justified only when the evidence identifies a concrete
problem or opportunity and the expected outcome is observable. Define the
production acceptance check before editing.

Prefer one coherent change such as:

- repairing canonical, sitemap, robots, redirect, or language-alternate behavior;
- improving one information architecture or internal-linking path;
- correcting structured data or metadata for a well-defined page family;
- removing an indexability or rendering defect;
- improving retrievability for a technically important topic supported by search
  or reader evidence.

Do not bundle unrelated cleanup, speculative keyword insertion, mass-generated
pages, promotion, or a routine skill-submodule update. Update the pinned SEO
skills only when a compatible change is needed for the current operation or
clearly fixes the operating mechanism.

## Delivery

Follow `DAILY_TASK.md`: use the daily branch and pull request, pass the
repository's authoritative checks, self-review the complete output, squash
merge, and verify the exact production deployment when the public site changes.
Do not create a separate closeout pull request; use the merged pull request's
compact closeout comment as the final delivery record.
