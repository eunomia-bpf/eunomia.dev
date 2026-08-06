# Daily SEO task

This is the technical SEO subtask of the root `DAILY_TASK.md`. The root file is
the only scheduled entrypoint and the authoritative daily operating contract. An
external scheduler must not target this file directly.

## Objective

Use measured search, acquisition, repository, and live-site evidence to maintain
and improve eunomia.dev's technical discoverability, indexability,
retrievability, and citation quality. Do not create a site edit merely because a
daily run occurred.

## Required sequence

1. Read `DAILY_TASK.md`, `CLAUDE.md`, every Markdown file under
   `.github/seo-data/`, the newest daily record when present,
   `.agents/skills/seo-geo/SKILL.md`, and the pinned collection, change, and
   validation skills under `.github/seo-skills`.
2. Use every enabled source in `site.md` and the windows defined there. Missing,
   stale, partial, disabled, or inaccessible sources are unavailable, not zero.
3. Evaluate crawl access, robots, sitemap coverage, status codes, redirects,
   broken links, canonical URLs, duplicate routes, indexability, pagination, URL
   stability, `hreflang`, language navigation, metadata, structured data, Open
   Graph, semantic page identity, internal linking, orphan pages, hub structure,
   repository/paper/tutorial pathways, rendering, performance, accessibility,
   and production deployment state.
4. Analyze important query, page, landing-page, referrer, acquisition, and
   outbound movements when source-native Search Console or analytics coverage
   exists. Public repository and live-site evidence may supplement those sources
   but does not replace them.
5. Record evidence and interpretation in the day's shared record under
   `.github/seo-data/daily/`; do not create a second SEO-only run log.
6. Make a technical change only when evidence identifies a concrete problem or
   opportunity and the expected production outcome is observable.

The root daily task controls branching, pull requests, merging, deployment,
verification, records, and completion. Do not duplicate those rules here.

## Technical analysis

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

## Daily completion

Follow `DAILY_TASK.md`: use the unified `daily/` branch and pull request, pass the
repository's authoritative checks, self-review the complete output, squash
merge, and verify the exact production deployment when the public site changes.
Do not create a separate closeout pull request; use the merged pull request's
compact closeout comment as the final delivery record.

A no-change result is valid when the data does not support a defensible public
edit. A draft, local commit, queued check, or unverified deployment is not
completion.
