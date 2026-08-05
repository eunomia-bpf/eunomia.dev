# Daily SEO task

## Objective

Run one fully autonomous, evidence-backed SEO operating cycle for eunomia.dev.
No normal collection, repository, review, merge, deployment, or verification
step requires human approval.

## Schedule

- Frequency: daily when invoked by an authorized agent scheduler
- Timezone: use `site.md`
- Data window: use the lookback and finalization lag in `site.md`
- Maximum site changes: one coherent, evidence-backed change per main pull request

## Scoped workflow override

For work invoked through this file, the fresh-branch and pull-request contract
below replaces the repository's general direct-`main` workflow. This standing
override is limited to SEO evidence collection, `.github/seo-data` maintenance,
submodule updates, and evidence-backed SEO/GEO site changes. Use the `seo/`
branch prefix from `site.md`; never push an automated SEO change directly to
`main`.

## Required sequence

1. Read the pinned `$collect-seo-data` skill and, when a site change is justified,
   `$change-seo-site`; also read repository instructions, all
   `.github/seo-data/*.md` files, and the newest daily records.
2. Fetch the remote default branch and create a fresh branch from it.
3. Check whether the `seo-skills` submodule has a newer allowed commit and include
   an available compatible update in the same main pull request.
4. Collect evidence only from sources marked enabled in `site.md`. Record a
   disabled, missing, stale, partial, or unavailable source explicitly; never
   convert missing evidence into zero. Public repository and live-site evidence
   may supplement analytics but does not replace source-native metrics.
5. Write or append `.github/seo-data/daily/YYYY-MM-DD.md`; refresh `status.md`,
   keep durable future work in `plan.md`, and reserve `block.md` for genuine
   human-only or permission blockers.
6. When evidence supports a site improvement, implement at most one coherent
   change and define its production acceptance check before editing. Use
   `.agents/skills/seo-geo/SKILL.md` as the site-specific technical checklist;
   the pinned skills remain authoritative for evidence collection and delivery.
7. Validate with the pinned validator and the smallest authoritative repository
   checks, inspect the intended diff, push the branch, and create a real
   non-draft pull request.
8. Wait for every required and expected CI check, then self-review the complete
   final diff, commits, generated output, and check results. Fix issues on the
   same branch and repeat CI and review as needed.
9. Squash-merge the pull request and delete its branch only after green CI and a
   clean final self-review.
10. For a site change, identify and wait for the production deployment triggered
    by the exact squash commit, then verify the defined behavior on the public
    site.
11. Open a metadata-only closeout pull request with the verified delivery facts;
    apply the same CI, self-review, squash-merge, and branch-deletion rules.
12. Do not manufacture content, site edits, promotion, or visible artifacts only
    to satisfy cadence. Promotion must follow the existing publication queue and
    platform-specific publisher skills.

## Daily completion

A day is complete only after its main pull request and closeout pull request are
squash-merged. A site-change day also requires a successful production deployment
for the exact squash commit and public verification. A failed, missing, queued,
skipped, or cancelled CI check, a local-only commit, issue, draft pull request,
workflow URL, or HTTP 200 alone is not completion.
