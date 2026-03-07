# Rollout Discipline

This file defines how the custom frontend should coexist with the current MkDocs deployment and when a route class is allowed into the public sitemap.

## Stages

### `shadow`

Goal:

- prove parity without changing the indexed route surface

Rules:

- sitemap must be a strict parity mirror of the legacy MkDocs sitemap
- no app-only routes are allowed into `sitemap.xml`
- dated `/blog/YYYY/MM/DD/...` permalinks must stay live, but they are not advertised yet

Verification:

- `test/scripts/sitemap-parity.mjs`
- `test/scripts/rollout-audit.mjs`

### `cutover`

Goal:

- switch the public site to the custom frontend while preserving all indexed legacy routes

Rules:

- every legacy sitemap route must still exist
- the only allowed app-only sitemap additions are dated `/blog/YYYY/MM/DD/...` permalinks
- search, metadata, alternates, feeds, browser smoke, runtime audit, and link crawl must all pass

Verification:

- `npm run audit`

### `growth`

Goal:

- allow deliberate route-surface expansion after cutover

Rules:

- `growth` must remain a superset of `cutover`
- any new sitemap route class needs an explicit parity decision and new audit coverage before it is admitted

## Environment Switch

Use `EUNOMIA_SITEMAP_STAGE` to choose the active sitemap surface:

- `shadow`
- `cutover`
- `growth`

If unset, the app defaults to `cutover`.

## Checklist

Before moving from `shadow` to `cutover`:

- all legacy sitemap paths are present
- app sitemap has no duplicate URLs
- dated blog additions are the only allowed extras
- `test:content`, `build`, `audit:http`, `audit:browser`, `audit:links`, `audit:runtime`, and `audit:rollout` all pass

Before moving from `cutover` to `growth`:

- the new route class is documented in `FEATURE_PARITY.md`
- the sitemap admission rule is updated in `app/lib/rollout.ts`
- audit coverage is added for the new route class

## Rollback

Rollback means switching the indexed surface back to `shadow` while keeping the app deployable for preview and direct validation.

Trigger rollback if any of these happen:

- sitemap parity drops below `479/479` for legacy routes
- `hreflang`, canonical, or `html lang` regress across audited routes
- runtime audit reports `large page data` warnings on heavy routes
- browser smoke or link crawl finds broken navigation or broken assets on parity-bound pages
