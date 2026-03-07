# App Migration TODO

This is the execution backlog for the custom frontend.

Status markers:

- `[todo]` not started
- `[doing]` in progress
- `[done]` implemented and verified
- `[blocked]` waiting on a prerequisite or decision

## Phase 0: Architecture Lock

- `[done]` Split `lib/content.ts` into `content/*` modules as the first structural refactor.
- `[done]` Introduce a `ContentManifest` as the route/source registry and migrate remaining loaders onto it.
- `[done]` Define the Markdown allowlist/sanitization policy in docs.
- `[done]` Define parity cut-lines for `shadow`, `cutover`, and `growth`.
- `[done]` Define Pages Router exit criteria and ban new duplicated locale route code.
- `[done]` Define redirect behavior for legacy, cutover, and canonicalized paths.
- `[done]` Define route-class rollback behavior during coexistence.
- `[done]` Define cache/revalidation behavior for static content and future ISR.

## Phase 1: Content Core

- `[done]` Build `fs-index.ts` for repo file inventories.
- `[done]` Build `types.ts` for parsed markdown, manifest, and page payload types.
- `[done]` Build `manifest.ts` for route identities, alternates, and sitemap candidates.
- `[done]` Build `collections.ts` for tutorials, blog, legacy blog, and section views.
- `[done]` Build `rewrite.ts` for local link and asset rewriting.
- `[done]` Build `assets.ts` for raw asset resolution and content type lookup.
- `[done]` Build `loaders.ts` for homepage, tutorials, blog, legacy blog, and section page data.

## Phase 2: Route and Locale Cleanup

- `[doing]` Replace duplicated route logic with shared page factories or render helpers.
- `[todo]` Collapse locale expansion onto one locale-aware URL builder.
- `[todo]` Keep one canonical route manifest for both English and Chinese.
- `[todo]` Stop adding framework logic to content helpers.

## Phase 3: Markdown and Docs Parity

- `[doing]` Enforce a raw-HTML sanitize allowlist after `rehype-raw`.
- `[doing]` Keep homepage rendering outside Markdown parity so docs hardening can stay strict.
- `[done]` Implement heading extraction and article TOC data.
- `[done]` Implement syntax highlighting parity for code blocks.
- `[todo]` Implement admonitions.
- `[todo]` Implement tabs.
- `[todo]` Validate unsupported constructs fail loudly in tests.
- `[todo]` Keep local asset resolution stable for nested tutorial paths.

## Phase 4: Metadata and SEO Parity

- `[done]` Keep title/description/canonical/hreflang generation centralized.
- `[done]` Improve Open Graph strategy beyond the current shared default.
- `[done]` Add article metadata rules for blog pages.
- `[done]` Add build-time git metadata for `last updated`.
- `[done]` Add build-time git metadata for `authors`.
- `[todo]` Add sitemap readiness gating by parity stage.

## Phase 5: Product Features

- `[done]` Replace the placeholder search input with a real search system.
- `[done]` Add full search results pages and keyboard navigation on top of quick search.
- `[done]` Add mobile navigation that preserves search and section discovery.
- `[done]` Add edit/source link support to all article-capable routes.
- `[done]` Add article continuation links for collection navigation.
- `[done]` Add feedback CTA component.
- `[done]` Add share buttons component.
- `[done]` Add RSS/feed generation and autodiscovery links.

## Phase 6: Testing

- `[done]` Add unit tests for slug generation.
- `[done]` Add unit tests for locale fallback resolution.
- `[done]` Add unit tests for route mapping.
- `[done]` Add unit tests for local path and asset rewriting.
- `[todo]` Add Markdown fixture tests for:
  - `[todo]` nested tutorial docs
  - `[todo]` legacy `/blogs/*`
  - `[todo]` `.en.md` and `.zh.md` variants
  - `[todo]` inline HTML allowed by policy
  - `[todo]` unsupported syntax failures
- `[todo]` Add a payload-size regression check for very large blog pages.

## Phase 7: Rollout

- `[todo]` Define which routes are allowed into sitemap during shadow mode.
- `[todo]` Define the cutover checklist for route classes.
- `[todo]` Define the fallback/rollback path while MkDocs still exists.
- `[done]` Define observability signals for route-class and locale health.
- `[done]` Define which audit failures or metrics should block rollout or trigger rollback.
- `[done]` Verify production parity against indexed and legacy routes.
- `[done]` Isolate production build output from `.next` so `next dev` and `next start` can be validated side-by-side.

## Current Working Order

1. `[done]` Design the full system and backlog.
2. `[done]` Review the design iteratively with codex reviewers/subagents.
3. `[done]` Implement the content subsystem split.
4. `[doing]` Add content-layer tests and finish migrating route/source lookups onto the manifest.
5. `[done]` Harden the Markdown pipeline.
6. `[done]` Implement heading extraction and article TOC data.
7. `[done]` Implement syntax highlighting parity for fenced code blocks.
8. `[done]` Restore search, git metadata, feedback CTA, and share actions.
9. `[todo]` Implement remaining Markdown cutover blockers: admonitions and tabs.
10. `[todo]` Collapse duplicated locale route logic.
11. `[done]` Verify build and audits still pass.
