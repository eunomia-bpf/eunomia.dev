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

- `[done]` Replace duplicated route logic with shared route builders and render helpers.
- `[done]` Collapse locale expansion onto one locale-aware URL builder.
- `[done]` Keep one canonical route manifest for both English and Chinese.
- `[done]` Stop adding framework logic to content helpers.

## Phase 3: Markdown and Docs Parity

- `[done]` Enforce a raw-HTML sanitize allowlist after `rehype-raw`.
- `[done]` Keep homepage rendering outside Markdown parity so docs hardening can stay strict.
- `[done]` Implement heading extraction and article TOC data.
- `[done]` Implement syntax highlighting parity for code blocks.
- `[done]` Implement Mermaid diagram parity for existing docs.
- `[done]` Implement admonitions.
- `[done]` Implement tabs.
- `[done]` Validate unsupported constructs fail loudly in tests.
- `[done]` Keep local asset resolution stable for nested tutorial paths.

## Phase 4: Metadata and SEO Parity

- `[done]` Keep title/description/canonical/hreflang generation centralized.
- `[done]` Improve Open Graph strategy beyond the current shared default.
- `[done]` Add article metadata rules for blog pages.
- `[done]` Add build-time git metadata for `last updated`.
- `[done]` Add build-time git metadata for `authors`.
- `[done]` Add sitemap readiness gating by parity stage.

## Phase 5: Product Features

- `[done]` Replace the placeholder search input with a real search system.
- `[done]` Add full search results pages and keyboard navigation on top of quick search.
- `[done]` Add mobile navigation that preserves search and section discovery.
- `[done]` Add edit/source link support to all article-capable routes.
- `[done]` Add article continuation links for collection navigation.
- `[done]` Add feedback CTA component.
- `[done]` Add share buttons component.
- `[done]` Add RSS/feed generation and autodiscovery links.
- `[done]` Add client and server fallback pages for render-time failures.

## Phase 6: Testing

- `[done]` Add unit tests for slug generation.
- `[done]` Add unit tests for locale fallback resolution.
- `[done]` Add unit tests for route mapping.
- `[done]` Add unit tests for local path and asset rewriting.
- `[done]` Add Markdown fixture tests for:
  - `[done]` nested tutorial docs
  - `[done]` legacy `/blogs/*`
  - `[done]` `.en.md` and `.zh.md` variants
  - `[done]` inline HTML allowed by policy
  - `[done]` unsupported syntax failures
- `[done]` Add a regression check that article routes stay on-demand and avoid large static page payloads.
- `[done]` Add browser coverage for Mermaid hydration on real docs pages.
- `[done]` Add rollout stage audit coverage for `shadow`, `cutover`, and `growth`.

## Phase 7: Rollout

- `[done]` Define which routes are allowed into sitemap during shadow mode.
- `[done]` Define the cutover checklist for route classes.
- `[done]` Define the fallback/rollback path while MkDocs still exists.
- `[done]` Define observability signals for route-class and locale health.
- `[done]` Define which audit failures or metrics should block rollout or trigger rollback.
- `[done]` Verify production parity against indexed and legacy routes.
- `[done]` Isolate production build output from `.next` so `next dev` and `next start` can be validated side-by-side.

## Phase 8: Maintainability Simplification

- `[done]` Move primary navigation, home tracks, home explore links, and footer IA into one site registry.
- `[done]` Generate the site IA registry as a build artifact so client components never import filesystem helpers.
- `[done]` Split discovered sections from published sections so new content trees do not auto-enter nav/home/footer.
- `[done]` Make site IA collection seeds consume the content family registry instead of duplicating taxonomy.
- `[done]` Replace the route-kind `switch` with a collection family registry shared by manifest expansion and route builders.
- `[done]` Move route-to-page resolution behind one content-layer resolver so Next route builders stay thin.
- `[done]` Prebuild a document index and migrate metadata consumers off repeated ad-hoc `parseMarkdown` calls.
- `[done]` Collapse non-home content rendering onto one `DocsPage` model and one docs shell.
- `[done]` Split the monolithic content loader into home, collection, section, and resolver modules.
- `[done]` Restrict search artifact fallback to development so production and verify stay fail-fast.
- `[done]` Remove implicit `process.cwd()` assumptions from content roots and make generated artifact paths explicit.
- `[done]` Add a single `npm run verify` entry that covers typecheck, content tests, build, and audits.
- `[done]` Make isolated `distDir` verification resilient by cleaning stale build artifacts before building.

## Current Working Order

1. `[done]` Design the full system and backlog.
2. `[done]` Review the design iteratively with codex reviewers/subagents.
3. `[done]` Implement the content subsystem split.
4. `[done]` Add content-layer tests and finish migrating route/source lookups onto the manifest.
5. `[done]` Harden the Markdown pipeline.
6. `[done]` Implement heading extraction and article TOC data.
7. `[done]` Implement syntax highlighting parity for fenced code blocks.
8. `[done]` Restore search, git metadata, feedback CTA, and share actions.
9. `[done]` Implement remaining Markdown cutover blockers: admonitions and tabs.
10. `[done]` Collapse duplicated locale route logic.
11. `[done]` Move oversized article routes off static payload generation while preserving URLs.
12. `[done]` Verify build and audits still pass.
13. `[done]` Prebuild static search indexes and keep the search UX/API compatible.
14. `[done]` Codify sitemap rollout stages and rollback discipline.
15. `[done]` Restore Mermaid parity and add real-doc fixture coverage.
16. `[done]` Centralize site IA and footer/header discovery rules.
17. `[done]` Introduce a collection family registry for manifest + route loader reuse.
18. `[done]` Prebuild document metadata and reuse it across collections, navigation, sidebars, feeds, and search.
19. `[done]` Collapse content pages onto one docs shell while keeping the homepage custom.
20. `[done]` Add a single verification entry point and clean isolated `distDir` builds.
21. `[done]` Separate section discovery from section publication in the generated site IA.
22. `[done]` Make collection families the seed source for site IA taxonomy.
23. `[done]` Split the monolithic loader file into focused loader modules.
24. `[done]` Tighten search artifacts so only development can rebuild them on the fly.
