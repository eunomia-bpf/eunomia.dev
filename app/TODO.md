# App Migration TODO

This is the execution backlog for the custom frontend.

Status markers:

- `[todo]` not started
- `[doing]` in progress
- `[done]` implemented and verified
- `[blocked]` waiting on a prerequisite or decision

## Static Export Lock

Hard target:

- `Cloudflare Pages static`
- true static export only
- no `pages/api/*`
- no `getServerSideProps`
- no production runtime server dependency

All remaining work is ordered around removing API, SSR, and runtime assumptions until the app can be exported and served as static files only.

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

## Phase 9: Docs-Site Convergence

- `[doing]` Keep the migration visually aligned with mature docs sites instead of inventing a separate product shell.
- `[doing]` Keep the homepage structurally close to the original MkDocs homepage even when using React/Tailwind.
- `[doing]` Keep the blog index structurally close to the original MkDocs blog index rather than a media landing page.
- `[doing]` Keep every non-home Markdown page on one generic docs shell.
- `[doing]` Finish moving collection-family behavior behind the registry so new families are single-point extensions.
- `[doing]` Finish consuming generated `content-model` and `manifest` artifacts at runtime instead of rebuilding live views ad hoc.
- `[doing]` Validate discovered-versus-published site IA overrides with fail-fast generation rules.
- `[doing]` Keep search on compact generated artifacts and document the dev-versus-prod fallback rules.
- `[todo]` Add a development watch flow so docs, IA, and search artifacts stay fresh without manual restart.
- `[done]` Remove the remaining runtime delivery assumptions so docs-shell convergence sits on a true static-export base.

## Phase 10: Static Export and Cloudflare Pages

- `[done]` Lock `static export only / no API / Cloudflare Pages static` into `README`, `ARCHITECTURE`, and `TODO`.
- `[done]` Remove `/api/search` and make quick search plus `/search` consume generated static search artifacts.
- `[done]` Remove `/api/raw-assets` and replace runtime proxying with build-time asset copying plus static URLs.
- `[done]` Remove `/api/og` and decide the static OG strategy.
- `[done]` Replace runtime `feed.xml`, `zh/feed.xml`, `sitemap.xml`, and `robots.txt` pages with build-time emitted static files.
- `[done]` Replace `getServerSideProps` content/search delivery with build-time path enumeration from the manifest.
- `[done]` Configure the app for true static export and add an export-oriented build contract.
- `[done]` Rewrite verification around exported files served from a dumb static server instead of `next start`.
- `[done]` Add explicit checks that no `pages/api/*` and no `getServerSideProps` remain.
- `[done]` Validate the exported artifact against Cloudflare Pages static assumptions before deployment handoff.

## Current Working Order

1. `[doing]` Keep docs, rollout rules, and parity status aligned with the static-only deployment contract.
2. `[doing]` Keep the homepage and blog index structurally close to the original MkDocs site while preserving the static-only build.
3. `[doing]` Finish moving collection-family behavior behind the registry so new families are single-point extensions.
4. `[doing]` Finish consuming generated `content-model` and `manifest` artifacts at runtime instead of rebuilding live views ad hoc.
5. `[doing]` Validate discovered-versus-published site IA overrides with fail-fast generation rules.
6. `[doing]` Keep search on compact generated artifacts and document the dev-versus-prod fallback rules.
7. `[todo]` Add a development watch flow so docs, IA, and search artifacts stay fresh without manual restart.
