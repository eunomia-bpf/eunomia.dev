# eunomia.dev Custom Frontend Plan

## Decision

If `eunomia.dev` moves away from MkDocs, the replacement must be:

- `Next.js` used as a static site compiler, not as a server runtime
- `Cloudflare Pages static` as the deployment target
- true static export output only
- Markdown content kept in-repo
- no `API route`
- no runtime server dependency in production

This is now a hard architectural constraint, not a preference.

A pure client-only SPA would still lose too much by default: HTML-first SEO, stable metadata generation, language alternates, sitemap generation, and predictable crawlability. But a server-rendered Next app is also no longer acceptable for the deployment target. The site must compile to static files.

## Current implementation

The current repository still reflects a compatibility-first `Next.js` slice under this `app/` directory.

That compatibility slice was useful for parity work, but it is no longer the target architecture.

The current implementation still has server assumptions that must be removed:

- `getServerSideProps`-backed content and search routes
- `pages/api/*` endpoints for search, raw assets, and OG generation
- runtime generation of feed, sitemap, and robots responses
- verification flows that assume `next start`

The migration target is now stricter:

- build a static artifact
- serve only static files
- deploy that artifact to `Cloudflare Pages static`

## Design Lock

This app should imitate mature docs sites, not invent a new product shell.

The practical reference set is:

- `MkDocs Material` for information density, navigation habits, and Markdown-first behavior
- `Nextra` and `Fumadocs` for the Next.js docs-shell model, not for bespoke landing-page styling

That means the design rules are:

- the homepage may be bespoke, but it should stay structurally close to the original `docs/index.md` homepage instead of becoming a marketing landing page
- the blog index should behave like a docs/blog index page, not a separate media product
- every non-home Markdown route should render through one generic docs shell
- collection-specific behavior should live in the collection family registry, not in scattered page or loader switches
- runtime should consume generated content artifacts rather than reconstructing the content model ad hoc
- the build must emit a true static export artifact; no production runtime is allowed to finish page resolution on demand
- search, raw assets, feed, sitemap, robots, and OG assets must be generated or copied at build time
- `pages/api/*` and other server-only routes are not allowed in the target state
- discovered content and published navigation are different states; new content trees do not auto-enter nav, home, or footer
- production export, local preview, and verification should fail fast when required artifacts are missing; only explicitly documented development flows may rebuild on the fly

## Engineering Status

This code is good enough as a working migration slice, but not yet good enough as the long-term production architecture.

What is already solid:

- strict TypeScript is enabled
- the app builds successfully with static output for the current route set
- sitemap-driven audits pass against the current implementation
- the app serves real Markdown instead of demo-only placeholder data
- route identity is derived from one canonical manifest for both locales
- search now runs against prebuilt static indexes under `.generated/search`
- content discovery and route modeling now have a generated-artifact direction: `documents`, `content-model`, `manifest`, `site-sections`, `search`
- Mermaid diagrams now render on real docs pages instead of silently degrading to code blocks
- rollout stages, sitemap gating, and rollback rules are documented and tested

What is not solid enough yet:

- the app still needs a full static-export refactor
- `docs/blog` and `docs/blogs` still overlap semantically
- the rendering pipeline still allows a documented raw-HTML subset
- production delivery still has to remove all API and SSR assumptions

## Docs-Site Emulation Rules

The UI target is a conservative docs shell:

- sticky header
- left sidebar
- main Markdown column
- right TOC when the page has headings
- article footer with edit/share/feedback metadata

What should not happen:

- tutorials, docs, and blog pages drifting into separate visual systems
- collection pages becoming card-heavy product pages
- homepage and blog index becoming detached from the original MkDocs information architecture

The one allowed exception is the homepage hero treatment, but even there the structure should still map back to the original MkDocs homepage content:

- primary intro
- primary actions
- project sections
- tutorial/docs/blog entry points

Operational discipline now lives in [ROLLOUT.md](./ROLLOUT.md). That file remains the source of truth for:

- which route classes are allowed into the sitemap in `shadow`, `cutover`, and `growth`
- what must pass before cutover
- what should trigger rollback to `shadow`

## Main Problems

### 1. The app is still not a true static export

The current site still assumes runtime delivery in several places:

- `getServerSideProps` content and search routes
- `pages/api/*` endpoints
- response-writer pages for feed, sitemap, and robots
- runtime raw-asset proxying

Until those are removed, the app cannot be deployed as `Cloudflare Pages static`.

### 2. Markdown rendering has an explicit trust boundary that still needs careful maintenance

The current pipeline allows raw HTML and then injects the rendered result into the page:

- `remarkRehype({ allowDangerousHtml: true })`
- `rehypeRaw`
- `dangerouslySetInnerHTML`

That is acceptable for trusted repository content, but it remains a deliberate policy surface that needs continued allowlist maintenance and fixture coverage.

### 3. Search and assets still need to move fully to build time

The content model already has generated `search` and `manifest` artifacts, but the delivery layer still needs to finish the transition:

- search must stop calling a live endpoint
- raw assets must stop resolving through an API proxy
- metadata files and OG strategy must become static outputs

### 4. Some UI features are still transitional rather than final systems

Examples:

- the search box is now backed by a prebuilt locale-aware content index, but ranking remains intentionally simple
- Open Graph support is route-aware, but the visual system is still transitional
- the page footer now restores git metadata, feedback CTA, and share actions, but the long-term chrome system is still transitional

These are feature gaps, but they also affect engineering quality because they leave temporary structure in the codebase.

## Missing Architectural Decisions

These decisions should be written down before more implementation lands.

### 1. Content module decomposition strategy

The repo needs a defined split for the current content monolith.

Recommended decision:

- `content/fs-index.ts`: walk the repo once and build the file inventory
- `content/discovery.ts`: perform source discovery from the file inventory
- `content/model.ts`: emit generated collection views over the discovered content
- `content/registry.ts`: define collection families and the single extension point for collection behavior
- `content/manifest.ts`: derive normalized route/content records from the generated content model
- `content/markdown.ts`: parse frontmatter and Markdown into typed render inputs
- `content/collections.ts`: expose blog/tutorial/section collections from the generated content model
- `content/assets.ts`: resolve static asset paths and public URLs
- `content/loaders.ts`: assemble page props from manifest records and the generic docs shell

The dependency direction should stay one-way:

- file inventory -> discovery -> content-model -> manifest/site-ia/search -> loaders -> pages/components

Pages should not scan the filesystem directly.

Artifact boundary:

- `documents.json`
- `content-model.json`
- `manifest.json`
- `site-sections.json`
- `search/*.json`

### 2. Locale routing architecture

The project needs one canonical route manifest and locale expansion on top of it, not duplicated route trees as the long-term pattern.

Recommended decision:

- define route identity without locale first, for example `tutorials/38-btf-uprobe/test-verify`
- expand to public URLs via a locale-aware route builder
- treat `/zh/` as a path prefix concern, not a separate page implementation concern
- do not add more duplicated `en`/`zh` route files while the refactor is in progress

### 3. Markdown trust and sanitization policy

The current design assumes repository-authored content is trusted, but that assumption is not documented.

Recommended decision:

- repository Markdown may contain a controlled subset of raw HTML
- rendered output must still pass through a documented allowlist or sanitization step
- feature syntax such as tabs and admonitions should become typed Markdown extensions or MDX components, not free-form HTML fragments
- if a construct is unsupported, the build should fail instead of silently degrading

Current inventory:

- the current docs tree does not use MkDocs admonition or tab syntax yet
- current raw HTML usage on parity-bound routes is concentrated in a few places: centered image wrappers in docs/blog posts, inline `<p><em>` captions, and a small number of inline `<br>` usages
- homepage hero markup is not a parity blocker because the homepage is allowed to become a bespoke Tailwind page
- the first hardening step should preserve `div`, `img`, inline text tags, and anchor/image attributes used by those pages while stripping scripts, event handlers, and dangerous URL schemes

### 4. Parity scope and rollout cut-lines

The design needs explicit milestones, otherwise "finish parity" stays too vague.

Recommended decision:

- `Shadow mode`: routes, SEO metadata, raw assets, docs/blog rendering, sitemap parity, analytics
- `Cutover ready`: search, TOC, heading anchors, admonitions, tabs, edit links, authors, last updated
- `Growth parity`: feedback CTA, share buttons, RSS/social image polish, homepage/product polish

The sitemap should only contain routes that satisfy the current cut-line for the environment being tested.

### 5. Static export delivery model

The design is no longer "static first". It is "static only".

Recommended decision:

- every public page must be emitted at build time
- every machine-readable endpoint must become a static file in the export output
- every client interaction must consume generated static artifacts rather than calling a local API
- no production or preview environment may depend on `next start`

### 6. Framework boundary

The current Pages Router implementation is only a compatibility shell while the content pipeline is refactored toward exportability.

Recommended decision:

- no new core abstractions should depend on `getServerSideProps`
- no new feature should introduce `pages/api/*`
- the content subsystem should stay framework-agnostic so route enumeration can feed a static export build
- future framework changes are allowed only if they preserve the static-export contract

### 7. Site IA publication model

Discovery and publication are separate concerns.

Recommended decision:

- discovered sections come from content and collection-family seeds
- published sections are the subset explicitly allowed into header, homepage, and footer surfaces
- overrides must be validated, not silently accepted
- duplicate ordering, unknown keys, and missing labels are build-time failures

### 8. Search artifact strategy

Search should be treated as a generated artifact, not an always-live content scan.

Recommended decision:

- search indexes are generated from the content manifest and document index
- production, `start`, and verification consume prebuilt search artifacts
- development may rebuild on demand only where explicitly allowed
- search artifacts should store compact search terms, not unbounded full-page bodies

## Non-Negotiables

The new frontend must preserve:

- every public URL that currently matters, including `/`, `/zh/`, docs routes, tutorials, blog, and legacy `/blogs/*` links
- current SEO infrastructure: `robots.txt`, `sitemap.xml`, canonical URLs, Open Graph tags, alternate language links, descriptive titles, and page descriptions
- current site behavior: search, blog index, dated posts, docs pages, multilingual routing, edit links, analytics, feedback entry points, and share buttons
- the existing Markdown content model instead of rewriting hundreds of documents as React pages
- `Cloudflare Pages static` compatibility
- no production API routes
- no runtime server dependency after build/export

## Current Site Inventory

Based on the repository today:

- `468` Markdown files under `docs/`
- `184` Chinese Markdown files
- `72` blog posts in `docs/blog/posts`
- `48` legacy blog files still living in `docs/blogs`
- MkDocs currently provides i18n, blog, search, tags, Git metadata, analytics, feedback, social cards, and edit links from one config file

This means the migration problem is mostly about rendering, routing, and parity, not about creating new content.

## Recommended Stack

## Rendering and Routing

- `Next.js` configured for true static export
- path-based i18n: `/` for English, `/zh/` for Chinese
- build-time page enumeration from the content manifest
- static files for `robots.txt`, `sitemap.xml`, RSS/feed, and any shared OG assets

## Content Pipeline

- keep source content in Markdown/MDX under a versioned content tree
- parse frontmatter, headings, TOC, and content metadata during build
- support current Markdown features with remark/rehype plugins:
  - fenced code blocks
  - syntax highlighting
  - tables
  - footnotes
  - task lists
  - heading anchors
  - admonitions/callouts
  - tabs
  - inline HTML where needed

## Search

- static generated search indexes under `.generated/search`
- one search surface across docs, tutorials, blog, and legacy blog
- fail-fast artifact loading in non-development environments
- preserve on-page search behavior for docs and blog content without any runtime `/api/search`

## Styling

- `Tailwind CSS` for layout and components
- keep the visual target close to mature docs sites rather than inventing a separate product design language
- homepage may be mildly customized, but blog/home structure should stay close to the original MkDocs site
- all non-home Markdown routes use one generic docs shell

## Git and Content Metadata

- preserve `last updated`
- preserve `authors`
- preserve `Edit this page`
- preserve publish dates for blog posts

## Migration Phases

## Phase 0: Lock the Static Constraint

- write `static export only / no API / Cloudflare Pages static` into design docs
- inventory every current runtime dependency
- make `legacy /blogs/*` and other public URLs explicit before delivery changes

## Phase 1: Remove Runtime Endpoints

- replace `/api/search` with static search artifact consumption
- replace `/api/raw-assets` with build-time asset copying and static URL rewriting
- replace `/api/og` with static OG assets or a bounded build-time generation strategy

## Phase 2: Emit Static Machine-Readable Files

- build `robots.txt`
- build `sitemap.xml`
- build `feed.xml` and `zh/feed.xml`
- ensure these files ship as static outputs, not response-writer pages

## Phase 3: Replace SSR Page Delivery

- remove `getServerSideProps`
- enumerate content paths from the manifest at build time
- keep search as a static shell with client-side querying over generated artifacts
- preserve all existing public URLs inside the static export

## Phase 4: Make Verification Static-First

- replace `next start`-based verification with “build/export + serve static dir”
- remove server-only runtime audits
- add explicit checks that no `pages/api/*` and no `getServerSideProps` remain

## Phase 5: Cloudflare Pages Static Handoff

- deploy only the exported static directory
- keep the deployment contract strict so regressions back to API/SSR fail early
- ship only after static-export parity checks pass

## Definition of Done

The migration is only done when:

- the new site serves exported static files for all major routes
- SEO parity checks pass
- browser smoke tests pass
- internal link crawl passes
- old URLs continue to resolve or redirect cleanly
- editors can still update content by modifying Markdown files in the repo
- there are no `pages/api/*` routes
- there are no `getServerSideProps` dependencies
- the output can be deployed to `Cloudflare Pages static` without a server runtime

## Design Backlog

### 1. Split the content pipeline

Goal:

- break `lib/content.ts` into a stable build-time content subsystem

Design:

- create a `ContentManifest` as the single source of truth for routes, source files, alternates, and asset references
- make filesystem walking happen once per build, not indirectly across unrelated helpers
- have page loaders consume manifest records instead of reconstructing routes ad hoc

Deliverables:

- `content/fs-index.ts`
- `content/discovery.ts`
- `content/model.ts`
- `content/registry.ts`
- `content/manifest.ts`
- `content/markdown.ts`
- `content/collections.ts`
- `content/assets.ts`
- `content/loaders.ts`

### 2. Remove duplicated locale route implementations

Goal:

- stop maintaining parallel English and Chinese page logic

Design:

- introduce shared render factories for landing pages, article pages, and collection pages
- keep locale-specific copy in config/data, not in page structure
- collapse route generation onto one locale-agnostic manifest and one locale-expansion helper

Deliverables:

- shared page factories or route render helpers
- one canonical route manifest
- one locale URL builder

### 2a. Keep collection families as the single extension point

Goal:

- adding a new collection family should not require touching multiple unrelated runtime modules

Design:

- collection family discovery, routing, landing-card behavior, and sidebar behavior should be defined from the registry
- page resolution should consume manifest records and family descriptors rather than a second runtime switch table

Deliverables:

- registry-owned family definitions
- collection loaders without family-specific runtime duplication

### 3. Harden the Markdown pipeline

Goal:

- make Markdown rendering explicit, typed, and safe

Design:

- keep Markdown as the source format
- adopt a documented allowlist for raw HTML handling
- implement admonitions, tabs, and other advanced syntax as explicit parser plugins or component transforms
- replace loose AST mutation with typed utility helpers

Deliverables:

- sanitization/allowlist policy
- plugin list for supported Markdown extensions
- failure policy for unsupported constructs

### 4. Add content-layer tests

Goal:

- protect the migration at the transformation boundary, not only at the browser boundary

Design:

- add fast unit tests for slug derivation, locale fallback, route mapping, and asset rewriting
- add fixture tests for Markdown samples that represent real docs edge cases
- keep the existing browser and sitemap audits as top-level regression checks

Deliverables:

- fixture corpus under `test/fixtures/`
- unit test suite for content helpers
- regression fixtures for `/blogs/*`, nested tutorial docs, `.en.md`, `.zh.md`, and raw assets

### 5. Reduce large page payloads

Goal:

- keep SEO parity while avoiding oversized serialized page props

Design:

- separate metadata/index data from heavy article body rendering
- do not optimize this inside the current monolith first; land it after the content split
- target the heaviest blog pages first and use them as the benchmark set

Deliverables:

- benchmark list of heavy pages
- agreed rendering strategy for large articles
- payload-size regression check in the verification flow

### 6. Finish parity features with clear milestones

Goal:

- stop treating feature parity as one undifferentiated bucket

Design:

- `Shadow mode` blockers: route parity, SEO parity, raw asset parity, analytics, sitemap parity
- `Cutover ready` blockers: search, TOC, heading anchors, admonitions, tabs, edit links, authors, last updated
- `Growth parity` features: RSS/social image polish and homepage CTA polish

Deliverables:

- milestone checklist
- route readiness matrix
- clear cutover gate for sitemap inclusion

### 7. Keep the UI aligned with mature docs sites

Goal:

- prevent the migration from recreating a separate product-shell design system

Design:

- homepage and blog index stay structurally close to the original MkDocs site
- all non-home Markdown pages use one conservative docs shell
- visual changes should favor clarity and familiarity over novelty

Deliverables:

- one generic docs shell
- MkDocs-like home/blog structure
- no collection-specific visual systems outside the homepage exception

## Suggested Implementation Order

1. Split the content pipeline
2. Stabilize one locale through the new manifest and loader boundaries
3. Add sanitization and typed Markdown extension handling
4. Collapse locale routing onto shared page abstractions
5. Fill cutover-blocking parity features
6. Remove runtime assumptions route class by route class until the site is a true static export

## Verification Snapshot

Last locally verified on `2026-03-07`:

```bash
cd app
npm run dev -- --hostname 0.0.0.0 --port 3000
npm run test:content
NEXT_DIST_DIR=.next-prod NEXT_PUBLIC_SITE_URL=http://127.0.0.1:3001 npm run build
NEXT_DIST_DIR=.next-prod NEXT_PUBLIC_SITE_URL=http://127.0.0.1:3001 npm run start -- --hostname 127.0.0.1 --port 3001

cd ../test
BASE_URL=http://127.0.0.1:3000 npm run audit:browser
BASE_URL=http://127.0.0.1:3001 npm run audit
```

Expected current result:

- `audit:http` passes on all sitemap routes
- `audit:parity` passes with zero missing legacy sitemap paths
- `audit:browser` passes
- `audit:links` passes
- `audit:runtime` passes after probing the heaviest tutorial, blog, and section routes
