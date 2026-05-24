# Custom Frontend Architecture

## Goal

Replace the current MkDocs rendering layer with a static-export frontend while keeping:

- stable URLs
- static-first SEO behavior
- multilingual routing
- Markdown-based content authoring
- blog/docs/tutorial functionality
- current analytics and sharing surfaces

This document is the target architecture for the migration, not a description of the current stopgap code.

Deployment is now locked:

- **GitHub Pages** (static artifact deploy via `actions/deploy-pages`) is the deployment target
- output is host-agnostic static files, so any static host works, but GitHub Pages is the one we ship to
- true static export only
- no `API route`
- no production server runtime

The design goal is to imitate mature docs sites, not invent a new product shell. The main behavioral references are:

- `MkDocs Material` for structure, navigation habits, and Markdown-first information density
- `Nextra` and `Fumadocs` for the Next.js docs-shell model

## System Boundaries

### In scope

- route generation
- content ingestion from `docs/**`
- Markdown transformation and rendering
- SEO metadata generation
- multilingual path handling
- edit links, git metadata, blog metadata
- search and docs navigation
- legacy URL compatibility
- verification and rollout gates

### Out of scope

- rewriting docs content into React pages
- changing the authoring model away from Markdown
- inventing a separate docs/blog product shell before parity is reached
- depending on `next start`, response-writer pages, or any server runtime in production

Homepage exception:

- the homepage may diverge modestly, but should still stay structurally close to the original `docs/index.md`
- the blog index should stay structurally close to the original MkDocs blog index
- all other docs, tutorials, blog articles, and legacy blog routes remain parity-bound to Markdown content and one generic docs shell

## Core Principles

### 1. Content is the source of truth

The site should derive routes, metadata, and renderable content from the Markdown tree, not from duplicated hand-maintained route files.

### 2. One route manifest, many renderers

Route identity should be locale-agnostic first. Public URLs are derived from one canonical route manifest.

### 3. Framework-specific code stays thin

The content pipeline should not depend on request-time Next.js APIs. Next.js should consume the content subsystem, not define it.

That means:

- no new `getServerSideProps`
- no new `pages/api/*`
- no feature whose delivery depends on a live application server after build completes

### 4. Unsupported syntax must fail loudly

If a Markdown construct cannot be rendered correctly, the build should fail or the test suite should flag it. Silent degradation is not acceptable for migration parity.

### 5. Sitemap inclusion is gated by parity

Routes should only be emitted to the sitemap once they meet the current parity bar for the active rollout stage.

### 6. Docs-site imitation beats novelty

The UI should optimize for familiar docs-site behavior:

- stable header
- left sidebar
- main Markdown column
- right TOC when headings exist
- restrained article footer

Novelty is not a goal for parity-bound routes.

### 7. Static export is the delivery model

The site must be emitted as static files.

That means:

- page paths are enumerated at build time from the manifest
- search consumes generated static artifacts
- raw assets resolve to copied static files, not a proxy API
- `robots.txt`, `sitemap.xml`, RSS/feed, and shared OG assets are emitted during build

## Target Module Layout

```text
app/
  components/
    chrome/
    article/
    markdown/
    seo/
  lib/
    content/
      fs-index.ts
      discovery.ts
      model.ts
      registry.ts
      types.ts
      manifest.ts
      markdown.ts
      rewrite.ts
      collections.ts
      assets.ts
      loaders.ts
    seo/
    routing/
    metadata/
    git/
  pages/ or src/app/
  public/
```

## Content Subsystem

### Responsibilities

- index repository files once
- generate content artifacts once per build snapshot
- normalize Markdown source identity
- resolve locale variants
- derive canonical route identities
- assemble collections for tutorials, blog, legacy blog, and section docs
- render Markdown into HTML or component-ready structures
- rewrite local links and local asset URLs
- expose page loaders and sitemap candidates

### Non-responsibilities

- rendering full page chrome
- emitting framework route files
- injecting analytics or other runtime scripts

### Generated artifacts

The runtime should consume generated artifacts rather than reconstructing the content model ad hoc.

Required artifacts:

- `documents.json`
- `content-model.json`
- `manifest.json`
- `site-sections.json`
- `search/*.json`
- static output files for:
  - `robots.txt`
  - `sitemap.xml`
  - `feed.xml`
  - `zh/feed.xml`

### Planned modules

#### `content/fs-index.ts`

Owns:

- filesystem walking
- file existence lookups
- cacheable repo inventories for `docs/` and `site/`

Exports:

- docs file set
- site file set
- top-level directory inventory

#### `content/types.ts`

Owns:

- parsed markdown types
- manifest record types
- page loader result types
- collection entry types

#### `content/discovery.ts`

Owns:

- source discovery for tutorial docs
- source discovery for blog and legacy blog entries
- section-route discovery

Design notes:

- this is the source-discovery boundary
- it should not leak framework or page concerns

#### `content/model.ts`

Owns:

- generated collection views over discovered content
- artifact read/write helpers for the content model

Design notes:

- collections should read from the generated model
- production should consume the artifact, not rescan docs live

#### `content/registry.ts`

Owns:

- collection family definitions
- collection family extension points
- collection-family-driven site IA seeds

Design notes:

- adding a new collection family should be a registry change first
- do not keep a second runtime switch table for collection behavior

#### `content/manifest.ts`

Owns:

- source-to-route identity normalization
- locale variant resolution
- canonical route manifest artifact generation
- sitemap candidate list

Exports:

- `ContentManifest`
- helpers to query pages by route identity or public path

#### `content/markdown.ts`

Owns:

- frontmatter parsing
- excerpt generation
- title/description fallback rules
- Markdown parsing pipeline
- heading extraction
- table of contents extraction

Design notes:

- keep Markdown parsing isolated from route logic
- expose typed render output, not only raw HTML strings

#### `content/rewrite.ts`

Owns:

- relative link rewriting
- absolute local path rewriting
- raw asset path resolution
- link validation helpers used by tests

#### `content/collections.ts`

Owns:

- tutorial collection
- blog collection
- legacy blog collection
- section collections

Design notes:

- collections are views over the generated content model, not separate scanners

#### `content/assets.ts`

Owns:

- raw asset export mapping
- public asset URL building
- MIME type mapping

#### `content/loaders.ts`

Owns:

- page-facing loader APIs
- composition of manifest + collections + markdown rendering
- homepage/blog/tutorial/section data assembly

Design notes:

- loaders consume artifacts plus registry definitions
- loaders do not rediscover content families on their own

## Routing Model

### Canonical route identity

Every page should have one locale-agnostic route identity, for example:

- `home`
- `tutorials/index`
- `tutorials/38-btf-uprobe/test-verify`
- `blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources`
- `blogs/bpftime`
- `section/bpftime/llvmbpf`

### Public URL expansion

Public URLs are derived from route identity plus locale:

- English: `/tutorials/38-btf-uprobe/test-verify/`
- Chinese: `/zh/tutorials/38-btf-uprobe/test-verify/`

### Legacy behavior

The route manifest must preserve:

- `/blogs/*`
- dated `/blog/YYYY/MM/DD/*`
- section paths already indexed by search engines

### Exit criteria for duplicated route files

Do not add new `en` and `zh` route implementations once the shared route manifest exists. Shared page helpers should become the only allowed path for new page logic.

### Build-time materialization

Public routes must be materialized at build time from the manifest.

Rules:

- request-time slug resolution is not allowed in the target state
- every exported page must have a manifest-backed static path
- verification must fail if a manifest-backed path is missing from the export

## Site IA Model

### Discovery versus publication

New content trees can be discovered without being published.

Rules:

- collection families seed the IA automatically
- generic top-level sections may be discovered automatically
- nothing newly discovered is implicitly published into the header, homepage, or footer
- publication overrides are explicit and validated

Validation requirements:

- unknown override keys are errors
- duplicate orders are errors
- missing labels are errors

## Markdown Pipeline

### Required supported features

- frontmatter
- fenced code blocks
- tables
- footnotes
- task lists
- heading anchors
- TOC extraction
- local images and static assets
- inline HTML within the allowed policy
- admonitions
- tabs

### Trust model

- repository-authored Markdown is treated as trusted input
- raw HTML is still constrained to an allowlist-based sanitization policy
- unsupported HTML or unsupported Markdown extensions should fail validation

Current syntax inventory:

- current docs do not rely on MkDocs admonition or tabs syntax yet
- raw HTML on parity-bound content routes is narrow: centered image wrappers, inline caption blocks, and a small number of inline `<br>` usages
- homepage hero/container markup is explicitly outside parity scope and can be replaced by native React/Tailwind markup
- literal `<script>` and full HTML document samples currently appear inside fenced code blocks and must stay escaped, not become executable DOM

### Rendering strategy

Near term:

- render to HTML for parity and speed

Longer term:

- allow a component-oriented path for advanced blocks where that reduces ad hoc HTML handling

## SEO and Metadata

### Required outputs

- canonical
- `hreflang`
- `robots.txt`
- `sitemap.xml`
- title
- description
- Open Graph tags
- article metadata for blog posts
- stable `html lang`

### Design rules

- metadata comes from the content subsystem, not from page components inventing their own rules
- sitemap generation should be backed by a readiness gate
- Open Graph images must be static assets or bounded build-time generated outputs

## Redirect and Rewrite Strategy

### Requirements

- old indexed URLs must continue to resolve
- canonical URLs must remain stable across the coexistence period
- route cutover must not require rewriting the whole site at once

### Design

- keep redirects as data derived from the route manifest, not as scattered page-level logic
- support three states per route:
  - served by legacy site
  - served by Next.js site
  - redirected to a new canonical path
- keep 301 rules explicit for any path that changes shape
- verify redirect behavior in audits for legacy and indexed paths

## Cache and Revalidation Model

### Requirement

Manifest-based export needs an explicit freshness model, otherwise content updates can drift by locale or route class.

### Design

- default mode is full rebuild for static content
- authors, last-updated, and search index generation must use the same content snapshot as the page build
- no runtime revalidation policy is assumed in the target design

## Search

### Target

- static generated search artifact with multilingual support
- docs and blog content searchable from one UI

### Design

- search indexing runs from generated content artifacts, not from live page scraping
- production and verification should fail fast when search artifacts are missing
- development may fall back to rebuild only where explicitly documented
- search artifacts should stay compact; keep normalized terms, not unbounded full-text bodies
- the search UI must not call a local API in the target state

## Git-backed Metadata

### Required

- last updated
- authors
- publish date for blog
- edit link

### Design

- edit link is content-manifest data
- authors and last-updated should come from a build-time git metadata step
- metadata collection must be cacheable and testable outside the page layer

## UI Systems

### Site chrome

- shared header/footer
- locale switcher
- navigation derived from config, not hardcoded per page file
- one generic docs shell for all non-home Markdown routes

### Article layout

- title block
- metadata row
- TOC slot
- source/edit link
- feedback/share/footer slots

### Collection pages

- tutorials index
- blog index
- legacy blog index
- section landing pages

Design rule:

- these are still docs pages, not product landing pages

### Home and blog structure

- homepage may have a stronger hero treatment, but should still mirror the original MkDocs homepage information architecture
- blog index should remain a docs/blog index, not a magazine or marketing page

## Verification Model

### Existing top-level checks

- sitemap HTTP audit
- sitemap parity audit
- browser smoke audit
- link and asset crawl

### Missing checks that must be added

- content-layer unit tests
- Markdown fixture tests
- export completeness checks
- parity-stage gating for sitemap inclusion
- rollout health signals by route class and locale
- assertions that no `pages/api/*` and no `getServerSideProps` remain

### Delivery checks

Final verification must be static-first:

- build/export the app
- serve the exported directory from a dumb static server
- run browser/link/SEO checks against that static server
- do not treat `next start` as the release path

## Rollout Stages

### Stage 1: Shadow mode

Must have:

- route parity
- SEO parity
- raw asset parity
- docs/blog/tutorial rendering parity
- analytics parity

### Stage 2: Cutover ready

Must have:

- search
- TOC
- heading anchors
- code highlighting
- admonitions
- tabs
- authors
- last updated
- edit links

### Stage 3: Growth parity

Must have:

- feedback CTA
- share buttons
- improved OG strategy
- RSS/feed polish
- homepage/product CTA polish

## Observability and Rollout Signals

### Requirement

Route-class rollback is only useful if the system exposes clear signals that a route class or locale has regressed.

### Design

- track health by route class and locale, not only site-wide
- define minimum rollout signals for:
  - render failures
  - missing localized content fallback events
  - broken asset/link rates from audits
  - search/index generation failures
  - unexpected sitemap exclusions or route drops
- make shadow-mode verification explicit for both English and Chinese paths
- keep observability decisions close to the rollout matrix so readiness and rollback use the same signals

## Rollback Model

### Requirement

If a route class regresses after cutover, the site needs a route-level fallback instead of requiring a full rollback.

### Design

- rollout decisions should be made per route class, not only per deployment
- keep legacy route availability until the new route class passes its cutover gate
- define a route readiness matrix that can be used to exclude route classes from sitemap and production cutover
- any future proxy or middleware cutover layer must support falling a route class back to legacy behavior without changing content logic

## Implementation Order

1. Split the content subsystem out of the monolith.
2. Introduce the shared route manifest and loader boundaries.
3. Add content-layer tests against real fixtures.
4. Lock down the Markdown trust/sanitization policy.
5. Collapse duplicated locale route implementations onto shared helpers.
6. Remove API and runtime delivery assumptions route class by route class.
7. Finish the static export path and make the verifier/static preview path the only release contract.
