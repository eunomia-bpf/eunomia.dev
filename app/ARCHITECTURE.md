# Custom Frontend Architecture

## Goal

Replace the current MkDocs rendering layer with a Next.js frontend while keeping:

- stable URLs
- static-first SEO behavior
- multilingual routing
- Markdown-based content authoring
- blog/docs/tutorial functionality
- current analytics and sharing surfaces

This document is the target architecture for the migration, not a description of the current stopgap code.

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
- redesigning every landing page before parity is reached

Homepage exception:

- the homepage may diverge into a bespoke Tailwind landing page before full docs parity
- all other docs, tutorials, blog, and legacy blog routes remain parity-bound to Markdown content

## Core Principles

### 1. Content is the source of truth

The site should derive routes, metadata, and renderable content from the Markdown tree, not from duplicated hand-maintained route files.

### 2. One route manifest, many renderers

Route identity should be locale-agnostic first. Public URLs are derived from one canonical route manifest.

### 3. Framework-specific code stays thin

The content pipeline should not depend on `getStaticProps`, `getStaticPaths`, or App Router APIs. Next.js should consume the content subsystem, not define it.

### 4. Unsupported syntax must fail loudly

If a Markdown construct cannot be rendered correctly, the build should fail or the test suite should flag it. Silent degradation is not acceptable for migration parity.

### 5. Sitemap inclusion is gated by parity

Routes should only be emitted to the sitemap once they meet the current parity bar for the active rollout stage.

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

#### `content/manifest.ts`

Owns:

- source-to-route identity normalization
- locale variant resolution
- canonical route manifest generation
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

- collections are views over the manifest, not separate scanners

#### `content/assets.ts`

Owns:

- raw asset serving resolution
- public asset URL building
- MIME type mapping

#### `content/loaders.ts`

Owns:

- page-facing loader APIs
- composition of manifest + collections + markdown rendering
- homepage/blog/tutorial/section data assembly

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
- Open Graph images can start with a shared default but need a route-class-specific strategy later

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

Manifest-based rendering needs an explicit freshness model, otherwise content updates can drift by locale or route class.

### Design

- default mode is full rebuild for static content until a more granular path is justified
- if route-level revalidation is added later, it must be driven by manifest records and not ad hoc page logic
- authors, last-updated, and search index generation must use the same content snapshot as the page build
- cache invalidation policy must be documented per route class before ISR or on-demand revalidation is introduced

## Search

### Target

- static search index with multilingual support
- docs and blog content searchable from one UI

### Design

- `Pagefind` remains the preferred path unless a stronger static alternative is required
- search indexing should run from the generated content manifest, not by scraping rendered pages
- the UI should not ship as a dead input before the index exists

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

## Verification Model

### Existing top-level checks

- sitemap HTTP audit
- sitemap parity audit
- browser smoke audit
- link and asset crawl

### Missing checks that must be added

- content-layer unit tests
- Markdown fixture tests
- payload-size regression check for large pages
- parity-stage gating for sitemap inclusion
- rollout health signals by route class and locale

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
6. Fill cutover-blocking parity features and fold payload-size work into that stage.
7. Move to App Router only after the shared abstractions hold.
