# eunomia.dev Custom Frontend Plan

## Decision

If `eunomia.dev` moves away from MkDocs, the replacement should be:

- `Next.js` App Router
- `React`
- `Tailwind CSS`
- Markdown/MDX content kept in-repo
- static rendering first (`SSG`/`ISR`), not a client-only SPA

A pure React SPA would lose too much by default: HTML-first SEO, stable metadata generation, language alternates, sitemap generation, and predictable crawlability.

## Current implementation

The first runnable slice in this repository uses `Next.js` with the Pages Router under this `app/` directory.

That is a deliberate compatibility choice for the initial implementation:

- it lets us emit route-specific `<html lang>` values immediately
- it keeps static routes and dated blog routes simple
- it gives us a working baseline before the full Markdown rendering pipeline is built

The long-term target can still move to a more modern App Router structure once parity is locked down.

## Engineering Status

This code is good enough as a working migration slice, but not yet good enough as the long-term production architecture.

What is already solid:

- strict TypeScript is enabled
- the app builds successfully with static output for the current route set
- sitemap-driven audits pass against the current implementation
- the app serves real Markdown instead of demo-only placeholder data

What is not solid enough yet:

- the content pipeline is too centralized in `lib/content.ts`
- route files are duplicated across locale and content type
- the rendering pipeline trusts raw HTML too broadly
- tests are strong at the black-box level but weak at the unit and fixture level
- some large blog pages still generate oversized static page payloads

## Main Problems

### 1. `lib/content.ts` is carrying too many responsibilities

`lib/content.ts` is currently the routing layer, file indexer, Markdown parser, blog registry, locale resolver, URL rewriter, sitemap source, and raw asset helper in one file.

That makes changes risky because unrelated concerns are coupled together. A small fix to localized routing or asset rewriting can accidentally affect blog lookup, sitemap generation, or Markdown rendering.

Current risk signal:

- `lib/content.ts` is over `1000` lines
- it mixes build-time concerns and request-time helpers
- it owns both data access and presentation-oriented transformations

### 2. Route implementation is duplicated and will drift

The English and Chinese page files under `pages/` are almost identical for:

- home
- tutorials
- blog
- legacy blogs
- generic sections

This is manageable for the first slice, but it is a bad steady-state design. Every new feature now has to be wired twice, and every SEO or layout fix has two chances to drift.

### 3. Markdown rendering has an explicit trust boundary but no safety layer

The current pipeline allows raw HTML and then injects the rendered result into the page:

- `remarkRehype({ allowDangerousHtml: true })`
- `rehypeRaw`
- `dangerouslySetInnerHTML`

That may be acceptable for trusted repository content, but it should be treated as a deliberate security policy, not an accidental default. Right now the policy is implicit and undocumented.

### 4. Tests do not protect the transformation logic well enough

The current `test/` directory is useful and already catches routing, SEO, and link regressions, but it mostly validates the app from the outside.

What is still missing:

- unit tests for slug generation
- unit tests for locale fallback resolution
- unit tests for relative link rewriting
- fixture tests for Markdown edge cases
- explicit checks for unsupported syntax that should fail loudly

The current design needs those tests because the logic is concentrated in the content layer.

### 5. The build still shows a performance smell on very large blog pages

`next build` still warns about large page data for the longest blog posts. The site works, but the current SSG payload shape is not ideal for very large documents.

That is not a release blocker for the migration slice, but it is real technical debt.

### 6. Some UI features are placeholders, not finished systems

Examples:

- the search box is present, but full search parity is not implemented yet
- TOC and heading-anchor behavior are only partial
- Open Graph support is only partial
- feedback and share surfaces are still missing

These are feature gaps, but they also affect engineering quality because they leave temporary structure in the codebase.

## Missing Architectural Decisions

These decisions should be written down before more implementation lands.

### 1. Content module decomposition strategy

The repo needs a defined split for the current content monolith.

Recommended decision:

- `content/fs-index.ts`: walk the repo once and build the file inventory
- `content/manifest.ts`: derive normalized route/content records from the file inventory
- `content/markdown.ts`: parse frontmatter and Markdown into typed render inputs
- `content/collections.ts`: expose blog/tutorial/section collections from the manifest
- `content/assets.ts`: resolve static asset paths and public URLs
- `content/loaders.ts`: assemble page props from manifest records

The dependency direction should stay one-way:

- file inventory -> manifest -> collections/loaders -> pages/components

Pages should not scan the filesystem directly.

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

### 4. Parity scope and rollout cut-lines

The design needs explicit milestones, otherwise "finish parity" stays too vague.

Recommended decision:

- `Shadow mode`: routes, SEO metadata, raw assets, docs/blog rendering, sitemap parity, analytics
- `Cutover ready`: search, TOC, heading anchors, code highlighting, admonitions, tabs, edit links, authors, last updated
- `Growth parity`: feedback CTA, share buttons, RSS/social image polish, homepage/product polish

The sitemap should only contain routes that satisfy the current cut-line for the environment being tested.

### 5. Rendering strategy by page class

The design currently says "static first", but large blog pages already show that one global policy is too coarse.

Recommended decision:

- homepage and landing pages: `SSG`
- tutorials and reference docs: `SSG`
- blog index pages: `SSG`
- very large articles: move toward App Router server rendering or a server-component path once route abstractions are stable

This should be decided per route class, not per framework preference.

### 6. Pages Router exit criteria

The current Pages Router implementation is a compatibility shell, not the target architecture.

Recommended decision:

- no new core abstractions should depend on `getStaticProps`, `getStaticPaths`, or duplicated `pages/zh/**` trees
- the content subsystem should stay framework-agnostic so it can be reused by App Router later
- move to App Router only after the content manifest, loader boundaries, and locale route model are stable

This prevents the migration from paying the routing rewrite cost twice.

## Non-Negotiables

The new frontend must preserve:

- every public URL that currently matters, including `/`, `/zh/`, docs routes, tutorials, blog, and legacy `/blogs/*` links
- current SEO infrastructure: `robots.txt`, `sitemap.xml`, canonical URLs, Open Graph tags, alternate language links, descriptive titles, and page descriptions
- current site behavior: search, blog index, dated posts, docs pages, multilingual routing, edit links, analytics, feedback entry points, and share buttons
- the existing Markdown content model instead of rewriting hundreds of documents as React pages

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

- `Next.js` App Router with static generation by default
- path-based i18n: `/` for English, `/zh/` for Chinese
- route handlers for `robots.txt`, `sitemap.xml`, RSS/Atom if enabled, OG image generation, and redirects

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

- use `Pagefind` or another static search index
- preserve on-page search behavior for docs and blog content

## Styling

- `Tailwind CSS` for layout and components
- custom design system for the homepage, project landing pages, and media/follow pages
- keep docs rendering readable and conservative even if the landing pages become more custom

## Git and Content Metadata

- preserve `last updated`
- preserve `authors`
- preserve `Edit this page`
- preserve publish dates for blog posts

## Proposed Repository Layout

One reasonable target layout:

```text
web/
  src/
    app/
      page.tsx
      zh/
      tutorials/
      blog/
      bpftime/
      eunomia-bpf/
      robots.ts
      sitemap.ts
    components/
    lib/
      content/
      seo/
      i18n/
      search/
      redirects/
  content/
    docs/
    tutorials/
    blog/
  public/
    assets/
  scripts/
    build-search-index.mjs
    verify-routes.mjs
```

## Migration Phases

## Phase 0: Lock the Parity Contract

- inventory all current routes from sitemap
- inventory all SEO outputs for representative pages
- inventory all current features and the exact user-visible behavior to keep
- make legacy `/blogs/*` handling explicit before anything else changes

## Phase 1: Build the Shell

- implement homepage, global header, footer, docs layout, and blog layout in Next.js
- wire path-based i18n and a shared metadata generator
- add route handlers for `robots.txt` and `sitemap.xml`

## Phase 2: Port Content Rendering

- ingest Markdown from the existing repo
- implement TOC, heading anchors, code rendering, admonitions, tabs, and local asset resolution
- reproduce current edit links, authors, and last-modified behavior

## Phase 3: Restore Growth Infrastructure

- restore analytics
- restore Open Graph and social image generation
- restore feedback CTA
- restore share buttons
- restore RSS/feed generation if needed

## Phase 4: Restore Search and Navigation

- build static search index
- create grouped navigation for tutorials and docs
- keep all current stable paths working

## Phase 5: Cut Over Safely

- run crawler parity against production
- add 301 redirects for every old path that changes
- compare indexed pages, titles, canonicals, hreflang tags, and sitemap coverage
- ship only after parity checks pass

## Definition of Done

The migration is only done when:

- the new site serves static HTML for all major routes
- SEO parity checks pass
- browser smoke tests pass
- internal link crawl passes
- old URLs continue to resolve or redirect cleanly
- editors can still update content by modifying Markdown files in the repo

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
- `Cutover ready` blockers: search, TOC, heading anchors, code highlighting, admonitions, tabs, edit links, authors, last updated
- `Growth parity` features: feedback CTA, share buttons, RSS/social image polish, homepage CTA polish

Deliverables:

- milestone checklist
- route readiness matrix
- clear cutover gate for sitemap inclusion

## Suggested Implementation Order

1. Split the content pipeline
2. Stabilize one locale through the new manifest and loader boundaries
3. Add sanitization and typed Markdown extension handling
4. Collapse locale routing onto shared page abstractions
5. Fill cutover-blocking parity features
6. Optimize large article payloads and then move toward App Router/server-component rendering where it actually helps

## Verification Snapshot

Last locally verified on `2026-03-06`:

```bash
cd app
NEXT_PUBLIC_SITE_URL=http://127.0.0.1:3000 npm run build
NEXT_PUBLIC_SITE_URL=http://127.0.0.1:3000 npm run start -- --hostname 127.0.0.1 --port 3000

cd ../test
BASE_URL=http://127.0.0.1:3000 npm run audit
```

Expected current result:

- `audit:http` passes on all sitemap routes
- `audit:parity` passes with zero missing legacy sitemap paths
- `audit:browser` passes
- `audit:links` passes
- `next build` still emits a non-blocking `large page data` warning for the largest blog pages
