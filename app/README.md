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
