# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source code for the eunomia-bpf project website (https://eunomia.dev). The site provides comprehensive tutorials and documentation for eBPF programming, the eunomia-bpf framework, bpftime, and related projects.

### Tech stack (current)

The site is rendered by a **custom Next.js + React + Tailwind CSS frontend** living in `app/`, statically exported to plain HTML/CSS/JS. It is **not** a runtime MkDocs site anymore.

- **Next.js** (pages router under `app/pages/`, `output: "export"` static export) — note the directory is named `app/` but uses the *pages* router, not the App Router
- **React 19** + **TypeScript** for components in `app/components/` and content/loader logic in `app/lib/`
- **Tailwind CSS** (`app/tailwind.config.ts`) for styling
- Content is still authored as **Markdown in `docs/**`**; a build-time pipeline (`app/scripts/generate-*`) parses it into JSON artifacts (content index, search index, manifest, static metadata) that the Next.js pages consume.
- **`mkdocs.yaml` is retained as the site IA / navigation configuration source** (parsed by `app/lib/content/mkdocs-config.ts`), not as a renderer. The MkDocs build itself is legacy — see `.github/workflows/mkdocs.yml` (`workflow_dispatch` only).

See `app/README.md` and `app/ARCHITECTURE.md` for the frontend in depth (note: `ARCHITECTURE.md` still names Cloudflare Pages as the target; the live deploy is GitHub Pages — see Deployment below).

## Common Development Commands

### Frontend Development (Next.js app)
```bash
cd app

# Install dependencies (Node.js 22+)
npm ci

# Run the local dev server (http://localhost:3000)
# Regenerates content artifacts, then runs `next dev`
npm run dev

# Production-compatible static export -> app/out
NEXT_PUBLIC_SITE_URL=https://eunomia.dev npm run build

# Quality gates
npm run lint        # eslint over components/lib/pages/scripts/tests
npm run typecheck   # tsc --noEmit
npm run verify      # lint + verify.mjs (used in CI)
```

> Legacy MkDocs commands (`mkdocs serve` / `make build`) only drive the deprecated
> `.github/workflows/mkdocs.yml` path and are no longer how the live site is built.

### Content Synchronization
```bash
# Clone/update the tutorial repository
make tutorial

# Clone/update other documentation repositories
make bpftime
make cuda-exp
make cupti-exp
```

**Important**: Tutorials are maintained in a separate repository (https://github.com/eunomia-bpf/bpf-developer-tutorial). Edit tutorials there, not in this repository.

## Architecture and Structure

### Documentation Organization
- `/docs/` - All website content
  - `tutorials/` - eBPF programming tutorials (synced from external repo)
  - `blog/` & `blogs/` - Technical blog posts
  - `eunomia-bpf/` - Framework documentation
  - `bpftime/` - Userspace eBPF runtime docs
  - `GPTtrace/` - GPTtrace tool documentation
  - `wasm-bpf/` - WebAssembly/eBPF integration docs
  - `others/` - Additional content (CUDA tutorials, ideas)

### Key Configuration
- `mkdocs.yaml` - Main site configuration
  - Configures Material theme, navigation, plugins
  - Sets up i18n for English/Chinese support
  - Defines site structure and features

### Deployment
- `.github/workflows/app-static-pages.yml` builds the Next.js app (`cd app && npm run build`),
  verifies the static export (no `localhost`/`127.0.0.1` leakage, correct canonical URLs in
  `sitemap.xml`/`robots.txt`/`feed.xml`), then deploys `app/out` to **GitHub Pages** via
  `actions/deploy-pages@v5`.
- The MkDocs workflow (`mkdocs.yml`) is legacy and only runs on manual `workflow_dispatch`.

### Content Management
- The site is multilingual (English and Chinese); locale routing is handled by the
  content manifest / loaders (`/zh/**` mirrors English route identities)
- Blog posts support tags and RSS feeds (`feed.xml`, `zh/feed.xml` emitted at build time)
- Git revision date and authors are collected at build time and surfaced per page

### Working with eBPF Tutorial Code
When working with eBPF examples in the tutorials:
- Each tutorial has its own Makefile using the libbpf build system
- BPF programs use `.bpf.c` extension
- User-space programs are regular `.c` files
- Build with `make` in the tutorial directory
- Clean with `make clean`

## Important Notes
- Do not edit tutorial content directly in this repo - edit in the bpf-developer-tutorial repository
- Project home pages should be edited in their respective repository README files
- The site automatically syncs content from external repositories during build
- Always test changes locally with `cd app && npm run dev` (and run `npm run verify`) before committing
- Never change existing public paths, source paths, route slugs, or existing navigation target hrefs unless the user explicitly requests that exact path change. Adding a new page is acceptable only when it does not move, rename, or reparent existing content paths. In particular, keep project documentation URL ownership stable: `/bpftime/` and `/bpftime/**` must remain bpftime paths, not be moved under `/products/` or another section.
- **`mkdocs.yaml` is the permanent, single source of truth for site IA configuration** (this is a fixed architectural decision, not a transitional state). All site URL, route, navigation, nav-dropdown, sidebar, and page-link configuration belongs in `mkdocs.yaml`. React/TypeScript components may render configured links, but must NOT define route tables, navigation entries, or hard-coded internal hrefs for site pages. Use generated content/IA data derived from `mkdocs.yaml` (via `app/lib/content/mkdocs-config.ts`) instead.
