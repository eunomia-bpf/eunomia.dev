# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the source code for the eunomia-bpf project website (https://eunomia.dev). The site provides comprehensive tutorials and documentation for eBPF programming, the eunomia-bpf framework, bpftime, and related projects.

This repository is also responsible for the complete operation of the
eunomia-bpf community, including documentation/site operations, community
coordination, media publishing, and external communications/promotion.

Public media work separates voice from ownership. Personal accounts carry the
maintainer's judgment, research choices, and engineering experience;
organization and project accounts carry formal releases and commitments.
Eunomia.dev is the institutional canonical archive and portfolio, with
eunomia-bpf, AgentSight, ActPlane, bpftime, papers, tutorials, and talks as the
core public evidence assets. Optimize platform posts for native account trust,
discussion, and community reach rather than website ranking alone.

## Required Workflow

For open-source code, documentation, synchronization, CI, release-readiness, or
PR-bound changes in this repository, use the `oss-change-workflow` skill before
editing. Follow its scope-control, validation, review, and CI guidance.

Before adding or keeping planning material in the repository, classify it by
lifespan. Short-term fixes, cleanup backlogs, one-off audits, and tactical
remediation plans should be tracked as GitHub issues, not long-lived draft
documents. Long-term decisions that are reusable, public-safe, and durable may
remain as documentation or skill guidance. When a document mixes both, split it:
move stable strategy or workflow guidance into the appropriate durable doc/skill,
open issues for concrete short-term fixes, then remove the temporary planning
file.

After every external platform publishing session, do a short publishing
lessons pass before reporting completion. Any concrete problem encountered
during drafting, preview, publishing, or public-page QA must be recorded in the
matching publisher skill or reference file, so the same platform mistake does
not recur in the next launch.

For long-form publishing on any external platform, prepare the platform-specific
upload/import artifact locally before opening the editor whenever practical.
Use a temporary file or `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md`
for the final title, body/H1 shape, image URLs or upload assets, table/code
fallbacks, links, tags/categories, and source/project note. Use platform
editors for import/upload, metadata/settings, preview, and QA, not for large
rewrites or fragile structural repairs. No platform requires a visible
canonical/source link in the article body; include one only when it helps the
reader.

If a same-day content-operations log is genuinely useful, write only
`draft/media/YYYY-MM-DD/run-log.md`. Do not create monthly daily-log files or
standalone figure inventories, platform-hook notes, per-article publish-QA
notes, or other disposable files whose only purpose is to prove that a workflow
step happened.

Work in this checkout directly on `main`. Do not create or switch to another
branch or worktree, and do not open a pull request. Before committing, inspect
the worktree, stage only explicit intended paths, run the smallest relevant
validation, and preserve unrelated user changes. Commit the validated change on
`main`, rebase onto `origin/main` when the remote has advanced, and push `main`
directly. This standing repository rule applies to daily `draft/` and
`.agents/skills/` maintenance as well as code, site, build, and documentation
work unless the user explicitly replaces it.

### Branch And Parallel Work

Treat this checkout as shared with the user and possibly other agents. Before
any branch, worktree, stash, rebase, reset, or commit operation, run
`git status --short --branch` and understand the current branch, upstream state,
dirty tracked files, and untracked files.

Keep this checkout on `main`; branch changes are prohibited by the standing
repository workflow. Preserve user work in place unless the user explicitly
asks you to move, stash, or discard it.

When another person or agent may be working in parallel, keep the scope narrow,
edit only requested files, and stage with explicit pathspecs. Never stage
unrelated dirty files or broad untracked directories.

Do not set a timeout when invoking a Subagent. Let it finish naturally; stop it
only when the user explicitly cancels it, its task has become obsolete, or it
has been confirmed stuck.

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

## Blog Writing Responsibilities

Keep blog style and blog production separate:

- `.agents/skills/blog-writing-style/SKILL.md` is the only source for how the
  finished English and Chinese prose should read. It may contain style guidance
  and examples, but no content requirements, model choices, editing procedure,
  or review workflow.
- `.agents/skills/blog-writer/SKILL.md` owns source preparation, writing and
  rewriting stages, model responsibilities, factual verification, and
  publication-integrity checks. It must not duplicate sentence-level or prose
  style rules.
- When both skills apply, `blog-writer` uses `blog-writing-style` as the target
  result. A style problem returns to the designated writer; Codex does not take
  over the body rewrite.

## Confidentiality Boundary

This is a PUBLIC repository. Business strategy, fundraising/incubator plans, pricing, customer lists or conversations, competitive analysis, personal constraints, and papers under review must NEVER be written into this repo (including `draft/`). That content belongs in the private strategy repo at `~/workspace/eunomia-strategy` (github.com/yunwei37/eunomia-strategy). Site operations, SEO/content plans, and brand style guidance are fine here.

## Important Notes
- Do not edit tutorial content directly in this repo - edit in the bpf-developer-tutorial repository
- Project home pages should be edited in their respective repository README files
- The site automatically syncs content from external repositories during build
- Always test changes locally with `cd app && npm run dev` (and run `npm run verify`) before committing
- Never change existing public paths, source paths, route slugs, or existing navigation target hrefs unless the user explicitly requests that exact path change. Adding a new page is acceptable only when it does not move, rename, or reparent existing content paths. In particular, keep project documentation URL ownership stable: `/bpftime/` and `/bpftime/**` must remain bpftime paths, not be moved under `/products/` or another section.
- **`mkdocs.yaml` is the permanent, single source of truth for site IA configuration** (this is a fixed architectural decision, not a transitional state). All site URL, route, navigation, nav-dropdown, sidebar, and page-link configuration belongs in `mkdocs.yaml`. React/TypeScript components may render configured links, but must NOT define route tables, navigation entries, or hard-coded internal hrefs for site pages. Use generated content/IA data derived from `mkdocs.yaml` (via `app/lib/content/mkdocs-config.ts`) instead.
