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

Daily changes limited to `draft/` and `.agents/skills/` are a direct
commit-and-push workflow. Do not open a pull request solely for those daily
draft/skill maintenance updates; still inspect the worktree, stage explicit
paths only, run a lightweight validation when relevant, and preserve unrelated
dirty files.

Treat this repository as a mature open-source project for every change. Never
push changes directly to `main` unless the user has explicitly allowed the
direct-push path for the current change type. Start from the current `main`
branch, create a feature or fix branch, and publish the change through a
normal, non-draft pull request for code, site, build, or other PR-bound changes.
Complete the independent review, Copilot-comment, validation, CI, and applicable
live-acceptance gates from `oss-change-workflow`. Do not merge the PR unless the
user explicitly asks.

Before publishing a PR, inspect the worktree, stage only the intended files, run
the smallest relevant validation, and preserve unrelated user changes.

### Branch And Parallel Work

Treat this checkout as shared with the user and possibly other agents. Before
any branch, worktree, stash, rebase, reset, or commit operation, run
`git status --short --branch` and understand the current branch, upstream state,
dirty tracked files, and untracked files.

Do not switch branches just to start from a clean base when the worktree
contains uncommitted changes, untracked files, or an active task branch. Continue
on the current branch when it matches the request. If a branch change is
necessary, state the source branch, target branch, and dirty-file handling first,
and preserve user work in place unless the user explicitly asks you to move,
stash, or discard it.

When another person or agent may be working in parallel, keep the scope narrow,
edit only requested files, and stage with explicit pathspecs. Never stage
unrelated dirty files or broad untracked directories. Prefer a separate worktree
only when the user asks for it or when the requested change truly cannot share
the current branch.

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

## Blog Writing Style Guide

When writing or editing blog posts in `docs/blog/posts/`, follow these rules strictly.

### Formatting rules (both languages)

- **No em dashes.** Never use "——"（中文）or " — "（English）in blog prose. Replace with commas, colons, semicolons, or conjunctions. The only exception is inside code blocks or CLI output examples.
- **No short choppy sentences.** Don't stack two or three short sentences that read like bullet points or notes. Merge them with connective tissue into flowing prose.
- **No "outline topic sentence" patterns.** Avoid structures like "X 有一个共同特征：Y", "X is another strong scenario", "根本问题是 X", "The fundamental problem is X". These are lazy structural crutches from note-taking. Rewrite as natural argumentative prose.
- **No numbered scenario lists** ("First scenario: ...", "Second scenario: ..."). Use natural transitions between examples ("Now consider...", "再看一个...", "Then there's...").
- **No trailing summaries** that just restate what the section already said.

### Prose quality

- Write blog prose, not notes. Every paragraph should read as a complete thought with a beginning, middle, and end.
- Maintain argumentative flow: each section should build on the previous one with a clear "so what" connection. Use progressive argument structure ("那就往下沉一层", "Push down one layer, then...") rather than enumerating features.
- Vary sentence structure. If three consecutive sentences start the same way or follow the same pattern, rewrite.
- Merge redundant restatements. If the same point appears twice in adjacent sentences, combine or cut.
- Keep technical depth but make it readable. Blog tone, not paper-abstract tone.

### Editing approach

- When fixing prose style, edit **one sentence at a time** using the Edit tool. Never overwrite entire sections or paragraphs at once.
- Do not change technical content, code blocks, YAML examples, or architecture diagrams when doing prose edits.
- After edits, verify no em dashes were introduced: `grep -c '——' file.zh.md` should return 0.

### Bilingual consistency

- English and Chinese versions of the same post should have matching structure: same sections, same argument flow, same examples in the same order.
- Section headings should correspond (e.g., "三层约束，三种盲区" ↔ "Three Layers, Three Blind Spots").
- When one version is rewritten, update the other to match structure (content can differ in natural expression).

## Confidentiality Boundary

This is a PUBLIC repository. Business strategy, fundraising/incubator plans, pricing, customer lists or conversations, competitive analysis, personal constraints, and papers under review must NEVER be written into this repo (including `draft/`). That content belongs in the private strategy repo at `~/workspace/eunomia-strategy` (github.com/yunwei37/eunomia-strategy). Site operations, SEO/content plans, and brand style guidance are fine here.

## Important Notes
- Do not edit tutorial content directly in this repo - edit in the bpf-developer-tutorial repository
- Project home pages should be edited in their respective repository README files
- The site automatically syncs content from external repositories during build
- Always test changes locally with `cd app && npm run dev` (and run `npm run verify`) before committing
- Never change existing public paths, source paths, route slugs, or existing navigation target hrefs unless the user explicitly requests that exact path change. Adding a new page is acceptable only when it does not move, rename, or reparent existing content paths. In particular, keep project documentation URL ownership stable: `/bpftime/` and `/bpftime/**` must remain bpftime paths, not be moved under `/products/` or another section.
- **`mkdocs.yaml` is the permanent, single source of truth for site IA configuration** (this is a fixed architectural decision, not a transitional state). All site URL, route, navigation, nav-dropdown, sidebar, and page-link configuration belongs in `mkdocs.yaml`. React/TypeScript components may render configured links, but must NOT define route tables, navigation entries, or hard-coded internal hrefs for site pages. Use generated content/IA data derived from `mkdocs.yaml` (via `app/lib/content/mkdocs-config.ts`) instead.
