# CLAUDE.md

Guidance for agents (Claude Code, Codex, and others) in this repository.
`AGENTS.md` is a symlink to this file.

## Project Overview

Source for the eunomia-bpf project website (https://eunomia.dev): tutorials
and docs for eBPF, eunomia-bpf, bpftime, and related projects, plus the
eunomia-bpf community's site operations, media publishing, and external
communications. Public media separates voice from ownership: personal
accounts carry the maintainer's judgment; org/project accounts carry formal
releases. Eunomia.dev is the institutional canonical archive (eunomia-bpf,
AgentSight, ActPlane, bpftime, papers, tutorials, talks), community-first in
identity, eBPF as the technical anchor, AI agents as a primary workload/user.
Eunomia Labs, Inc. may be named transparently as steward/builder, but don't
turn the site into a sales-first company page; keep pricing/CTAs on product
surfaces unless explicitly asked otherwise.

## Precedence Rule

Inside this repo's already-authorized goals (site/content changes, scheduled
patrols, publishing runs, skill-routed tasks), continue, repair, and deliver
automatically — don't stop to ask for approval. Ask or wait only where a
genuine external constraint requires it: the team's branch/PR/review process
(below), publishing beyond what's already authorized (see Publishing),
spending money, or handling credentials. Nothing else is an approval gate.
Report a genuine external blocker honestly and keep working on unaffected
tasks rather than stalling the whole run.

**Rule hygiene:** keep rules few and non-contradictory. Before adding a rule
here or in a skill, delete or merge an existing one instead of stacking an
exception on top. Each rule has one home — skills/automations link to this
file instead of restating it. Incident stories and dated logs belong in git
history or an issue; a one-line pointer here is enough.

## Workflow: Branch, Commit, Push

Work directly on `main`: no new branches/worktrees, no PRs, unless the user
explicitly asks for one. This checkout may be shared with the user or other
agents, so run `git status --short --branch` before any branch/stash/rebase/
reset/commit, stage only the explicit paths you intended to change, and
rebase onto `origin/main` when it has advanced before pushing `main` directly.
For OSS code/docs/sync/CI/release changes, follow `oss-change-workflow` for
scope, validation, and review. Every configured agent/model may do real work
within its task's already-authorized scope; coordinator roles only prevent
duplicate work, never reduce another model to read-only. Don't set a timeout
on a subagent — let it finish; stop it only if cancelled, obsolete, or stuck.

## Skills & Planning Material

See `.agents/README.md` for how the skills bridge (submodule + sync script)
works and how to change a shared vs. repo-specific skill. Track short-term
fixes as GitHub issues, not draft docs; only reusable, durable, public-safe
decisions belong in documentation/skills.

## Publishing

- Standing authorization: a request to publish/post/submit, or a queue item
  marked `排队`, authorizes completing that publication end to end (prep,
  preview, publish, QA, ledger update) without re-asking at the last step —
  except private messages, follows, likes, votes, account settings, payments,
  and deletions. Stop only for a draft/preview-only task or a real external
  blocker (report it, keep working on other eligible tasks).
- Medium/DEV.to are API-first: use local `MEDIUM_API_KEY`/`DEV_TO_API_KEY`,
  never print or commit them; follow with visible-browser QA. Other platforms
  are visible-browser-only.
- Prepare the platform artifact locally first (typically
  `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md`); use the platform
  editor for import/settings/preview/QA, not large rewrites. Record any new
  concrete platform problem/workaround in the matching publisher skill instead
  of a disposable per-run file.

## Blog Writing & Confidentiality

`blog-writing-style` owns finished-prose style (English/Chinese);
`blog-writer` owns source prep, writing/editing, verification, and
publication-integrity checks — neither duplicates the other.

PUBLIC repository: never write business strategy, fundraising plans, pricing,
customer info, competitive analysis, personal constraints, or papers under
review here (including `draft/`) — that belongs in the private
`~/workspace/eunomia-strategy` repo. Site ops, SEO/content plans, and brand
style guidance are fine here.

## Tech Stack, Commands, Architecture

Custom **Next.js (pages router) + React 19 + Tailwind** frontend in `app/`,
statically exported (`output: "export"`) — not a runtime MkDocs site. Content
is Markdown in `docs/**` (`tutorials/` synced from bpf-developer-tutorial —
edit there, not here); `app/scripts/generate-*` builds it into JSON artifacts
consumed by the Next.js pages. **`mkdocs.yaml` is the permanent, single source
of truth for site IA/navigation** (parsed by `app/lib/content/mkdocs-config.ts`);
components render generated data from it, never their own route tables or
hard-coded hrefs. Details: `app/README.md`, `app/ARCHITECTURE.md`. Deploy via
`.github/workflows/app-static-pages.yml` (`app/out` -> GitHub Pages);
`mkdocs.yml` is legacy/manual-only.

```bash
cd app && npm ci                                          # Node.js 22+
npm run dev                                 # dev server :3000
NEXT_PUBLIC_SITE_URL=https://eunomia.dev npm run build     # static export -> app/out
npm run lint && npm run typecheck && npm run verify        # quality gates (verify = CI)
```

```bash
make tutorial  # clone/update bpf-developer-tutorial; also: make bpftime / cuda-exp / cupti-exp
```

Multilingual: `/zh/**` mirrors English routes; RSS at `feed.xml`/`zh/feed.xml`.
Project home pages live in their own repo's README, not here. eBPF tutorial
code: each tutorial has its own libbpf Makefile (`.bpf.c` for BPF programs,
`.c` for user space; `make` / `make clean`).

## Invariants

- Test with `cd app && npm run dev` and `npm run verify` before committing.
- Never change existing public paths, route slugs, or nav hrefs without an
  explicit request for that exact change; new pages are fine if they don't
  move/reparent existing content. `/bpftime/**` stays under bpftime.
