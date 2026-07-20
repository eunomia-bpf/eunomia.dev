---
name: eunomia-research-report
description: Research and draft source-grounded Eunomia public reports or daily research blogs. Use when Codex needs to analyze the latest broad AI, Agent, and infrastructure news; scan papers, engineering practice, commercial products, open source, public institutions, and social discussion; select a topic and thesis only after research; create a distinct deep report; or continue a report invoked by eunomia-content-patrol. This skill prepares and validates local drafts but does not authorize final publishing.
---

# Eunomia Research Report

Research current technical change before choosing a thesis, then write a concise
public report whose depth comes from evidence and analysis rather than length.

## Required Context

Read these before acting:

- `CLAUDE.md`
- current month plan: `draft/plan/YYYY-MM.zh.md`
- current monthly log, if present: `draft/content-daily-log-YYYY-MM.md`
- previous reports under `draft/media/` and `docs/reports/` when checking topic
  duplication
- `references/research-method.md`

This skill may be invoked directly or by `eunomia-content-patrol`. The patrol
decides when research work belongs in the daily operating loop; this skill owns
the research, topic selection, report drafting, and report QA.

## Workflow

### 1. Start With A Broad Direction

Begin with a wide editorial direction such as current AI, Agent, or systems
infrastructure. Do not decide the thesis, preferred conclusion, or relationship
to Eunomia projects before searching.

### 2. Run The Current-News Scan

Search the latest 48 hours first. Cover every source family in
`references/research-method.md`, including academic research, engineering
practice, commercial products, open source, public institutions, and social or
community discussion. Also consider benchmarks, datasets, incidents, security
advisories, protocols, and standards.

Search broad AI, Agent, and infrastructure topics before narrowing. Include
model serving and inference, training systems, GPU/runtime performance,
observability, security, evaluation, developer tooling, eBPF, and open-source
infrastructure when they carry current signal.

If the 48-hour window is sparse, as on weekends or holidays, keep that result
visible and widen to the latest working day and then 7-30 days. Distinguish the
date of the underlying event from the article, repost, or indexing date.

### 3. Search Efficiently And Verify Selectively

Use web search, feeds, paper indexes, official release pages, repository search,
and public project pages for discovery. Do not open an interactive browser for
every result. Search snippets, aggregators, newsletters, and reposts identify
leads but do not support factual claims by themselves.

Open and read primary sources for serious candidates. Before citing, quoting,
or acting on a social post, inspect the public post and its context through the
normal visible browser UI. Never use hidden platform APIs, background endpoints,
or scraping datasets.

### 4. Build The Signal Map

Capture serious candidates compactly with source type, primary URL, publication
date, event date, concrete claim, evidence strength, limitation, and any
contradictory signal. Keep this in working context or the daily log; do not
create a visible source-inventory artifact.

Cluster candidates by mechanism, change, or tension rather than shared keywords.
Generate candidate research questions only after clustering. Compare candidates
for timeliness, evidence independence, technical consequence, disagreement, and
reader usefulness.

### 5. Form And Test The Thesis

Choose a clear, contestable thesis only after the scan. It may support, qualify,
contradict, or be unrelated to existing Eunomia projects and editorial views.

Before drafting, look for the strongest alternative explanation and evidence
that could overturn the thesis. Narrow or discard a thesis that depends on one
marketing claim, one repeated press release, or social posts quoting each other.

### 6. Draft The Public Report

Create the working report at:

`draft/media/YYYY-MM-DD/<topic-slug>/deep-report.zh.md`

Give it a stable `report_id`, research question, research window, source cutoff
date, status, and thesis. Make the latest 48-hour developments identifiable in
the report while using older sources only where they add mechanism, prior art,
baseline, or contradiction.

Use the editorial functions in `references/research-method.md` without turning
them into a rigid heading checklist. Keep the evidence chain in the article and
the raw browsing process out of it.

### 7. Review For Publication Value

Check that:

- the thesis emerged from the evidence and is not product advocacy by default
- every central factual claim resolves to a primary or clearly labeled source
- current events are separated from older context and republished material
- independent sources are truly independent
- contradictions, uncertainty, and source limitations remain visible
- analysis explains mechanism and second-order effects instead of summarizing
- developer or operator implications follow from the evidence
- the report is materially different from previous report questions and theses
- no private strategy, customer information, pricing plans, or unreleased work
  entered the public repository
- no em dash appears in public prose

Prefer the shortest version that preserves the thesis, evidence chain,
mechanism, counterargument, boundary, and practical significance. Do not pad a
report to look deep.

### 8. Record And Hand Back

Update `draft/content-daily-log-YYYY-MM.md` with the research window, source
families checked, selected topic or "no defensible thesis", report path, and
next concrete action. When invoked by `eunomia-content-patrol`, return control to
that skill for platform adaptation, publishing authorization, and ledger work.

This skill does not authorize a final publish, repost, comment, or other social
action.
