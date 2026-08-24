---
name: eunomia-research-report
description: Research, write, validate, and publish source-grounded Eunomia Daily Reports for technical readers. Use when an agent needs to use platform Deep Research or direct primary-source research, identify a non-trivial systems gap, propose high-quality academic and production ideas, write bilingual reports under the stable `/research/` URLs, or revise an existing Daily Report.
---

# Eunomia Daily Report

Produce technically serious, source-grounded analysis for the public **Daily Report** section. The repository is the source of truth for the publication contract, content, routes, topic mix, and delivery workflow.

The purpose is to help engineers and researchers understand a real problem, see what current work still misses, and find ideas worth implementing and evaluating. Depth comes from evidence, mechanism, comparison, and testable design, not from academic tone or terminology density.

## Public Contract

Every Daily Report must satisfy these rules:

- The title and opening are understandable to a technically qualified reader who has not read the source corpus.
- Facts, measurements, cited claims, synthesis, and proposed ideas remain distinguishable.
- The report contains a concrete section explaining where current research or production practice is still weak.
- The report contains a small number of developed directions with both academic and production relevance.
- The report states the assumptions behind its conclusion and the evidence that would change it, using reader-facing language.
- Every scheduled daily run publishes one new report. A weak candidate is rejected, but the run must continue researching another approved question until one passes the quality gates.
- The rolling topic mix in `.github/seo-data/content-series.md` is mandatory: normally 5–7 of the most recent 10 reports explicitly center eBPF, while pure Agent topics remain at most 1–2 of 10.
- Do not add provenance banners, warning boxes, generation-process badges, review-status text, or footer disclaimers to the public page.
- Do not infer an article byline from Git commit metadata.

## Required Context

Read these before acting:

- `CLAUDE.md`
- `.agents/skills/blog-writing-style/SKILL.md`
- `.github/seo-data/content-series.md`
- existing pages under `docs/research/`
- related posts under `docs/blog/` and `docs/reports/`
- `references/research-method.md`
- `docs/papers/registry.yaml` only after a candidate question exists

When invoked by a scheduled repository task, also read the task entrypoint, current operating records, rolling topic mix, and any explicit publication constraints stored in the repository.

## Output Location And Metadata

A bilingual report uses:

- `docs/research/<topic>.md`
- `docs/research/<topic>.zh.md`

Keep an existing URL when revising an article. Scheduled daily publication, however, requires one **new** report page; revising an existing page does not satisfy that day's publication requirement.

New Daily Report frontmatter should include:

```yaml
date: YYYY-MM-DD
title: "Reader-facing title"
description: "Concrete problem, main analysis, and practical consequence."
tags:
  - Daily Report
  - <precise topic tags>
research_question: "The exact question the report tries to answer"
source_cutoff: YYYY-MM-DD
status: daily-report
```

## Topic Selection

Daily Report is eBPF-first.

Before research begins:

1. calculate the rolling mix from the most recent published reports;
2. prefer the active series in `.github/seo-data/content-series.md`;
3. keep 5–7 of the most recent 10 reports centered on eBPF;
4. keep pure Agent reports to at most 1–2 of 10;
5. use the remaining slots for adjacent systems topics such as Linux, profiling, networking, security, runtimes, GPU/heterogeneous systems, distributed systems, storage, and compilers.

An eBPF-centered report must make eBPF essential to the mechanism, comparison, runtime boundary, or experiment. Mentioning eBPF in the introduction or future-work section does not make a report eBPF-centered.

Because the current small archive already contains two Agent-centered reports, the next reports should strongly favor eBPF before another pure Agent report is considered.

## Workflow

### 1. Start With A Real Reader Question

Begin from the active eBPF or approved adjacent-systems series, then search before choosing a thesis. Prefer a question that changes an architecture, implementation, security, operations, or research decision.

Good questions expose one of these:

- a mechanism missing from current systems;
- conflicting results that need reconciliation;
- a production failure absent from benchmarks;
- an abstraction that breaks under new workloads;
- a capability that exists but lacks a correct interface;
- a measurement problem that prevents comparing alternatives.

Do not choose a topic merely because it is trending or can mention an Eunomia project.

### 2. Use Platform Deep Research As Input

When platform Deep Research is available or requested, use it for broad source discovery, competing explanations, and citation collection.

Treat its report as research input, not publishable prose. Before using it:

- inspect the important primary sources;
- verify dates, metrics, experimental conditions, and quoted conclusions;
- remove unsupported extrapolation;
- identify which sources are independent;
- compare the candidate with existing Eunomia content;
- rebuild the argument around the reader's question rather than the tool's outline;
- record internally when the platform report was incomplete or unavailable.

### 3. Build An Evidence Map

For every serious source, capture privately:

- source type and primary URL;
- publication date and event date;
- concrete fact, mechanism, or measured result;
- workload, scale, method, and assumptions;
- implementation or artifact availability;
- possible conflict of interest;
- independent support or counterevidence;
- which reader decision it changes.

Count independent evidence, not repeated coverage. A focused report should use the smallest corpus that adequately supports the mechanism, alternatives, gap, and boundary. Do not pad a narrow report to imitate a survey.

### 4. Pass The Thesis And Gap Gate

Before drafting, state in ordinary language:

1. the concrete problem;
2. the current default mental model;
3. why that model fails;
4. the central claim;
5. the strongest alternative explanation;
6. the research or production gap exposed by the evidence;
7. what result would change the claim.

The gap must identify a missing benchmark, interface, mechanism, guarantee, dataset, measurement, deployment property, or boundary condition. Reject generic statements such as "more research is needed" or "scalability remains challenging."

If the candidate fails this gate, discard it and immediately research the next candidate in the approved series roadmap. The daily run does not terminate without a report.

### 5. Design The Reader Path

Use a concrete recurring scenario before introducing a taxonomy, formal model, or coined term.

A common progression is:

- concrete situation and consequence;
- why existing practice appears sufficient;
- the failure mechanism;
- what existing systems solve and leave open;
- the proposed architecture or decision;
- evidence and competing explanations;
- current gaps;
- promising directions;
- evaluation and evidence that would change the conclusion.

Do not front-load a source matrix, equation, related-work parade, or newly named property.

### 6. Write The Required Gap Section

Use a reader-facing heading such as:

- `## Where current work is still weak`
- `## 现有研究还缺什么`

Cover two to five concrete gaps. For each gap, explain:

- what leading papers or systems already achieve;
- the exact missing capability or evidence;
- why the omission matters to correctness, performance, security, operation, cost, or adoption;
- what observation or experiment would establish whether the gap is material.

Do not repeat individual papers' future-work paragraphs. Synthesize across evidence.

### 7. Write The Required Ideas Section

Use a heading such as:

- `## Promising directions with academic and production value`
- `## 兼具学术价值与生产价值的方向`

Prefer two or three developed ideas over a brainstorm list. Every idea must include:

1. **Gap:** the missing mechanism or evidence it addresses.
2. **Mechanism:** what the proposed system, abstraction, dataset, protocol, or algorithm actually does.
3. **Delta:** how it differs from the strongest adjacent work.
4. **Artifact:** what can be implemented, released, or reproduced.
5. **Evaluation:** workloads, baselines, metrics, and an ablation or counterexample that distinguishes the idea.
6. **Academic value:** the generalizable question, property, or method.
7. **Production value:** who can deploy it and which cost or failure it reduces.
8. **Failure condition:** a result showing the extra mechanism is not worth its complexity.

Reject ideas that are only "apply an LLM," "build a platform," "add eBPF," or "use a better model." The mechanism must remain meaningful when the label is removed.

### 8. Write Bilingual Pages Naturally

English and Chinese versions share evidence, claims, and idea quality, not sentence boundaries.

In Chinese:

- use Chinese for ordinary concepts;
- retain English only for proper nouns, identifiers, code, and useful terms of art;
- do not make the grammar depend on a stack of English nouns;
- explain a formal term in Chinese before relying on its English name.

### 9. Use Reader-Facing Conclusion Boundaries

Do not publish headings such as `Scope, limitations, and falsification` or direct Chinese translations of that template.

Use a natural heading such as:

- `## What would change this conclusion?`
- `## 哪些结果会改变这个判断？`

Explain assumptions, counterexamples, and decisive future evidence in prose. Keep the concepts, remove the template language.

### 10. Validate And Publish

Every scheduled daily run must finish with one new bilingual report deployed publicly.

1. create a focused branch;
2. add exactly one new Daily Report topic in English and Chinese;
3. change only files in scope plus directly coupled navigation/metadata/operating records;
4. run the repository's full verification workflow;
5. inspect the complete PR diff;
6. merge only after required checks pass;
7. verify that the exact merge commit deploys;
8. inspect the public English and Chinese pages, navigation, canonical URLs, gap sections, idea sections, and rendered conclusion heading;
9. record the report's series and topic classification for the next rolling-mix calculation.

A draft, open PR, green pre-merge check, revision-only day, or unpublished report is not completed daily delivery.

## Publication Review

Reject or revise the candidate when any of these are true:

- the title depends on an unexplained coined term;
- the opening reads like an abstract rather than a human explanation;
- the report summarizes papers one by one;
- the gap section contains generic complaints;
- the ideas section lacks an implementable artifact or discriminating evaluation;
- academic value is claimed only because the topic is new;
- production value is claimed without a deployable boundary or user;
- repository-owned work is treated more generously than outside work;
- the page contains process-oriented warning boxes, provenance notes, review-status text, or footer disclaimers;
- the new page substantially duplicates an existing question or thesis;
- important claims cannot be traced to primary evidence;
- the topic would violate the rolling eBPF/Agent editorial mix.

Rejecting one candidate does not end the daily run. Select another question from the active or approved series and repeat the research process until one report passes.

## Daily Fallback

There is no `no-report` outcome for the scheduled daily operation.

If the initial candidate lacks a defensible gap, mechanism, evidence base, or non-trivial idea:

1. preserve any useful evidence internally;
2. reject the candidate;
3. move to the next question in the active eBPF series;
4. if necessary, move to another approved eBPF or adjacent-systems series while respecting the rolling mix;
5. continue until one candidate meets the full publication standard.

The fallback is broader research, not weaker writing.
