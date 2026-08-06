---
name: eunomia-research-report
description: Research, write, validate, and publish source-grounded Eunomia Daily Reports for technical readers. Use when an agent needs to use platform Deep Research or direct primary-source research, identify a non-trivial systems gap, propose high-quality academic and production ideas, write bilingual AI-generated reports under the stable `/research/` URLs, or revise an existing Daily Report. Public reports are AI-generated and must never be represented as human-reviewed unless a named human review actually occurred.
---

# Eunomia Daily Report

Produce technically serious AI-generated analysis for the public **Daily Report**
section. The filesystem and public URLs remain `docs/research/` and
`/research/` for compatibility, but the reader-facing name is Daily Report.

The purpose is not to imitate a paper. The purpose is to help engineers and
researchers understand a real problem, see what current work still misses, and
find ideas worth implementing and evaluating.

## Non-Negotiable Public Contract

Every public Daily Report must satisfy all of these rules:

- It is explicitly labeled **AI-generated and not human-reviewed** near the top
  and again at the bottom of the rendered page.
- It never claims to be reviewed, peer-reviewed, maintainer-endorsed, or written
  by a human unless that review or authorship actually happened and is recorded.
- Human Git commit authors are not presented as article authors merely because
  they merged generated text.
- The title and opening are understandable to a technically qualified reader who
  has not read the source corpus.
- Facts, measurements, cited claims, generated synthesis, and speculative ideas
  remain distinguishable.
- The report contains a section explaining where current research or production
  practice is still weak.
- The report contains a section proposing a small number of non-trivial research
  or systems directions with both academic and production relevance.
- The report is not published when the scan yields only a fluent summary and no
  defensible gap, mechanism, or useful idea.

A disclosure does not excuse weak sourcing. It tells readers the correct
editorial status while the report still aims for strong evidence and reasoning.

## Required Context

Read these before acting:

- `CLAUDE.md`
- `.agents/skills/blog-writing-style/SKILL.md`
- existing pages under `docs/research/`
- previous related posts under `docs/blog/` and `docs/reports/`
- `references/research-method.md`
- `docs/papers/registry.yaml` only after a candidate question exists
- the current rendered Daily Report disclosure implementation

When invoked by `eunomia-content-patrol`, also read the rolling publication
queue and the current dated media workspace.

## Output Location And Metadata

A public bilingual report uses:

- `docs/research/<topic>.md`
- `docs/research/<topic>.zh.md`

Keep the existing URL when revising an article. Do not rename `/research/` URLs
only because the public section is called Daily Report.

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
status: ai-generated-unreviewed
```

Do not use `reviewed-research-brief` or a human byline by default.

## Workflow

### 1. Start With A Real Reader Question

Begin from a broad systems area, then search before choosing a thesis. Prefer a
question that changes an architecture, implementation, security, operations, or
research decision.

Good questions expose one of these:

- a mechanism missing from current systems;
- conflicting results that need reconciliation;
- a production failure not represented in benchmarks;
- an abstraction that breaks under new workloads;
- a capability that became possible but lacks a correct interface;
- a measurement problem that prevents comparing alternatives.

Do not choose a topic because it is trending or because it can mention an
Eunomia project.

### 2. Use Platform Deep Research As Input

When platform Deep Research is available or requested, use it for broad source
discovery, competing explanations, and citation collection.

Its report is research input, not publishable prose. Before using it:

- open and inspect the important primary sources;
- verify dates, metrics, experimental scope, and quoted conclusions;
- remove unsupported extrapolation;
- identify which sources are independent;
- compare the candidate with existing Eunomia content;
- rebuild the article around the reader's question rather than the tool's
  outline;
- record honestly when the platform report was incomplete or unavailable.

Never say a page was produced by platform Deep Research unless a complete report
was actually returned and materially used.

### 3. Build An Evidence Map

For each serious source, capture privately:

- source type and primary URL;
- publication date and event date;
- concrete fact, mechanism, or measured result;
- implementation or artifact availability;
- limitation and possible conflict of interest;
- independent support or counterevidence;
- which reader decision it changes.

Count independent evidence, not repeated coverage. A focused report should use
the smallest corpus that adequately supports the mechanism, alternatives, gap,
and boundary. The broad scheduled landscape scan may use larger numeric gates,
but a narrow Daily Report must not be padded to imitate a survey.

### 4. Pass The Thesis And Gap Gate

Before drafting, state in ordinary language:

1. the concrete problem;
2. the current default mental model;
3. why that model fails;
4. the central claim;
5. the strongest alternative explanation;
6. the research or production gap exposed by the evidence;
7. what result would weaken or falsify the claim.

The gap must be specific. Avoid statements such as "more research is needed" or
"scalability remains challenging."

A useful gap identifies a missing benchmark, interface, mechanism, guarantee,
dataset, measurement, deployment property, or boundary condition.

### 5. Design The Reader Path

Use a concrete recurring scenario before introducing a taxonomy, formal model,
or coined term.

A common progression is:

- concrete situation and consequence;
- why existing practice appears sufficient;
- the failure mechanism;
- what existing systems solve and leave open;
- the proposed architecture or decision;
- evidence and competing explanations;
- research gaps;
- promising directions;
- evaluation, limits, and falsification.

Do not front-load a source matrix, equation, related-work parade, or newly named
property.

### 6. Write The Required Gap Section

Use a reader-facing heading such as:

- `## Where current work is still weak`
- `## 现有研究还缺什么`

Cover two to five concrete gaps. For each gap, explain:

- what current papers or systems already do;
- the exact missing capability or evidence;
- why the omission matters to correctness, performance, security, operation, or
  adoption;
- what observation or experiment would show that the gap is real.

Do not confuse an author's limitation paragraph with a field-level gap. The
report must synthesize across evidence.

### 7. Write The Required Ideas Section

Use a heading such as:

- `## Promising directions with academic and production value`
- `## 兼具学术价值与生产价值的方向`

Prefer two or three developed ideas over a long list. Every idea must include:

1. **Gap:** the concrete missing mechanism or evidence it addresses.
2. **Mechanism:** what the proposed system, abstraction, dataset, or algorithm
   would actually do.
3. **Delta:** how it differs from the strongest related work.
4. **Artifact:** what can be implemented, released, or reproduced.
5. **Evaluation:** workloads, baselines, metrics, and an ablation or
   counterexample that distinguishes the idea.
6. **Academic value:** the generalizable question, property, or method.
7. **Production value:** who could deploy it and what cost or failure it reduces.
8. **Failure condition:** a result that would show the extra mechanism is not
   worth its complexity.

Reject ideas that are only "apply an LLM," "build a platform," "add eBPF," or
"use a better model." The mechanism must remain meaningful when the marketing
label is removed.

### 8. Write Bilingual Pages Naturally

English and Chinese versions share evidence, claims, and idea quality, not
sentence boundaries.

In Chinese:

- use Chinese for ordinary concepts;
- retain English only for proper nouns, identifiers, code, and useful terms of
  art;
- do not make the grammar depend on a stack of English nouns;
- explain a formal term in Chinese before relying on its English name.

### 9. Preserve Editorial Status In Rendering

Before publication, verify that every `/research/` and `/zh/research/` page:

- displays the AI-generated, not-human-reviewed notice near the top;
- repeats the notice at the bottom;
- uses the public label Daily Report or 每日报告 in navigation and section UI;
- does not expose a human author through article metadata by default;
- keeps canonical and hreflang URLs unchanged;
- remains linked from the Blog menu and Blog landing page.

### 10. Validate And Publish

When the user or an authorized queue item requests publication, complete the
whole delivery:

1. create a focused branch;
2. change only the Daily Report content, rendering, or skill files in scope;
3. run the repository's full verification workflow;
4. inspect the complete PR diff;
5. merge only after required checks pass;
6. verify that the exact merge commit deploys;
7. check the public English and Chinese pages, disclosure text, navigation,
   canonical URLs, and the new gap and idea sections.

A draft, open PR, or green pre-merge check is not a completed deployment.

## Publication Review

Reject or revise the report when any of these are true:

- the title depends on an unexplained coined term;
- the opening reads like an abstract rather than a human explanation;
- the report summarizes papers one by one;
- the gap section contains generic limitations;
- the ideas section lacks an implementable artifact or discriminating
  evaluation;
- academic value is claimed only because the topic is new;
- production value is claimed without a deployable boundary or user;
- repository-owned work is treated more generously than outside work;
- the disclosure is missing or suggests human review that did not occur;
- the new page substantially duplicates an existing question or thesis;
- important claims cannot be traced to primary evidence.

## No-Report Outcome

If research produces interesting sources but no defensible gap and no
non-trivial idea, do not publish. Record the strongest unresolved question and
what evidence would unblock it. A truthful no-report result is better than an
AI-generated summary added only to satisfy cadence.
