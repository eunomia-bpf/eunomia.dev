# Daily Report Research Method

Use this reference for source coverage, evidence grading, gap analysis, idea design, and public report structure. The repository is the authoritative specification for the Daily Report workflow.

## Research Windows

Use the latest seven days for current releases, papers, incidents, and engineering discussion when the assignment depends on recent change. Read the latest 48 hours first, then widen to 7-30 days for mechanisms, contradictions, and deployment evidence.

Use older primary sources for theory, standards, prior art, and baselines. Confirm both the publication date and the date of the underlying event.

A durable Daily Report does not need a news hook. It still needs a source cutoff so later updates can tell which evidence the analysis included.

## Source Families

Prefer sources that expose inspectable evidence:

- peer-reviewed papers and method-complete preprints;
- datasets, benchmarks, and reproducible artifacts;
- standards and official technical specifications;
- source code, releases, commits, issues, and maintainer discussions;
- first-person engineering reports and postmortems;
- production measurements with enough methodology to interpret them;
- official documentation for current runtime behavior.

Marketing pages and social posts can establish what someone announced or observed. They do not establish a general technical fact without corroboration.

Count independent evidence, not links. Ten articles repeating one announcement are one evidence cluster.

## Evidence Map

For every serious source, record privately:

- primary URL and source family;
- publication date and underlying event date;
- concrete claim or measured result;
- method, workload, scale, and assumptions;
- available code, data, logs, or implementation detail;
- possible conflict of interest;
- independent support or counterevidence;
- which architecture or engineering decision it changes.

A source counts only when it changes the analysis through a fact, mechanism, comparison, contradiction, deployment failure, or useful question.

## Evidence Lattice

A strong report normally combines several roles:

- **Mechanism evidence:** why a failure or trade-off occurs.
- **Implementation evidence:** how a system or interface actually works.
- **Deployment evidence:** where the mechanism matters in practice.
- **Counterevidence:** a workload or result that narrows the central claim.
- **Measurement evidence:** a method that can distinguish alternatives.

A focused report should use the smallest corpus that supports its question, strongest alternative, gap, and boundary. Do not pad it to resemble a survey.

## Topic Selection

Prefer a question that changes a decision and exposes a concrete gap. Useful signals include:

- papers and production experience disagree;
- a new workload breaks an old abstraction;
- current benchmarks omit an important failure mode;
- a capability exists but lacks a safe or portable interface;
- systems optimize a proxy that does not measure the real outcome;
- a mechanism works only because of an unstated workload assumption;
- an operational problem cannot be compared because no shared metric or dataset exists.

Compare the candidate with existing Daily Reports and Blog posts. A new title, product, example, or coined term does not make the thesis new.

## Gap Analysis

The gap section must synthesize the field rather than repeat future-work paragraphs from individual papers.

A strong gap has four parts:

1. **Current capability:** what leading systems already achieve.
2. **Missing element:** the exact interface, guarantee, dataset, measurement, mechanism, or deployment property that is absent.
3. **Consequence:** why the absence matters to correctness, performance, security, cost, operation, or adoption.
4. **Test:** what experiment or observation would establish whether the gap is material.

Examples of specific gaps:

- no benchmark contains hidden cross-resource conflicts;
- no schema connects model decisions to system effects and authority;
- evaluation assumes a complete trace although production retention is lossy;
- a runtime exposes parallel execution but no commit semantics;
- published metrics count tool success rather than final task correctness;
- privacy cost is omitted from adaptive observability evaluation.

Reject generic statements such as:

- more research is needed;
- scalability remains a challenge;
- security could be improved;
- future work should evaluate more models;
- a model could automate the process.

## Idea Design

The ideas section is the main value of the Daily Report. Prefer two or three ideas that can support a real artifact and discriminating evaluation.

For each idea, write an internal design card before public prose.

### Gap

What exact missing capability or evidence does the idea address?

### Mechanism

What components, state, control path, algorithm, abstraction, or protocol would exist? Describe the system without relying on marketing labels.

### Delta From Related Work

Name the strongest adjacent approach and state the technical difference. A new application domain alone is usually not enough novelty.

### Artifact

What can be implemented or released?

Examples include:

- a runtime or proxy;
- a kernel or userspace mechanism;
- a portable schema or protocol;
- a benchmark and ground-truth dataset;
- a compiler or verifier extension;
- a trace corpus and replay harness;
- a scheduling or retention controller;
- a reproducible production study.

### Evaluation

Specify:

- workload families;
- strongest baselines;
- correctness and performance metrics;
- a fixed resource, privacy, or review budget when relevant;
- an ablation that isolates the new mechanism;
- a counterexample or workload where the simpler design should win.

### Academic Value

State the generalizable research question, property, model, or method. Do not claim academic value only because the topic is recent.

### Production Value

Identify the operator or developer who could deploy the artifact, the boundary where it integrates, and the failure, cost, or manual work it reduces.

### Failure Condition

State what result would show that the extra mechanism is unnecessary, too expensive, too inaccurate, or not general enough.

## Idea Quality Tests

An idea is not ready when it is only:

- apply a model to classify events;
- use eBPF to observe more data;
- build a unified platform;
- add a control plane;
- use multi-agent collaboration;
- make the model more accurate;
- combine several existing systems without a new property or trade-off.

Ask instead:

- What information becomes available that was previously unavailable?
- What decision becomes enforceable or measurable?
- What cost or correctness property changes?
- Why can existing interfaces not provide it?
- What workload exposes the difference?
- What would make the proposed mechanism lose to a simpler baseline?

## Human-Readable Synthesis

The public page is not a research notebook and not the direct output of a search tool.

Write for a technically qualified reader who has not read the sources. A useful progression is:

1. concrete situation and consequence;
2. why the obvious solution appears sufficient;
3. the missing mechanism;
4. evidence and competing explanations;
5. architecture or engineering consequence;
6. current gaps;
7. promising directions;
8. evidence that would change the conclusion.

Use a recurring scenario when possible. Introduce a formal term only after the reader understands the pattern it names.

Do not begin with a source matrix, field overview, taxonomy, formula, or coined abstraction.

## Required Public Sections

Every Daily Report article must render these functions, either from Markdown or from a route-specific rendering component:

- `Where current work is still weak` / `现有研究还缺什么`
- `Promising directions with academic and production value` / `兼具学术价值与生产价值的方向`

The gap section should contain two to five concrete gaps. The idea section should contain a small number of developed directions, not a brainstorm list.

For assumptions, counterexamples, and decisive future evidence, use a natural heading such as:

- `What would change this conclusion?`
- `哪些结果会改变这个判断？`

Do not use a templated heading built around the words `scope` or `limitations`.

## Public Presentation

The public page should read as a carefully edited technical report. Do not add process-oriented banners, warning boxes, generation badges, provenance callouts, review-status text, or footer disclaimers. Keep the reader's attention on the question, evidence, reasoning, gaps, and proposed directions.

Do not expose a Git committer as the article author merely because they merged the text.

## Bilingual Writing

English and Chinese versions share evidence and claims, not sentence boundaries. Write each version naturally.

In Chinese, use Chinese for ordinary concepts. Keep English for proper nouns, identifiers, code, and useful terms of art. Do not let a sentence's grammar rely on a stack of English nouns.

## Publication Validation

Before merge, verify:

- title, description, opening, gaps, ideas, decisive counterevidence, and references;
- primary-source support for central factual claims;
- no process-oriented callout, provenance note, review-status line, or footer disclaimer;
- Daily Report labels in desktop and mobile navigation and the Blog call to action;
- stable `/research/` and `/zh/research/` canonical URLs;
- bilingual hreflang pairs;
- no inferred human author metadata;
- sitemap, structured data, internal links, and browser rendering;
- full repository CI.

After merge, verify that the exact merge commit deployed and inspect the public English and Chinese pages.

## No-Report Outcome

Do not publish when the scan produces only interesting sources or a fluent summary. Record the strongest unresolved question and the evidence required to support a real gap and testable idea. Cadence never overrides the publication contract.
