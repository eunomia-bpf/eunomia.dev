# Research Method

Use this reference for source coverage, topic selection, evidence grading, and
the shape of a public report.

## Research Windows

Use the latest seven days to find current releases, papers, incidents, and
discussion when the assignment is time-sensitive. Read the latest 48 hours
first, then widen to the previous 7-30 days for mechanisms, contradictions,
deployments, and engineering context.

Use older primary sources for standards, prior art, baselines, and theory.
Confirm both the publication date and the date of the underlying event. A recent
repost of an old result is not a recent event.

A durable Research brief does not need a news hook. Its source cutoff should
still be explicit so later readers know which evidence the claim includes.

## Source Coverage

For a scheduled broad weekly analysis, materially review at least:

- 20 distinct papers
- 20 distinct industry or open-source projects
- 10 distinct current-event sources published within the latest seven days

This gate exists to prevent a broad landscape report from being built from a
small or repetitive sample. It does not apply mechanically to every focused
Research brief. A focused brief should use enough primary evidence to establish
the mechanism, the strongest alternative, and the boundary of the claim. Do not
pad a narrow question with unrelated sources.

Count a source only when it changes the analysis by adding a fact, mechanism,
comparison, contradiction, adoption signal, failure, or useful question.
Several articles repeating one announcement remain one piece of evidence.

Useful source families include:

- **Academic research:** conference papers, journals, arXiv papers, workshop
  material, datasets, benchmarks, and author artifacts.
- **Engineering practice:** architecture notes, postmortems, performance
  studies, migration reports, and security disclosures.
- **Official product material:** release notes, specifications, system cards,
  and detailed documentation. Treat marketing metrics as claims until they are
  independently supported.
- **Open source:** repositories, releases, commits, issues, pull requests,
  discussions, maintainer notes, and reproducible artifacts.
- **Public institutions:** standards bodies, foundations, agencies, research
  laboratories, and formal consultations.
- **Community evidence:** public maintainer discussions and practitioner reports
  that expose disagreement or operational friction.

Discovery sources such as newsletters, aggregators, and social posts are leads,
not final support for factual claims.

## Evidence Lattice

Before choosing a broad thesis, try to fill several independent evidence roles:

- **Current change:** a release, incident, paper, policy change, or adoption
  event that explains why the question matters now.
- **Mechanism evidence:** a paper, benchmark, dataset, or systematic study that
  tests or defines the underlying mechanism.
- **Implementation evidence:** source code, a standard, detailed documentation,
  or a reproducible artifact showing how the mechanism is built.
- **Deployment evidence:** a postmortem, issue, maintainer discussion, or
  first-person report showing where the mechanism succeeds or fails.
- **Counterevidence:** a result, workload, or explanation that would narrow or
  overturn the thesis.

A broad report should normally use at least three roles, including mechanism
evidence and one non-vendor technical artifact. A narrow report may use fewer
when one primary source directly answers the question. Never add a weak citation
solely to complete the pattern.

## Candidate Record

For each serious candidate, capture:

- title and primary URL
- source family and owner
- publication date and underlying event date
- concrete new fact or claim
- available data, code, logs, or implementation detail
- independent support
- limitation, conflict of interest, or missing evidence
- possible mechanism or tension
- question it could help answer

Keep this compact and private to the research process.

## Topic Selection

A useful topic commonly has more than one of these properties:

- independent sources point to the same underlying change;
- a new artifact or dataset makes a claim inspectable;
- research results and production experience disagree;
- a release changes what developers can build, operate, secure, or afford;
- a failure mode or second-order effect is missing from surface coverage;
- the evidence can change a reader's technical decision.

Prefer a narrow question with strong evidence over a broad trend supported by
repetition.

Compare the candidate with existing Eunomia Research pages and blog posts before
drafting. The central question, argument, or conclusion must be materially
different. A new headline, product, example, or named abstraction does not make
a thesis new.

## Evidence Roles And Attribution

Prefer primary data, code, experiments, standards, official documentation, and
first-person engineering reports. Peer-reviewed work and method-complete
preprints can establish mechanisms and measured results. Commercial pages can
establish what an organization announced, not a general fact without
corroboration.

Cross-validate a central inference with independent source types when the claim
extends beyond one source's own system. Look for agreement on mechanism rather
than identical wording.

Place attribution near the claim it supports. A reference list is not a
substitute for an evidence chain in the article.

## Repository-Owned Work

After a thesis exists, inspect `docs/papers/registry.yaml`, public paper text,
project artifacts, and related Eunomia posts for genuinely relevant evidence.

Use repository-owned work as one evidence node. Give it the same editorial
distance, limitations, and verification standard as outside work. Avoid a
dedicated promotional section or a forced tie-back to Eunomia products.

Omit the connection when it does not improve the reader's model.

## Human-Readable Synthesis

The public article is not the research notebook and not the output of a search
tool.

Write for a technically qualified reader who knows the broad domain but has not
read the source corpus. Build the article around the reader's decisions:

1. show a concrete situation;
2. explain why the obvious mental model fails;
3. identify the mechanism;
4. state the design or decision that changes;
5. introduce evidence where it changes the argument;
6. test alternatives and boundaries.

Use one recurring scenario when possible. It gives the reader a stable object
to revisit as the article moves from failure to mechanism and design.

Do not start with:

- a field overview;
- a dense table of products or papers;
- a source-by-source literature survey;
- a taxonomy with no prior example;
- a formula;
- a coined term.

A new abstraction should compress a pattern the reader already understands. If
the reader must learn the term before seeing the problem, the order is wrong.

### Titles And Openings

A title should name the problem or decision in ordinary language. A subtitle or
later section can introduce the formal term.

Within the first 250 English words or 500 Chinese characters, establish:

- the affected actor or system;
- the concrete failure or unmet need;
- why existing practice is insufficient;
- the article's main claim.

Do not use the opening to list credentials, sources, or claims of novelty.

### Technical Depth

Depth should appear as:

- a causal mechanism;
- a meaningful comparison;
- a non-obvious consequence;
- an architecture choice;
- an evaluation that can distinguish alternatives;
- an explicit condition that would falsify the claim.

Term density, equations, and article length are not evidence of depth.

Formal notation is optional. Use it after an intuitive explanation and only when
it removes ambiguity. Define every symbol locally and explain what engineering
decision the formalism changes.

### Literature And Tables

Synthesize literature by mechanism or disagreement. A paragraph should not exist
merely to mention another paper.

Use a table only when all rows share the same comparison dimensions. Put dense
runtime or related-work tables after the reader understands the problem, not in
the opening third.

### Chinese And English

Chinese and English versions share evidence and claims, not sentence boundaries.
Write each version naturally.

In Chinese prose, use Chinese for ordinary concepts. Include an English term on
first mention when it is a recognized term of art or useful search key. Do not
make a sentence's grammar depend on a stack of English nouns.

## Public Report Shape

A report normally needs these editorial functions, but they do not have to be
rigid headings:

- a concrete problem and reader stakes;
- a clear, contestable thesis in ordinary language;
- the mechanism connecting the evidence;
- what existing mechanisms solve and leave open;
- a design, architecture, or decision consequence;
- competing evidence and alternative explanations;
- implementation or operational implications;
- evaluation criteria;
- uncertainty, applicability limits, and falsification conditions.

Long form is justified when the argument needs room. It is not a requirement to
include every source reviewed.

End Chinese reports with `## 参考资料`. List only sources materially used in the
argument.

## One-Pass Reader Test

Before publication, read only the title, description, opening, headings, and
conclusion. A qualified reader should be able to answer:

- What problem is being solved?
- Who experiences it?
- What is the central claim?
- What design or operational decision changes?
- What evidence matters most?
- What would make the claim false?

Then read the full article once. Rewrite if the reader must:

- remember several undefined terms;
- reconstruct missing premises;
- infer why a source was mentioned;
- translate English-heavy Chinese clauses;
- read a formal definition before seeing an example;
- confuse a worker's local success with the workflow's final correctness.

## No-Report Outcome

If the scan produces interesting links but no defensible thesis, do not draft a
deep report. Record the strongest unresolved question and the evidence that
would unblock it. A truthful no-report outcome is better than a topical summary
disguised as analysis.
