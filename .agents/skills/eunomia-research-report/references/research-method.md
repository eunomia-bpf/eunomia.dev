# Research Method

Use this reference for source coverage, topic selection, evidence grading, and
public report shape.

## Research Windows

Start with the latest 48 hours to find current news, releases, papers, incidents,
and discussion. Confirm both the publication date and the date of the underlying
event. A new repost of an old result is not a new event.

Widen to the latest working day and then the previous 7-30 days for slow-moving
research, engineering writeups, and context that has not yet reached broad
discussion. Use older primary sources for mechanisms, prior art, baselines, and
counterevidence.

The 48-hour window sets discovery priority. It is not a freshness requirement
for every citation.

## Source Coverage

Run a lightweight search across every family. Do not require a minimum number of
results from any family, and do not create a separate artifact for each one.

- **Academic research:** conference papers, journals, arXiv, workshop material,
  datasets, benchmark papers, and author project pages.
- **Engineering practice:** engineering blogs, architecture notes, postmortems,
  performance studies, migration reports, and security disclosures.
- **Commercial products:** official launches, release notes, documentation,
  system cards, engineering-relevant pricing or limit changes, and concrete case
  studies. Treat marketing claims as claims until corroborated.
- **Open source:** repositories, releases, commits, issues, pull requests,
  discussions, maintainer notes, adoption signals, and reproducible artifacts.
- **Public institutions:** government agencies, standards bodies, foundations,
  research laboratories, specifications, consultations, and regulatory notices.
- **Social and community discussion:** LinkedIn, Xiaohongshu, X, Reddit, Hacker
  News, Lobsters, Zhihu, Juejin, and public maintainer discussions. Use these to
  find practitioner experience, disagreement, and emerging questions.

Searching every family does not mean citing every family. Include a source in
reader-facing work only when it adds a fact, mechanism, contradiction, adoption
signal, real failure, or useful research question.

## Candidate Record

For each serious candidate, capture:

- title and primary URL
- source family and source owner
- publication date and underlying event date
- concrete new fact or claim
- available data, code, logs, or implementation detail
- whether another source independently supports it
- limitation, conflict of interest, or missing evidence
- possible mechanism or tension
- question it could help answer

Keep this compact. The record is a research aid, not a public artifact.

## Topic Selection

A useful topic commonly has more than one of these properties:

- independent sources point to the same underlying change
- a new artifact or dataset makes a claim inspectable
- research results and production experience disagree
- a release changes what developers can build, operate, secure, or afford
- a failure, limit, or second-order effect is missing from surface coverage
- the evidence can change a reader's technical decision

Prefer a narrower question with strong evidence over a broad trend supported by
weak repetition. Do not choose a topic because it promotes a repository project.

## Evidence Roles

Prefer primary data, code, experiments, standards, official documentation, and
first-person engineering reports for factual claims. Peer-reviewed work and
method-complete preprints can establish mechanisms and measured results.
Commercial pages and social posts can establish what an organization announced
or what a practitioner observed, but not a broader fact without corroboration.

Use multiple independent evidence clusters for a broad thesis when available.
One decisive source can justify a narrow report, but source volume must not be
mistaken for independence. Ten articles repeating one press release remain one
piece of evidence.

## Public Report Shape

A public report should be independently readable and useful to a technical
reader. It normally needs these editorial functions, but they do not have to be
separate headings or appear in a rigid order:

- a clear, contestable thesis
- the most important facts and what changed recently
- why the timing matters
- an evidence chain grounded in primary material
- the mechanism connecting the evidence
- patterns, contradictions, and alternative explanations
- original analysis, boundaries, and second-order effects
- practical implications for developers or operators
- uncertainty, falsification conditions, and questions to track next

Keep the argument tighter than the research dossier. Chinese reports often land
around 4,000-7,000 Chinese characters, but this is a soft editorial range, not a
requirement. A shorter report is better when the thesis, evidence, mechanism,
counterargument, and implications are already complete. Use 7,000-10,000 only
when the topic genuinely needs additional cases or methodological explanation.
Go beyond that for occasional flagship work, not routine depth. Never pad or cut
solely to hit a number.

## No-Report Outcome

If the scan produces interesting links but no defensible thesis, do not draft a
deep report. Record the source families checked, strongest unresolved question,
and what future evidence would unblock it. A truthful no-report result is better
than a topical summary disguised as analysis.
