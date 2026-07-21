# Research Method

Use this reference for source coverage, topic selection, evidence grading, and
public report shape.

## Research Windows

Cover the latest seven days to find current news, releases, papers, incidents,
and discussion, reading the latest 48 hours first. Confirm both the publication
date and the date of the underlying event. A new repost of an old result is not
a new event.

Widen to the latest working day and then the previous 7-30 days for slow-moving
research, engineering writeups, and context that has not yet reached broad
discussion. Use older primary sources for mechanisms, prior art, baselines, and
counterevidence.

The seven-day window sets the reporting cadence, while the latest 48 hours set
discovery priority. Neither is a freshness requirement for every citation.

## Source Coverage

For a scheduled weekly analysis, materially review at least 20 distinct papers,
20 distinct industry or open-source projects, and 10 distinct news or
current-event sources published within the latest seven days. A paper counts in
the first group; a vendor system, production implementation, or open-source
repository counts in the second. The third group exists to establish what
changed during the reporting window. Older standards, datasets, documentation,
and background material may strengthen the analysis but do not satisfy that
current-news requirement. Count a source once even when several articles repeat
it. Do not create a separate inventory artifact.

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

Every counted source must materially affect the report by adding a fact,
mechanism, comparison, contradiction, adoption signal, real failure, or useful
research question. If a source does not affect the analysis, it does not satisfy
the corpus gate.

## Evidence Lattice

Before selecting a broad thesis, try to fill four different evidence roles:

- **Current change:** a recent release, paper, incident, policy change, or
  measurable adoption event that explains why the question matters now.
- **Academic or methodological evidence:** a paper, benchmark, dataset, or
  systematic study that tests a mechanism, defines a construct, or exposes a
  measurement boundary.
- **Implementation evidence:** source code, a reproducible artifact, a standard,
  an engineering report, or detailed technical documentation showing how the
  mechanism is built or operated.
- **Deployment evidence:** a postmortem, issue, maintainer discussion, user
  report, or public community thread showing where the mechanism succeeds,
  fails, or creates friction in practice.

A broad report should normally use at least three roles, including academic or
methodological evidence and one non-vendor technical artifact. A narrow report
may use fewer when one primary source directly answers the question. Never add a
weak citation solely to satisfy a role.

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

Compare the candidate with existing Eunomia articles before drafting. Its
central question, argument, or conclusion must be materially different. A new
headline, recent news hook, product, or example does not create a new thesis by
itself.

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

Cross-validate each central inference with at least two independent source types
when the claim extends beyond one source's own product or experiment. Look for
agreement on mechanism, not identical wording. Record disagreement explicitly:
a paper can challenge a product metric, an issue can expose an operational limit
missing from documentation, and a standard can show which data remains portable.

## Repository-Owned Work

After the thesis exists, inspect `docs/papers/registry.yaml`, public paper text,
project artifacts, and related Eunomia posts for genuinely relevant evidence.
Use repository-owned work as one evidence node, never as the destination of the
argument. In reader-facing prose, give it the same editorial distance as any
third-party paper or project. Lead with the mechanism or result, name the work
directly, and avoid first-person possessives, "our research," or explanations of
its relationship to the repository. Usually one or two sentences inside the
surrounding argument are enough. Link the source nearby, state its scope and
limitations, and let independent evidence carry the broader claim. Omit the
connection when it does not improve the reader's model. Expand only when the
paper or project is itself the report's explicit subject.

Do not create a dedicated promotional section, project roundup, call to action,
or paragraph whose only purpose is to establish Eunomia's involvement.

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

Write future-tracking questions in capability or mechanism terms. Avoid a list
of named products, vendors, or anticipated launches unless the report studies
that product directly. Product names belong in the evidence chain when needed,
not in a closing watchlist that reads like promotion.

End Chinese reports with `## 参考资料`. A scheduled weekly report must account
for the sources that materially informed it, including at least 20 papers, 20
industry or open-source projects, and 10 other useful materials. Inline
attribution still belongs beside the claim it supports. Do not inflate the list
with unread or unused links.

Write a long-form analysis rather than a compressed news summary. Give the
argument enough space to synthesize the required corpus, compare evidence,
explain mechanisms, test alternatives, and derive practical consequences. There
is no word-count quota; length follows the work the evidence requires. Never pad
or cut solely to hit a number.

## No-Report Outcome

If the scan produces interesting links but no defensible thesis, do not draft a
deep report. Record the source families checked, strongest unresolved question,
and what future evidence would unblock it. A truthful no-report result is better
than a topical summary disguised as analysis.
