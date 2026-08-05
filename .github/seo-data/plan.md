# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, AI-agent runtime
infrastructure, observability, enforcement, and heterogeneous-systems research.
Optimize for technically useful discovery and citation by people and software
agents without creating low-value, mass-generated content.

## Success signals

- Search Console clicks, impressions, click-through rate, and canonical/indexing
  state once a public-safe export path is configured; preserve each metric's
  source-native meaning.
- Public-safe GA4 acquisition and engaged-session signals once configured, used
  to distinguish discovery from useful on-site follow-through.
- Stable crawlability, canonical ownership, language alternates, structured data,
  internal links, and production verification enforced by repository CI and live
  checks.
- Qualified movement from relevant technical pages to public repositories,
  papers, tutorials, and demos, without inventing a blended SEO score.
- Repeatable evidence that important pages are accurately retrievable and cited
  for relevant technical questions; record method and date rather than claiming
  universal AI visibility.

## Operating constraints

- Raw analytics and private identifiers stay outside Git.
- Every automated SEO run uses a fresh `seo/` branch and a real non-draft pull
  request; this scoped contract overrides the general direct-`main` repository
  rule for work invoked by `daily-task.md`.
- Required and expected CI must pass before final automated self-review.
- A clean final review is followed by squash merge and branch deletion; normal
  operation does not require human review.
- Site changes wait for the exact squash commit's production deployment and
  public verification.
- Post-merge evidence is recorded through a metadata-only closeout pull request
  using the same CI, self-review, and squash-merge rules.
- Implement at most one coherent site change per main pull request, and permit a
  no-change evidence run when no defensible improvement is supported.
- `.agents/skills/seo-geo` remains the Eunomia-specific technical checklist;
  `.github/seo-skills` owns shared collection and delivery mechanics.

Short-term fixes, one-off audits, and remediation backlogs belong in GitHub
issues, not in this durable plan.
