# SEO plan

## Purpose

Keep the daily operation evidence-driven: analyze enabled sources, publish exactly one technically strong bilingual Daily Report, change search-facing behavior only when evidence supports it, and deliver through CI, squash merge, exact deployment verification, and one merged-PR closeout comment.

## Current priorities

1. Finish the September 29 map-reuse semantic-compatibility publication on a fresh branch and non-draft PR.
2. Keep the newest-ten editorial window within the repository target. Today's selected eBPF report keeps the window at 7 eBPF / 1 Agent / 2 adjacent.
3. Do not make an unrelated technical SEO/GEO change without a reproducible defect or measurable opportunity.
4. Treat incomplete Search Console date coverage and partial GA4 aggregates as partial evidence rather than filling gaps.
5. Continue the **eBPF Deployment Compatibility and Lifecycle** series with boundaries that remain distinct from capability admission, cross-kernel semantics, interface negotiation, and transactional live upgrade.
6. Candidate future boundaries include reboot-safe state reconstruction and persistent BPF-link ownership after controller restart, but neither should be duplicated while an unpublished attempt already exists.
7. Keep the shared SEO skill submodule pinned until the repository's single-PR daily delivery contract is migrated cleanly.

## Delivery policy

- One fresh `daily/` branch and one real non-draft PR per run.
- Exactly one new bilingual Daily Report topic per run.
- Terminal-green expected and required CI before merge.
- Full final diff and generated-output self-review before squash merge.
- Exact squash-commit production deployment verification.
- Public English and Chinese acceptance verification.
- Exactly one compact closeout comment on the merged PR.
- Never disable the recurring operations schedule because a run is blocked.
