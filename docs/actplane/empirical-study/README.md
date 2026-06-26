# ActPlane Empirical Study

This directory keeps the curated empirical-study artifacts that are useful to
understand why ActPlane's policy language targets process, file, network, and
temporal agent behavior. Scratch notes, raw model runs, tuning logs, and full
corpus workspaces stay on the artifact or backup refs described in
`docs/ARTIFACT.md`.

## Snapshot

The study analyzed agent instruction files (`CLAUDE.md` and `AGENTS.md`) from
popular AI-agent code repositories. The retained product-branch artifacts are
aggregate outputs, not the full raw corpus.

| Metric | Value |
| --- | ---: |
| In-corpus repositories | 144 |
| Instruction files | 228 |
| Instruction-file text | 39,803 lines |
| Candidate normative lines | 3,762 |
| ActPlane-related candidate lines | 529 |
| Repositories with at least one ActPlane-related candidate | 101 |

The candidate-line counts are keyword/category extraction results and should be
read as aggregate evidence for prevalence, not as a hand-labeled ground truth.
The product branch keeps the aggregate study outputs that are independent of
compiler syntax. DSL-specific ruleset artifacts should live only on artifact
refs after they have been regenerated and validated against the current
compiler.

## Main Findings

- Agent instruction files frequently contain operational guardrails that are
  below the tool layer: VCS gates, secrets handling, test-before-commit rules,
  workspace boundaries, destructive-operation guards, network egress limits, and
  mediation through project tools.
- These guardrails map to ActPlane primitives: exec argument matching, labeled
  file and endpoint sources, source-to-sink label flow, `after` ordering,
  lineage gates, target scoping, and declassification.
- Style and code-quality instructions are common in the broader corpus but are
  intentionally outside ActPlane's enforcement scope unless they correspond to a
  concrete OS-observable action.

## Retained Artifacts

- `candidate_rules_144.tsv`: aggregate candidate-line extraction with repo,
  file family, line number, category guess, and source text.
- `figures/`: generated summary figures for the empirical study.

The raw corpus, raw traces, intermediate coding notes, old evaluation drafts,
and exploratory scripts are intentionally not kept in the product branch.
