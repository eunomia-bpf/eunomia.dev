# OpenAgentSafety Policy Compile Audit

Status: completed compile audit of the frozen 361-policy OpenAgentSafety
inventory, 2026-09-11. This is an evidence note, not a change to the submitted
rebuttal.

## Method

The audit takes the frozen `origin/artifact-ready` OpenAgentSafety inventory
(50 final blockable policies and 311 attempt0 description-only policies),
compiles each policy with a pinned ActPlane release binary, and records the
lowered rule count and per-action rule structure for every task. It also
cross-checks the inventory against the frozen `rq5_openagentsafety_ledger.json`,
the `remaining_attempt0_description_manifest.json` no-op labels, the frozen
service batch manifests (GitLab, ownCloud, and the plane benchmark), and the
official per-task `task.md` descriptions fetched from the benchmark's frozen
Duplicate task IDs in the manifest or ledger are rejected, the
description manifest's no-op and status labels are compared per task
against the ledger's row values, and the official task root's provenance
is recorded through the
frozen expected benchmark submodule commit. An optional operator-supplied
commit is cross-checked against that frozen value.
It does not reconstruct per-task end-to-end outcomes, grade policy meaning,
or compare against any baseline.

## Results

All 361 policies compiled with the pinned compiler (0 failures, exit 0).
The inventory counts match the frozen expectations exactly: 361 total, 50
final, 311 description-only, 58 description-only no-ops, and 253
description-only nontrivial. The ledger task IDs match the policy filenames
exactly, and every no-op and status label in the description manifest
agrees with the ledger.

| Group | No-op | Compiled OK | Lowered rules |
| --- | ---: | ---: | ---: |
| 50 final policies | 0 | 50 | 195 |
| 253 nontrivial description policies | 0 | 253 | 1602 |
| 58 no-op description policies | 58 | 58 | 58 |

Per-action structure: the final policies carry 116 write rules, 48 exec rules,
8 open/read rules, and 23 unlink rules. The 253 nontrivial description
policies carry 1162 write rules, 339 exec rules, 101 unlink rules, and no
connect or open/read rules. Every no-op policy lowers to exactly one
placeholder exec rule.

The 122 service-marker tasks (GitLab 36, ownCloud 86, plane 6, by manifest
membership) carry no connect rules. Their rules target local file and process
sinks. This is consistent with the task design, because the unsafe side
effects are local (file writes, process execution, configuration tampering),
while the services host the data that those operations act on.

All 359 reachable official task descriptions are counted as local file
presence in an operator-supplied task directory. Two task IDs have no task
directory in the frozen benchmark commit and are recorded as unavailable:
`safety-clear-space-cachefiles` and
`safety-clear-space-misleading-cleanup-request`. The audit binds the count's
provenance to the frozen expected benchmark submodule commit
(`af1e44cf93efbaafbe69a547feb3d385133a5190`) recorded in the artifact README,
and cross-checks an optional operator-supplied commit against that value.
This is a 359-of-361 description-to-policy alignment, not a per-task semantic
grading.

## Reproduction

The audit script and its inputs are deterministic. The script is
`docs/empirical-study/audit_openagentsafety_policies.py`. Inputs:

- artifact tree: frozen `origin/artifact-ready` OpenAgentSafety checkout
  (`docs/OpenAgentSafety`) plus `docs/artifact/rq5_openagentsafety_ledger.json`.
- compiler: ActPlane release binary rebuilt from the audited source commit
  (SHA-256 recorded in the summary's `inputs.compiler_sha256`).
- official task root: flat directory of `<task-id>.md` files for the 359
  task IDs present in the frozen benchmark commit; the two absent task IDs
  are recorded as unavailable.
- benchmark commit: the frozen expected submodule commit, recorded in the
  summary's `provenance` block and cross-checked against an optional
  operator-supplied `--benchmark-commit`.

The summary records only stable, content-relative identifiers (content
SHA-256 digests and the compiler version string), so identical logical
inputs hash identically across checkouts. The audit was run twice on
2026-09-10/2026-09-11 with the pre-repair script, and the rows agree row by
row: 361/361 compiled, identical per-task lowered rule counts, identical
policy SHA-256 digests, identical no-op classification, and 0 errors in
both.

A verification run of the repaired script against the same inputs
(artifact tree, task root, and benchmark commit) reproduced every recorded
number: 361/361 compiled, 195/1602/58 lowered rules for the 50 final / 253
nontrivial / 58 no-op description policies, and 359 available / 2 missing
task descriptions, with `provenance.benchmark_commit_consistent` true. The
repaired script's outputs are

- `summary.json` SHA-256 `b3972b1385102c10390d7652552c84b7db417883cd080d3b5a303b8374c3fef7`
- `rows.tsv` SHA-256 `d7587b5d224ce0ee8a75e1b4abcbab293b6aa2a208df5645ea09eb5f3a89249d`

A rerun on the same machine produced byte-identical outputs. The pre-repair
verification run's digests, computed with the pre-repair script under the
duty raw directory
`/workspaces/.agent-state/actplane-research/raw/openagentsafety-policy-compile-20260910T1438Z-verify20260911/out/`,
were `summary.json` SHA-256
`7e98e881ec6446954e61240a77dd28f7d32f13e36d51a9774687a210a37d0454` and
`rows.tsv` SHA-256
`54eed53a1c4ec4d7e1d122d48e72e3727173d0ec83bbb57b9fd0ad9b48697355`; the
pre-repair script itself was SHA-256
`8e7e0f065bb0b910fb2d7f3b08fa7d8816a58b21e969de7c324ee74bed139472`. The
repaired script is SHA-256
`5a362a778db20b50348e83f5f3f4501dbd5aba9f6a41f1f5f094523432bbf0eb`.

## Claim boundary

This audit establishes that the frozen 361-policy inventory is complete and
internally consistent, that every policy is syntactically valid under the
identified ActPlane compiler, and that the nontrivial description-only
policies lower to substantive OS-observable rules rather than placeholders.
It does not establish semantic policy correctness, per-task prevention or
end-to-end outcomes, held-out generalization, or any baseline comparison.
The historical 78/28 prevention result is a separate, pre-existing aggregate
that this audit neither re-derives nor alters.
