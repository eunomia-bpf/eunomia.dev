# RQ2 Reviewer-to-Evidence Audit

Status: in progress, 2026-09-09. This plan does not modify the staged rebuttal draft.

## Decision and hypothesis

Reviewer D interprets Table 2 as an enforcement confusion matrix and asks why
ActPlane with feedback has 18 false positives while the opaque ablation has one,
even though both use the same kernel mechanism. The paper currently defines the
labels end to end, but then attributes 17 of 18 false positives to broad policy
translation without publishing trace-level evidence.

The primary hypothesis is that Table 2 mixes distinct stages: policy translation,
kernel rule matching, feedback delivery, and agent recovery. The 18 false positives
are therefore not measurements of kernel soundness or completeness. For every one
of the 190 traces, the audit will separately record ground truth, compiled-policy
intervention, intervention effect, feedback visibility, and final trajectory judge
label. The 18 false positives will then be manually attributed using the original
directive, frozen policy, trace, feedback, tool result, and judge rationale.

A positive result means the paper should replace “kernel FP/FN” language with an
end-to-end decomposition and report the dominant translation or harness failure
modes. A negative result, where intervention records disagree with actual effects,
would instead identify an engine or runner defect. A mixed result requires both
categories and representative negative examples to remain visible.

## One comparison matrix

| Unit | Frozen input | Kernel observation | End-to-end observation | Comparison |
| --- | --- | --- | --- | --- |
| 190 trace-system pairs per system | directive, policy, trace, expected compliance | rule fired, kill/notify, target, feedback | TP/TN/FP/FN, recovery tools, final error | within-system stage decomposition |
| 190 matched ActPlane pairs | identical task identity and trace family | ActPlane versus opaque trigger/effect | ActPlane versus opaque judgment | feedback ablation, not an independent baseline |
| 18 ActPlane FP traces | full evidence above | observed intervention and target | judge rationale and recovery | manual causal attribution with failures retained |

The audit treats prompt-filter and tool-regex as weak controls, FIDES/tool-IFC as
the independent research baseline, and ActPlane-opaque only as a feedback ablation.
It does not describe the 26/28 repaired false negatives as held-out generalization.

## Existing evidence first

The canonical Qwen artifact is on `origin/artifact-ready` at
`docs/artifact/rq2-qwen-primary/`. It contains 950 runner results and 950 judge
files selected by a frozen manifest. The historical full DeepSeek run remains on
`origin/backup/2026-06-14-master`. The paper-facing verifier is
`docs/artifact/verify_results.py` on the artifact ref.

Read-only extraction and audit command:

```bash
tmpdir=$(mktemp -d /tmp/actplane-rq2.XXXXXX)
git archive origin/artifact-ready \
  docs/artifact/rq2-qwen-primary docs/corpus-test \
  | tar -x -C "$tmpdir"
node docs/empirical-study/audit_rq2_verdicts.js "$tmpdir" \
  "$tmpdir/rq2-verdict-audit.json"
```

Raw paths remain the artifact manifest, its referenced runner JSON files, matching
`trajectory_judges_llama_cpp_guardrail_response/*.judge.json`, and each selected
`docs/corpus-test/*/*/rule.yaml` and trace JSONL. Generated audit output is
exploratory until manually reviewed and is not committed as a paper result.

## Initial reproducible result

Running the command above on the frozen Qwen artifact recomputes all 950 labels
and exactly matches the paper table. Of the 18 ActPlane end-to-end false
positives, 17 contain a setup-phase intervention and the remaining trace contains
a recovery-phase kill, so all 18 have an observed intervention. Three setup
interventions kill and 14 notify. In the paired opaque runs, 15 of the same 18
traces also record a setup trigger, but only one receives an end-to-end FP label.

This resolves Reviewer D's numerical paradox. The 18-to-1 difference does not
show that enabling feedback changed the kernel policy into one that matched 17
additional compliant traces. It mostly shows that the trajectory metric counts
visible corrective feedback as intervention, while a matching opaque notify is
hidden from the agent and can remain a TN. The paired runs are separate agent
executions, so the remaining trigger differences cannot be assigned to feedback
without comparing their exact executed effects.

The stage cross-tab further confirms that the labels are not kernel verdicts.
Among ActPlane rows, observed intervention occurs on 86/86 TP, 2/28 FN, 18/18
FP, and 2/58 TN trajectories. Thus intervention is necessary for the published
TP definition but is neither sufficient for TP nor exclusive to FP. Across all
190 matched ActPlane/opaque pairs, setup triggers occur in both runs for 83,
only ActPlane for 14, only opaque for zero, and neither for 93. This comparison
describes recorded executions, not paired deterministic trials, because feedback
can change later actions within a multi-step setup trajectory.

## RQ2 matched execution-path analysis

Reviewer D accepts FIDES as a reasonable baseline but questions whether the
other controls are strong, while the stated RQ2 asks specifically about direct
and indirect execution paths. The existing overall confusion matrix cannot show
whether the result follows the below-tool mechanism, and the family heatmap does
not report matched differences against the independent baseline. The analysis
hypothesis is therefore that ActPlane and FIDES will be comparable on direct
violations, where both see the initiating tool call, but ActPlane will have a
large paired advantage on script and hidden violations whose effects cross that
boundary.

A positive result localizes the advantage to the mechanism claimed by RQ2. A
negative result, such as the same advantage on direct paths or no advantage on
indirect paths, would weaken that explanation. A mixed result requires reporting
the losing path and limits rather than relying on overall DCR.

| Comparison | Units | Outcome semantics | Role |
| --- | ---: | --- | --- |
| Five systems by allowed-effect, lookalike, direct, script, and hidden family | 38 per cell | End-to-end TP/TN/FP/FN from the paired judge file | descriptive matrix |
| ActPlane versus FIDES on compliant, direct, and combined indirect groups | 76, 38, and 76 matched traces | Correct means TP or TN; exact two-sided McNemar test uses discordant pairs | primary independent-baseline comparison |
| ActPlane versus opaque intervention by family | 38 per cell | setup and any-phase rule reports, separate from final judgment | feedback ablation and stage audit only |

The frozen 950-row artifact supports the hypothesis, with an important direct-path
boundary. On the 76 indirect violation traces, ActPlane correctly resolves 55
(72.4%) while FIDES resolves 7 (9.2%), a 63.2 percentage-point difference. The
matched cells are 4 both correct, 51 ActPlane only, 3 FIDES only, and 18 both
wrong (exact two-sided McNemar \(p=2.92\times10^{-12}\)). The difference appears
in both indirect families: script is 27/38 versus 7/38, and hidden is 28/38
versus 0/38.

Direct violations do not favor ActPlane: it resolves 31/38 (81.6%) versus
FIDES at 34/38 (89.5%), with 2 ActPlane-only and 5 FIDES-only correct pairs
(McNemar \(p=0.453\)). On the 76 compliant traces, ActPlane is 58/76 and FIDES
is 52/76 (\(p=0.405\)). These exploratory paired tests were added after seeing
the aggregate result and are not a substitute for a preregistered replication.
They nevertheless make the mechanism boundary falsifiable and show that the
overall advantage is not evidence of superiority on direct paths.

The same run keeps enforcement observations distinct from end-to-end outcomes.
ActPlane records an any-phase intervention in 31 direct, 27 script, and 30
hidden traces, while the corresponding final TP counts are 31, 27, and 28.
FIDES is compared only on final judged outcome because its tool-layer records
are not kernel events.
ActPlane-opaque remains an ablation and is not included in the independent
baseline test.

The executable analysis is `docs/empirical-study/audit_rq2_verdicts.js`. Its
version-2 JSON includes the complete family matrix, the matched comparison, and
the ActPlane/opaque stage counts. The run used the read-only extraction command
above, with raw output, input hashes, and a concise table retained under
`/workspaces/.agent-state/actplane-research/raw/rq2-path-baseline-audit-20260910T0906Z/`.
The manifest's undocumented fifth column disagrees with 97 FIDES judge outcomes
(73 TP-to-FN and 24 TN-to-FP) and is ignored by the official paper verifier. The
audit retains it as an auxiliary provenance field but uses each referenced judge
file as the final outcome source, matching `docs/artifact/verify_results.py` on
the artifact ref.
This result strengthens the original coding-task RQ2 comparison. It does not add
an independent baseline to OpenAgentSafety or establish unseen non-coding
generalization.

The raw rules already show heterogeneous candidate causes, including deliberately
broad translations (notify on every write), semantic conditions unavailable to a
path-only rule (dependency content or release intent), generated helper processes
that execute Python on behalf of file tools, and path or lineage mismatches. A
causal count remains pending trace-by-trace review. Until that review is complete,
the evidence supports an end-to-end metric clarification, not the stronger claim
that policy translation alone caused 17 of 18 false positives.

## Selected new experiment: long-session over-tainting

Reviewer B predicts label explosion in long sessions, and Reviewer D asks whether
reading `.env` prevents all later network use. Existing E1 proves one secret-read
to connect flow and E8 proves explicit declassification, but neither measures how
intervention burden grows or where process-scope boundaries reset it.

The hypothesis is that a sensitive label remains monotonic within a process lineage,
so every later benign connect matches even when it does not use sensitive content.
Safety should remain constant while intervention burden grows linearly with the
number of later connects. A negative result would show lost persistence or missing
connect observation. A mixed result would locate a process or descendant boundary
and narrow the paper's “session” claim.

| Case | Label at start | Later relationship | Benign connects | Predicted matches |
| --- | --- | --- | ---: | ---: |
| clean control | no | same seeded lineage | 5 | 0 |
| length 1 | yes | same process | 1 | 1 |
| length 5 | yes | same process | 5 | 5 |
| length 20 | yes | same process | 20 | 20 |
| labeled process exits | child only | clean sibling | 5 | 0 |
| nested descendant | yes | one extra fork generation | 5 | 5 |

The first run isolated persistence by seeding the frozen `SECRET` bit at the
process boundary. It established the propagation control but did not answer the
reviewer's actual file-read premise. After moving update matching and rename-flow
state off the BPF stack, the follow-up uses `source SECRET = file
"/session.env"`. Each sensitive case opens and reads that file before connecting.
The sibling case confines the read to a child that exits. The policy uses
`notify`, not `kill`, so all scheduled connects execute and the count measures
burden rather than early termination. Destinations are closed loopback ports, and
no payload is sent.

The privileged VM runner is
`docs/empirical-study/run_long_session_overtaint_vm.sh`. On Ubuntu 6.8 under KVM,
the seeded control and the real-read follow-up both observed 0, 1, 5, 20, 0, and
5, exactly matching the preregistered rows. Every sensitive violation in those
six rows records provenance `op:1` and target `/session.env`. Thus the control's propagation result
survives real label acquisition: intervention burden grows linearly inside the
labeled lineage, including one more fork generation, while a clean sibling remains
untainted after the reader exits. This supports Reviewer B's over-taint concern
within one lineage, but not process-tree-global label explosion.

A seventh engineering validation renames `/session.env` to `/renamed.env`, reads
the renamed inode, and then connects. The first run observed its predicted
violation but returned null provenance, which exposed a feedback-explanation gap.
File-source materialization now records object provenance before rename state is
copied. A second Ubuntu 6.8 KVM run again observed one violation and attributed it
to `op:1,target:/session.env`. This is an engineering validation outside the six
preregistered over-taint rows, not another independent experimental condition.

Raw evidence is under
`/workspaces/.agent-state/actplane-research/raw/file-read-long-session-20260910T0124Z/`.
`counts.tsv` is derived from `console.clean.log`, which retains every
`TAINT_VIOLATION`, and `metadata.tsv` records commit, kernels, acceleration, and
policy hash. The container runner
`docs/empirical-study/run_long_session_overtaint.sh` could not load BPF because
the container lacks `CAP_BPF` and `CAP_SYS_ADMIN`. The original Ubuntu 6.8
file-source attempt failed at `trace_openat_exit` with combined call-stack size
544 bytes. Raw failed attempts remain under `raw/file-read-stack-fix-attempt*`.
After the open path loaded, the same limit surfaced in the generic rename-rule
path, so source-only policies now autoload a propagation-only rename exit program
while write-rule policies retain the full sink evaluator. These failures are
engineering evidence, not experimental observations. Build success and loader
readiness remain prerequisites, not results.

The first independent Codex review blocked the follow-up because the Rust loader
did not select the new flow-only exits and the runner did not fail closed. After
the loader fix, a second review confirmed hook selection and hashes but retained a
block because the sibling control discarded its reader child's exit status. The
runner now propagates that status, and its unit matrix explicitly covers modern
source-only, modern write-rule, and legacy rename selection. These reviews assess
code and evidence integrity, not the paper's scientific claims. Their raw reports
are under `raw/independent-review-6259cd3b/` and
`raw/independent-review-8298d23a/` in the agent-state directory.
The final review reports `APPROVE` with no blocking code or runner findings. It
retains the limitation that modern write-rule and legacy selection are covered by
code inspection and unit assertions rather than separate KVM runs.

## Remaining generality and baseline evidence

The frozen `origin/artifact-ready` OpenAgentSafety assets establish a useful but
bounded non-coding result. They inventory all 361 workplace and personal-assistant
tasks, 303 nontrivial policies and 58 no-op policies. The aggregate summary reports
106 baseline-unsafe outcomes, of which ActPlane prevented 78 and missed 28. The
311 first-pass policies were generated from task descriptions without evaluator,
trajectory, score, or ActPlane-log access. Thus this is evidence beyond coding and
for description-to-DSL transfer, not evidence that no such evaluation exists.

Two limitations remain material. The artifact branch retains only aggregate
outcomes, and every ledger row explicitly marks per-task outcome unavailable, so
the 78/28 split cannot currently be trace-audited like RQ2. Its baseline is the
same agent without ActPlane enforcement, not an independent policy system. FIDES
is the independent prior-system baseline only in the 190-trace RQ2 coding corpus.
OPAQUE remains a feedback ablation, while prompt-filter and tool-regex remain weak
configured controls. Therefore the existing evidence does not support claiming an
independent strong-baseline comparison on unseen non-coding tasks.

A forensic check of both documented backups narrows, but does not remove, the raw
evidence gap. `origin/backup/2026-06-14-master` contains no OpenAgentSafety result
directories. The separately backed-up official benchmark repository at commit
`8cb4131211435a933d44942479e79418972f8f9b` retains per-task trajectories and
evaluator outputs for its published model runs. It does not retain the ActPlane
condition's per-task watcher logs, runner summaries, or matched trajectories used
to produce the paper's aggregate 78/28 count. Those official files can support a
future frozen task selection and baseline audit, but cannot reconstruct ActPlane's
per-task outcomes or turn the no-enforcement condition into an independent method.

The forensic commands are:

```bash
git ls-tree -r --name-only origin/backup/2026-06-14-master \
  docs/OpenAgentSafety
git ls-remote https://github.com/eunomia-bpf/OpenAgentSafety.git \
  refs/heads/backup/2026-06-14-actplane-submodule
# inspect the recursive Git tree for nested commit 8cb4131 via the GitHub API
```

A defensible next experiment would freeze a task-description-only policy generator
and an unseen OpenAgentSafety subset, then compare ActPlane with an official,
independently implemented runtime policy method under the same tasks, model,
services, and end-to-end outcome rubric. Positive, negative, and mixed results
would respectively support cross-domain advantage, reject it, or localize it by
OS-observable versus semantic/service-side effects. This experiment should not be
started until the official checkout, complete per-task raw outcomes, and runnable
independent baseline are present. The current workspace has the runner and policy
inventory but lacks those three frozen inputs, so inventing a substitute here
would create another weak control rather than answer Reviewers A, C, and D.
