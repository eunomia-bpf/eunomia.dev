---
date: 2026-09-11
title: "When Is an eBPF Optimization Result Strong Enough to Ship?"
description: "eBPF optimizations can win one microbenchmark and regress real applications. This report develops evidence envelopes, holdout tests, and promotion gates."
tags:
  - Daily Report
  - eBPF
  - Optimization
  - JIT
  - Benchmarking
  - Verification
research_question: "How can eBPF optimization evidence show that a speedup survives verifier, JIT, kernel, architecture, application, and workload variation instead of reflecting one tuned benchmark?"
source_cutoff: 2026-09-11
status: daily-report
---

# When Is an eBPF Optimization Result Strong Enough to Ship?

Suppose an eBPF optimization makes a tight `BPF_PROG_TEST_RUN` loop 18% faster. The verifier still accepts the program, the JIT emits fewer instructions, and repeated runs on the same machine are stable. Is that enough evidence to enable the optimization for Cilium, Katran, Tracee, or a fleet of mixed x86 and ARM machines?

Not really. The optimization can be completely correct and still be a poor production choice. It may speed up a hot arithmetic loop but add code size that hurts the instruction cache in a larger program. It may help one JIT backend and do nothing on another. It may optimize a program that accounts for almost none of the application's workload time. A profile-guided pass can look excellent on the workload used to tune it and lose when branch bias changes. An automated optimizer can even discover ways to improve the benchmark rather than the intended application behavior.

This creates a different problem from proving semantic equivalence. **For an eBPF optimization, correctness tells us whether the transformed program is allowed to replace the original. It does not tell us how broad the performance claim is, where the optimization is profitable, or whether the evidence is strong enough to promote it into production.**

<!-- more -->

This report closes the current optimization series after [runtime-profile specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/), [portable architecture specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/), [execution provenance](https://eunomia.dev/research/ebpf-specialization-debug-provenance/), [native-operation trust](https://eunomia.dev/research/ebpf-native-operation-trust-boundary/), and [cross-backend operation semantics](https://eunomia.dev/research/ebpf-cross-backend-operation-semantics/).

Those reports establish five different questions: is the transformation equivalent, are runtime assumptions still valid, can the implementation run on this architecture, which specialization actually executed, and does delegated stateful behavior preserve one semantic contract? Here we assume those gates pass. The remaining question is whether the *performance evidence* justifies saying that an optimization is useful outside the exact experiment that discovered it.

## The same optimization can have several different performance truths

Kops is a useful example because it reports both microbenchmark and application results. Its EInsn operations replace verifier-visible BPF instruction sequences with native machine idioms. The paper reports microbenchmark improvements up to 24%, while production applications improve by up to 12% on x86-64 and ARM64. Those numbers are not contradictory. They answer different questions.

A microbenchmark can isolate whether a rotate, conditional select, extract, or similar sequence becomes cheaper. A production application additionally asks how often that sequence executes, what surrounds it, whether code layout changes, whether helper/map costs dominate, and whether the application-level workload is sensitive to the saved cycles.

Linux BPF infrastructure already reflects a similar distinction. BPF CI uses `veristat` on complex programs to compare verifier behavior and catch verifier-performance regressions. `veristat` can compare properties such as processed instructions and verifier statistics between revisions. That is valuable, but a verifier regression is not the same thing as runtime throughput or tail latency. One optimization therefore needs several oracles rather than one universal score.

The current [`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark) framework makes this explicit for optimization research. Its current corpus contains six production eBPF applications, 146 comparable program measurements, and 42 microbenchmark tasks. It records verifier outcomes, JIT code size, application lifecycle state, workload correctness, raw workload metrics, and per-program kernel run counters. The framework also notes a practical fact that is easy to miss: static rewrite counts do not reliably predict speedup, and a pass that improves one program can regress another.

That is the evidence problem in one sentence. An optimization result is not a scalar. It is a conditional claim over a program, runtime, architecture, kernel, workload, and measurement procedure.

## What current evaluation still leaves underspecified

The first gap is **the unit of a performance claim**. “Up to 20% faster” can mean one instruction sequence, one BPF program, one application phase, or one end-to-end workload. Without an explicit claim boundary, a valid local result can be read as a much broader result.

The second gap is **profitability coverage**. Correctness can often be binary: the transformed program is equivalent or it is not. Profitability is rarely binary. A pass may help 30 programs, be neutral on 100, and regress 16. A single geomean can hide whether those regressions occur on rare utilities or on the programs consuming most fleet CPU time.

The third gap is **environment generalization**. eBPF sits in a compilation pipeline with architecture-specific JITs, kernel versions, feature backports, helper implementations, map behavior, CPU microarchitecture, and application loaders. An optimization that is semantically portable can still have sharply different performance on another JIT or CPU. The September 6 report addressed safe architectural eligibility and fallback. It did not define how much evidence is required before claiming that the *performance benefit* generalizes.

The fourth gap is **adaptive-search bias**. An optimizer, especially an automated or agentic optimizer, may try many transformations and repeatedly observe the same benchmark. Eventually it can overfit to measurement noise, one workload phase, a warm cache state, or even a loophole in the harness. Traditional compiler evaluation worries about benchmark selection; closed-loop search adds the risk that the benchmark itself becomes the optimization target.

The fifth gap is **production promotion**. Papers normally stop after evaluation. Production systems need another decision: on which machines and programs should this optimization be enabled tomorrow, what evidence should travel with that decision, and what observation should revoke it?

## Promising directions with academic and production value

### 1. Make every optimization claim carry an evidence envelope

Instead of publishing or storing only a speedup, treat the result as a structured artifact. An evidence envelope would bind the optimization identity to the conditions under which its benefit was measured.

A minimal record might include:

```text
optimization = rotate_v3
semantic_witness = proof-sequence-hash
kernel = 6.x + commit
jit = x86_64 / commit
cpu = family-model-stepping
program = object + program tag
application = katran + revision
workload = replay-manifest + digest
baseline = exact artifact
samples = raw run counters + workload metrics
correctness = pass/fail + oracle identity
performance = distribution + confidence interval
regression_budget = <= 1% on guarded metrics
claim_scope = {this program, this workload family, this target class}
```

The key difference from ordinary benchmark metadata is the final field. The artifact states what the result is allowed to justify. A microbenchmark can support “this instruction idiom is cheaper on this target.” It cannot automatically support “the application is faster.” A production replay can support an application-level claim, but only for the workload and target family represented by the replay.

The research artifact would be a schema plus a comparison engine that can combine envelopes only when their claim scopes are compatible. Evaluation should take published-style optimization summaries and ask whether the envelope prevents over-broad conclusions while remaining compact enough to generate automatically. A useful metric is *claim escape*: how often a result accepted under a weaker metadata scheme fails when rerun outside its hidden assumptions.

The idea is not worthwhile if ordinary benchmark manifests already capture every condition that materially changes the result and reviewers or deployment systems consistently enforce that scope. The experiment should try to falsify the need for another artifact.

### 2. Use holdout and counterexample suites for optimizer evaluation

Optimization evaluation usually asks whether a pass wins on selected programs. An adaptive optimizer needs a stronger test: can it improve the search set without learning how to game it?

Split evidence into at least three groups. A development set is visible to the optimizer. A holdout set contains unseen applications, workload phases, kernels, or architectures. A counterexample set contains cases designed to punish common shortcuts: programs where code-size growth should hurt, workloads where the optimized sequence is cold, branch distributions that invert the training profile, verifier-sensitive programs, and applications whose real loader or lifecycle must remain intact.

The [`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark) integrity model already points in this direction by treating the optimizer as untrusted and forbidding tricks such as shortening workloads, filtering failed programs, bypassing real loaders, or fabricating result files. A research system can make that principle quantitative.

The primary score should not be “best speedup found.” Report at least acceptance coverage, worst guarded regression, holdout speedup distribution, invalid-result rate, and the fraction of development wins that reproduce on holdout targets. For agentic search, freeze the holdout oracle until final evaluation so the search loop cannot adapt to it.

An ablation can intentionally leak holdout results back to the optimizer. If the final score rises while independent reproduction gets worse, the benchmark has demonstrated the exact overfitting problem it is meant to measure.

This would produce a reusable benchmark methodology rather than another optimization pass. The production analogue is equally useful: canary workloads and machines become holdouts for a newly discovered optimization policy.

### 3. Promote optimizations through a profitability contract, not a global enable bit

Many optimizations do not need to be universally good. They need a reliable applicability rule.

A promotion system could learn or derive a small profitability predicate from the evidence envelope: target architecture, JIT capabilities, program features, code-size delta, runtime profile stability, and application/workload class. The optimization is enabled only when the predicate matches. The runtime then records whether the expected benefit appears and disables or rolls back the specialization when a guarded regression budget is exceeded.

This is deliberately different from the September 5 deoptimization mechanism. That report asks whether a stale runtime assumption can change program semantics. Here semantics remain valid; the predicate controls *economic usefulness*. The fallback may be triggered because p99 latency regressed 3%, because JIT size crossed an instruction-cache budget, or because the optimized program accounts for too little runtime to justify added complexity.

The evaluation should compare three policies: globally enable the optimization, statically enable it on a hand-written target allowlist, and evidence-driven promotion. Test across kernel versions, x86-64 and ARM64, several CPU generations, microbenchmarks, and real application workloads. Measure realized fleet-weighted speedup, worst regression, fraction of eligible executions, decision overhead, rollback frequency, and how quickly the policy adapts after a workload or kernel change.

The system loses if a simple static allowlist achieves the same realized benefit and regression bound. That failure condition matters: not every optimization needs an online controller.

## A practical evaluation contract

These directions suggest a simple hierarchy for future eBPF optimization claims.

First, prove or otherwise validate **semantic admissibility**. That is the work covered by verifier safety, equivalence, trust, and stateful operation contracts.

Second, measure **local mechanism gain** with microbenchmarks. This tells us whether the intended machine-level effect exists.

Third, measure **program gain** using kernel run counters, JIT size, and repeated execution of the actual BPF program. This catches interactions with surrounding bytecode.

Fourth, measure **application gain** through the real loader and workload. This determines whether the optimized BPF program matters to the system using it.

Fifth, test **generalization** on holdout kernels, architectures, and workload phases. This defines how broad the claim can be.

Finally, measure **promotion safety**: how many regressions escape the applicability rule, how quickly they are detected, and whether rollback restores the baseline.

A paper does not need every layer for every experiment. It should, however, stop its claim at the highest layer actually tested. A production rollout should be even stricter because the workload distribution, not the best benchmark, determines the realized value.

## What would change this conclusion?

The proposed evidence machinery would be unnecessary if local eBPF benchmark wins already predicted production benefit with high reliability. A large corpus could test that directly: correlate isolated instruction/program speedups with application throughput, latency, and BPF CPU time across architectures and workloads. If the relationship is strong and stable, elaborate evidence envelopes and holdout gates would add process without much information.

The argument would also weaken if existing BPF CI already provided one end-to-end optimization contract combining verifier behavior, exact JIT output, real application lifecycle, workload correctness, runtime performance, architecture diversity, and regression promotion. Today the pieces exist, but they are normally separate: `veristat` is strong verifier evidence, application benchmark suites provide workload evidence, and individual optimization papers define their own performance matrices.

Finally, a profitability contract is not justified if the optimization has negligible downside and an overwhelming benefit on every supported target. The right answer for a universally profitable transformation is still to make the compiler better, not to build a control plane around it.

But for optimizations that are architecture-sensitive, profile-sensitive, or discovered by adaptive search, “the benchmark got faster” is only the beginning of the evidence. The result is strong enough to ship when the system can state **where the claim applies, which counterexamples were tested, what regressions are bounded, and what observation will revoke the optimization when reality leaves the measured envelope.**

## References

- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Linux BPF CI, [kernel-patches/bpf](https://github.com/kernel-patches/bpf), accessed 2026-09-11.
- libbpf, [`veristat`](https://github.com/libbpf/veristat), accessed 2026-09-11.
- Eunomia, [`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark), accessed 2026-09-11.
- sched_ext, [Developer Guide](https://github.com/sched-ext/scx/blob/main/DEVELOPER_GUIDE.md), accessed 2026-09-11.
