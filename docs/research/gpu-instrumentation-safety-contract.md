---
date: 2026-08-30
title: "Can GPU Instrumentation Change a Kernel Without Changing Its Meaning?"
description: "Dynamic GPU binary instrumentation can observe code after SASS lowering, but probes can perturb registers, occupancy, and control flow. This report defines a safer contract."
tags:
  - Daily Report
  - GPU
  - Instrumentation
  - Profiling
  - Runtime
research_question: "What contract should dynamic GPU instrumentation expose so inserted device-side probes remain semantically safe, bounded in resource perturbation, and explicit about unsupported sites?"
source_cutoff: 2026-08-30
status: daily-report
---

# Can GPU Instrumentation Change a Kernel Without Changing Its Meaning?

A production GPU kernel is slow only on one customer workload. Source-level logging cannot reproduce it, and recompiling the application would change the binary that failed. Dynamic binary instrumentation looks ideal: patch the already compiled GPU code, run a small probe near the suspicious instruction, and collect the missing state.

But the probe now executes inside the same GPU program it is trying to observe. It needs registers. It adds instructions. It may touch memory, change control flow, or execute under the same SIMT and synchronization rules as the application. On a kernel already close to an occupancy or resource boundary, even a semantically correct probe can change scheduling enough to hide the original performance problem. A less careful probe can change the program itself.

GPU instrumentation therefore has two jobs. It must collect useful evidence, and it must make a defensible statement about how much of the original execution it preserved. Today's tools provide strong mechanisms for tracing, sampling, and binary rewriting, but that preservation statement is still fragmented across tool-specific assumptions.

<!-- more -->

This report argues for an explicit **GPU instrumentation safety contract** between an instrumentation runtime and the tool loaded into it. The contract should describe which architectural state a probe may read or modify, which dynamic sites are actually supported, how much execution-resource perturbation the runtime permits, and what evidence is retained when the runtime skips or throttles a probe. The point is not to make instrumentation free. It is to prevent "the tool ran" from being confused with "the tool observed the original program faithfully."

The question is distinct from the site's earlier reports on [GPU launch-latency attribution](https://eunomia.dev/research/gpu-kernel-launch-latency/) and [host/device causality](https://eunomia.dev/research/gpu-host-device-causality/). Those reports ask whether a trace can explain when and why GPU work ran. Here the problem begins one level lower: whether inserting the observer changes the execution whose behavior we want to explain.

## Dynamic GPU instrumentation is already powerful enough to need a contract

[NVBit](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus) established that GPU binary instrumentation can work after CUDA source and PTX have already been lowered to NVIDIA SASS. Its runtime can inspect precompiled kernels and libraries and inject device-function calls before or after selected SASS instructions. The current [NVBit repository](https://github.com/NVlabs/NVBit) still exposes that model: tools can inspect and modify SASS, inject arbitrary device functions, and even remove instructions. The repository explicitly warns that removing instructions does not guarantee the application will continue to work correctly, and notes that function injection carries save-and-restore cost around application state.

Intel's instrumentation stack reaches a similar layer from a different ecosystem. The 2026 ISPASS artifact for [GTPin](https://zenodo.org/records/18911052) describes high-level binary instrumentation for Intel GPUs and publishes the code used to reproduce the paper's results. The existence of mature tooling on both NVIDIA and Intel hardware makes the core question broader than one ISA or vendor: once a runtime can rewrite executed GPU code, what exactly does it promise to preserve?

There is also a useful contrast with vendor profiling interfaces. Current [CUPTI documentation](https://docs.nvidia.com/cupti/index.html) exposes activity tracing, callbacks, PC sampling, SASS metrics, performance-monitor sampling, checkpointing, and profiling APIs. PC sampling periodically records a selected active warp's program counter and scheduler state; SASS metrics use instruction-level support and SASS patching for selected metrics. These interfaces provide valuable evidence without giving an arbitrary tool the same mutation freedom as a general binary rewriter.

That difference matters. A programmable instrumentation runtime can answer questions a fixed profiler did not anticipate, but it also moves more responsibility from the vendor API into the instrumentation runtime and tool.

## Saving registers is necessary, but transparency is larger than register state

A common intuition is that instrumentation is transparent if the runtime saves the application's registers, calls the probe, and restores those registers afterward. NVBit's implementation work makes this practical, and register preservation is a necessary part of any sane design. It is not the whole property.

GPU execution depends on finite per-SM resources. NVIDIA's current [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) documents that register use is one factor that limits how many thread blocks and warps can be resident on an SM. A small increase in per-thread or per-block resource demand can cross an allocation boundary and reduce occupancy. The CUDA programming model also makes shared memory, resident blocks, and resident warps part of that scheduling envelope.

An injected probe can therefore preserve every application register value and still perturb performance through several paths:

- extra registers or local-memory spills change the resource footprint;
- extra loads, stores, atomics, or cache traffic compete with the application;
- probe branches execute under warp-level control flow and can add divergence;
- probe-side synchronization can interact badly with partial-warp or conditional execution;
- instruction expansion changes issue pressure and timing around races or contention;
- buffering and flushing telemetry can move pressure from the probe site to a later phase.

For correctness tools, there is an additional boundary: some sites are easier to reason about than others. A post-dominator-free branch target, an indirect control-flow transfer, a barrier region, or code whose register/stack behavior cannot be recovered should not silently receive the same "instrumented" label as a simple arithmetic instruction.

The recent [WarpGuard](https://arxiv.org/abs/2606.11871) work on CUDA SASS control-flow integrity is useful here because it makes the denominator explicit. Across 77 CUDA artifacts it classifies tens of thousands of SASS control-flow sites and separates protected sites from unsupported, profile-excluded, fallback, and no-surface outcomes. Its main goal is CFI rather than general profiling, but the reporting discipline is important: unsupported instrumentation is a first-class result rather than a hidden hole in coverage.

## The missing property is observable non-interference, not zero overhead

Demanding zero overhead would make most useful dynamic instrumentation impossible. The more practical requirement is **observable non-interference**: the runtime states which program properties it intends to preserve, measures the resource changes it can observe, and refuses to turn unknown or unsupported cases into silent success.

For a performance probe, the contract might allow timing perturbation up to a declared budget but forbid application-visible memory writes outside a private telemetry region. For a security monitor, it might allow a larger execution cost but require fail-closed checks before protected control transfers. For a debugging probe, changing selected application registers may be intentional, but that mutation must be declared rather than hidden behind the word "instrumentation."

This makes two dimensions explicit:

1. **semantic effect:** what application-visible state and control behavior a tool may change;
2. **measurement effect:** how much the observer changes scheduling, resource occupancy, memory traffic, and timing.

A runtime can be strong on one dimension and weak on the other. PC sampling, for example, has a narrow semantic footprint but limited observability compared with arbitrary injected code. A binary rewriting tool can observe much more, but it needs a stronger contract and better accounting.

The site's [bpftime GPU work](https://eunomia.dev/bpftime/documents/gpu/) is one example of pushing programmable runtime logic toward device execution. An eBPF-like verifier can help constrain memory and control-flow behavior, but the GPU-specific problem remains even with safe bytecode: verifier safety alone does not tell us whether probe resource use changed occupancy, whether an attach site was actually instrumentable, or whether sampling/throttling left an important coverage hole.

## Where current work is still weak

### Tool safety and program transparency are usually different interfaces

Binary instrumentation frameworks necessarily manage ABI state, code relocation, injected calls, and tool-private storage. Profilers separately expose overhead controls and supported metric sets. What is missing is one machine-readable contract that connects these facts to the question an operator actually asks: "Did this probe preserve the behavior I am diagnosing?"

A useful test would instrument kernels close to register, shared-memory, and occupancy boundaries, then compare both functional outputs and hardware resource state before and after instrumentation. The runtime should be able to distinguish a semantically preserved but performance-perturbed run from a run that stayed inside both budgets.

### Unsupported sites and skipped observations are not consistently part of the result

A dynamic tool can fail to decode a binary pattern, refuse a site, run out of telemetry space, sample only part of execution, or throttle itself under load. If the final report only shows observations that succeeded, a user cannot distinguish "nothing happened" from "the observer had no evidence."

WarpGuard's explicit site categories show one possible reporting model. General instrumentation needs the same idea for profiling and debugging: supported sites, rejected sites, runtime-skipped instances, sampled instances, and lost records should all contribute to a coverage statement.

The discriminating experiment is a workload containing deliberately unsupported or high-pressure sites. A trustworthy tool should lower confidence or expose missing coverage instead of producing the same confident summary it would produce after complete observation.

### Resource perturbation is measured after the fact, not enforced as an attach-time budget

GPU tools often report total slowdown, which is useful but coarse. A 2% whole-program slowdown can hide a large local perturbation in a kernel that runs infrequently, while a 20% slowdown in a microbenchmark can be harmless if the probe is only enabled for one request.

What is missing is a budget that can be checked near attachment: additional registers, private/local memory, shared memory, instrumentation instructions, telemetry bytes, and expected probe frequency. The budget does not need to predict exact runtime slowdown, but it can reject clearly dangerous combinations before they invalidate the measurement.

A material test is to compare a fixed global-overhead policy with a resource-aware per-kernel budget. If both preserve ranking, diagnosis, and application behavior equally well, the more complicated budget is unnecessary.

### Cross-vendor instrumentation lacks a common preservation vocabulary

NVBit operates on NVIDIA SASS; GTPin instruments Intel GPU binaries; vendor profiler APIs expose different sampling and metric capabilities. A portable tool cannot assume that "instruction probe" or "PC sample" means the same coverage or perturbation on each backend.

A useful portable layer should therefore standardize preservation claims and uncertainty, not pretend to standardize the underlying ISA. Backends can report their native capabilities while sharing concepts such as state clobber set, attach-site class, coverage, resource delta, and telemetry-loss state.

## Promising directions with academic and production value

### 1. A verified probe-effect manifest

**Gap.** A tool can describe where it wants to attach, but the runtime does not have a portable object describing what the probe is allowed to change and what the backend must preserve.

**Mechanism.** Attach every device-side probe with a compact effect manifest. It declares readable architectural state, writable state, permitted memory regions, possible control transfers, synchronization use, maximum telemetry writes, and whether application-state mutation is intentional. The backend combines this with an attach-site capability record derived from SASS, another device ISA, or an intermediate representation.

The loader then produces one of several explicit outcomes: verified attach, attach with declared degradation, unsupported site, or rejected effect. The runtime must not silently weaken the manifest to make a tool load.

For an eBPF-like backend, a bytecode verifier can prove part of the effect manifest. For native CUDA/C++ instrumentation callbacks, static validation may be weaker and the runtime may need a conservative capability class. The abstraction survives either implementation because the public object is the promised effect, not the verifier technology.

**Delta from related work.** NVBit already handles ABI-compliant instrumentation and state access; WarpGuard derives policies for supported SASS control-flow sites. The proposed layer generalizes those ideas into an attach-time contract for arbitrary observability and debugging probes rather than another binary-rewriting engine.

**Artifact.** A vendor-neutral manifest schema plus NVBit and GTPin adapters, with an optional eBPF-like verified probe backend. A small CLI could answer `why-not-attached <kernel,site>` and show the exact violated constraint.

**Evaluation.** Instrument CUDA and Intel GPU workloads containing arithmetic, memory, indirect control flow, barriers, divergent branches, and library kernels. Compare raw framework behavior with manifest-gated instrumentation. Measure unsupported-site recall, false acceptance, functional divergence, attach latency, and the fraction of useful probes admitted.

**Academic value.** The general question is whether dynamic GPU instrumentation can expose a portable effect system despite vendor-specific binary semantics.

**Production value.** Profilers and security tools gain a machine-checkable boundary for third-party probes instead of relying only on tool review and crash testing.

**Failure condition.** If practical probes require effects too dynamic for the manifest to validate without rejecting most useful sites, a static attach contract is the wrong abstraction.

### 2. Resource-budgeted instrumentation with explicit coverage

**Gap.** A semantically safe probe can still destroy the performance behavior being measured by crossing a register, occupancy, bandwidth, or telemetry threshold.

**Mechanism.** Before activation, the backend estimates the probe's incremental resource footprint for each kernel variant. The runtime admits instrumentation under a per-kernel budget such as:

```text
max_register_delta
max_local_memory_delta
max_shared_memory_delta
max_injected_instructions_per_event
max_telemetry_bytes_per_second
max_sampled_event_fraction
```

When a full probe exceeds budget, the runtime can rotate sites, sample invocations, or fall back to a lower-cost observation mode such as PC sampling. Every reduction must update a coverage record. The result says both what was observed and what the runtime intentionally skipped.

The important design choice is to make coverage part of the query result. An operator should be able to see that a hot instruction accounts for 40% of collected samples while only 12% of eligible events were observed, rather than treating that percentage as if coverage were complete.

**Delta from related work.** CUPTI already offers lower-overhead sampling modes and NVBit enables selective instrumentation. The proposed mechanism makes resource admission and observation coverage a shared runtime policy instead of leaving each tool to invent its own sampling switch.

**Artifact.** A resource estimator, admission controller, and coverage-carrying telemetry format. A prototype can use CUPTI counters or occupancy calculations for validation while driving NVBit-style binary instrumentation.

**Evaluation.** Use kernels placed just below and just above known register/occupancy boundaries, plus memory-bound, compute-bound, divergent, and synchronization-heavy workloads. Compare always-on full instrumentation, fixed-rate sampling, PC sampling, and resource-budgeted instrumentation. Measure diagnosis accuracy, occupancy changes, register spills, kernel-duration distortion, telemetry loss, and total overhead.

**Academic value.** This asks whether observer perturbation can be treated as a schedulable resource with explicit accuracy trade-offs.

**Production value.** An always-on GPU observability service can keep a hard operational budget while still escalating to richer probes when a specific kernel needs diagnosis.

**Failure condition.** If simple fixed-rate sampling gives the same diagnosis quality and perturbation bounds across workloads, per-kernel resource admission adds complexity without enough value.

### 3. A counterexample benchmark for transparent GPU instrumentation

**Gap.** Tool evaluations usually measure average slowdown and collected information, but they rarely construct cases where instrumentation changes the answer to the debugging or performance question.

**Mechanism.** Build paired kernels whose uninstrumented behavior is known but whose instrumentation sensitivity differs:

| Pair | Same apparent target | Hidden boundary |
| --- | --- | --- |
| equal kernel time | same hot instruction | one kernel sits one register-allocation step below an occupancy drop |
| same branch shape | same attach site | one executes the site under partial-warp divergence |
| same memory trace | same load/store probe | one is near a cache or atomic-contention threshold |
| same barrier count | same synchronization region | one reaches the site with different active-lane structure |
| same event rate | same telemetry record | one saturates the probe buffer and silently loses observations |

The harness runs native, sampled, and instrumented variants and records a ground-truth answer to a concrete question such as hotspot ranking, stall cause, race manifestation, or security violation. A tool fails when its own instrumentation changes that answer while still reporting high confidence.

**Delta from related work.** NVBit, GTPin, CUPTI, and newer SASS security instrumentation demonstrate useful mechanisms. This benchmark evaluates the observer itself as a possible source of counterexamples rather than only evaluating whether the observer collected data.

**Artifact.** An open CUDA-first benchmark suite with binary variants, expected outputs, occupancy/resource boundaries, trace-loss injection, and a backend interface for Intel or future GPU runtimes.

**Evaluation.** Primary metrics are semantic divergence, diagnosis-rank stability, resource delta, coverage, false-confidence rate, and overhead. The most important ablation removes the safety contract while keeping the same probe code, which isolates the value of admission and coverage reporting from the probe's analysis logic.

**Academic value.** The benchmark turns instrumentation transparency from an informal expectation into a falsifiable systems property.

**Production value.** Tool vendors and runtime teams can regression-test new GPU architectures and driver releases for observer-induced failures before enabling deep instrumentation in production.

**Failure condition.** If real frameworks already preserve the benchmark's answers whenever total overhead is below a simple threshold, then the benchmark would show that a smaller slowdown-only contract is sufficient.

## What would change this conclusion?

The argument assumes that device-side programmable instrumentation will continue to be useful precisely because fixed vendor telemetry cannot anticipate every debugging, security, and research question. If future hardware exposes rich, low-overhead, non-mutating observation for nearly all questions operators care about, general binary instrumentation could become a niche offline tool and a broad runtime contract would matter less.

The proposed contract is also unnecessary if resource perturbation turns out to be well predicted by one scalar such as whole-kernel slowdown. The strongest counterevidence would be a cross-architecture study showing that, below a simple overhead threshold, dynamic probes preserve hotspot ranking, synchronization behavior, race manifestation, and security outcomes even near register and occupancy boundaries.

Finally, portability may be the wrong goal. If NVIDIA SASS, Intel GPU binaries, and future accelerators expose incompatible notions of attach sites and preserved state, a shared schema could erase important backend facts. In that case the right design would be vendor-specific contracts with only a thin common envelope for coverage and uncertainty.

Until those results exist, programmable GPU instrumentation should be treated like any other code inserted into a live system: useful because it can observe what fixed interfaces miss, but trustworthy only when its effects and blind spots are part of the result.