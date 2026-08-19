---
date: 2026-08-19
title: "When Does Profiler Sampling Become Biased?"
description: "Sampling profiles look precise even when phase locking, skid, and missed short-lived code bias them. This report develops a confidence-aware profiler contract."
tags:
  - Daily Report
  - Profiling
  - Performance
  - Sampling
  - Linux perf
  - Statistics
research_question: "How can a sampling profiler detect and quantify phase locking, incomplete coverage, skid, and other systematic sampling errors instead of presenting sample percentages as if they were exact measurements?"
source_cutoff: 2026-08-19
status: daily-report
---

# When Does Profiler Sampling Become Biased?

A profiler records 1000 samples per second from a service with a repeated 1 ms control loop. The report says function A consumes 8.1% of CPU time and function B consumes 5.7%. A second run reverses the order. Raising the sampling frequency changes both percentages again.

Which result should an engineer trust?

The usual answer is to collect more samples. That works when the error is mainly random variance. It does not solve a sampler that repeatedly observes the same phase of a periodic workload, misses short-lived functions, lands after the instruction that caused an event, or is throttled in a workload-dependent way.

<!-- more -->

The central problem is that a sampling profile is an **estimator**, but profiler UIs often present it like a counter. A percentage such as 8.1% hides how samples were scheduled, which code was eligible to be observed, how many samples were lost, whether the interval synchronized with the workload, and whether independent runs agree.

This is not a new observation. Steven McCanne and Chris Torek published [A Randomized Sampling Clock for CPU Utilization Estimation and Code Profiling](https://www.usenix.org/conference/usenix-winter-1993-conference/randomized-sampling-clock-cpu-utilization-estimation-and) in 1993 specifically to reduce synchronization between periodic sampling and program behavior. A later USENIX profiling study noted that DCPI randomized intervals and that unpredictable one-shot timer periods help guard against accidental synchronization. Current Linux kernel documentation for Propeller profile collection still recommends a suitable prime-number event period rather than an arbitrary round value.

The problem is also not only phase locking. The OSDI 2026 operational paper [When Sampling Lies: Trustworthy Performance Profiling for Flat Workloads with Blink](https://www.usenix.org/conference/osdi26/presentation/devsot) reports that flat mobile workloads with thousands of short-lived routines can make sampling profilers systematically wrong because of skid, shadow effects, and incomplete function coverage. Blink uses lightweight instrumentation instead and reports 99.999% accuracy at 1% overhead in its evaluated workloads.

Those results suggest a stronger design goal than "sample faster": **a profiler should expose whether its own estimator is trustworthy for the current workload, and should have a bounded fallback when it is not.**

This report is intentionally an adjacent-systems report rather than an eBPF report. The sampling problem applies to `perf`, PMU profilers, runtime profilers, mobile profilers, GPU profilers, and other periodic or event-driven samplers. eBPF can be one implementation tool, but it is not necessary to the mechanism.

## A sample percentage has hidden assumptions

Linux `perf_event_open()` exposes both `sample_period` and `sample_freq`. With a period, an overflow occurs after a configured number of events. With frequency mode, the kernel adjusts the period to try to maintain the requested rate. Samples can include the current period, timestamp, CPU, instruction pointer, callchain, and lost-sample accounting.

That is enough to build powerful profilers, but the resulting fraction

```text
samples attributed to X / total samples
```

is only a good estimator for time or event share when the sampling process observes execution in a sufficiently representative way.

Several assumptions can fail.

### The sampling clock can synchronize with the workload

Suppose a program executes a repeating cycle:

```text
A for 300 us -> B for 300 us -> C for 400 us -> repeat
```

A profiler that samples every 1 ms at the same phase can observe almost only one region. Collecting ten times longer does not repair the estimate because the additional observations repeat the same mistake.

This is why randomized sampling is old but still important. The 1993 randomized-clock paper made synchronization a first-class sampling problem. The Linux Propeller documentation gives a modern engineering hint in the same direction by suggesting a large prime sample period, such as `500009`, for hardware-sampled profiles.

A prime period is not a proof of independence, and frequency feedback is not the same as randomized sampling. The useful lesson is that **sample timing is part of the measurement design**, not just an overhead knob.

### Hardware samples can skid

A PMU overflow is not always attributed exactly to the instruction that logically caused the event. Modern hardware and `perf` expose mechanisms such as precise-event support where available, but precision differs by event and architecture.

For broad CPU-time profiling, a small amount of skid may not change the answer. For short functions, flat profiles, or event attribution at instruction granularity, it can change rankings. Blink's OSDI 2026 evaluation treats this as one source of systematic error rather than ordinary sampling variance.

### Short-lived code can have near-zero inclusion probability

A routine that runs for 20 microseconds many times may contribute meaningful total work while rarely being active at the exact instant a sampler interrupts execution. If surrounding code is longer-lived, its apparent share can be inflated even when total sample count is high.

This is different from saying "we need more samples." If the observation process systematically shadows some functions, increasing the duration can converge to the wrong distribution.

### The profiler can change its own sampling process

Sampling has overhead. Linux exposes `perf_cpu_time_max_percent` and can throttle sampling when collection consumes too much CPU time. Frequency mode also adjusts the period to chase a target rate.

Those mechanisms are reasonable, but they mean the effective sampling process can vary over time. A report that preserves only the final symbol histogram loses evidence needed to decide whether the histogram was collected under one stable design.

## Where current work is still weak

### Randomization is treated as a configuration trick, not a measurement contract

Randomized or carefully chosen sample periods are not novel. The gap is that many profiler outputs do not make the realized sampling schedule observable enough to diagnose aliasing after the fact.

A robust profile should be able to answer:

- What distribution generated the next sampling interval?
- What interval was actually realized?
- Did the kernel or runtime throttle or retune it?
- Are sample times clustered at a stable phase relative to a repeating workload signal?
- Do independent schedules produce the same hot-code ranking?

Without those facts, a user can change `-F` or `-c` and compare screenshots, but the profiler cannot explain why the result moved.

### A high sample count is often mistaken for high confidence

Ten million samples sound convincing. They are not ten million independent observations if the workload is periodic, if many samples share one phase, or if a class of short functions is systematically missed.

Classical confidence intervals based on independent Bernoulli samples can therefore be too optimistic. A profiler needs uncertainty that reflects time structure, repeated collection epochs, and coverage evidence rather than only `sqrt(n)` intuition.

### Profiler UIs rarely distinguish variance from structural bias

Two failure modes require different responses:

1. **random variance:** independent samples disagree because the sample set is small;
2. **structural bias:** the sampling mechanism systematically sees the wrong parts of execution.

More samples help the first. They can make the second look more confidently wrong.

A useful profiler should label these separately. Rank instability across independent randomized epochs suggests insufficient evidence. Persistent disagreement with selective instrumentation or a known workload oracle suggests structural bias.

### Instrumentation is usually presented as the opposite of sampling

Sampling is low overhead and broadly deployable. Instrumentation can provide much stronger coverage but historically costs more and may perturb execution.

Blink shows that this trade-off is workload- and implementation-dependent. Its reported 1% overhead does not mean full instrumentation is always cheap. It does show that a profiler can use instrumentation as a **targeted oracle** instead of treating the choice as all-sampling or all-instrumentation.

That creates a practical research question: can a profiler spend an instrumentation budget only on regions where sampling evidence is demonstrably weak?

## Promising directions with academic and production value

### 1. A randomized sampling contract with aliasing diagnostics

**Gap.** Existing interfaces configure sample period or target frequency, but the final profile usually does not describe the realized stochastic process well enough to diagnose synchronization or retuning.

**Mechanism.** Define sampling as a first-class schedule with recorded provenance. Each collection epoch specifies a target budget and an interval distribution, for example a bounded exponential or jittered renewal process rather than one fixed period. Each sample records its intended and realized interval, trigger source, current hardware period when available, CPU, timestamp, and any throttle or loss metadata.

The analysis layer then performs an aliasing check. It can compare sample timestamps against dominant workload periods found from scheduler, request, runtime, or application markers. It can also examine the phase distribution modulo candidate periods. A suspiciously concentrated phase distribution is evidence that the sample clock is not exploring execution uniformly.

The sampler should not randomize blindly. Hardware events with strong semantic meaning may require event-count sampling rather than time sampling, and some profilers need deterministic reproducibility. The contract therefore records the schedule and lets the workload choose among fixed, frequency-controlled, randomized, or mixed modes.

**Delta.** Randomized clocks already exist in prior work. The contribution is making the schedule and its realized behavior part of the profile artifact, with an explicit aliasing diagnostic instead of an undocumented tuning trick.

**Artifact.** A `perf`-compatible prototype or standalone collector that emits a sidecar sampling manifest plus a report showing interval distribution, phase concentration, throttling, sample loss, and per-epoch profile differences.

**Evaluation.** Build periodic microbenchmarks whose phase length is known and sweep fixed periods, prime periods, frequency mode, and several randomized schedules under the same overhead budget. Add phase drift, CPU migration, and load changes. Measure per-function estimation error, top-k rank error, aliasing-detection precision/recall, and overhead.

**Academic value.** This makes the relationship between a sampler's stochastic process and profile bias directly measurable.

**Production value.** When a hot-function ranking changes, an engineer can tell whether the workload changed or the sampler synchronized with it.

**Failure condition.** If ordinary `perf` frequency mode or a well-chosen fixed period already eliminates meaningful phase bias across realistic periodic workloads, the extra randomized contract adds complexity without enough benefit.

### 2. Replicated profile epochs with uncertainty and rank stability

**Gap.** Profiler percentages normally have no uncertainty attached, and sample count alone cannot distinguish independent evidence from correlated observations.

**Mechanism.** Divide a collection budget into several independent epochs, each with a separately seeded sampling schedule. For every symbol or stack, keep the per-epoch estimate rather than immediately merging all samples. Report a central estimate plus uncertainty across epochs and a rank-stability score.

For long-running correlated workloads, treat an epoch as the resampling unit rather than pretending every individual sample is independent. A block bootstrap or another dependence-aware estimator can quantify variation without requiring a false IID assumption.

The UI should surface unresolved comparisons. If A is estimated at 8% and B at 7% but their epoch-level intervals overlap heavily, the report should say "ordering unresolved" rather than sorting them with two decimal places. If the top five remain stable across independent schedules, that is stronger evidence than one large merged histogram.

**Delta.** Statistical uncertainty is standard methodology, and this report does not claim to invent confidence intervals. The systems contribution is to carry enough collection structure through the profiler pipeline that uncertainty and rank stability are valid properties of the recorded measurement.

**Artifact.** A profile format extension and analysis tool that preserves epoch identity, effective sample periods, lost samples, per-epoch histograms, uncertainty intervals, and stable/unstable rank groups.

**Evaluation.** Use workloads with known CPU shares, repeated phase changes, and flat short-function distributions. Compare naive sample-count intervals, epoch-based intervals, and ground truth. Score interval coverage, top-k stability, false confidence rate, and time required to reach a stable optimization decision.

**Academic value.** The key question becomes "when is a profile decision statistically resolved?" rather than only "how many samples were collected?"

**Production value.** Engineers can stop collecting when a decision is stable, or avoid optimizing noise when it is not.

**Failure condition.** If epoch-level uncertainty is so wide on normal production windows that it rarely resolves useful rankings, the approach needs stronger stratification or a different estimator rather than another UI confidence badge.

### 3. Uncertainty-triggered selective instrumentation

**Gap.** Sampling remains attractive because it is cheap, while the workloads most likely to fool it can require stronger coverage.

**Mechanism.** Start with low-overhead sampling. Use the diagnostics above to identify uncertain regions: symbols with unstable rank, short routines with suspiciously low coverage, phase-sensitive groups, or event classes with large skid risk. Spend a bounded instrumentation budget only on those regions for a short validation window.

The instrumented window becomes a local oracle. It can measure entry counts, bounded timing, or exact events for the selected functions. The profiler compares sampled estimates with that oracle, learns a correction or declares the region not safely sampleable, then removes the instrumentation.

This design should be conservative about correction. A one-time ratio should not silently "fix" future profiles if workload structure changes. The output should preserve which values are sampled, instrumented, corrected, or unresolved.

**Delta.** Hybrid profilers already combine mechanisms, and Blink demonstrates that lightweight instrumentation can be practical. The new question is whether **uncertainty itself can drive where instrumentation is activated**, under a fixed overhead budget.

**Artifact.** A hybrid profiler with a budget manager, sampling diagnostics, temporary function instrumentation, and a provenance-aware report format.

**Evaluation.** Reproduce flat workloads similar to Blink's target class, plus server workloads with a few dominant hot functions. Compare sampling only, full instrumentation, and uncertainty-triggered instrumentation at equal overhead budgets. Measure coverage, attribution error, top-k correctness, instrumentation footprint, and time to diagnosis.

**Academic value.** This turns measurement confidence into an online control signal for the profiler.

**Production value.** A fleet profiler can stay cheap most of the time and pay for precision only where sampling evidence says it is needed.

**Failure condition.** If identifying uncertain regions already requires enough instrumentation to erase the overhead advantage, or if dynamic instrumentation perturbs the flat workload more than it helps, the hybrid design is not worthwhile.

## A benchmark needs adversarial sampling workloads, not only ordinary applications

A profiler can look correct on workloads with one dominant function and still fail badly on periodic or flat code. Evaluation should therefore include workloads designed to stress the estimator itself.

A useful benchmark matrix would contain:

| Workload | What it tests |
| --- | --- |
| fixed periodic phases | phase locking and aliasing |
| slowly drifting phase | whether randomization remains representative |
| thousands of short equal-weight routines | incomplete coverage and rank error |
| one dominant hot function | simple case where sampling should win |
| bursty request phases | time correlation and epoch design |
| PMU event with configurable precision | skid sensitivity |
| sampler under CPU pressure | throttling and effective-rate changes |
| selectively instrumented ground truth | estimator error and interval coverage |

The benchmark should score more than overhead and visual similarity. At minimum it should measure:

- per-symbol relative error;
- top-k precision and recall;
- pairwise rank reversals;
- fraction of executed functions ever observed;
- uncertainty-interval coverage;
- false-confidence rate, where the profile declares a stable ranking that ground truth disproves;
- effective sample rate and loss;
- CPU and memory overhead.

The most important metric may be **decision error**. If the profiler causes an engineer or optimizer to choose the wrong function to optimize, a tiny aggregate histogram error is not reassuring.

## The first prototype should be deliberately boring

This problem does not initially require a new kernel subsystem.

A useful first prototype can run in userspace around existing `perf_event_open()` interfaces:

1. collect several short independent epochs;
2. vary or jitter supported sampling periods while preserving the same total overhead budget;
3. record timestamps, periods, loss, and throttle evidence already available from perf;
4. compute phase concentration and rank stability;
5. selectively instrument a small set of ambiguous functions in a controlled benchmark;
6. compare both paths against known ground truth.

Only after that experiment shows a missing primitive should the work ask for kernel support, such as a stronger randomized-overflow mode or better sample-schedule provenance.

This order matters. McCanne and Torek already showed that the clock matters more than thirty years ago. A 2026 systems contribution should not be "randomize the timer again." It should show that **modern profilers can detect when their measurement is biased, quantify unresolved uncertainty, and escalate precision without violating a fixed overhead budget.**

## What would change this conclusion?

Three results would weaken the case substantially.

First, if current `perf` frequency control and common event-period practices produce unbiased, stable rankings across periodic, flat, short-function, and phase-changing workloads at low overhead, there is little reason to add a richer sampling contract.

Second, if Blink-style instrumentation or another low-overhead instrumentation method is consistently cheap enough and more accurate across the same workload classes, the right answer may be to instrument rather than make sampling more statistically sophisticated.

Third, if epoch-level uncertainty and aliasing diagnostics do not predict real ground-truth errors, then they are only plausible-looking statistics. The benchmark must show that a warning corresponds to a higher probability of a wrong optimization decision.

Until those tests are run, sample percentages should be treated as estimates with a measurement design behind them, not as exact CPU accounting. The interesting systems problem is no longer how to collect more samples. It is how to make a profiler explain **when its own samples are enough to trust**.
