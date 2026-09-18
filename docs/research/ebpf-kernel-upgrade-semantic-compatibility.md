---
date: 2026-09-18
slug: ebpf-kernel-upgrade-semantic-compatibility
title: "Can an eBPF Object Keep Its Meaning After a Kernel Upgrade?"
description: "CO-RE can repair type offsets, but kernel upgrades can still change verifier admission, kfunc contracts, tracepoints, and feature-probe behavior."
tags:
  - Daily Report
  - eBPF
  - Linux
  - Compatibility
  - BTF
  - CO-RE
research_question: "How should an eBPF deployment prove that a previously accepted object still has the intended observable behavior after a kernel upgrade, rather than only proving that relocation and loading succeed?"
source_cutoff: 2026-09-18
status: daily-report
---

# Can an eBPF Object Keep Its Meaning After a Kernel Upgrade?

Suppose an eBPF object loads successfully on kernel A. The host upgrades to kernel B. The same object still passes CO-RE relocation and the verifier, and all links attach. Is that enough to say the application is compatible?

Not necessarily.

Linux gives BPF a deliberately stable core ABI, but an actual application often depends on interfaces outside that core: tracepoint payloads, attachable kernel functions, BTF-described kernel internals, kfunc contracts, verifier rules, and loader-side feature probes. CO-RE can adapt a field offset or test whether a type exists. It cannot prove that the field still means what the application assumes, that a probe still classifies a verifier result correctly, or that an unstable kernel-facing interface preserves the same behavioral contract.

A recent failure makes the distinction concrete. After Linux 7.2 changed verifier behavior around a `bpf_set_retval` probe, Cilium 1.20 could fail during startup because its feature detection saw a different verifier error and classified the helper probe incorrectly. The BPF capability had not simply become a different version number. The *observation used to infer capability* had changed.

This report asks the next question after [capability-based admission](https://eunomia.dev/research/ebpf-kernel-capability-evidence/): once an object is admitted on both sides of an upgrade, how can a deployment establish that the object still performs the same intended job?

<!-- more -->

## eBPF kernel upgrade compatibility is wider than CO-RE relocation

CO-RE solves a specific and valuable problem. The BPF object carries BTF and CO-RE relocation records, and the loader patches instruction offsets or immediates using the target kernel's BTF. Current Linux documentation groups relocations around field properties, type properties, and enum values. This lets one object tolerate many structural changes that would otherwise require rebuilding against each kernel.

That is structural compatibility, not a general semantic equivalence proof.

Consider a program that reads a kernel field after CO-RE finds it successfully. Three different claims are easy to conflate:

1. **The field exists and the instruction can address it.** CO-RE can often establish this.
2. **The verifier accepts the resulting program.** The target kernel establishes this at load time.
3. **The value still represents the application-level fact the program thinks it represents.** Neither relocation nor admission proves this in general.

The third claim matters for monitoring and policy. A tracing program may continue to read a valid integer while the surrounding kernel lifecycle changes when that integer is updated. A hook may still exist while no longer covering the full operation the application intended to observe. A kfunc can keep a recognizable name while its allowed calling context, ownership rules, or argument contract evolves.

The important boundary is therefore not "CO-RE works" versus "CO-RE fails." It is **structural adaptation versus behavioral compatibility**.

This also differs from [architecture-specific eBPF specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/). Architecture specialization asks whether different JIT or native backends preserve one BPF meaning. Kernel-upgrade compatibility asks whether the kernel-facing environment around the same object still satisfies the assumptions that gave the object its meaning in the first place.

## Linux explicitly keeps some BPF-facing surfaces unstable

The Linux BPF design documentation draws a useful line. BPF instructions, program arguments, helpers and helper arguments, and recognized return codes are part of the stable ABI. But it also names exceptions and non-ABIs that many production BPF applications use.

Tracepoints are not a stable ABI. Kprobe attachment locations are not a stable ABI. Kernel internals walked by tracing programs can change. The documentation recommends CO-RE to make attachment and kernel-data adaptation easier, but that does not turn internal kernel functions into a stable public contract.

Kfuncs are even more explicit. Current kernel documentation says that, unlike helpers, kfuncs do not have a stable interface and may change from one kernel release to another. Visibility can be program-type specific, pointer validity rules can evolve, and BTF is part of how the verifier reasons about those calls.

This means a realistic compatibility statement should be decomposed by dependency:

```text
stable BPF ABI dependency
    -> expect compatibility unless a kernel regression exists

CO-RE/BTF structural dependency
    -> relocate and validate target structure/type evidence

unstable tracepoint/kprobe/kfunc dependency
    -> require version-specific or behavior-specific evidence

loader/probe dependency
    -> validate that feature inference still means what the loader thinks it means
```

A single "loaded successfully" bit collapses all four into one answer.

## Feature probes can drift even when they are trying to avoid version checks

Active probing is usually better than assuming that `uname -r` implies a capability. But a feature probe is still a program, and its interpretation can itself become a compatibility surface.

The Linux 7.2 Cilium incident is a useful example because the failure happened in the probe rather than in the datapath program. Cilium reported a fatal startup error while detecting `FnSetRetval` for `CGroupSock`. The probe load returned a verifier message that `R1 is not a scalar`. The detection logic expected a different failure shape to distinguish "helper exists" from "helper does not exist," so a verifier-side change broke the inference.

That does not make active probing a bad strategy. It makes the contract more precise:

> A probe result is only evidence when the loader's interpretation of that result is also valid for the target kernel.

A robust probe should therefore prefer machine-readable outcomes where possible, minimize dependence on verifier log wording, and carry a probe implementation identity in compatibility evidence. When the probe deliberately relies on verifier rejection to establish that a helper or program type exists, CI should run that exact probe across the supported kernel matrix.

The same principle applies to BTF tooling. In August 2026, cilium/ebpf fixed a name-indexing bug that prevented lookup of kernel BTF functions whose names start with triple underscores, such as `___pskb_trim`. The kernel function and BTF information could be present while a userspace normalization rule made an fentry/fexit target appear unavailable. Compatibility is a property of the **kernel + BTF + loader/tooling + object** path, not the kernel alone.

## The unresolved gap is behavioral compatibility after successful admission

The previous capability-evidence report proposed receipts that explain why a concrete artifact was admitted or rejected on one host. That is necessary, but it still leaves an upgrade gap.

Suppose an object receives a successful receipt before and after an upgrade:

```text
kernel A: relocate pass -> verifier pass -> attach pass
kernel B: relocate pass -> verifier pass -> attach pass
```

The two receipts establish that both environments accepted the artifact. They do not by themselves establish that the artifact observed the same events, enforced the same policy boundary, or produced equivalent state transitions under the same workload.

Existing mechanisms cover pieces of this problem:

- CO-RE handles structural relocation against target BTF.
- The verifier checks safety and target-specific admissibility.
- BPF selftests exercise kernel BPF behavior and regression cases.
- Projects such as cilium/ebpf test across multiple kernel versions and expose feature-probe helpers.
- Production systems can run canaries before broad rollout.

What is still missing as a common deployment abstraction is a **behavior contract bound to one BPF artifact and exercised across the upgrade boundary**.

Such a contract should not try to freeze all kernel internals. It should describe only the observable invariants that make this application useful. For a network policy program that might mean verdicts and map transitions for a small packet corpus. For a profiler it might mean event coverage and attribution relationships. For a tracing program it might mean which operation lifecycle is represented by a hook and which fields must satisfy cross-field invariants.

## Research direction 1: build cross-kernel semantic witnesses for BPF artifacts

The first direction is to make compatibility tests artifact-specific instead of relying on a generic kernel matrix alone.

**Gap.** A kernel can accept one object on two releases while the application's observation or policy assumptions differ. Generic BPF selftests cannot know the application-level invariant.

**Mechanism.** Alongside the BPF object, generate a small set of semantic witnesses. Each witness contains a controlled stimulus, expected observable invariants, and an explicit tolerance policy. The deployment harness runs the same witness set against the old and candidate kernels using the same object and loader generation.

A witness should compare semantics rather than raw timestamps or incidental implementation details. Examples include:

- a network flow that must yield the same allow/drop decision and policy-generation transition;
- a file-operation sequence that must produce one logical open lifecycle even if internal functions differ;
- a scheduler event that must preserve a stated task-state invariant;
- a profiler workload whose causal attribution must retain the same parent/child relationship while sample counts may vary within a declared tolerance.

The output is not a single checksum. It is a structured result such as:

```text
artifact: sha256:...
loader: ...
old_kernel: ...
new_kernel: ...
witnesses:
  policy_allow: equivalent
  policy_revoke: equivalent
  event_lifecycle: changed
  attribution: equivalent-with-tolerance
```

**Delta from existing practice.** Kernel selftests test the kernel. Unit tests test the application implementation. The proposed artifact witness is a deployment object that explicitly spans **two target kernels** and encodes the application semantics that must survive the transition.

**Prototype.** Start with 20 to 40 small CO-RE programs covering fentry/fexit, tracepoints, helpers, kfuncs, maps, and one networking policy path. Package each with deterministic namespace or VM stimuli and normalized outputs.

**Evaluation.** Build a kernel matrix with upstream LTS releases, current releases, selected distribution kernels, and intentionally mutated kernels that change attach targets, BTF layouts, verifier rules, or lifecycle timing. Measure how often load-only testing declares compatibility while a witness detects a behavior change. Also measure false alarms caused by witnesses that are too implementation-specific.

**Academic value.** This turns "portable BPF" from a loadability claim into a falsifiable behavioral compatibility claim.

**Production value.** A fleet upgrade can be blocked by one named violated invariant instead of a generic unsupported-kernel label.

**Failure condition.** If useful witnesses must reproduce almost the entire application or are so brittle that routine harmless kernel changes fail them, the abstraction is too expensive and should be narrowed.

## Research direction 2: localize semantic drift to the dependency that changed

A failing witness is useful, but an operator still needs to know why it failed.

**Gap.** Today a post-upgrade difference can be blamed on BTF, CO-RE, the verifier, an attach target, a kfunc, loader probing, map behavior, or application logic. These layers often appear in one startup or runtime failure path.

**Mechanism.** Build a compatibility dependency graph from the artifact and loader. Nodes represent concrete dependencies such as a CO-RE field relocation, program type, helper/kfunc signature, attach target, map feature, verifier-sensitive construct, and loader probe. Edges connect each application witness to the dependencies it exercises.

During an upgrade, capture both sides:

```text
witness changed
    |
    +-- target BTF digest changed
    +-- CO-RE resolution changed
    +-- verifier decision/log class changed
    +-- attach target identity changed
    +-- kfunc signature/visibility changed
    +-- feature probe outcome changed
```

The system should then report the smallest changed dependency set consistent with the failed witness, rather than only dumping two verifier logs.

**Delta from the September 15 capability receipt.** A receipt explains one admission decision. This graph links **behavioral regressions across two admitted environments** back to the exact compatibility evidence that changed.

**Prototype.** Extend a libbpf-based loader to emit normalized dependency records and link them to witness IDs. Add adapters for target BTF hashes, CO-RE relocation results, helper/kfunc availability, attach metadata, and normalized verifier outcomes.

**Evaluation.** Inject one controlled incompatibility at a time, then combinations of two or three. Score root-cause localization precision, time to diagnosis, and the fraction of failures that remain ambiguous. Compare against kernel-version diffing, raw verifier logs, and a broad `bpftool feature` diff.

**Academic value.** This tests whether compatibility can be represented as a causal dependency problem rather than an opaque binary matrix.

**Production value.** Operators can distinguish "rebuild the object," "update the loader probe," "switch to the tracepoint fallback," and "the kernel changed the observed lifecycle" without manual archaeology.

**Failure condition.** If the dependency graph becomes a second hand-maintained kernel model, or if most semantic changes cannot be localized to observable dependencies, it is not a practical improvement over a support table.

## Research direction 3: make kernel upgrades pass an eBPF semantic promotion gate

The third direction is operational: treat a kernel upgrade like a software rollout that must preserve BPF application semantics.

**Gap.** Fleet kernel qualification often proves boot, generic workload health, and package compatibility, while BPF applications separately prove that they load. Neither necessarily checks the application-level meaning of the probes and policies that remain attached after rollout.

**Mechanism.** Before broad deployment, run the candidate kernel on a small set of representative nodes or VMs. Replay the artifact witnesses and dependency capture from the currently accepted kernel and the candidate kernel. Promote only when all hard invariants match and all tolerated changes remain within declared envelopes.

The gate should classify outcomes rather than reduce them to pass/fail:

- **equivalent:** hard invariants match;
- **compatible with declared variance:** metrics differ only inside a defined tolerance;
- **fallback required:** a supported alternative attachment or implementation preserves the contract;
- **behavioral regression:** the artifact loads but a hard witness changes;
- **admission regression:** relocation, verifier, or attach fails;
- **probe regression:** feature inference changes before application load.

**Prototype.** Integrate the gate with a CI kernel matrix and a canary-node upgrader. Preserve old/new receipts, witness outputs, and dependency deltas as one content-addressed promotion artifact.

**Evaluation.** Replay historical compatibility failures and synthetic mutations, including the Linux 7.2 feature-probe class. Compare four policies: version allowlist, load-only gate, capability-receipt gate, and semantic promotion gate. The main metrics are unsafe promotions, unnecessary blocks, diagnosis latency, and qualification cost.

**Academic value.** This gives kernel/BPF compatibility a measurable false-admission and false-rejection problem instead of treating support as a static matrix.

**Production value.** It creates a concrete place to catch BPF regressions before a fleet upgrade reaches every node.

**Failure condition.** If canary kernels and witness workloads do not predict production behavior well enough to reduce unsafe promotions, then the gate adds ceremony without useful protection.

## A practical deployment can layer evidence instead of testing everything every time

A semantic gate does not require a full compatibility lab on every process start.

The expensive work belongs at artifact build time, kernel qualification time, and controlled fleet canaries. Runtime loading can keep using cached target evidence keyed by the kernel/BTF/loader/artifact identities described in the earlier [capability-evidence report](https://eunomia.dev/research/ebpf-kernel-capability-evidence/).

A reasonable production ladder is:

```text
known-bad or unsupported kernel?
        -> reject

artifact capability receipt valid?
        -> no: probe / relocate / load / attach

kernel generation already passed artifact witnesses?
        -> yes: reuse qualified result
        -> no: run upgrade qualification before broad rollout

behavioral difference found?
        -> supported fallback, hold rollout, or update artifact/loader
```

The key is to preserve claim scope. If the only evidence is "CO-RE relocation succeeded," say that. If the actual object loaded, say that. If a witness suite exercised the application invariant across old and new kernels, then the deployment can make the stronger statement.

This is particularly useful for evolving interfaces such as kfuncs and kernel tracing targets. Rather than pretending they are stable forever, the system can accept that they move and require fresh evidence when they do.

## What would change this conclusion?

Three findings would weaken the need for an explicit semantic upgrade contract.

First, if a broad kernel and distribution study shows that successful CO-RE relocation plus verifier/attach success predicts application-level behavior with negligible false admission across realistic BPF workloads, load-time evidence may already be strong enough.

Second, if almost all observed upgrade failures are clean admission failures rather than silent or probe-level semantic mismatches, then richer admission receipts may capture most of the practical value without cross-kernel witnesses.

Third, if application-specific witnesses prove too brittle to distinguish harmless kernel implementation changes from meaningful behavior changes, a shared semantic gate would create more false blocks than useful protection.

The current evidence points the other way. Linux deliberately exposes both stable and unstable BPF-facing interfaces, CO-RE is explicitly a relocation mechanism, and recent production failures show that even feature-probe interpretation can drift with verifier behavior. The safer deployment model is therefore layered: **relocate structure, prove admission, then test the application semantics that matter across the upgrade boundary.**

## Sources

- [Linux BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html)
- [Linux BPF LLVM Relocations and CO-RE relocations](https://docs.kernel.org/bpf/llvm_reloc.html)
- [Linux BPF Type Format documentation](https://docs.kernel.org/bpf/btf.html)
- [Linux BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)
- [cilium/ebpf issue #2084: Linux 7.2-rc6 feature-probe failure](https://github.com/cilium/ebpf/issues/2084)
- [Cilium issue #48016: Linux 7.2 `bpf_set_retval` probe failure](https://github.com/cilium/cilium/issues/48016)
- [cilium/ebpf PR #2086: fix BTF essential-name indexing for leading triple underscores](https://github.com/cilium/ebpf/pull/2086)
- [cilium/ebpf repository and kernel-version CI matrix](https://github.com/cilium/ebpf)
