---
date: 2026-09-15
slug: ebpf-kernel-capability-evidence
title: "Can an eBPF Loader Trust the Kernel Version?"
description: "Kernel versions do not prove eBPF capability on backported systems. This report develops capability receipts, semantic canaries, and support envelopes."
tags:
  - Daily Report
  - eBPF
  - Linux
  - Compatibility
  - libbpf
  - BTF
research_question: "How should an eBPF loader decide what a real deployed kernel can safely load and attach when distribution backports, configuration, BTF, privileges, and rapidly evolving BPF interfaces make version checks incomplete?"
source_cutoff: 2026-09-15
status: daily-report
---

# Can an eBPF Loader Trust the Kernel Version?

A loader often starts compatibility logic with a simple question: what does `uname -r` say?

That is useful inventory information. It is not a sufficient capability contract.

A distribution kernel can keep an old upstream base number while backporting years of subsystem work. A host can compile out one BPF facility, disable unprivileged BPF, expose a different BTF surface, or carry verifier and kfunc behavior that does not match an upstream version table. Even two machines with apparently similar kernel versions can give the same BPF object different answers because the relevant facts are not encoded in one version integer.

The practical conclusion is narrower than “never check versions.” Version and distribution metadata are still useful for support policy, known-bad ranges, and debugging. But a loader that must decide whether a concrete BPF artifact can run should prefer **evidence from the kernel it is about to use** over an inference from the kernel version it resembles.

This report develops that idea into a deployment contract: a loader should build a reproducible **capability receipt** from active feature probes, the running kernel's BTF surface, the artifact's own relocation and verifier result, the privilege context, and, where existence is not enough, a small semantic canary. The receipt should explain not only why the fast path was selected, but also why a fallback was selected or why loading was refused.

<!-- more -->

This is different from the earlier report on [architecture-specific eBPF specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/). That report asks whether an optional native optimization is eligible on a CPU/JIT backend while keeping one portable semantic witness. Here the BPF object itself is unchanged. The problem is deciding whether a real Linux deployment, often with distribution backports and local policy, actually provides the BPF interfaces the object expects.

It is also different from [stateful eBPF transactional upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). Transactional upgrade is about changing a running multi-object application without corrupting state. Capability evidence comes earlier: can the candidate object be admitted on this host at all, and what evidence supports that answer?

## A distribution kernel version is a lineage label, not a feature bitmap

Red Hat's current RHEL kernel documentation states the problem unusually directly. The upstream base number in a RHEL kernel version does not list every source change, and the version string alone cannot determine whether a specific upstream feature, API, or driver behavior is present. Red Hat keeps a stable base while backporting fixes, features, hardware support, and distribution-specific work.

RHEL 9.6 makes the mismatch concrete. The release still reports a `5.14.0`-based kernel package, while the release notes say the **eBPF facility has been rebased to Linux 6.12**. The same release documents BPF token, BPF arena, new kfuncs, and detection of kfuncs for the running kernel. A policy such as `kernel >= 6.12 means feature X` would therefore misdescribe a system whose BPF subsystem incorporates much newer work under a 5.14 base version.

Red Hat also publishes a full “Available BPF features” chapter generated from `bpftool feature`. That is a useful operational signal in itself: the distribution does not ask users to reconstruct BPF support solely from the upstream base number. It reports the capabilities observed for the shipped kernel, including configuration, program types, helper availability, map types, and other BPF properties.

Backports are not only a distribution feature story. Security and correctness fixes also move across version lines. Ubuntu's record for CVE-2021-3490, for example, notes that an eBPF verifier fix associated with upstream development was backported into several stable kernel series. The exact patch history matters more than a naive major/minor threshold.

This does not mean version checks are worthless. They remain good for questions such as:

- whether a vendor officially supports the host;
- whether a specific known regression range should be blocked;
- whether a package, kernel ABI, or distribution release belongs to a tested support tier;
- which compatibility evidence should be collected next.

The mistake is using the version string as if it were direct proof that `this program type + this helper + this kfunc + this attach path + this verifier behavior` exists.

## Linux and libbpf already prefer active evidence for many capabilities

`bpftool feature probe kernel` actively interrogates the running kernel. Current bpftool documentation describes probes for the `bpf()` system call, JIT state, program types, helper functions, and other BPF-related parameters. Newer bpftool versions also distinguish the built-in items the tool knows about from what the target system actually supports.

libbpf exposes the same idea as APIs. `libbpf_probe_bpf_prog_type()` attempts a minimal program load to determine whether the host kernel supports a program type. `libbpf_probe_bpf_map_type()` probes map support. `libbpf_probe_bpf_helper()` asks whether a helper is supported for a particular program type. These are not static tables keyed by `LINUX_VERSION_CODE`; they ask the kernel.

BTF and CO-RE add a different kind of evidence. The running kernel exposes authoritative BTF at `/sys/kernel/btf/vmlinux` when configured. libbpf can match the BPF object's recorded type and relocation information against that target BTF and repair field offsets and related type-dependent references. A loader can therefore reason about the target's actual type surface instead of assuming that `struct task_struct` on one kernel version has the layout of another.

These mechanisms already imply an evidence ladder:

```text
version / distro metadata
        |
        v
broad active capability probes
        |
        v
target BTF + CO-RE relocation
        |
        v
object-specific verifier/load result
        |
        v
attach / semantic canary when needed
```

The lower a decision goes on this ladder, the more directly it answers the question the loader actually cares about.

## Feature presence still does not prove object compatibility

A general capability probe can say that the kernel supports a program type or helper while the actual object still fails. Several boundaries remain object-specific.

### CO-RE solves structural relocation, not every BPF contract

CO-RE is very good at a particular problem: making references to kernel types and fields relocatable against the target BTF. It does not prove that every helper, map, kfunc, attach type, verifier rule, or runtime semantic assumption used by the program remains valid.

Linux's BPF design documentation explicitly separates stable BPF ABI pieces from unstable kernel internals and tracepoints. It also treats kfuncs differently from stable helpers. Current kernel documentation says kfuncs are kernel-to-kernel APIs without hard stability guarantees; maintainers may change or remove them when justified, and kfunc visibility can depend on program type.

That matters more as new BPF functionality moves toward kfuncs. A loader cannot safely reduce `the symbol exists in some documentation` to `this object may call it here`.

### Privilege is part of observed capability

Feature probing is also permission-sensitive. libbpf's probe APIs warn that the process needs the required capabilities or root privileges for feature checks. `bpftool feature` has an explicit `unprivileged` mode because otherwise a non-root probe could misclassify kernel capability as kernel absence.

A production receipt therefore needs to record the **probe authority context**. “Unsupported” and “not observable with these credentials” are different states. Likewise, a privileged deployment controller should not use its own result to promise that an unprivileged tenant can perform the same operation.

### The verifier is an executable compatibility boundary

The verifier combines program type, helper/kfunc contracts, pointer and lifetime rules, BTF knowledge, kernel configuration, and the concrete instruction graph. A successful load of the actual object is stronger evidence than a broad feature flag.

The BPF subsystem is still evolving here. A September 2026 bpf-next patch series, for example, is actively unifying helper and kfunc argument checking, including type admission, nullability, memory, BTF, packet access, and resource ownership handling. Whether or not that series lands in a particular form, it is current evidence that the verifier-side contract is an active implementation surface rather than a frozen lookup table.

For a loader, the verifier should therefore be treated as an admission oracle whose result is captured, not merely an inconvenience behind a compatibility table.

## The missing abstraction is a capability receipt bound to the artifact

Most systems have pieces of the evidence but do not bind them into one durable answer.

A useful capability receipt should identify four things independently:

1. **Target identity.** Distribution/package identity, full kernel release, architecture, boot or build identity where available, relevant config evidence, BTF digest, and security/privilege context.
2. **Probe identity.** bpftool/libbpf versions, probe mode, effective capabilities, and the exact general feature results used by policy.
3. **Artifact identity.** Hash of the BPF object or skeleton, compiler and libbpf compatibility metadata when relevant, expected program/map/attach types, kfunc/helper dependencies, and CO-RE relocation results.
4. **Admission result.** Verifier/load outcome, normalized error class, verifier-log digest or retained log according to policy, attach result if attempted, selected fallback, and any semantic canary result.

For example:

```text
target:
  kernel_release: 5.14.0-...el9
  distro_package: ...
  btf_sha256: ...
  privilege_profile: deployment-controller-v2

artifact:
  object_sha256: ...
  requires:
    prog_types: [tracing]
    maps: [ringbuf]
    kfuncs: [...]

probe:
  bpftool: ...
  libbpf: ...
  general_features: ...
  core_relocations: pass

admission:
  verifier: pass
  attach_canary: pass
  selected_path: fentry
  fallback: tracepoint
```

This is deliberately not a proposal to freeze all kernel internals into a new ABI. It is an evidence artifact. Its job is to let the deployment system answer a postmortem question: **what did we observe on this host, for this BPF artifact, under this authority, when we decided it was compatible?**

## Where current work is still weak

### General feature matrices are broader than application requirements

`bpftool feature` can produce a rich host capability view, and vendor release notes can publish that view for a shipped kernel. But an application usually depends on a much smaller conjunction: one program type, several helpers or kfuncs, a few map types, a particular attach mechanism, specific BTF types, and concrete verifier behavior.

A large matrix says what the host can generally do. It does not explain which rows were necessary for one deployment decision or which fallback becomes valid if one row is missing.

The missing element is an object-bound dependency projection: derive the smallest relevant capability claim from the artifact, test it, and preserve the evidence.

### Load success can still be weaker than runtime-semantic compatibility

Some properties are exercised only when a program attaches or when the relevant hook executes. A load-only test can prove verifier acceptance without proving that the expected attachment exists, that the expected BTF-backed kfunc is available in that exact context, or that a distribution-specific behavior matches the application's assumption.

The missing element is a safe way to test semantic edges without turning compatibility probing into production side effects.

### Compatibility incidents are hard to reproduce after the host changes

When a fleet rolls forward, `the object failed on kernel 5.14` is often too little evidence. Was it the exact distribution build, a config difference, BTF, a permission change, libbpf, a verifier backport, a kfunc contract, or the object itself?

Without a durable receipt, the evidence disappears when the host is upgraded or rebooted. A support matrix can tell you what should have worked; it cannot replay what the loader actually observed.

## Research directions worth building

### 1. Generate minimal, artifact-bound capability receipts

Build a loader pass that extracts the artifact's actual compatibility dependencies before loading. The prototype should combine ELF/BTF/CO-RE metadata with loader-known program, map, helper, kfunc, and attach requirements, then run only the probes needed to decide this artifact's policy.

The output is a signed or content-addressed receipt that can be cached per target identity and attached to deployment telemetry.

The academic question is whether a small automatically derived capability set predicts real load/attach success better than version tables or broad distribution support matrices. The production value is explainable admission and deterministic fallback without probing every BPF feature on every start.

Evaluate across upstream kernels plus long-lived distribution kernels with aggressive backports. Use multiple privilege profiles. For each artifact/host pair, compare four predictors: version threshold, vendor support table, broad `bpftool feature` output, and artifact-bound receipt. The primary metric is false admission and false rejection, not how many features are discovered.

This direction fails if the derived dependency set is unstable, nearly as large as the full matrix, or does not materially reduce compatibility misclassification.

### 2. Add side-effect-bounded semantic canaries for ambiguous edges

When general probing plus object load is insufficient, run a deliberately tiny canary that exercises only the uncertain boundary.

A map canary can create and destroy the required map type. A kfunc canary can load the smallest legal program that calls the required function in the intended program context. An attach canary may use a namespace, temporary cgroup, or test interface only when that hook is proven to be scoped to that isolation boundary; a disposable link limits lifetime, not observation scope. Host-wide hooks such as fentry or tracepoints should be exercised on a dedicated test VM or host, or not probed in production. Every canary should have an explicit side-effect budget and cleanup contract.

The research problem is choosing canaries that are strong enough to predict the real application while remaining safe and cheap. One can model this as **compatibility test selection**: given a dependency graph and prior failures, choose the smallest set of probes that reduces uncertainty below a deployment threshold.

Evaluate canary predictive power on a matrix with intentionally missing config, permission changes, BTF differences, backported verifier behavior, absent attach targets, and changing kfunc surfaces. Compare load-only admission with load-plus-canary admission.

This direction fails if canaries themselves become a fragile parallel implementation of the application, require unsafe production mutations, or still miss the failure classes operators care about.

### 3. Build a replayable artifact-to-kernel support envelope

A single receipt explains one decision. A support envelope aggregates receipts across a fleet or CI matrix.

For each BPF artifact generation, maintain a set of tested target identities and outcomes, but key them by observed capability evidence rather than only `kernel >= X`. The envelope can answer questions such as:

- Which target capabilities have actually admitted this object?
- Which verifier or attach failure classes recur?
- Which fallback paths were exercised rather than merely compiled?
- Did a distribution update change capability evidence while keeping the same upstream base version?
- Can a new artifact generation shrink or expand the proven support set?

The research artifact would combine CI, production receipts, and a replay harness. Feed a saved receipt into a synthetic target model or matching kernel VM and check whether the loader makes the same decision. Use Linux BPF selftests as a methodological precedent: kernel BPF development already prefers regularly executed selftests for functional and corner-case regression coverage.

The production value is replacing hand-maintained version folklore with evidence that accumulates as the system is tested and deployed.

This direction fails if target identities fragment so aggressively that receipts never generalize, or if replay cannot preserve the verifier, BTF, privilege, and attach conditions needed to reproduce outcomes.

## A practical loader policy can be conservative without becoming slow

Active evidence does not require probing the whole kernel on every process start.

A production implementation can cache receipts by a composite identity containing the kernel package/build identity, BTF digest, relevant config or security-policy identity, privilege profile, BPF artifact hash, and a probe/policy digest covering bpftool/libbpf, dependency-extractor, and admission-policy versions. Reuse is valid only while every component matches; changes in probe logic or policy invalidate the result just as target or artifact changes do.

The decision policy can also preserve version metadata as an outer support boundary:

```text
unsupported vendor / known-bad build?
        |
        +-- yes -> reject
        |
        +-- no  -> evaluate artifact requirements
                     |
                     +-- active evidence sufficient -> load
                     |
                     +-- ambiguous edge -> bounded canary
                     |
                     +-- missing capability -> explicit fallback or reject
```

This is more defensible than either extreme. A loader does not have to ignore vendor support policy, and it does not have to blindly trust a version number when the kernel can answer the narrower capability question directly.

The fallback also becomes observable. If fentry is unavailable but a tracepoint path is supported, record that selection. If a kfunc-based implementation is unavailable but a stable-helper implementation exists, record the downgrade. A compatibility system that silently picks a different path loses much of the value of active probing because operators still cannot explain what ran.

## What would make this conclusion wrong?

Three findings would weaken the case for artifact-bound capability evidence.

First, large-scale evaluation might show that full distribution package versions plus vendor release matrices already predict BPF load, attach, and semantic success with essentially no false decisions. In that world active probing adds operational complexity without improving admission quality.

Second, the kernel and libbpf could converge on one stable, comprehensive capability-query ABI that directly describes every dependency an application needs, including kfunc and attach semantics. If that interface were authoritative and cheap, custom receipts could collapse into a signed snapshot of the standard query.

Third, object-specific verifier/load tests might dominate every broader signal. If simply attempting to load the final artifact under the final privilege context predicts all relevant compatibility and can always be done safely before activation, a separate layer of general probes and semantic canaries may not be worth maintaining.

Current evidence does not support those simplifications. RHEL explicitly documents that version strings cannot determine feature presence and publishes BPF capability tables generated from active probing. libbpf contains kernel-probing APIs. CO-RE consumes the target's actual BTF rather than a guessed layout. kfuncs intentionally lack hard stability guarantees, and current verifier work continues to evolve their argument contracts. **The stronger deployment abstraction is therefore not “kernel X is new enough.” It is “this exact artifact was admitted on this exact target under this authority, and here is the evidence and fallback decision that prove what we meant by compatible.”**

## References

- Red Hat. [Managing, monitoring, and updating the kernel: What the version string does not mean](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/managing_monitoring_and_updating_the_kernel/index), accessed 2026-09-15.
- Red Hat. [Red Hat Enterprise Linux 9.6 Release Notes: Kernel and eBPF facility](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/9.6_release_notes/new-features), accessed 2026-09-15.
- Red Hat. [RHEL 9.6 Available BPF features](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/9.6_release_notes/index), accessed 2026-09-15.
- Linux kernel documentation. [libbpf Overview](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html), accessed 2026-09-15.
- Linux kernel documentation. [BPF Type Format](https://docs.kernel.org/bpf/btf.html), accessed 2026-09-15.
- Linux kernel documentation. [BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html), accessed 2026-09-15.
- Linux kernel documentation. [BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html), accessed 2026-09-15.
- Linux kernel documentation. [HOWTO interact with BPF subsystem](https://docs.kernel.org/bpf/bpf_devel_QA.html), accessed 2026-09-15.
- libbpf. [`libbpf_probe_bpf_prog_type`, `libbpf_probe_bpf_map_type`, and `libbpf_probe_bpf_helper`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h), accessed 2026-09-15.
- bpftool. [`bpftool feature` manual](https://manpages.debian.org/unstable/bpftool/bpftool-feature.8.en.html), accessed 2026-09-15.
- Ubuntu Security. [CVE-2021-3490](https://ubuntu.com/security/CVE-2021-3490), updated 2026, accessed 2026-09-15.
- Amery Hung. [PATCH bpf-next v2 00/23: Unify helper and kfunc argument checks](https://lore-kernel.gnuweeb.org/bpf/20260911221956.1F62A1F00893%40smtp.kernel.org/T/), BPF mailing-list archive, 2026-09-11.