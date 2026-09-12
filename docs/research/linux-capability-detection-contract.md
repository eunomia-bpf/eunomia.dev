---
date: 2026-09-12
slug: linux-capability-detection-contract
title: "Why Is Linux Capability Detection Harder Than Checking the Kernel Version?"
description: "Linux capability detection fails when backports, configuration, and semantic fixes diverge from kernel versions. This report proposes capability receipts."
tags:
  - Daily Report
  - Linux
  - Compatibility
  - System Calls
  - Runtime
  - Portability
research_question: "How should portable Linux software decide that a runtime feature is actually usable when distribution backports, configuration, policy, and semantic fixes diverge from the kernel version string?"
source_cutoff: 2026-09-12
status: daily-report
---

# Why Is Linux Capability Detection Harder Than Checking the Kernel Version?

Suppose a service starts on a machine reporting Linux 5.14. It wants a security feature, a newer `io_uring` operation, or an `openat2()` behavior. A simple deployment rule says that the feature arrived in Linux version X, compares X with `uname -r`, and chooses the fast path or fallback.

That rule is attractive because it turns compatibility into one integer comparison. Real Linux deployments do not preserve that relationship. Distribution kernels backport newer subsystem changes onto older upstream version bases. Features can be compiled out or disabled at boot. A sandbox or compatibility layer can hide a system call that the host kernel implements. Even when an API exists, a later semantic fix can matter to the application without changing the API version it originally checked.

**Linux capability detection therefore needs evidence from the running interface, not only a guess derived from the kernel version. The harder problem is preserving enough evidence to explain exactly which behavior was proved, in which execution context, and why the program selected one path.**

<!-- more -->

This is an adjacent-systems detour from the current eBPF roadmap. The newest-ten Daily Report window is already at the configured maximum of seven eBPF-centered reports, so another eBPF report would violate the editorial mix. The compatibility problem is still useful background for portable runtimes such as [bpftime](https://github.com/eunomia-bpf/bpftime), but this report is about Linux userspace interfaces in general rather than BPF-specific capability negotiation.

## Linux capability detection is already a collection of different contracts

The strongest evidence against version-only logic comes from Linux interfaces themselves: they have independently invented several ways to negotiate compatibility.

Red Hat documents why the release string is an incomplete capability description. A RHEL 9 kernel can report a 5.14 upstream base while individual subsystems contain changes from much newer upstream kernels. The version string identifies the packaged kernel lineage; it does not enumerate every backported subsystem feature.

Landlock goes further and explicitly tells userspace not to infer support from the kernel version. An application can query the Landlock ABI version with `landlock_create_ruleset(..., LANDLOCK_CREATE_RULESET_VERSION)`, then enable only access rights present in that ABI. The current August 2026 kernel documentation also exposes `LANDLOCK_CREATE_RULESET_ERRATA`: a bitmask for semantic fixes that may require userspace awareness. That distinction is important. A feature can exist, have the same broad ABI generation, and still differ in behavior because a correctness fix is present or absent.

`openat2()` uses another pattern. Its `struct open_how` is explicitly extensible, and the structure size acts as an implicit version. If userspace passes fields newer than the running kernel understands, nonzero unknown extension data produces `E2BIG`. The manual page even describes probing the largest supported structure size. Here compatibility is negotiated through structure shape rather than a named ABI number.

`io_uring` has accumulated still another set of mechanisms. `IORING_REGISTER_PROBE` reports supported operation codes. Feature bits on `io_uring_setup()` describe selected behavior. Current man-pages also document `IORING_REGISTER_QUERY`, available since kernel 6.15, which can query supported opcodes, flags, and subsystem-specific capabilities without first creating a ring. The subsystem has moved toward richer runtime capability queries because a single release number cannot conveniently describe the matrix.

These interfaces are not inconsistent mistakes. Each reflects a different compatibility problem. The issue for portable userspace is that the evidence is fragmented across versioned ABIs, bitmaps, opcode probes, extensible structs, configuration, return codes, and execution policy.

## Four different questions get collapsed into “is this feature supported?”

The first question is **implementation presence**. Does this kernel or compatibility layer implement the syscall, opcode, flag, or object shape at all? `ENOSYS`, an opcode probe, or a query interface can often answer this.

The second is **runtime availability**. Landlock can be compiled into a kernel yet disabled by boot configuration. `io_uring` operations may have privilege or setup-mode requirements. A feature can exist in source and remain unusable in the current boot or process environment.

The third is **semantic level**. An ABI number or opcode bit may establish a coarse feature generation while a later fix changes an edge case the application cares about. Landlock's errata mechanism makes this explicit: userspace may occasionally need to know not only that the interface exists, but whether a particular semantic correction is present.

The fourth is **authorization in this execution context**. Containers, seccomp policies, virtualized kernels, LSM configuration, namespaces, and credentials can make two processes on the same nominal kernel observe different usable capability sets. A host-global cache can therefore answer the wrong question for a sandboxed process.

A version comparison compresses all four questions into one proxy. A direct probe is better, but even a probe is only useful if the software remembers what that probe established and which decision depended on it.

## Where current work is still weak

The first gap is **a common evidence model across Linux APIs**. Landlock can return an ABI and errata mask, `io_uring` can return operation and flag support, and `openat2()` can negotiate structure size. Portable applications still encode each answer as ad hoc booleans in loader code. After an incident, it can be difficult to tell whether a fallback was selected because the kernel lacked a feature, the feature was disabled, the process was denied, or the application simply did not recognize a newer interface.

The second gap is **semantic confidence beyond feature presence**. Capability APIs normally answer questions defined by their own subsystem. They rarely express the higher-level guarantee the application wants, such as “this path resolution cannot escape this root under these flags” or “this sandbox correctly handles the network behavior my policy relies on.” Landlock's new errata query is evidence that behavior-level distinctions sometimes matter after an ABI already exists.

The third gap is **scope and lifetime**. Some observations are stable for the boot; others depend on process credentials, namespace, seccomp state, container runtime, or a virtualized syscall surface. Capability caches seldom declare their validity scope. Reusing a host probe inside a more restricted execution context creates a stale-evidence bug rather than a missing-feature bug.

The fourth gap is **reproducible fleet compatibility testing**. Distribution backports intentionally break the simple mapping from upstream version to feature set, but many CI matrices still label machines primarily by kernel release. That makes it hard to distinguish a failure caused by version lineage from one caused by a specific backport, configuration choice, semantic fix, or policy layer.

## Promising directions with academic and production value

### 1. Turn capability probes into typed receipts

The gap is not lack of probes. Linux already has many good ones. The missing piece is a common record of what each successful or failed probe actually proves.

A small capability receipt could contain:

```text
requirement = landlock.net.bind-tcp
probe = LANDLOCK_CREATE_RULESET_VERSION
result = abi>=4
semantic_fixes = {erratum-1, erratum-2}
kernel_build = package + build-id
execution_scope = boot + userns + seccomp-profile + credentials
observed_at = process-start
chosen_path = landlock-network-policy-v2
fallback = filesystem-only-policy
```

The mechanism would be a userspace library and schema that adapters populate from native subsystem probes. The library should not replace Landlock, `io_uring`, or `openat2()` negotiation. It should preserve their source-native meaning and bind the result to the application decision that consumed it.

The strongest adjacent baseline is direct hand-written feature detection. Evaluate both approaches across mainline kernels, long-lived distribution kernels, different boot configurations, containers, and syscall-virtualization environments. Inject backports, disabled features, denied syscalls, and selected semantic errata. Measure false-positive enablement, unnecessary fallback, time to explain a decision, receipt size, and startup overhead.

The academic value is a model of capability evidence that separates presence, availability, semantics, and authorization instead of treating support as one bit. The production user is a runtime, database, storage engine, sandbox, or agent executor that already contains kernel-version conditionals and feature probes; the integration boundary is its startup and backend-selection path.

This idea loses if ordinary per-API probes plus a small amount of logging achieve the same decision accuracy and postmortem explainability at materially lower complexity. A general schema is useful only if it captures recurring structure rather than normalizing everything into vague metadata.

### 2. Build a counterexample corpus for version-based compatibility rules

A compatibility test suite can deliberately create cases where a release-number heuristic gives the wrong answer. One machine can use a distribution backport on an older upstream base. Another can run a newer kernel with the feature disabled. A container can deny the syscall. A virtualized kernel can omit it. Two kernels can expose the same coarse ABI while differing in a documented erratum.

The artifact would contain small requirement-specific probes plus expected behavior tests, not a table claiming that “kernel X supports feature Y.” Each test records the version heuristic's prediction, the native probe result, and an operation that exercises the semantic property the application actually needs.

Evaluate several real Linux libraries or runtimes that currently gate features by version, compile-time macros, or direct probes. The primary metric is not test coverage. It is the rate of **capability misprediction**: enabling an unusable or semantically unsuitable path, and unnecessarily disabling a usable one. An ablation removes backports and policy layers to show how much each source of divergence contributes.

The academic contribution is a measurable taxonomy of compatibility failures in mixed Linux fleets. Production teams can run the corpus against the exact kernel images and sandbox profiles they ship before widening a rollout.

This direction should be abandoned if representative production kernels and environments almost never disagree with a simple version rule, or if direct native probes already catch every meaningful counterexample. The point is to expose real divergence, not manufacture exotic failures.

### 3. Give capability evidence an explicit validity lease

A capability result should state how long and where it is valid. A kernel-build property may survive for a whole boot. A boot-time LSM setting should be invalidated at reboot. A seccomp or namespace-dependent result may be valid only for one process lineage. A result observed outside a sandbox should not automatically authorize a path inside it.

The mechanism can be simple: attach a scope key to each receipt, such as kernel build identity, boot ID, namespace identities, sandbox-policy digest, and credential class. Cache reuse is allowed only when the requirement declares which components matter and the key still matches. A new environment either re-probes or takes the conservative fallback.

The evaluation should compare global host caching, probe-on-every-use, and lease-scoped caching across container starts, sandbox changes, privilege drops, kernel upgrades, and restarts. Measure stale-positive decisions, redundant probes, startup latency, and the cost of conservative fallback. A useful adversarial test moves a process into a more restricted execution environment after a host-level probe has succeeded.

The academic question is how to infer the minimum validity scope for heterogeneous OS capabilities without turning every check into a one-shot probe. Production value appears in long-running runtimes and fleet agents that cache backend choices and then create workers under different policies.

This mechanism is unnecessary when all relevant capability observations are immutable for the application's lifetime and process-start probing is already cheap. In that case a lease system would add state without removing failures.

## A practical rule for portable Linux software

The simplest usable policy is hierarchical.

Use the kernel release as **orientation**, not proof. It is useful for logging, coarse support policy, and deciding which probe code is worth attempting.

Use the subsystem's native negotiation mechanism as **capability evidence**. Prefer Landlock ABI queries, `io_uring` probes and queries, extensible-structure negotiation, or the documented syscall behavior over a guessed minimum version.

Then test the **semantic property that matters** when the feature is security- or correctness-sensitive. A successful syscall is not automatically proof of every property the application expects from it.

Finally, bind the result to its **execution scope and fallback decision**. When an operator asks why a worker used the slow path while another did not, the system should be able to answer with evidence rather than reconstructing an `if (kernel_version >= ...)` branch from source code.

This is related to the earlier report on [architecture-specific eBPF portability](https://eunomia.dev/research/ebpf-portable-architecture-specialization/), but the boundary is different. That report asks how one portable BPF semantic artifact chooses architecture-specific implementations. Here the object is ordinary Linux userspace software facing several unrelated kernel APIs and distribution backports. The lesson for the future eBPF compatibility series is narrower: BPF-specific work should focus on verifier, CO-RE, kfunc, attachment, and persistent-state semantics rather than repeat a generic Linux capability-receipt design.

## What would change this conclusion?

The argument weakens if distribution kernel versions become reliable capability identifiers in practice. A broad fleet study could test that: take commonly deployed RHEL, Ubuntu, Debian, cloud, mainline, and virtualized kernels, predict usable features only from release strings, and compare those predictions with native probes plus semantic tests. If false positives and false negatives are negligible, direct capability evidence may not justify its engineering cost for most applications.

The conclusion also changes if Linux converges on a common machine-readable negotiation contract that already reports presence, configuration, semantic revision, and execution-policy constraints. `IORING_REGISTER_QUERY` and Landlock's ABI/errata queries move in that direction inside individual subsystems, but there is no single cross-subsystem contract today.

Finally, not every feature deserves semantic testing or durable receipts. A best-effort performance optimization with a safe fallback can simply try the operation and fall back on failure. The stronger machinery is justified when a wrong positive can violate security or correctness, when a fleet decision is expensive to reverse, or when operators need to reproduce why two apparently similar Linux hosts behaved differently.

The practical standard is therefore not “never check `uname`.” It is: **use version strings to locate the neighborhood, use runtime interfaces to prove capability, and keep enough scoped evidence to explain the choice when backports, configuration, or semantics make two similar-looking kernels behave differently.**

## References

- Linux kernel documentation, [Landlock: unprivileged access control](https://kernel.org/doc/html/latest/userspace-api/landlock.html), including ABI and errata negotiation, accessed 2026-09-12.
- Linux man-pages, [`io_uring_register(2)`](https://man7.org/linux/man-pages/man2/io_uring_register.2.html), including `IORING_REGISTER_PROBE` and `IORING_REGISTER_QUERY`, accessed 2026-09-12.
- Linux man-pages, [`openat2(2)`](https://man7.org/linux/man-pages/man2/openat2.2.html), including extensible-structure negotiation, accessed 2026-09-12.
- Red Hat Enterprise Linux documentation, [Managing, monitoring, and updating the kernel](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_monitoring_and_updating_the_kernel/), including the RHEL backport model, accessed 2026-09-12.
- Linux kernel documentation, [Linux ABI description](https://kernel.org/doc/html/latest/admin-guide/abi.html), accessed 2026-09-12.
