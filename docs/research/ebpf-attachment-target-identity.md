---
date: 2026-09-28
slug: ebpf-attachment-target-identity
title: "Can an eBPF Attachment Follow a Recreated Workload?"
description: "BPF links bind programs to concrete kernel targets, while containers, cgroups, namespaces, and interfaces can be replaced under the same logical workload. This report asks how attachment continuity should be proved across target generations."
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF link
  - Kubernetes
  - Compatibility
research_question: "When a logical workload survives but its kernel attachment target is recreated, how can an eBPF controller prove that instrumentation or policy follows the new target without a silent coverage gap?"
source_cutoff: 2026-09-28
status: daily-report
---

# Can an eBPF Attachment Follow a Recreated Workload?

Suppose an agent attaches an eBPF program to a container's cgroup and pins the resulting BPF link. The loader exits. The link is still there, so the operator reasonably says, "the attachment survives controller restart."

Then Kubernetes replaces the Pod.

The replacement can have the same workload role and even the same Pod name, but Kubernetes gives it a different Pod UID and may place it on another node. At the Linux layer, the new instance is represented by new namespaces, cgroups, network devices, and other kernel objects. A BPF link that was created for the old target does not contain a Kubernetes selector saying "follow whatever becomes the next replica of this workload." It represents an attachment to a concrete kernel target.

That creates a deployment boundary which is easy to miss: **persistence of the BPF object is not continuity of the logical attachment.**

The kernel API makes the distinction visible. `BPF_LINK_CREATE` attaches a program to a target and returns a link file descriptor. For cgroup programs the target is a file descriptor for a particular cgroup directory; for XDP the target is a network-interface index; other link types use similarly concrete targets. Pinning a link can preserve the link beyond the loader process, but it does not turn that target into a logical workload selector.

This report continues the [deployment compatibility and lifecycle](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/) series. Earlier reports asked whether a host can admit an artifact, whether its behavior remains correct after a kernel upgrade, and how a loader selects among evolving kernel interfaces. Here the artifact and kernel may be completely compatible. The problem is that **the thing the program was attached to has been replaced**.

<!-- more -->

## BPF links solve ownership, not logical target rebinding

BPF links are a major improvement over older attachment APIs because they give an attachment an explicit kernel object and file-descriptor lifetime. The userspace API describes `BPF_LINK_CREATE` as attaching a program to `target_fd` at an `attach_type` hook and returning a file descriptor for the link. Like other BPF objects, links can be pinned in bpffs so that closing the original loader process does not necessarily remove the last reference.

That is exactly the property a long-running observability or networking agent wants when its controller restarts.

But the API also shows the boundary:

```text
logical intent
    "observe workload checkout-api"

        resolve

kernel target generation G17
    cgroup fd / netns fd / ifindex / other target handle

        attach

BPF link L42 -> target generation G17
```

If the workload is replaced, the controller may resolve the same logical intent to generation G18:

```text
logical intent
    "observe workload checkout-api"

        resolve again

kernel target generation G18

        attach again

BPF link L43 -> target generation G18
```

Nothing about keeping `L42` alive proves that `G18` is covered.

This is not a Kubernetes-specific quirk. Linux exposes concrete target identities because the kernel must know exactly where a program executes. Libbpf's cgroup attach API accepts a `cgroup_fd`. TCX and XDP attachment select a network interface by `ifindex`. Network-namespace-aware hooks use namespace file descriptors. These identifiers are useful precisely because they identify a real object, not an abstract service.

## Target lifetime is part of BPF semantics

The current Linux cgroup/BPF implementation makes target lifetime an explicit kernel concern. `kernel/bpf/cgroup.c` registers a cgroup lifetime notifier and handles cgroup online/offline transitions. It also has dedicated destruction work for cgroup BPF state.

The useful conclusion is not that every link type has identical teardown behavior. They do not. The conclusion is that an attachment's lifecycle is coupled to the lifecycle rules of its target type.

This matters in both directions.

First, a controller must not infer logical coverage from the existence of a pinned link alone. A pinned object can prove that a kernel BPF object still has a reference. It cannot prove that the intended workload is still represented by the same target.

Second, a controller must not assume that a reusable-looking identifier is a durable workload identity. An interface index is a kernel networking identifier, not a Kubernetes Pod UID. A cgroup path is a location in the cgroup hierarchy, not a proof that the process population behind the path is the same logical generation that was originally selected.

The right question is therefore:

> Which identity is stable enough for workload intent, which identity names the current kernel target, and what evidence proves the mapping between them?

## Kubernetes makes the identity split concrete

Kubernetes explicitly describes Pods as relatively ephemeral. A replacement Pod can have the same name as the old Pod but receives a different UID. It can also be scheduled to a different node. The Pod's execution context includes Linux namespaces and cgroups.

For an eBPF system this gives at least three identity layers:

1. **Logical workload identity.** Deployment, DaemonSet, StatefulSet member, service role, tenant, selector, or another operator-facing intent.
2. **Orchestrator generation.** A particular Pod UID, sandbox, container attempt, or rollout generation.
3. **Kernel target identity.** The cgroup, namespace, network interface, task, socket, or other object to which a BPF program is actually attached.

A production controller normally needs all three. If it stores only layer 3, it cannot explain whether the target still corresponds to the intended workload. If it stores only layer 1, it cannot prove where the program executes. If it stores layers 1 and 3 but not the generation transition, it can miss a replacement window.

The distinction matters even for pure observability. A short gap may be acceptable, but it should be measured as a gap rather than silently counted as continuous coverage.

## "Attach higher and filter" is a real baseline, not a universal solution

There is a strong simple alternative: avoid attaching to short-lived objects. Attach at a longer-lived ancestor or node-global hook, then classify events inside the BPF program using cgroup IDs, namespace information, marks, addresses, or map state.

This design can be excellent. It reduces attachment churn and can make replacement almost invisible to the hook itself.

But it moves the same continuity problem into classification state:

```text
node-global hook
    -> observe event
    -> resolve kernel identity to logical workload
    -> look up current workload generation
    -> record / act
```

Now the dangerous transition is not "new link missing"; it is "new target identity not yet mapped" or "reused identity still mapped to the old generation."

The baseline also has different cost and isolation properties. A broad hook sees events for workloads that are not relevant to the application, increases map and classification pressure, and can enlarge the blast radius of mistakes. Some BPF program types are naturally target-specific and cannot simply be lifted to one global hook.

So the research question is not whether every agent should attach locally. It is whether **local attachment and broad-hook filtering can share an explicit continuity contract** instead of each inventing ad hoc reconciliation.

## Where current work is still weak

### A live link does not prove that the intended workload is covered

BPF object discovery can tell an operator that a link exists and expose link-specific kernel information. Orchestrators can tell the operator which Pod or container generation should exist. What is missing is a standard evidence object connecting the two statements.

A useful record would say:

```text
intent: workload selector / generation
orchestrator: pod UID / container attempt
target: kind + kernel identity + node
link: link ID + program identity
resolved_at: monotonic generation
state: prepared | active | retiring | stale
```

The material test is a replacement storm: continuously recreate workloads while independently checking the kernel targets and observed events. If the controller ever reports "covered" while the current target has no matching active attachment or classification entry, the evidence model is insufficient.

### Replacement is a transition, but most attachment APIs expose endpoints

Kernel attachment APIs correctly operate on concrete targets. Kubernetes controllers correctly reconcile desired objects. The missing layer is the transition protocol between old and new target generations.

For monitoring, "stop seeing old, discover new, attach new" may be acceptable. For stronger continuity requirements it creates an observable gap. "Attach new, then retire old" is better, but only if the new target can be discovered and prepared before the application considers the new generation fully ready.

The gap is therefore not another attach syscall. It is a coordination boundary between workload lifecycle and BPF attachment readiness.

A discriminating experiment should inject delay and failure into every phase: target discovery, BPF load, link creation, map initialization, orchestrator watch delivery, and old-target retirement. Measure uncovered time and incorrect overlap, not just reconciliation latency.

### Kernel identifiers and logical identities have different reuse rules

Logical names can intentionally be reused. Kernel identifiers can also be recycled after an object disappears. A controller that caches only `ifindex`, cgroup path, PID, or another local identifier can therefore mistake a new object for an old one unless it binds the identifier to a stronger generation context.

This is a general systems problem: an identifier is only meaningful with the lifetime in which uniqueness is guaranteed.

A useful benchmark should force rapid create/destroy/recreate cycles and attempt identifier reuse. The controller should prove that stale state cannot become valid for a new workload merely because one numeric or pathname identifier reappears.

### There is no common correctness metric for attachment continuity

Most systems can report load success, attach success, or controller reconciliation time. Those are operational metrics, but they do not directly measure the property an instrumentation user cares about: "which operations from which workload generation were actually covered by which BPF generation?"

Without a ground-truth workload/target timeline, two agents can both claim successful reconciliation while having very different blind windows.

The missing artifact is a benchmark and trace schema that treats target replacement as a first-class fault.

## Promising directions with academic and production value

### Direction 1: target-generation receipts for every logical attachment

**Gap.** Controllers know the workload intent and kernel attachment APIs know the concrete target, but the mapping between them is usually implicit and hard to audit after a replacement.

**Mechanism.** Represent each resolved attachment as a target-generation receipt. The receipt binds:

- a stable logical selector and application generation;
- orchestrator identity such as Pod UID and container attempt;
- target type and node;
- a target-specific kernel identity;
- program/link identity and creation result;
- the predecessor generation, if any.

Target-specific identity should use the strongest practical tuple instead of a naked local number. For a network device, for example, the receipt can bind interface identity to its network namespace and observed generation rather than trusting `ifindex` globally. For a cgroup, the controller can retain the opened target and independently record a kernel-visible cgroup identity plus orchestrator ownership.

The receipt is evidence, not a new kernel primitive. The target fd and kernel remain authoritative for attachment.

**Delta from related work.** BPF links make attachment lifetime explicit. Kubernetes UIDs make orchestrator object lifetime explicit. The proposed receipt connects those two lifetime domains and records generation transitions.

**Artifact.** A small libbpf-side identity library, a target resolver for cgroup/netns/netdevice attachments, and a machine-readable receipt format inspectable with a `bpftool`-style command.

**Evaluation.** Recreate Pods, containers, cgroups, namespaces, and veth pairs under high churn. Compare path-only, numeric-ID-only, link-only, and generation-bound tracking. Measure false continuity, stale-object matches, time to diagnosis, receipt size, and resolver overhead.

**Academic value.** The general question is how to compose identities whose uniqueness guarantees live in different lifetime domains.

**Production value.** An operator can answer "which workload generation did this link actually cover?" after a rollout or incident.

**Failure condition.** If existing link metadata plus orchestrator state already reconstruct the mapping reliably and cheaply under churn, a dedicated receipt format adds little value.

### Direction 2: replacement-aware two-generation reconciliation

**Gap.** Ordinary controllers reconcile eventual state, but stronger continuity requires a defined handoff between target generations.

**Mechanism.** Model replacement as a two-generation transaction:

```text
resolve G(next)
    -> create/initialize attachment for G(next)
    -> verify target + program + workload generation
    -> mark G(next) attachment-ready
    -> admit normal workload traffic/measurement
    -> retire G(prev)
```

The controller keeps the old generation active while the new one is prepared when the hook permits overlap. If the hook cannot overlap, the controller explicitly records the uncovered interval rather than hiding it.

For Kubernetes, an instrumentation agent could expose a readiness signal that lets the platform delay normal service readiness until the required attachment receipt is active. The exact integration is workload-dependent; the important property is that workload readiness and attachment readiness become one protocol when continuity matters.

**Delta from related work.** This is not the August transactional eBPF program upgrade problem, where one logical target remains and program/state generations change. Here the BPF program can stay identical while the **kernel target itself changes**.

**Artifact.** A reconciliation library with state-machine persistence, Kubernetes integration, fault injection, and target adapters for cgroup and network-device hooks.

**Evaluation.** Compare naive watch-and-attach, periodic polling, broad-hook filtering, and two-generation reconciliation during rollouts, container restarts, node drains, CNI recreation, and controller restarts. Measure uncovered operations, duplicate observation, replacement-to-ready latency, API load, and recovery from lost watch events.

**Academic value.** The mechanism studies continuity across resource replacement when the kernel API and orchestrator expose different transaction boundaries.

**Production value.** Observability and networking agents can turn "eventual reattachment" into an explicit SLO with a measurable failure mode.

**Failure condition.** If broad stable hooks plus classification state eliminate attachment gaps with materially lower complexity and cost for nearly all relevant program types, target-specific transactional reconciliation should remain a niche design.

### Direction 3: a coverage-witness benchmark for ephemeral targets

**Gap.** Attach success and controller latency do not reveal which workload operations escaped observation during target churn.

**Mechanism.** Build a ground-truth harness that gives every workload generation and test operation a monotonically ordered identity. In parallel, record target-generation receipts and BPF observations. The checker then asks:

```text
for every test operation O:
    expected logical workload generation
    expected target generation
    observed BPF attachment/classification generation
    sample or result produced
```

Fault injection destroys and recreates cgroups, network namespaces, veth devices, Pods, and nodes; delays orchestration events; restarts the controller; and pressures identifier reuse. The primary result is not "reattached in 40 ms" but a set of operations that were uncovered, multiply covered, or attributed to the wrong generation.

**Delta from related work.** Existing BPF selftests establish kernel API correctness for attachment mechanisms. This benchmark evaluates end-to-end continuity between logical orchestration identity and kernel target identity.

**Artifact.** A reproducible Kubernetes/Linux testbed, event trace schema, checker, and corpus of replacement faults.

**Evaluation.** Run several real agents or prototype instrumentation paths under identical churn. Report uncovered-operation rate, maximum blind interval, stale-generation rate, false attribution, controller CPU/API cost, and workload latency added by readiness coordination. Include an observability-only workload where accepting bounded loss should beat the stricter mechanism.

**Academic value.** It provides a correctness metric and workload for dynamic instrumentation systems whose targets are not stable.

**Production value.** Vendors and operators can test whether a claimed "persistent attachment" survives realistic workload replacement rather than only controller restart.

**Failure condition.** If uncovered operations are vanishingly rare and have no measurable consequence across realistic churn, stronger continuity machinery may not justify its operational cost.

## What a production controller should distinguish today

Even without a new kernel API, a controller can avoid the most dangerous category error by tracking separate states:

```text
program loaded
link object alive
old target alive
logical workload desired
current target resolved
current target attached
current workload generation active
```

Those are not synonyms.

For a controller restart, a pinned link may let the new controller rediscover and adopt an existing attachment. That is an ownership-recovery problem.

For a workload replacement, the controller must decide whether the logical intent should follow to a different kernel target. That is a target-reconciliation problem.

For a host reboot, the kernel object graph is rebuilt. That is a durability and reconstruction problem.

Conflating the three makes a system look more persistent than it is.

## What would change this conclusion?

The strongest counterexample would be evidence that production eBPF agents overwhelmingly attach at stable ancestor or node-global hooks and that target replacement only updates ordinary classification state with no distinct continuity failure. If broad hooks cover the important observability and networking cases with bounded cost and clear identity semantics, a general target-rebinding protocol would be unnecessary.

The conclusion would also weaken if current kernel link types already exposed enough durable target-generation metadata for controllers to detect every relevant replacement without additional receipts, or if orchestrators guaranteed that a workload could not become active until all required node-local instrumentation had independently reconciled.

Finally, strict continuity is not always the right objective. A profiler can rationally accept a short blind interval if eliminating it requires delaying workload readiness. The mechanism should therefore expose the gap and its cost rather than impose one universal policy.

The narrow conclusion is still useful: **a persistent BPF link proves persistence of an attachment object, not persistence of the logical workload-to-target mapping.** Any system that promises observability or networking continuity across workload replacement needs evidence for that mapping and a defined transition between target generations.

## Sources

- [Linux kernel documentation: eBPF syscall API](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel source: cgroup BPF lifecycle handling](https://github.com/torvalds/linux/blob/master/kernel/bpf/cgroup.c)
- [Libbpf API: attach a program to a cgroup](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_cgroup/)
- [Libbpf API: TCX attachment by interface index](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_tcx/)
- [Libbpf API: pin a BPF link](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_link__pin/)
- [Kubernetes documentation: Pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Kubernetes documentation: Pods and their Linux isolation context](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Eunomia Daily Report: Can an eBPF Loader Treat a kfunc as Just Present or Missing?](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/)
- [Eunomia Daily Report: Can an eBPF Loader Trust the Kernel Version?](https://eunomia.dev/research/ebpf-kernel-capability-evidence/)
