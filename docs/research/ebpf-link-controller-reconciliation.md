---
date: 2026-09-21
slug: ebpf-link-controller-reconciliation
title: "Can an eBPF Link Outlive Its Controller Safely?"
description: "Pinned BPF links can survive controller exits, but restart logic can lose ownership and duplicate attachments. This report develops reconciliation contracts."
tags:
  - Daily Report
  - eBPF
  - Linux
  - Operations
  - Kubernetes
research_question: "What ownership and reconciliation contract lets a restarted eBPF controller distinguish the live links it should adopt, replace, or detach without duplicating attachments or deleting another controller's state?"
source_cutoff: 2026-09-21
status: daily-report
---

# Can an eBPF Link Outlive Its Controller Safely?

A BPF link has one operational property that is easy to celebrate and easy to misuse: it can outlive the userspace process that created it.

That is exactly what pinning is for. The current libbpf API documents that [`bpf_link__pin()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.h) increments the link's reference count so the link can remain loaded after the creating process exits. The kernel UAPI likewise treats BPF objects as reference-counted objects that disappear only after file descriptors, pins, and other references are gone.

For a short-lived command, this is convenient. For a long-running controller, it creates a harder question. After a crash, upgrade, container restart, or partial cleanup, the new controller process has to decide whether an already-live kernel link is:

- the exact attachment it still wants;
- an older generation that should be replaced;
- an orphan left by its previous instance;
- a link owned by another controller;
- or an attachment whose userspace metadata has disappeared while the kernel object is still valid.

The kernel can tell userspace which links exist. It cannot tell a restarted application what those links *mean* to that application's desired state.

A real Cilium regression makes the distinction concrete. In [Cilium issue #46065](https://github.com/cilium/cilium/issues/46065), an agent restart could leave a `cgroup_inet_sock_release` BPF link alive even when the pin file used by restart logic was absent. Repeated restarts could accumulate duplicate links. The reported agent still passed readiness while endpoints remained stuck in restoration and policy-denied traffic appeared. The issue was closed as stale on September 21, 2026, and an earlier proposed cleanup PR, [#46389](https://github.com/cilium/cilium/pull/46389), was closed without merge.

This is not a claim that BPF links are broken. The opposite is true: the kernel kept the attachment alive because some valid lifetime reference still existed. The failure was in reconciling kernel reality with controller intent.

The systems question is therefore broader than one Cilium bug: **what contract should an eBPF control plane use when attachment lifetime is intentionally independent from process lifetime?**

<!-- more -->

## A pin is a lifetime reference, not an ownership proof

It is tempting to use a bpffs path as the controller's source of truth:

```text
pin exists     -> attachment exists and is mine
pin missing    -> attachment is absent
```

That model is too strong.

The BPF object model is reference counted. A pin is one reference. An open file descriptor is another. Some attach mechanisms and object relationships can retain additional references. Removing one userspace-visible name does not prove the underlying link has already disappeared.

The same asymmetry appears in the opposite direction. A pin can exist while the controller's desired state has changed. The kernel does not know that a deployment generation was rolled back, a Kubernetes object was deleted, or another controller instance has taken responsibility for the hook. It only knows that an object remains referenced.

This makes bpffs a useful durable handle, but not a complete ownership ledger.

The kernel exposes the pieces needed for inspection. The current BPF syscall UAPI includes [`BPF_LINK_GET_NEXT_ID`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h), `BPF_LINK_GET_FD_BY_ID`, and `BPF_OBJ_GET_INFO_BY_FD`. Libbpf exposes corresponding helpers such as `bpf_link_get_next_id()` and `bpf_link_get_info_by_fd()`. `BPF_PROG_QUERY` can also report link IDs for supported attachment points.

Those interfaces answer questions such as:

```text
Which BPF links exist?
Which program does this link reference?
What type of link is it?
What attachment target information does this link expose?
```

They do not answer:

```text
Which deployment generation created it?
Which controller is responsible for it now?
Was this link committed, or was the controller killed halfway through setup?
Is a missing pin evidence of deletion or only missing bookkeeping?
May I safely detach it without disrupting another owner?
```

That missing meaning is where restart bugs appear.

## The Cilium failure shows why existence and ownership are different states

The useful detail in the Cilium report is not simply that an old link survived. Survival was expected behavior for a persistent attachment.

The failure sequence was closer to this:

```text
old controller
    creates/retains BPF link
    |
    | controller restarts and userspace bookkeeping changes
    v
new controller
    does not find the expected pin
    assumes no owned link is present
    creates or tries to detach through a different API path
    |
    v
kernel
    still has the previous bpf_link
```

The issue reporter observed old links accumulating across sequential restarts. A manual `bpftool link detach id ...` removed the stale attachment and let the next reconciliation attempt complete.

The proposed Cilium PR is also instructive. Its description explicitly notes the problematic state: the pin file can be gone while the `bpf_link` itself remains in the kernel. The patch attempted to query the cgroup for link-attached programs and detach orphan links by `LinkID` before creating a new one.

That is a reasonable local repair, but it exposes the larger design problem. "No pin, therefore detach every matching link" is only safe if the controller can prove that every matching link belongs to it. On a shared hook, in a multi-controller environment, during a rolling upgrade, or after partial migration between attachment mechanisms, that proof can be nontrivial.

A robust controller therefore needs at least three states, not two:

```text
owned and desired
owned but undesired
present but ownership is uncertain
```

The third state should not silently collapse into either "delete" or "ignore."

## This is not the same problem as atomic eBPF upgrade

The earlier report on [stateful eBPF transactional upgrades](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/) asks how a multi-object application moves from one committed generation to another without exposing a half-upgraded datapath.

Restart reconciliation begins from a different failure model. The controller may have lost the in-memory transaction state entirely. It wakes up after the fact and has to reconstruct what the kernel is already running.

Likewise, [kernel capability admission](https://eunomia.dev/research/ebpf-kernel-capability-evidence/) asks whether an artifact can load on a target kernel, while [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/) asks whether admitted behavior still means the same thing after a kernel change.

Here the kernel can be unchanged and the program can be perfectly valid. The failure is control-plane identity:

```text
actual attachment set != controller's reconstructed ownership model
```

This boundary matters because BPF links are increasingly the normal way to represent persistent, updateable attachment state. Making links easier to keep alive increases the value of a principled restart protocol.

## Where current work is still weak

Linux provides strong object-lifetime primitives, but controller-level lifecycle semantics are mostly application policy.

First, a link ID is an inspection handle, not a durable application identity. It identifies one live kernel object. It does not encode a deployment, tenant, controller generation, or desired-state key.

Second, a bpffs path is a durable name but not a complete proof of kernel state. A controller can lose or remove the path while another reference keeps the object alive. Conversely, the path can remain after the desired attachment has changed.

Third, matching only by program identity is often insufficient. Two controllers can intentionally attach the same program image to the same target with different operational ownership. A program tag or object digest answers "same code?" rather than "same responsibility?"

Fourth, health checks frequently observe the controller process, not reconciliation convergence. The Cilium report is especially useful here because readiness and status remained healthy while the datapath orchestrator repeatedly failed and endpoints stayed in restoration.

Finally, current test suites usually test successful create, attach, update, and detach operations. They less often kill the controller after every side effect and ask whether the next process reconstructs exactly one desired attachment without deleting foreign state.

The missing abstraction is not another attach API. It is an ownership-aware reconciliation contract over existing kernel objects.

## Research direction 1: generation-scoped attachment receipts

The first direction is to give each intended attachment a durable identity that is separate from its current kernel link ID.

**Gap.** A restarted controller can enumerate links, but it needs to map those links back to a logical desired-state record. Pin paths, program tags, and link IDs each encode only part of that mapping.

**Mechanism.** Define an attachment receipt with a stable logical key and generation:

```text
attachment_key: socketlb/cgroup-inet-sock-release
owner_scope: node-agent
controller_generation: 417
kernel_boot_id: ...
target_identity: cgroup + attach_type + target generation
program_identity: object digest + program name/tag
link_id: 1851
pin_path: /sys/fs/bpf/.../link
state: prepared | committed | retiring
```

The exact storage can start entirely in userspace. A controller can write the receipt beside its bpffs objects or into a small pinned metadata map. The important rule is that `link_id` and `pin_path` are evidence attached to the logical key, not the logical key itself.

The `kernel_boot_id` prevents an old userspace receipt from accidentally referring to a numerically reused kernel object after reboot. A target generation is equally important for targets that can be destroyed and recreated while keeping a familiar userspace name.

On restart, the controller reads receipts, enumerates live links through `BPF_LINK_GET_NEXT_ID` or target-specific queries, reads `bpf_link_info`, and attempts a join:

```text
desired attachment
    <-> durable receipt
    <-> live kernel link
```

Any missing edge becomes an explicit recovery case rather than an assumption.

A later kernel extension could make this cheaper by exposing an immutable opaque reconciliation tag in generic link info. Such a tag would be metadata only. It would not grant authority or replace verifier checks. The first prototype does not need that UAPI change.

**Delta from current practice.** This is stronger than "remember the pin path" and more specific than a general deployment manifest. It names the control-plane identity that must survive process loss while kernel link IDs remain runtime facts.

**Prototype.** Add receipt generation and restart reconciliation to a small libbpf controller that manages cgroup and TCX links. Run the controller both with normal bpffs pins and with deliberately removed pins while another kernel reference keeps the link alive.

**Evaluation.** Measure correct adoption rate, false orphan classification, and accidental foreign-link deletion across crash/restart cases. Repeat with two controllers intentionally attaching identical program images to the same supported multi-attach hook.

**Academic value.** This turns BPF attachment recovery into an identity-reconciliation problem with a measurable invariant rather than ad hoc cleanup code.

**Production value.** Operators get an inspectable answer to "why did this controller adopt or detach this link?" and a stable key for alerts, logs, and rollback tooling.

**Failure condition.** If pin paths plus existing link metadata uniquely and reliably recover ownership across realistic crashes, multi-controller hooks, and target recreation, the extra receipt layer is unnecessary.

## Research direction 2: quarantine ambiguous links before detach

The second direction is to make uncertainty a first-class reconciliation state.

**Gap.** A restart routine often wants to converge quickly: delete old objects and recreate the desired set. That is dangerous when ownership evidence is incomplete. Leaving a duplicate link can be wrong, but detaching a foreign or still-authoritative link can be worse.

**Mechanism.** Use a reconciliation state machine that classifies each observed link before mutation:

```text
EXACT        receipt + target + program + generation all match
STALE        owned receipt proves an older generation
MISSING      desired receipt has no live link
FOREIGN      evidence points to another owner
AMBIGUOUS    live link exists but ownership proof is incomplete
```

Only `STALE` links are immediately eligible for automatic detach. `EXACT` links are adopted. `MISSING` attachments are recreated. `FOREIGN` links are preserved. `AMBIGUOUS` links enter a quarantine path that gathers more evidence before mutation.

For a hook where duplicate execution is unsafe, quarantine should also affect readiness. The controller should not report datapath convergence while it knows that an ambiguous extra attachment may still run.

A practical recovery sequence could be:

1. snapshot the target's current link set and query revision when available;
2. classify links without mutation;
3. create any missing candidate attachments;
4. verify the desired program is active on the intended target;
5. detach only links whose stale ownership is proven;
6. query again and require a stable desired set before declaring ready.

This is deliberately not a general multi-object upgrade transaction. It is a restart protocol for reconstructing ownership after the old process is already gone.

**Delta from current practice.** Cleanup logic often treats absence of a pin or mismatch in one lookup as enough evidence to recreate or delete. Quarantine requires positive evidence for destructive cleanup and exposes unresolved ambiguity to health status.

**Prototype.** Implement the state machine around cgroup links first because `BPF_PROG_QUERY` can expose attached programs and link IDs. Extend to TCX and tracing links where target metadata differs.

**Evaluation.** Inject missing pins, stale receipts, duplicate links, controller overlap, reused cgroup paths, and concurrent attach/update operations. The safety metric is accidental deletion of a valid foreign link. The liveness metric is time to converge to exactly the desired set.

**Academic value.** The mechanism gives control-plane reconciliation a safety/liveness tradeoff that can be tested under partial knowledge.

**Production value.** A controller can fail closed on ownership ambiguity instead of oscillating between duplicate attachments and destructive cleanup.

**Failure condition.** If ambiguity is so common that quarantine routinely prevents recovery, or existing kernel metadata cannot disambiguate ownership without operationally unacceptable delays, the model needs a stronger kernel-visible ownership primitive.

## Research direction 3: crash-fuzz the link lifecycle, not only the datapath

The third direction is an evaluation system for the control plane itself.

**Gap.** A datapath can pass packet tests while the controller's recovery protocol is broken. The failure only appears after the process dies between two lifecycle operations.

**Mechanism.** Build a deterministic crash-fuzz harness that places failure points after every relevant side effect:

```text
create program
create link
pin link
write receipt
mark committed
unlink pin
update program
start retirement
detach old link
delete receipt
```

At each point, kill the controller without cleanup, restart it, and compare three views:

- desired attachment state from configuration;
- actual link state from kernel enumeration and target queries;
- observed hook execution, including the number and order of programs that actually ran.

The last view matters. Two links can look superficially similar in metadata while duplicate execution changes policy, counters, socket state, or packet mutation.

The harness should also run with two controller instances and inject overlapping restarts. For cgroup multi-attach, it should distinguish a legitimate multi-program configuration from accidental duplicate ownership.

A convergence oracle can be simple:

```text
for every logical attachment key:
    exactly one intended owner-generation is active

for every active owned link:
    a committed desired-state key explains it

for every foreign link:
    reconciliation never detaches it
```

Readiness should become part of the test. A controller that reports healthy before these invariants hold fails even if it eventually converges.

**Delta from ordinary BPF selftests.** Kernel selftests are good at validating BPF object and attach semantics. This harness tests the userspace lifecycle protocol against real kernel persistence and controller crashes.

**Prototype.** Start with a tiny libbpf daemon in a VM, one cgroup target, and a counter program whose execution count exposes duplicates. Add TCX and tracing targets after the oracle is stable.

**Evaluation.** Compare three controllers: pin-path-only recovery, enumerate-and-delete recovery, and receipt-plus-quarantine recovery. Score false detach, duplicate execution time, recovery latency, and incorrect readiness under thousands of injected crash points.

**Academic value.** It creates a reproducible benchmark for persistent-kernel-object reconciliation, a problem that also appears in networking, storage, and device control planes.

**Production value.** The same harness can become an upgrade gate for agents that manage persistent BPF state.

**Failure condition.** If lifecycle bugs are already caught by existing integration tests with equivalent crash-point coverage and kernel-state oracles, a separate harness would duplicate existing infrastructure.

## Practical deployment guidance today

For current production controllers, the safest rule is simple: **do not treat a missing pin as proof that no link exists.**

On restart, inspect the actual attachment point as well as bpffs. Use `bpftool link show`, `BPF_LINK_GET_NEXT_ID`, `BPF_LINK_GET_FD_BY_ID`, `bpf_link_info`, and target-specific query APIs as appropriate. Keep enough durable metadata to explain which live link belongs to which desired attachment.

Do not automatically detach every matching program on a shared hook unless ownership is provable. Program identity and ownership identity are different things.

Make reconciliation convergence part of health. A process can be alive while the datapath is still half-restored or duplicated. If the controller knows that attachment state is ambiguous, surface that state rather than reporting a generic healthy status.

Finally, test restarts as faults, not only as graceful shutdowns. Kill the controller after link creation, after pinning, after unlinking, and during replacement. The most useful test is the one where userspace bookkeeping and kernel lifetime disagree.

For readers implementing these paths, the existing [detach tutorial](https://eunomia.dev/tutorials/28-detach/) is a useful low-level companion, but production control planes need a stronger ownership protocol around those primitives.

## What would change this conclusion?

The main conclusion is that persistent BPF links need controller-level ownership and reconciliation semantics in addition to kernel-level lifetime semantics.

That conclusion would weaken if one of three things became true.

First, if generic BPF link metadata gained a stable, sufficiently expressive owner identity that real controllers could use directly across crashes and rolling restarts, much of the userspace receipt machinery could disappear.

Second, if production evidence showed that pin paths plus existing target queries already recover ownership without ambiguity across shared hooks, process crashes, and target recreation, the problem would be smaller than the Cilium failure suggests.

Third, if broad crash-injection testing found that duplicate or foreign-link failure modes are rare and always harmless because supported hooks are idempotent under repeated attachment, the safety case for quarantine would be weak.

None of those statements is established today. Linux gives eBPF a durable link object and good introspection primitives. What remains underspecified is the control-plane contract that turns those kernel objects back into one correct desired attachment set after the controller that created them is gone.