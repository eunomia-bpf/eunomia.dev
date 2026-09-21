---
date: 2026-09-21
slug: ebpf-controller-restart-link-adoption
title: "Can an eBPF Control Plane Restart Without Duplicating the Datapath?"
description: "Pinned BPF links can outlive their loader, but a restarted controller can lose the identity needed to adopt them. This report develops restart receipts, kernel-first reconciliation, and adversarial restart testing."
tags:
  - Daily Report
  - eBPF
  - Linux
  - Lifecycle
  - libbpf
  - bpffs
research_question: "How should an eBPF controller recover after its userspace process restarts while the kernel, pinned links, maps, and attach targets remain live, without duplicating attachments or silently adopting the wrong datapath state?"
source_cutoff: 2026-09-21
status: daily-report
---

# Can an eBPF Control Plane Restart Without Duplicating the Datapath?

Suppose a networking daemon loads an eBPF program, attaches it with a BPF link, pins that link in `bpffs`, and then crashes.

The kernel does exactly what the daemon asked. The pinned link keeps a reference to the attachment, so the program can keep running after the process that created it is gone. The replacement daemon starts a few seconds later and tries to recover.

Now the difficult question begins: **how does the new process prove that the attachment it sees is the one it should adopt?**

A pathname such as `/sys/fs/bpf/foo/ingress` looks like an answer, but it is only one view of kernel state. The new process may run in another mount namespace. The path may resolve to a different `bpffs` instance. The pin may have been deleted while the link remains referenced elsewhere. A target interface or cgroup may have been recreated. A legacy attachment may coexist with a link-based attachment. Another controller instance may be recovering at the same time.

A recent Cilium report gives a particularly sharp example. In [Cilium issue #47847](https://github.com/cilium/cilium/issues/47847), an additional `bpffs` mount was placed over `/sys/fs/bpf` while Cilium was running. After the agent restarted, the new process could no longer see the pins created through the old mount view and created new TCX attachments. `bpftool` then showed duplicate ingress and egress TCX programs on the same devices. The reporter later identified the shadow mount as the reproduction condition and closed the issue as an environment problem rather than a Cilium bug.

That distinction is useful. The failure did not require BPF link lifetime to be broken. It required two individually reasonable mechanisms to disagree about identity:

1. the kernel still had live attached links;
2. the restarted userspace controller used a different pathname view to decide whether those links existed.

This report argues that restart recovery needs a stronger contract than "pin it and reopen the same path." A restartable eBPF control plane should reconcile **kernel-visible attachment state, pin-namespace identity, attach-target identity, and controller generation** before it creates, updates, or removes anything.

The scope is deliberately narrow. This is not the same problem as [admitting an object on a particular kernel](https://eunomia.dev/research/ebpf-kernel-capability-evidence/) or [proving that an object keeps its behavior across a kernel upgrade](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/). It is also different from [transactionally upgrading a stateful eBPF application](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/), where the intended program generation changes. Here the intended datapath may be completely unchanged. Only the userspace controller died and came back while the kernel object graph remained live.

<!-- more -->

## Pinning solves lifetime, not recovery identity

Linux gives BPF objects reference-counted lifetimes. The current [eBPF syscall documentation](https://docs.kernel.org/userspace-api/ebpf/syscall.html) says that `BPF_OBJ_PIN` adds a filesystem reference to a BPF object, preventing deallocation when the original file descriptor is closed. `BPF_OBJ_GET` opens a new file descriptor for a pinned object. The same API also exposes `BPF_LINK_GET_NEXT_ID`, `BPF_LINK_GET_FD_BY_ID`, and `BPF_OBJ_GET_INFO_BY_FD`, which allow userspace to enumerate and inspect live link objects independently of a remembered pin pathname.

For attachments, `BPF_LINK_CREATE` returns a file descriptor that manages the link, and `BPF_LINK_UPDATE` can replace the associated program without dropping and recreating the attachment. libbpf's [`bpf_link__pin()`](https://libbpf.readthedocs.io/en/latest/api.html) documentation makes the intended process-lifetime behavior explicit: pinning increments the link reference count so the link can remain loaded after the process that created it exits.

This is a strong primitive. It is not a complete restart protocol.

Pinning answers:

> Should this kernel object remain referenced after my file descriptor closes?

Restart recovery asks several additional questions:

> Is this live object mine?
>
> Is it attached to the target I think it is attached to?
>
> Is it the generation and policy configuration I intend to run?
>
> Is this pathname still looking at the same `bpffs` instance that the previous controller used?
>
> If the pin is missing, is the attachment absent, or merely invisible through this namespace?

Those are different questions.

## A production loader already contains an implicit recovery protocol

Current Cilium source makes the control-flow boundary concrete. In [`pkg/datapath/loader/tcx.go`](https://github.com/cilium/cilium/blob/main/pkg/datapath/loader/tcx.go), the TCX loader first calls an update path using the expected pinned link. If the existing link is found, it updates the program. If the pin is missing or the link is defunct, it falls back to creating a new TCX link and pins that new link under the expected name.

The code also documents the key lifetime fact: after a TCX link is successfully pinned, closing the userspace link handle does not detach the program.

This design is sensible when the pathname is a faithful index of the controller's live links. Issue #47847 shows the boundary where that assumption can fail. A second `bpffs` mount can make the path appear empty even though the original link is still attached in the kernel. "Pin not found" then means "not visible through this mount," not necessarily "no managed attachment exists."

That is a general systems pattern. A controller reconstructs ownership through one metadata plane while the effect itself lives in another plane. If the metadata plane is lost, remounted, stale, or only partially restored, blindly recreating effects can duplicate them.

The kernel exposes enough information to do better than a pathname-only test. `BPF_LINK_GET_NEXT_ID` and `BPF_LINK_GET_FD_BY_ID` can enumerate live links, `BPF_OBJ_GET_INFO_BY_FD` can inspect them, and `BPF_PROG_QUERY` can query programs associated with several attach targets. Tooling such as `bpftool link show` already uses kernel-visible object state for inspection.

The missing piece is a standard rule for turning those observations into **safe adoption**.

## The same path can name a different BPF filesystem

The Cilium reproduction is valuable because it is not an exotic kernel race. It uses ordinary Linux mount behavior.

A pathname is resolved through the calling process's mount namespace. Mounting a second `bpffs` instance on `/sys/fs/bpf` changes what later path lookups at that location see. The previous mount and its pinned objects can remain alive and reachable through other references even though the new controller's `/sys/fs/bpf/...` path resolves somewhere else.

That means the following startup logic is unsafe as a general invariant:

```text
if expected_pin_exists():
    update_it()
else:
    attach_new_link()
```

The safe interpretation of a missing pin is weaker:

```text
expected pin not visible in this pathname view
```

Before creating a replacement effect, the controller has to ask whether a matching attachment already exists in kernel state.

This is similar to storage recovery. A missing directory entry does not prove that an external effect never happened. Recovery must reconcile the durable effect with the journal or intent record.

## A link ID is useful evidence, but it is not enough by itself

One tempting fix is to record the BPF link ID and reopen it after restart with `BPF_LINK_GET_FD_BY_ID`.

That improves recovery, but one integer still does not define the controller's intended datapath.

A robust identity needs at least four layers.

### 1. Attachment identity

The controller needs to identify the hook and target, not only the link object. Depending on the program type, that may involve:

- attach type and hook kind;
- network namespace plus interface identity;
- cgroup identity;
- tracing target;
- TCX ordering or other multi-program position;
- expected multiplicity at the hook.

Interface names and cgroup path strings are not always stable identities. An interface can be deleted and recreated under the same name with another ifindex. A cgroup path can refer to a newly created cgroup after the old one disappears. The recovery contract should use the strongest kernel identity available for that hook and explicitly record where the kernel API does not expose enough identity.

### 2. Program identity

The controller should verify which program the link references. Useful evidence can include program ID, program tag, BTF-related metadata, expected program name, and an artifact hash retained by the controller.

A program name alone is weak because names are short labels, not cryptographic identities. A kernel object ID alone is also only a live-kernel identifier. The controller needs to connect the live object back to the artifact and configuration generation that created it.

### 3. State identity

A datapath is rarely one program. It may depend on pinned maps, map-in-map relationships, configuration maps, policy generations, tail-call targets, or shared state also used by another program generation.

Adopting a link while silently pairing it with the wrong map generation can be worse than creating a duplicate attachment. The program executes, but against state with different ownership or semantics.

### 4. Pin-namespace identity

The recovery system should record enough information to detect that `/sys/fs/bpf` is no longer the same filesystem instance or mount view used by the previous controller. The exact representation can be implementation-specific, but a pathname alone is insufficient.

The important policy is simple: **a changed pin namespace invalidates pathname-only absence claims.**

## Restart races turn recovery into a distributed ownership problem

A daemon restart often looks single-process in a diagram, but production systems can make it concurrent.

Kubernetes may start a replacement pod before every old helper process exits. A supervisor may retry aggressively. An operator may launch a diagnostic or repair process. A host agent may have a separate component managing some links. Two controller generations can therefore overlap.

If both run this logic:

```text
observe no expected pin
create link
pin link
```

then even a perfectly stable `bpffs` mount can produce duplicate effects when observation and creation are not serialized.

The deeper contract is not just object persistence. It is **single-writer ownership of reconciliation**.

A restarted controller needs a generation or lease concept outside the live link itself. Before mutating the datapath it should be able to prove either:

- it is the unique controller generation allowed to reconcile this attachment set; or
- its mutation is idempotent against the exact live kernel state it just observed.

For many hook types, the second option is difficult because attaching another valid program is a legal kernel operation. The kernel cannot infer that the new program is an accidental duplicate of an older controller generation.

## Real incident reports show why startup health is not enough

A second Cilium community report, [issue #46065](https://github.com/cilium/cilium/issues/46065), described `cgroup_inet_sock_release` links that remained visible across Cilium agent restarts and accumulated on a cgroup while the new agent entered a retry loop. The issue was ultimately closed without a maintained fix and should not be treated as a verified kernel defect. It is still useful as an operational observation: a controller can report broadly healthy process-level status while its datapath reconciliation is incomplete or ambiguous.

This suggests a readiness rule that is easy to miss:

> A restarted eBPF control plane should not become "ready" merely because it can load new programs. It should become ready only after it has reconciled the expected live attachment graph.

For a networking agent, the acceptance condition may include:

- exactly the expected effective attachments at each managed hook;
- no unexpected older managed generation still active;
- all required maps opened and schema-checked;
- program-to-map and link-to-program relationships matching the intended generation;
- no unresolved attachment whose ownership cannot be classified.

Unknown state should be visible as unknown. Turning it into "attach another copy and see whether traffic works" hides the evidence needed to debug the next restart.

## The unresolved gap: there is no first-class adoption contract

Linux provides the object primitives. Production loaders provide application-specific recovery logic. What is still weak is the layer between them.

### Pin paths are names, not ownership proofs

A pin lets another process obtain a reference to an object. It does not, by itself, prove which controller generation owns that object or whether the current pathname view is the same one used by the old process.

### Kernel enumeration is observational, not declarative

The kernel can enumerate links, programs, maps, and information about them. It does not know the application-level invariant "there must be exactly one `cil_from_netdev` ingress attachment for this controller generation and it must use these maps."

That invariant belongs to userspace.

### Hook types expose different recovery surfaces

Modern BPF links are easier to enumerate and manage than older attachment mechanisms, but real applications can use a mixture of links, legacy TC attachments, cgroup attachment APIs, perf-event based tracing, XDP, and subsystem-specific attachment models.

A general recovery layer needs hook-specific adapters while preserving one application-level ownership model.

### Reconciliation is rarely tested as a state-space problem

Most integration tests check clean start, clean stop, and perhaps one restart. The hard failures live in partial states:

- process dies after attach but before pin;
- pin succeeds but controller metadata is not committed;
- mount namespace changes before restart;
- only some links are visible;
- one target is recreated under the same human-readable name;
- two controllers recover concurrently;
- an existing link references an old program generation but current maps;
- legacy and link-based attachments coexist;
- update fails after the controller has taken new references.

That state space is where a restart contract should be evaluated.

## Research direction 1: create a restart receipt for the live BPF object graph

The first mechanism is a **restart receipt** written by the controller after a datapath generation becomes active.

The receipt is not a raw dump of every BPF object. It is the minimal ownership graph needed to decide whether a future process may adopt, update, quarantine, or recreate the deployment.

For each managed attachment, record evidence such as:

```text
controller_generation: 184
artifact_digest: sha256:...
policy_generation: 9271

pin_namespace:
  expected_bpffs_mount_identity: ...
  expected_pin: /sys/fs/bpf/app/eth0/ingress

attachment:
  kind: tcx/ingress
  target_identity: netns + ifindex + stable device evidence
  expected_multiplicity: 1
  expected_order: ...

link:
  observed_link_id: 314
  program_id: 9256
  link_info_digest: ...

program:
  program_tag: ...
  btf_id: ...
  artifact_section: ...

state:
  map_manifest_digest: ...
```

The receipt should live outside `bpffs` as well as refer to it. Otherwise a changed or shadowed BPF filesystem can hide both the object pin and the only record explaining what should exist.

The research question is how small this receipt can be while still preventing false adoption. Different hook types expose different target metadata, so the prototype should define a common core plus hook-specific identity fields.

### Evaluation

Build a test matrix across TCX, cgroup links, XDP or another network hook, and one tracing link type. For every restart, compare four policies:

1. pin-path-only recovery;
2. link-ID-only recovery;
3. kernel enumeration plus names;
4. receipt-based recovery with target, program, state, and mount identity.

Inject target recreation, changed mount namespaces, partial pin deletion, and stale controller metadata. Measure false adoption, unnecessary reattachment, unresolved ambiguity, and recovery latency.

This direction fails if the receipt cannot distinguish common stale-state cases without becoming a complete duplicate of kernel state.

## Research direction 2: make startup a kernel-first reconciliation transaction

The second mechanism changes controller order of operations.

A restart should begin from kernel ground truth, not from the assumption that the expected pin directory is complete.

A possible protocol is:

```text
1. acquire controller-generation ownership
2. inspect bpffs mount identity
3. enumerate/query relevant live attachments from the kernel
4. open FDs for candidate links/programs/maps before mutating them
5. match candidates against the restart receipt
6. classify each expected attachment:
      exact match -> adopt
      compatible old program -> update existing link
      absent with strong evidence -> create
      ambiguous / duplicate -> quarantine or fail closed
7. publish/repair pins for adopted objects in the current intended bpffs
8. verify the full effective attachment graph
9. mark datapath ready
10. release or garbage-collect only objects proven stale
```

The key safety rule is step 6: **"pin not found" is not enough evidence for "create."**

Where `BPF_LINK_UPDATE` is supported, updating an adopted link can preserve attachment continuity instead of detach-and-reattach. Where kernel query APIs can enumerate effective programs, the controller can detect an already-running effect even if the original pin is invisible.

The protocol also makes ambiguity explicit. If two live links both plausibly match one expected attachment and the controller cannot prove which one owns current state, automatically deleting one may be unsafe. A correct system can fail closed, mark the node degraded, or invoke a hook-specific repair policy instead of guessing.

### Evaluation

Kill the controller at every transition in the protocol and restart it. The invariant is stronger than "eventually becomes healthy":

- at most one intended effective attachment is active unless the hook contract explicitly allows more;
- no policy generation is paired with incompatible state;
- no active attachment is deleted without an ownership proof;
- after a bounded recovery interval, every managed object is classified as adopted, replaced, stale, or unknown.

This direction fails if kernel inspection cannot expose enough attachment identity for major production hook types. That failure would itself identify where new kernel introspection is needed.

## Research direction 3: build an adversarial restart benchmark for eBPF control planes

The third mechanism is an evaluation artifact rather than a new API.

A restart benchmark should treat BPF lifecycle recovery like a crash-consistency test. Instead of only restarting the daemon at a clean point, it injects failures around every externally visible transition.

Useful perturbations include:

- `SIGKILL` after link creation and before pinning;
- `SIGKILL` after pinning and before controller-state commit;
- overlaying or changing the `bpffs` mount before restart;
- restarting in a different mount namespace;
- deleting one pin while retaining another live reference;
- recreating an interface or cgroup under the same path/name;
- leaving legacy and link-based attachments together;
- racing two controller generations;
- forcing `BPF_LINK_UPDATE` failure;
- changing one map generation while leaving the old link active;
- injecting permission differences into the replacement process.

The benchmark needs kernel-side ground truth. Before and after each recovery attempt, enumerate links and programs, query effective attachment state where possible, and drive traffic or events through the hook to count actual executions.

The most useful metrics are not startup time alone:

- duplicate effective attachment count;
- orphan lifetime;
- wrong-generation execution count;
- policy discontinuity duration;
- false adoption rate;
- destructive cleanup of a still-owned object;
- time spent in unresolved state;
- percentage of injected crash points that converge without operator repair.

A production controller should be able to state its restart envelope: which crash points and namespace mutations it can recover from automatically, which it detects but refuses to repair, and which remain unsupported.

This direction fails if the benchmark only tests bookkeeping while missing whether packets, syscalls, or traced events are actually processed twice. The ground truth must include effect execution, not only object counts.

## A practical restart policy can be stricter today

A production implementation does not need a new kernel API to improve significantly.

Several rules are implementable with existing interfaces:

1. **Pin the attachment object when attachment survival is intended.** Pinning only maps or programs does not give the controller the same first-class handle over the attachment lifecycle.
2. **Treat the `bpffs` mount as configuration with identity, not merely a path.** Detect an unexpected mount change before interpreting missing pins.
3. **Query kernel attachment state before creating a replacement.** Use link enumeration, link info, and hook-specific query APIs where available.
4. **Adopt by evidence, not by name.** Match target identity, program identity, state generation, and controller generation.
5. **Use one reconciliation owner.** Prevent overlapping controller generations from independently deciding to recreate effects.
6. **Do not report readiness before reconciliation finishes.** Process health and datapath ownership are separate states.
7. **Keep unknown objects visible.** An unmanaged or ambiguous live link should produce an explicit diagnostic rather than be silently ignored.
8. **Prefer in-place link update when the intended attachment is already present and the hook supports it.** This avoids creating a second attachment just to change the program.
9. **Test mount and namespace failures deliberately.** A shadowed `bpffs` mount is not a theoretical edge once it has produced a real duplicate-TCX reproduction.

These rules make restart behavior less convenient in one sense: the controller may refuse to self-heal when it cannot prove ownership. That is preferable to a controller that "heals" by installing another policy program on top of an already-live one.

## The larger lesson is that persistence creates an adoption problem

Pinning is often described as a way to make BPF state survive a process exit. That is correct, but survival changes the control-plane problem.

Without persistence, process death destroys the last reference and cleanup is implicit. With persistence, the next process inherits a world containing effects it did not create itself. It needs a protocol for recognizing and adopting those effects.

This is the same transition seen in other systems. Persistent resources move correctness from object lifetime into recovery identity. A database needs transaction recovery, an orchestrator needs resource ownership and generation, and a restartable eBPF controller needs more than a directory of pins.

The kernel already provides strong pieces: reference-counted BPF objects, pins, link IDs, object-info queries, link update, and attachment query interfaces. The missing abstraction is a controller-level contract that binds those pieces into one statement:

> **These are the live kernel effects for generation G; these are the targets and state they belong to; this new controller has proved they match; and no second effect needs to be created.**

That contract would make eBPF lifecycle recovery much easier to reason about than today's mixture of pathname conventions, hook-specific cleanup, and best-effort startup reconciliation.

## What would change this conclusion?

Three findings would weaken the case for a stronger adoption contract.

First, if production controllers using persistent BPF links can demonstrate that stable mount setup plus pin-path reopening eliminates duplicate and orphan attachment failures across realistic restart, namespace, and target-recreation tests, then a richer receipt may add complexity without enough value.

Second, if kernel attachment enumeration cannot be made complete and cheap enough across important hook types, kernel-first reconciliation may need to stay hook-specific rather than becoming a general control-plane layer.

Third, if a simpler primitive emerges that atomically binds a persistent link to an application-defined owner generation and exposes that ownership through kernel query APIs, much of the userspace receipt machinery could collapse into that primitive.

Until then, the evidence points in the other direction. `bpffs` pinning is a lifetime mechanism. A restartable eBPF control plane still needs an **adoption protocol** that reconciles the live kernel object graph before it creates another one.

## Sources

- [Linux kernel documentation: eBPF syscall commands](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [libbpf API documentation](https://libbpf.readthedocs.io/en/latest/api.html)
- [Cilium TCX loader implementation](https://github.com/cilium/cilium/blob/main/pkg/datapath/loader/tcx.go)
- [Cilium issue #47847: shadow bpffs mount and duplicate TCX links after restart](https://github.com/cilium/cilium/issues/47847)
- [Cilium issue #46065: reported orphaned cgroup BPF link behavior across agent restarts](https://github.com/cilium/cilium/issues/46065)
