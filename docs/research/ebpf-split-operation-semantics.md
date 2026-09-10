---
date: 2026-09-10
title: "Can eBPF Split One Operation Across Host and Hardware?"
description: "High-level eBPF offload can split one policy effect across host and hardware. This report asks how to preserve atomicity, ordering, and replay-safe recovery."
tags:
  - Daily Report
  - eBPF
  - SmartNIC
  - DPU
  - Offload
  - Distributed Systems
  - Verification
research_question: "How can eBPF split one high-level semantic operation across kernel, NIC/DPU, or other hardware execution without changing its observable result under partial completion, retry, reordering, fallback, or device failure?"
source_cutoff: 2026-09-10
status: daily-report
---

# Can eBPF Split One Operation Across Host and Hardware?

Suppose an eBPF policy receives a packet, checks a host-owned tenant budget, asks a SmartNIC to transform or redirect the packet, and then updates accounting state. Keeping the whole path on the CPU is easy to reason about but may waste the hardware fast path. Moving the whole program to the NIC may be impossible because the authoritative budget or one required operation still lives on the host.

The tempting compromise is to split the operation. Let the device do the cheap local work and let the host finish the part that needs host state or authority. That can be fast, but it creates a new correctness question: what happens if the device completes its half and the host never commits its half, or if a retry causes one side to execute twice?

Linux and existing research already contain pieces of this design space. XDP can hand a frame to another CPU or device-specific program, Linux can bind BPF programs to offload devices, networking subsystems split high-level functions between software and hardware, and systems such as fabric_ext compile one policy across several heterogeneous targets. The missing piece is a general eBPF contract for **one logical effect whose implementation crosses execution domains**. The contract has to say where the operation commits, which effects may be replayed, which state is authoritative, and how both sides agree that they are executing the same generation.

<!-- more -->

This is different from [heterogeneous eBPF execution placement](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/), which asks where a valid policy should run. It is also different from [architecture-specific specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/), which asks whether one portable operation has an eligible native implementation, and from the recent [native-operation trust report](https://eunomia.dev/research/ebpf-native-operation-trust-boundary/), which asks what code must be trusted after delegation. Here the target is already chosen and trusted enough to use. The question is whether an operation can be **partly completed in more than one domain without creating an externally visible half-result**.

## BPF already has explicit handoffs, but not a general split-operation contract

Linux XDP provides useful examples because it does not pretend that a handoff is ordinary local execution. `BPF_MAP_TYPE_CPUMAP` can redirect an `xdp_frame` to another CPU, where a second XDP program may run. `BPF_MAP_TYPE_DEVMAP` can associate another XDP program with a destination device, and that program executes after `XDP_REDIRECT` before the frame is queued for transmission. The handoff point is part of the API.

These mechanisms are easier to reason about than an invisible compiler split. The first program has an explicit return action, ownership of the frame changes at a defined boundary, and the next stage runs in a documented context. But they still leave application-level composition to the program. If stage one increments a quota counter and stage two drops, duplicates, or reroutes the frame, the kernel does not automatically define a transaction spanning both effects.

The current Linux hardware-offload path is similarly explicit about device binding. `kernel/bpf/offload.c` restricts device-bound program initialization to XDP and `SCHED_CLS`, associates the program with a network device, and lets device-specific offload operations participate in verifier preparation, instruction checks, and finalization. That is a real offload contract, but it is primarily a **program-to-device** contract. It does not provide a generic abstraction saying that instructions 1-40 and one map update form the host half of a semantic operation while instructions 41-80 and a packet transformation form the device half.

This distinction matters because whole-program equivalence and split-operation correctness are different properties. A whole-program device implementation can be checked against the original BPF behavior. Once the implementation is partitioned, correctness also depends on the protocol between the partitions.

## Other Linux offloads show the protocol problem clearly

The broader Linux networking stack already exposes what happens when a high-level operation is divided between host software and hardware.

The XFRM device interface distinguishes **crypto offload**, where the NIC performs encryption or decryption while the kernel does the rest, from **packet offload**, where the NIC also handles encapsulation and keeps security-association and policy state synchronized with the kernel. The difference is not just how much work moves. The second mode requires a stronger shared-state contract because hardware participates in more of the logical IPsec operation.

Netfilter flowtable hardware offload shows a second failure mode. The kernel documentation notes that hardware installation is asynchronous, so a few packets can still traverse the software fast path before a flow reaches hardware. It also warns that offloaded flow state can become stale when forwarding information changes. The system therefore has periods in which software and hardware coexist for the same logical traffic class.

Those mechanisms are not eBPF APIs, but they are useful counterexamples to a simple model of offload. An operator cannot infer semantic atomicity from the fact that "hardware offload is enabled." Correctness depends on which state is synchronized, when ownership changes, what packets can still use the old path, and how stale or failed hardware state is repaired.

A high-level eBPF delegation mechanism will face the same questions as soon as it splits an effect instead of moving an entire self-contained program.

## Research systems make richer cross-device decomposition plausible

[hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) showed that real XDP programs can execute on an FPGA NIC by providing an optimizing compiler, an extended BPF instruction set, and FPGA implementations of XDP maps and helper functions. Its implementation used about 15% of the FPGA resources and reported packet-processing throughput comparable to a high-end CPU core with roughly 10x lower forwarding latency in the evaluated setup. The important point for this question is that preserving familiar XDP behavior required reproducing more than BPF arithmetic. The device needed the surrounding map and helper environment.

More recent [fabric_ext](https://arxiv.org/abs/2607.26335) goes beyond whole-program offload. Its semantic movement graph describes data movement with ordering, ownership, source and destination, and transformations such as move, checksum, filter, reduce, replicate, and persist. The compiler lowers one policy into per-device BPF programs, verifier obligations, consistency-classed maps, and backend artifacts across GPU, driver/runtime, DPU/NIC, and CXL-side targets.

That is strong evidence that higher-level decomposition is useful. It also sharpens the unsolved boundary. A compiler may know that an operation contains a `Reduce` followed by `Persist`, or that two stages share an ownership edge, but production recovery still needs an answer when one target observes completion and another target resets, retries, or remains on an older policy generation.

The next abstraction should therefore be smaller than a universal distributed transaction system and stronger than a best-effort handoff. It should describe the **commit semantics of the operations that a compiler is allowed to split**.

## The split must preserve one observable result

Consider a semantic operation `charge_and_redirect(tenant, packet)` with two intended effects:

1. consume one unit from an authoritative tenant budget;
2. emit the packet on a selected device path.

There are several ways to partition it. The host can reserve the budget first and give the NIC a capability to send exactly one packet. The NIC can tentatively process the packet and ask the host to commit. Or the operation can be defined as replay-safe so that either side can retry using the same identity.

What cannot remain implicit is the point at which the operation becomes externally committed.

A useful split-operation descriptor needs at least these properties:

- a stable operation type and policy generation;
- an operation-instance identity that survives retry;
- the authoritative owner of every state object the operation reads or mutates;
- the effects permitted before commit and the effects that constitute commit;
- whether each stage is idempotent, commutative, compensatable, or non-repeatable;
- the ordering relation required between host and device effects;
- the terminal outcomes after timeout, device reset, generation change, or duplicate completion.

This does not require two-phase commit for every packet. Many fast-path operations can use a much cheaper protocol. A host can mint a bounded one-use capability, a device can apply an idempotent transformation keyed by an operation ID, or a counter update can be declared commutative and reconciled later. The point is to make the chosen failure semantics explicit enough that the compiler and runtime can reject an unsafe split.

A prototype could be built above existing kernels in a runtime such as [bpftime](https://github.com/eunomia-bpf/bpftime), with host-side BPF stages, device emulation, and fault injection before attempting a new Linux ABI. The first research result should be the contract and its measurable value, not a broad claim that arbitrary eBPF can be distributed automatically.

## Where current work is still weak

### Split semantics are usually encoded in each subsystem

CPUMAP and DEVMAP define specific XDP handoffs. XFRM defines its own software/hardware split and state synchronization. Flowtable offload defines another transition model. Hardware BPF offload exposes device-specific verifier and translation hooks. These interfaces work because each subsystem can encode its own assumptions.

What is missing is a reusable eBPF-level vocabulary for a high-level operation that spans two execution domains. Without it, every new compiler or accelerator has to rediscover which effects may happen before handoff, which state must remain authoritative, and which retries are safe.

### Verifier safety stops before cross-domain commit semantics

The BPF verifier can establish memory and type safety for a BPF program and can enforce declared call constraints. A device backend can additionally verify or translate an offloaded program. Neither result proves that a host-side reservation and a device-side packet effect happen once, in order, as one logical operation.

The missing property is not another instruction safety rule. It is a protocol property over multiple verified stages.

### Failure and generation changes are underrepresented in offload evaluation

Throughput and latency are natural accelerator metrics. They rarely expose the cases that make a split wrong: duplicate device completion after a timeout, an old device rule firing after a host policy update, a host state update succeeding while the device resets, or a fallback path executing the same effect again.

A split-operation mechanism needs tests where these events are deliberate rather than accidental. Otherwise an implementation can look equivalent during steady state while diverging exactly during the transition that operators care about most.

### The useful subset of splittable operations is unknown

A completely general protocol could be too expensive. Stateless transforms, monotonic counters, bounded reservations, and idempotent writes may cover a large fraction of profitable offloads with simple rules. Operations that combine irreversible external effects with strongly consistent host state may be better kept in one domain.

The field needs evidence about that boundary before standardizing a general cross-device transaction abstraction.

## Promising directions with academic and production value

### 1. Compile split operations from explicit effect classes

**Gap.** Current BPF types describe values and verifier-visible safety, but a compiler deciding to partition a high-level operation lacks a machine-readable statement of retry and commit behavior.

**Mechanism.** Let a high-level operation declare its effects as a small set of classes: pure, idempotent, commutative, reservable, compensatable, or non-repeatable. The descriptor also names authoritative state and the commit effect. The compiler may split an operation only when it can select a protocol valid for those classes. For example, a reservable host quota plus a one-use device capability needs no general transaction coordinator; a non-repeatable external write followed by host bookkeeping may be rejected as unsplittable.

**Delta.** Architecture capability manifests answer whether a target can execute an implementation. Native-operation contracts bound what one trusted implementation may do. This proposal instead decides whether **multiple implementations may jointly realize one logical effect** under failure.

**Artifact.** A descriptor format, compiler pass, and runtime library with three or four concrete protocols, integrated first with host BPF plus a software SmartNIC/DPU emulator.

**Evaluation.** Use packet transformation, quota enforcement, telemetry aggregation, and movement/checksum pipelines. Compare whole-host execution, whole-device execution where available, ad hoc split code, and effect-typed splitting. Inject retries and device resets. Measure incorrect external outcomes, duplicate or lost effects, runtime overhead, added latency, and how often the compiler can safely accept a split. The idea loses if ordinary explicit stage APIs achieve the same fault coverage with materially less metadata and machinery.

### 2. Add generation-bound operation receipts at the handoff

**Gap.** A timeout does not tell the host whether the device never saw an operation, completed it, or completed it under an older policy generation.

**Mechanism.** Every split instance carries a compact identity such as `(policy_generation, operation_id, stage)`. The device records only the minimum durable or replayable state needed for the operation's declared retry class. Completion returns a receipt containing the generation, stage outcome, and effect identity. A host retry with the same operation ID can be deduplicated; a completion from the wrong generation cannot silently satisfy the new operation.

For very fast paths, receipts need not be exported per packet to userspace. They can exist in bounded device state, compact rings, or sampled audit mode. The protocol should scale assurance with effect severity rather than turning every packet into a distributed log record.

**Artifact.** A host/device handoff library plus fault-injection hooks that model delayed completion, duplicate completion, device reboot, and stale generation state.

**Evaluation.** Compare no identity, generation-only identity, and full operation identity. Measure duplicate effects, stale-generation acceptance, memory cost, lookup cost, recovery latency, and the maximum sustainable operation rate. An ablation should show whether operation-level identity adds value beyond an existing generation fence.

**Academic value.** The general question is how little cross-domain state is sufficient to obtain replay-safe semantics for high-rate programmable datapaths.

**Production value.** Operators get a bounded recovery rule after device resets or timeouts instead of choosing between blind replay and dropping uncertain work.

### 3. Build a semantic fault benchmark for partial offload

**Gap.** Existing offload benchmarks can prove that one target is fast while saying little about whether a split preserves the same outcome during transition and failure.

**Mechanism.** Define each workload by an observable semantic oracle, then run it in pure host, pure device, and split configurations. Faults occur at controlled points around the handoff: before device acceptance, after device effect but before completion, after host state mutation, during policy-generation replacement, and during fallback activation.

**Artifact.** A reproducible benchmark with packet traces, state snapshots, fault schedules, and an oracle that detects duplicate, lost, reordered, and stale-generation effects. Include Linux XDP multi-stage paths as a readily available baseline and at least one SmartNIC/DPU or faithful emulator path.

**Evaluation.** Report semantic divergence rate first, then throughput, p99 latency, recovery time, state/receipt memory, and host-device traffic. Compare best-effort split, stop-the-world handoff, generation fencing, and effect-aware replay. Include a workload of stateless independent packets where the simplest best-effort design should win; otherwise the benchmark would only reward extra machinery.

**Academic value.** This supplies a common measurement method for the correctness/performance frontier of partial offload.

**Production value.** The benchmark can become a release gate for drivers, compiler partitioners, and programmable-NIC runtimes before they enable a new split mode in production.

## What would change this conclusion?

The case for a reusable split-operation contract would weaken if real eBPF offloads rarely need to divide one logical effect. Whole-program placement, explicit XDP stage boundaries, or stateless device functions may cover most useful deployments. In that world, a general descriptor and receipt protocol would add complexity without changing many failures.

It would also weaken if existing subsystem-specific protocols already provide a small common interface that can be reused without new BPF semantics. XFRM-style state synchronization, generation fencing, and ordinary idempotent request IDs might be enough once they are composed carefully by a runtime.

The strongest falsifier is empirical. If fault-injection experiments show that effect typing and operation receipts catch no additional externally visible errors over a simpler explicit-stage baseline, or if they impose enough state traffic to erase the offload benefit, the extra abstraction should be dropped.

Until that evidence exists, high-level eBPF offload should distinguish **moving an implementation** from **splitting an operation**. The first needs capability, equivalence, and trust evidence. The second also needs a commit and replay contract, because two individually correct stages can still compose into one incorrect external effect.

## References

- Linux kernel source, [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c), accessed 2026-09-10.
- Linux kernel documentation, [`BPF_MAP_TYPE_CPUMAP`](https://docs.kernel.org/bpf/map_cpumap.html), accessed 2026-09-10.
- Linux kernel documentation, [`BPF_MAP_TYPE_DEVMAP` and `BPF_MAP_TYPE_DEVMAP_HASH`](https://docs.kernel.org/bpf/map_devmap.html), accessed 2026-09-10.
- Linux kernel documentation, [XFRM device: offloading IPsec computations](https://docs.kernel.org/networking/xfrm_device.html), accessed 2026-09-10.
- Linux kernel documentation, [Netfilter flowtable infrastructure](https://docs.kernel.org/networking/nf_flowtable.html), accessed 2026-09-10.
- Marco Spaziani Brunella et al., [hXDP: Efficient Software Packet Processing on FPGA NICs](https://www.usenix.org/conference/osdi20/presentation/brunella), OSDI 2020.
- Yiwei Yang and Andi Quinn, [The Fabric Is the Cluster Driver: Cross-Layer eBPF Policies for GPU-CXL Fabrics](https://arxiv.org/abs/2607.26335), arXiv:2607.26335, 2026.
