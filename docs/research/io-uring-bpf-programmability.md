---
date: 2026-08-14
title: "How Far Can eBPF Programmability Move Into io_uring?"
description: "Current Linux now exposes cBPF admission and eBPF struct_ops control in io_uring. This report maps capability, authority, lifecycle, and safe composition."
tags:
  - Daily Report
  - eBPF
  - io_uring
  - Linux
  - I/O
  - Security
  - Runtime
research_question: "What becomes possible now that current Linux io_uring has both in-ring BPF request filtering and an eBPF struct_ops execution path, and what authority, context, lifecycle, and evaluation contracts are still missing before eBPF can safely control asynchronous I/O?"
source_cutoff: 2026-08-14
status: daily-report
---

# How Far Can eBPF Programmability Move Into io_uring?

Suppose a storage or networking service owns one `io_uring` and submits work for many logical clients. The same ring may open files, connect sockets, issue device-specific `URING_CMD` requests, receive network payloads directly into registered userspace memory, and eventually carry filesystem work through FUSE. At that point, the ring is no longer just a faster syscall batching interface. It is an execution boundary with its own queues, registered resources, scheduling rules, and increasingly its own policy surface.

Linux has traditionally offered two obvious places to control such work. A program can configure static `io_uring` restrictions when the ring is created, and system security policy can act through Linux Security Module hooks around the operations that eventually execute. Current Linux source now contains a third class of mechanism: BPF code can run *inside* the `io_uring` path itself.

<!-- more -->

The important detail is that there are actually two different BPF mechanisms, and treating them both as generic "eBPF hooks" hides the interesting systems problem.

`IORING_REGISTER_BPF_FILTER` installs per-opcode **classic BPF (cBPF)** programs that decide whether a submission is allowed. Separately, `io_uring_bpf_ops` is an **eBPF `struct_ops`** interface that can replace a ring's loop step and call io_uring-specific kfuncs. One mechanism is deliberately small and admission-oriented. The other can participate in the ring's execution loop and access ring memory regions under verifier constraints.

That split is more interesting than adding another probe point. It means `io_uring` is starting to expose several programmable control planes with different trust models, contexts, and lifetimes. The research question is no longer "where can eBPF observe I/O?" It is **how eBPF should participate in I/O execution without creating an incoherent second security and scheduling stack**.

This report continues the [eBPF runtime and extensibility series](https://eunomia.dev/research/async-ebpf-causal-profiler/). The previous reports asked how userspace eBPF runtimes should define capabilities, how multiple programs compose on one hook, how stateful applications upgrade, and how asynchronous work preserves causality. `io_uring` now provides a concrete kernel subsystem where all four questions meet.

## io_uring already has four different kinds of control

The easiest way to see the architectural change is to compare what current Linux already exposes.

| Mechanism | Scope | Programmability | Main role |
| --- | --- | --- | --- |
| `IORING_RESTRICTION_*` | one ring | static declarative rules | allow selected registration operations, SQE opcodes, and SQE flags |
| `IORING_REGISTER_BPF_FILTER` | one ring and opcode | cBPF program | inspect submission context and allow or deny a request |
| Linux Security Modules | system security boundary | LSM policy, including BPF LSM where applicable | enforce host-wide security decisions independently of one ring |
| `io_uring_bpf_ops` | one ring | eBPF `struct_ops` plus io_uring kfuncs | participate in the ring execution loop, submit SQEs, and inspect permitted ring regions |

The [io_uring UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h) still contains the older restriction interface. It can allow a registration opcode, allow an SQE opcode, or constrain SQE flags. That is useful for sandbox-style rings because it can permanently reduce what a ring may submit.

The same UAPI now also includes `IORING_REGISTER_BPF_FILTER`. Unlike a static opcode allowlist, a filter can make a decision from request-specific state. This creates a natural hierarchy: restrictions describe a coarse capability envelope, while a filter can decide whether one request fits the policy inside that envelope.

The eBPF `struct_ops` path goes further. It is not just a finer allowlist. It can change how an eligible ring runs work.

The challenge is that these mechanisms were designed for different jobs. A production system that combines them needs to know which one is authoritative, which state each layer may observe or mutate, what happens during updates, and how an operator can explain a decision after the fact.

## The new request filter is BPF, but it is deliberately not eBPF

The current [BPF filter UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring/bpf_filter.h) defines an `io_uring_bpf_ctx` containing `user_data`, the SQE opcode, SQE flags, and optional opcode-specific auxiliary data. The current auxiliary union includes fields for `SOCKET`, `OPENAT`/`OPENAT2`, and `CONNECT`, such as socket family and protocol, open flags and resolve mode, or a destination address and port.

The kernel's [filter implementation](https://github.com/torvalds/linux/blob/master/io_uring/bpf_filter.c) makes the intended semantics quite explicit. Filters are registered per opcode. A return value of 1 allows the request and 0 denies it. If several filters exist for one opcode, every filter must allow the request. A denial becomes `-EACCES` before the request is queued.

The submission path in [`io_uring.c`](https://github.com/torvalds/linux/blob/master/io_uring/io_uring.c) runs those filters immediately after request initialization and before `trace_io_uring_submit_req()` and normal queueing. That is a useful location for admission control because the kernel has already interpreted enough of the SQE to construct a request, while execution has not yet escaped into worker, polling, device, or completion paths.

There is also a compatibility mechanism that is easy to miss. A userspace filter declares the auxiliary PDU size it expects. The kernel reports the actual size for that opcode, and `IO_URING_BPF_FILTER_SZ_STRICT` can make a size mismatch reject registration. `IO_URING_BPF_FILTER_DENY_REST` can populate unconfigured opcodes with a deny filter. These are small features, but they already encode two properties that a larger programmable interface needs: explicit context version expectations and an allowlist-by-default mode.

The surprising part is how the filter program is loaded. The implementation builds a `sock_fprog` and calls `bpf_prog_create_from_user()`. In other words, this is a constrained cBPF filter path, not an eBPF program loaded through a normal `BPF_PROG_LOAD` workflow.

That choice makes sense for a narrow gate. A cBPF program with a fixed context is easier to reason about than a general eBPF program with maps, kfuncs, mutable shared state, and a growing helper surface. The request filter can remain close to "pure predicate over this submission."

It also means that `IORING_REGISTER_BPF_FILTER` should not be described as the point where arbitrary eBPF enters io_uring. The deeper eBPF boundary lives elsewhere.

## eBPF struct_ops can participate in the ring's execution loop

The current [`io_uring/bpf-ops.h`](https://github.com/torvalds/linux/blob/master/io_uring/bpf-ops.h) defines `struct io_uring_bpf_ops` with a `loop_step` callback and a `ring_fd`. The implementation in [`bpf-ops.c`](https://github.com/torvalds/linux/blob/master/io_uring/bpf-ops.c) registers that structure as a BPF `struct_ops` type named `io_uring_bpf_ops`.

This is real eBPF programmability. The implementation registers io_uring-specific kfuncs for `BPF_PROG_TYPE_STRUCT_OPS`. Two are especially important:

- `bpf_io_uring_submit_sqes()` can submit a requested number of SQEs for the ring;
- `bpf_io_uring_get_region()` can return a bounded pointer to selected io_uring memory regions, including the parameter memory, completion-queue region, or submission-queue region.

The verifier path restricts access to the callback context, and the region kfunc checks the requested size before returning a pointer. The interface therefore does not simply expose an arbitrary `io_ring_ctx *` to eBPF. It defines a small subsystem-specific capability surface.

The installation rules also show that this is not yet a generic replacement for every io_uring execution mode. Current code rejects `io_uring_bpf_ops` on rings using `IORING_SETUP_SQPOLL` or `IORING_SETUP_IOPOLL`, requires `IORING_SETUP_DEFER_TASKRUN`, and permits only one installed `bpf_ops` instance per ring. The BPF object is attached to a specific `ring_fd` and removed when its `struct_ops` link is unregistered.

Those constraints are useful. They give us a concrete boundary to evaluate rather than an unlimited "programmable I/O" slogan. The eBPF program is allowed to participate in a particular ring's deferred task-run loop, with a defined set of kfuncs and memory regions.

The architectural distinction is now clear:

```text
SQE submitted by application
        |
        v
static ring restrictions
        |
        v
cBPF per-opcode admission filter
        |
        v
normal request initialization / security / issue paths
        |
        +----------> asynchronous workers, poll, device paths
        |
        v
eBPF struct_ops-controlled ring loop
        |
        +----------> submit SQEs / inspect permitted ring regions
        v
completion handling
```

This is not the exact call graph for every operation, but it captures the useful design separation. The cBPF filter decides whether a request may enter. The eBPF `struct_ops` path can influence how a supported ring progresses work. LSM and subsystem security checks remain separate authority boundaries.

## Why this matters more as io_uring absorbs new I/O objects

If io_uring only batched `read()` and `write()`, an in-ring execution control plane would still be interesting, but the scope would be limited. Current Linux is moving more subsystem-specific work into the ring.

The kernel's [FUSE-over-io-uring documentation](https://docs.kernel.org/next/filesystems/fuse-io-uring.html) describes a userspace FUSE daemon registering `IORING_OP_URING_CMD` entries on the FUSE connection. Once registration is active, the kernel can enqueue FUSE requests to per-CPU io_uring queues and the daemon returns a result while fetching the next request. The documentation explicitly says the interface is still in development and does not yet support every request type.

This changes what a ring represents. It can become the transport for a filesystem control path rather than merely a queue of syscalls initiated by the application.

The [io_uring zero-copy receive documentation](https://docs.kernel.org/networking/iou-zcrx.html) moves another boundary. Packet headers remain in the normal kernel TCP stack, but payload data can land directly in registered userspace memory. Flow steering and RSS decide which hardware receive queues feed the zero-copy path, and the application recycles buffers through an io_uring refill ring.

Again, the ring is now tied to resources that live beyond one ordinary syscall argument: NIC queues, registered memory, refill metadata, multishot receive state, and application buffer lifetimes.

The block layer is evolving similarly. The kernel's [ublk documentation](https://docs.kernel.org/block/ublk.html) describes userspace block servers built around io_uring commands, including newer zero-copy support using registered kernel buffers. A trusted userspace server becomes responsible for preserving buffer correctness when servicing client I/O.

These examples make the eBPF `struct_ops` interface more consequential. An eBPF program that controls a ring loop may eventually sit near filesystem, networking, block, or device-specific work. A subsystem-specific kfunc that is harmless for a synthetic benchmark can become part of a real storage or networking authority path when the ring owns registered resources.

The right abstraction therefore cannot be "give eBPF more io_uring internals." It needs to state exactly which I/O capabilities the program controls and which system security decisions remain outside its authority.

## eBPF should compose with LSM, not become a parallel security stack

Current [`io_uring.c`](https://github.com/torvalds/linux/blob/master/io_uring/io_uring.c) calls `security_uring_allowed()` during ring setup. io_uring also invokes security checks in credential and operation-specific paths. More generally, Linux Security Modules exist to make host security policy authoritative across subsystems, not only inside one file descriptor.

A ring-local BPF program solves a different problem. It can express policy from the perspective of one application or one ring. For example, a service might want one tenant-facing ring to connect only to a known address class, deny filesystem opens with selected flags, or use a custom deferred-task execution policy. Those are useful local constraints even when the host LSM already permits the process to perform the underlying operations.

The dangerous design would let a ring-local eBPF program reinterpret system authority. If the struct_ops program can create or submit work through a path that no longer encounters the same LSM checks as a normal request, programmability becomes a bypass surface. Conversely, if every ring-local decision has to re-run a large host policy engine, the hot-path benefit may disappear.

The clean model is asymmetric:

1. **LSM remains the upper security authority.** Ring-local programs may further restrict or schedule work, but cannot grant an operation that host policy denies.
2. **cBPF filters provide cheap request admission.** Their small context and pure allow/deny semantics make them suitable for a narrow predicate.
3. **eBPF struct_ops provides execution policy inside an explicitly supported ring mode.** Its kfunc and memory access surface is capability-limited and should stay auditable.
4. **Static restrictions define irreversible or hard-to-broaden capability envelopes when that is desired.** A dynamic program should not silently widen a ring that was intentionally created with a smaller static authority set.

Linux already contains pieces of this hierarchy. What is missing is a machine-readable contract that makes the relationship explicit enough for application authors, verifiers, and operators to rely on it.

## Where current work is still weak

### Filter context is much richer for some opcodes than others

The base cBPF context always contains opcode, SQE flags, and `user_data`. The current [`opdef.c`](https://github.com/torvalds/linux/blob/master/io_uring/opdef.c) adds semantic filter payloads for `CONNECT`, `OPENAT`/`OPENAT2`, and `SOCKET`. Many other operations have no equivalent opcode-specific payload.

That means two policies with the same conceptual goal can have very different observability. A filter can inspect an address for `CONNECT`, but a future policy over `URING_CMD`, zero-copy receive resources, registered files, or subsystem-specific command fields may see far less semantic information.

The missing property is not "more fields." It is a principled rule for **which semantic state is safe and stable enough to expose at the admission boundary**. The current PDU-size negotiation is a useful compatibility seed, but it does not define how new fields acquire semantics or how policy remains portable across kernel versions.

A decisive test would take the same policy, such as "this ring may access only resources in tenant T," across file, socket, FUSE, ublk, and zero-copy receive operations. If each opcode needs unrelated ad hoc parsing or privileged side channels, the context model is too fragmented.

### The control planes have no common precedence or provenance model

Static restrictions, cBPF filters, LSM decisions, and eBPF `struct_ops` all answer different questions. Their interaction is currently implicit in code paths rather than represented as one policy graph.

Operators will eventually ask simple questions that are hard to answer from that arrangement:

- Which rule denied this SQE?
- Which version of the ring-local policy was active?
- Did an eBPF loop submit the request or did userspace submit it directly?
- Which LSM decision still applies to a request generated by an in-ring execution policy?
- Did the ring inherit or copy a filter set from another restriction object?

The cBPF filter implementation already uses copy-on-write behavior for filter sets, and the eBPF struct_ops object has its own link lifecycle. These mechanisms are individually understandable, but they do not produce one common provenance record.

A production runtime needs a stable answer for every admitted, denied, generated, and completed request. Without that, programmability can improve control while making incidents harder to explain.

### Updating one ring's programmable state is not a transaction

The active series already examined [transactional upgrades for stateful eBPF applications](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). io_uring exposes the same problem in a narrower and more operationally important form.

A ring may simultaneously depend on static restrictions, a set of cBPF opcode filters, one eBPF struct_ops link, registered files, memory regions, personalities, buffer rings, FUSE entries, or zero-copy network resources. Changing only one of those objects can temporarily create a policy that no intended generation ever specified.

For example, an operator might install a new execution policy before the matching filter context is available, or update a resource registration while an old struct_ops program still assumes the previous layout. The kernel's current individual APIs can each be correct while the application-level transition is inconsistent.

The missing mechanism is a generation boundary across *related programmable I/O state*. Whether that belongs in io_uring itself or in a userspace controller is an open design choice, but the evaluation must include concurrent submissions and failure in the middle of an update.

### Resource accounting is weaker than the new execution boundary

Once eBPF can drive a ring loop and the ring can own substantial registered resources, CPU instruction count is not enough for accounting. A policy may trigger submissions, retain registered memory, interact with FUSE queues, or change how often the application crosses into the kernel.

A safe runtime needs attribution for at least:

- eBPF execution time and invocation count;
- SQEs submitted by userspace versus generated or progressed through the BPF-controlled loop;
- bytes and lifetime of registered memory reachable by the ring;
- completion-queue pressure and dropped or deferred work;
- subsystem-specific resources such as FUSE queue entries or zero-copy receive buffers.

Without this accounting, a program can be verifier-safe but still be operationally expensive. This is the same distinction the [userspace eBPF runtime contract](https://eunomia.dev/research/userspace-ebpf-runtime-contract/) made between memory safety and runtime resource authority, now applied to a kernel I/O substrate.

## Promising directions with academic and production value

### 1. A typed capability contract for programmable io_uring

**Gap.** The current interfaces expose a mixture of static restrictions, cBPF context fields, eBPF `struct_ops` callbacks, and kfunc-accessible regions, but no single description of what a ring-local program is allowed to observe or cause.

**Mechanism.** Define a machine-readable capability descriptor for a programmable ring. It would name the permitted SQE classes, semantic context fields, registered-resource classes, eBPF kfuncs, writable or read-only regions, and whether BPF-originated submission is allowed. The descriptor would be bound to the ring and queryable from userspace. BTF can describe eBPF-side typed structures, while the existing filter PDU-size negotiation can remain the compatibility mechanism for the intentionally small cBPF gate.

The key is to keep cBPF and eBPF separate where their trust models differ while giving both a common capability vocabulary. A descriptor might say that `CONNECT` admission exposes destination family/address/port to cBPF, while the eBPF loop can submit only opcodes A and B and access only the CQ region.

**Delta.** This is different from adding more filter fields or more kfuncs. The artifact is the contract that constrains and explains those interfaces.

**Artifact and evaluation.** Implement a prototype around current io_uring query/registration APIs and a liburing inspection tool. Test file, network, FUSE, and ublk workloads. Measure whether the same high-level capability policy maps cleanly across opcodes, how many kernel-version conditionals remain, and whether a verifier or controller can reject incompatible program/ring combinations before activation.

**Academic value.** The general question is how a subsystem-specific eBPF runtime exposes heterogeneous capabilities without collapsing into an unstable internal API.

**Production value.** Operators can inspect what a ring-local program can actually do before attaching it, rather than reviewing BPF source plus kernel internals.

**Failure condition.** If the descriptor simply mirrors dozens of unstable implementation fields and cannot reduce compatibility logic, it adds bureaucracy without providing a useful abstraction.

### 2. Versioned ring policy generations with explicit authority ordering

**Gap.** A ring's effective policy spans objects that update independently and whose precedence is implicit.

**Mechanism.** Treat related restrictions, cBPF filters, eBPF struct_ops, and registered-resource assumptions as one versioned policy generation. Build the next generation off-path, validate it against the target ring capabilities, then activate it with one generation switch. Requests record the generation under which they entered the ring. Completions and audit events preserve that generation even if policy changes while the request is in flight.

The generation also defines authority ordering: static restriction envelope first, host LSM authority always applicable, ring-local cBPF admission next, and eBPF execution policy last within its granted capabilities. A generated submission inherits the same or a strictly narrower generation instead of starting an untracked authority chain.

**Delta.** The previous transactional-upgrade report focused on general stateful BPF applications. Here the test case is stricter: the transaction must cover a live asynchronous I/O queue whose old requests and new requests overlap in time.

**Artifact and evaluation.** Implement generation tagging in a userspace controller first, then evaluate whether a small kernel primitive is needed for atomic activation. Inject process crashes and update failures while fio, a network proxy, FUSE, and ublk workloads submit continuously. Score unauthorized windows, request loss, generation ambiguity, rollback time, and hot-path overhead.

**Academic value.** This exposes a general concurrency problem at the boundary between policy versioning and asynchronous execution.

**Production value.** A service can update ring-local policy without draining every queue or accepting a temporary mixed configuration that is difficult to audit.

**Failure condition.** If normal link replacement plus application-level quiescence already provides negligible downtime and no measurable inconsistent window, a new generation primitive is unnecessary.

### 3. A benchmark for in-ring eBPF control versus external hooks

**Gap.** It is easy to argue that an in-ring eBPF control point is faster or more expressive than LSM, tracing, seccomp-style constraints, or userspace validation. There is not yet a standard workload that tells us when the extra control surface is actually worth its complexity.

**Mechanism.** Build one benchmark suite where the same policy is implemented at several boundaries:

1. userspace validation before SQE submission;
2. static `IORING_RESTRICTION_*` where expressible;
3. cBPF `IORING_REGISTER_BPF_FILTER`;
4. host security policy at an LSM-relevant boundary;
5. eBPF `io_uring_bpf_ops` for execution policies that need the ring loop;
6. a tracing-only baseline that observes but does not enforce.

Use policies that force the mechanisms to differ. Examples include destination-aware socket admission, file-open constraints, tenant-specific registered-buffer use, request scheduling under completion pressure, and a BPF-driven loop that batches submission adaptively.

**Artifact and evaluation.** Use microbenchmarks plus realistic network, storage, FUSE, and ublk services. Measure per-I/O latency, throughput, CPU cycles, cache effects, policy precision, bypass attempts, tail latency under overload, and update behavior. Run ablations with no semantic PDU, no region kfunc access, and no BPF-originated submission.

**Academic value.** The result would identify which control decisions benefit from being moved inside an asynchronous I/O runtime rather than attached before or after it.

**Production value.** Kernel and runtime maintainers get evidence for whether a proposed new io_uring BPF context field or kfunc solves a real deployment problem.

**Failure condition.** If external hooks and userspace validation match the in-ring mechanisms on both overhead and policy precision for realistic workloads, the correct design is to keep io_uring's BPF surface small.

## The design target is a narrow programmable boundary, not unlimited in-kernel plugins

Current Linux source is already enough to reject two extremes.

The first extreme says io_uring only needs observability. That is no longer accurate because the kernel contains both a submission-time BPF decision path and an eBPF execution-control interface.

The second extreme says this should grow into a general-purpose eBPF runtime with access to arbitrary io_uring internals. The current implementation points in the opposite direction. The cBPF filter intentionally has a small predicate context. The eBPF path exposes specific BTF-checked callback state and selected kfuncs. It restricts which ring modes can attach the struct_ops implementation. These constraints are features if the goal is a maintainable subsystem interface.

The more useful target is a **narrow programmable boundary with explicit authority**:

- stable semantic context where request admission needs it;
- a deliberately small eBPF execution interface where in-ring control has measurable value;
- host security policy that remains authoritative;
- versioned lifecycle and provenance across the pieces;
- resource accounting that measures effects beyond verifier safety.

That design would make io_uring a strong case study for a broader eBPF question. As BPF moves from observing subsystems to participating in their control loops, the main interface problem shifts from "which hook can I attach to?" to "what capability does this attachment grant, and how does it compose with the rest of the system?"

## What would change this conclusion?

This argument assumes that in-ring decisions provide either lower overhead, better semantic context, or tighter execution control than policies placed outside io_uring. The benchmark above could show that the assumption is wrong.

If static restrictions plus existing LSM and userspace validation cover nearly every production policy with comparable overhead, io_uring should keep the new BPF surfaces narrow and specialized. If eBPF struct_ops cannot expose useful execution control without depending on unstable ring internals, the interface may be better treated as an experimental optimization hook rather than a runtime boundary. If cBPF filter overhead becomes material on high-IOPS paths, the right answer may be to push more decisions into static capabilities rather than make the filter richer.

The opposite result would be more interesting. If FUSE, zero-copy networking, ublk, and other ring-centric workloads repeatedly need request semantics and execution control that external hooks cannot express cheaply, then io_uring offers a concrete path toward subsystem-native eBPF programmability. In that case, the next work should not be a larger pile of hooks. It should be a coherent contract for capability, authority, updates, provenance, and resource cost.