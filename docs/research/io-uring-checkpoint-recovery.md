---
date: 2026-09-13
slug: io-uring-checkpoint-recovery
title: "What Must Be Saved to Checkpoint a Process Using io_uring?"
description: "An io_uring checkpoint must preserve more than process memory: in-flight requests, ring resources, completions, and external effects need recovery semantics."
tags:
  - Daily Report
  - Linux
  - io_uring
  - Checkpoint Restore
  - Async I/O
research_question: "What state and recovery protocol are required to checkpoint and restore an io_uring application without losing completions, duplicating external effects, or rebinding ring-owned resources incorrectly?"
source_cutoff: 2026-09-13
status: daily-report
---

# What Must Be Saved to Checkpoint a Process Using io_uring?

A normal process checkpoint starts from a familiar picture: stop the threads, save memory and registers, preserve file descriptors and other kernel-visible resources, then recreate them on another process or host. `io_uring` makes that picture incomplete because part of the application's live state can sit between userspace and the kernel.

A submitted request may already hold a kernel reference to a file even after the application's ordinary file descriptor is closed. A registered buffer may still be pinned because an in-flight operation uses it. A multishot accept can remain active after producing several completions. A completion may already describe an externally visible effect that userspace has not processed yet. For some operations, cancellation is inherently racy or no longer possible after the request reaches hardware.

This means an `io_uring` checkpoint cannot be defined as "save the mappings that contain the SQ and CQ rings." The real question is which asynchronous operations and ring-owned resources belong to the checkpoint, which effects have already happened, which completions the application has observed, and what restore is allowed to replay.

The practical consequence is straightforward: **a correct checkpoint needs an operation-level recovery contract, not only a byte-for-byte snapshot of process state.** Without that contract, restore can lose a completed result, repeat a write or network action, attach a fixed-file index to the wrong object, or reuse a buffer whose ownership had already moved.

<!-- more -->

## Why freezing memory is not enough for an io_uring checkpoint

`io_uring` deliberately moves work out of the synchronous syscall boundary. Userspace prepares submission queue entries, the kernel consumes them, and completion queue entries report results later. That separation is useful for performance, but it creates several distinct states that a checkpoint system has to distinguish.

Consider a server that has submitted a multishot accept, registered a table of fixed files, and uses a provided-buffer ring for receives. At the instant the process is frozen, one accepted connection may already exist in a direct descriptor slot even though its CQE has not been consumed. Another connection may still be waiting inside the persistent multishot request. A receive may have consumed one buffer ID from the shared pool. Simply copying userspace memory does not tell restore which of those transitions have already become real kernel or external state.

The upstream liburing manual pages make these boundaries explicit. Registered files are not just copies of the process file-descriptor table. The kernel holds its own references, the application may close the original descriptors, and direct descriptors can exist only inside the registered file table. Registered buffers are pinned and remain valid while in-flight operations still reference them; replacing or unregistering a buffer does not immediately release it. Provided-buffer rings add another ownership transition because the kernel consumes buffer IDs and returns the selected ID in the completion.

Cancellation does not turn this into a simple drain problem. `IORING_OP_ASYNC_CANCEL` itself races with normal completion, and the two resulting CQEs can arrive in either order. The current liburing documentation also states that not all operations are cancelable: I/O already submitted to hardware, such as disk I/O in progress, typically cannot be canceled. Closing an ordinary file descriptor does not necessarily stop pending `io_uring` work because the request can hold its own file reference.

Multishot operations widen the state space further. One SQE can produce many CQEs while remaining active. A multishot accept can create a sequence of accepted sockets, and a multishot receive can consume a sequence of provided buffers. At checkpoint time, "the SQE has completed" is therefore not even a binary property.

CRIU provides useful evidence that this is still a real compatibility boundary rather than a solved deployment detail. Its issue **#2131, `io_uring support in CRIU`**, remains open as of the source cutoff. The reported reproducer fails while dumping a process with an established ring because of the `anon_inode:[io_uring]` mapping. Supporting the mapping is only the first layer. A production-quality restore must also decide what to do with live requests and ring-owned resources whose semantics are not captured by ordinary memory and file-descriptor images.

## The missing distinction: request state, completion state, and effect state

For checkpoint and restore, one request can cross at least three boundaries:

| Boundary | Example question | Why restore cares |
| --- | --- | --- |
| Request ownership | Is the SQE still only in userspace, accepted by the kernel, queued, or executing? | Determines whether the old ring can still produce a result and whether replay is possible. |
| Completion visibility | Has a CQE been produced, and has userspace consumed it? | Prevents losing a result or delivering the same logical completion twice. |
| External effect | Has the operation changed a file, socket, namespace object, peer, or device state? | Prevents replay from duplicating an effect that already happened. |

These boundaries are related but not identical. A write can have reached the storage stack before userspace sees the CQE. An accept can have created a socket before the application processes the returned direct-descriptor index. A receive can consume bytes from a peer and a buffer from a provided-buffer pool before higher-level code records that progress.

That makes blind replay unsafe. Recreating a fresh ring and resubmitting every request that lacks an application-consumed CQE may repeat side effects. Refusing to replay anything may instead lose operations that were accepted by the old ring but never took effect. The checkpoint layer needs a classification finer than "pending" and "done."

A useful recovery vocabulary is:

- **recreate:** rebuild ring configuration and resource tables from durable identities;
- **replay-safe:** resubmit because repeating the operation cannot create an invalid external state;
- **reconcile:** query or reconstruct external state before deciding whether to resubmit;
- **deliver-only:** preserve a completion whose effect already happened but whose result userspace had not consumed;
- **quiesce-required:** refuse the checkpoint until the request reaches a known boundary;
- **non-restorable:** reject migration because no safe reconstruction exists for that operation and environment.

The exact classes can differ by runtime, but a restore system needs some equivalent distinction. Otherwise it silently treats an asynchronous execution protocol as a memory-copy problem.

## Where current work is still weak

The first gap is **an inspectable ring-state export boundary**. Linux and liburing expose powerful setup, registration, cancellation, and completion interfaces, but there is no general checkpoint image that serializes an `io_uring` instance as a set of restorable operations and resource identities. Reconstructing only userspace mappings is insufficient because the kernel owns request references, fixed-file entries, pinned buffers, provided-buffer consumption, worker state, and other execution state.

The second gap is **effect-aware replay semantics**. `user_data` gives applications a way to correlate requests and completions, but it does not tell a checkpoint system whether an operation is idempotent, whether an external side effect already occurred, or how to reconcile that effect after migration. This is especially important for writes, sends, accepts, opens, and operations that allocate direct descriptors.

The third gap is **resource rebinding across restore**. A fixed-file index is meaningful only relative to one ring's registered file table. A buffer ID is meaningful relative to one buffer group. A direct descriptor may not exist in the ordinary process descriptor table at all. Restoring numeric indices without restoring the object identity and ownership graph can produce a syntactically valid ring that points at the wrong resources.

The fourth gap is **a failure-oriented benchmark**. A checkpoint test that only proves "the process resumed and continued serving requests" can miss duplicate writes, silently dropped CQEs, duplicated accepts, wrong fixed-file bindings, buffer reuse before ownership returned, or multishot history being restarted from the wrong point. These failures require an oracle that understands logical operations and external effects.

## Promising directions with academic and production value

### 1. Define an io_uring recovery manifest for operations and resources

The first direction is a machine-readable checkpoint manifest that describes both ring configuration and the logical resources it refers to. The manifest would not dump arbitrary kernel internals. It would record the minimum stable information needed to reconstruct or reject the ring:

```text
ring = setup flags + supported features + registration generation
files = fixed index -> stable object identity + reopen/reconnect method
buffers = group/index -> memory region generation + ownership state
requests = logical request id + opcode + dependencies + recovery class
multishot = logical request id + delivered completion frontier
completion = result present? userspace consumed? associated resource/effect?
```

The central mechanism is **generation-scoped identity**. Fixed index 7 after restore is valid only if it is rebound to the same logical file or socket generation that index 7 represented when the checkpoint was taken. The same rule applies to buffer groups and direct descriptors. Numeric equality alone is not enough.

A prototype could integrate with a CRIU plugin or a userspace runtime that wraps liburing. Evaluate file I/O, TCP servers, multishot accept/recv, registered files, registered buffers, provided-buffer rings, and direct descriptors. Compare a memory-plus-FD baseline, a drain-and-recreate baseline, and the manifest design. The primary metric should be invalid recovery events, followed by checkpoint downtime, restore latency, extra bookkeeping, and the fraction of workloads that require quiescence.

The academic value is a portable model for reconstructing asynchronous kernel-owned resources without requiring a full serialization of opaque kernel implementation state. The production value is live migration and restart for storage engines, proxies, runtimes, and services that increasingly depend on `io_uring`.

The idea should be rejected if a simple drain, destroy, and recreate protocol reaches the same correctness with bounded downtime across realistic workloads. In that case, a richer manifest would add complexity without buying availability.

### 2. Track a completion-and-effect frontier instead of only pending SQEs

A second direction is an operation ledger that separates four facts for each logical request: submitted, effect possibly committed, CQE produced, and result consumed by the application. The checkpoint would preserve the frontier between these states rather than infer everything from SQ/CQ indices.

For read-only operations, replay may be safe when the data source itself has suitable semantics. For side-effecting operations, the runtime could require an application-supplied reconciliation key or idempotency rule. A storage write might bind to an application generation or transaction identifier. A network send might be restorable only when the protocol already carries sequence or request IDs. An accept might be delivered from a preserved completion rather than repeated if the connection already exists.

The key design constraint is that the kernel should not be expected to understand application transactions. The runtime records enough operation identity to connect kernel completion with the application's existing recovery protocol. When no such protocol exists, the honest answer may be "quiesce before checkpoint" rather than speculative replay.

Evaluation should inject checkpoint freezes at several points: before kernel consumption, while an operation is in flight, after the external effect but before CQE observation, after CQE production but before consumption, and after application acknowledgment. Count duplicate external effects, lost results, repeated protocol messages, and false quiescence requirements. Include cancellation races and non-cancelable disk I/O so that the design cannot rely on successful cancel as an oracle.

The academic question is how to define exactly-once or at-least-once recovery boundaries for an asynchronous kernel interface without pretending the kernel controls remote or durable effects. The production integration point is the I/O runtime layer where `user_data`, request metadata, and application transaction IDs already meet.

This direction loses if applications already expose enough durable idempotency and sequence information that ordinary process checkpointing plus replay gives the same result without a dedicated ledger.

### 3. Build an adversarial io_uring checkpoint benchmark

A useful benchmark should make the common wrong implementations fail visibly. It can combine a checkpoint controller with workloads whose ground truth is known:

- a file writer where every logical write has a unique generation and duplicate/missing writes are detectable;
- a TCP server using multishot accept and direct descriptors, with clients recording exactly which connections were established;
- a multishot receive workload using provided buffers, where each byte range and buffer ID has an ownership history;
- fixed-file table updates racing with in-flight requests;
- registered-buffer replacement where old memory remains referenced until completion;
- cancellation races, including operations that have already reached a non-cancelable state.

The benchmark should freeze at controlled lifecycle points and then either restore on the same kernel or migrate to another compatible host. Score failure classes separately: lost CQE, duplicate effect, stale request replay, wrong resource binding, buffer ownership violation, multishot truncation/restart, and unrecoverable external state. Report downtime and throughput only after correctness.

The strongest baseline is not a deliberately broken snapshotter. It is a conservative **drain-everything** design that waits until the ring is quiescent, recreates it, and resumes. The research system is useful only if it preserves correctness while reducing downtime or supporting requests that cannot cheaply reach a global idle point.

The academic contribution is a repeatable semantics benchmark for asynchronous-I/O migration. The production artifact is a qualification suite that runtime and checkpoint-system maintainers can run against new kernels, liburing releases, and application I/O patterns.

The benchmark is unnecessary if existing CRIU and io_uring test suites already expose the same logical failure classes with a comparable ground-truth oracle. A first experiment should try to prove that before inventing another suite.

## A practical checkpoint rule today

Until there is a stronger restore interface, an application or checkpoint system should prefer **explicit quiescence over guessed replay**.

Stop new submissions, cancel requests that are safely cancelable, and keep consuming CQEs until the runtime can classify what remains. Do not assume that closing an ordinary file descriptor terminates its pending `io_uring` operations. Record how fixed files, direct descriptors, registered buffers, and provided-buffer groups map to application objects. Treat multishot requests as live streams with a completion frontier rather than one-shot SQEs. For any side-effecting request whose completion/effect relationship is uncertain, either reconcile it through the application's own protocol or refuse to checkpoint at that point.

This does not mean every service needs a new kernel API. Many applications can intentionally build a short quiescent point and recreate the ring from ordinary durable state. That is the baseline a more ambitious checkpoint mechanism has to beat.

The earlier [io_uring programmability report](https://eunomia.dev/research/io-uring-bpf-programmability/) asks how policy and execution control should compose as more resources move behind `io_uring`. This report covers a different boundary: how those asynchronous resources survive process restart or migration. The [GPU checkpoint report](https://eunomia.dev/research/gpu-checkpoint-recovery-consistency/) provides a useful comparison because it also separates component restoration from a legal application recovery cut, but the concrete state here is Linux async I/O and ring-owned resources.

## What would change this conclusion?

The case for richer recovery metadata would weaken if CRIU or another general checkpoint system gains an `io_uring` restore interface that can prove correct reconstruction of live requests, completions, registered resources, multishot progress, and side-effect boundaries across the workloads above. A kernel-supported export/import mechanism could move much of the hard state out of userspace, although external effects would still need application semantics.

It would also weaken if measurements show that real `io_uring` services can always stop submissions and drain or cancel their rings within an acceptable migration pause. If conservative quiescence is cheap and reliable, replaying live asynchronous state is not worth the extra protocol.

Finally, some applications may already have strong idempotency. A replicated storage engine with durable request IDs or a network protocol with exact replay detection may safely reconstruct pending work from higher-level state. For those workloads, the right checkpoint contract can be much smaller than a generic operation ledger.

The boundary to remember is this: **an io_uring ring is not only shared memory. It is an asynchronous execution protocol with kernel-owned resources and effects that can outlive the userspace action that submitted them. A safe checkpoint must either quiesce that protocol or preserve enough identity and progress to recover it deliberately.**

## References

- CRIU, [`io_uring support in CRIU` issue #2131](https://github.com/checkpoint-restore/criu/issues/2131), open as of 2026-09-13.
- liburing manual, [`io_uring_cancelation(7)`](https://man7.org/linux/man-pages/man7/io_uring_cancelation.7.html), accessed 2026-09-13.
- liburing manual, [`io_uring_multishot(7)`](https://man7.org/linux/man-pages/man7/io_uring_multishot.7.html), accessed 2026-09-13.
- liburing manual, [`io_uring_registered_files(7)`](https://man7.org/linux/man-pages/man7/io_uring_registered_files.7.html), accessed 2026-09-13.
- liburing manual, [`io_uring_registered_buffers(7)`](https://man7.org/linux/man-pages/man7/io_uring_registered_buffers.7.html), accessed 2026-09-13.
- liburing manual, [`io_uring_provided_buffers(7)`](https://man7.org/linux/man-pages/man7/io_uring_provided_buffers.7.html), accessed 2026-09-13.
