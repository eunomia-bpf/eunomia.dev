---
date: 2026-09-16
slug: io-uring-cancel-terminal-state
title: "When io_uring Cancels I/O, Is the Operation Really Over?"
description: "io_uring cancellation is racy: an I/O may finish while cancel runs. This report designs terminal-state receipts, resource fences, and adversarial race tests."
tags:
  - Daily Report
  - Linux
  - io_uring
  - Asynchronous I/O
  - Cancellation
research_question: "What contract should io_uring runtimes use to distinguish a cancellation request from terminal operation completion, so buffers, file references, and side effects are not retired too early?"
source_cutoff: 2026-09-16
status: daily-report
---

# When io_uring Cancels I/O, Is the Operation Really Over?

A server submits an `io_uring` read, starts shutting a connection down, and sends `IORING_OP_ASYNC_CANCEL`. The cancel CQE says the request was found. Is it now safe to reuse the buffer, drop the file-related state, and tell the rest of the runtime that the operation never happened?

Not necessarily.

`io_uring` cancellation is a race with an operation that already has its own execution path. The target may finish normally before cancellation wins. It may be found but already be completing. A request that reached hardware may no longer be cancelable. Closing the application's file descriptor does not automatically cancel pending `io_uring` work because the ring can hold its own file reference. The API exposes enough information to build a correct runtime, but a single Boolean such as `cancelled = true` is too weak to describe the state transition.

The practical rule is simple: **a cancellation request is an attempt to change an operation's future, while the target operation's completion and any opcode-specific retirement events determine when its resources can actually be reused.** A robust runtime needs to track both.

<!-- more -->

## What io_uring cancellation CQEs actually tell you

Upstream liburing's current cancellation documentation describes two completion events. The cancellation operation gets its own CQE: success means a matching request was found and canceled; `-ENOENT` means no matching request was found; `-EALREADY` means the target was found but was already completing. Separately, the target request eventually gets its own CQE, commonly `-ECANCELED` when cancellation wins, but possibly a normal result or another error when it does not.

The order of those CQEs is not guaranteed. That matters because the cancel request and the I/O request represent different state machines. One says what happened to the *attempt to cancel*. The other says how the *original operation terminated*.

The current kernel implementation makes the same distinction visible. In `io_uring/cancel.c`, cancellation of io-wq work maps a running request to `-EALREADY` and a missing request to `-ENOENT`; it then continues checking other cancellation paths such as poll, waitid, futex, and timeouts. The result is intentionally more precise than a global "stopped" bit because different request classes reach different execution stages.

There is another boundary that surprises code written with synchronous I/O intuitions. liburing documents that closing a file descriptor does not automatically cancel pending operations submitted through the ring. `io_uring` has already taken the file reference needed by the request, so the integer descriptor can disappear from the application while the operation remains live.

Synchronous cancellation exists as well through `io_uring_register_sync_cancel()`. It is useful when a thread needs to block while cancellation is processed. It still does not turn every operation into something that can be rolled back after the device or protocol has crossed an irreversible boundary. Upstream documentation explicitly notes that operations already submitted to hardware, such as disk I/O in progress, typically cannot be canceled.

These facts suggest states that an application should not collapse:

| Runtime state | What is known | Safe action |
| --- | --- | --- |
| cancel requested | a cancellation operation was submitted | keep target resources live |
| cancellation accepted | cancellation found a target and executed the supported cancel path | wait for target and retirement evidence |
| target already completing | cancel returned `-EALREADY`, or the race otherwise became visible | wait for the target CQE |
| target completion observed | the original request emitted its final result CQE | evaluate any opcode-specific retirement obligation |
| resources retirable | all completion and ownership-release events required by that opcode are observed | retire or reuse resources |

For a multishot operation, the target is not terminal until the final CQE no longer carries `IORING_CQE_F_MORE`. Zero-copy sends add a different obligation: liburing documents a send-result CQE followed by a notification CQE with `IORING_CQE_F_NOTIF`, and the send buffer is not safe to reuse until that notification arrives. The useful boundary is therefore not "one target CQE". It is **the complete retirement evidence defined by the operation's completion protocol**.

This is different from the application-level effect ambiguity in [the earlier agent retry report](https://eunomia.dev/research/agent-tool-retry-effect-idempotency/). There, a network or process failure can hide whether an external side effect committed. Here, the kernel exposes explicit completion channels, but user space still has to preserve the distinction between cancellation control flow, target completion, and ownership release.

## Why common runtime shortcuts fail

A tempting implementation is to put each pending operation in a table keyed by `user_data`, submit a cancel request, and remove the entry as soon as the cancel CQE returns success. The table is tidy, but the lifetime rule is wrong. The original request still owns a completion obligation, and freeing or reusing its buffer, request object, or generation token can turn a benign race into use-after-reuse or result misattribution.

A second shortcut is to close the file descriptor and use that as the teardown barrier. This fails for the same reason: the ring's request has its own file reference. Descriptor identity and request lifetime are related but not identical.

A third shortcut is to interpret `-ENOENT` as proof that the operation never existed. It can also mean the operation completed before the cancellation lookup. If the application has not yet consumed or reconciled that target CQE, treating `-ENOENT` as "nothing happened" loses information.

A fourth shortcut is to stop after one target CQE for every opcode. That is enough for many one-shot operations, but it is wrong for completion protocols with additional lifetime events. A multishot request can produce more CQEs while `IORING_CQE_F_MORE` is set, and a zero-copy send buffer remains owned until its notification CQE. A generic runtime needs an opcode-aware retirement rule rather than one universal callback convention.

Finally, a timeout is not necessarily an effect rollback. A linked timeout can arrange cancellation of the linked request, but an operation can race with timeout handling just as explicit cancellation can. For network sends, storage writes, or device commands, the application must still define what a successful target completion means for visible effects and what cleanup is permitted after each terminal result.

## Where current work is still weak

The first gap is **a user-space terminal-state contract that spans operation classes**. `io_uring` correctly exposes operation-specific behavior and cancellation results, but runtimes often wrap them behind one future/promise abstraction. That abstraction needs a precise rule for when memory, registered resources, file-related state, and logical request identity stop being reachable by the kernel. The useful test is whether a generic runtime can run cancellation-heavy workloads under address sanitizers and generation-tagged buffers without stale completions being attributed to reused objects.

The second gap is **effect-aware cancellation semantics**. A final result CQE tells the runtime how one kernel request completed, but applications care about a higher boundary: whether bytes were sent, a write became visible through the filesystem, a device command became irreversible, a notification still owns a zero-copy buffer, or a multishot source can still produce events. The right contract differs by opcode and backing object. A useful evaluation needs deliberately racy cancellation points and an oracle for the specific effect being claimed, not just a count of `-ECANCELED` CQEs.

The third gap is **teardown evidence**. Production runtimes need to answer why a resource was freed: target completed normally, cancellation won, a zero-copy notification released a buffer, shutdown drained the ring, or an operation crossed into a non-cancelable stage and later completed. Today that reasoning is usually implicit in callbacks and counters. A crash dump with only `user_data` and result codes can be insufficient once identifiers are reused or operations have multiple completion events.

The fourth gap is **a cross-kernel, cross-opcode cancellation benchmark**. Cancellation behavior depends on where a request is executing: poll state, io-wq, protocol code, or hardware. A benchmark that tests only one socket read says little about file I/O, multishot network operations, zero-copy sends, `uring_cmd`, or requests that are already in flight on a device. A useful benchmark should report race outcomes by operation class and execution stage rather than one cancellation-success percentage.

## Promising directions with academic and production value

### 1. Give every asynchronous operation a terminal-state receipt

**Gap.** A cancel CQE describes the cancellation request, while the target completion protocol describes the target and resource ownership. Generic runtimes frequently need one auditable object that joins those facts before retirement.

**Mechanism.** Assign each logical operation a generation-stable identity and keep a small state record until its retirement contract is satisfied. The record would capture submission identity, opcode, owned resources, cancel attempts, cancel CQEs, target CQEs, `IORING_CQE_F_MORE`, notification CQEs such as `IORING_CQE_F_NOTIF`, and an operation-specific effect classification. State transitions are monotonic: a cancel attempt may move the operation into `cancel-pending`, but only the required completion and ownership-release evidence moves its resources into `retirable`.

A compact record could look like this:

```text
operation = {slot, generation}
opcode = SEND_ZC
resources = {fixed_file_slot, send_buffer_generation}
submit_seq = 1842
cancel_attempt = {seq=1901, result=0}
target_result = {result=4096}
retirement_event = {notif=true}
effect = peer_may_have_observed_payload
retire_after = retirement_event
```

**Delta.** This is not a new cancellation primitive. The kernel already supplies the relevant completion events. The new part is a runtime invariant that joins cancellation, target result, and operation-specific ownership release into one retirement proof instead of letting each callback infer lifetime independently.

**Artifact.** Build a small liburing-compatible runtime shim plus a trace/replay format. It should support ordinary one-shot requests, multishot operations, zero-copy sends, file-descriptor cancellation, linked timeouts, and synchronous cancellation.

**Evaluation.** Compare the receipt model with a conventional future/callback wrapper across socket reads/writes, zero-copy sends, poll, file I/O, and multishot operations. Force races by varying cancel delay around completion. Measure stale-completion attribution, premature resource reuse, memory-safety failures, state bytes per live request, completion latency, and throughput. The decisive ablation removes generation identity while keeping the same completion events; if reused `user_data` or slot IDs then cause false joins, the generation is doing real work.

**Academic value.** The general question is how to define a reusable terminal-state property for asynchronous operations whose cancellation, result delivery, and ownership release may be separate events.

**Production value.** Networking runtimes, storage engines, language runtimes, and proxies can use the receipt at their `io_uring` abstraction boundary to make teardown auditable and to prevent buffer or request-object reuse before the kernel is finished.

**Failure condition.** If a simpler opcode-specific state machine provides identical safety and diagnostic value across the supported operation set, the generalized receipt is unnecessary overhead.

### 2. Make resource retirement a fence, not a callback convention

**Gap.** Even a correct cancellation state machine can fail if the buffer pool, fixed-file table, connection object, or application request is recycled by another subsystem before the original operation releases ownership.

**Mechanism.** Introduce a lightweight retirement fence that holds the resource generation until every operation that references it has satisfied its retirement evidence. The fence is not a global drain. It can be scoped to a connection, fixed-file slot, buffer-group entry, send buffer, or request arena. Cancellation marks operations as no longer *wanted*; completion events discharge the references that make reuse safe.

For many ordinary one-shot requests, the final target CQE discharges the operation reference. For multishot operations, the final CQE without `IORING_CQE_F_MORE` does. For zero-copy sends, the notification CQE carrying `IORING_CQE_F_NOTIF` releases the buffer-lifetime obligation. Shutdown can aggregate the remaining operation identities rather than relying on file-descriptor closure as an implicit barrier.

**Delta.** Existing reference counting protects memory only when every subsystem holds and releases the right reference. The proposed fence makes asynchronous kernel ownership explicit and generation-aware, so a stale completion cannot silently attach to a newly allocated object that happens to reuse the same slot.

**Artifact.** Implement a connection/request arena with generation-tagged slots and per-scope retirement fences. Provide adapters for registered buffers, zero-copy send buffers, and fixed files, then integrate it into a small echo proxy and a file-I/O worker.

**Evaluation.** Stress repeated connect/cancel/close/reuse loops, zero-copy send-buffer reuse, buffer-ring recycling, fixed-file slot reuse, and ring teardown. Baselines are raw callback lifetime management and ordinary reference counting without generation checks. Metrics are stale CQEs accepted, premature buffer reuse, extra retained memory, teardown latency, slot-reuse delay, and steady-state throughput. Inject intentionally delayed result and notification CQEs after attempted slot reuse to test whether the generation fence rejects the old generation.

**Academic value.** This asks whether asynchronous kernel ownership can be modeled as a compositional lifetime capability rather than as library-specific cleanup code.

**Production value.** The mechanism targets the exact place where high-throughput runtimes are fragile: aggressive object and buffer reuse during disconnect storms, timeouts, and rolling shutdown.

**Failure condition.** If conventional reference counting plus opcode-specific completion handling provides identical safety with lower memory and latency under the same reuse stress, the generation fence should not be adopted.

### 3. Benchmark cancellation by race outcome and external effect

**Gap.** A test that reports "90% of cancels returned success" does not tell an operator whether an application is safe. A cancel can lose the race and still be correct if the target's normal completion is handled. Conversely, a test can collect many `-ECANCELED` results while still recycling resources too early.

**Mechanism.** Build an adversarial harness that controls the relative timing among submission, execution, cancellation, file-descriptor close, resource reuse, target-CQE consumption, and extra lifetime events such as zero-copy notifications. Label each run with both the kernel-visible completion sequence and an operation-specific external-effect oracle.

For network I/O, the peer can record bytes observed. For file writes, the harness can read back the target after completion to measure only post-completion read visibility; stable-storage or crash-survival claims require a separate persistence oracle and are outside this test unless explicitly added. For poll/multishot operations, it can count events emitted before and after the final CQE. For zero-copy send, it can attempt buffer mutation between the result CQE and notification CQE to verify that the harness catches premature reuse. For io-wq paths, controlled blocking points can widen the `-EALREADY` race window.

**Delta.** liburing already carries extensive regression tests. This benchmark targets a different question: whether a *user-space lifetime and effect contract* remains correct across race outcomes and operation families. It treats `0`, `-ENOENT`, `-EALREADY`, target success, target failure, `-ECANCELED`, `F_MORE`, and `F_NOTIF` as inputs to the oracle rather than as the final score.

**Artifact.** Release a workload matrix, race scheduler, ground-truth effect collectors, and a compact outcome schema that can run in CI across kernel versions.

**Evaluation.** Test multiple kernel versions and operation classes with fixed random seeds plus targeted race windows. Report forbidden resource reuse, misattributed completion, effect disagreement, cancellation latency, and test reproducibility. Include a no-cancellation baseline so the harness itself does not manufacture lifetime failures.

**Academic value.** The benchmark provides a measurement method for cancellation semantics under adversarial timing rather than assuming cancellation is one atomic event.

**Production value.** Runtime maintainers can gate kernel/liburing upgrades and teardown-path changes on the same race corpus that models connection storms and timeout-heavy workloads.

**Failure condition.** If ordinary liburing regression tests plus sanitizers already expose every contract violation found by the new harness, a separate benchmark adds little value.

## A practical runtime rule

Treat cancellation as control flow, not as resource reclamation.

Give the cancel SQE its own `user_data` so its CQE cannot be confused with the target. Keep the target's identity valid until its completion protocol reaches the point where the logical request can be retired. Handle `-ENOENT` as "not found by cancellation", not as proof that no completion exists. Handle `-EALREADY` as an explicit instruction to reconcile the target. For multishot requests, wait for the final CQE without `IORING_CQE_F_MORE`. For zero-copy sends, keep the send buffer valid until the `IORING_CQE_F_NOTIF` notification. Do not use `close(fd)` as a cancellation barrier for pending ring requests.

When the operation can create an externally visible effect, define that effect separately from kernel lifetime. A normal target completion after a cancel race is still a real completion and must be processed according to the application's protocol. The [atomic-write crash report](https://eunomia.dev/research/linux-atomic-write-crash-semantics/) made a similar distinction at the storage layer: one local mechanism does not automatically provide the larger application contract. Here the local mechanism is cancellation, and the larger contract is safe operation retirement plus effect reconciliation.

## What would change this conclusion?

This argument would weaken if `io_uring` gained a universal cancellation primitive whose successful completion guaranteed, for every opcode and backend, that the target could no longer complete, no longer owned any user-visible resource, and had produced no unreconciled external effect. Current APIs deliberately expose more varied states, so runtimes have to compose the stronger property themselves.

It would also weaken for applications that use only a narrow operation set with simple lifetime rules. If a runtime supports one-shot pollable reads, never reuses identities before consuming their target CQEs, and has no multishot, zero-copy send, or device I/O, a small state machine may already be enough. The proposed receipts and fences should earn their complexity on workloads that actually exercise reuse and cancellation races.

Finally, the research directions are falsifiable. If race-controlled tests across sockets, files, poll, multishot operations, zero-copy sends, and device-backed I/O show that a conventional opcode-aware wrapper has zero lifetime or attribution failures and the richer state adds no diagnostic value, the extra machinery should be discarded.

The useful mental model is therefore narrow: **cancellation requests try to stop future work; the operation's completion protocol proves when the original asynchronous operation and its resource ownership are over. Safe runtimes keep that boundary explicit.**

## References

- liburing, [`io_uring_cancelation(7)`](https://github.com/axboe/liburing/blob/master/man/io_uring_cancelation.7), upstream manual page, accessed 2026-09-16.
- liburing, [`io_uring_register_sync_cancel(3)`](https://github.com/axboe/liburing/blob/master/man/io_uring_register_sync_cancel.3), upstream manual page, accessed 2026-09-16.
- liburing, [`io_uring_prep_send_zc(3)`](https://github.com/axboe/liburing/blob/master/man/io_uring_prep_send_zc.3), zero-copy send completion and notification semantics, accessed 2026-09-16.
- Linux kernel, [`io_uring/cancel.c`](https://github.com/torvalds/linux/blob/master/io_uring/cancel.c), current cancellation implementation, accessed 2026-09-16.
- Linux kernel UAPI, [`include/uapi/linux/io_uring.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h), cancellation flags and CQE definitions, accessed 2026-09-16.
- Jens Axboe, [Reliable cancelation or wait for completion of specified SQE](https://github.com/axboe/liburing/discussions/608), liburing maintainer discussion, 2022-06-17.
