# Why can a syscall-rewriting trampoline crash threads created by `clone` or `clone3`?

**Short answer:** an x86-64 tracer can turn a normal `syscall` into a function-like path that ends in `syscall; ret`. That works for an ordinary, single-return system call because the `call` into the trampoline put a continuation address on the current stack. It can fail for `clone` and `clone3`: the system call returns once in the parent and once in the child, but a thread child resumes with the new stack requested by the caller. The child reaches the trampoline's `ret` without the return address that was pushed on the parent's old stack. `ret` then treats the first word of the fresh child stack as an instruction pointer, commonly causing a seemingly unrelated `SIGSEGV` before the thread start routine runs.

The bug is not that Linux returned to the wrong instruction. It is that the rewriter introduced a stack-based return contract that the original `syscall` instruction did not have.

## The original instruction does not use the userspace stack

On x86-64, the hardware `SYSCALL` entry saves the userspace return instruction pointer in `rcx` and flags in `r11`. The Linux entry code explicitly notes that the instruction does not save anything on the userspace stack and does not change `rsp`.

An instruction rewriter may need more room than the original two-byte `syscall`, so one possible design replaces it with an indirect `call` into a nearby or specially mapped stub. The stub runs a hook and eventually executes the real system call:

```text
application
    call trampoline        # pushes application continuation on current stack

trampoline
    ... run hook ...
    syscall
    ret                    # pops continuation from current stack
```

For `read`, `write`, or another ordinary system call, control returns only to the calling thread. Its `rsp` still addresses the stack on which `call` pushed the continuation, so the final `ret` can work.

`clone` and `clone3` change that assumption. The raw interfaces return in both execution contexts: the parent receives the child's ID and the child receives zero. The kernel copies the parent's saved register image, sets the child's return register to zero, and, when a new stack was supplied, replaces the child's stack pointer with that requested stack. The instruction pointer still resumes after the `syscall` inside the trampoline.

The resulting control flow is asymmetric:

- **Parent path:** the next instruction is the trampoline's `ret`, and `rsp` still points into the original stack containing the pushed continuation. The return works.
- **Child with a new stack:** the next instruction is the same `ret`, but `rsp` points into the fresh child stack. No trampoline continuation was pushed there, so `ret` pops unrelated data and branches to it.
- **Child with a null, fork-like stack:** the child keeps a copied version of the parent's stack layout, so it may retain the expected continuation.

Intel defines near `RET` as loading the next instruction pointer by popping the top of the stack. The processor does not know which `call` the programmer intended to match. If the top word is zero, a data pointer, a canary, or another non-code value, the resulting fault address can be far removed from the tracer. That is why the symptom often looks like a libc or application crash rather than a syscall-rewriting bug.

## Why `pthread_create` is a high-value reproducer

glibc's current `pthread_create` path prepares `clone_args` with thread-sharing flags, an explicit stack address and size, TLS, and parent/child TID locations, then calls its internal clone wrapper. This makes a minimal `pthread_create`/`pthread_join` program a better regression test than a single-threaded loop that only issues `getpid`, `read`, or `write`.

A useful reproducer should prove all of these facts separately:

1. The same binary starts and joins a worker without instruction rewriting.
2. The rewriting mode is actually active; a stale preload path or an unloaded transformer must fail the test instead of silently producing a green result.
3. The parent observes successful thread creation.
4. The worker reaches its first instruction and changes an observable flag.
5. The parent joins the worker and the process exits normally.

The fourth check matters. A successful return from `pthread_create` in the parent does not prove that the child survived the return path or entered the user start routine.

## Diagnose the continuation, not only the fault address

Start with a minimal threaded program and compare rewriting off versus on. If only the rewritten case fails, inspect the path at the instruction boundary rather than assuming the crashing symbol is the cause.

On x86-64, capture the following state in both parent and child immediately after the real system call returns:

- system call number and return value;
- `rip`, confirming that both contexts resume after the same `syscall` in the trampoline;
- `rsp`, confirming whether the child switched to an explicitly supplied stack;
- the word at the top of each stack before `ret`; and
- the expected application continuation that the original rewriting `call` pushed.

Disassemble both the patched application site and trampoline. Verify that the site really transfers through the expected stub and that the fallback path is exactly the one being debugged. A fault reported as an instruction fetch from a data page is consistent with a bad `ret`, but it is not sufficient evidence by itself; the stack and continuation comparison establishes the mechanism.

Also distinguish the glibc wrapper from the raw interfaces. The glibc `clone()` wrapper arranges for the child to call a supplied function. The raw `clone` system call and `clone3` return from the system call in both parent and child. A rewriter operates below the library abstraction and must preserve the raw architectural behavior that the library depends on.

## Repair options

The safest design rule is: do not model a two-return system call as an ordinary one-return function unless the trampoline explicitly constructs a valid continuation for every returned context.

There are two practical repair families.

### 1. Bypass the hook for the special system call

Dispatch `clone` and `clone3` to a small untraced path that executes the real system call without entering a C hook. This avoids running complex hook code in the half-created child context, but it does not by itself fix a trailing `ret`: if the child uses a new stack, the untraced path must still provide a valid continuation or use a non-stack-based transfer.

This trade-off should be explicit in product behavior and tests. If creation calls bypass the hook, tracing output must not claim complete syscall coverage.

### 2. Seed the child stack with the continuation

For an x86-64 raw `clone` with a non-null child stack, reserve one word below the requested stack pointer, store the application continuation there, and pass the adjusted pointer to the kernel. The child's `ret` pops that address and restores `rsp` to the originally requested top of stack.

For `clone3`, `stack` identifies the lowest byte and `stack_size` the extent of the child's stack. A corresponding repair can store the continuation at `stack + stack_size - 8` and reduce `stack_size` by eight before the system call. The current bpftime x86-64 transformer uses this shape while routing both calls around the C hook. It leaves a null child stack on the fork-like path, where the parent stack layout is copied rather than replaced.

This repair is not a generic recipe to paste across architectures. It depends on:

- the architecture's syscall ABI and syscall numbers;
- stack growth direction and alignment;
- the exact raw `clone` argument order;
- `clone3` structure layout and size validation;
- whether the requested memory is writable;
- control-flow protection such as userspace shadow stacks; and
- the rewriter's own register, red-zone, vector-state, signal, and unwind contracts.

In particular, a regular-stack fix should not be described as covering Intel CET shadow-stack execution without a dedicated test. The ordinary and shadow return-address stacks must agree for `ret` to succeed when shadow stacks are active.

An alternative trampoline can transfer to a saved continuation with a `jmp` instead of `ret`, but the child still needs a trustworthy place from which to obtain that continuation. A pointer kept only on the parent's stack does not become valid merely because the final instruction changed.

## Regression tests should exercise both returned contexts

A complete test matrix is broader than “the process stopped crashing”:

- `pthread_create` followed by `pthread_join`, with proof that the worker ran;
- raw `clone` with an explicit child stack;
- raw `clone` with a null, fork-like stack where the flags permit it;
- `clone3` with an explicit stack and valid minimum size;
- invalid stack and size inputs, confirming that error behavior is unchanged;
- many concurrent creations, so stack preparation and continuation storage are not accidentally shared;
- real syscalls issued from each child after startup;
- a single-threaded victim, confirming no regression on the common path;
- tracing-enabled and tracing-disabled modes, with an assertion that the transformer was actually loaded; and
- every supported architecture and control-flow-protection mode, rather than assuming the x86-64 result transfers.

Check semantic coverage as well as process exit status. If the repair deliberately bypasses creation calls, assert that downstream consumers see the documented gap. If the hook is retained, verify parent and child return values, preserved registers, stack alignment, signal behavior, and unwindability.

The bad `ret` is a control-flow integrity defect because it derives an instruction pointer from unrelated stack contents. A crash is the demonstrated outcome; exploitability depends on memory layout and mitigations and should not be inferred without separate evidence.

## References

- [Linux `clone(2)` manual: raw return behavior and child-stack semantics](https://man7.org/linux/man-pages/man2/clone.2.html)
- [Linux x86 `copy_thread`: child return value and optional stack-pointer replacement](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/process.c)
- [Linux x86-64 syscall entry: `SYSCALL` saves no data on the userspace stack and does not change `rsp`](https://github.com/torvalds/linux/blob/master/arch/x86/entry/entry_64.S)
- [Intel 64 and IA-32 instruction reference for `RET`](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2b-manual.pdf)
- [glibc `pthread_create` implementation and its `clone_args` setup](https://github.com/bminor/glibc/blob/master/nptl/pthread_create.c)
- [Current bpftime x86-64 syscall transformer and `clone`/`clone3` handling](https://github.com/eunomia-bpf/bpftime/blob/master/attach/text_segment_transformer/text_segment_transformer.cpp)
- [Intel overview of Control-flow Enforcement Technology and shadow-stack returns](https://www.intel.com/content/www/us/en/developer/articles/technical/technical-look-control-flow-enforcement-technology.html)
- [Linux BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf)
- [Linux BPF verifier implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c)
- [Linux AF_XDP socket implementation](https://github.com/torvalds/linux/blob/master/net/xdp/xsk.c)
- [OpenTelemetry GenAI operation-cost proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)
- [OpenTelemetry GenAI event conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained.

### Runtime instrumentation must preserve the system call's control-flow shape

The strongest engineering question concerned a syscall-rewriting mode that worked for single-threaded programs but failed when a program created a thread. The key debugging shift was from “which library address crashed?” to “which continuation exists in each returned context?” Linux's [x86 child setup](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/process.c) and [syscall entry contract](https://github.com/torvalds/linux/blob/master/arch/x86/entry/entry_64.S) explain why a `call`/`ret` wrapper adds an obligation that raw `clone` did not have. The current [runtime transformer](https://github.com/eunomia-bpf/bpftime/blob/master/attach/text_segment_transformer/text_segment_transformer.cpp) now gives the child a continuation and keeps creation calls out of the C hook.

The broader lesson is about coverage evidence. A green single-threaded benchmark can prove that ordinary returns work while saying nothing about system calls that create a second userspace continuation. Loader success is also not proof that the intended transformer was present. Tests need an observable child-side action and an explicit assertion that rewriting was enabled.

### Kernel review focused on failure paths, ownership, and diagnostic precision

The public kernel surface was busy. Repeated themes included private-stack JIT state, stack bounds during link update, resizable-hash-table lifetime, arena reclaim under memory limits, page ownership in generic XDP and AF_XDP refill paths, and clearer verifier diagnostics for composite return values. These are different subsystems, but they share an invariant: an error, retry, or alternate execution path must not retain stale ownership or an impossible machine state.

The practical response is to turn each review concern into a failure-injection or boundary test under [BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf). For verifier work, diagnostics should identify the exact member or register state that makes a return type unsupported rather than only rejecting the program; the [verifier implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) is the semantic boundary. For packet-buffer work, retry and release tests must show that [AF_XDP socket state](https://github.com/torvalds/linux/blob/master/net/xdp/xsk.c) transfers ownership exactly once.

### Observability conventions are still separating observed data from standardized meaning

Another active discussion asked how to represent model-operation cost when exporters already emit proprietary attributes. The public [operation-cost proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/287) remains open, so an exporter-specific field should not be presented as a released OpenTelemetry convention. Producers also need to say whether a number is billed, estimated, or derived, and which currency or unit applies.

Discussion also continued around linking an evaluation result to external evidence. The useful boundary remains: standardize an optional reference and integrity binding without turning telemetry into the artifact store or implying that the evidence was trusted. The [current GenAI event convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) should remain the baseline until new names complete the specification process.

### Quiet targets were still checked

The scheduler help surfaces, public practitioner forum, and most project-specific support channels had no substantive new technical exchange in the daily window. One networking surface was primarily discussing meeting and contribution onboarding. Several project channels contained only introductions, automated maintenance notices, or no messages. The newest public forum post was about a crash-safe BPF loader but was outside the 24-hour window, so it was not used as fallback evidence.
