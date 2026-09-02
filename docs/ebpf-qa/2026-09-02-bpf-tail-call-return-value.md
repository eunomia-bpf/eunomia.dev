# Why can't an eBPF program read `bpf_tail_call()`'s return value?

**Short answer:** because a BPF tail call is a control-transfer operation, not an ordinary helper call with a usable result. On success, execution jumps into the selected program and never returns to the caller. On failure, execution falls through to the next instruction, but the kernel verifier deliberately models the helper as `RET_VOID`. Therefore `R0` is unreadable after the call, and code that assigns or prints a supposed return value is rejected with an error such as `R0 !read_ok`.

The practical rule is simple: call `bpf_tail_call()` as a statement. Any code immediately after it is the failure path. Do not assign its apparent C return value.

## Why the C declaration is misleading

Many BPF helper declarations use a function-pointer-shaped C interface. Depending on the headers and generated documentation in use, `bpf_tail_call()` may look as though it returns `long`, and older helper text may even describe zero or a negative error. That surface syntax does not define what the verifier lets a BPF program consume.

The authoritative verifier prototype in the current Linux source sets `bpf_tail_call_proto.ret_type` to [`RET_VOID`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L3037-L3047). `RET_VOID` means that the call does not establish a readable value in `R0`. This is also reflected by libbpf's constant-slot wrapper, [`bpf_tail_call_static()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h#L133-L162), whose C return type is `void` even though it emits helper call 12.

That distinction explains this verifier sequence:

```text
call bpf_tail_call#12
...
w3 = w0
R0 !read_ok
```

The call itself is valid. The later attempt to copy `R0` is not. The ordinary BPF calling convention uses `R0` for function results, but only when the called operation actually defines one; the [BPF ABI guidance](https://docs.kernel.org/bpf/standardization/abi.html) does not override a helper's verifier prototype.

## What happens on success and failure

The interpreter makes the two paths explicit. It first rejects an out-of-range index, an exhausted tail-call budget, or an empty program-array slot. Those cases branch to `out` and continue at the caller's next instruction. If the slot is valid, the interpreter replaces the current instruction pointer with the selected program and resumes dispatch there. It never writes an error code to `R0`; see the [`JMP_TAIL_CALL` implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L2057-L2082).

The x86-64 JIT implements the same contract: it checks the index, chain budget, and target pointer, branches to a fall-through label when a check fails, and otherwise jumps to the target program. Its source-level pseudocode is documented directly above [`emit_bpf_tail_call_indirect()`](https://github.com/torvalds/linux/blob/master/arch/x86/net/bpf_jit_comp.c#L701-L715).

This means there cannot be a useful “success return value”: success has no returning continuation. Nor is there a portable failure code for BPF bytecode to inspect, because the failure paths do not define `R0`. Reading whatever native register happens to be present would be architecture-dependent and unsafe, so the verifier prevents it.

The selected program's eventual `EXIT` result becomes the result of the active program chain. It is not returned to the program that issued the tail call.

## The verifier-safe pattern

Write the tail call and its fallback as control flow:

```c
SEC("xdp")
int dispatch(struct xdp_md *ctx)
{
    __u32 slot = choose_slot(ctx);

    bpf_tail_call(ctx, &programs, slot);

    /* Reached only when the tail call did not transfer control. */
    count_tail_call_fallthrough(slot);
    return XDP_ABORTED;
}
```

Do not write:

```c
long err = bpf_tail_call(ctx, &programs, slot);
bpf_printk("tail call returned %ld", err);
```

Casting the result to an integer or storing it before printing does not help; every such form still asks the verifier to read an undefined `R0`. A source-level cast to `void`, or simply ignoring the expression result, is acceptable. When the slot is a compile-time constant and the toolchain supports it, `bpf_tail_call_static()` states the intended `void` contract more clearly and may enable JIT optimization.

## How to diagnose failure without a return code

Treat fall-through as one observable outcome, then validate likely causes at the layer that owns them:

1. Count fall-throughs in a per-CPU map, labeled by a small bounded dispatch class or slot. This measures failure without relying on an undefined register.
2. Prove the index is within the program array's declared range before the call. If the index is derived from packet or task data, clamp or reject it explicitly.
3. In user space, verify that every expected program-array slot is populated with a compatible loaded program, and recheck it after updates or reloads. Do not infer population from the BPF caller's return value.
4. Bound chain depth by design. A cycle or unexpectedly long dispatch chain eventually falls through when the kernel's tail-call budget is exhausted.
5. Exercise one known-valid slot, one empty slot, one out-of-range index, and a deliberately overlong chain. Assert both the final program return value and the caller's fall-through counter.

The caller cannot distinguish an empty slot from an exhausted chain budget through `bpf_tail_call()` itself. If operations require that distinction, record configuration state in user space and expose separate, non-sensitive health metrics. Avoid using a second mutable map as an assumed mirror unless updates to both maps have an explicit consistency protocol.

## What should documentation and tooling say?

The most accurate user-facing contract is: “On success, control transfers to the selected program and does not return. On failure, execution continues at the next instruction. No return value is available to the BPF program.” A C declaration that looks integer-valued should not be presented as permission to consume `R0`.

This is not merely a wording preference. Verifier acceptance, interpreter behavior, and JIT behavior must agree across architectures. The verifier's `RET_VOID` model prevents source code from depending on register contents that the execution engines intentionally leave unspecified. Documentation generators and library wrappers should express that control-flow contract, while tests should compile a failure-only continuation rather than assert a numeric helper result.

## References

- [Linux kernel: verifier prototype for `bpf_tail_call`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L3037-L3047)
- [Linux kernel: interpreter implementation of `JMP_TAIL_CALL`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L2057-L2082)
- [Linux kernel: x86-64 JIT tail-call checks and control transfer](https://github.com/torvalds/linux/blob/master/arch/x86/net/bpf_jit_comp.c#L701-L715)
- [libbpf: the `void` constant-slot `bpf_tail_call_static()` wrapper](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h#L133-L162)
- [Linux kernel documentation: BPF register and calling conventions](https://docs.kernel.org/bpf/standardization/abi.html)
- [Linux kernel selftests: program-array and tail-call test construction](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/test_verifier.c)

## Community discussion today

The visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question appeared within the strict 24-hour window, so the seven-day fallback was not used. This synthesis removes participant and channel identities, message links, exact times, private topology, raw logs, and searchable original wording. No raw transcript was retained, and no reply, reaction, direct message, follow, invitation, or moderation action was performed.

### Tail-call control flow exposed a documentation trap

The strongest troubleshooting thread began with a verifier rejection immediately after helper call 12. The key observation was that the program tried to move `R0` into another argument register. Discussion then compared the user-facing helper declaration with the verifier prototype and execution path, converging on the mismatch explained above: success cannot return, while failure fall-through does not define a readable result. The useful remediation is a failure-only continuation plus independent health counters, not a cast or a different integer type.

### Kernel review focused on failure-state correctness

The public kernel development archive was active around verifier backtracking, socket-reference lifetime, BPF LSM attachment, terminal control-flow instructions, batch-map overflow handling, and a compiler regression involving a socket context field. Although these patches address different subsystems, their shared concern is whether exceptional paths preserve ownership and register-state invariants. That same discipline applies here: a fall-through edge exists, but it must not manufacture a value that the execution engine never defined.

### Practitioners are measuring both overhead and semantic usefulness

A public practitioner forum featured one performance study and one process-behavior reconstruction project during the window. The discussions point to two complementary expectations for observability tools: quantify instruction and data-path cost, and show that low-level events can be correlated into a useful, reproducible explanation. For a tail-call dispatcher, that suggests reporting bounded fall-through counts and configuration state rather than emitting misleading per-call error numbers.

### Instrumentation work emphasized migration and reviewability

The active observability discussions were mostly implementation coordination: migrating library instrumentation, splitting dependent changes, and deciding which work was ready for review. No stronger in-window user question emerged there. The operational lesson is to make dependency and readiness state explicit; for BPF program arrays, the analogous requirement is to verify slot population and compatible program types before treating a dispatcher as healthy.

### Quiet targets were still checked

Project help and feature areas were either empty, automated-notification-only, or quiet in the window. The scheduling support surfaces had no new 24-hour question, and the eBPF instrumentation area had no new exchange in that period. These are accessible-but-quiet findings, not missing coverage disguised as zero activity.
