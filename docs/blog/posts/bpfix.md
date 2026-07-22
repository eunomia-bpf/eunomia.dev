---
date: 2026-07-25
slug: ebpf-verifier-errors-bpfix
description: A study of 235 reproduced eBPF verifier rejections shows that the terminal error identifies where verification stopped, not where the program lost the proof the verifier required. bpfix reconstructs the proof lifecycle from the verifier log to close the diagnostic gap.
---

# Why eBPF Verifier Errors Are Hard to Fix: The Diagnostic Gap

When developers load an eBPF program into the Linux kernel, the verifier must prove the program safe before any bytecode runs. The verifier walks every possible execution path through the program, tracking what it knows about each register and stack slot at each instruction. If it finds an instruction it cannot prove safe (a memory access through an unvalidated pointer, a read past the end of a packet, or an unbounded loop), it rejects the program and prints an error.

The problem is that the verifier's error message names the instruction where it got stuck, not the instruction where the program went wrong. The two can be far apart. A bounds check might be missing twenty instructions earlier; a pointer might have lost its type information after passing through a branch; a helper function might have returned a value the verifier can no longer track. The developer sees the final symptom, not the root cause.

The paper [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) studies this gap systematically. We reproduced 235 real verifier rejections under a fixed kernel and compiler, then asked: how much does the terminal error message actually tell you about what to fix? The answer is surprisingly little. `EINVAL` covers 47% of the cases; a single normalized error string can map to as many as nine completely different root causes.

<!-- more -->

## How the Verifier Thinks

To understand why verifier errors are so hard to debug, it helps to understand what the verifier is actually doing. The verifier performs abstract interpretation: it simulates executing the program without running it, tracking a simplified model of what each register and memory location might contain at each instruction.

This model is called the abstract state. For each register, the verifier might know: this is a pointer to the packet data, valid from offset 0 to 42. Or: this is a scalar whose value is between 0 and 255. Or: this is a pointer returned by `bpf_map_lookup_elem`, and the program has not yet checked whether it is null. These facts constrain what operations the program can safely perform.

The verifier builds these facts as it walks the program. When the program performs a bounds check (`if (ptr + 8 > data_end) return`), the verifier records that `ptr` is now known to be at least 8 bytes from the end. When the program calls a helper that returns a map value, the verifier records the pointer type and the requirement for a null check. When the program branches, the verifier explores both paths and tracks which facts hold on each.

These accumulated facts are the *proofs* that later instructions depend on. A packet read is safe only if the verifier can still see a proof that the access is within bounds. A map-value write is safe only if the pointer came from a lookup helper and passed a null check. The site's [eBPF security overview](https://eunomia.dev/blog/2024/02/11/the-secure-path-forward-for-ebpf-runtime-challenges-and-innovations/) covers the verifier's safety role more broadly; this post focuses on the diagnostic problem.

The critical point: proofs can be lost. A register that held a bounded packet pointer might get overwritten. A branch might merge two paths where one has a proof and one does not. The compiler might optimize away the operation that established the proof, or reorder instructions so the verifier no longer sees the connection. When this happens, the verifier rejects the program at the instruction that *needed* the proof, not at the instruction that *lost* it.

## Where Verification Stops vs. Where the Proof Was Lost

Consider a packet-parsing example from the paper. The program computes a UDP header pointer, checks it against `data_end`, then reads `dest`.

```c
if (udph + sizeof(struct udphdr) > data_end)
    return 1;

dst_port = __constant_ntohs(((struct udphdr *)udph)->dest);
```

![A real eBPF verifier rejection shown as source, verifier log, and proof-oriented diagnostic](imgs/bpfix-figure-1-diagnostic-gap.png)

Figure 1 from the paper puts three views side by side: the source reads from a UDP header, the raw verifier log stops at `R5 invalid mem access 'scalar'`, and the proof-oriented diagnostic identifies what the load required: a packet pointer that the verifier could still recognize at the dereference.

The snippet looks guarded, but the bytecode no longer preserves the packet-pointer proof at the load. The terminal line `R5 invalid mem access 'scalar'` says the verifier sees a scalar where it expected a packet pointer. It does not say when the packet pointer became a scalar, whether the source forgot a bounds check, whether compiler lowering merged provenance away, or whether the developer should rederive a pointer.

This is the diagnostic gap: the error names the symptom, not the cause. The source code might be entirely correct; the problem might be in how the compiler lowered it, or in how the verifier tracks types across branches. Alternatively, the source might have a real bug, but twenty lines earlier than the error points to. Either way, the terminal message alone does not tell you.

## What 235 Reproduced Rejections Reveal

To study this problem with real data, we assembled a corpus of verifier rejections that developers actually encountered. We started with 936 candidate reports from Stack Overflow questions, GitHub issues, GitHub fix commits, and Linux kernel selftests. Each candidate was rebuilt and loaded with Linux 6.15.11, clang 18, and verifier log level 2. Only 235 still produced a verifier rejection under that fixed setup; the rest depended on a different environment, no longer failed with the selected toolchain, or lacked the source material to rebuild.

This filtering gives the corpus a clear scope: a reproducible sample rather than an estimate of every verifier failure developers encounter. Each retained case includes the faulty source and the developer's own fix from the original report. Those paired artifacts allowed us to label both the root cause and the layer where the accepted repair landed.

Source changes repaired 191 cases, or 81% of the corpus. The remaining 44 cases involved source that was correct for the intended operation: 18 were repaired in the compiler, 14 through the environment, and 12 in the verifier. A context-field read compiled with `-O0`, for example, could lose its verifier-visible pointer type during lowering; changing the compilation setting repaired the program while leaving its C logic intact. The rejected instruction alone gives no way to choose among these layers.

The 191 source bugs fall into 12 root-cause categories. Ten are eBPF-specific, arising from the verifier's requirements around bounds, pointer provenance, object lifetimes, and helper protocols, concepts that do not exist in normal C programming.

| Root-cause category | Cases |
|---|---:|
| Unclamped scalar used as an offset or length | 24 |
| Corrupted or stale dynptr object | 23 |
| Packet access without a bound on every path | 22 |
| Missing null check | 19 |
| Pointer type or provenance mismatch | 16 |
| Unverified address dereferenced | 16 |
| Index exceeds object capacity | 15 |
| Context or contract misuse | 15 |
| Unpaired resource reference | 15 |
| Interrupt flag restored out of order | 11 |
| Probe signature mismatched with the ABI | 9 |
| Oversized or uninitialized stack buffer | 6 |

These categories require different repairs. Tightening a scalar range, preserving a packet bound across every path, checking a map lookup for null, and releasing a reference in the right order all satisfy the verifier, but they restore different facts. A developer who sees `invalid mem access 'scalar'` cannot tell from that message alone whether the fix is a bounds check, a null check, a type cast, or a compiler flag.

The paper measures this ambiguity by normalizing register numbers and offsets in the final verifier line. The 235 rejections produced 167 distinct strings, which collapsed to 82 message templates; 15 templates covered more than one root cause. The four most common templates show how quickly a familiar message spreads across unrelated mistakes.

| Terminal message template | Cases | Root-cause categories |
|---|---:|---:|
| `R# invalid mem access 'scalar'` | 28 | 9 |
| `invalid access to packet` | 26 | 5 |
| `invalid access to map value` | 18 | 4 |
| `R# !read_ok` | 13 | 4 |

`EINVAL` is broader still, appearing in 47% of all reproduced rejections.

The verifier log actually contains far more information than the terminal error: at log level 2, it prints the abstract state after every instruction. The problem is that developers must manually trace through this state to figure out where the proof was lost. The terminal line discards the history needed to connect the rejected operation to a root cause and a repair layer.

## From Rejection Location to Repair Information

A useful diagnostic would answer the questions the terminal error leaves open: what proof did the verifier need at the rejected instruction, and where did the program lose it?

The paper introduces a research prototype called [bpfix](https://github.com/eunomia-bpf/bpfix) that attempts this reconstruction. It reads the verifier's log-level-2 output (the per-instruction abstract states) and traces backward from the rejected operation. It identifies what proof was required (packet bounds, pointer provenance, null check, etc.), when that proof first appeared in the state, and when it disappeared. If debug metadata is available, it maps these transitions back to source lines.

A map-value case from the paper illustrates the difference. A developer cast the address of a BPF map object directly to a pointer and tried to write through it:

```c
__u64 *v = (__u64 *)&globals;
*v += 1;
```

The verifier rejected the write with `only read from bpf_array is supported`. The error names the rejected operation, but not the underlying problem: you cannot write through a map object pointer directly. The verifier expects a map-value pointer, which must come from a helper like `bpf_map_lookup_elem`. The correct fix follows that protocol:

```c
__u32 key = 0;
__u64 *v = bpf_map_lookup_elem(&globals, &key);
if (!v)
    return 0;
*v += 1;
```

The repair establishes the missing proof in three steps: look up the map element, check the returned pointer for null, then write through it. A diagnostic that names the required proof (a map-value pointer from a helper) and the loss point (the direct cast) gives the developer more to work with than the terminal error alone.

This approach also distinguishes cases that share an error message but need different fixes. One program constructs a packet address from an integer offset; it never establishes packet-pointer provenance and needs a source change. Another program derives and bounds-checks its pointer correctly, but compiler optimization merges the value into a scalar before the load; the fix is a compiler flag, not a source change. Both produce `invalid mem access 'scalar'`, but they belong to different repair layers.

## Can LLMs Fix Verifier Errors?

The paper tests whether better diagnostic context improves automated repair. If the diagnostic gap matters, models should perform better when given the missing proof information than when given only the raw verifier log.

We built bpfix-bench, a benchmark of 75 source-level repair tasks. Forty are constructed around specific verifier proofs that the repaired program must re-establish; 35 are minimized from open-source projects like Cilium, xdp-tools, and [bpftime](https://eunomia.dev/blog/2023/11/11/bpftime-extending-ebpf-from-kernel-to-user-space/).

Each task has an executable test suite independent of the diagnostic tool. A candidate fix must compile, load through the kernel verifier, pass a functional test, and pass a source-semantics check. The last requirement matters: it catches patches that make the error disappear by deleting the offending code path or changing the program's behavior. Success means restoring a verifier-acceptable program that still does what it was supposed to do.

The experiment compared two prompt conditions: one where the model received the raw verifier log, and one where it received a shorter diagnostic that named the required proof and relevant source span. Three models were tested at temperature zero: Qwen3.6 27B, GLM 5.2, and Qwen2.5 3B (as a lower-capacity comparison). One-shot mode judged the first candidate; retry mode returned failure information once and allowed a second attempt.

| Model | Raw log, one shot | Localized diagnostic, one shot | Raw log, one retry | Localized diagnostic, one retry |
|---|---:|---:|---:|---:|
| Qwen3.6 27B | 22/75 | 38/75 | 30/75 | 44/75 |
| GLM 5.2 | 28/75 | 38/75 | 47/75 | 52/75 |
| Qwen2.5 3B | 0/75 | 8/75 | 0/75 | 10/75 |

![Repair success across three models given a raw verifier log or a localized proof diagnostic](imgs/bpfix-figure-6-repair-success.png)

The results show a consistent improvement when models receive proof-localized context instead of raw logs. Qwen3.6 27B improved from 29% to 51% one-shot success; GLM 5.2 from 37% to 51%; Qwen2.5 3B from 0% to 11%. The gains persist with retry: Qwen3.6 goes from 40% to 59%, GLM from 63% to 69%.

The benchmark records the first stage where each one-shot candidate fails, which makes the aggregate result easier to interpret.

| Model | Input | Compile | Verifier load | Functional test | Source semantics | No program returned | Accepted |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B | Raw log | 3 | 19 | 9 | 22 | 0 | 22 |
| Qwen3.6 27B | Localized | 1 | 10 | 10 | 16 | 0 | 38 |
| GLM 5.2 | Raw log | 1 | 10 | 11 | 25 | 0 | 28 |
| GLM 5.2 | Localized | 1 | 5 | 9 | 22 | 0 | 38 |
| Qwen2.5 3B | Raw log | 7 | 62 | 0 | 3 | 3 | 0 |
| Qwen2.5 3B | Localized | 14 | 39 | 6 | 8 | 0 | 8 |

The stage-by-stage breakdown shows where the improvement comes from. For Qwen3.6, verifier-load failures dropped from 19 to 10 and source-semantics failures from 22 to 16. These are exactly the stages tied to restoring proofs while preserving behavior, which is the information the diagnostic adds.

The 3B model shows a different pattern. Verifier-load failures dropped from 62 to 39, and prompts that previously exceeded its context window now fit. But compile failures rose from 7 to 14; the model reached more repair attempts but still struggled with basic code generation.

These results are narrow: three models, 75 tasks, temperature zero, at most one retry. Even the best result (52/75 with retry) leaves many tasks unsolved. But the consistent improvement across models supports a specific conclusion: repair is easier when you know which proof was lost, not just where verification stopped.

## Debugging Verifier Errors in Practice

The paper's findings suggest a different approach to debugging verifier rejections. Instead of asking "what is wrong with this line?", start by asking what the rejected operation *required*.

Many verifier rejections happen because some instruction needed a proof the verifier did not have. A packet load needs packet-pointer provenance and a valid bound. A map-value write needs a pointer from a lookup helper with a null check. A dynptr slice needs a live dynptr object. Identifying the required proof is the first step.

The next step is tracing backward through the abstract states to see when that proof appeared, disappeared, or never existed. The verifier log at level 2 contains this information (it prints the state after every instruction), but you have to read it manually. Look for where a register changes from a typed pointer to a scalar, where a bound disappears after a branch merge, or where a required check never appears.

A practical reading order:

1. **Identify the rejected operation:** what kind of access is it? Packet read, map write, helper call?
2. **Name the required proof:** packet bounds, pointer provenance, null check, scalar range?
3. **Trace backward:** when did the register have that proof? When did it lose it? Did it ever have it?
4. **Determine the repair layer:** is the problem in source (missing check), compiler (optimization hid the proof), environment (wrong kernel version), or verifier (precision limit)?

Sometimes this trace leads outside the source. If the abstract states show a proof established and then discarded during compilation, the fix might be a compiler flag. If the source never established the proof, you need a code change. The terminal error is where you start; the proof history tells you where to fix.

## References

- [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748)
- [bpfix GitHub repository](https://github.com/eunomia-bpf/bpfix)
- [Linux kernel documentation on the eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [BPF Verifier Visualizer](https://github.com/libbpf/bpfvv)
- [An Empirical Study on the Challenges of eBPF Application Development](https://doi.org/10.1145/3672197.3673429)
- [bpftime userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
