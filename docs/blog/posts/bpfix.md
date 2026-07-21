---
date: 2026-07-25
slug: ebpf-verifier-errors-bpfix
description: A study of 235 reproduced eBPF verifier rejections shows that the terminal error identifies where verification stopped, not where the program lost the proof the verifier required. bpfix reconstructs the proof lifecycle from the verifier log to close the diagnostic gap.
---

# Why eBPF Verifier Errors Are Hard to Fix: The Diagnostic Gap

An eBPF verifier rejection ends with a concrete instruction and a terse message, which makes the failure look more localized than it is. The named instruction is where verification stopped; the source-level mistake may lie several operations earlier, where the program lost a pointer type, scalar range, lifetime, or provenance fact that a later access required.

The paper [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) examines this problem through 235 failures reproduced under a single kernel and compiler configuration. Its central question is diagnostic: how much repair information survives in the terminal message, and what additional context does a human or LLM need to choose the right fix? The study finds that `EINVAL` covers 47% of the corpus and that a single normalized message can map to as many as nine distinct root causes.

<!-- more -->

## A Rejection Is the End of a Proof Lifecycle

Before an eBPF program runs in the kernel, the verifier must prove that every execution path is safe. It tracks abstract values for registers and memory at each instruction, building facts such as whether a packet pointer remains within bounds, whether a map-value pointer came from the correct helper, whether a dynptr is still valid, and whether a scalar range is tight enough for the access that follows. These facts are the proofs a later instruction can rely on, and they survive only as long as the verifier can see them in its abstract state. The site's [eBPF security overview](https://eunomia.dev/blog/2024/02/11/the-secure-path-forward-for-ebpf-runtime-challenges-and-innovations/) covers the verifier's safety role more broadly; this post focuses on the diagnostic problem.

That model gives eBPF its safety boundary, but it also changes what a rejection means. A C line can be rejected even when the source-level mistake happened several operations earlier; the rejected instruction is the first place where the verifier needed a proof it no longer had.

Consider a packet-parsing example from the paper. The program computes a UDP header pointer, checks it against `data_end`, then reads `dest`.

```c
if (udph + sizeof(struct udphdr) > data_end)
    return 1;

dst_port = __constant_ntohs(((struct udphdr *)udph)->dest);
```

![A real eBPF verifier rejection shown as source, verifier log, and proof-oriented diagnostic](imgs/bpfix-figure-1-diagnostic-gap.png)

Figure 1 from the paper puts three views side by side: the source reads from a UDP header, the raw verifier log stops at `R5 invalid mem access 'scalar'`, and the proof-oriented diagnostic identifies what the load required: a packet pointer that the verifier could still recognize at the dereference.

The snippet looks guarded, but the bytecode no longer preserves the packet-pointer proof at the load. The terminal line `R5 invalid mem access 'scalar'` says the verifier sees a scalar where it expected a packet pointer. It does not say when the packet pointer became a scalar, whether the source forgot a bounds check, whether compiler lowering merged provenance away, or whether the developer should rederive a pointer.

In this post, a *proof* is the verifier-visible fact that makes an instruction safe. A packet load needs a register the verifier still classifies as a packet pointer, with a range that stays inside `data_end`. A map-value write needs a pointer returned by the proper helper and checked for null. A scalar offset needs a bounded range small enough for the object being accessed. The source may express the programmer's intent, but the verifier accepts only the facts that survive in its abstract state.

For a human, that difference matters: the line with the failed load may be reasonable source code, while the repair must restore the verifier-visible proof before that load.

## What 235 Reproduced Rejections Reveal

The empirical study began with 936 candidate reports from four sources: Stack Overflow questions, GitHub issues, GitHub fix commits, and Linux kernel selftests. Each candidate was rebuilt and loaded with Linux 6.15.11, clang 18, and verifier log level 2. Only 235 still produced a verifier rejection under that fixed setup; the rest depended on a different environment, no longer failed with the selected toolchain, or lacked the source material to rebuild.

This filtering gives the corpus a clear scope: a reproducible sample rather than an estimate of every verifier failure developers encounter. Each retained case includes the faulty source and the developer's own fix from the original report. Those paired artifacts let the authors label both the root cause and the layer where the accepted repair landed.

Source changes repaired 191 cases, or 81% of the corpus. The remaining 44 cases involved source that was correct for the intended operation: 18 were repaired in the compiler, 14 through the environment, and 12 in the verifier. A context-field read compiled with `-O0`, for example, could lose its verifier-visible pointer type during lowering; changing the compilation setting repaired the program while leaving its C logic intact. The rejected instruction alone gives no way to choose among these layers.

The 191 source bugs fall into 12 root-cause categories. Ten arise from eBPF-specific contracts: verifier-visible bounds, provenance, object lifetimes, and helper protocols.

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

These categories describe different repairs. Tightening a scalar range, preserving a packet bound across every path, checking a map lookup for null, and releasing a reference in the right order all satisfy the verifier, but they restore different facts. The terminal message rarely identifies that distinction.

The paper measures this ambiguity by normalizing register numbers and offsets in the final verifier line. The 235 rejections produced 167 distinct strings, which collapsed to 82 message templates; 15 templates covered more than one root cause. The four most common templates show how quickly a familiar message spreads across unrelated mistakes.

| Terminal message template | Cases | Root-cause categories |
|---|---:|---:|
| `R# invalid mem access 'scalar'` | 28 | 9 |
| `invalid access to packet` | 26 | 5 |
| `invalid access to map value` | 18 | 4 |
| `R# !read_ok` | 13 | 4 |

`EINVAL` is broader still, appearing in 47% of all reproduced rejections. These measurements do not say the verifier log lacks information; log level 2 records abstract state after each instruction. They show that the terminal line discards the history needed to connect the rejected operation to a root cause and a repair layer.

## From Rejection Location to Repair Information

A useful diagnostic must reconstruct the part of the proof lifecycle that the terminal line omits. Starting from the rejected operation, it identifies the verifier-visible fact required at that point, traces when evidence for that fact appeared or disappeared, and distinguishes the layer responsible for restoring it. The rejected instruction anchors the trace, the required proof describes the obligation, the loss point narrows the relevant transition, and the repair layer tells the developer whether to inspect source, compiler output, environment, or verifier behavior.

The paper implements this reconstruction in a research prototype called [bpfix](https://github.com/eunomia-bpf/bpfix), which reads log-level-2 abstract states and maps evidence back to source spans when debug metadata is available. The prototype tests whether per-instruction state can recover the repair information that the terminal line omits.

A map-value case in the paper shows why these pieces belong together. The program treats the map object itself as ordinary memory by casting `&globals` to a value pointer, so the terminal line points at the write even though the mistake occurred when the pointer was derived.

```c
__u64 *v = (__u64 *)&globals;
*v += 1;
```

A program can hold a pointer to a map object while lacking the proof needed for a map-value write, because the verifier expects a value pointer derived through a helper. The accepted repair follows that protocol.

```c
__u32 key = 0;
__u64 *v = bpf_map_lookup_elem(&globals, &key);
if (!v)
    return 0;
*v += 1;
```

The terminal line names only the rejected access; the repair establishes the missing proof in three steps: look up the map element, check the returned pointer for null, then write through it. In the earlier packet example, the same reasoning applies: a register first appears as a bounded packet pointer and later appears as a scalar before the load. The transition between those states is more useful for repair than the final load alone.

Repair-layer attribution also prevents two identical messages from being treated as one bug. A source program that constructs a packet address from an integer offset never establishes packet-pointer provenance and needs a source change. Another program derives and bounds-checks its pointer correctly, yet compiler lowering merges the value into a scalar before the load; the accepted repair changes the compilation behavior. Both cases end with `invalid mem access 'scalar'`, but their fixes belong to different layers.

## LLM Repair with Better Context

The paper uses automated repair to test whether proof context changes downstream decisions. Its benchmark, bpfix-bench, contains 75 source-level verifier repair tasks: 40 constructed around a specific verifier proof that the repaired program must re-establish, and 35 minimized from open-source projects such as Cilium, xdp-tools, and [bpftime](https://eunomia.dev/blog/2023/11/11/bpftime-extending-ebpf-from-kernel-to-user-space/). The two groups combine controlled proof failures with failures from real project histories.

Every task ships with an executable test suite maintained separately from the diagnostic. A candidate must first return a program and compile; it must then load through the kernel verifier, pass a functional test, and pass a source-semantics check. That final check catches superficially successful patches that remove the rejected path or alter the intended behavior. Acceptance therefore means more than making the error disappear: the patch must restore a verifier-acceptable program while preserving the task's semantics.

Three models were evaluated at temperature zero: Qwen3.6 27B as the primary model, hosted GLM 5.2, and Qwen2.5 3B as a lower-capacity comparison. For each task, the model received the same buggy program and produced a candidate patch; the experimental variable was whether the prompt contained the raw verifier log or a shorter diagnostic naming the required proof and relevant source span. One-shot mode judged the first candidate; retry mode returned failure information once and allowed a second attempt.

| Model | Raw log, one shot | Localized diagnostic, one shot | Raw log, one retry | Localized diagnostic, one retry |
|---|---:|---:|---:|---:|
| Qwen3.6 27B | 22/75 | 38/75 | 30/75 | 44/75 |
| GLM 5.2 | 28/75 | 38/75 | 47/75 | 52/75 |
| Qwen2.5 3B | 0/75 | 8/75 | 0/75 | 10/75 |

![Repair success across three models given a raw verifier log or a localized proof diagnostic](imgs/bpfix-figure-6-repair-success.png)

In the one-shot comparison, the absolute improvement ranges from 11 to 21 percentage points across the three models. Qwen3.6 shows the largest gain, from 29.3% to 50.7%; GLM rises from 37.3% to 50.7%; the 3B model rises from 0% to 10.7%. The retry results address a different question: failure feedback raises the counts for the two larger models under both prompt conditions, yet the localized context still retains an advantage.

The benchmark records the first stage where each one-shot candidate fails, which makes the aggregate result easier to interpret.

| Model | Input | Compile | Verifier load | Functional test | Source semantics | No program returned | Accepted |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B | Raw log | 3 | 19 | 9 | 22 | 0 | 22 |
| Qwen3.6 27B | Localized | 1 | 10 | 10 | 16 | 0 | 38 |
| GLM 5.2 | Raw log | 1 | 10 | 11 | 25 | 0 | 28 |
| GLM 5.2 | Localized | 1 | 5 | 9 | 22 | 0 | 38 |
| Qwen2.5 3B | Raw log | 7 | 62 | 0 | 3 | 3 | 0 |
| Qwen2.5 3B | Localized | 14 | 39 | 6 | 8 | 0 | 8 |

For Qwen3.6, verifier-load failures drop from 19 to 10, and source-semantics failures from 22 to 16. GLM follows the same pattern: load failures fall from 10 to 5, source-semantics failures from 25 to 22. These are the stages most closely tied to restoring the verifier-visible proof while preserving program behavior.

The smaller model exposes a different tradeoff. Its verifier-load failures drop from 62 to 39, and the three prompts that previously exceeded its context window now return programs, but compile failures rise from 7 to 14. Shorter, more targeted context helps the model reach a plausible repair attempt, but limited capacity still constrains code generation and semantic preservation.

The scope of these results is narrow. The best one-shot count is 38 of 75, so half the tasks still fail; even the best retry result accepts only 52 of 75. The experiment covers three models, one benchmark, temperature-zero generation, and at most one retry; it measures accepted patches rather than developer time or production reliability. Within those boundaries, the evidence supports a specific conclusion: repair improves when the model receives the missing proof and its location, not just the instruction where verification stopped.

## What to Check When the Verifier Fails

The corpus suggests a more useful first debugging question than "what is wrong with this line?" Start by asking what the rejected operation required: a packet load needs packet-pointer provenance and a valid bound; a map-value write needs a pointer derived through the correct helper; a dynptr access needs an object whose lifetime still matches the verifier's model.

This shift gives developers a better reading order for raw verifier logs: move from the rejected operation to the required proof, then walk backward through the abstract states until the proof appeared, disappeared, or never existed. The repair layer often becomes clear at that point: source bugs, compiler-lowering artifacts, environment problems, and verifier limitations can all produce the same terminal string, but they do not call for the same fix.

A practical pass over the log has four checkpoints:

- Start at the rejected operation and name what kind of access it performs.
- Translate that access into the proof it required, such as packet-pointer provenance, a map-value pointer, a scalar range, or a live dynptr.
- Walk backward through the abstract states until the proof appears, disappears, or never appears.
- Decide the repair layer from that transition: a missing source check, a lost compiler-visible pointer, and a verifier precision limit leave different traces.

This reading order sometimes leads outside the source. When the abstract states show a proof established by the program and then discarded in bytecode, compiler output becomes the next place to inspect. A verifier limitation or environment mismatch leaves a different trace. The final error remains the starting point; the proof history determines where the repair belongs.

## References

- [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748)
- [bpfix GitHub repository](https://github.com/eunomia-bpf/bpfix)
- [Linux kernel documentation on the eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [BPF Verifier Visualizer](https://github.com/libbpf/bpfvv)
- [An Empirical Study on the Challenges of eBPF Application Development](https://doi.org/10.1145/3672197.3673429)
- [bpftime userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
