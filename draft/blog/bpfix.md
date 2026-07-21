---
date: 2026-07-25
slug: ebpf-verifier-errors-bpfix
description: Across 235 eBPF verifier rejections, bpfix shows why the final error hides the repair point and how proof-aware diagnostics guide real fixes.
---

# Why eBPF Verifier Errors Are Hard to Fix: 235 Rejections, Lost Proofs, and bpfix

An eBPF verifier error can look precise enough to be actionable, until the line it names keeps failing after the obvious fix. The kernel reports the instruction where verification stopped, but the repair often belongs earlier, at the point where the program lost the pointer, range, lifetime, or provenance proof the verifier later needed.

The [bpfix paper](https://arxiv.org/abs/2607.02748) studies that gap directly across 235 reproduced verifier rejections, where `EINVAL` was the errno in 47% of cases and one normalized terminal error string mapped to as many as nine distinct root causes. The verifier log contains useful state, but the final error line shows only the last frame of a proof story. This article presents our bpfix study and tool through that diagnostic gap, including the cases the current implementation still does not solve.

<!-- more -->

## eBPF Verifier Errors Are Proof Failures

Before an eBPF program runs in the kernel, the verifier has to prove that every path is safe. It does this by tracking abstract values for registers and memory at each instruction, building up facts such as whether a packet pointer is still within bounds, whether a map-value pointer came from the right helper, whether a dynptr is still valid, and whether a scalar range is tight enough for the access that follows. These facts are the proofs a later instruction can rely on, and they survive only as long as the verifier can see them in its abstract state. The site's [eBPF security overview](https://eunomia.dev/blog/2024/02/11/the-secure-path-forward-for-ebpf-runtime-challenges-and-innovations/) covers that safety role more broadly, while this post stays with the narrower diagnostic problem.

That model gives eBPF its safety boundary, but it also changes what a rejection means, because a C line can be rejected even when the source-level mistake happened several operations earlier, and the rejected instruction is simply the first place where the verifier needed a proof it no longer had.

Consider a packet parsing example from the bpfix paper. The program computes a UDP header pointer, checks the pointer against `data_end`, and then reads `dest`.

```c
if (udph + sizeof(struct udphdr) > data_end)
    return 1;

dst_port = __constant_ntohs(((struct udphdr *)udph)->dest);
```

![eBPF verifier diagnostic gap shown by bpfix on a real rejection](imgs/bpfix-figure-1-diagnostic-gap.png)

Figure 1 from the paper is useful because it puts the three views next to each other. The source line reads from a UDP header, the raw verifier log stops at `R5 invalid mem access 'scalar'`, and bpfix turns that last line into a source span plus the required proof, namely a verifier-recognized packet pointer at the rejected dereference.

The snippet looks guarded, but the bytecode no longer preserves the packet-pointer proof at the load. The terminal verifier line, `R5 invalid mem access 'scalar'`, says the verifier sees a scalar where it expected a packet pointer, yet it does not say when the packet pointer became a scalar, whether the source forgot a bounds check, whether compiler lowering merged pointer provenance away, or whether the developer should rederive a pointer that the verifier can recognize.

In this post, a proof is the verifier-visible fact that makes one instruction safe. A packet load needs a register that the verifier still classifies as a packet pointer and a range that stays inside `data_end`, a map-value write needs a pointer returned by the proper helper and checked for null, and a scalar offset needs a bounded range small enough for the object being accessed. The source may express the programmer's intent, but the verifier accepts only the facts that survive in its abstract state.

For a human, that difference matters because the line with the failed load may be perfectly reasonable source code, while the repair has to restore the verifier-visible proof before that load.

## Why the Terminal Verifier Error Is Too Coarse

bpfix starts from a practical question that most eBPF developers have met in some form. When the verifier rejects a program, how much does the terminal error actually narrow the repair?

The authors assembled `bpfix-empirical` from 936 candidate reports drawn from Stack Overflow questions, GitHub issues, GitHub fix commits, and Linux kernel selftests. They rebuilt each candidate under one fixed toolchain, Linux 6.15.11 with clang 18 at verifier log level 2, where the verifier prints abstract state after each instruction along with the terminal error. The 235 candidates that reproduced as verifier rejections became the study corpus.

That reproduction filter is important. The study is not counting every complaint about eBPF or every historical bug report. It keeps only cases that can still be rebuilt, loaded, and rejected under one kernel and compiler configuration, so every terminal message, root-cause label, and repair layer is measured against the same verifier behavior.

The first split changes how a developer should read the error, because in 191 of the 235 rejections, the developer's repair changed the program source, while the other 44 rejected correct source and were repaired outside the source, with 18 repaired in the compiler, 14 in the environment, and 12 in the verifier. A verifier-layer repair means the source program was judged correct after a kernel-side precision or verifier-behavior fix, so about one in five reproduced rejections did not ask the developer to change the eBPF program at all.

The 191 source bugs fell into 12 root-cause categories, and 10 of those categories were eBPF-specific. The common cases involved proof families visible to the verifier. A proof family is the class of fact the verifier is trying to establish, such as scalar range, dynptr lifetime, packet bounds on every path, null map lookup checks, and pointer provenance. Different proof families require different source-level repairs, so knowing which family the rejection belongs to narrows the repair target in a way that the terminal error string alone cannot.

A root cause, a proof family, and a repair layer answer different questions. The root cause names the developer or toolchain mistake, the proof family names the verifier-visible fact that went missing, and the repair layer names where the accepted fix belongs. Keeping those three labels separate is what prevents a terminal message from collapsing several repairs into one vague diagnosis.

The corpus also separates three questions that are easy to blur together when reading a raw log.

| Debugging question | What the paper measures | Why it changes the repair |
|---|---|---|
| Did the source need to change? | 191 of 235 rejections were repaired in source, while 44 were repaired in the compiler, environment, or verifier. | A terminal line can describe a real verifier failure even when the source program is not the layer to edit. |
| Which proof was missing? | The 191 source bugs covered 12 root causes, including scalar range, packet bounds, pointer provenance, dynptr lifetime, and map lookup checks. | The repair must re-establish the specific proof family, because silencing one error string is not enough. |
| How ambiguous is the last line? | The template `R# invalid mem access 'scalar'` covered 28 cases across nine root-cause categories. | The same terminal message can require different source edits, build changes, or verifier-side fixes. |

The terminal messages do not carry that structure. `EINVAL` was the errno in 47% of all reproduced rejections, and after normalizing registers and offsets, 167 distinct error strings collapsed into 82 templates. Fifteen templates each spanned more than one root cause, and the most common template, `R# invalid mem access 'scalar'`, covered 28 cases across nine root-cause categories.

That ambiguity is where bpfix enters the story. The terminal line still matters as the symptom, but the missing repair information sits in the proof lifecycle that led to that symptom.

## bpfix Reconstructs Where the Proof Was Lost

bpfix treats the verifier log as a proof trace. Under `log_level=2`, the verifier prints abstract state after each instruction, showing register types, scalar ranges, pointer provenance, and reference counts as the analysis progresses. The terminal line marks the rejection, but the earlier states often show when the relevant proof appeared, how long it survived, and where it became incompatible with the rejected operation.

bpfix first parses the terminal line and the per-instruction abstract states into a normalized evidence stream, abstracting away register numbers and offsets so that the same proof family can be recognized across different programs. From that stream, bpfix identifies the missing proof family, such as pointer provenance, packet bounds, scalar range, or map-value derivation, then tracks the evidence for that proof through the log. The final diagnostic reports the required proof, relevant source spans, the observed loss point when bpfix can see one, and guidance for re-establishing the proof.

![bpfix workflow from verifier log to proof reconstruction and repair-oriented diagnostic](imgs/bpfix-figure-3-workflow.png)

The workflow matters because bpfix does not need to understand every source-level intention before it can help. Since the evidence comes from the verifier's own abstract states and does not rely on an inferred model of developer intent, the analysis can work on programs bpfix has never seen, while source metadata improves how the result is displayed.

The useful diagnostic objects are small but different.

| Diagnostic object | What it answers | Why the developer cares |
|---|---|---|
| Rejected operation | Which instruction made the verifier stop? | This is the symptom and the anchor for reading the log. |
| Required proof | What fact did that operation need? | This names the verifier obligation the repair must restore. |
| Loss point | Where did evidence for that proof disappear? | This is often closer to the real source or lowering problem. |
| Repair layer | Should the fix land in source, compiler settings, environment, or verifier behavior? | This prevents editing correct C when the bytecode or kernel-side analysis is the problem. |

The map-value case in the paper shows a different proof family. The rejected program treats the map object itself as ordinary memory by casting `&globals` to a value pointer, so the terminal line points at the write even though the missing proof is earlier.

```c
__u64 *v = (__u64 *)&globals;
*v += 1;
```

A program can hold a pointer to a map object and still lack the proof needed to write a map value, because the verifier expects a value pointer derived through a helper. The repair follows that required proof directly.

```c
__u32 key = 0;
__u64 *v = bpf_map_lookup_elem(&globals, &key);
if (!v)
    return 0;
*v += 1;
```

That example is useful because the terminal line names only the rejected access, while the repair is a small protocol that establishes the missing fact. Look up the map element, check the returned pointer, then write through that pointer.

In the packet example, bpfix can report that the required proof is a verifier-recognized packet pointer at the rejected load. The log first shows the register as a packet pointer with bounds and later shows the same value as a scalar before the load, making that transition the useful debugging target.

The same idea separates bugs that look identical in the terminal line. One case may need a source fix because the code never derived a proper map-value pointer, while another may need a compiler or build-setting change because correct source was lowered into bytecode that hides the verifier-visible pointer. Both can end with `invalid mem access 'scalar'`, but they belong to different repair layers.

The compiler-lowering case makes the layer distinction concrete. The paper includes a correct source expression whose context-field read becomes a scalar in bytecode under one compilation setting, so the fix is a compiler flag that preserves verifier-visible pointer information and leaves the C source untouched.

The bpfix diagnostic does more than make verifier output prettier. Pretty output helps, but localization is the important move because the diagnostic names the proof family that failed and points to the program transition that made the proof unavailable.

The maintained 0.1.x CLI keeps that responsibility deliberately narrow. It reads verifier, build, `bpftool`, libbpf, Aya, or BCC logs produced by the workflow a developer already uses; optional object analysis can add control-flow context. The default path does not execute a loader command, replace the kernel verifier, check the semantics of an accepted program, or edit source automatically. bpfix supplies structured evidence for a repair, while the developer or repair agent still owns the change and must validate it against the kernel and the program's tests.

## Why This Matters for LLM Repair

The verifier's diagnostic gap becomes even more visible when a model tries to repair a rejected program. A human can sometimes infer missing verifier state by experience, but an LLM only gets the text it is given, and the raw terminal error often leaves too many plausible repairs.

To measure that effect, the paper builds `bpfix-bench`, a benchmark of 75 source-level verifier repair tasks. Forty tasks are constructed around a required verifier proof, meaning the task's verifier rejection can only be resolved by re-establishing a specific proof family in the eBPF program. The other 35 tasks are minimized from open-source projects including Cilium, xdp-tools, and [bpftime](https://eunomia.dev/blog/2023/11/11/bpftime-extending-ebpf-from-kernel-to-user-space/). Each task includes an executable test suite independent of bpfix, so the kernel verifier and task tests judge the repair separately from bpfix. A repair has to load through the kernel verifier and pass functional and source-semantics checks.

That last requirement makes `bpfix-bench` stricter than a compile-only patch benchmark. A candidate fix first has to return a program, compile, load through the kernel verifier, pass the functional test, and then pass a source-semantics check that rejects repairs which simply delete behavior or change the intended program. A model that silences the verifier by removing the rejected code path would fail the functional or semantics check, so the benchmark measures whether the diagnostic helps a model repair the same verifier-visible proof while preserving what the original program was supposed to do.

The paper evaluates Qwen3.6 27B, GLM 5.2, and Qwen2.5 3B. With the raw verifier log, one-shot repair ranged from 0% for Qwen2.5 3B to 37% for GLM 5.2, while replacing the raw log with the bpfix diagnostic improved one-shot repair by 11 to 21 percentage points. For Qwen3.6 27B, the anchor model in the evaluation, one-shot success rose from 22 of 75 tasks to 38 of 75, and with one failure-informed retry, the same model went from 30 to 44 accepted repairs.

![bpfix-bench repair success across three models with raw verifier logs and bpfix diagnostics](imgs/bpfix-figure-6-repair-success.png)

Each model contributes paired bars comparing the raw verifier log with the bpfix diagnostic, so the chart is best read as localization evidence. The input program and test suite stay the same, while the diagnostic text changes from a raw verifier log to a bpfix explanation of the missing proof and relevant span. When success rises under that intervention, the evidence points to a better repair target, with the program, model, and acceptance tests held fixed.

The failure-stage breakdown makes the result more specific. The bpfix diagnostic mainly reduced verifier-load failures and source-semantics failures, the two stages where a repair must restore the verifier-visible proof and still preserve the program's intended source semantics. Compile failures stayed low for Qwen3.6 27B and GLM 5.2, so the gain did not come from making ordinary code generation easier. For Qwen2.5 3B, the raw log produced no accepted one-shot repairs, while the bpfix diagnostic produced 8 accepted repairs and eliminated three context-window failures that had occurred when the full raw log exceeded the small model's input budget.

That pattern matters beyond bpfix itself. In this benchmark, what changed the repair outcome was proof context in the prompt, with the program, model, and test suite otherwise held fixed.

The remaining failures matter just as much as the gain. The strongest reported one-shot result accepts 38 of 75 repairs, leaving 37 unresolved; one failure-informed retry raises the count to 44, not to complete coverage. The experiment therefore supports localization as a useful input to repair, not the stronger claim that a proof-aware diagnostic turns current LLMs into reliable automatic eBPF repair systems.

## What to Check When the Verifier Fails

bpfix changes the first debugging question. Instead of starting with the rejected instruction alone, start with the proof that instruction required, since a packet load needs packet-pointer provenance and a valid bound, a map-value write needs a pointer derived through the right helper, and a dynptr access needs an object whose lifetime still matches the verifier's model.

That shift gives developers a better reading order for raw verifier logs, moving from the rejected operation to the required proof, then walking backward through the abstract states until the proof appeared, disappeared, or never existed. The repair layer often becomes clearer at that point, because source bugs, compiler lowering artifacts, environment problems, and verifier limitations can end in the same terminal string, but they do not call for the same fix.

A practical pass over the log has four checkpoints.

- Start at the rejected operation and name what kind of access it performs.
- Translate that access into the proof it required, such as packet-pointer provenance, a map-value pointer, a scalar range, or a live dynptr.
- Walk backward through the abstract states until the proof appears, disappears, or never appears.
- Decide the repair layer from that transition, because a missing source check, a lost compiler-visible pointer, and a verifier precision limit leave different traces.

That reading order will not always terminate in the source. When the abstract states show a proof that the source did establish and the bytecode then discarded, the next place to look is the compiler, not the C code.

## References

- [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748)
- [bpfix GitHub repository](https://github.com/eunomia-bpf/bpfix)
- [Linux kernel documentation on the eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [BPF Verifier Visualizer](https://github.com/libbpf/bpfvv)
- [An Empirical Study on the Challenges of eBPF Application Development](https://doi.org/10.1145/3672197.3673429)
- [bpftime userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
