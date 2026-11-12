---
date: 2026-09-07
title: "Can You Debug the eBPF Program That Actually Ran?"
description: "Dynamic eBPF specialization can replace bytecode and JIT images at runtime. This report defines the provenance needed to debug the code that actually ran."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Debugging
  - Observability
  - Compilers
research_question: "What provenance must a dynamically specialized eBPF runtime preserve so an operator can reconstruct which BPF generation, optimization assumptions, and native JIT image actually executed during an incident?"
source_cutoff: 2026-09-07
status: daily-report
---

# Can You Debug the eBPF Program That Actually Ran?

Imagine a production XDP program that is re-specialized several times during one afternoon. A userspace optimizer sees a stable branch bias, emits a faster BPF variant, loads it through the normal verifier and JIT, then later falls back after the workload changes. At 14:07, latency spikes for thirty seconds. By the time an operator opens `bpftool`, the optimized generation is already gone.

The source file is still available. The currently loaded BPF program is inspectable. The incident trace is available. None of those facts alone proves which rewritten BPF instructions and which native JIT image handled the packets at 14:07, or why that generation had been selected.

Linux already provides unusually good BPF introspection. Program IDs, tags, translated instructions, JITed machine code, BTF line information, runtime statistics, and load metadata are all useful. The missing boundary appears when optimization becomes dynamic: **debugging needs an execution-time provenance chain that binds an observation to the exact specialization generation, the assumptions and transformations that created it, and the native image that was active then.**

<!-- more -->

This question is narrower than the previous report on [runtime profile-guided eBPF specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/). That report asks when a rewrite is still semantically justified and proposes generation-scoped equivalence and assumption evidence. Here we assume the optimizer has already produced a valid generation. The failure is operational: can a postmortem connect a sample, packet, or latency interval back to that generation after later re-JITs and deoptimizations?

It is also different from [architecture-specific specialization portability](https://eunomia.dev/research/ebpf-portable-architecture-specialization/). Portability asks whether a native fast path is eligible on this machine and whether a portable fallback exists. Provenance asks which eligible implementation actually ran at one point in time and how an operator can prove it.

## Linux can show the loaded BPF program and its JIT image

The current BPF userspace API already exposes much of the state a debugger would want.

The kernel's [`struct bpf_prog_info`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) includes a program ID and tag, translated-program length and bytes, JITed-program length and bytes, BTF identity, function information, source line information, JITed line information, map IDs, load time, and optional runtime statistics. The [BTF documentation](https://kernel.org/doc/html/next/bpf/btf.html) explains that an introspection tool can retrieve `bpf_prog_info`, BTF, translated bytecode, and JIT line information and use them to print line information alongside BPF and JIT code.

`bpftool` turns these interfaces into practical commands. Its current [`prog dump`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8) interface can dump translated BPF instructions, dump the JITed host image, emit raw opcodes, and show source lines when line information exists. `prog show` can expose IDs, tags, load time, translated and JITed sizes, maps, process holders, and runtime counters when statistics are enabled.

This is a strong baseline. A static BPF deployment can often answer: what program is loaded now, what did the verifier/JIT produce, and where does this instruction map back to source?

The Linux tag is also useful as a compact identity. In current [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c), `bpf_prog_calc_tag()` hashes the BPF instruction stream after normalizing unstable map file-descriptor fields. The kernel also uses that tag when naming JITed BPF symbols. That gives tools a stable content-derived handle for the loaded BPF program rather than only a process-local file descriptor.

But a tag answers an identity question, not a history question.

## Dynamic re-JIT creates a version-history problem

The public [BpfReJIT design](https://lpc.events/event/20/contributions/2445/) makes the problem concrete. Its userspace shim can intercept load and attach operations, rewrite bytecode based on configuration, workload behavior, and kernel version, and then submit each candidate through the normal verifier and JIT. Runtime speculative optimization is explicitly part of the design.

Once a runtime can do that, one logical application may produce a sequence like:

```text
source object
   |
   +-- generation 41: portable BPF -> JIT image A
   |
   +-- generation 42: branch-specialized BPF -> JIT image B
   |
   +-- generation 43: deoptimized BPF -> JIT image C
```

All three generations can be verifier-safe. All three can have valid BPF tags. Each one may even carry useful BTF line information. The debugging gap is that these facts are normally attached to individual loaded objects, while an incident is attached to **time and execution**.

An operator therefore needs to answer four different questions:

1. Which generation was active for this hook or link at the time of the observation?
2. Which source artifact and transformation sequence produced that BPF generation?
3. Which profile, configuration, capability, or other assumption caused the optimizer to select it?
4. Which JIT backend and native image actually executed on that machine?

Current kernel introspection exposes valuable pieces of this chain. It does not define one durable record connecting all four.

## BTF line information is source mapping, not optimization provenance

BTF makes BPF debugging much better because translated and JITed instructions can carry source line information. For an ordinary compiler pipeline, that is often enough to move from an instruction back to the source line that produced it.

A dynamic optimizer adds another mapping layer:

```text
source line
   -> original BPF instruction
      -> rewritten BPF instruction
         -> JIT instruction range
            -> observed sample/event
```

If a rewrite clones a block, folds a condition, introduces a guard, replaces a sequence with a native operation, or later deoptimizes, a single source line can correspond to several generated instruction ranges across time. Preserving the original line number is useful, but it does not tell the operator which transform created the instruction or which assumption justified that transform.

This is the same distinction compiler engineers make between debug location and optimization history. A location says where code came from. A provenance record says what happened to it.

For BPF, that distinction matters even more because the verifier and JIT are independent trust and lowering stages. A useful postmortem may need to distinguish:

- the original application BPF artifact;
- the userspace optimizer's rewritten BPF artifact;
- verifier acceptance of that rewritten artifact;
- the kernel/JIT version that lowered it;
- an optional architecture-specific fast path;
- the generation that was actually attached and receiving execution;
- the moment that generation was retired or replaced.

## Where current work is still weak

### Loaded-object identity is not durable incident identity

The BTF documentation states that a loaded BPF program has a unique ID for its lifetime. That is the correct object-lifetime model, but an incident archive often outlives the loaded object. A program can be replaced and released before the postmortem begins.

A BPF tag survives better because it is content-derived, yet it still identifies BPF instructions rather than the whole optimization decision. Two deployments can use the same rewritten BPF bytes under different optimizer versions, profile evidence, kernel builds, or native lowering decisions. Conversely, semantically equivalent generations can have different tags after harmless rewrites.

The missing property is a durable execution-generation identity whose meaning includes the transformation and deployment context needed to reproduce the code path.

### Current dumps describe code, not why that code was selected

`bpftool prog dump xlated` and `bpftool prog dump jited` are excellent forensic primitives. They can reveal the actual translated and native instructions while a program is available. They do not record the optimizer predicate that selected the program.

For a profile-guided optimizer, a useful explanation may be `branch 17 was specialized because profile epoch P observed a 99.8% bias and guard G remained valid`. For an architecture-specific operation, it may be `native implementation N was selected because capability set C matched and proof witness W was accepted`.

Without that decision record, seeing the machine code can establish **what** ran but not **why this variant existed**.

### Samples can become ambiguous after rapid generation churn

A profiler observes execution over time. A re-JIT runtime changes code over time. The two timelines have to agree.

If a trace stores only a source symbol or logical program name, samples from several generations can collapse together. If it stores only a transient program ID or address, later symbolization may fail after the object disappears. If it stores the BPF tag but not activation intervals and transform metadata, an operator can identify the bytecode but still miss the decision that produced it.

Dynamic specialization therefore needs a low-cost way to attribute execution evidence to a generation without copying a full manifest into every sample.

## Promising directions with academic and production value

### 1. Make every specialization generation produce an execution receipt

The first artifact should be an append-only, content-addressed receipt for each generation. It can live primarily in userspace rather than expanding the hot-path kernel ABI.

A minimal receipt could contain:

```text
generation_id
parent_generation_id
source_build_id
original_bpf_tag
specialized_bpf_tag
optimizer_build_id
transform_ids[]
assumption_ids[]
verifier_result_digest
kernel_build_id
jit_backend
native_image_digest
line_mapping_digest
activated_at
retired_at
retirement_reason
```

Large verifier logs, profiles, proof objects, and JIT images can remain in an external artifact store keyed by digests. The receipt only has to make the chain immutable and joinable.

The important design question is identity. `generation_id` should not be a reused counter scoped to one process. It should be stable enough that a trace collected on one host can be joined with optimizer and deployment evidence later. A hash over the parent receipt, specialized BPF bytes, optimizer version, assumption set, and target context is one prototype option.

This extends the equivalence certificate from the September 5 report in one specific way: the certificate proves or records why a generation is semantically acceptable, while the execution receipt persists enough deployment state to prove that this exact generation was the one activated and observed.

### 2. Attribute samples through a generation interval map

The second artifact should connect runtime observations to receipts without putting large metadata on every event.

One design is an interval map maintained at activation time:

```text
[time_start, time_end, hook/link identity, JIT address range]
    -> generation_id
```

When a new BPF generation becomes active, the runtime records the old generation's retirement boundary and the new generation's JIT address range plus stable generation ID. Perf samples, BPF-side counters, packet traces, or application events only need enough identity and time information to join against that map.

A kernel-integrated prototype could expose a short generation cookie alongside existing BPF program metadata or JIT symbol events. A userspace-first prototype can instead subscribe to load/attach/replace operations and preserve retired mappings externally. The research question is how much kernel participation is needed to make the activation boundary race-free.

The hard cases are the useful ones: link replacement while CPUs still finish old invocations, tail calls crossing programs, freplace/fentry relationships, per-CPU execution, rapid deoptimization, and address reuse after a JIT image is freed. The representation needs an explicit `unknown` state when the collector loses the handoff rather than silently attributing a sample to the newest generation.

### 3. Evaluate provenance with adversarial re-JIT forensics

The third artifact should be a benchmark where the answer is known before the debugger runs.

Start from one BPF source program and generate tens or hundreds of generations. Inject one behavior or performance anomaly into a specific transformation under a specific assumption epoch. Then churn through later specializations and unload the problematic generation before analysis begins.

Compare at least these baselines:

- current `bpftool` program metadata plus translated/JIT dumps and BTF line information;
- profiler/JIT symbol records available from the normal Linux toolchain;
- full optimizer logs and per-generation snapshots;
- execution receipts plus generation-aware sample attribution.

The primary metrics should not be only storage overhead. Measure:

- exact-generation attribution precision and recall;
- fraction of observations that correctly remain `unknown` when evidence is lost;
- ability to reconstruct source -> rewrite -> verifier -> JIT -> execution;
- ability to identify the optimization decision that introduced the anomaly;
- replay success on the original and a different kernel/JIT version;
- bytes retained per generation and runtime overhead per activation and sample;
- time-to-root-cause for an operator who did not watch the optimization happen.

A strong system should make stale or missing provenance visible. A debugger that confidently assigns a sample to the wrong generation is worse than one that says the chain is incomplete.

## What would change this conclusion?

Three results would weaken the need for a separate specialization-provenance layer.

First, current Linux metadata, BPF tags, BTF/JIT line information, and normal profiler load records may already be sufficient to reconstruct exact historical generation identity after unload and replacement, including the optimizer decision that created each variant. If a realistic re-JIT benchmark shows unambiguous postmortem reconstruction without extra metadata, another receipt format is unnecessary.

Second, dynamic BPF specialization may remain rare enough that preserving a complete bytecode/JIT snapshot and optimizer log on every rewrite is operationally cheap. If full snapshots solve the problem with acceptable storage, synchronization, and retention cost, an optimized provenance graph adds complexity without value.

Third, the execution boundary may prove too expensive to observe precisely. If race-free generation attribution requires per-event metadata or synchronization that materially erases the optimization benefit, the right answer may be coarse epoch logging plus targeted debug mode rather than always-on attribution.

Current evidence suggests the problem is real before it is large. Linux already exposes the bytecode, JIT image, tags, BTF line mappings, and runtime metadata needed for strong debugging of loaded programs. BpfReJIT makes repeated runtime rewriting a concrete design rather than a hypothetical compiler pass. **The missing mechanism is not another disassembler. It is a durable join key between an incident observation and the exact optimization generation that produced the code executing at that moment.**

## References

- Linux kernel UAPI. [`struct bpf_prog_info` in `include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h), accessed 2026-09-07.
- Linux kernel documentation. [BPF Type Format (BTF)](https://kernel.org/doc/html/next/bpf/btf.html), accessed 2026-09-07.
- Linux bpftool documentation. [`bpftool-prog(8)`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8), accessed 2026-09-07.
- Linux kernel source. [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c), accessed 2026-09-07.
- Linux Plumbers Conference 2026. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/), accessed 2026-09-07.
