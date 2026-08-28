---
date: 2026-08-27
slug: ebpf-foundation-fellowship-final
description: "A final reflection on the eBPF Foundation Community & Advocacy Fellowship: what changed after the first update, how the research directions evolved, and what maintaining eBPF resources taught me."
---

# What Changed After Six Months of eBPF Work: A Final Fellowship Update

The eBPF Foundation Community & Advocacy Fellowship is now complete. When I joined the inaugural cohort in October 2025, I planned to spend the fellowship modernizing the [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial), adding material for newer eBPF features, and exploring emerging areas such as GPU and machine-learning observability.

My [first fellowship update](https://ebpf.foundation/ebpf-fellowship-update-tutorials-research-and-expanding-ebpf-into-gpu-and-ai/) already covered the main output from that work: nine new tutorials, updates across more than 30 existing tutorials, community contributions, and research around eBPF for GPU systems and AI agents. Rather than repeat that list, I want to use this final post to describe what changed after that update, how those research directions became more concrete, and what I learned from maintaining educational material alongside research projects.

## The Tutorial Became a Living Compatibility Test

One thing became much clearer during the fellowship: writing an eBPF tutorial is easier than keeping it correct.

A small example depends on more than the BPF program itself. Kernel capabilities change, libbpf evolves, compiler behavior moves, distributions ship different configurations, and previously reasonable examples can become misleading even when they still compile. Community reports repeatedly exposed these boundaries. We fixed a Rust uprobe lesson whose release build could inline away the function being traced, corrected an incorrect cgroup-filtering assumption in `runqlat`, repaired `ecc` compilation for the minimal tutorial, and added troubleshooting for fentry attachment failures on kernels that do not provide the expected tracing support.

That changed how I think about the tutorial project. It is not only documentation. It is also a continuously exercised compatibility surface for the eBPF ecosystem. A useful lesson needs to explain the mechanism, run on a well-defined environment, fail in an understandable way when the environment is unsupported, and stay maintainable as the surrounding stack changes.

The repository has continued in this direction with newer runnable lessons for [TCX](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/50-tcx) and [BPF Token](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/features/bpf_token). These were added after the original fellowship work, but they follow the same principle: new kernel mechanisms become much easier to understand once they are reduced to a small program that a developer can build, run, modify, and break themselves.

## The Research Directions Became More Specific

In the first update, I grouped most of the research into two broad themes: GPU systems and AI agents. By the end of the fellowship, both themes had moved from "where else can eBPF be useful?" toward narrower systems questions about policy, semantics, and analysis.

### From observing agents to enforcing agent policies

AgentSight started from an observability problem: AI agents expose semantic intent in prompts and model responses, while their real effects happen through processes, files, networks, and other OS resources. Boundary tracing with eBPF lets us correlate those two views without modifying each agent framework.

The next question was what to do once the system can see those effects. This led to [ActPlane](https://arxiv.org/abs/2606.25189), which moves from observation to policy enforcement for agent harnesses. The central idea is that policy context often lives closest to the agent, while complete enforcement needs to happen at the OS boundary. ActPlane therefore lets an agent or operator express cross-event and information-flow policies, then enforces concrete process, file, and network behavior in the kernel with eBPF and returns semantic feedback to the agent when a rule is violated.

This was a meaningful change in research direction for me. A sandbox can answer whether a process may access a resource, and a tool gateway can authorize a visible tool call. Long-running agents also need policies such as "run the relevant tests after the source changes and before committing" or "data derived from this file must not reach that endpoint." These are stateful, causal rules. They require connecting agent-level context with system-level enforcement rather than choosing one layer or the other.

### From tracing one run to profiling many runs

The AgentSight work also moved beyond collecting individual traces. In May, we added [OpenTelemetry GenAI export](https://github.com/eunomia-bpf/agentsight/pull/49), allowing eBPF-observed LLM traffic from closed-source or unmodified agents to enter a standard OTLP observability pipeline. The more interesting research question, however, appeared when the amount of history became large: how do we understand an agent after hundreds or thousands of sessions rather than debug one trace?

That led to [agentpprof](https://github.com/eunomia-bpf/agentsight/pull/84) and [semantic flamegraphs](https://eunomia.dev/blog/2026/06/24/semantic-flamegraphs-for-ai-agent-traces/). The analogy with CPU profiling is useful but imperfect. CPU profilers aggregate deterministic function names; agent traces contain natural-language prompts whose wording changes even when the underlying task is the same. agentpprof maps these operations into stable semantic categories and uses pprof-compatible profiles to aggregate tokens, time, file effects, and network effects across sessions.

The broader research question is no longer only whether eBPF can observe an AI agent. It is how system evidence and semantic intent can be transformed into stable, reusable performance and behavior abstractions. That matters for comparing agent workflows, identifying repeated work, understanding resource cost, and eventually giving agents feedback about their own behavior.

### From GPU programmability to heterogeneous eBPF semantics

The GPU work developed in a similar way. [gpu_ext](https://arxiv.org/abs/2512.12615) showed that the GPU driver and device layer can expose eBPF-style programmable policy hooks, including device-side execution. Once that works, the next problem is not simply running more BPF bytecode on a GPU. The harder question is what the eBPF safety and execution contract means on hardware with a different execution model.

In bpftime, we began prototyping a GPU-specific verifier that combines ordinary eBPF safety analysis with SIMT properties such as lane uniformity, divergent control flow, atomic-address safety, map-key behavior, and device resource budgets. We also explored multi-GPU attachment and device-local runtime state. These prototypes are still evolving, but they changed the direction of the work: portability across heterogeneous eBPF targets cannot mean only accepting the same bytecode. The runtime has to preserve the relevant event, memory, state, authority, and verifier semantics of the target as well.

This connects back to the userspace eBPF work in bpftime and to NCCL policy extensions. Kernel, userspace, communication libraries, and accelerators have very different execution environments, but the attraction of eBPF is the possibility of giving them a common programmable policy model without making every extension a new privileged native plugin.

## Research and Tutorials Started Feeding Each Other

The most useful part of doing research and education at the same time was the feedback loop between them.

Research prototypes tend to expose a large design all at once. Turning one part into a tutorial forces the mechanism to become smaller and more reproducible. What is the minimum environment? Which API is actually essential? What assumption breaks on another kernel? Can somebody who did not build the research system reproduce the interesting part in ten minutes?

The reverse direction is just as useful. Tutorial issues come from real machines and real developers rather than a controlled evaluation environment. A report that a tracing example fails on a particular kernel, architecture, compiler, or distribution often points directly at an abstraction boundary that the research version ignored. Maintenance work is therefore not separate from systems research; it is one source of its edge cases.

This is also why I think educational resources should follow eBPF into newer domains rather than stop at the traditional networking and tracing examples. A GPU profiler tutorial can make cross-layer attribution concrete. An agent observability example can show why application hooks and system effects disagree. A policy example can show why a single event is sometimes not enough to express the intended rule.

## What the Fellowship Changed for Me

The fellowship gave me something that is difficult to allocate in normal research and open-source work: time for maintenance, explanation, and community feedback alongside new systems work.

A paper naturally rewards the new mechanism. A release rewards the new feature. Neither strongly rewards revisiting thirty older examples, answering an environment-specific issue, translating a new lesson, or reducing a research prototype into something another developer can understand. The fellowship made those activities part of the work rather than something postponed until later.

It also changed my view of community work. I started the program thinking mostly about producing more material. I finished it thinking more about maintaining a loop: publish a runnable example, let people try it on environments I did not test, use the failures to improve both the example and the underlying design, and carry the useful systems questions back into research.

For future fellows, that is the part I would recommend preserving. Pick at least one resource that should still be useful after the fellowship ends, keep it runnable, and treat community questions as technical input rather than only support work.

## What Continues

The fellowship has ended, but these directions are continuing. I plan to keep the [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial) current with new kernel mechanisms and real systems examples, while continuing research on eBPF as a programmable layer for heterogeneous systems and AI-agent runtimes. On the agent side, I am especially interested in connecting observability, semantic profiling, and deterministic OS-level policy enforcement. On the GPU side, the open question is increasingly how to preserve useful eBPF semantics across devices rather than simply how to execute BPF there.

I am grateful to the eBPF Foundation for supporting the fellowship, and to everyone who opened an issue, sent a pull request, reviewed a project, tried an example on an unexpected machine, or joined a discussion. Those interactions shaped the work much more than a list of completed tutorials can show.
