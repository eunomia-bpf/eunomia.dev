---
date: 2026-08-27
slug: ebpf-foundation-fellowship-final
description: "A final update from the eBPF Foundation Community & Advocacy Fellowship, covering six months of tutorial work, GPU and AI systems research, and community collaboration."
---

# Six Months as an eBPF Foundation Fellow: Tutorials, GPUs, and AI Systems

The six-month eBPF Foundation Community & Advocacy Fellowship is now complete. When I joined the inaugural cohort in October 2025, my original goal was fairly concrete: modernize the [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial), make newer eBPF features easier to learn, and expand the material toward emerging areas such as accelerator and machine learning observability.

In my [first fellowship update](https://ebpf.foundation/ebpf-fellowship-update-tutorials-research-and-expanding-ebpf-into-gpu-and-ai/), I wrote about how that work was already extending beyond traditional networking, tracing, and security. Looking back at the full fellowship, that became the main theme of the six months. The tutorial work, research projects, and community discussions increasingly converged on the same question: how can we make eBPF useful and understandable at new system boundaries, especially GPUs and AI agents?

## Keeping Modern eBPF Learnable

The tutorial remained the center of the fellowship. During the program, we added nine new tutorials covering newer kernel features such as BPF Arena, Workqueues, `struct_ops`, and `dynptr`, together with accelerator-oriented examples including GPU flamegraph profiling, GPU driver tracing for Intel, AMD, and Nouveau, and Intel NPU tracing. We also added practical system examples around HID-BPF and cgroup policy control.

Adding new lessons was only part of the work. I revised documentation across more than 30 existing tutorials, merged seven community pull requests, and improved the project's automation with documentation generation and Rust-based CI. This maintenance work matters because eBPF examples age quickly. Kernel APIs, libbpf behavior, toolchains, and recommended development patterns keep changing, so a tutorial that once compiled is not necessarily a tutorial that still teaches the right thing.

The repository has continued to grow after the fellowship as well. Recent additions include runnable lessons for [TCX](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/50-tcx) and [BPF Token](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/features/bpf_token). The goal remains the same: keep examples small enough to learn from, but real enough that readers can connect them to current kernels and production systems.

## Following eBPF into GPU Systems

One reason the tutorial expanded into accelerators is that eBPF itself is starting to become relevant outside the CPU-centric paths where most developers first encounter it.

In [gpu_ext](https://arxiv.org/abs/2512.12615), we explored treating GPU drivers and the device layer as a programmable OS subsystem. The design exposes safe hooks in the GPU software stack and adds a device-side eBPF runtime for verified policy logic inside GPU kernels. Across inference, training, and vector-search workloads, the prototype improved throughput by up to 4.8x while allowing policies to be changed without modifying or restarting applications.

[NCCLbpf](https://arxiv.org/abs/2603.11438) applied a similar idea to collective communication. Instead of loading unverified native plugins into NCCL, it embeds a userspace eBPF runtime into existing plugin interfaces, with verification, shared maps, and atomic policy updates. In our evaluation, policy decisions added only 80 to 130 ns of overhead, and a message-size-aware policy improved AllReduce throughput by up to 27% over NCCL's default for the evaluated 4 to 128 MiB range.

The production side of this work is represented by [SysOM-AI](https://arxiv.org/abs/2603.29235), which combines CPU profiling, GPU kernel tracing, NCCL instrumentation, and eBPF-based cross-layer tracing. It has been deployed at Alibaba across more than 80,000 GPUs, with less than 0.4% overhead, and helped reduce median diagnosis time for confirmed production issues from days to around ten minutes.

These projects also fed back into the educational work. A GPU flamegraph tutorial is more useful when it explains why cross-layer attribution is hard. A driver-tracing example is more useful when readers can connect a kernel event to a CUDA or collective-communication problem. Research and tutorials became two views of the same systems problem rather than separate activities.

## eBPF at the Boundary of AI Agents

The other direction was AI agents. Agents can generate code, start subprocesses, access files and networks, and retry actions in ways that are difficult to predict from the application layer alone. That makes the operating-system boundary a useful place to observe and control them.

[AgentSight](https://github.com/eunomia-bpf/agentsight) uses eBPF to observe both sides of that boundary. It traces decrypted LLM traffic at TLS library functions and system activity at the kernel, then correlates prompts and responses with process, file, and network behavior. This makes it possible to inspect closed-source or rapidly changing agent frameworks without adding an SDK to each one.

Related work explored other parts of the same problem. [AgentCgroup](https://eunomia.dev/blog/2026/02/17/agentcgroup-what-happens-when-ai-coding-agents-meet-os-resources/) studies the bursty CPU and memory behavior of coding agents and how cgroup hierarchy can isolate short-lived tool execution from long-lived agent state. [ACRFence](https://arxiv.org/abs/2603.20625) examines semantic rollback attacks in agent checkpoint and restore, where a restored agent may re-synthesize a slightly different irreversible request rather than replaying the original one. [SchedCP](https://arxiv.org/abs/2509.01245) looks in the other direction, using LLM agents to analyze workloads and synthesize verified `sched_ext` scheduling policies.

The common thread is that eBPF can serve as a stable systems interface even when the software above it is changing quickly. Agent frameworks may change their internal APIs every few months, while process, file, network, scheduler, and resource-control boundaries remain much more stable.

## Turning the Work into Community Discussion

The fellowship also gave me more opportunities to discuss these ideas outside the repositories. At [Linux Plumbers Conference 2025](https://eunomia.dev/others/miscellaneous/linux-plumbers-talk/), I presented bpftime and discussed why a userspace eBPF runtime can complement kernel eBPF, including support for existing eBPF toolchains and lower-overhead userspace instrumentation.

In March, I also co-organized the first [AgenticOS workshop at ASPLOS 2026](https://os-for-agent.github.io/asplos-2026.html), focused on operating-system support for AI agents. The program included work on resource control, agent execution, sandboxing, scheduling, and eBPF-based security. I presented our AgentCgroup work there, and the discussions reinforced that observability and control for agents are increasingly systems problems rather than only model or application problems.

## What I Take Away from the Fellowship

Looking back, the clearest lesson is that eBPF education has to move with the boundaries of eBPF itself. Teaching a helper or attach type is useful, but newer use cases often require readers to understand the surrounding system first: where a GPU policy can safely run, how an AI agent crosses process boundaries, or why a collective-communication plugin needs verification and hot updates.

The second lesson is that maintenance and research reinforce each other. Community issues expose examples that no longer work on current kernels. Tutorial work forces an experimental idea into a small, reproducible form. Research provides concrete problems that make a new eBPF feature worth learning. The most useful outcome of the fellowship was not a single tutorial or project, but a tighter loop between these activities.

I am grateful to the eBPF Foundation for supporting the time needed to do this work, and to everyone who opened issues, sent pull requests, reviewed projects, or joined the discussions. The fellowship has ended, but the work has not. I plan to keep expanding the tutorial, especially around newer kernel features, GPU systems, and AI-agent infrastructure, while making it easier for more contributors to add and maintain examples.

If any of these areas are useful to you, contributions and feedback are always welcome in the [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial) and the broader [eunomia-bpf](https://github.com/eunomia-bpf) community.
