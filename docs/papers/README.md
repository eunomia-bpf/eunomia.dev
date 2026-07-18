---
title: Papers and Research
description: Full-text research library (PDF + plain text per paper) from the eunomia-bpf community across eBPF, userspace runtimes, GPU systems, and AI agent infrastructure.
keywords: eunomia-bpf papers, eBPF research, bpftime, AI agent systems, GPU eBPF, systems research
author: eunomia-bpf community
---

# Papers and Research

This page is the canonical library of research produced by or directly connected to the eunomia-bpf community. Each paper carries a link to its published or arXiv version, a local PDF copy, a plain-text extraction readable by both people and LLM tools, the public artifact, and the developer-oriented blog explanation when one exists. Only published papers or public arXiv versions enter this library; work under review never appears here. Metadata and pinned arXiv versions live in `registry.yaml` in this directory.

## The research arc

The papers below form one continuous line of work, told in four threads that build on each other.

### Making eBPF usable (2022 to 2024)

The eunomia-bpf toolchain and the bpf-developer-tutorial together established an educational base for eBPF programming, but distribution and authoring remained hard problems. Wasm-bpf ([arXiv 2408.04856](https://arxiv.org/abs/2408.04856)) tackled distribution by packaging eBPF programs inside WebAssembly modules so they could be shipped and instantiated like ordinary cloud workloads. KEN ([arXiv 2312.05531](https://arxiv.org/abs/2312.05531), eBPF@SIGCOMM 2024) tackled authoring by generating verified kernel extensions from natural-language intent, the first demonstration that an LLM could produce programs the kernel verifier would accept. Code-Survey ([arXiv 2410.01837](https://arxiv.org/abs/2410.01837)) then turned the LLM lens inward, providing an automated methodology to analyze the kernel's own eBPF subsystem history at scale.

These three papers lowered the barrier from complementary sides: packaging and portability, generation and verification, and large-scale codebase understanding.

### Taking the extension runtime beyond the kernel

Kernel eBPF is powerful but constrained to the kernel boundary. bpftime began as an arXiv preprint ([2311.07923](https://arxiv.org/abs/2311.07923)) describing a userspace uprobe and syscall extension runtime fully compatible with kernel eBPF toolchains, and matured into the OSDI 2025 paper "Extending Applications Safely and Efficiently," which demonstrates that the same extension model works outside the kernel with competitive performance. MVVM ([arXiv 2410.15894](https://arxiv.org/abs/2410.15894)) complements this story with Wasm checkpoint-restore and live migration, enabling agent deployment across heterogeneous nodes.

Robustness of the extension pipeline itself became its own concern. Kops ([arXiv 2606.24213](https://arxiv.org/abs/2606.24213)) introduces native operations into the eBPF compilation pipeline without breaking safety guarantees. bpfix ([arXiv 2607.02748](https://arxiv.org/abs/2607.02748)) characterizes the diagnostic gap when the verifier rejects a program, the first systematic study of why developers struggle to interpret rejection messages and how tooling can bridge that gap.

CET-disassembly ([arXiv 2506.09426](https://arxiv.org/abs/2506.09426)) sits at a different layer: sound and precise static binary disassembly using control-flow enforcement metadata. It provides the binary-level foundation for instrumenting closed-source software where source-level eBPF attachment is unavailable.

### Into GPUs and the I/O path

Once the extension philosophy proved viable in userspace, the natural question was whether it could reach hardware domains the CPU and syscall view cannot see. gpu_ext ([arXiv 2512.12615](https://arxiv.org/abs/2512.12615)) answers for GPUs: it attaches eBPF-style extensible OS policies to GPU scheduling and memory management. NCCLbpf ([arXiv 2603.11438](https://arxiv.org/abs/2603.11438)) pushes further into GPU collective communication, composing verified policies over NCCL operations. ChainIO ([DOI 10.1145/3748355.3748371](https://doi.org/10.1145/3748355.3748371), eBPF@SIGCOMM 2025) applies the same principle to bridging the disk and network I/O domains.

Together these papers demonstrate that the extension model generalizes beyond CPU-bound system calls into the hardware paths where modern infrastructure spends most of its time.

### Observing and governing AI agents

The current center of gravity connects back to the runtime and tooling groundwork of the earlier threads: kernel-level infrastructure is what makes deep agent observability and enforcement possible without modifying agent code.

SchedCP ([arXiv 2509.01245](https://arxiv.org/abs/2509.01245), MLforSystem 2025) showed that LLM agents can safely drive Linux schedulers, casting the agent as an autonomous operator rather than a passive tool. AgentSight ([arXiv 2508.02736](https://arxiv.org/abs/2508.02736), PACMI 2025) provides system-level observability for AI agents using eBPF, capturing behavior below the SDK and below HTTP. AgentCgroup ([arXiv 2602.09345](https://arxiv.org/abs/2602.09345)) characterizes and controls the OS resources agents consume, exposing how agent workloads differ from traditional server processes.

ACRFence ([arXiv 2603.20625](https://arxiv.org/abs/2603.20625)) identifies a new attack surface: semantic rollback attacks in agent checkpoint-restore, where restoring a prior state can silently undo safety-critical decisions. The paper proposes prevention mechanisms that preserve checkpoint-restore utility while blocking rollback exploitation.

ActPlane ([arXiv 2606.25189](https://arxiv.org/abs/2606.25189)) synthesizes these threads into OS-level enforcement of the policies developers already write. An empirical study of 84 CLAUDE.md/AGENTS.md files from 64 repositories found 2,116 policy-relevant statements; 64% of those statements are policies (the remainder are descriptions), and 83% of the policies are system-observable. ActPlane enforces these policies at runtime, achieving 75.8% dangerous-command refusal compared to 53.7% for the best prior baseline (measured over 190 traces and 38 rules), at 1.9% end-to-end overhead.

## Full index

### 2026

| Paper | PDF / Text | Artifact | Technical blog | Status |
|---|---|---|---|---|
| [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) | [PDF](bpfix.pdf) · [txt](bpfix.txt) | [bpfix](https://github.com/eunomia-bpf/bpfix) | Blog pending | arXiv preprint |
| [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses](https://arxiv.org/abs/2606.25189) | [PDF](actplane.pdf) · [txt](actplane.txt) | [ActPlane](https://github.com/eunomia-bpf/ActPlane) | [Read the blog](../blog/posts/actplane.md) | arXiv preprint, v2 |
| [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213) | [PDF](kops.pdf) · [txt](kops.txt) | [Kops artifact](https://github.com/eunomia-bpf/bpf-benchmark) | Blog pending | arXiv preprint |
| [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625) | [PDF](acrfence.pdf) · [txt](acrfence.txt) | Paper artifact described in the paper | [Read the blog](../blog/posts/agent-check-restore-safety.md) | arXiv preprint |
| [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438) | [PDF](ncclbpf.pdf) · [txt](ncclbpf.txt) | [nccl-eBPF](https://github.com/eunomia-bpf/nccl-eBPF) | Blog pending | arXiv preprint |
| [AgentCgroup: Understanding and Controlling OS Resources of AI Agents](https://arxiv.org/abs/2602.09345) | [PDF](agentcgroup.pdf) · [txt](agentcgroup.txt) | [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) | [Read the blog](../blog/posts/agentcgroup-characterization.md) | arXiv preprint, v2 |

### 2025

| Paper | PDF / Text | Artifact | Technical blog | Status |
|---|---|---|---|---|
| [gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615) | [PDF](gpu-ext.pdf) · [txt](gpu-ext.txt) | [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | [GPU verification background](../blog/posts/gpu-bug-taxonomy.md), paper explainer pending | arXiv preprint, v2 |
| [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245) | [PDF](schedcp.pdf) · [txt](schedcp.txt) | [SchedCP](https://github.com/eunomia-bpf/schedcp) | [Read the blog](../blog/posts/schedcp-agentic-os.md) | MLforSystem 2025 |
| [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736) | [PDF](agentsight.pdf) · [txt](agentsight.txt) | [AgentSight](https://github.com/eunomia-bpf/agentsight) | [Read the blog](../blog/posts/agentsight_paper.md) | PACMI 2025 |
| [ChainIO: Bridging Disk and Network Domains with eBPF](https://doi.org/10.1145/3748355.3748371) | [ACM DL](https://doi.org/10.1145/3748355.3748371) | [ChainIO](https://github.com/eunomia-bpf/ChainIO) | Blog pending | eBPF@SIGCOMM 2025 |
| [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | [PDF](bpftime-osdi25.pdf) · [txt](bpftime-osdi25.txt) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime background](../blog/posts/bpftime.md), OSDI paper explainer pending | OSDI 2025 |
| [Exploiting Control-flow Enforcement Technology for Sound and Precise Static Binary Disassembly](https://arxiv.org/abs/2506.09426) | [PDF](cet-disassembly.pdf) · [txt](cet-disassembly.txt) | None | Blog pending | arXiv preprint |

### Earlier work

| Paper | PDF / Text | Artifact | Technical blog | Status |
|---|---|---|---|---|
| [MVVM: Deploy Your AI Agents Securely, Efficiently, Everywhere](https://arxiv.org/abs/2410.15894) | [PDF](mvvm.pdf) · [txt](mvvm.txt) | [MVVM](https://github.com/Multi-V-VM/MVVM) | Blog pending | arXiv preprint |
| [Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases](https://arxiv.org/abs/2410.01837) | [PDF](code-survey.pdf) · [txt](code-survey.txt) | [code-survey](https://github.com/eunomia-bpf/code-survey) | [Read the blog](../blog/posts/code-survey.md) | arXiv preprint |
| [Wasm-bpf: Streamlining eBPF Deployment in Cloud Environments with WebAssembly](https://arxiv.org/abs/2408.04856) | [PDF](wasm-bpf.pdf) · [txt](wasm-bpf.txt) | [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf) | [Read the blog](../blog/posts/wasm-bpf.md) | arXiv preprint |
| [KEN: Kernel Extensions using Natural Language](https://arxiv.org/abs/2312.05531) | [PDF](ken.pdf) · [txt](ken.txt) | [KEN](https://github.com/eunomia-bpf/KEN) | [Read the blog](../blog/posts/kgent.md) | eBPF@SIGCOMM 2024 |
| [bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User Interactions](https://arxiv.org/abs/2311.07923) | [PDF](bpftime-arxiv.pdf) · [txt](bpftime-arxiv.txt) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime background](../blog/posts/bpftime.md) | arXiv preprint; superseded by the OSDI 2025 paper |

## Reading notes

- [osdi20-brunella.md](osdi20-brunella.md): bilingual reading note and full text of hXDP (OSDI 2020), an external paper on efficient software packet processing on FPGA NICs.
- [uXDP camera-ready text](uXDP__Frictionless_XDP_Deployments_in_Userspace___Camera_Ready.txt): reference material kept alongside the library.

## Keeping this library current

A weekly audit compares this index and `registry.yaml` against arXiv, checking for new versions, new papers, and link rot. When it finds drift, it opens a GitHub issue. When a paper receives a new arXiv version, refresh the local PDF and plain-text pair, update the version pin in `registry.yaml`, and re-verify every number cited across the site. New papers follow the checklist in `.github/PAPER_PUBLICATION_CHECKLIST.md`.
