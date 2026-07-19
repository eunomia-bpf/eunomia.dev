# Paper-Angle Blog Ideas

This draft file tracks blog ideas from the current paper library. It is not a publishing plan and does not include papers under review. Public claims should still be checked against `docs/papers/registry.yaml` and the corresponding paper text before drafting.

## Ready to Write

### bpfix

Working title: **Why eBPF Verifier Errors Point to the Wrong Place**

Angle: eBPF verifier errors name where verification stopped, while developers need to find where the verifier-visible proof was lost. This should be a paper explainer with a small walkthrough, not a pure tool tutorial.

Evidence to foreground: 235 reproduced rejections, 47% returning `EINVAL`, one normalized error string mapping to nine root causes, and bpfix improving LLM one-shot repair by 11 to 21 percentage points on `bpfix-bench`.

Draft files:

- `draft/blog/bpfix.md`
- `draft/blog/bpfix.zh.md`

### Kops

Working title: **How eBPF Can Gain Native Operations Without Growing the Kernel JIT**

Angle: Kops is about the performance gap created by a deliberately simple kernel JIT. The blog should explain the tension between optimization and trusted computing base, then show how proof sequences plus native emits let userspace introduce operations while the existing verifier keeps its role.

Evidence to foreground: eBPF can run up to twice as slow as directly compiled native code in the paper's characterization. EInsn speeds up microbenchmarks by up to 24% on x86-64 and 22% on ARM64, improves production applications by up to 12%, and shrinks native code size by 12 to 23%.

Why it matters: This is the strongest follow-up to bpfix because both posts are about the eBPF compilation and verification pipeline rather than a single deployment domain.

### NCCLbpf

Working title: **Verified NCCL Policies With eBPF**

Angle: NCCL plugins give operators powerful tuning hooks, but native plugins run inside NCCL's process without verification or structured cross-plugin state. NCCLbpf should be framed as bringing the eBPF extension model to GPU collective communication.

Evidence to foreground: 80 to 130 ns overhead per tuner decision, less than 0.03% of collective latency, 1.07 us atomic hot-reload downtime with zero dropped calls across 400,000 invocations, and up to 27% AllReduce throughput improvement for 4 to 128 MiB messages on 8 NVIDIA B300 GPUs.

Why it matters: This reaches ML infrastructure readers and connects eBPF to distributed training performance, not only observability.

### gpu_ext

Working title: **GPU Drivers Need an eBPF Policy Layer**

Angle: Existing GPU policy mechanisms sit either too high in user space or too deep in vendor driver code. gpu_ext argues that GPU drivers and device kernels should become programmable OS subsystems with safe hooks and device-side eBPF execution.

Evidence to foreground: throughput up to 4.8x and tail latency up to 2x across inference, training, vector search, and multi-tenant workloads, with minimal overhead in instrumentation-only deployments.

Relationship to existing posts: Existing GPU observability and GPU bug taxonomy posts can serve as background links, but this needs a direct paper explainer that owns the phrase "GPU eBPF policy runtime."

### bpftime OSDI 2025

Working title: **Extending Applications Safely and Efficiently With bpftime**

Angle: The existing `bpftime.md` is useful background, but the OSDI 2025 paper deserves its own explainer. The post should distinguish kernel eBPF, classic user-space tracing, and bpftime's application extension model.

Evidence to foreground: Use numbers from `docs/papers/bpftime-osdi25.txt`, not the older bpftime blog, before drafting.

Why it matters: This is a flagship systems paper and should become the canonical blog link for the paper library.

## Good Second Wave

### MVVM

Working title: **Secure AI Agent Deployment Across Edge and Cloud With Wasm**

Angle: MVVM complements the agent runtime safety line. Where ACRFence focuses on semantic rollback and AgentCgroup focuses on resource control, MVVM focuses on portable, private, fault-tolerant execution across heterogeneous devices.

Evidence to foreground: 200 ms failover, 8.9x speedup through speculation, 94.2% harmful-output detection accuracy, and 3 to 5% overhead, checked against `docs/papers/mvvm.txt`.

Series fit: Link to ACRFence, AgentCgroup, AgentSight, and ActPlane as pieces of the larger "agent runtime substrate" story.

### CET disassembly

Working title: **Using CET Metadata to Make Binary Rewriting Less Wasteful**

Angle: TVA uses Intel CET `endbr64` metadata to prune spurious disassembly paths while preserving soundness. The blog should focus on why closed-source instrumentation needs a sound binary-level foundation when source-level hooks or uprobes are not enough.

Evidence to foreground: SPEC CPU2017 and real-world application results show up to 1.3x faster instrumentation time. Check the paper text before adding any broader performance claim.

Series fit: This can connect to bpftime and AgentSight as a lower-level instrumentation foundation.

### Code-Survey Chinese Pair

Working title: **LLM 如何读懂 Linux eBPF 子系统的演化**

Angle: The paper already has an English blog, but `docs/blog/posts/code-survey.zh.md` is missing. This should be a Chinese-native rewrite rather than a line-by-line translation.

Evidence to foreground: The existing English post says Code-Survey analyzed over 16,000 commits and 150,000 emails. Verify those numbers against `docs/papers/code-survey.txt` before publishing.

Why it matters: This is quick content debt and improves bilingual consistency in the paper library.

## Needs Source Refresh First

### ChainIO

Working title: **Bridging Disk and Network I/O With eBPF**

Angle: ChainIO looks like a natural I/O-path extension story, but the registry currently has `pdf: null` and `text: null`. Avoid writing the explainer until the ACM copy or a local text extraction is available.

Next step: Add the local paper text or stable source notes, then draft from the actual paper rather than the title and DOI alone.

## Possible Series Frame

Working title: **From eBPF to Agentic OS**

Angle: Turn the four-thread research arc already written in `docs/papers/README.md` into a blog/feed entry. The post would connect eBPF usability, userspace extension runtimes, GPU and I/O policy, and AI agent observability/enforcement.

Use carefully: This should not replace individual paper explainers. It is better as a hub post after bpfix, Kops, NCCLbpf, gpu_ext, and bpftime OSDI each have their own canonical blog.
