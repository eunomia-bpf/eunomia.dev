---
title: 每日报告
description: "面向系统研究与生产实践的技术分析，比较一手证据，识别尚未解决的问题，并提出具有学术价值和生产价值的可验证方向。"
---

# 每日报告

Eunomia 每日报告围绕具体系统问题展开，比较一手证据，分析现有研究或生产实践仍然解释不了、测不出来或落不了地的部分，并提出可以继续做成研究系统或工程机制的方向。

## 当前报告

### [eBPF 能看清 GPU Megakernel 里面到底在跑什么吗？](https://eunomia.dev/zh/research/ebpf-gpu-megakernel-observability/)

Megakernel 把 operator 与 dependency boundary 移进一个 persistent GPU kernel。本文提出 compiler-exported semantic task hook、携带 coverage 的 eBPF aggregation，以及专门判断 kernel/PC-level evidence 何时已经不能保持正确诊断的 counterexample benchmark。

### [GPU 通信组成员变化后，怎么证明训练状态还是一致的？](https://eunomia.dev/zh/research/gpu-membership-generation-continuity/)

新 communicator 建好并不等于所有 rank 已经回到同一个逻辑状态。本文提出带 generation 的 reconfiguration certificate、ownership-aware state reconstruction，以及专门攻击 stale operation、rank reuse 与 repartition 的成员切换 correctness benchmark。

### [GPU 利用率能告诉你还能不能安全塞进一个任务吗？](https://eunomia.dev/zh/research/gpu-utilization-allocatability/)

GPU 利用率描述近期活动，却不能证明一个具体的新任务可以安全共置。本文把硬资源可容纳性和共享干扰分开，提出可分配性证书、带风险预算的两阶段准入，以及专门攻击“GPU 还有空位”判断的反例 benchmark。

### [GPU 插桩能改写内核，却不改变内核原本的行为吗？](https://eunomia.dev/zh/research/gpu-instrumentation-safety-contract/)

动态 GPU 插桩可以直接观察已经编译好的设备代码，但探针本身也会改变寄存器、占用率、控制流与观测覆盖率。本文提出 probe-effect manifest、带资源预算和显式 coverage 的插桩机制，以及专门检测 observer-induced failure 的 counterexample benchmark。

### [GPU 运行时只看页故障，能把内存放对地方吗？](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/)

GPU 显存超配会把 migration 和 eviction 都变成 policy decision。本文比较 fault、采样访问、object/phase 与 scheduling evidence，并提出携带证据的 placement record、可观测 compliance 的 placement intent，以及固定观测预算下测量 decision regret 的 counterexample benchmark。

### [eBPF 在 L7 代理交接时还能保住策略身份吗？](https://eunomia.dev/zh/research/ebpf-l7-proxy-policy-identity/)

L7 proxy 会终止一条带策略身份的连接，再创建或复用新的上游连接，因此 socket identity 可能不再代表真正触发请求的 principal。本文提出 generation-scoped handoff capability、policy-safe multiplexing，以及检测 fast/slow path authorization-lineage violation 的 benchmark。

### [eBPF 在主机与卸载路径之间还能保证完整中介吗？](https://eunomia.dev/zh/research/ebpf-complete-mediation-offload/)

主机软件、SmartNIC 快速路径与 DPU 卸载都可能执行网络策略，而流量会在 miss、更新和故障期间改变路径。本文提出显式 path-coverage contract、跨 generation 连续的 fallback，以及直接测量 policy escape 而不只测 offload 吞吐的 benchmark。

### [一次已经撤销的授权，能在 eBPF 数据路径里存活多久？](https://eunomia.dev/zh/research/ebpf-authorization-revocation/)

高性能 eBPF datapath 会通过 conntrack、auth map、socket-local storage 等持久状态复用已经做过的授权判断，即使产生这些状态的策略已经改变。本文提出 scoped revocation epoch、cross-layer completion barrier，以及直接测量最后一次 stale allow 的 benchmark。

### [eBPF 能验证有状态安全策略，而不只是验证字节码安全吗？](https://eunomia.dev/zh/research/ebpf-stateful-policy-verification/)

有状态 eBPF 安全系统会让 BPF map 在 packet、syscall、CPU 与用户态更新之间保存策略状态，即使每段程序都能通过 verifier，状态 transition 仍可能违反安全意图。本文提出小型 temporal policy contract、verifier-cooperative runtime guard，以及专门制造策略状态错误的 benchmark。

### [零拷贝 eBPF 数据路径里，谁拥有数据包缓冲区？](https://eunomia.dev/zh/research/ebpf-zero-copy-buffer-ownership/)

AF_XDP、io_uring ZC Rx 和 DPDK 都会高频复用 packet buffer，却使用不同的 ownership 与回收协议。本文提出 generation-scoped buffer capability、绑定 policy generation 的 handoff witness，以及专门检测跨路径 ownership 与 provenance 错误的 zero-copy fault benchmark。

### [多租户网络策略应该怎样在 eBPF 数据面里组合？](https://eunomia.dev/zh/research/ebpf-network-policy-composition/)

多租户集群可能同时运行 additive 的 Kubernetes NetworkPolicy、带 tier 的 ClusterNetworkPolicy 和 Cilium L3-L7 policy。本文提出带 authority 的 composition IR、跨 generation 稳定的 verdict witness，以及专门检查多 owner 策略组合和 explanation correctness 的 counterexample benchmark。

### [eBPF 能压缩遥测数据而不丢失诊断依据吗？](https://eunomia.dev/zh/research/ebpf-diagnostic-telemetry-compression/)

持续运行的 eBPF 遥测可能产生过多数据，而 counter 和 histogram 又会删掉后续诊断需要的上下文。本文提出 diagnostic contract、状态变化 exemplar、携带 coverage 的 summary，并用相同预算 benchmark 比较不同表示保留 root cause 的能力，而不是只比较 compression ratio。

### [eBPF 能理解应用自己定义的资源吗？](https://eunomia.dev/zh/research/ebpf-application-resource-semantics/)

应用内部的 pool、queue、cache 和 credit 可以直接决定性能，却不一定是操作系统资源。本文提出可版本化的 resource-semantics manifest，把语义编译成 eBPF attach plan，并用运行时 confidence loss 与 mutation benchmark 检查软件升级后这些语义是否仍然可信。

### [GPU 内核是真的慢，还是只是启动晚了？](https://eunomia.dev/zh/research/gpu-kernel-launch-latency/)

CUDA kernel 启动晚可能来自 host scheduling、runtime、command-buffer queueing、dependency 或 device availability，即使 kernel 本身的执行时间完全没变。本文提出带显式未知状态的 launch-state ledger、跨 host/device 的 launch lineage，以及以已知 delay source 为 ground truth 的归因 benchmark。

### [GPU Kernel 变慢时，Profiler 能证明是谁造成的吗？](https://eunomia.dev/zh/research/gpu-host-device-causality/)

异步 CUDA trace 可以同时看到 host call、stream、graph node 与 GPU kernel，却未必能证明哪一个更早的动作真正造成延迟。本文提出带 generation 的 host-device causal identity、保留 unknown edge 的 dependency-aware critical path，以及专门让 timestamp-only 解释失败的 ground-truth benchmark。

### [性能分析器的采样什么时候会产生偏差？](https://eunomia.dev/zh/research/profiler-sampling-bias/)

当 sampler 与周期 workload 相位锁定、hardware sample 出现 skid，或者短函数持续被漏采时，profile 百分比可能系统性出错。本文提出带 aliasing 诊断的 sampling-schedule contract、用独立 profile epoch 表达 rank uncertainty，以及在固定 overhead budget 下由 uncertainty 触发 selective instrumentation。

### [eBPF 能把内存开销归因到真正使用的页面吗？](https://eunomia.dev/zh/research/page-level-ebpf-memory-attribution/)

分配调用栈、RSS、页面热度、回收、迁移和硬件内存采样描述的是不同层次的成本。本文提出从应用分配到虚拟区间 generation 和页面活动的生命周期 provenance、带明确置信度的访问加权归因，以及用于判断逐页 lineage 是否值得其开销的 ground-truth benchmark。

### [异构系统里的 eBPF 到底应该运行在哪里？](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/)

内核、用户态、SmartNIC 与 GPU-side runtime 都可能是 eBPF 的合法执行位置，但它们暴露的事件、状态、内存、权限与 verifier 环境并不相同。本文提出 placement-aware target manifest、以 generation 为边界的状态归属，以及用于选择执行位置并检查语义是否保持一致的 ground-truth benchmark。

### [eBPF 可编程能力能在 io_uring 里面走多远？](https://eunomia.dev/zh/research/io-uring-bpf-programmability/)

当前 Linux 的 io_uring 同时出现了按 opcode 的 BPF 请求过滤和 eBPF `struct_ops` 执行路径。本文区分 cBPF admission gate 与 eBPF ring-loop control surface，并进一步分析 restriction、LSM 权限、policy generation、provenance 和资源归属怎样组合，尤其是在 io_uring 开始承载 FUSE、zero-copy networking、ublk 等注册 I/O 资源之后。

### [异步系统里，eBPF Profiler 还要追踪什么？](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)

异步工作会经过 `io_uring`、workqueue、runtime task 与 application-defined resource 离开原来的线程，因此即使 CPU 与 off-CPU sample 本身准确，也可能失去逻辑归属。本文提出 typed causal-edge 模型、把 topology edge 与 context sample 分开预算的采集方法，以及用于跨线程 attribution 的 ground-truth benchmark。

### [有状态 eBPF 应用能不能原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)

单个 BPF link 可以干净地替换一个程序，但真实有状态应用还跨越 maps、pinned objects、多个 hooks 和用户态控制器。本文区分简单 state reuse 与 semantic state migration，并把 generation-gated activation、BTF-aware migration 和 crash-consistent recovery 发展成可验证的升级机制。

### [多个 eBPF 程序如何安全共享同一个 Hook？](https://eunomia.dev/zh/research/ebpf-hook-composition-contract/)

Linux、libxdp 与 TCX 已经能让多个 eBPF 程序共享执行点，但排序本身无法定义数据修改、共享状态、竞争结果和更新应该怎样组合。本文比较现有多程序语义以及近期隔离和 bytecode dependency 工作，并提出类型化组合 manifest、显式 outcome algebra 与 versioned hook generation。

### [用户态 eBPF 要成为真正的运行时，还缺什么？](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/)

一个 BPF VM 可以执行指令，却未必定义程序如何挂载、拥有哪些能力、状态由谁持有，以及扩展如何撤销和做资源归属。本文比较 Linux eBPF、uBPF、bpftime 与 eBPF for Windows，并提出机器可读的运行时契约、绑定能力的 attach handle 和 per-extension 资源账本。

### [多个 AI Agent 同时工作时，谁来保证最终结果是对的？](https://eunomia.dev/zh/research/parallel-agent-effect-serializability/)

`worktree`、沙箱和并行工具调用可以隔离 worker，却仍可能产生错误的组合结果。本文用代码修改、共享预算、审批和不可逆操作说明：并行 Agent 的结果在真正生效前，需要经过一次统一的验证和提交。文章还分析现有 benchmark 与工具效果契约的不足，并提出 Agent 事务层、语义冲突 benchmark 和自适应并发控制等方向。

### [AI Agent 轨迹到底该保留什么：固定证据预算下的可观测性设计](https://eunomia.dev/zh/research/agent-trace-evidence-budget/)

一次模型调用周围可能产生数百个系统事件，完整轨迹却仍可能缺少决定性的状态、权限和 provenance。本文提出 evidence portfolio，并进一步分析证据效用、可移植 schema、无偏自适应采集与同预算评测仍然存在的研究空白。
