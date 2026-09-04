---
date: 2026-09-04
title: "GPU Checkpoint 恢复成功，应用状态就一定一致吗？"
description: "GPU checkpoint/restore 已经能保存 CPU 与设备状态，但可恢复的镜像仍可能跨越不同应用 epoch。本文讨论如何定义可验证的一致恢复点。"
tags:
  - Daily Report
  - GPU
  - Checkpoint Restore
  - Runtime Systems
  - Fault Tolerance
  - Distributed Systems
research_question: "透明 GPU checkpoint/restore 能否在 CPU 状态、GPU 状态、分布式通信与外部可见副作用之间保证一个应用一致的恢复点？"
source_cutoff: 2026-09-04
status: daily-report
---

# GPU Checkpoint 恢复成功，应用状态就一定一致吗？

集群调度器想抢占一个 CUDA 服务，释放 GPU，之后再把它恢复到另一张兼容的卡上。Checkpoint 工具报告成功，显存已经保存，CUDA 对象能够重建，CPU 进程也能恢复。可服务重新运行以后，仍然可能重复返回一次结果、再次执行一轮 collective，或者从一个应用从未真正处于过的混合状态继续执行。

问题不在于 checkpoint 没有保存足够多的字节，而在于保存边界和应用语义边界并不相同。现在的 GPU checkpoint/restore 已经是可用的系统原语，但**一个能够 restore 的 GPU 进程镜像，不自动等于一个应用一致的恢复点**。CUDA 状态、CPU 状态、通信状态和外部可见副作用都可能分别合法，却属于不同的逻辑 epoch。

本文的判断是：恢复点本身应该成为可验证对象。Checkpoint 需要说明，参与同一次逻辑操作的每个状态域，是否都包含在同一个 cut 之前；位于 cut 之后的状态，要么没有进入镜像，要么被明确标记为可以 replay。这样调度器才能区分“这些字节可以恢复”和“应用可以安全地从这里继续运行”。

<!-- more -->

这和之前的 [GPU membership generation continuity](https://eunomia.dev/zh/research/gpu-membership-generation-continuity/) 问题不同。那篇文章讨论 communicator 重建或 world size 改变之后，所有 rank 是否重新收敛到同一个状态 generation。它也不同于 [GPU memory placement](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/)：这里即使 GPU 成员和数据放置完全不变，checkpoint 仍然可能把 CPU、GPU 和外部世界截在不同的应用时刻。

## GPU checkpoint 冻结的是 CUDA 状态，不是整个应用

NVIDIA 已经把 checkpoint/restore 放进 CUDA Driver API。当前 CUDA 13.3.1 文档中，`cuCheckpointProcessLock()` 会把运行中的 CUDA 进程切换到 locked 状态，并阻止后续 CUDA API 调用继续改变 GPU 状态；`cuCheckpointProcessCheckpoint()` 随后把 GPU memory 搬到 host memory，并释放底层 GPU 引用。Restore 时还可以把旧 GPU 映射到同一芯片类型且显存容量足够的新 GPU，恢复完成后进程回到 locked 状态，再由 unlock 继续执行。

最新的 [NVIDIA `cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) 把这个流程表现得更直观。暂停 CUDA 时，它会锁住会修改 GPU 状态的 driver call，等待已经提交的 GPU 工作和 stream callback 完成，把 device memory 拷到主机侧，再释放 GPU 资源。这里有一个很重要的边界：这个 CUDA 操作本身不会暂停 CPU thread。因此 NVIDIA 需要把它和 CRIU 组合，才能得到完整的 Linux 进程 checkpoint。

这种分层很有价值。应用不必为每个 framework 自己重写一套 GPU save/restore，GPU 也可以在进程还存在时先被释放。但它同时暴露了一个一致性窗口：CUDA cut 与 CPU process cut 之间，普通 host code 仍然有自己的状态，也可能已经对 checkpoint 之外的系统产生了副作用。

例如，一个服务收到请求 `R`，在 GPU 上更新状态，把结果写回 host memory，然后向客户端发送 reply。如果 checkpoint 落在 GPU 更新之后、reply 之前，它不一定错误，只要恢复逻辑知道 reply 仍待发送，而且只会发送一次。更麻烦的是，reply 已经被远端接收，但本地记录“已发送”的状态还没有进入 checkpoint。这时 restore 以后重新执行发送，就会把一个合法的本地镜像变成重复的外部效果。

CUDA 无法单独解决这个问题，因为远端服务不属于 CUDA process。CRIU 可以恢复本地进程资源，但它不会因为本地 socket 被恢复，就自动回滚数据库、另一个服务或已经推进的远端 collective participant。应用真正需要的一致性边界，比任何一个单独的 serialization boundary 都更大。

## 透明 GPU checkpoint 已经很擅长重建运行时状态

这并不是说现有 GPU checkpoint 系统还停留在“拷显存”的阶段。最近的工作已经解决了不少非常难的重建问题。

[CRIUgpu](https://arxiv.org/abs/2502.16631) 把 GPU-aware checkpointing 和 CRIU、vendor mechanism 结合起来，让 CUDA 与 ROCm workload 可以在没有持续 API log 开销的情况下做透明 checkpoint。这类工作把 GPU workload 拉回到普通 process checkpoint 的系统模型里，而不是让每个 framework 都走一条专用 restart 路径。

[FlowGPU](https://doi.org/10.1007/978-3-032-35251-4_20) 在 2026 年 8 月以 Euro-Par 2026 论文形式上线，重点处理 system-level GPU checkpoint/restore 的 correctness 与 performance 问题。它用 per-task interception 和 ghost process，只在 checkpoint 时拆分 GPU 与 non-GPU state；用 CUDA VMM 保持 restore 前后的 GPU virtual address identity；通过 record/replay 重建 opaque runtime object；还协调 distributed pause。论文专门处理了一个 NCCL blocking communication 场景：如果一边已经暂停，另一边卡在匹配操作上，checkpoint protocol 自己就可能死锁，因此 FlowGPU 会在无法完整 pause 时走有界失败路径，而不是把任务永久卡住。

这些机制越来越能回答“这个进程以及它的 GPU runtime state 能不能被正确重建”。但它们没有自动回答另一个问题：重建出来的状态究竟代表应用的哪一次 transaction。Restore 后 pointer 完全一样，也不代表 pointer 对应的请求还没有向外部世界 commit。

这个区别和 filesystem crash consistency 很像。把每一个 block 都恢复成语法上合法的值，并不等于这些 block 一定来自同一个允许的 transaction prefix。

## Persistent GPU runtime 会让这个语义边界更明显

Persistent kernel 把越来越多控制状态搬进 GPU。前一篇关于 [megakernel observability](https://eunomia.dev/zh/research/ebpf-gpu-megakernel-observability/) 的 Daily Report 已经讨论过：task identity、dependency 和 scheduling decision 可能全部发生在一个长期运行的 kernel 内部，CUDA launch boundary 不再对应逻辑 operator boundary。

Checkpoint 也会遇到同样的变化。

[Concordia](https://arxiv.org/abs/2606.23521) 在 device-resident persistent kernel 内实现 checkpoint hook，用于长时间运行的 LLM inference。它注册 KV-cache block、scheduler state 等 GPU-resident region，为不同状态生成专门的 checkpoint handler，并把 recovery record 追加到 CPU 可见的 memory。这个结果说明，语义更细的恢复点可以放到 framework 以下、离被保护状态很近的位置，而不一定只能靠上层 framework 定期保存整个模型状态。

但这反而把问题说得更清楚。假设 persistent runtime 能证明“sequence 804 之前的 KV block 已经 commit”，host request queue、通信进度、输出流和任何外部副作用也必须能解释同一个 sequence 804。否则 device checkpoint 很精确，整个应用的恢复语义仍然含糊。

因此，真正有用的 checkpoint interface 需要少量应用语义，而不是无限扩大 byte coverage。

## 现有工作仍然薄弱在哪里

### Restore 成功没有说明哪些逻辑副作用可以重放

多数透明 checkpoint 机制把成功定义为 process、runtime 与 memory state 能被重建。这当然是必要条件，但生产恢复还需要一条 effect rule。

对于每个 in-flight operation，恢复逻辑至少要知道三种状态：已经 commit、必须 replay，或者必须丢弃。没有这个分类，checkpoint manager 无法区分“GPU computation 重算一次但没有任何外部影响”和“网络 reply 重复发送一次”“storage mutation 又执行一次”或者“给另一个 rank 重复发送一条消息”。

最直接的反例测试，是让外部 receiver 记录严格递增的 operation ID。一个 checkpoint 系统即使在进程层面 100% restore 成功，只要恢复后出现 ID 缺失或重复，就说明它没有捕获正确的应用恢复点。

### 所有 rank 都暂停，并不代表应用已经处于同一个 epoch

FlowGPU 已经说明 distributed pause 本身就是协议，而不是简单对每个进程做 `SIGSTOP`。它会协调多个 rank，并避免某些 blocking NCCL pattern 导致 checkpoint 永远无法完成。这比独立 snapshot 每个进程已经强很多。

但“所有 rank 现在都停了”仍然不是应用一致性的充分条件。某个 rank 可能已经更新 optimizer state、消费下一条 input 或发布 output，而另一个状态域还没同步到同一个逻辑位置。调度器真正需要的是类似“epoch 184 之前的操作全部 commit，184 之后的操作没有进入 checkpoint 或者都可重放”这样的条件。

### Checkpoint 的 coverage 通常是隐含知识

透明工具通常知道自己能保存哪些对象，但使用 checkpoint 的 scheduler 往往拿不到一个 machine-checkable coverage statement。`cuda-checkpoint` 的功能支持也会随着 driver version 演进。生产系统不应该靠工具版本、文档和 workload 行为去猜：IPC memory、Unified Memory、device-side runtime metadata、远端 storage 或 peer protocol state 是否属于本次 checkpoint 的必要状态。

Checkpoint artifact 应该显式写出它覆盖了什么，以及 restore 正确性依赖哪些假设。

## 值得继续做的研究方向

### 1. 给每个 checkpoint 一个 recovery-cut certificate

第一个可实现 artifact 是由 checkpoint coordinator 生成的小型 manifest。它记录逻辑 checkpoint epoch，以及哪些状态域真正加入了同一个 cut：

```text
checkpoint = 91
application_epoch = 184
cpu_process = frozen@184
cuda_process = CHECKPOINTED@184
rank_set = {0,1,2,3}@184
persistent_kernel = quiescent@184
external_effect_fence = committed_through(183)
replayable = [request_7712]
coverage = [cpu, cuda, nccl, request_log]
unknown = []
```

具体字段可以由 runtime 决定，但验证规则很简单：只有每个 required domain 都能证明自己属于兼容的 cut，或者已经被 replay protocol 明确处理，这个 checkpoint 才能进入可恢复状态。

它不需要替换现有工具。CUDA 负责 device-process state transition，CRIU 负责 Linux process state，FlowGPU 类的 interceptor 提供 GPU object 和 distributed pause evidence，framework 或 persistent runtime 发布 semantic epoch。Certificate 的工作只是把这些状态绑定到同一次恢复决策里。

学术问题在于：跨 CPU、GPU、通信与外部效果的一致性模型，能不能压缩成足够小、仍适合透明 runtime 的 contract。生产价值更直接：certificate 不完整时，cluster scheduler 可以拒绝 migration，或者退化到完整 restart，而不是赌一次 restore。

### 2. 暴露 semantic quiescence 和 effect fence，不要把整个应用塞进 checkpoint 工具

第二个 artifact 是一组很窄的 adapter，用于处理仅靠冻结 CUDA call 无法一致化的状态。

Persistent-kernel runtime 可以提供 `prepare_checkpoint(epoch)`，等所有小于等于该 epoch 的 device task 都完成 commit 后返回；通信库可以报告同一 epoch 之前的 collective 是否完成；serving runtime 可以把 request/result ID 写入 write-ahead log，在 coordinator 完成 fence 或 replay classification 之前暂缓对外发布。

Checkpoint system 不需要理解整个模型、scheduler 或 RPC framework。每个 subsystem 只需要暴露加入一致 cut 所需的少量 transition：prepare、quiesce、commit、abort，以及 replay classification。

失败路径必须有界。如果一个 rank 或外部 subsystem 在 timeout 内到不了目标 cut，就取消这次 checkpoint 并继续运行。FlowGPU 对 incomplete distributed pause 的处理已经给出了一个很好的系统原则：失败的 checkpoint 比“看起来成功但一致性未知”的 checkpoint 更安全。

### 3. Benchmark 应该测恢复语义，而不只测 pause time 和 image size

Checkpoint 论文通常会测 runtime overhead、image size、pause duration、restore duration 和 migration time。这些都是必要指标，但如果目标是恢复一致性，还需要直接测“恢复后的 execution 是否等价于某个允许的 prefix 加上合法 replay”。

可以做一个 deterministic testbed，同时维护四类 ground truth：CPU memory、GPU memory、NCCL 或其他 peer protocol、以及一个外部 transactional log service。Fault injector 专门把 checkpoint 放在 GPU completion、host callback、collective boundary、allocator change 和 output publication 前后。

每次 restore 后检查：

- 已经 commit 的 operation 是否丢失；
- 外部可见 effect 是否重复；
- 是否混入不兼容 epoch 的状态；
- pointer identity 是否变化或失效；
- restore 后通信是否死锁；
- coverage 不支持时，系统是否仍错误地接受 recovery。

主要指标应该是**adversarial cut placement 下的 invalid recovery rate**，而不是 checkpoint 本身有多快。只有恢复语义相同，速度比较才有意义。

## 哪些结果会改变这个判断？

有三类结果会让单独的 recovery-cut contract 变得多余。

第一，vendor 或 process-checkpoint interface 直接把 atomic boundary 扩展到 CPU thread、GPU runtime state、multi-process communication，以及真实应用关心的全部外部副作用。如果底层 primitive 本身就拥有完整应用 cut，额外 certificate 没有必要。当前 CUDA checkpoint 的设计仍然明确与 CPU checkpoint 组合，而不是声称覆盖这个更大的边界。

第二，主流 production framework 最终统一到一个 application-level checkpoint protocol，本身已经提供 durable operation epoch 和 exactly-once replay semantics。透明 GPU checkpoint 只负责加速这个协议的实现，那么语义 contract 可以完全留在 GPU 层之上。

第三，实验可能发现透明 preemption 和 migration 在生产中只发生于天然 quiescent 的 framework boundary，根本不存在含糊的外部 effect。若 adversarial cut placement 无法制造出现有系统检测不到的 semantic recovery failure，这套 certificate 只会增加复杂度。

目前的证据更支持相反方向。CUDA 已经有正式的 process checkpoint state machine，CRIUgpu 与 FlowGPU 正在让透明重建真正可用，persistent-kernel 系统又继续把高价值状态移到 framework boundary 以下。**下一步值得补的，不是另一种显存拷贝方法，而是一条 machine-checkable 的声明：这份 GPU state 究竟属于应用的哪一个一致恢复点。**

## 参考资料

- NVIDIA：[CUDA Driver API: CUDA Checkpointing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html)，CUDA 13.3.1 文档，访问于 2026-09-04。
- NVIDIA：[`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint)，checkpoint/restore utility 与当前 driver feature notes，访问于 2026-09-04。
- Radostin Stoyanov 等：[CRIUgpu: Transparent Checkpointing of GPU-Accelerated Workloads](https://arxiv.org/abs/2502.16631)，2025。
- Zehua Yang 等：[FlowGPU: Transparent and Efficient GPU Checkpointing and Restore](https://doi.org/10.1007/978-3-032-35251-4_20)，Euro-Par 2026，2026-08-15 首次上线。
- Yuhang Gan 等：[Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference](https://arxiv.org/abs/2606.23521)，2026。
