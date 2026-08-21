---
date: 2026-08-21
title: "eBPF 能理解应用自己定义的资源吗？"
description: "eBPF 可以在不重建应用的情况下追踪内部池、队列与缓存，但原始探针并不知道这些对象的语义。本文提出带版本和运行时校验的资源契约。"
tags:
  - 每日报告
  - eBPF
  - 性能分析
  - 可观测性
  - Uprobe
  - USDT
research_question: "eBPF profiler 怎样在不把应用语义硬编码成过期探针的前提下，追踪应用自定义资源，并保留资源身份、生命周期与跨系统层的真实影响？"
source_cutoff: 2026-08-21
status: daily-report
---

# eBPF 能理解应用自己定义的资源吗？

MySQL 的 buffer pool 可能已经塞满了不该留下的页面，但 Linux 看到的进程内存仍然完全正常。一个 work queue 可能已经过载，CPU utilization 却并不高。query cache、connection pool、token bucket、临时表或者应用内部的 credit，都可能直接决定请求为什么变慢，却都不是操作系统原生认识的资源。

这正是系统 profiler 很容易踩到的边界：eBPF 很擅长观察**代码和操作系统实际做了什么**，但它不会自动知道**一个应用对象到底代表什么**。

<!-- more -->

OSDI 2026 的 [Diagnosing Performance Issues in Application-Defined Resources](https://www.usenix.org/conference/osdi26/presentation/hu-yigong) 把这个可见性缺口量化得很清楚。论文从 45 个真实性能问题中整理出 38 种不同的 application-defined resource，并把很多资源交互压缩成四类语义事件：`WAIT`、`ACQUIRE`、`USE` 和 `RELEASE`。它实现的 gigiprofiler 用 LLM 找语义候选，再用静态分析验证，通过 LLVM pass 插入探针，最后根据运行时证据诊断 15 个复现问题，还找到了两个后来被 MariaDB 开发者确认的新 bug。

这篇论文同时给动态 profiler 一个很重要的警告。它的实验里，单独依赖 LLM 找资源事件时 false positive 高达 45–60%。静态验证会进一步提高精度，post-profiling validation 还能继续去掉错误候选；但在对 MySQL buffer pool 做穷举检查时，最终 pipeline 平均仍漏掉大约 23% 的 usage event。九个可以做端到端性能比较的 case 里，运行时 overhead 平均 3.7%，最高 7.8%。

这些结果说明两件事：应用资源语义非常有价值，而且一定会出错。于是问题可以进一步收窄成一个 eBPF 研究问题：**能不能把“资源语义”与“插桩机制”分开，让语义作为一个可版本化契约，再由 eBPF 在运行时持续检查这个契约是否还成立？**

本文主张建立一个**由 eBPF 消费的 versioned application-resource contract**。静态分析、LLM、开发者 annotation，或者已有的 USDT/user_events 都可以提出资源语义；eBPF 负责部署与验证：动态挂到现有二进制、把类型化描述传到探针、把资源事件与 scheduler/I/O/memory 等系统效果关联起来，并在真实运行已经不符合描述时明确降低 confidence。

目标不是简单地“用 eBPF 采四类事件”，而是实现一种**不重编应用的动态语义插桩，以及跨应用/OS 边界的独立运行时校验**。如果 eBPF 做不到这个性质，或者还不如编译期插桩可靠，这条方向就应该失败。

## 现有系统已经给了我们什么

### gigiprofiler 给出了一个很有用的资源事件模型

gigiprofiler 最容易复用的部分并不是某一个 detector，而是一个更简单的观察：很多完全不同的应用资源，都可以用一组小而明确的交互事件来描述。

- `WAIT(resource_id, wait)`：任务走到慢 acquire path 并等待。
- `ACQUIRE(resource_id, units)`：任务取得资源或者某种容量。
- `USE(resource_id, use_type, target_id)`：任务真正使用资源，还可以标记关联的另一个资源。
- `RELEASE(resource_id, units)`：资源或容量被归还。

这个接口已经足够表达几类常见病理。长时间 `WAIT` 可以暴露 contention；反复 acquire/release 却没有有效 work，可能说明 policy 很低效；`ACQUIRE` 与 `RELEASE` 长期不平衡，可能暴露 unbounded growth 或 leak；某个请求长期占有资源，也可以解释为什么后续请求被迫走 slow path。

关键是：这些是**语义事件**。一个函数返回 pointer，并不意味着它就是 `ACQUIRE`；一次 mutex wait，也不一定是在等待真正造成问题的应用资源。事件含义必须来自应用本身的行为。

### Linux 已经有多种传输 application-level evidence 的机制

Linux 本身并不缺“怎么把事件送出来”的接口。

[`user_events`](https://docs.kernel.org/6.18/trace/user_events.html) 允许用户进程注册带类型的 trace event，ftrace、perf 等现有工具可以直接消费。当前接口还支持 `USER_EVENT_REG_MULTI_FORMAT`，让同一个逻辑事件名同时存在多种 format，这对 schema 会随版本变化的应用很有用。

[uprobes](https://docs.kernel.org/trace/uprobetracer.html) 可以直接挂到已有 executable 或 shared library 的指定位置，而不需要应用主动写 trace event。libbpf 的 [`bpf_program__attach_uprobe_multi`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_uprobe_multi/) 可以让一个 BPF program 同时挂很多函数或 offset，并给每个 attach point 配不同 cookie；[`bpf_program__attach_usdt`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_usdt/) 则提供 USDT 对应的 attach path。

BPF ring buffer 还有一个适合因果记录的性质：同一个 shared FIFO 能保留跨 CPU 的事件顺序。BPF map 则可以只保存生命周期、identity 与 join 所需要的小量状态，让用户态 analyzer 不必保留整份机器事件流。

这些接口解决了**在哪观察、怎样传输状态**。它们并没有解决**pool、queue、cache entry、lease 或 credit 到底是什么**。

### 动态 attach 不等于动态语义

最容易出现的误判是：既然 uprobe 能动态挂函数，问题是不是已经解决了？其实这只解决了探针位置。

假设版本 A 里的 `pool_get()` 每次都返回一个真正归当前请求拥有的对象。版本 B 改成某些路径返回 borrowed object，这种返回不应该计作 acquire。函数名可能没变，uprobe 仍然能成功 attach，参数也完全能读出来，但原来的**语义契约已经过期**。

直接用 pointer 当 identity 同样危险。对象被销毁后，地址可能很快被新对象复用。如果 profiler 把 raw address 当成永不变化的资源 ID，它会把两个生命周期拼成一个。机器层面的 trace 可以精确到每一个地址，却在应用层得到错误结论。

所以真正的问题不是 attach，而是**semantic versioning 与 runtime validation**。

## 当前工作仍然薄弱的地方

### 语义模型通常和一次 build 或一次分析绑在一起

gigiprofiler 会针对一个 application version 做静态分析，再通过 LLVM pass 注入轻量探针。对 on-demand diagnosis 来说这是合理的，但资源模型自然会绑定到当时分析和 instrument 的代码版本。

生产环境经常不满足这个假设。profiler 可能面对已经构建、打包甚至由别的团队交付的 binary；源码未必就在现场，fleet 里也可能同时跑几个不同 build。

这里缺少的是一个可携带的 artifact：它应该明确写出**这个语义 claim 对哪个 build 有效、事件在哪里、resource instance 怎样从运行时值派生、生命周期怎样划分**。

### 探针在语法上仍然有效，语义上却可能已经错了

symbol resolve 成功、attach 成功、argument decode 成功，都只能证明 probe 能执行，不能证明它仍然代表 `ACQUIRE`、`USE`、`WAIT` 或 `RELEASE`。

生产 profiler 因此需要 semantic health signal。比如一个 `ACQUIRE` descriptor 说返回对象之后应该进入 `USE`，新版本却出现成千上万个从来没有被使用或释放的 acquisition。collector 不应该直接宣布“发现资源 leak”。更合理的状态可能是“model stale”。

### application-resource effect 与 system effect 仍然是两套证据

应用资源之所以影响性能，是因为它最后一定改变真实执行。buffer-pool miss 会带来 I/O，满队列会推迟 runnable work，credit 用完会让请求 stall，cache eviction 会把别的线程推到 page fault 或额外 CPU work 上。

站点之前的 [异步 eBPF profiler 报告](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/) 已经讨论过 causal topology 怎样跨线程保持；[逐页内存归因报告](https://eunomia.dev/zh/research/page-level-ebpf-memory-attribution/) 则讨论了 lifetime-aware provenance。application-resource profiling 实际上需要两者一起成立：资源自己要有正确的 lifetime identity，它造成的效果还要能 join 到 scheduler、I/O、memory 和 request。

编译期插进去的 event stream 可以很准确地讲应用语义，却可能看不到这些系统效果；纯 system profiler 能看到效果，却不知道应用资源是什么。真正开放的问题是怎样把两者连起来。

### 评测往往只奖励最终 diagnosis，不评 semantic contract 本身

一个系统可能因为错误的 probe 仍然和某个 workload phase 高度相关，于是最后碰巧给出正确 bottleneck label。只看 diagnosis accuracy 会把这种问题隐藏起来。

如果要做可复用的 semantic instrumentation layer，至少应该分开评四件事：

1. event-site precision / recall；
2. resource instance 与 lifetime identity accuracy；
3. stale-contract detection；
4. 固定 overhead budget 下的端到端 diagnosis accuracy。

没有这些维度，很难判断 profiler 真正理解了 application resource，还是只学会一个 workload-specific proxy。

## 值得继续做的研究与生产方向

### 1. 把 versioned resource-semantics manifest 编译成 eBPF attach plan

**Gap。** 资源发现工具可以给出 candidate semantics，eBPF 可以动态 attach，但缺少一个把 semantic claim、目标 binary 与具体 probe plan 连起来的公共 artifact。

**Mechanism。** 定义一个小型 `resource.manifest`，例如：

```yaml
resource: mysql.buffer_pool.page
build_id: 9f2c…
event: ACQUIRE
site: buf_LRU_get_free_block+0x1a4
instance_key: arg0
unit_key: retval
generation: allocation_epoch
confidence: validated
```

真正 schema 还应该包含 resource class、units/capacity、可选 `target_id`、预期 lifetime transition、symbol/offset provenance，以及这个 descriptor 是根据什么证据生成的。

loader 先把 manifest 与目标 build 对齐。大量 site 可以共用一个 `uprobe_multi` program，用 attach cookie 索引不同 descriptor；USDT site 也可以沿用同一份 semantic descriptor，只把 argument decode 交给 libbpf。BPF map 只保存从 raw argument 推导 `(resource_class, instance, generation)` 所需的 compact state。

默认情况下，build-ID 不匹配就拒绝加载。也可以提供 best-effort symbol re-resolution，但这个模式应该从 degraded confidence 开始，而不是把旧版本的“validated”状态无条件继承过去。

**Delta。** 现有 semantic profiler 会发现并 instrument event，eBPF loader 会动态挂 program。新对象把两者拆开，让资源语义本身可以独立生成、审计、分发、attach、撤销和版本化。

**Artifact。** manifest compiler、libbpf loader、可复用的 BPF resource-event program，以及把四类事件输出成 generation-scoped identity 的小型 analyzer。

**Evaluation。** 给 MySQL、PostgreSQL、Apache，再加一个类似 llama.cpp 的 runtime-heavy application 建 manifest。与 compiler-inserted instrumentation、手写 USDT/user_events 比较 attach coverage 与 event correctness，同时测 load time、event overhead、probe 数和 event-site precision/recall。

**Academic value。** 核心问题是：application semantics 能不能成为 program analysis 与 dynamic instrumentation 之间可版本化的接口，而不是永远埋在某一个 profiler 里。

**Production value。** fleet 可以直接给现有 binary 部署 resource-aware profiling，也可以单独升级或 rollback semantic descriptor，而不用重建目标应用。

**Failure condition。** 如果真实资源事件必须写大量 per-application BPF 特例才能表达，这个 manifest 就只是一层 config wrapper，不构成有效抽象。

### 2. 带显式 confidence loss 的运行时语义校验

**Gap。** 动态 attach 的 probe 在语义变化后仍然可能一直 firing。attach success 与 event count 都无法区分真实 pathology 和 stale model。

**Mechanism。** 每个 manifest entry 同时编译一组 validation invariant，例如：

- acquire 的 instance 必须属于当前 active resource generation；
- release 后的 unit 不应该继续留在 active ownership set；
- `USE` 通常应该指向之前 acquire 的 instance，或者被明确声明为 externally owned；
- 同一个 instance key 不应该在两个重叠 generation 里被复用；
- value distribution 与 transition ratio 可以和宽松的训练区间对比，但只能作为 warning evidence，不能当 correctness proof。

BPF map 保存这些检查所需的有界在线状态。违反 invariant 时先增加 typed counter，必要时通过 ring buffer 送少量 sample。用户态 controller 再把持续 violation 转成 `validated -> suspect -> stale` 这样的 confidence transition。

一个很重要的限制是：看到 violation **不能直接让系统自己改语义定义**。运行时证据只能证明旧 contract 已经解释不了现实，并不能告诉我们新的正确 contract 是什么。下一版 descriptor 仍然要通过新的静态/LLM 分析或人工 review 生成。

同一个 eBPF 系统还能独立观察 scheduler、block-I/O、page fault 与 process event，因此 validator 可以检查某类 resource event 是否真的带来模型预测的系统效果。这让 eBPF 不只是 transport，而是横跨应用和 OS 边界的 independent checker。

**Delta。** gigiprofiler 已经使用 post-profiling validation 去掉错误 event candidate。这里进一步把这个思路变成每个 deployed semantic descriptor 都有的持久、可版本化 health state，并加入独立 OS evidence。

**Artifact。** 一套 BPF validation library，以及记录 per-descriptor confidence、evidence count 与具体 failed invariant 的 controller。

**Evaluation。** 构造 symbol 不变但语义变化的版本 mutation：owned return 变 borrowed return、queue capacity 单位改变、handle reuse、release path 移动、函数被 inline/split、resource ownership 改变。比较 stale-model detection latency、false alarm，以及相比 attach-only baseline 能避免多少错误 diagnosis。

**Academic value。** 更一般的问题是：怎样在不把“观测成功”误当成“语义正确”的情况下，用运行程序本身持续验证 inferred instrumentation。

**Production value。** 运维人员能明确区分“观察到了资源 contention”和“profiler 对这个资源的模型已经不可信”。

**Failure condition。** 如果有效 invariant 需要保存太多 per-resource state，最终抹掉 eBPF 的部署与 overhead 优势，那么这类验证应该留给更重的 application-specific runtime。

### 3. 面向 semantic profiling 的 mutation benchmark

**Gap。** 现有 profiler 评测很少把 event correctness 与 diagnosis correctness 分开，也很少检查 semantic instrumentation 在软件升级后是否仍然可靠。

**Mechanism。** 围绕真实 application resource 建一套 ground-truth benchmark。每个 resource 都先定义正确的 event/lifetime trace，再做可控 mutation：

- rename 或移动 operator function；
- symbol 保留，但 ownership semantics 改变；
- unit 从 object 改成 byte；
- 对象销毁后复用 handle；
- `USE` 之前增加一次 async handoff；
- 让某个 resource event 在另一个线程上引起 I/O 或 scheduler delay。

对同一个 workload 跑四种 instrumentation：compiler-inserted semantic event、显式 `user_events`/USDT、versioned eBPF manifest，以及只看系统事件的 tracing。benchmark 分别评 event-site accuracy、lifetime identity、stale-model detection、cross-layer causal attribution、final diagnosis 和 overhead。

**Delta。** 普通 benchmark 问 profiler 能不能找到已知 bug；这里问的是 profiler 能不能知道**自己的语义假设什么时候已经不再匹配程序**。

**Artifact。** 一组可复现 application version、workload generator、ground-truth event trace 和 scoring script。第一版可以从 MySQL buffer-pool mutation 开始，再扩展到 queue、cache、connection pool 与 runtime scheduler。

**Evaluation。** benchmark 本身就是 evaluation artifact。结果应该按 mutation 给 confusion matrix 和 confidence calibration，而不是只汇总一个 diagnosis accuracy。

**Academic value。** 它把“软件演进中的 semantic observability”变成一个可测性质，而这正是现有 profiling benchmark 经常隐藏的部分。

**Production value。** 团队在把 observability rule 推到 fleet 之前，可以先检查它是否能撑过一次真实 release。

**Failure condition。** 如果这些 mutation 与真实升级中出现的 profiler failure 没有相关性，它就只是一套 synthetic stale-probe test，而不是有价值的 semantic profiler benchmark。

## 什么情况会改变这个结论？

这里最强的结论并不是“eBPF 应该替代 source instrumentation”。它不应该。

如果应用本身已经提供稳定、版本化良好的 `user_events`、USDT、metrics 或 tracing span，并且里面直接有 resource identity 与 lifetime，这些接口就是比 reverse-engineering function call 更好的语义来源。eBPF 仍然可以把它们与 OS effect 关联起来，但没有必要重新猜一遍应用已经公开的语义。

本文提出的 eBPF layer 只有在三个条件同时成立时最有价值：

1. application-defined resource 确实决定性能；
2. 运行 binary 没有足够稳定的 semantic interface；
3. operator 需要在不重建应用的情况下动态 attach，并独立关联 system-level evidence。

最能否定本文的实验，是让 versioned eBPF manifest 与 gigiprofiler 的 compiler instrumentation 跨多个真实软件版本直接比较。如果 eBPF 路径的 semantic precision 更低、检测不出 stale contract，或者花了差不多的工程成本却没有更好的 cross-layer diagnosis，那么编译期或者 source-defined instrumentation 才是更好的设计。

反过来，如果一份很小的 semantic manifest 能跨部署边界稳定存在，而 eBPF 又能用真实执行持续验证它的假设，那么 system observability 就终于可以讨论 application-defined resource，而不需要假装一个 pointer 或一个 function name 本身就是资源。