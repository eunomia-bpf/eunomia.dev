---
date: 2026-08-24
title: "eBPF 能跨 TLS 和共享进程执行信息流策略吗？"
description: "eBPF 能追踪进程、文件与套接字之间的数据流，但 TLS 和共享运行时会模糊真正携带标签的数据。本文提出面向 flow 的跨边界执行机制。"
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Information Flow
research_question: "当一个进程同时处理不同安全标签的数据，而且 TLS 或多路复用打破应用数据与内核套接字之间的一一对应关系时，eBPF 应该怎样继续执行信息流策略？"
source_cutoff: 2026-08-24
status: daily-report
---

# eBPF 能跨 TLS 和共享进程执行信息流策略吗？

设想一个常驻服务同时读取公开配置和客户密钥，随后通过同一个 HTTPS 连接池发送两个请求。按进程做信息流控制，可以知道这个服务曾经接触过密钥；按套接字做策略，可以知道它连接到了哪个目的地址。但仅靠这两个事实，内核无法判断究竟是哪一个请求真的带上了密钥内容。

这个区别决定了信息流控制能否长期运行。如果整个进程一旦读过敏感数据就永久带上敏感标签，那么安全策略可能会把它后续所有网络访问都阻断。反过来，如果标签清除得太早，数据又可能经过堆缓冲区、TLS 库、复用套接字、代理或异步 I/O 路径，在没有匹配策略判定的情况下离开机器。

<!-- more -->

Linux 已经给 eBPF 提供了很强的执行点。[BPF LSM](https://docs.kernel.org/bpf/prog_lsm.html) 可以挂载到 LSM hook，实现系统级强制访问控制和审计；cgroup 与 socket 类程序可以约束网络操作；[sockmap 和 sockhash](https://docs.kernel.org/bpf/map_sockmap.html) 还能在 socket message 上运行 verdict program，并把一个 verdict 精确应用到指定长度的数据。这些机制的重要价值在于，它们位于应用工具 API 之下，不容易被换一个 shell、库或子进程绕开。

Whole-system provenance 研究已经说明了为什么这一层有价值。[CamFlow](https://camflow.org/publications/socc-2017.pdf) 使用 LSM 和网络 hook 追踪内核对象之间的数据 provenance，[CamQuery](https://camflow.org/publications/ccs-2018.pdf) 则进一步把实时 provenance 分析放到运行路径中，用于 data-loss prevention 等安全任务。近期的 [ActPlane](https://github.com/eunomia-bpf/ActPlane) 使用 eBPF 和 BPF LSM，把标签沿进程、文件和网络边传播。它公开的[规则语义](https://eunomia.dev/zh/actplane/rule-language/)也明确采用保守策略：进程读取带标签文件后获得该标签，后续写文件或建立网络连接时继续传播。

这个模型很适合作为起点，但也把下一层问题暴露出来。现代服务通常不是“一条安全主体对应一个进程”或“一份数据对应一个套接字”。事件循环、语言运行时、worker pool、HTTP/2、连接复用、RPC multiplexing 和异步 I/O 都在有意共享进程与传输通道。一旦一个进程同时处理多条相互独立的数据流，只把标签挂在进程上，就无法继续说明究竟是哪份数据造成了这个标签。

TLS 会让这种错位更加明显。Linux 的 [kernel TLS](https://docs.kernel.org/networking/tls.html) 可以在用户态完成握手后，把 TLS record layer 移进内核；[TLS device offload](https://docs.kernel.org/networking/tls-offload.html) 又可以把部分加解密继续下移到网卡。另一些应用则始终在用户态 TLS 库中完成完整 record path。因此，同一个 HTTPS 请求在不同部署中会经过不同的可观测和可执行边界。一个策略如果假设某个固定 socket hook 总能看到明文，就不能算可移植的信息流策略。

本文的判断是：eBPF 信息流执行需要在“应用语义数据”和“内核对象”之间补上一层 **带显式 coverage 的 flow 级身份**。进程、文件和套接字标签仍然有用，但精确执行还要回答三个问题：这个标签属于哪一条逻辑数据流，它怎样绑定到当前内核操作，以及当这条关联无法证明时系统怎样明确返回 unknown，而不是默认为安全。

这个问题不同于前一篇[多租户网络策略组合报告](https://eunomia.dev/zh/research/ebpf-network-policy-composition/)。上一篇关心多个 policy owner 和 policy language 编译到同一个 datapath 后，谁拥有最终 verdict，以及 verdict 怎样追溯到来源规则。它也不同于 [BPFflow](https://vtechworks.lib.vt.edu/items/c88095ab-6dd3-4589-bd40-d0f79939a18f)：BPFflow 约束的是 eBPF 程序自身读取的敏感内核数据，防止 eBPF 程序把这些数据泄漏到非可信 sink。本文保护的是应用数据，难点在于数据跨越应用、进程、TLS 与 socket 边界之后，身份是否还能保持。

## 内核对象标签在一个对象承载多条 flow 时开始失去精度

进程标签能回答一个很有价值的问题：来自敏感 source 的信息是否曾经到达这个进程。但它回答不了后面的某次计算究竟是否依赖那个 source。

考虑一个事件循环服务：

```text
read public.json  ─┐
                   ├─> server process ─> shared TLS connection ─> api.example
read secret.key   ─┘
```

如果使用保守的进程级传播，服务读取 `secret.key` 后就变成 `SECRET`。假设规则是“`SECRET` 不得流向 api.example”，那么以后发往这个 endpoint 的请求都会被拒绝，包括完全由 `public.json` 产生的请求。它是安全的，却可能让常驻服务失去可用性。

反过来，只在一个请求结束后清除进程标签，也很难从内核事件得到充分依据。密钥可能早已被复制到 heap object，排进另一个 task，保存在 cache 中，经过 pipe 传递，或者进入稍后才消费的 buffer。内核可以看到内存和对象上的操作，但看不到语言级依赖关系，也就不知道后续哪几个字节仍然由这份密钥派生。

ActPlane 把这种 over-tainting 明确当成保守设计。对于围绕短生命周期 AI Agent process tree 的 harness，这通常是合理取舍；但对于 multiplexed server、数据库客户端、语言运行时和代理，一个进程可能比一条敏感 flow 活得长几个数量级，永久标签的成本会迅速放大。

因此这里缺的并不是“再加几个 hook”，而是 **能和内核执行点重新关联的进程内 flow identity**。

## TLS 会改变有用数据出现的位置

网络策略通常讨论目的地址、身份、HTTP method、path 或数据类别，但内核在任意一个时间点只看得到其中一部分。

普通 userspace TLS 会在 TCP socket 接收数据之前就把应用明文变成 TLS record。XDP 或 tc 位于 packet path，能看到传输元数据，却只能看到密文，无法直接判断哪些应用数据触发了 data-loss rule。Uprobe 可以在加密之前观察 TLS library call，但 tracing hook 本身并不等价于强制的 pre-operation deny point，而且不同 TLS 实现的调用路径也不同。

kTLS 又会改变这条路径：用户态负责握手与安装 crypto state，kernel TLS ULP 可以处理 record 加解密；硬件 offload 还可能让 record crypto 继续向 NIC 移动。性能上这是好事，但意味着“TLS 边界”不是一个可移植策略可以永远假定的固定位置。

Sockmap 的 message verdict program 说明 eBPF 确实可以在 socket 层按 byte range 做判定。内核文档甚至明确支持让 verdict 作用于接下来的若干字节，或者先 cork，等累计到足够字节再决定。这是很有价值的 primitive，但它仍需要一个可信答案：**这些字节究竟带哪个安全标签？** 如果数据已经加密、多条请求共享同一 stream，或者分类来自更早的文件/数据库读取，仅靠解析当前 bytes 并不能恢复 provenance。

所以一个稳健设计应该拆开两个事实：

- **语义 provenance**：哪条逻辑数据 flow 带有 `CUSTOMER_SECRET` 之类的标签；
- **执行绑定**：当前哪一个内核可见的操作或 byte range 正在承载这条 flow。

前者可能需要应用或 runtime 语义提供，后者必须在 eBPF 能执行或审计的边界上重新验证。

## 现有研究还缺什么

### 1. Process-wide taint 在 multiplexed runtime 里精度不足

Whole-system provenance 和现有 eBPF IFC 可以沿进程、文件和 endpoint 传播标签。这足以证明“敏感 source 可能影响过这个进程”，而且保守传播本身是合理的安全策略。

缺失的是一个生命周期受限、能表示同一进程内多条并发逻辑 flow 的身份。没有它，一个进程同时处理 secret request 与 public request 时，系统无法表达只有其中一条 outbound operation 是敏感的，最终只能选择永久 over-taint，或者使用缺少证明的标签清理 heuristic。

决定性实验应该让单个 event-loop process 交错处理 public/secret request，同时加入共享 worker、connection pooling 和主动 buffer reuse。只有在不依赖“一请求一进程”的条件下，系统仍能阻断 ground-truth secret egress，并允许独立 public traffic，才说明精度问题真的被解决。

### 2. TLS 与 offload 会让 hook placement 变成策略正确性的一部分

Linux 可以把 TLS record processing 留在用户态、移到 kTLS，或者进一步使用硬件 offload。同样一份数据，在不同部署里明文和密文出现的位置并不一致。

缺失的是 machine-readable coverage contract：它要说明 flow label 在哪里引入、在哪里绑定到 transport state、哪些 enforcement hook 当前有效、哪些 path transition 暂时无法支持。只要其中一环无法证明，策略应该明确返回 `unknown`，而不是默认认为 library-level 标签一定跟随数据进入了 socket。

测试应当让同一个应用策略分别运行在 userspace TLS、kTLS software mode、可用的 TLS hardware offload、plain TCP，以及“终止 TLS 后重新建连”的 proxy 路径上。如果数据流完全相同，仅仅因为 instrumentation boundary 移动就改变策略结果，说明执行 contract 还不完整。

### 3. 应用语义到 BPF 强制执行之间缺少通用可信交接

Kernel hook 是强执行点，但内核不会自动知道某段字节属于“客户 A 的密钥”还是“公开 metrics response B”。应用插桩可能知道这一区别，但一个已经被攻陷的进程不能被允许随意制造 label 或 declassification claim 来绕过策略。

缺少的是一个窄且带 freshness 的 handoff：flow identity、label set、generation，以及允许的 transformation 怎样绑定到内核对象或 message；谁有权引入标签，谁有权去标签；fd、buffer、socket、request ID 被复用时又怎样识别旧绑定已经失效。

对抗性测试应该主动尝试 stale-token replay、fd reuse、buffer address reuse、child-process handoff、proxying 与未授权 declassification。如果旧 semantic label 能被重新绑到新 socket，或者应用能自行声称“这份敏感输出已经公开”，这个接口就不具备可执行 provenance。

### 4. 现有评测很少把安全 over-taint 和精确执行区分开

把整个常驻服务永久标成 sensitive，可以得到非常漂亮的“零泄漏”结果；跳过难追踪路径，也可以得到漂亮的性能结果。这两种数字都不能告诉 operator 机制能否部署。

缺失的 benchmark 应该提供 byte/request 级 ground truth，同时统计 false allow 和 false deny，并把无法覆盖的路径单独记为 unknown，因为“没有看到”与“经过验证后允许”不是同一件事。

工作负载至少应包含 shared runtime、HTTP/2 或 RPC multiplexing、async queue、fd passing、`sendfile`、userspace TLS、kTLS、proxy，以及一个简单的 one-process-per-flow baseline。最后这个 baseline 很重要，因为在它上面 coarse label 应该以更低复杂度胜出。

## 兼具学术价值与生产价值的方向

### 1. 把安全标签绑定到 flow generation，而不是进程生命周期

**Gap。** Process label 能保存 provenance，却会把同一进程中的所有并发工作压成一个安全状态。

**Mechanism。** 引入生命周期短于进程的 `flow_id`，并用 generation 防止 ID 复用。可信 source adapter 在分类数据进入 runtime 时创建 flow；runtime instrumentation 沿已知 task、queue 和 buffer transition 传播 flow ID；到达 kernel boundary 后，再把 flow 与稳定的 kernel identity 连接起来，例如 process lineage、socket cookie 与 generation-scoped message sequence。BPF map 只保留执行需要的活跃 binding，完整语义元数据留在用户态。

这里最重要的规则是：裸 pointer、fd number 和 request ID 都不能直接当 flow identity，它们都会被复用。binding 必须携带 lifetime evidence，并在 task、buffer ownership 或 socket generation 结束时失效。

**Delta。** CamFlow 类系统会给 kernel object 做 versioning，ActPlane 会沿 process/file/network object 传播标签。这里新增的性质，是为 **同一进程内部并发 flow** 提供 versioned identity，并能把它重新 join 到可执行的 kernel object，而不需要永久污染共享 owner。

**Artifact。** 一个小型 flow-label ABI、一个 async runtime adapter、一个 TLS stack adapter、BPF LSM/socket enforcement backend，以及能够解释每次 verdict 使用了哪条 flow-to-socket binding 的 debugger。

**Evaluation。** 使用 event-loop HTTP client、worker pool、connection reuse、async queue，以及 one-process-per-request 对照组。测量 false allow、false deny、unknown coverage、binding lifetime bug、CPU、map memory 与 tail latency。删除 generation tracking 做 ablation，并强制 fd/buffer reuse，检查 stale binding 是否造成错误判定。

**Academic value。** 一般化问题是：whole-system IFC 能否只增加有界 runtime metadata，就获得有用的 sub-process precision，而不需要退化成完整 language-level dynamic taint tracking。

**Production value。** 常驻服务和 agent runtime 可以在接触一份敏感对象之后继续进行与它无关的正常网络访问，同时仍然保留可执行的数据泄漏约束。

**Failure condition。** 如果正确传播 flow 必须插桩绝大多数 memory operation 或大量 application-specific path，这套抽象已经过于接近 full dynamic taint analysis，也就失去了 eBPF 易部署的优势。

### 2. 把 TLS path coverage 做成可验证的执行 contract

**Gap。** 同一个逻辑请求可能经过 userspace TLS、kTLS 或 hardware offload，把策略绑在某个固定 plaintext hook 上会在部署变化后悄悄失效。

**Mechanism。** 为每类受保护连接定义 path manifest，记录 semantic label source、预期 TLS mode、flow 与 socket/message state 的绑定 hook、最终 enforcement hook，以及任一步不可用时的 fallback。Loader 在声明策略 active 之前验证 kernel feature 与 attachment。TLS/runtime mode 一旦变化，就让旧 binding generation 失效，必须重新建立，而不是沿用旧信任。

某个部署可能在 userspace TLS write 之前绑定标签，再在 connect/send 边界执行 endpoint rule；另一个部署使用 kTLS 后可能拥有不同的可用路径。这套设计不要求每种模式看到相同 bytes，而要求每种模式明确说明自己究竟证明了什么、没证明什么。

**Delta。** 现有 BPF attach API 定义程序在哪里运行；这里增加的是 **跨变化 cryptographic path 的 policy-level coverage proof**，并在当前路径无法支持目标 claim 时保留显式 unknown 状态。

**Artifact。** Feature detector、path-manifest schema、userspace TLS 与 kTLS adapter，以及一个在应用不变时主动切换 TLS/offload mode 的 conformance suite。

**Evaluation。** 在 plain TCP、OpenSSL 或另一种 userspace TLS、kTLS software mode、可用的 device offload 和 TLS-terminating proxy 上重放相同 labeled request trace。将 policy decision 与 coverage report 对照 ground truth，并测量初始化成本、per-request overhead 与降级为 `unknown` 的 flow 比例。

**Academic value。** 这会把 hook placement 从实现细节提升为 cross-layer security policy 的一等正确性条件。

**Production value。** Operator 更换 TLS library 或启用 offload 时，不会在不知情的情况下改变 eBPF data-egress rule 的含义。

**Failure condition。** 如果一个稳定 kernel hook 已经能在所有相关 TLS mode 中提供等价 semantic coverage，那么 path manifest 只是额外复杂度，不产生新的保证。

### 3. 为跨边界 IFC 构建 counterexample benchmark

**Gap。** 只测安全性会奖励粗粒度永久 taint，只测吞吐量又会奖励跳过困难路径。

**Mechanism。** 构建带 ground-truth label 的 workload，并故意加入容易让 provenance 丢失的执行路径。每个 test case 记录 source label、transformation、预期 sink decision，以及最可能丢失 provenance 的边界。Harness 注入并发、multiplexing、fd reuse、buffer reuse、proxy hop、async handoff 与 TLS mode change。每套系统都必须返回 `allow`、`deny` 或 `unknown`，未观测到的路径不能计成正确 allow。

**Delta。** Whole-system provenance benchmark 往往测 capture/query 能力，普通 policy test 又常常只验证某条已知 rule 有没有触发。这里测的是 **shared object 与变化 enforcement boundary 下的 decision precision**。

**Artifact。** 可复现 Linux workload suite、ground-truth trace format、coarse process IFC / flow-scoped IFC / application-only policy adapter，以及 false allow、false deny、unknown coverage 和 overhead scorer。

**Evaluation。** 比较 process-wide label baseline、application-instrumented baseline、ActPlane 风格 object propagation 和 proposed flow-scoped mechanism。加入短生命周期进程场景，让 process-wide taint 在合适 workload 上以简单和低成本胜出。所有系统在相同 workload rate 下报告 correctness、置信区间与资源成本。

**Academic value。** 这套 benchmark 把 precision-versus-coverage trade-off 变成可以比较的实验，而不是把 conservative over-taint 自动当成完整成功。

**Production value。** 安全团队可以根据可接受 false-deny budget 与要求的 path coverage，选择满足需求的最简单机制，而不是默认采用更复杂的跨层系统。

**Failure condition。** 如果真实生产策略几乎不需要 sub-process distinction，而且 coarse label 在代表性 trace 上已经有可接受的 false-deny rate，那么 benchmark 应该明确表明简单设计足够。

## 哪些结果会改变这个判断？

如果生产 workload 大多把 security principal 映射到短生命周期进程或独占 socket，那么 flow-scoped identity 的必要性会明显下降。在这种环境里，保守的 process/file/socket label 更容易审计，也可能已经是正确边界。

如果某个稳定 BPF hook 能在 userspace TLS、kTLS、multiplexing 与 offload 下都恢复所需 application label，而且不需要额外 runtime cooperation，同样会削弱本文判断。当前接口已经提供强制执行与 socket byte-level primitive，但它们本身还没有证明 semantic data 与 kernel operation 之间的对应关系。

最有说服力的证据来自真实服务上的 cross-boundary benchmark。如果 process-wide label 几乎不阻断合法流量，同时能抓到 flow-scoped 设计抓到的同一批泄漏，那么额外 metadata 没有必要。反过来，如果 flow-scoped design 明显减少 false deny，同时没有增加 false allow，也没有留下大面积 `unknown`，那么进程内 flow identity 就值得成为 eBPF security policy 的下一层能力。

## 参考资料

- [Linux kernel：LSM BPF Programs](https://docs.kernel.org/bpf/prog_lsm.html)
- [Linux kernel：BPF sockmap 与 sockhash](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux kernel：Kernel TLS](https://docs.kernel.org/networking/tls.html)
- [Linux kernel：Kernel TLS offload](https://docs.kernel.org/networking/tls-offload.html)
- [Pasquier 等，Practical Whole-System Provenance Capture，SoCC 2017](https://camflow.org/publications/socc-2017.pdf)
- [Pasquier 等，Runtime Analysis of Whole-System Provenance，CCS 2018](https://camflow.org/publications/ccs-2018.pdf)
- [Dimobi 等，BPFflow: Preventing information leaks from eBPF，eBPF 2025](https://vtechworks.lib.vt.edu/items/c88095ab-6dd3-4589-bd40-d0f79939a18f)
- [ActPlane 源码仓库](https://github.com/eunomia-bpf/ActPlane)
