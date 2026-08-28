---
date: 2026-08-28
title: "eBPF 在 L7 代理交接时还能保住策略身份吗？"
description: "Cilium 已经会把源身份带过代理连接，但在身份更新、连接复用、重试和快慢路径切换时，请求级授权链路仍可能失去一致性。"
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - Service Mesh
  - Envoy
  - L7
research_question: "当流量经过 L7 代理并从新的 socket 或连接发出时，eBPF 安全数据路径怎样保留请求级身份、策略 generation 和授权来源？"
source_cutoff: 2026-08-28
status: daily-report
---

# eBPF 在 L7 代理交接时还能保住策略身份吗？

网络策略可以在请求进入代理之前做出正确判断，却在后续路径上失去足够的上下文，使下一次判断变得含糊。

这里真正的问题并不是“生产系统完全不会把身份带过代理”。Cilium 已经做得更多：Envoy 会参与策略查询，代理数据路径会给上游流量携带源身份，近期的实际 bug 也说明连接跟踪里的源安全身份会被 L7 策略处理继续使用。

因此更值得研究的问题是：**当安全属性属于一个逻辑请求，而实现状态主要绑定在连接上时，哪些授权信息必须继续保持一致？**

<!-- more -->

L7 代理会终止下游连接，解析 HTTP 或 gRPC，然后创建或复用上游连接。一个逻辑请求因此会先后表现为 packet、socket、代理内的 request，再变成新的 socket。连接池会让多个请求共用一条上游连接；重试会让一个请求先后使用多条连接；策略或身份更新则可能在旧连接仍存活时改变当前有效的授权状态。

本文提出一个 **policy-identity handoff contract**。现有的连接级身份传递应该被视为很强的基线，而不是问题的终点。真正缺少的是请求级授权 lineage：它要跨表示变化保持一致，并且明确绑定到批准该请求的策略 generation。

这和上一篇 [主机与卸载路径之间的 complete mediation](https://eunomia.dev/zh/research/ebpf-complete-mediation-offload/) 不同。Complete mediation 关心每条可达路径是否都经过有效 enforcement point。这里先假设 enforcement point 都存在，再问这些位置看到的是否仍然是同一个安全主体和同一代授权。

它也不同于 [多 owner 网络策略组合](https://eunomia.dev/zh/research/ebpf-network-policy-composition/)。Policy composition 解决“哪条策略最终生效”；proxy handoff 则从这个结果出发，检查产生的 authority 是否仍绑定在正确的逻辑请求上。

## 生产系统已经会传递身份，但主要仍以连接为边界

Cilium 当前文档说明，L7 policy 流量会经过 node-local Envoy。Cilium 的 Envoy 包含定制 policy-enforcement filter，Cilium agent 与 Envoy 通过 Unix-domain socket 交换配置、access log 和管理信息。可参考当前的 [Cilium Envoy 文档](https://docs.cilium.io/en/latest/security/network/proxy/envoy/) 和 [Layer 7 Policy 文档](https://docs.cilium.io/en/latest/security/policy/layer7/)。

在 Ingress 和 Gateway API 中，Cilium 还明确记录了 Envoy 前后的两个逻辑策略执行点。外部流量通常先带 `world` identity，经过代理边界后则通过特殊的 `ingress` identity 进入后续策略判断。见 [Cilium Ingress 与 Network Policy](https://docs.cilium.io/en/latest/network/servicemesh/ingress/)。

实现本身比高层文档走得更远。Cilium 当前的 BPF 头文件定义了 `MARK_MAGIC_PROXY_INGRESS` 和 `MARK_MAGIC_PROXY_EGRESS` 等 mark，用来在上游代理流量中携带 source identity。见 [`bpf/lib/common.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/common.h)。近期的生产问题也展示了 Envoy 的 BPF metadata 路径把 source identity 写到上游连接上的实际行为。

这条证据排除了一个太弱的论文命题：问题不是“proxy 一定会忘记源身份”。Cilium 已经有连接级身份连续性机制。

但同一实现也暴露了更精确的失败模式。[Cilium issue #44912](https://github.com/cilium/cilium/issues/44912) 在 2026 年 3 月报告：endpoint identity 发生变化后，已有 conntrack entry 可能仍保留旧 `src_sec_id`。旧 identity 被回收后，已建立连接的 L7 policy 处理会失败，而新连接已经使用新的 identity。这个 bug 说明，跨边界传递一个数字身份还不够，身份还需要明确的生命周期和 generation 语义。

Linux BPF 的 socket 原语也体现了这一点。`BPF_MAP_TYPE_SOCKMAP` 和 `BPF_MAP_TYPE_SOCKHASH` 可以在 socket 之间 redirect message，也能挂 socket-level verdict program。用户态查询拿到的是 socket cookie，而不是 kernel socket pointer。这些机制很适合标识 transport lifetime，却不会天然标识任意 HTTP/2 stream 或一次跨多连接的 retry。见 Linux [sockmap / sockhash 文档](https://docs.kernel.org/next/bpf/map_sockmap.html)。

当更多 L7 工作进入 kernel fast path 后，这个区别更明显。[L7FP](https://arxiv.org/abs/2605.31084) 在 2026 年 5 月提出为常见 service-mesh L7 policy 合成 eBPF fast path，不支持的情况再 fallback 到已有 userspace proxy。这个设计有明显性能吸引力，但也增加了一条 correctness requirement：快路径和慢路径不仅要给出相同 allow/deny verdict，还应该保持等价的授权 lineage。

## Socket identity 不等于 request identity

对简单 TCP forwarding 来说，把 source identity 绑到 upstream socket 上可能已经足够。Multiplexing 和 retry 会打破“一条连接对应一个请求”的假设。

例如 proxy 到 backend 之间只有一条 HTTP/2 connection，却可以承载多个 downstream request。如果后续策略仍然区分 caller、policy generation 或某些 L7 授权，那么一个 socket-level identity 只有在这些 request 对后续策略完全等价时才足够。

Retry 则是相反的映射：一个 logical request 可能先后走多条 upstream connection。授权应该跟随 request，而 socket-local state 只跟随其中一段 transport lifetime。

因此更合适的模型至少区分三类 identity：

- **transport identity**：绑定具体 packet、flow 或 socket 生命周期；
- **logical request identity**：绑定 HTTP、gRPC 等一次逻辑请求，并跨 retry 和调度保持一致；
- **policy identity**：包含 principal、授权范围以及批准该请求的 policy generation。

实现可以在证明安全等价时压缩这些状态，但不能默认三者天然相同。

## 当前工作还薄弱在哪里

### 连接级 source identity 缺少显式的 request lineage contract

Cilium 的 source-identity propagation 是很好的基线。文档和现有接口没有明确给出的，是这份连接级 identity 在 pooling、multiplexing、retry 或 upstream transport 变化后，怎样对应到某一个 logical request。

缺少的抽象是一条可验证 lineage relation：后续 enforcement point 应该能回答“哪个 downstream principal、哪次 L7 decision、哪个 policy generation 产生了这个具体 upstream request”。

一个有区分力的测试应该让多个 downstream identity 共用 proxy，强制 connection reuse 与 retry，并检查每个 accepted upstream request 是否都只能映射到一条有效授权 lineage。

### Identity transition 需要 generation-aware 生命周期

Cilium issue #44912 说明，endpoint 已切换到新 security identity 后，旧 established connection 仍可能保留过时身份。它不只是普通 cache invalidation bug，而是直接说明 transport lifetime 可能长于授权状态的有效期。

因此 handoff state 需要显式 generation 或 revocation epoch。消费 propagated identity 的位置必须能判断“这个 identity 对当前 request 是否仍有效”，而不能把 established connection 理解为永久冻结授权。

### Fast-path fallback 需要 lineage equivalence，而不只是 verdict equivalence

L7FP 让 kernel fast path 与 proxy slow path 成为一个很具体的现代设计。两条路径可以在功能测试里都返回相同 allow/deny，却给 accepted request 留下不同 provenance。

因此真正缺少的 evaluation 是 **authorization-lineage equivalence**，并且要覆盖 pooling、retry、identity transition、policy update 和 proxy restart。

## 兼具学术价值和生产价值的方向

### 1. Generation-scoped proxy handoff capability

**Gap。** 现有系统可以把 source identity 带到 proxy-related connection 上，但连接级 identity 自己无法把一个 logical request 绑定到批准它的 policy generation。

**Mechanism。** 在 eBPF-to-proxy 边界生成一个 compact capability，绑定 principal、policy generation、destination scope、nonce、expiry/revocation epoch，以及允许 refine 它的 proxy trust domain。完成 L7 parsing 后，proxy 再派生 request-scoped witness。Backend-side eBPF 在把流量视为原始授权的 continuation 之前，先验证 witness 和当前 generation。

第一版应该建立在已有 source-identity propagation 之上，而不是重做整套数据路径。Connection mark 或 conntrack identity 可以继续承担 common fast path，generation-scoped handle 只补上生命周期与 request lineage 语义。

**Delta。** Complete mediation 证明 enforcement 确实发生。Cilium 已经证明 identity 可以跨 proxy connection 传递。这里增加的是 request scope 与 policy-generation validity。

**Artifact。** 一个 eBPF redirect 组件、一个小型 Envoy extension、generation table，以及能输出 machine-readable lineage 的 egress validator。

**Evaluation。** 先复现 issue #44912 一类 identity transition，再加入 pooling、HTTP/2 multiplexing、retry、proxy restart、backend change 和 policy churn。测量 stale-generation accept、principal misattribution、false reject、state size、lookup cost 和 request latency。

**Academic value。** 把 reference-monitor reasoning 扩展到 semantic transformation 和授权状态生命周期变化同时存在的场景。

**Production value。** 给今天的连接级身份传播补上一条可审计的 request-level least-privilege 路径。

**Failure condition。** 如果现有 production mesh 已经有 request-scoped、generation-aware 机制，并能在 pooling、retry、restart 和 identity transition 下保持无歧义，那么研究重点应该转向形式化和评测已有机制。

### 2. Policy-safe multiplexing 与 request-to-socket coalescing

**Gap。** eBPF 很自然地按 packet、flow 或 socket cache，而现代 proxy 可能让许多 logical request 共用一条 upstream transport。

**Mechanism。** 把 connection-pool coalescing 变成 security proof obligation。只有当多个 request 对 proxy 之后仍要执行的所有 policy property 都具有等价 security context 时，才允许共用 upstream transport；否则就保留 request-level witness，或者按 versioned policy-equivalence class 拆分 pool。

这个 equivalence class 可以包含 destination identity、principal class、policy generation，以及下游仍然需要的少量 L7 attribute。Policy update 一旦改变 equivalence，就必须使不兼容的 pooled state 失效或 drain。

**Delta。** 这不是为了性能做普通 pool partition，分组规则来自下游 authorization semantics。

**Artifact。** Envoy connection-pool extension，加一个供 eBPF control plane 使用的 policy-equivalence API。Debug view 需要解释两个 request 为什么可以安全共享 transport。

**Evaluation。** 覆盖 HTTP/1.1 keepalive、HTTP/2、gRPC stream、retry、hedged request、identity change 和多 principal 的 backend reuse。比较 naive one-identity-per-socket、禁用 pooling 和 policy-aware coalescing。

**Academic value。** 它问的是一个经典系统问题：多个 logical principal 在什么条件下可以安全共享一个 cached physical resource？

**Production value。** 保住 proxy efficiency，同时不让 pooling 抹掉后续 enforcement 仍然需要的身份差异。

**Failure condition。** 如果 L7 授权之后的下游 enforcement 从不依赖 caller identity，或者现有 proxy 已经按等价 security context 完整拆分 pool，那么额外机制价值有限。

### 3. Confused-deputy 与 fast/slow-path lineage benchmark

**Gap。** 现有 network benchmark 通常关注 throughput、latency、update time 或最终 allow/deny，很少检查 accepted request 在多次表示变化后是否仍带着正确的授权来源。

**Mechanism。** 构建 ground-truth harness，为每个 logical request 指定 principal、policy generation、允许的 L7 operation、destination scope 和预期 backend identity，再强制请求经过 eBPF fast path、proxy slow path、混合 fallback、pooling、multiplexing、retry、identity transition、policy update 和 proxy restart。

主要 metric 是 **authorization-lineage violation**：一个被接受的 upstream request，如果其 principal、generation、destination scope 或 L7 authorization 无法对应到真正触发它的 ground-truth request，就记为 violation。

**Delta。** Complete-mediation benchmark 找完全逃过 enforcement 的 packet。这里先假设 mediation 已发生，再检查 request 是否带着 stale 或错误 authority 穿过边界。

**Artifact。** 可复现 Kubernetes testbed、Cilium/Envoy 配置、mixed-principal workload generator、fault injector、reference authorization log，以及可比较 fast/slow path 的 trace format。

**Evaluation。** 从 one-request-per-connection 开始，再逐步加入 pooling、multiplexing、retry、identity transition 和 policy churn。只有当新机制能消除 connection-level baseline 暴露出的 lineage violation 时，复杂度才值得。

**Academic value。** 把一个 cross-layer security property 变成 kernel、CNI、service mesh 和 proxy 都能测的共同目标。

**Production value。** Operator 可以检查 L7 acceleration 与 proxy integration 在真实生命周期变化下是否保持授权含义，而不是只看最后得到 HTTP 200 还是 403。

**Failure condition。** 如果多个独立实现都能在这些 adversarial workload 下保持零 lineage violation，那么这个 gap 很可能已经被现有实践解决。

## 什么证据会改变这个结论？

最强的反证是一套 production-grade 实现：它已经能把 request-scoped、不可伪造的 authorization identity 与 policy generation 穿过 downstream termination、upstream pooling、HTTP/2 或 gRPC multiplexing、retry、proxy restart、identity transition、policy update 和 fast-path fallback。

Cilium 已有的 source-identity propagation 说明基本的连接级 handoff 是可以解决的；issue #44912 则说明生命周期变化仍可能让连接级 identity 失效。只有 benchmark 还能找出已有 propagation 与 invalidation 机制覆盖不到的 failure，新增 handoff mechanism 才有充分理由。

另一个边界是 enforcement placement。有些部署明确让 Envoy 成为所有 L7 decision 的最终 authority，后续策略只信 proxy identity。在这种模型里，把 original caller 继续带给后面的 eBPF decision 未必有价值。只有 downstream enforcement、audit、revocation、rate control 或 provenance 仍依赖原始授权上下文时，request-scoped lineage 才值得保留。

因此最实际的下一步不是先设计新的 metadata format，而是先做 benchmark：复现 policy/endpoint transition 后的 stale identity，再加入 pooling 和 retry，最后确认 compact generation-scoped handoff 是否真的解决了现有 connection-level identity propagation 无法覆盖的问题。