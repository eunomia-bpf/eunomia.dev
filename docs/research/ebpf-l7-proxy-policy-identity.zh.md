---
date: 2026-08-28
title: "eBPF 在 L7 代理交接时还能保住策略身份吗？"
description: "当 eBPF 把带策略身份的流量交给 L7 代理时，上游新连接可能丢失最初的主体身份、策略 generation 和授权来源。"
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - Service Mesh
  - Envoy
  - L7
research_question: "当流量进入 L7 代理，并从另一个 socket 或连接重新发出时，eBPF 安全数据路径怎样保留请求原始身份、策略 generation 和授权 provenance？"
source_cutoff: 2026-08-28
status: daily-report
---

# eBPF 在 L7 代理交接时还能保住策略身份吗？

一个网络策略可以在请求进入代理之前做出完全正确的判断，却在请求真正到达后端之前丢掉这个判断成立的理由。

问题出现在数据路径跨过语义边界时。eBPF 程序先看到一个带源身份和连接 tuple 的 packet 或 socket，然后把流量重定向到 Envoy 等 L7 proxy。代理终止下游连接，解析 HTTP、gRPC 或其他应用层协议，再创建或复用另一条上游连接。到了后端一侧，packet 属于代理自己的 socket，tuple、security identity、连接生命周期和 policy context 都可能已经改变。

策略的重要性没有改变。系统仍然需要回答一个简单的问题：**这个具体的上游请求，究竟是哪个主体、在哪个 policy generation 下被允许的？**

<!-- more -->

本文认为，跨边界的 eBPF 安全设计除了 path coverage，还需要一个 **policy-identity handoff contract**。请求从 eBPF enforcement domain 进入 L7 proxy，再回到新的 kernel/network path 时，真正决定授权的上下文必须以一种不会被另一个请求、另一条连接或旧策略 generation 混淆的形式继续存在。

这比上一篇 [主机与卸载路径之间的 complete mediation](https://eunomia.dev/zh/research/ebpf-complete-mediation-offload/) 更窄。Complete mediation 问的是每条可达 packet path 是否都经过有效 enforcement point。本文先假设请求确实经过了多个 enforcement point，然后追问：proxy 把流量终止并重新发起之后，这些 enforcement point 还在讨论**同一个安全主体和同一次授权判断**吗？

它也不同于 [多 owner 网络策略组合](https://eunomia.dev/zh/research/ebpf-network-policy-composition/)。Policy composition 决定哪一个 owner 和 rule 应该产生最终 verdict；proxy handoff 则问这个 verdict 在表示从 packet 变成 socket、L7 request，再变成另一个 socket 时，是否仍然绑定在原来的请求上。

## 生产系统已经把这个边界暴露出来了

Cilium 当前的 L7 policy 文档明确说明：L7 policy traffic 会经过 node-local Envoy。Cilium 的 Envoy 构建包含自定义 policy enforcement filter，Cilium agent 通过 Unix-domain socket 给 Envoy 下发配置并接收 access log。可以直接参考当前的 [Cilium Envoy 文档](https://docs.cilium.io/en/latest/security/network/proxy/envoy/) 和 [Layer 7 policy 文档](https://docs.cilium.io/en/latest/security/policy/layer7/)。

在 Cilium Ingress 和 Gateway API 中，这个边界更加明显。Cilium 文档说明 per-node Envoy 会和 eBPF policy engine 交互并执行 policy lookup，而且存在两个逻辑 policy enforcement point：流量在进入 Envoy 前可以检查一次，在 Envoy 即将把流量发往 backend 时再检查一次。外部流量通常先带有 `world` identity，到了 Envoy 边界又会被赋予特殊的 `ingress` identity。见 [Cilium Ingress 与 Network Policy 文档](https://docs.cilium.io/en/latest/network/servicemesh/ingress/)。

这个设计本身很实用，但它也暴露出更一般的系统问题。第二个 enforcement point 看到的 identity 不一定就是最初发起请求的 identity。Proxy 本来就是一个 deputy：它接受调用者的连接，再代表调用者去创建或复用另一条连接。

Linux BPF 的很多原语天然落在 packet 和 socket 边界。`BPF_MAP_TYPE_SOCKMAP` 与 `BPF_MAP_TYPE_SOCKHASH` 可以在 socket 之间 redirect message，也可以挂 socket-level verdict program。内核文档还会把 socket cookie 暴露给用户态，而不是把 kernel socket pointer 直接暴露出去。这些机制非常适合标识一个 transport lifetime，但 proxy handoff 会创建新的 socket，它已经无法单独表示原始请求。见 Linux 当前的 [sockmap / sockhash 文档](https://docs.kernel.org/next/bpf/map_sockmap.html)。

随着更多 L7 工作进入 kernel fast path，这个问题会更加重要。2026 年 5 月发布的 [L7FP](https://arxiv.org/abs/2605.31084) 会为常见 service-mesh L7 policy 合成 eBPF fast path，对无法支持的 policy 再 fallback 到现有 userspace proxy。它的性能结果说明这种 fast/slow split 很有吸引力，但也带来新的 correctness requirement：一个 request 不能仅仅因为这一次留在 eBPF fast path，而另一次经过 proxy，就悄悄换掉 authorization identity。

## Socket identity 不等于 request identity

对于简单 TCP forwarding，把 policy state 绑定在 socket 上可能足够。但一旦 proxy 开始 multiplex 或 retry，这个假设就很脆弱。

例如 Envoy 到 backend 之间只有一条 HTTP/2 connection，但里面可能同时承载多个 downstream client 的请求。如果 eBPF egress path 给这个 upstream socket 绑定一个 source identity 或一个 authorization generation，它到底描述的是哪个请求？同一个 socket 不变，stream 之间的答案却可能完全不同。

Retry 是相反的变化。一个 logical request 可能先后使用多条 upstream connection。Request-level authorization 应该跨 retry 保留，但 socket-local record 自己做不到。

Connection pooling 让这两种情况成为日常行为，而不是 corner case。因此一个正确系统至少需要区分三种 identity：

- 具体 packet/socket lifetime 的 **transport identity**；
- 能跨 proxy scheduling、multiplexing 和 retry 的 **logical request / stream identity**；
- 包含安全主体、policy generation 和授权范围的 **policy identity**。

它们可以关联，但不应该因为 benchmark 恰好使用 one-request-per-TCP-connection 就被压成同一个字段。

## Handoff 更像 capability，而不是普通 trace annotation

实际系统不需要把完整 policy object 放进每个 request。它需要的是下一层 enforcement domain 能验证的一小段授权声明。

假设 eBPF datapath 判断 downstream principal `A` 可以在 policy generation `g` 下向 service `B` 发起某一类请求。在 redirect 到 proxy 之前，datapath 或 control plane 可以生成一个短生命周期的 **handoff capability**，其中包含或引用：

- 原始 security identity 或 principal class；
- 授权这次 handoff 的 policy generation；
- 防止 capability 被拿去复用到无关流量的 flow/request nonce；
- 允许的 destination 或 service scope；
- proxy 可以进一步收窄的 L7 capability class，例如某一组 HTTP method/path rule；
- expiry 或 revocation epoch；
- 允许消费这个 capability 的 proxy instance 或 trust domain。

Proxy 在解析 L7 request 之后可以进一步 refine 这个 capability。例如 packet-level decision 只允许“principal A 可以进入 proxy P 去访问 service B”，L7 filter 再证明某一个 HTTP request 同时匹配 method/path rule `r`。当 Envoy 创建或复用 upstream connection 时，它产生一个绑定 logical request 的 derivative witness，而不是只给整个 proxy socket 一个 identity。

后端一侧的 eBPF 不需要再次解析 HTTP。它只要验证 derivative witness、当前 generation、destination scope 和 proxy authority，就可以判断这个 packet 是否真的是原始授权 request 的 continuation。

这更接近 capability system，而不是 tracing label。Trace label 主要帮助事后解释发生了什么；handoff capability 直接参与 authorization decision，在 stale、缺失或绑定到错误 request 时应该 fail closed。

## 当前工作还薄弱在哪里

### 两个 enforcement point 并不会自动保留同一个 principal

Cilium 已经明确记录 Envoy 前后都存在 policy enforcement。这比把 proxy 当作透明 middlebox 强得多。但前后两个 check 完全可能合法地使用不同 identity，例如 `world`、`ingress` 或 workload identity。

缺少的抽象是一条显式 **lineage relation**：第二个 decision 应该能证明是哪一个原始 principal 和 policy generation 导致了这个 request，而不是只证明 packet 来自一个受信任的 proxy process。

当 proxy 变成 confused deputy 时，这个 gap 会直接影响安全性。如果一个被允许的 downstream request 可以让 proxy 发出超出原 principal scope 的 upstream operation，那么 backend 只信 proxy identity 的规则就比原始 policy 更弱。反过来，如果 backend 想重新执行原始 L3/L4 policy，却拿不到转换后的上下文，也可能误杀合法流量。

一个有区分力的测试应该让多个 downstream identity 共用一个 proxy，故意让 backend destination 重叠，再检查任意 upstream request 是否可能被归因到错误 principal 或旧 policy generation。

### Per-socket policy state 在 multiplexing 下会失效

Socket cookie、socket-local storage、sockmap entry 和 connection-tracking state 都很有价值，因为它们能避免每个 packet 重做昂贵工作。它们也确实适合描述 transport lifetime。

但这些状态自己无法描述 HTTP/2 stream、gRPC request、retry，或者多个 caller 共享的 connection pool。如果 policy cache 默认“一条 upstream socket 对应一个 principal”，它可能在不用 pooling 的 HTTP/1.1 benchmark 里完全正确，在真实 HTTP/2 workload 里却是错的。

缺少的机制是：什么时候 request-level authorization 可以安全地 **coalesce** 成 connection-level state？只有当同一条 connection 上所有 active request 对下游仍需执行的 policy property 都等价时，系统才应该聚合。否则就要保留 request/stream-level lineage，或者把最终 decision 留在 proxy 内部。

### Fast-path fallback 需要 identity equivalence，不只是 verdict equivalence

L7FP 展示了一种很有用的 split：常见 policy 走 eBPF fast path，unsupported case 回到 userspace proxy。功能测试可以验证两条路径对同一个 request 是否都 allow 或 deny。

这是必要条件，但还不够。简单测试中，两条路径即使给出相同 verdict，也可能给 accepted request 留下不同 provenance。等到 backend policy、audit、rate limiter 或 revocation mechanism 依赖原始 principal 时，这个差异才暴露出来。

所以缺少的 evaluation 不只是 verdict equivalence，而是 fast/slow path 之间的 **authorization-lineage equivalence**，而且要覆盖 retry、pooling、policy update 和 proxy restart。

## 兼具学术价值和生产价值的方向

### 1. Generation-scoped proxy handoff capability

**Gap。** eBPF 可以授权并 redirect 一个 flow 到 L7 proxy，但 proxy 的 upstream socket 并不会天然携带原始 principal 或 policy generation。

**Mechanism。** 在 eBPF-to-proxy 边界定义一个 compact handoff capability，由可信 datapath 生成，只允许被授权 proxy instance 消费。Capability 绑定 principal identity、policy generation、destination scope、nonce、expiry/revocation epoch 和允许 refine 的 capability class。完成 L7 parsing 之后，proxy 再产生 request-scoped witness，让 backend-side datapath 可以验证原始授权，而不是把 proxy socket identity 当成原始 principal。

Capability 不能被普通 workload 伪造。第一版实现可以选择 kernel-managed opaque handle、受限访问的 map-backed generation record，或者经本地可信 channel 传递的 authenticated token。重点应该先放在 trust boundary 清晰，而不是追求密码学花样。

**Delta。** Complete mediation 证明 request 穿过了 enforcement point；这里证明 security subject 和 policy generation 跨这些 point 仍然是同一个。Multi-owner composition 决定 winning rule；handoff capability 把这个 authority 跨 proxy transformation 带过去。

**Artifact。** 一个小型 eBPF redirect 组件、Cilium/Envoy extension 或 filter、map-backed capability table 和 egress validator。Debug view 应该能把 downstream identity、L7 rule、upstream request 与 policy generation 连起来。

**Evaluation。** 让多个 identity 共用 node-local proxy，同时反复更新 policy、重启 Envoy、替换 proxy instance、触发 retry 和改变 backend endpoint。测量 unauthorized upstream request、stale-generation accept、principal misattribution、capability lookup cost、额外 request latency 和 state size。

**Academic value。** 这个问题可以推广为：一个 trusted component 终止并重新发起通信时，reference-monitor reasoning 怎样跨 semantic transformation 保持成立。

**Production value。** Service mesh 与 eBPF policy engine 可以保住 least-privilege intent，而不是一旦 redirect 就把 proxy 当成万能身份。

**Failure condition。** 如果现有 production mesh 已经有等价的 request-scoped、generation-aware handoff，而且在 proxy restart、pooling 和 retry 下都不会 identity confusion，那么研究重点应该转向形式化和测量已有机制，而不是再造一个协议。

### 2. Policy-safe multiplexing 与 request-to-socket coalescing

**Gap。** eBPF 自然容易用 packet、flow 或 socket 做 cache key，但现代 L7 proxy 会在共享 upstream connection 上调度多个 logical request。

**Mechanism。** 把 coalescing 变成一个 proof obligation。只有多个 request 对 proxy 之后仍需执行的 policy 来说具有等价 downstream security context，proxy/runtime 才允许它们共享 upstream transport；如果 context 不同，就保留 stream-level witness，或者按照显式 policy-equivalence class 拆分 connection pool。

一个 compact equivalence class 可以包含 destination identity、当前 policy generation、downstream principal class，以及 downstream enforcement 仍然需要的 L7 attribute。这个 class 必须 versioned。Policy update 一旦改变 equivalence，旧 pooled state 就不能继续无条件复用。

**Delta。** 这不是为了性能做普通 connection-pool partition。Grouping rule 来自下游 security semantics，并用 authorization lineage 来评估。它也不要求把每个 HTTP header 暴露给 eBPF；proxy 可以保留 protocol detail，只输出下一层 enforcement 所需的 security equivalence class。

**Artifact。** Envoy connection-pool extension，加一个由 eBPF control plane 消费的 policy-equivalence API。Debug view 需要解释两个 request 为什么可以安全共享一条 upstream socket，以及依据了哪些 policy field。

**Evaluation。** 覆盖 HTTP/1.1 keepalive、HTTP/2 multiplexing、gRPC stream、retry、hedged request，以及多 source identity 的 backend connection reuse。比较 naive one-identity-per-socket cache、完全禁用 pooling 和 policy-aware coalescing。测量 misattribution、false reject、connection count、latency、CPU 与 policy-update convergence。

**Academic value。** 它把 network policy 连到一个经典系统问题：多个 logical principal 在什么条件下可以安全共享一个 cached physical resource？

**Production value。** 在保留 proxy efficiency 的同时，避免 connection pooling 把 kernel-side policy 仍然需要的身份差异抹掉。

**Failure condition。** 如果 proxy 之后的 backend enforcement 永远不依赖原始 principal，或者现有 production proxy 已经按等价 security context 把所有 pool 完整分开，这个机制只会增加状态而不会改善 correctness。

### 3. Confused-deputy 与 fast/slow-path lineage benchmark

**Gap。** 现有 network benchmark 通常强调 throughput、latency、policy-update time 或 allow/deny correctness，很少故意让 proxy 同时代表多个 principal，再检查 authorization lineage 是否跨每次 representation change 保留下来。

**Mechanism。** 构建 ground-truth harness，为每个 logical request 指定 principal、policy generation、允许的 L7 operation、destination scope 和预期 backend identity，然后强制 request 经过 direct eBPF fast path、Envoy slow path、混合 fallback、connection pooling、HTTP/2 multiplexing、retry、policy update 和 proxy restart。

主要 correctness metric 是 **authorization-lineage violation**：一个被接受的 upstream request，如果其 principal、generation、destination scope 或 L7 authorization 无法和真正触发它的 ground-truth request 对应，就记为 violation。Secondary metric 再包括 revocation 后的 stale accept、false reject、lineage-loss rate、throughput、CPU 和 latency。

**Delta。** 上一篇 complete-mediation benchmark 找的是完全绕过 current enforcement 的 packet。这里假设 mediation 已经发生，继续检查被 mediation 的 request 是否带着错误 identity 穿过 proxy boundary。

**Artifact。** 可复现 Kubernetes testbed、Cilium/Envoy 配置、mixed-principal workload generator、fault injector、reference authorization log，以及可以对比 kernel fast path 和 proxy slow path 的 trace format。

**Evaluation。** 先包含 one-request-per-connection 这种所有实现都应通过的简单 workload，再逐步加入 pooling、multiplexing、retry 和 policy churn。新机制必须在 ordinary socket/proxy identity 会产生 lineage failure 的场景里真正消除错误，才值得增加复杂度。

**Academic value。** Benchmark 把 cross-layer security property 变成可测量目标，也给 kernel、CNI、service-mesh 和 proxy research 一个共同评测面。

**Production value。** Operator 可以测试一个 L7 acceleration 或 proxy integration 在真实 failure/pooling 行为下是否保住 policy subject，而不是只看 demo request 最后得到 HTTP 200 还是 403。

**Failure condition。** 如果多个独立实现都能在 adversarial multiplexing、restart、update 和 fallback workload 下保持零 lineage violation，那么这个 gap 可能已经被现有实践解决得足够好。

## 什么证据会改变这个结论？

本文依赖一个假设：生产数据路径里，L7 proxy 前使用的 security identity 并不会自动、无歧义地绑定到 proxy 重新发出的 logical request。

最强的反证会是一套 production-grade 实现：它已经能把 request-scoped、不可伪造的 authorization identity 和 policy generation 穿过 downstream termination、upstream connection pooling、HTTP/2 或 gRPC multiplexing、retry、proxy restart、policy update 和 fast-path fallback；而且 adversarial benchmark 无法制造 stale、confused 或无法归因却被接受的 request。那时真正需要做的是标准化和系统评测，而不是再增加新的 handoff protocol。

第二个边界是 enforcement placement。有些部署明确把 Envoy 当成所有 L7 decision 的最终 authority，并要求 backend policy 只信 proxy identity。在这种模型里，把原始 principal 继续带到后面的 eBPF enforcement point 可能没有价值。只有当 downstream enforcement、audit、revocation、rate control 或 provenance 仍然依赖 original request identity 时，handoff capability 才值得存在。

所以最实际的下一步不是先给每个 packet 塞更多 metadata，而是先做 benchmark，让 proxy identity confusion 可观测，再验证一个 compact generation-scoped handoff 是否真的能消除普通 socket/proxy identity 无法避免的错误。