---
date: 2026-08-26
title: "一次已经撤销的授权，能在 eBPF 数据路径里存活多久？"
description: "eBPF 数据路径会把授权结果保存在连接、socket、认证和策略状态里。本文提出 scoped revocation epoch、跨层 barrier 与撤销延迟 benchmark。"
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Revocation
research_question: "当一次已经允许的决策被写进持久数据路径状态后，eBPF 安全系统怎样在不清空所有状态、也不让每个事件都回到用户态查策略的前提下，为 stale authorization 建立可验证的时间上界？"
source_cutoff: 2026-08-26
status: daily-report
---

# 一次已经撤销的授权，能在 eBPF 数据路径里存活多久？

假设一个 workload 在 10:00 被允许建立连接。10:01，它的身份被撤销。策略对象已经更新，但 packet 仍然经过一个记得旧决定的数据路径。

此时授权到底还存在哪里？

它可能已经变成 endpoint 的 policy revision、BPF policy map 里的 entry、conntrack 状态、authentication cache、socket-local BPF storage，或者 sockmap 里保存的 socket 引用。其中一些状态正是因为旧策略曾经允许这个 flow 才被创建出来，而且可能跟 socket 或 connection 一样存活很久。更新策略源，并不能自动证明所有携带旧授权的副本已经失效。

<!-- more -->

这和“状态转换是否合法”不是同一个问题。上一篇每日报告 [eBPF 能验证有状态安全策略，而不只是验证字节码安全吗？](https://eunomia.dev/zh/research/ebpf-stateful-policy-verification/) 关注 persistent security state 是否遵守合法的 transition relation。今天即使 `ALLOWED -> REVOKED` 这个 transition 完全正确，仍然有另一个问题：**旧的 allow 到底什么时候才在所有缓存过它的位置上真正不能再用？**

生产系统需要的也不只是 eventual convergence。有些 revoke 是普通配置变化，有些则来自 credential 泄漏、workload identity 改变、endpoint 被隔离，或者 tenant 失去访问权限。后几种情况下，operator 真正想得到的是一个 bound，比如：revision 42 撤销 principal X 以后，任何 enforcement path 使用 revision 41 接受 X 的时间都不能超过 100 ms。

看起来这是 control plane 的同步问题，但 eBPF 让它同时变成 datapath state 问题。fast path 之所以快，恰恰是因为它会复用已经计算出来的状态，而不是每次都回用户态重新判断。

## 授权本来就可能比一次 policy lookup 活得更久

Linux 已经提供了很多适合保存这种状态的 primitive。

[`BPF_MAP_TYPE_SK_STORAGE`](https://docs.kernel.org/bpf/map_sk_storage.html) 会把 BPF value 直接挂在 socket 上。kernel 会在 socket 或 map 删除时释放它，BPF program 和用户态也都可以创建、更新、读取或删除这份状态。把 per-socket metadata 放到 enforcement 附近非常方便，但如果系统没有额外定义 validity rule，它自然会跟着 socket lifetime 走。

[`BPF_MAP_TYPE_SOCKMAP` 和 `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html) 会保存 socket reference，并允许 BPF verdict 与 redirect program 使用这些 socket。也就是说，一次 decision 完全可能被物化成在原始事件结束后继续存在的数据路径状态。

Cilium 在更大规模上展示了同一种模式。它的 [policy 文档](https://docs.cilium.io/en/latest/security/network/policyenforcement/) 明确把 session protocol 的网络策略定义成 stateful：如果 `A => B` 被允许，B 到 A 的 reply traffic 会自动被允许，但 B 重新发起一个新 connection 仍然需要自己的规则。它的 [eBPF map 文档](https://docs.cilium.io/en/latest/network/ebpf/maps/) 也列出了 node-scoped connection-tracking map 和 authentication map，这些都是生产数据路径里的持久状态。

当前 Cilium source 进一步暴露了 rollout 的边界。endpoint policy 会维护 desired 与 realized policy revision，endpoint regeneration 可以异步发生，而且同一次 policy update 可能在一部分 endpoint 上成功、在另一部分 endpoint 上失败。CLI 甚至提供了 [`cilium-dbg policy wait`](https://docs.cilium.io/en/stable/cmdref/cilium-dbg_policy_wait/) 来等待所有 endpoint 更新到指定 revision。换句话说，policy convergence 本身就是一个需要观察和等待的过程，并不是瞬间成立的性质。

更窄的 expiration 机制也已经存在。Cilium mutual authentication 的设计会给 auth cache 保存 expiry，当前 Helm 配置也暴露 authentication garbage-collection interval。expiry 很有用，但 certificate timeout 和 emergency revoke 是两种不同语义。除非 datapath 每次复用 entry 时都会检查某个 validity condition，否则“五分钟做一次 GC”本身并不能证明 stale authorization 最多只存活五分钟。

所以真正值得问的不是这些系统“有没有 revocation bug”。更一般的问题是：eBPF 安全系统有没有一种清晰、可检查的方式来说明 **已经被 admit 的 authority 最多还能继续生效多久**。

## Policy revision 还不是 revocation proof

revision number 能说明某个 component 认为自己已经实现了哪个版本的策略，却未必枚举了所有还携带旧 authority 的派生对象。

可以想象下面四个状态同时存在：

```text
policy map:       revision 42 已经 deny principal X
endpoint:         realized policy revision = 42
conntrack entry:  flow F 是在 revision 41 下 admit 的
socket storage:   auth_generation = 41, authenticated = true
```

前两行完全可以是正确的，但后两行是否危险，取决于 datapath 怎样处理 established flow。反过来，如果为了保证重新决策而把所有 connection state 都删掉，也可能中断大量与这次 revoke 完全无关的合法流量。

这形成了一个很直接的 systems trade-off。每次重新 policy lookup 可以得到最新 authority，但会牺牲 fast path；永远复用 cache 很便宜，但旧决定可能活得比产生它的策略更久；固定 timeout 处在中间，但 stale window 由 timeout 决定，而不是由 revoke 的紧急程度决定。

更好的接口应该把这个 trade-off 变成显式 contract。

## 现有研究还缺什么

### 1. 缓存的授权通常没有显式 stale-allow budget

connection 或 socket 可以记住一次旧 decision 已经成功，但这些状态往往没有说明：一旦 policy 或 identity 改变，这个成功结果还能作为 authority 使用多久。

缺少的是直接绑定在 authorization-bearing state 上的 validity contract。它至少要记录这次 allow 来自哪个 policy/principal generation，以及对应 generation 被撤销后允许残留多久。

后果是 operator 只能看到新 policy revision 已安装，却无法回答旧 connection、authentication record 或 socket-local cache 是否还能继续放行。

最直接的 test 是：先建立 long-lived flow，再 revoke 对应 identity 或 rule，然后测出最后一个仍然因为 pre-revocation state 而被 accept 的 packet。对 conntrack、auth cache、socket-local state、endpoint regeneration 与 controller failure 都重复一次。

### 2. Control-plane convergence 和 datapath invalidation 不是同一个完成条件

Cilium 暴露 desired/realized policy revision，也有等待 endpoint update 的命令。这已经承认 policy rollout 是异步的。但一个通用 revocation contract 还需要多问一句：哪些由旧 policy 派生出来的 authorization cache 必须失效以后，才能说 revoke 已经完成？

缺少的是 cross-layer completion barrier。如果 endpoint 已经到 revision 42，但某个 endpoint policy map 之外的 authorization-bearing object 仍能使用 revision 41，那么 revision 42 还不是完整的撤销证明。

后果是 control plane 可能报告“已经 converged”，而旧 authority 仍然存在于另一个 map、socket、node 或 userspace redirect path。

可以故意让一个 enforcement domain 延迟，其他 domain 正常 convergence，然后检查系统是否会过早报告成功。

### 3. Flush-all 的正确性来自它过度破坏

处理 stale state 最简单的安全方式是把所有可能依赖旧策略的状态都删掉。这样当然能去掉 stale allow，但也会丢掉无关 connection state 与 load-balancing decision。Cilium 自己的配置注释也说明，重建 BPF state 可能扰动已有 connection，并改变 established traffic 的 decision。

缺少的是 selective invalidation：只找到或廉价拒绝由被撤销 subject、policy generation、credential 或 authority domain 派生出来的状态。

否则 emergency revoke 可能只能在“制造较大 outage”和“容忍较长 stale window”之间二选一。

一个有区分度的实验是：在大量 unrelated long-lived connection 中只 revoke 一个 principal，然后比较 flush-all、timeout-only 与 selective mechanism 的 stale permit、false drop、reconnect 数量和 convergence time。

### 4. Controller downtime 可能让“最终会过期”变成没有上界的假设

用户态 controller 可以负责 GC map entry 与修复状态，但安全 bound 不应该默默依赖 controller 一定能准时调度。Cilium 较早的 mutual-authentication design 就指出过这个问题：如果 datapath 自己看不到 expiry，那么 agent 长时间 down 时，已经认证的 session 可能继续保持认证状态。

缺少的是即使用户态延迟，datapath 仍能独立检查的 safety condition。

否则 incident 中最糟糕的 failure mode，也就是 controller 同时挂掉，反而可能让 revocation 变弱。

测试方法是在 revoke 前一刻停掉或 partition controller，再观察 stale-allow bound 是否仍然成立。

## 兼具学术价值与生产价值的方向

### 1. 给所有 authorization-bearing state 加 scoped revocation epoch

**Gap。** 已经缓存的 allow 缺少一种便宜、统一的方法来证明创建它的 authority 仍然有效。

**Mechanism。** 为每个 authorization domain 维护单调递增的 revocation epoch。domain 可以是 identity、endpoint、policy subject set、credential class，或者 policy compiler 选择的其他有限集合。所有可能绕过 fresh policy lookup 的 cached allow 都保存创建它时的 epoch：

```text
cached_allow = {
  principal_id,
  policy_id,
  revocation_epoch,
  optional_deadline,
  decision_metadata
}
```

fast path 在复用 allow 前，把 cached epoch 与当前 domain epoch 比较。revoke 只需要递增 epoch，旧 entry 即使没有同步删除，也立刻变成 unusable。

这里不应该只有一个 global epoch，因为一个 tenant 的 revoke 会把所有人的 cached decision 都打掉。可以设计很小的 hierarchy，比如 endpoint epoch 加 principal epoch，由 compiler 决定某类 cache entry 必须检查哪一个。这样 hot path 的 lookup 数量仍然有固定上界。

对于 socket-local storage，epoch 跟 socket 一起保存；对于 conntrack 或 auth map，则成为 value 的一部分，或者从一个紧凑 side map 得到。某些 domain 无法精确 index 时，还可以用 deadline 作为 fallback。

**与现有工作的差别。** 普通 policy revision 描述 configuration progress，authentication expiry 描述时间有效性。scoped revocation epoch 则直接让 cached authorization 依赖一个具体 invalidation generation，使 datapath 不必同步删除 object 也能拒绝 stale authority。它也比上一篇 temporal state contract 更窄，不描述任意合法 state transition，只针对一个性质：revoke 后旧 allow 还能被复用多久。

**Artifact。** 一套 libbpf-friendly authorization-state schema、generated lookup helper、scoped epoch map，以及 socket storage 和 connection/auth map adapter。

**Evaluation。** 比较 no-recheck、TTL-only、global epoch、scoped epoch 与 synchronous deletion。workload 包含 long-lived TCP、UDP request/reply、mutual-auth cache、socket-local state 和 mixed tenant。测 stale-allow duration、stale packet、false invalidation、per-packet cycle、cache miss、map memory 和 update fan-out。还要包含完全不发生 revoke 的 workload，暴露 steady-state overhead。

**学术价值。** 一般问题是：怎样给 in-kernel cached authorization 加 revocation consistency，同时又不把每个事件变成 remote policy lookup。

**生产价值。** operator 可以得到可测的 emergency-revocation bound，同时保住无关的 long-lived flow。

**失败条件。** 如果额外 epoch lookup 的成本已经接近重新执行目标 policy，或者真实 authorization domain 根本无法细分、每次 revoke 最终还是要 invalidate 大多数状态，那这个机制就没有优势。

### 2. 把 revocation completion 做成 cross-layer barrier，而不是 controller 猜测

**Gap。** policy object 已经更新、endpoint revision 已经追上，并不能单独证明所有 authorization-bearing domain 都跨过了 revocation boundary。

**Mechanism。** 每次 revoke 分配一个单调递增 `revocation_id`，并声明它要求哪些 enforcement domain 完成。domain 可以包括 endpoint policy map、identity cache、auth map、connection state、socket policy，以及 userspace-managed redirect state。每个 domain 在能够保证早于这次 revoke 的 decision 已经被 epoch check 拒绝或物理删除后，发布自己的 high-watermark。

只有所有 required watermark 都达到目标 ID，controller 才把 revoke 标记为 effective：

```text
revocation 82 requires:
  endpoint_policy >= 82
  auth_cache      >= 82
  connection_auth >= 82
  socket_policy   >= 82
```

某个 domain 卡住时不能静默成功。policy 可以选择 bounded fail-closed、short lease，或者显式 `unknown`，而不是假装已经 converged。

这和 distributed system 常见 rollout barrier 有相似结构，但这里真正要研究的是 control-plane policy revision 与 datapath-derived authority 的边界。Cilium 的 `policy wait` 是很强的 endpoint-revision baseline。实验必须找到 endpoint convergence 仍不足以说明 revoke 完成的 case，并证明额外 barrier 确实能抓住它。

**与现有工作的差别。** transactional upgrade protocol 追求 coherent program/map generation 的切换。这里不要求所有 state 原子变化，而是要求能验证旧 authorization 在多个独立更新的 enforcement domain 上什么时候都变得 unusable。

**Artifact。** revocation controller、紧凑的 per-domain acknowledgement map、把 requested/realized/effective revision 分开的 status API，以及 fault injection hook。

**Evaluation。** 注入 delayed endpoint regeneration、一个 stale node、controller crash/restart、map update failure、socket lifetime extension 与 auth-cache delay。baseline 包括 endpoint-revision waiting、fixed sleep、global flush 和 barrier。指标是 early-success error、revocation completion latency、系统已经报告成功后仍出现的 stale permit、false deny 与 recovery time。

**学术价值。** 把 revoke 从一个 eventually consistent 运维动作变成可以跨 heterogeneous enforcement layer 测量的 consistency property。

**生产价值。** incident responder 可以问“撤销是否真的已经生效”，而不是“configuration object 有没有被接受”。

**失败条件。** 如果实际系统所有 authorization path 本来就共享同一个 generation 与 update boundary，这个 barrier 只是多余 plumbing。benchmark 应该包含这种简单系统，让 simpler design 在该赢的地方赢。

### 3. 用“最后一次 stale allow”定义 revocation benchmark

**Gap。** 很多 security benchmark 测 policy throughput 或 rule-update rate，但它们不回答 incident-response 最重要的问题：发出 revoke request 之后，最后一个仍然因为旧 authority 被接受的 packet、syscall 或 message 是什么时候？

**Mechanism。** 构造 authorization lineage 已知的 workload，在受控时间点注入 revoke。每个 accepted event 带足够的 test-only identity，让 harness 能判断它是被 pre-revocation 还是 post-revocation state 授权。benchmark 至少记录四个时间：

1. revoke requested；
2. policy source updated；
3. 每个 enforcement domain 报告 revocation realized；
4. 最后一个使用 stale authority 被接受的 event。

fault 包含 control-plane pause、endpoint regeneration delay、CPU contention、map pressure、long-lived TCP flow、UDP reply state、socket-local cache、program replacement 与 node partition。另一组实验同时修改一个 unrelated identity，用来测 collateral invalidation。

**Artifact。** open trace/replay corpus、eBPF fault-injection adapter、ground-truth revocation label，以及统一 stale-allow distribution report。

**Evaluation。** 在同一 policy-update workload 下比较当前系统、TTL-only expiry、flush-all、scoped epoch 和 epoch-plus-barrier。核心指标是 P50/P99/max stale-allow duration、stale permit 数量、false deny、connection disruption、CPU overhead 与 map update amplification。这里 max 特别重要，因为 emergency-revocation contract 的目的就是限制 long tail。

**学术价值。** 把一个 security consistency property 变成跨系统可以比较的指标，而不是只能比较彼此不兼容的 policy-update API。

**生产价值。** CNI、runtime security、service mesh 与 socket-policy 项目可以得到一套 regression test，专门检查最容易被“配置更新成功”掩盖的 revocation failure。

**失败条件。** 如果真实 deployment 无法可靠知道某个 event 究竟用了哪份 cached decision，那么 benchmark 可能只能用于 controlled prototype。此时更应该先做 authorization provenance，而不是急着宣称有 universal score。

## 哪些结果会改变这个判断？

最强的反例是：现有 policy regeneration、connection-state handling、authentication expiry 与 targeted map update 已经能把 stale authorization 压到真实 deployment 需要的范围内。如果 ground-truth benchmark 证明当前系统在 long-lived flow、controller failure、map pressure 与 mixed tenant 下仍然稳定满足目标 revoke bound，那么新的 revocation protocol 只会增加复杂度，并不会提高 security。

第二个反例是成本。每次 fast-path reuse 都做 generation compare，可能增加一次 map lookup 或 cache miss。如果这个 overhead 抹掉了 caching 本来的性能收益，那么 selective synchronous deletion 或更短 lease 可能是更好的工程选择。

所以本文真正想得到的结果并不是“所有 eBPF 系统都应该用 epoch”。更合理的 contract 是：**只要安全系统会复用 cached authorization，它就应该能说明最大 stale-allow window 是多少、哪些 state 还能携带旧 authority，并证明 control plane 延迟时这个 bound 仍然成立。**

epoch、lease、targeted invalidation 或其他机制都可以实现这个性质。真正不应该继续默认的是：把“新 policy revision 已安装”直接当成“所有旧 allow 已经死亡”的证明。

## 参考资料

- Linux kernel documentation: [`BPF_MAP_TYPE_SK_STORAGE`](https://docs.kernel.org/bpf/map_sk_storage.html)
- Linux kernel documentation: [`BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- Cilium documentation: [Policy Enforcement](https://docs.cilium.io/en/latest/security/network/policyenforcement/)
- Cilium documentation: [Endpoint Lifecycle](https://docs.cilium.io/en/latest/security/policy/lifecycle/)
- Cilium documentation: [eBPF Maps](https://docs.cilium.io/en/latest/network/ebpf/maps/)
- Cilium CLI documentation: [`cilium-dbg policy wait`](https://docs.cilium.io/en/stable/cmdref/cilium-dbg_policy_wait/)
- Cilium source: [`pkg/endpoint/policy.go`](https://github.com/cilium/cilium/blob/main/pkg/endpoint/policy.go)
- Cilium source: [`bpf/lib/conntrack.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/conntrack.h)
- Cilium source: [`bpf/lib/host_firewall.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/host_firewall.h)
- Cilium Helm configuration: [authentication and BPF map settings](https://github.com/cilium/cilium/blob/main/install/kubernetes/cilium/README.md)
- Cilium design CFP: [Mutual authentication updates](https://github.com/cilium/design-cfps/blob/main/cilium/CFP-28986-mutual-auth-updates.md)
