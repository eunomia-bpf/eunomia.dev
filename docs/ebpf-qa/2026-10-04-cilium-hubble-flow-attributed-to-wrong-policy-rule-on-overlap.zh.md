# 当两条重叠的 L3/L4 策略条目同时命中时，为什么 Hubble 会把流量归因到错误的策略规则，而 BPF 数据面实际执行的是正确的那条？

执行不受影响——BPF 数据面选中的是正确的条目。误归因发生在策略决策的用户态副本里：`pkg/policy/mapstate.go` 是 C 数据面同一套条目选择逻辑的 Go 重新实现，而 v1.20 里这次查表重构把 LPM 前缀长度比较写反了。`mapState.lookup()` 在两条同优先级的重叠规则下会返回另一条条目，于是 Hubble 在 `egress_allowed_by` 里把流量记到错误的规则上，而记录的 `policy_match_type` 描述的却是数据面实际命中的条目——这条记录自相矛盾。

## 机制

Cilium 把 L3/L4 策略存在一张以 LPM-trie 为键的 map 里。对一条流量，数据面做两次查表：一次用具体的远端身份，一次用该身份的聚合身份（例如集群内 pod 用 `aggregate-cluster`，全网流量用 0）。当两条条目都存在时，`bpf/lib/policy.h` 按固定的优先级阶梯裁决：

1. 具体身份条目处于最高优先级（优先级 0 的拒绝）时，直接选中，不再看聚合条目。
2. 否则，优先级更高的条目胜出。
3. 优先级相同时，`lpm_prefix_length` 更长的条目胜出——即 L4 更具体的那条。通配端口的条目前缀短（只有协议位），固定端口的条目则是完整的协议加端口长度，所以同一优先级下 port-80 规则压过任意端口规则。
4. 前缀长度打平时，选具体身份条目。

策略会计（policy accounting）维护的每条条目字节/包计数器让这一切可观测：数据面命中哪条，哪条的计数器就会动。

v1.20 的 Go 副本在第 3 步上把 allow 路径写反了：

```go
if idKey.PrefixLength() > aggKey.PrefixLength() {
    return authOverride(aggEntry, idEntry), true
}
```

对照数据面读，这个条件方向反了：具体条目前缀更长时它返回聚合条目，聚合条目前缀更长时它落到具体条目。只有打平那一个点巧合一致。这个反转是在查表重构时引入的；重构之前该比较与数据面一致。

deny 路径同病异症：同优先级的拒绝对总会直接返回具体条目，而数据面只有在优先级 0 的拒绝时才这样做，其余情况仍比较前缀长度。两侧裁决结果相同，所以执行同样不受影响——差别只在归因到哪条规则。

## 验证与调试路径

上游 issue 里的复现（见下方参考）给出了完整公开路径：

1. 为同一条流量造两条同优先级、L4 具体度不同的重叠条目。例如：具体身份的任意端口 allow，配一条 `toEntities: all` 的聚合 allow 且指定端口——数据面必须选中带端口那条。
2. 读数据面真相：`cilium-dbg bpf policy get <endpoint-id>` 列出每条策略 map 条目及其逐条 BYTES/PACKETS 计数器（策略会计默认开启）。发流量后，前缀更长那条的计数器在动，另一条纹丝不动。
3. 读用户态归因：`hubble observe` 开 JSON，对比 `egress_allowed_by`（Hubble 把流量归给哪条规则）与 `policy_match_type`（它报告的是哪条条目特征）。
4. 这个 bug 的特征就是两者对不上：计数器说命中了一条，归因却说是另一条规则给的裁决，而 match type 描述的形状与被归因的规则对不上（例如 L4-only 的 match type 挂到一条没有任何端口的 L3-only 规则上）。数据面裁决——放行还是拒绝——始终正确，偏的只有归因。

## 局限

- 数据包执行从未受影响，任何方向都没有。C 数据面是正确的；缺陷只限于用户态副本，因此所有读取它的东西——Hubble 的策略归因，以及使用同一查表的 Go 侧测试——继承的都是错误的规则归因，永远不会是错误裁决。
- 截至本次运行日，stable v1.20 线（至 v1.20.2）仍带着反转的比较。修复于 2026-09-29 以两个提交落在 `main` 上，由 issue 报告者推送——allow 路径修正，外加一个后续提交把同优先级拒绝改走前缀比较而不是无条件返回具体条目——它随 v1.21.0 预发布线首发（已确认 v1.21.0-pre.3 中带上）。运行日没有 1.20.x 反向移植在进行中。
- 上游 issue 在关闭时引用的是自动化报告刷屏、且当时没有已确认真实用户投诉；上面的复现展示了 Hubble 输出上具体的用户可见影响。在升级到带上该修复的版本之前，不要让 Hubble 的规则归因独自背书同优先级、端口具体度不同的重叠规则——用逐条计数器交叉核对，或规划升级到 1.21 线。

## 参考

- [cilium/cilium issue 48945](https://github.com/cilium/cilium/issues/48945) — 上游报告：反转的比较、它矛盾的数据面 C 规则，以及 kind 集群复现与计数器/Hubble 输出。
- [cilium/cilium 拉取请求 49062](https://github.com/cilium/cilium/pull/49062) — 一个被拒绝、未合并关闭的社区修复 PR；其审阅意见独立指出了 deny 路径的同症状变体。
- [Cilium v1.20.2 — bpf/lib/policy.h](https://github.com/cilium/cilium/blob/v1.20.2/bpf/lib/policy.h) — 数据面优先级阶梯、LPM 前缀长度常量与逐条策略会计。
- [Cilium main — pkg/policy/mapstate.go](https://github.com/cilium/cilium/blob/main/pkg/policy/mapstate.go) — 修正后的用户态查表，比较方向已改正。
- [Cilium — Hubble 可观测性](https://docs.cilium.io/en/stable/observability/hubble/) — 流记录如何携带策略归因。

## 当日社区讨论

选取的问题来自一个 opt-in 存档：一条讨论报告了命中两条重叠策略规则的流量被 Hubble 归因到错误规则的问题，线内回复指向了已跟踪的上游 issue 及其审阅中的修复——数据面正确、只有 Go 侧副本有偏，并附注 main 上包装结果的那个辅助函数已经没了、但比较本身仍是错的。该讨论前一天已被发现但被搁置，因为当时修复仍在审阅中；到本次运行修复已落入 main、上游 issue 已关闭，于是可以作为完整且已核对的答案发布。

当日其他讨论：一条 Cilium LoadBalancer 共享 VIP frontend 所有权讨论（两天前发布问题的再次提交，附维护者指向一个已跟踪的上游 issue 与一个审阅中的修复）、一条 GnuTLS HTTP/2 每连接 HPACK 解码器讨论（前一天发布的问题）、一条高频 socket 层 drop 时延与用户态 context switch 的基准请求（太薄未发布）、一条 OTEL Kubernetes 缓存环境变量讨论（前一天发布的问题）、一条 OpenTelemetry GenAI 语义约定讨论（记录智能体调用开始时哪些技能可用，属语义约定拉取请求，不在本页范围内）。

本次运行的渠道覆盖：两个 opt-in 存档共 11 条消息，全部如上覆盖。visible-browser-only 来源（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能审阅——没有可用的 visible-browser 会话——因此标记为未覆盖，而非平静。
