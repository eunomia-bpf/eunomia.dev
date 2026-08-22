# 为什么无人抓取的 OBI Prometheus endpoint 仍会让内存持续增长？

**简短回答：** “没有被 scrape”不等于“已经禁用”。OpenTelemetry eBPF Instrumentation（OBI）即使没有收到任何 HTTP 请求，也可能继续发现服务并更新带 label 的 Prometheus metric children。当前上游 Helm chart 的默认 OBI 配置会在 `9090` 端口启用 `prometheus_export`，但默认不创建 metrics Service 或 ServiceMonitor。一项仍然 open 的 chart 修复说明，OBI 是在处理 scrape 时清理过期 metric children；如果没有 scrape 驱动这条路径，旧 series 就可能留在进程中，并随着 label set 变化而持续占用更多内存。

如果部署只通过 OTLP 导出 metrics，应关闭内置 Prometheus exporter，而不是让一个不可达 endpoint 一直运行。如果确实要让 Prometheus 抓取 OBI，则要显式配置并测试完整路径：exporter、container port、discovery 或 Service、ServiceMonitor 或 scrape configuration，以及 target health。对一个从未进入 collection path 的 endpoint，仅调低 Prometheus TTL 不是可靠修复。

## 为什么 TTL 不一定能限制内存

当 `prometheus_export.port` 非零时，OBI 会启用 Prometheus exporter。文档中 `ttl` 的含义是：未更新的 metric instance 经过这段时间后不再报告，默认值为五分钟。这个描述看起来像 wall-clock retention 上限，但当前 chart 讨论揭示了更关键的实现细节：**expiration 在哪里执行**。讨论报告的清理动作发生在处理 scrape 的过程中。

于是会形成下面的失效路径：

```text
被观测流量持续到达
        -> 新增或变化的 label set 更新 metric children
Prometheus endpoint 已配置
        -> 进程内 exporter 保留这些 children
没有 scraper 到达 endpoint
        -> collection-time expiration 没有被驱动
        -> 旧 children 累积，resident memory 可能持续上升
```

增长的是 user-space exporter state，不能因为 OBI 使用 eBPF 采集就直接归因于 eBPF map。长时间运行的 DaemonSet 更容易暴露这个现象：应用实例、route、status code、peer identity 或其他维度会不断变化，而 exporter 进程一直不退出。

在正常发生 collection 时，TTL 仍然有意义，它决定 exporter 何时忽略或删除不活跃 metric instance。但除非实现明确提供独立于 scrape 的定时 expiration，否则不能把 TTL 当作后台垃圾回收保证。

## Helm 默认值如何产生不可达 exporter

在本文写作时，chart 默认值同时包含下面三个设置：

```yaml
service:
  enabled: false

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics

serviceMonitor:
  enabled: false
```

因此，render 后的 OBI 配置会启动 Prometheus exporter，但 chart 自己没有创建 discovery path。如果另一个系统直接发现 pod、添加 annotations 或提供独立 Service，这可能是有意配置；如果部署只使用 OTLP，并且没有任何组件抓取 `9090`，这就是没有消费者的多余状态。

一项仍为 open 的 Helm-chart PR 提议：除非启用 chart 管理的 Service，否则从 render 配置中删除 `prometheus_export`。它的测试覆盖默认、启用 Service 和只启用 ServiceMonitor 三种情况。由于 PR 尚未合并，不能把这种行为描述为已发布功能。它也暴露了兼容性选择：`service.enabled: false` 并不能证明没有 scraper，因为 direct pod discovery 和独立管理的 Service 都是有效拓扑。

比“Service 关闭则 exporter 关闭”更安全的不变量是：

```text
Prometheus exporter 已启用 <=> 存在有意配置且经过测试的 scrape consumer
```

Chart 应让这个不变量显式化：提供独立 exporter 开关，或清楚定义用户自带 `prometheus_export` 配置的优先级。

## 有意识地选择三种部署模式之一

### 1. 只用 OTLP 导出 metrics

保留 `otel_metrics_export`，关闭内置 scrape endpoint。按当前 OBI 配置模型，port 为 `0` 或未设置时不会打开 Prometheus endpoint：

```yaml
config:
  data:
    otel_metrics_export:
      endpoint: "http://${HOST_IP}:4318"
    prometheus_export:
      port: 0

service:
  enabled: false

serviceMonitor:
  enabled: false
```

上线前必须 render 正在使用的准确 chart 版本。Helm map merge、外层 values wrapper 或后续 chart revision 都可能重新引入默认 stanza。测试应断言 render 后的 OBI 配置中 Prometheus port 为零或不存在，同时 DaemonSet 不声明 application-metrics port。

### 2. 由 chart 管理 Prometheus scrape

为 Prometheus 到 exporter 建立一致路径：

```yaml
service:
  enabled: true

serviceMonitor:
  enabled: true

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics
```

仅创建对象还不够。需要确认 Service selector 能匹配 OBI pods，Service port 能解析到 exporter container port，ServiceMonitor selector 会被目标 Prometheus 实例选中，并且 target 确实健康。如果 ServiceMonitor 存在却没有被 Prometheus 采用，仍然属于没有 scrape 的状态。

### 3. Direct pod discovery 或独立管理的 Service

显式保留 exporter，但不要假设 chart 自己的 Service 是唯一合法 consumer。应在 values 和测试中记录外部 ownership：

```yaml
service:
  enabled: false

serviceMonitor:
  enabled: false

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics
```

然后针对 render 后的 pod 验证独立 scrape configuration。这种模式正是 chart 不能仅因为自身 Service 关闭，就静默删除用户显式 exporter 配置的原因。在采用改变 render 规则的后续版本前，应先针对 pending 行为完成升级测试。

## 不要混淆应用 metrics 与 OBI internal metrics

`prometheus_export` 暴露 OBI metrics features 选中的应用、网络及其他被观测 metrics；`internal_metrics` 报告 OBI 自身行为，并且可以独立选择 Prometheus 或 OTLP。两者设置成相同端口时可以共享 HTTP server，但配置和 metric families 仍然不同。

关闭没有使用的 application Prometheus exporter，不代表必须丢失 OTLP application metrics 或 OBI self-observability。应分别选择每条信号路径：

- application metrics 使用 OTLP、Prometheus 或两者；
- OBI internal metrics 禁用、通过 OTLP 发送，或暴露给 Prometheus；
- 每个启用的 pull endpoint 都要有一个可达 consumer。

这种区分也能改进诊断。如果 application series 数量稳定但 RSS 上升，应继续检查其他来源；如果 application series label churn 上升，同时 scrape request 为零，就更支持不可达 exporter 这一机制。

## 有边界的生产诊断方法

使用 canary，把配置意图与运行时证据逐项对齐。

1. **Render release。** 检查生成的 ConfigMap、DaemonSet ports、Service 和 ServiceMonitor。不要只根据一段 values 推断运行时配置。
2. **证明 scrape 是否发生。** 检查 Prometheus target health 和 scrape counters。如果启用了 internal metrics，`obi_prometheus_http_requests_total` 可以显示是否有请求到达 OBI scrape endpoint。
3. **区分进程与内核内存。** 同时观察 container RSS 或 working set、eBPF map memory 和 map entry 数。Exporter retention 应主要体现在 OBI 进程内。
4. **跟踪 churn，而不只是流量。** 统计活跃服务以及可能产生 series 的 label 组合。请求量稳定并不代表历史 label turnover 有界。
5. **每次只改一个变量。** 在一个 canary 中只关闭 `prometheus_export`，保留 OTLP path 和相同 workload。内存斜率停止增长，比重启更有解释力，因为重启会清空全部进程状态。
6. **验证相反 canary。** 如果需要 Prometheus 输出，保持 exporter 启用，并让真实 scraper 以正常 interval 抓取。确认 series expiration 和 target continuity 都正常。
7. **把 memory limit 当作 containment。** Limit 能保护 node，但不是修复；OOM restart 可能掩盖持续存在的配置错配。

比较结果时至少等待完整 TTL 加上多个 scrape intervals。立刻变平不代表旧 children 已经被清理，立刻下降也可能只是进程重启造成的。

## 上游应该修复什么？

长期方案需要同时满足两个属性：

1. **Lifecycle cleanup 不能意外依赖 consumer activity。** 即使 scrape 延迟或完全不存在，exporter state 也应有有界 expiration 机制。
2. **Packaging 不应隐式启用未使用组件。** 只有 operator 明确选择 scrape mode 时，chart 才应 render Prometheus exporter，同时保留用户显式配置的 direct-scrape 路径。

在两者都实现前，operator 应显式关闭未使用 exporter，并为选定模式添加 render tests。当前 PR 是有用证据和 packaging mitigation 提案，但尚不是 release guarantee，也不能替代 runtime expiration。

## 参考资料

- [OBI 数据导出配置：Prometheus endpoint、port、TTL 与 instrumentations](https://opentelemetry.io/docs/zero-code/obi/configure/export-data/#prometheus-exporter-component)
- [当前 OpenTelemetry eBPF Instrumentation Helm values](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/values.yaml)
- [关于省略无人抓取 `prometheus_export` 的 open Helm-chart PR](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360)
- [OBI internal-metrics reporter 配置](https://opentelemetry.io/docs/zero-code/obi/configure/internal-metrics-reporter/)
- [OBI 已导出 metrics，包括 Prometheus scrape-request telemetry](https://opentelemetry.io/docs/zero-code/obi/metrics/)
- [关于统一 OBI OTLP 与 Prometheus export paths 的 open design issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974)
- [关于 operation cost 的 GenAI 语义约定 open issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)
- [关于 `gen_ai.usage.cost.*` 的 GenAI 语义约定 open PR](https://github.com/open-telemetry/semantic-conventions-genai/pull/443)
- [BPF 线程：为 JITed programs 添加 KASAN checks](https://lore.kernel.org/bpf/20260822-kasan-v7-0-99afee6ef7fd@bootlin.com/T/#t)
- [BPF 线程：为 signed loaders 提供专用 keyring 和 ML-DSA 支持](https://lore.kernel.org/bpf/20260821214111.1120748-1-daniel@iogearbox.net/T/#t)
- [BPF 线程：修复 conditional jump 的 verifier non-null inference](https://lore.kernel.org/bpf/20260821-bug-029-bad-non-null-inference-v1-1-45ddc0f7c308@gmail.com/T/#t)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 和频道身份、封闭聊天链接、精确时间、私有拓扑、原始日志及可搜索回原讨论的措辞均已删除，也没有保留原始 transcript。

### Pull exporter 需要显式 consumer 和独立 lifecycle

当天最强的运维问题涉及一项小型 Helm 改动：当基于 eBPF 的 instrumentation pod 的内置 Prometheus endpoint 没有被 scrape 时，减少其内存占用。公开配置确认了错配：chart 默认启用 exporter，同时默认关闭 Service 和 ServiceMonitor。[仍然 open 的 chart 变更](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360)会在 chart-managed Service 不存在时删除 exporter，并记录 collection-time series expiration 机制。

当前应先把每个部署归类为 OTLP-only、chart-managed scrape 或 externally managed scrape，然后 render 并测试准确模式。未决设计问题是 packaging switch 是否已经足够：direct pod scraping 必须继续可用，而 exporter cleanup 也不应依赖 consumer 主动调用 endpoint。

### GenAI cost semantics 已变得具体，但仍是提案

另一条可观测性讨论从“是否应该记录 cost”进入更难的问题：这个值究竟表示什么。一项 [open tracking issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)和一项[正在推进的 PR](https://github.com/open-telemetry/semantic-conventions-genai/pull/443)提出 `gen_ai.usage.cost.*` 字段，但约定尚未合并或稳定。讨论区分了 provider billed value 与本地 estimated value，并指出 currency 和 pricing-source provenance 是可互操作 schema 的必要组成。

生产 telemetry 目前应保留版本化、带 namespace 的实验字段，记录值属于 billed 还是 estimated，并保留计算使用的 currency 与 pricing revision。在约定发布前，不要让 alert 或跨 provider dashboard 依赖这些 proposed names。

### 内核工作把可观测性推进到生成代码与信任边界

公开 BPF review 集中在 verifier 正常通过仍不能覆盖的 failure domains。一组 [KASAN-for-JIT series](https://lore.kernel.org/bpf/20260822-kasan-v7-0-99afee6ef7fd@bootlin.com/T/#t)提议在 x86 BPF instructions 翻译为 native code 时加入 memory-access checks。当前 revision 明确只覆盖 x86 上的 generic KASAN，并排除 BPF-stack 和 potentially faulting access classes，因此它是带明确边界的 review-stage diagnostic capability，不是通用 JIT memory safety。

另一组 [signed-loader series](https://lore.kernel.org/bpf/20260821214111.1120748-1-daniel@iogearbox.net/T/#t)提议提供专用 BPF-only keyring：默认以 sealed empty 状态启动，并加入 ML-DSA tooling 与 end-to-end tests。这里有价值的 security boundary 是 scope：loader 自己控制的 keyring 本身不能证明多少信任，而由 boot 阶段配置并受限制的 BPF keyring 可以表达 operator policy，又不必把这把 key 扩展到更广泛的 kernel trust hierarchy。两个系列目前都仍是上游提案。

项目支持、调度支持、网络和公开论坛类目标在当日窗口内没有实质新技术交流，或只有例行自动通知与非技术协调。它们均可访问，并计为已检查，而不是跳过。
