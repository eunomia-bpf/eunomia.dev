# 为什么在 OBI 的 Config v2 文档下，Helm 图自动注入的 Kubernetes 缓存地址环境变量会被忽略，缓存地址又该写在哪里？

这个变量没有坏——它是一个 Config v1 的运行时覆盖项，而 Config v2 不再把 `OTEL_EBPF_*` 变量作为叠加层应用到文档上。从 OBI v0.11.0 起，独立二进制和 OBI Collector receiver 都加载单一 Config v2 文档，而这个加载器不会把遗留的 `OTEL_EBPF_KUBE_META_CACHE_ADDRESS`（或其他任何 `OTEL_EBPF_*` 覆盖）读成配置。v2 文档有自己的规范字段——Kubernetes enricher 块下的 `metadata_cache.address` 字段——只有这个字段把 OBI pod 接到共享的 `k8s-cache` 上。Helm 图在 `k8sCache.replicas` 设定时仍按 v1 风格在 OBI daemonset 上导出那个环境变量，但它渲染出的 Config v2 文档里没有任何字段指向缓存服务，于是这个变量是无效的：每个 OBI pod 都回退到自带的进程内 informer 缓存，缓存服务收不到任何订阅，每个 pod 继续各自对着 Kubernetes API server 开 `LIST`/`WATCH` 流。修复办法是把地址写进 Config v2 文档，而不是继续导出变量。

## 机制

OBI 用一个可选的 `k8s-cache` 服务集中 Kubernetes 元数据。不再是每个 OBI pod 各开自己的 informer `LIST`/`WATCH` 流去连 API server，而是每个 pod 开一条 gRPC 流连到缓存；缓存统一运行一次 `Pod`/`Node`/`Service` informer，然后重放并流式推送元数据事件（`informer.EventStreamService/Subscribe`，其中 `SYNC_FINISHED` 事件标记初始重放结束）。"本地进程内 informer"与"订阅远端缓存"之间的开关是单个配置字段：

- Config v1：`attributes.kubernetes` 块下的 `meta_cache_address`，运行时可用 `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` 环境变量覆盖。
- Config v2：该字段迁移并改名为 Kubernetes enricher 块下的 `metadata_cache.address`。v1 路径不再被读取；只有 v2 路径有效。

Config v2 是单文档配置模型。迁移指南明确：v2 部署里 `OTEL_EBPF_*` 这类运行时环境变量不会被读作额外的 v1 配置——"只是把遗留变量带进 v2 部署并不能保留它的覆盖效果"。变量在 v2 里要起作用，只有当文档引用它时才行；而即便如此也取决于加载器——上游 `otelconf/x.ParseYAML` 路径在解码前展开 `${VAR}`、`${env:VAR}`、`${VAR:-fallback}` 与 `${env:VAR:-fallback}`，而 OBI 内部独立解析器直接解码文档字节，所以替换占位符只在执行展开的加载器里才存活。

这正是图（chart）的缺口。daemonset 模板输出：

```yaml
{{- if .Values.k8sCache.replicas }}
- name: OTEL_EBPF_KUBE_META_CACHE_ADDRESS
  value: {{ .Values.k8sCache.service.name }}:{{ .Values.k8sCache.service.port }}
{{- end }}
```

……但图渲染进 configmap 的 Config v2 文档里没有任何 `metadata_cache.address` 字段指向缓存服务。环境变量是 v1 覆盖项，v2 文档会忽略它。实际结果是：`k8sCache.replicas: 1` 部署了缓存服务和 daemonset 环境变量，却没有一个 OBI pod 真正连上。k8s-cache 设计说明把回退写得很清楚："如果未提供 k8s 缓存地址，OBI 会发起自己的本地进程内缓存"——这正是缓存服务想要避免的每个 pod 各自开 `LIST`/`WATCH` 的扩展模式。同样的缺口也适用于用户自带 v2 文档（例如经 `config.data`）的情形：缓存地址不会自动接入，除非写进那份文档。

## 验证与调试路径

确认地址是否真的在 OBI 加载的文档里，以及缓存是否看到任何订阅者：

1. 渲染或读取 daemonset 加载的 Config v2 文档（图 configmap，或带 `k8sCache.replicas: 1` 的 `helm template`）。在图的修复之前，`enrich.enrichers.kubernetes` 下没有 `metadata_cache:` 块；在 configmap 里 grep 缓存服务地址——它只出现在 daemonset 的 `env` 里，不在文档里。
2. 确认 pod 跑的是 Config v2 文档（schema `version: "2.0"`）。若仍是 v1 文件，环境变量是生效的，缓存收不到连接就指向缓存缺失或不可达，而不是变量被忽略。
3. 检查缓存侧。`k8s-cache` 在 Prometheus `/metrics` 端点暴露内部指标，并记录重连/订阅行为；一个正在运行却零 OBI 订阅的缓存正是该缺口的特征。OBI 也会记录自己用的是本地缓存还是订阅远端。
4. 用 `obi config migrate` 交叉核对：迁移 v1 文件并读报告。未被实例化进 v2 字段的 v1 环境覆盖不会被保留——"未被 v2 文档引用的变量没有任何迁移效果，除非目标版本明确把它文档化为独立运行时输入"。

## 局限

环境变量形式是 v1 覆盖项，被刻意排除在 v2 文档模型之外。有两个正确修复，选择由部署方式决定：

- 升级图。图修复（open-telemetry/opentelemetry-helm-charts 拉取请求 2440，"fix(obi): set the k8s cache address in the Config v2 document"）已合并进 0.14.2：当 `k8sCache.replicas` 设定时，渲染出的 v2 configmap 现在包含 Kubernetes enricher 块下指向缓存服务的 `metadata_cache.address` 字段。在带上该变更的图上，地址会被自动接好。
- 在那之前，把地址写进 Config v2 文档——经图的 `config.data`（Kubernetes enricher 块下的 `metadata_cache.address` 字段）——或让部署停留在已冻结但仍受支持的 v1 配置上，那里变量依然生效。

不要在 v2 部署里单独依赖导出 `OTEL_EBPF_KUBE_META_CACHE_ADDRESS`。除非文档引用它，否则变量是无效的；而让被引用占位符起作用的替换又取决于加载器。受支持、可复现的形式是 v2 文档里的 `metadata_cache.address` 字段——由修复后的图写死，或在 values 里显式设定。

## 参考

- [OBI — `devdocs/k8s-cache.md`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/k8s-cache.md) — k8s-cache 设计；`meta_cache_address` / 环境变量开关；"无地址即本地进程内缓存"；内部指标。
- [OBI — Config v2 迁移指南](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/config/version-2.0/migration.md) — "OTEL_EBPF_* 运行时覆盖不会被读作额外的 v1 配置"；"Rewire v1 environment overrides"；`obi config migrate` 行为。
- [OBI — Config v2 指南](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/config/version-2.0/config-v2.md) — v1 字段 `meta_cache_address` 迁移并改名为 Kubernetes enricher 块下的 `metadata_cache.address`（迁移加改名）；环境变量替换的加载器注意事项。
- [OBI Helm 图 — daemonset 模板](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/templates/daemonset.yaml) — 当 `k8sCache.replicas` 设定时注入 `OTEL_EBPF_KUBE_META_CACHE_ADDRESS`。
- [open-telemetry/opentelemetry-helm-charts 拉取请求 2440](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2440) — "fix(obi): set the k8s cache address in the Config v2 document"（2026-10-02 合并，图 0.14.2），把 `metadata_cache.address` 接入渲染出的 v2 configmap。

## 当日社区讨论

选取的问题来自一个 opt-in 存档：一位实践者注意到，在官方 OBI Helm 图里，开启 k8s 缓存时 `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` 会被自动注入 OBI daemonset pod，而 Config v2 文档又不接受环境变量作为配置，因此不确定该变量在 v2 下是否受支持。同一讨论的后续在真实图上证实了顾虑：只设 `k8sCache.replicas: 1` 的 values 会得到一份默认省略缓存地址的 v2 配置，OBI 会忽略导出的变量，缓存收不到连接；用户自带 v2 文档（经 config data）时同样如此。讨论中的临时 workaround 是把地址手工接进 v2 文档（Kubernetes enricher 块下的 `metadata_cache.address` 字段，指向缓存服务及其 gRPC 端口），并附注图应当像仍为 v1 那样为 v2 文档设置它。本篇对照公开 OBI 配置文档与图确认了该机制：v1 环境变量不是 v2 覆盖项，该字段在 v2 里已迁移改名，把地址接进 v2 文档的图修复（拉取请求 2440）才是持久解法。

当日其他讨论：一条 Cilium LoadBalancer 共享 VIP frontend 所有权讨论（两天前发布问题的再次提交，附维护者指向一个已跟踪的上游 issue 与一个审阅中的修复）、一条 GnuTLS HTTP/2 每连接 HPACK 解码器讨论（前一天发布的问题）、一条 Hubble 流到规则误归因讨论（作为已跟踪的上游 issue 被回复，数据面正确，只有 Go 侧副本有偏差）、一条高频 socket 层 drop 时延与用户态 context switch 的基准请求（太薄未发布）、一条 OpenTelemetry GenAI 语义约定讨论（记录智能体调用开始时哪些技能可用，属语义约定拉取请求，不在本页范围内）。

本次运行的渠道覆盖：两个 opt-in 存档共 11 条消息，全部如上覆盖。visible-browser-only 来源（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能审阅——没有可用的 visible-browser 会话——因此标记为未覆盖，而非平静。
