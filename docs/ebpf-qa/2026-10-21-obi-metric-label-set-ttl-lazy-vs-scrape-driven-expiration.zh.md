# OpenTelemetry OBI 指标为什么会在 TTL 之后还在 store 里保留最长达两倍时长，Prometheus 导出器会不会在每次 scrape 时都删掉同一组 label？

答案取决于用的是哪个导出器。OBI 在 OpenTelemetry 导出器和 Prometheus 导出器里都用同一个 TTL 来限制每个 label 集合的指标基数，但两者删除的时机不同。OTEL 导出器是**惰性**删除：只有当记录路径再次触到导出器时才会触发一次清扫，而且这次清扫本身被限制为「最多每 TTL 一次」，所以一个死掉的 label 集合在最后一次更新后最多可以再存活**两个 TTL**。Prometheus 导出器则是**每次 scrape 都删**：它的 collector 在 `Collect` 里直接调用过期清扫，所以一个死掉的 label 集合会在 TTL 之后落地的第一次 scrape 上消失。两者默认都是同一个 5 分钟 TTL。

一条指标序列活得比产生它的 label 集合还久，是基数泄漏，不是正确性 bug：值本身没问题，问题只是这条死序列多占了多少时间的内存和 scrape 负载。

## 机制

1. **同一个过期存储，按 label 集合建键。** `pkg/export/expire/expiry_map.go` 定义了 `ExpiryMap`，一个带同步锁的从 label 集合 key 到条目的 map，条目里存值和一个 `lastAccess` 时间。热路径上唯一的写入口是 `GetOrCreate`：未命中就插入新条目，命中就刷新 `lastAccess`。`DeleteExpired` 扫一遍 map，把所有 `lastAccess` 严格早于 `now - ttl` 的条目（判定式是 `now.Sub(e.lastAccess) > ex.ttl`，也就是安静满一个 TTL 之后立刻到期）连同它们的 label 集合一起返回；而 `ttl == 0` 直接关闭过期功能。key 由 `LabelsKey` 生成，把每个 label 值编码成「长度 + 值」，因此 `["a", "bc"]` 和 `["ab", "c"]` 不会互相碰撞。

2. **OTEL 过期器由记录驱动，且被限制为每 TTL 最多一次。** `pkg/export/otel/expirer.go` 在 `ExpiryMap[attribute.Set]` 外面包了一层，持有一个 `lastExpiration` 时钟。删除发生在 `ForRecord` 里：在返回数据点之前，它先检查 `now.Sub(ex.lastExpiration) >= ex.ttl`，只有满足条件才调用 `removeOutdated`，扫出每个过期的 attribute 集合，并对每个调用 `metric.Remove` 以该集合作为删除选项，真正从 SDK 指标上摘掉这个数据点。由于清扫被限制为「距上次清扫不足一个 TTL 就跳过」，一个 TTL 前就过期的条目只可能被下一次清扫发现，而那次清扫最早也要再过 TTL 才发生——所以总寿命最长可达 `2*TTL`，这正是代码注释里写的那句话：「一个过期指标在最后一次更新后可能继续可见 2*TTL」。清扫不是后台定时器：它跑在那些调用 `ForRecord` 的记录路径上，也就是 `pkg/export/otel/metrics.go` 里的 RED 时延与大小指标、以及 `pkg/export/otel/metrics_svc_graph.go` 里的 service-graph 汇总。`RemoveAllMetrics` 是显式的关闭钩子，会立即删掉所有条目。

3. **Prometheus 过期器由 scrape 驱动。** `pkg/export/prom/expirer.go` 用同一个 `ExpiryMap` 包住一个 `prometheus.MetricVec`，并以 `Collector` 形式暴露。它的 `Collect` 先调用 `ex.entries.DeleteExpired()`，再用 `DeleteLabelValues` 把返回的每个 label 集合从包住的向量里删掉，最后才把剩下的输出。这里没有 `lastExpiration` 门控：每次 scrape 都是一次清扫，所以最大滞留时间是一个 TTL 加一个 scrape 间隔。过期器上的 `DeleteLabelValues` 提供对单个 label 集合的即时手动删除，`prom_nodejs.go` 这类按进程的 runtime 指标封装在 Node.js 进程消失时就是用它的。

4. **TTL 的取值与来源。** `pkg/obi/config.go` 里 `defaultMetricsTTL = 5 * time.Minute`，同时赋给 `OTELMetrics.TTL` 和 `Prometheus.TTL`。OTEL 这个旋钮是 OTEL metrics 配置上的 `TTL` 字段，对应 YAML 里的 `ttl`、环境变量 `OTEL_EBPF_METRICS_TTL`；Prometheus 的旋钮是同名的 YAML `ttl`、环境变量 `OTEL_EBPF_PROMETHEUS_TTL`。一个 TTL 值，两套删除节奏。

## 验证与排障路径

1. **先确认配的是哪个导出器，再读生效的 TTL。** OTEL 导出器读 `OTEL_EBPF_METRICS_TTL`（OTEL metrics 块下的 YAML `ttl`）；Prometheus 导出器读 `OTEL_EBPF_PROMETHEUS_TTL`（Prometheus 块下的 YAML `ttl`）。哪边没设都会回落到 5 分钟默认，所以一条「看起来陈旧」的序列如果比 5 分钟还年轻，那就还不是配置问题。

2. **直接在 Prometheus 端点上观察差异。** 对一组不再产生流量的 label 集合（一条冷掉的路由或一个停掉的服务），Prometheus 的 `/metrics` 输出会在安静满一个 TTL 后的第一次 scrape 上丢掉它，下一次 scrape 已经干净了。而在 OTEL collector 上，同一条序列可能再多撑两次 scrape 才消失，因为触发删除的那次清扫只有在「某条」label 集合的记录再次路过导出器时才会发生。如果 OTEL 指标看起来总在「边界附近才过期」而 Prometheus 端点看起来收得很紧，那就是这个预期形状，不是 bug。

3. **看 debug 日志。** 两个过期器都有 debug 日志：OTEL 过期器在删时会打 `deleting old OTEL metric` 并带 attribute 集合，Prometheus 过期器打 `deleting old Prometheus metric` 并带 label 值，`GetOrCreate` 打 `storing new metric label set`。把其中一侧导出器的日志级别调到 debug，清扫边界就能直接在时间线上读出来。

4. **约束最坏情况，而不是干等。** 当基数很高（大量短命路由或进程）时，OTEL 的「两个 TTL 尾巴」加上 scrape 间隔就是死序列的实际留存时间。缩短 TTL 是两侧唯一能由操作者控制的旋钮；那个门控本身不可配置。

## 限制

- OTEL 过期器的清扫门控挂在记录路径上，而不是挂在墙钟定时器上：当导出器安静时，清扫就不会跑，所以「最多两个 TTL」是上界，不是保证。一条完全空闲的 OTEL 管道会一直保留它最后一批 label 集合，因为没有任何记录会调用 `ForRecord` 来触发清扫。
- 两个导出器对同一个 TTL 值没有统一的语义：OTEL 的边界是安静 `2*TTL`，Prometheus 的边界是 `TTL` 加 scrape 间隔。只拿一个数去理解两边的操作者会误判一个死掉的服务留下的是哪批序列。
- `ttl: 0` 会彻底关闭被设为 0 的那一侧的过期功能，把基数泄漏换成 label 集合的无界增长；这是显式的可选项，不是默认。
- `RemoveAllMetrics` 只存在于 OTEL 过期器上，是单个服务实例 reporter 的关闭路径；它不在普通安静期触发，不能替代 TTL。
- 本回答描述的是 OBI `main` 分支行为。`ExpiryMap` 的形状、每 TTL 一次的门控、以及 scrape 驱动的 `Collect` 清扫，都是 OBI 的实现细节；「label 集合应在一段安静期后过期」这个语义是通用的基数管理，任何按 label 的指标存储都会面对。

## 参考资料

- [OBI `pkg/export/expire/expiry_map.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/expire/expiry_map.go) - `ExpiryMap` 存储：`GetOrCreate` 在命中时刷新最后访问时间，`DeleteExpired` 在 `now.Sub(lastAccess) > ttl` 时切割，`LabelsKey` 使 label 集合 key 无歧义。
- [OBI `pkg/export/otel/expirer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/otel/expirer.go) - `ForRecord` 的「每 TTL 一次」清扫门控及其「最多 2*TTL」注释、`removeOutdated` 调用 `metric.Remove`、以及用于 reporter 关闭的 `RemoveAllMetrics`。
- [OBI `pkg/export/prom/expirer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/prom/expirer.go) - Prometheus 过期器：`Collect` 在每次 scrape 调用 `DeleteExpired`，并把返回的 label 集合从包住的 `MetricVec` 中删掉。
- [OBI `pkg/obi/config.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/obi/config.go) - `defaultMetricsTTL = 5 * time.Minute` 同时赋给 `OTELMetrics.TTL` 与 `Prometheus.TTL`。
- [OBI `pkg/export/prom/prom.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/prom/prom.go) - Prometheus 配置的 `TTL` 字段，对应 YAML `ttl`、环境变量 `OTEL_EBPF_PROMETHEUS_TTL`。

## 今日社区讨论

今日没有可供总结的社区讨论：本次运行所覆盖的监控窗口一条消息都没有返回——两个 opt-in 归档在 24 小时窗口内返回了 0 条消息（`result=ok`，零条）——所以本条回答的是上文公开一手资料里一个长存的问题，而不是某条在线讨论，因此不附匿名的讨论摘要。

本次运行的频道覆盖：两个 opt-in 归档频道在监控窗口内均返回零条消息。可见浏览器专属来源——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本次因没有可用的可见浏览器会话而无法审阅，故标记为「未覆盖」而非「安静」。
