# 2026-10-21 内容巡检运行日志

## 2026-10-21 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`obi-metric-label-set-ttl-lazy-vs-scrape-driven-expiration`
- 问题：OpenTelemetry OBI 指标为什么会在 TTL 之后还在 store 里保留最长达两倍时长，Prometheus 导出器会不会在每次 scrape 时都删掉同一组 label？（EN H1：Why can an OpenTelemetry OBI metric stay in the store up to twice as long as its TTL, and does the Prometheus exporter drop the same label set on every scrape?）
- 来源：10-21 快照（两条 opt-in CNCF 归档渠道，本次 0 条消息，空窗口）→ 公开一手源回退路径。问题取自 OBI 用户态指标过期器「OTEL 导出器 2×TTL 懒过期 vs Prometheus 导出器 scrape 驱动逐次删除」这条未充分文档化的行为边界，与 10-20（HTTP span 命名/基数）、10-19（LRU 哈希淘汰）、10-16/10-17/10-18（trace 上下文 map）均不重复，是新的「用户态指标 store 生命周期」轴。空窗口与回退来源已在两页正文中如实披露。
- git 基线：run 开始时本地 `main` 在 `c15835311`（10-20 run-log）。验证器首轮 commit（`6111de92d`）后 push 被拒（远程推进 `c15835311..cff454507`，共 3 个提交：`cff454507` daily data brief、`f468e9ec2` media run-log、`71c904d10` juejin 发布，均非 QA 自有路径），已按规则 `git reset --soft origin/main` 重提 4 条 owned paths 并成功 push。

### 对照公开一手源的验证

- OBI `pkg/export/expire/expiry_map.go`：`ExpiryMap` 以 label 集合键 → 条目（值 + `lastAccess`）建表；`GetOrCreate` 命中即刷新 `lastAccess`；`DeleteExpired` 以 `now.Sub(e.lastAccess) > ex.ttl` 为截止；`ttl == 0` 关闭过期；`LabelsKey` 以「长度前缀」编码避免碰撞。
- OBI `pkg/export/otel/expirer.go`：`ForRecord` 返回前检查 `now.Sub(ex.lastExpiration) >= ex.ttl`，满足才 `removeOutdated` 清扫并对过期集合调用 `metric.Remove`；代码注释明确「an expired metric might stay visible after 2*TTL time after not being updated」；`RemoveAllMetrics` 为显式关闭钩子。调用点在 `pkg/export/otel/metrics.go`（RED 时延/大小）与 `pkg/export/otel/metrics_svc_graph.go`（service-graph 汇总）。
- OBI `pkg/export/prom/expirer.go`：`Collect` 每次先 `DeleteExpired()` 再以 `DeleteLabelValues` 删掉过期 label 集合 → 最大滞留 = 一个 TTL + 一个 scrape 间隔；`DeleteLabelValues` 亦供 `prom_nodejs.go` 等按进程 runtime 指标封装在进程消失时手动删除。
- OBI `pkg/obi/config.go`：`defaultMetricsTTL = 5 * time.Minute`，分别映射到 `OTELMetrics.TTL` 与 `Prometheus.TTL`；env 键为 `OTEL_EBPF_METRICS_TTL` / `OTEL_EBPF_PROMETHEUS_TTL`。

### 隐私与内容

- 空窗口，无快照消息，故两页正文不涉及任何私人讨论；「Community discussion today」段如实披露快照窗口为空（2 条 opt-in 渠道共 0 条消息）且可见浏览器专属渠道本次无可用会话、一并标记「未覆盖」。无姓名 / handle / 链接 / 时间戳 / 内部仓库名。公开 OBI 仓库链接仅出现在 `## References`。
- 内容顺序符合标准：直接回答、机制、验证与排障路径、局限、References、当日社区讨论（含覆盖披露）。H1 不含下划线、反引号、撇号；EN 与 ZH 索引 link text 与各自 H1 逐字一致。最长 backticked token 为 `pkg/export/otel/metrics_svc_graph.go`（36 字符）——成稿前将原 50 字符的 `metric.Remove(ctx, …)` 调用改写为短反引号 + 散文，低于 390px 移动端溢出门槛。

### 覆盖披露

本 run 覆盖两条 opt-in 归档渠道（两条 CNCF OpenTelemetry 插桩渠道），10-21 快照 0 条消息（空窗口）。问题改由公开一手源（OBI 用户态指标过期器源码）供给，已在页面中注明。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可用可见浏览器会话，故一并标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-21-obi-metric-label-set-ttl-lazy-vs-scrape-driven-expiration.md`
- ZH：`docs/ebpf-qa/2026-10-21-obi-metric-label-set-ttl-lazy-vs-scrape-driven-expiration.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- QA commit（`origin/main`）：`3ee52851f`（完整 `3ee52851f4050bf574ad90bb31fe86dd92709e2f`，`docs(ebpf-qa): obi-metric-label-set-ttl-lazy-vs-scrape-driven-expiration (2026-10-21)`），恰含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-21.json`，仅由验证器写入，最终 `status=published`，commit `3ee52851f4050bf574ad90bb31fe86dd92709e2f`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- 验证器首轮：content-test → build → 2× Chromium render（桌面 1440×900 + 移动 390×844）→ scoped commit 全部通过，本地 commit `6111de92d` 已落地；push 被拒（远程 3 提交推进，均非 QA 路径），首轮 receipt 记为 `failed`（commit 已落地，仅 push 未过）。
- 恢复：`git reset --soft origin/main`（外部 staged 集合原样保留）→ 重提 4 条 owned paths 为 `3ee52851f` → push 成功 `cff454507..3ee52851f`。
- 验证器二轮（`skipped_already_published` 重验路径）：不重复 commit / build / render，`remote_contains_commit` 通过；但 20 分钟 public-poll 窗口内 Pages 部署（run `39695465820`，实际 25m23s 才绿）尚未完成，public 检查 404 后窗口闭合，receipt 保持 `failed`。
- 验证器三轮（同路径，Pages 部署绿后）：约 2 秒完成全部复查，receipt 重写为 `status=published`。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39695465820` 为绿色（`conclusion: success`，25m23s）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中（各 3 处），新 slug 同时出现在两个 live 索引页（各 3 处），`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：本 run 遇到的两个故障模式均已见于发布标准（远程推进导致 push 被拒 → `reset --soft` 前移后重提，标准第 132–136 行；Pages 部署超出 20 分钟 public-poll 窗口导致 404 → 重跑验证器走 `skipped_already_published` 重验，10-11 / 10-17 已记录模式），无新故障模式；fresh 宿主的 Chromium 共享库教训未复现（本 run 运行前 `ldd` 检查干净，二进制可直接启动）。故发布标准无需新增条目，本次 lessons pass 为 no-op。

This run-log commits separately as the 10-21 same-day artifact.
