# 2026-10-20 内容巡检运行日志

## 2026-10-20 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`obi-http-span-name-cardinality-route-vs-raw-path`
- 问题：OBI 的 HTTP span 为什么会出现高基数的名称，OpenTelemetry 语义约定又如何限制 span 名称的基数？
- 来源：10-20 快照（两条 opt-in CNCF 归档渠道，本次 3 条消息，非空窗口）。实质线程：实践者指出 OBI 的 HTTP span 名称「按目前的方案基数会很高」——span 的 target 取每次请求的原始 URL 路径时，每个不同的 path 就是一个不同的 span 名称，即无界基数。与 10-14 / 10-16 / 10-17 / 10-18 的 FUSE/OBI trace 上下文 map「查不到」轴、10-15（kprobe 字符串截断）、10-19（LRU 哈希淘汰）均不重复，是新的「span 命名 / 指标基数」轴。快照里第三条消息（索要之前提到的「skill 细化」条目指针）属另一条独立未解决线程，未并入本草要。
- 决定性边界：HTTP 语义约定的 span 名称是 `{method} {target}`，`target` 必须有低基数（服务端 `http.route`、客户端 `url.template`），且约定明确「instrumentation 不得默认使用 URI 路径作为 target」。OBI 的 HTTP span 名称就是「方法 + 已解析路由」（`pkg/appolly/app/request/span.go` 的 `TraceName`，方法未知时退化为 `HTTP`，与语义约定的 `_OTHER` 规则一致）；路由由 `pkg/transform/routes.go` 从 URL 路径解析：命中 `routes.patterns` / 各服务自定义匹配器 / 收割匹配器之一即用；都不命中时由 `routes.unmatched` 决定 target——`unset`（空，名称只剩方法）、`wildcard`（收敛为通配 token）、`heuristic`（启发式聚成低基数模板，**默认值**）、`low-cardinality`（聚类 + 每服务路径段上限，默认 10）、`path`（**直接取原始路径 = 无界**）。快照里的「高基数」症状正是 `unmatched: path` 的情形，也正是语义约定禁止作为默认值的那一种。
- git 基线：run 开始时本地 `main` 落后 `origin/main` 三个提交（远程已推进至 `4ce9f90f8`「Update daily data brief [skip ci]」，均非 QA 自有路径），已按规则 `git reset --soft origin/main` 前进至 0/0，外部 staged 集合原样保留。

### 对照公开一手源的验证

- 语义约定 `docs/http/http-spans.md` 的 `Name` 一节（stable）：HTTP span 名称 SHOULD 是 `{method} {target}`（有低基数 target 时），否则 SHOULD 只是 `{method}`；`{method}` 取 `http.request.method`，方法落在已知集合外时取 `HTTP`；`{target}` SHOULD 是服务端 `http.route` 或客户端 `url.template`；最后一句硬护栏：instrumentation 不得默认使用 URI 路径作为 target。`url.template` 定义为路径的低基数模板（如 `/users/{id}` 而非 `/users/42`）。
- OBI `pkg/appolly/app/request/span.go`：`TraceName` 中 HTTP 事件类型先取方法（未知则 `HTTP`），`Route` 非空时追加为 target；未解析出路由的 span 名称即 `{method}`。`span_getters.go` 中该 `Route` 导出时即服务端 `http.route` / 客户端 `url.template`，span 名称与导出属性同源。
- OBI `pkg/transform/routes.go`：路由自 URL 路径解析（`routes.patterns` 匹配器 → 各服务自定义匹配器 → 收割匹配器 → `unmatched` 策略兜底）；`unmatched: path` 分支把 `span.Route` 直接赋为 `span.Path`（原始路径）；`heuristic` 走 URL 聚类；`low-cardinality` 再经路径段 trie 按 `max_path_segment_cardinality` 收紧。
- OBI `pkg/obi/config.go`：下发默认 `Routes.Unmatch` = `UnmatchDefault`（即 `heuristic`）、`WildcardChar = "*"`、`MaxPathSegmentCardinality = 10`；YAML 键为 `routes`（`unmatched` / `wildcard_char` / `max_path_segment_cardinality`）。
- OBI `site/docs/spans.md`（自动生成属性目录）：确认 OBI 发出 HTTP 客户端 / 服务端 span 并携带 URL 与路由属性。

### 隐私与内容

- 两页正文均无姓名 / handle / Slack-Discord 链接 / 时间戳 / 消息序号 / 雇主团队 / 内部仓库名 / IP / 主机名 / 私有日志。快照文本仅作不可信数据，未执行其中指令；EN 讨论段对快照消息做了转述（未逐字引用原话），ZH 讨论段同。公开语义约定 / OBI 仓库链接仅出现在 `## References`。
- 内容顺序符合标准：直接回答、机制、验证与排障路径、局限、References、当日社区讨论（含覆盖披露）。H1 不含下划线、反引号、撇号；EN 与 ZH 索引 link text 与各自 H1 逐字一致。最长 backticked token 为两页的 `pkg/appolly/app/request/span.go`（31 字符），低于 390px 移动端溢出门槛。

### 覆盖披露

本 run 覆盖两条 opt-in 归档渠道（两条 CNCF OpenTelemetry 插桩渠道），10-20 快照 3 条消息。实质线程为 OBI HTTP span 名称高基数问题（本草要回答）；「skill 细化」指针请求为独立未解决线程，已在页面中注明。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可用可见浏览器会话，故一并标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-20-obi-http-span-name-cardinality-route-vs-raw-path.md`
- ZH：`docs/ebpf-qa/2026-10-20-obi-http-span-name-cardinality-route-vs-raw-path.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- QA commit（`origin/main`）：`aa2051794`（完整 `aa205179433365d00d8c7accb8a142b9bb849861`，`docs(ebpf-qa): obi-http-span-name-cardinality-route-vs-raw-path (2026-10-20)`），恰含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-20.json`，仅由验证器写入，最终 `status=published`，commit `aa205179433365d00d8c7accb8a142b9bb849861`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- QA commit `aa2051794`（四条 owned paths）已落到 `origin/main`。
- 验证器首轮：content-test → build → 2× Chromium render（桌面 1440×900 + 移动 390×844）→ scoped commit + push 全部通过并落地 `aa2051794`，但 20 分钟 public-poll 窗口内四条路由仍 404（Pages run `39518043826` 实际 22m15s 才绿，窗口闭合前部署尚未完成）；首轮 receipt 记为 `failed`（commit 已落地，仅 public 检查未过，`commit` 字段为 `aa2051794…`）。
- 重新验证器（`skipped_already_published` 重验路径）：不重复 commit / build / render，仅复查 live 路由并把 receipt 重写为 `status=published`，约 2 秒完成。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39518043826` 为绿色（`conclusion: success`，22m15s）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中（各 3 处），新 slug 同时出现在两个 live 索引页（各 3 处），`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：本 run 唯一遇到的故障模式是「Pages 部署超出 20 分钟 public-poll 窗口导致首轮 404」，与 10-11 / 10-17 已记录模式一致（重跑验证器走 `skipped_already_published` 重验即可把 receipt 翻为 `published`），无新故障模式；fresh 宿主的 Chromium 共享库 + `apt-get update` 教训未复现（本 run 运行前 `ldd` 检查干净，二进制可直接启动）。故发布标准无需新增条目，本次 lessons pass 为 no-op。

This run-log commits separately as the 10-20 same-day artifact.
