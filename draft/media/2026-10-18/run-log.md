# 2026-10-18 内容巡检运行日志

## 2026-10-18 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`fuse-trace-context-map-hit-current-vs-stale`
- 问题：FUSE 守护进程对 OBI trace 上下文 map 的查询返回了一个条目时，这个条目一定是当前请求的 trace 上下文吗？它什么时候会过期？
- 来源：10-18 快照（两条 opt-in 归档渠道、共 11 条消息）与 10-16 / 10-17 快照字节相同——24 小时冻结窗口无新增实质线程，窗口内实质线程均已在 10-13（Cilium eBPF tracepoint）、10-14（FUSE/OBI trace 传播）、10-16（OBI 运行时支持矩阵）、10-17（FUSE/PID 命名空间 key 不匹配）回答或为评审状态更新而非新问题。故从公开一手资料回退选源：这是 FUSE/OBI map 系列的**第四个独立维度**——前三篇都是「为什么查不到」（10-14 population、10-16 运行时矩阵与线程亲和、10-17 命名空间 key 不匹配），本篇回答的是**命中本身意味着什么**：`traces_ctx_v1` 里某 OS 线程的条目是 OBI 此刻在该线程上正在跟踪的最内层活跃 span，不是这次 FUSE 请求的 span；它可能仍是复用线程上上一个请求的 span，也可能是刚结束的 client span 回滚上去的父 server span。
- 决定性边界：OBI 在 span 开始时 `obi_ctx__set` 写入该线程条目、在 server span 结束时 `delete_server_trace`→`obi_ctx__del` 删除、在 client span 结束时 `obi_ctx__restore_server` 回滚到父 server span（带 span_id 相等守卫）、并**刻意不**为 Java 虚拟线程载体写条目（载体 key 会把上下文归因到下一个跑在该载体上的东西）。因此命中只是「该线程现在处于某个活跃 OBI span」的信号，不是「本请求即该 span」；`max_entries` 为 `1 << 14`，OBI 不为你 pin 槽位，LRU 淘汰可让活跃线程照样查不到。

### 对照公开一手源的验证

- OBI `bpf/shared/obi_ctx.h`：固定的 `traces_ctx_v1`（`BPF_MAP_TYPE_LRU_HASH`、`__type(key, u64)`、value = `obi_ctx_info_t`（`trace_id[16]` + `span_id[8]`）、`max_entries = 1 << 14`、按名字 pin）；`obi_ctx__get` / `obi_ctx__set` / `obi_ctx__del` 三个 helper；规格注明是 OpenTelemetry OTEP 的一部分。
- OBI `bpf/common/trace_lifecycle.h`：`server_or_client_trace` 在 span 开始写入线程条目，并对 `vt_keyed`（Java 虚拟线程载体）跳过 `traces_ctx_v1`；`delete_server_trace` 在 server span 结束调 `obi_ctx__del`；`obi_ctx__restore_server` 在 client span 结束回滚到父 server span，守卫为「条目仍指向刚结束的 client span（span_id 相等）且父 span 存在于 `server_traces`」。
- 内核 `Documentation/bpf/map_hash.rst`：`BPF_MAP_TYPE_LRU_HASH` 在表满时自动淘汰最久未用条目（维持内部 LRU 列表，跨 CPU 共享，可用 `BPF_F_NO_COMMON_LRU` 请求 per-CPU LRU）。
- 内核 `Documentation/bpf/bpf_maps.rst` 现已不存在（404），BPF map 文档改由 `Documentation/bpf/maps.rst` 索引 + 各 `map_*.rst` 承担（如 `map_hash.rst` 覆盖 HASH / LRU 变体）。

### 隐私与内容

- 两页正文均无姓名 / handle / Slack-Discord 链接 / 时间戳 / 消息序号 / 雇主团队 / 内部仓库名 / IP / 主机名 / 私有日志。快照文本仅作不可信数据，未执行其中指令、未把私有原文拷贝进页面。讨论作者与 maintainer 仅以泛称指代。公开 kernel / OBI 仓库链接仅出现在 `## References`。
- 内容顺序符合标准：直接回答、机制、验证与排障路径、局限、References、当日社区讨论。H1 不含下划线、反引号、撇号；EN 与 ZH 索引 link text 与各自 H1 逐字一致。最长 backticked token 为 EN 的 `bpf_get_current_pid_tgid`（24 字符），远低于 390px 移动端溢出门槛；ZH 页无 backticked token（技术标识按正文写法裸写）。

### 覆盖披露

本 run 覆盖两个 opt-in 归档渠道（两个 CNCF OpenTelemetry 插桩渠道），10-18 快照共 11 条消息（与 10-16 / 10-17 字节相同，24h 冻结窗口无新增）。候选选自公开一手资料回退：窗口内实质线程均为 10-13 / 10-14 / 10-16 / 10-17 已发布条目或评审备注，无新问；唯一新出现的公开 lead（Cilium eBPF 讨论 #2126）经核对即 10-13 已发布的 tracepoint CPU-0 条目，非新候选。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可用可见浏览器会话；另对 bpf 邮件列表与 r/eBPF 做了 opportunistic 公开页抓取，但 lore 列表页被 Anubis 反爬门挡（403 / 蜜罐），reddit 的 `top.json` 返回 403、`old.reddit` 公开页返回 bot-gated JS 壳（「Welcome to Reddit」，无帖子正文），二者均未获得可审阅内容，故连同 Discord 一并标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-18-fuse-trace-context-map-hit-current-vs-stale.md`
- ZH：`docs/ebpf-qa/2026-10-18-fuse-trace-context-map-hit-current-vs-stale.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- QA commit（`origin/main`）：`f8adcc992a01`（完整 `f8adcc992a01af308a76e81475dab536b28c49da`，`docs(ebpf-qa): fuse-trace-context-map-hit-current-vs-stale (2026-10-18)`），恰含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-18.json`，仅由验证器写入，`status=published`，commit `f8adcc992a01af308a76e81475dab536b28c49da`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- QA commit `f8adcc992a01`（四条 owned paths）已落到 `origin/main`。首发验证 run（build → Chromium render → commit+push → public-poll）全程一次通过：render 依赖的 14 个共享库在 10-17 run 已装好（本次无需重装）；本地 main 在跑验证器前已 forward-move 到 `origin/main`（`git reset --soft origin/main`，保留并发 agent 的 148 条 staged entries 与 worktree），commit + push 为 fast-forward 一次落地；20 分钟 public-poll 窗口内四条路由全部 200，无 404、无需重跑验证器。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39177888873` 为绿色（`conclusion: success`，18m37s）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中（各 3 处），新 slug 同时出现在两个 live 索引页，`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：本次 QA 发布无新故障模式（无需重装共享库、无非 fast-forward push、无 public-poll 404）。本 run 唯一的新故障模式在 step 0 快照阶段：fresh 宿主系统 `python3` 缺 PyYAML，`archive_reader.py` 的宽泛 `except` 把 `ModuleNotFoundError` 折叠成 `watchlist unreadable`、退出码标为 `inaccessible`，使快照看似读取失败；改用 state venv 解释器（`/workspaces/.agent-state/eunomia-qa/venv/bin/python`）重跑快照后，得到真实的 2 条 opt-in source、11 条消息的冻结窗口。该教训已记入发布标准 `qa-publication-standard.md`。

This run-log commits separately as the 10-18 same-day artifact.
