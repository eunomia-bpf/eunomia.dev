# 2026-10-18 内容巡检运行日志
运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-18。唯一提交时刻为 2026-10-19 07:23 +08（= 2026-10-18 16:23 PDT），落在 LA 2026-10-18。巡检开始时补发缺口为 0 条（2026-10-02 巡检已核销 2026-09-03 与 2026-09-16 两条缺口；10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 → 10-18 连续巡检，无新增补发缺口），struct_ops 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 struct_ops（队列第 129 行；第 128 行掘金 dynptr 已于 10-17 巡检完成）；其后下一可执行任务为第 130 行 xpu/flamegraph（LA 2026-10-19 正常额度）。

## 队列任务：掘金 struct_ops（LA 10-18 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 129 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 128 行 dynptr 已于 10-17 巡检完成；下一可执行掘金行第 130 行为 xpu/flamegraph（LA 2026-10-19 正常额度）。
- 源文：`docs/tutorials/features/struct_ops/README.zh.md`；发布稿 `draft/media/2026-10-18/struct_ops/juejin-body.md`（移除源 H1，其余正文逐字保留；20570 字符、31555 字节、714 行、8 个 H2 / 16 个 H3 / 0 个 H4、20 个代码块 [8 c、8 bash、4 无语言标签]、0 张图片、0 个表格、6 条唯一外链目标 [lpc.events LPC 2024 演讲、github.com/eunomia-bpf/gpu_ext、github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/features/struct_ops、github.com/eunomia-bpf/bpf-developer-tutorial、eunomia.dev/tutorials/、www.kernel.org/doc/html/latest/bpf/]）。
- 标题逐字取源 H1「eBPF 教程：使用 BPF struct_ops 扩展内核子系统」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（每个 chip 经 native-setter 搜索 + 真实 CDP 指针点击可见 `.byte-select-option`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr 教程标签组）。

## 提交

- 提交时刻 2026-10-19 07:23 +08 = 2026-10-18 16:23 PDT；提交前创作者中心基线 全部 87 / 已发布 87 / 审核中 0 / 未通过 0（dynptr 放行后基线）。单次弹窗「确定并发布」（完整 CDP 指针序列）返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7697622920013709347`，URL `https://juejin.cn/post/7697622920013709347`。提交后创作者中心 全部 88 / 已发布 87 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入本系列最长审核间隔：提交后暂存地址 `/spost/7697622920013709347`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；08:12 +08 前后轮询翻为 全部 88 / 已发布 88 / 审核中 0（审核间隔约 49 分钟，慢于 26-sudo / 20-tc / bpftrace / bpf_arena / bpf_iters / bpf_token / dynptr 的 1–23 分钟先例，为本系列最长审核间隔），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr；区别于 bpftrace / 26-sudo / 20-tc 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- struct_ops `/post/7697622920013709347`：标题逐字「eBPF 教程：使用 BPF struct_ops 扩展内核子系统」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（20570 字符 artifact；渲染 DOM innerText 读为 19740）；8 个 H2 / 16 个 H3 / 0 个 H4；20 个代码块 [8 c / 8 bash / 4 无语言标签，渲染 DOM 自动标/拆分为 21 个 `<pre>`]；0 张图片；0 个表格；6 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；0 评论 [QA 读取时] / 3 评论 [后期计数读取]（评论卡片懒渲染，QA 时刻静态选择器读 0，计数正则 `(\d+)\s*评论` 读为 "3 评论"）；早期计数 2 阅读 / 1 点赞 / 3 评论。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7697622920013709347` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 75 → 76，`last_checked` → 2026-10-18。
- `sources.json`：`last_checked` → 2026-10-18（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-18；dynptr 叙述段后补记 2026-10-18 struct_ops 叙述行（16:23 PDT 提交、08:12 +08 翻为已发布、约 49 分钟审核间隔、本系列最长先例）；Juejin confirmed 表格首行置顶 struct_ops（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：`Last checked:` → 2026-10-18；知乎未映射 40 → 41、掘金未映射 34 不变（分母经并发合并自 108 → 109；74/108 → 75/109 mapped），滚动队列 4 → 3 个 Juejin tasks（共 28 → 27 平台任务）；Juejin 状态行修复为完整 4 列行并前置 struct_ops 子句、机器台账计数刷新为 75/109 / 34 未映射 / 3 open rolling-queue tasks / next xpu/flamegraph；知乎状态行同步刷新为 68/109 mapped / 41 未映射。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-18；补发缺口段落前置 10-18 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、08:12 +08 翻为已发布、约 49 分钟、本系列最长审核间隔）；Ledger 基线按检查器更新（知乎 41、掘金 34，映射 75/109）；剩余队列 24 个知乎任务（23 `排队` + 1 `阻塞`）、3 个掘金任务，共 27 个平台任务；第 129 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-17` 块上新增 `### 2026-10-18` 块（struct_ops 发布 + QA + 放行后 top-ten 复核 + 前向指针指向 10-19 正常额度掘金 xpu/flamegraph）；未在 10-17 块内再织入。
- 发布稿 `draft/media/2026-10-18/struct_ops/juejin.md`：状态改「已确认公开」（post `7697622920013709347`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要，并保留正文/标题/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（分母经并发 Gthulhu 合并 + 日报自 108 → 109，知乎未映射 40 → 41；Juejin 75/109 映射、34 未发布、76 confirmed [2 条无 `source_path`]；Zhihu 41 未发布；exit 0）。三个 CRLF 台账经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 沿用 10-17 经验：正文 base64 分块写入 `window.__b64` 后按 UTF-8 解码注入 CodeMirror（直接 `atob` 为乱码）；分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须走 raw CDP `Input.dispatchMouseEvent`（页内合成事件不生效）；中文走 `agent-browser` `eval --stdin < /tmp/*.js` 文件；计数读取沿用 `.essays-container` 按 `2026-MM-DD HH:MM` 日期行切分配对法。
- 渲染 DOM 自动标/拆分：源 20 个代码块 [8 c / 8 bash / 4 无标签] 在 Juejin 渲染 DOM 中为 21 个 `<pre>`；记录以源计数 20 为准。

## 跟踪 top ten 复核（08:12 +08 创作者中心放行后逐行读取）

- 新文 struct_ops 首读 1 展现 / 2 阅读。
- 流量累积（对比 10-17 放行后 07:47 +08 复核）：[dynptr](https://juejin.cn/post/7697186290753159202) 0 / 1 → 1792 / 46；[bpf_wq](https://juejin.cn/post/7697094073967116323) 3 / 8 → 5 / 14；[bpf_token](https://juejin.cn/post/7696759034846445583) 804 / 20 → 845 / 24；[bpf_iters](https://juejin.cn/post/7696384745624141824) 694 / 16 → 728 / 19；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2322 / 53 → 2397 / 57；[cgroup](https://juejin.cn/post/7695667191396040738) 8 / 12 → 8 / 14；[bpftrace](https://juejin.cn/post/7695068144415244323) 4 / 9 → 4 / 10；[1-helloworld](https://juejin.cn/post/7694934986782064649) 1457 / 40 → 1474 / 43；[17-biopattern](https://juejin.cn/post/7694534084639195174) 1262 / 26 → 1277 / 28；19-lsm-connect 移出前十（其 10-17 读数为 8 / 17）。
- 评论活动全部集中于新文 struct_ops（后期计数 3 评论）；被跟踪旧帖 0 新增评论/回复。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。

## 后续

- 下一条可执行任务为队列第 130 行掘金 xpu/flamegraph（LA 2026-10-19 正常额度）；本次不越范围执行。

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
