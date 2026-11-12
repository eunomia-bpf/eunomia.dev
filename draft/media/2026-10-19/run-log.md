# 2026-10-19 内容巡检运行日志
运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-19。唯一提交时刻为 2026-10-20 08:44 +08（= 2026-10-19 17:44 PDT），落在 LA 2026-10-19。巡检开始时补发缺口为 0 条（2026-10-02 巡检已核销 2026-09-03 与 2026-09-16 两条缺口；10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 → 10-18 → 10-19 连续巡检，无新增补发缺口），xpu/flamegraph 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 xpu/flamegraph（队列第 130 行）。

## 队列任务：掘金 xpu/flamegraph（LA 10-19 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 130 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 129 行 struct_ops 已于 10-18 巡检完成；下一可执行掘金行第 131 行为 xpu/gpu-kernel-driver（LA 2026-10-20 正常额度）。
- 源文：`docs/tutorials/xpu/flamegraph/README.zh.md`；发布稿 `draft/media/2026-10-19/xpu-flamegraph/juejin-body.md`（移除源 H1，源第 11 行图片引用 `qwen3_flamegraph.svg` 刻意省略 [该 `.svg` 为教程 `.gitignore` 的 `*.svg` 规则忽略的构建产物：仓库内无此文件、上游 bpf-developer-tutorial 无、eunomia.dev 线上 404，无法转存或上传，故从发布稿省略该图片行，其余正文逐字保留]；31718 字符、57736 字节、723 行、11 个 H2 / 17 个 H3 / 5 个 H4、25 个代码块 [1 c、1 cpp、2 python、18 bash、3 无语言标签]、0 张图片、0 个表格、11 条唯一外链目标 [docs.nvidia.com/nsight-systems、github.com/eunomia-bpf/bpftime/tree/master/example/gpu、github.com/eunomia-bpf/xpu-perf、github.com/intel/iaprof、perfetto.dev、pytorch.org 博客与文档、rocm.docs.amd.com rocprofiler-sdk、alcf.anl.gov HPCToolkit 2024 幻灯片、brendangregg.com AI 火焰图博文、intel.com Level Zero tracer]，0 条 eunomia.dev 链接）。
- 标题逐字取源 H1「eBPF 示例：使用 CUPTI 构建 GPU 火焰图分析器」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（每个 chip 经 native-setter 搜索 + 真实 CDP 指针点击可见 `.byte-select-option`，沿用 17-biopattern / 1-helloworld / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops 教程标签组）。

## 提交

- 提交时刻 2026-10-20 08:44 +08 = 2026-10-19 17:44 PDT；提交前创作者中心基线 全部 88 / 已发布 88 / 审核中 0 / 未通过 0（struct_ops 放行后基线）。单次弹窗「确定并发布」（完整 CDP 指针序列，`scrollIntoView({block:"center"})` 后真实 CDP 指针点击）返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7698287001586040872`，URL `https://juejin.cn/post/7698287001586040872`（编辑器过渡 URL 上的 draft id `7698287148908511266` 非发布 post id；实际发布 post id 取自创作者中心审核中列表行）。提交后创作者中心 全部 89 / 已发布 88 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7698287001586040872`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；09:01 +08 前后轮询翻为 全部 89 / 已发布 89 / 审核中 0（审核间隔约 17 分钟，快于 dynptr 的 23 分钟与 struct_ops 的 49 分钟先例，慢于 bpf_wq / bpf_arena / bpf_iters / bpf_token 的 1–5 分钟快档），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops；区别于 bpftrace / 26-sudo / 20-tc / 31-goroutine 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- xpu/flamegraph `/post/7698287001586040872`：标题逐字「eBPF 示例：使用 CUPTI 构建 GPU 火焰图分析器」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（31718 字符 artifact；渲染 DOM innerText 读为 31260，代码块空白归一化）；11 个 H2 / 17 个 H3 / 5 个 H4；25 个代码块 [1 c、1 cpp、2 python、18 bash、3 无语言标签，渲染 DOM 25 个 `<pre>`，与源 25 个围栏代码块 1:1]；0 张图片 [源第 11 行 `qwen3_flamegraph.svg` 图片引用为 gitignored 构建产物、线上 404，刻意省略]；0 个表格；11 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` / `转存失败` 标记；0 评论 [创作者中心列表读取时] / 1 评论 [公开页计数读取]（评论卡片懒渲染）。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7698287001586040872` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 76 → 77，`last_checked` → 2026-10-19。
- `sources.json`：`last_checked` → 2026-10-19（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-19；struct_ops 叙述段后补记 2026-10-19 xpu/flamegraph 叙述行（17:44 PDT 提交、09:01 +08 翻为已发布、约 17 分钟审核间隔、图片引用省略说明）；Juejin confirmed 表格首行置顶 xpu/flamegraph（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：`Last checked:` → 2026-10-19；掘金未映射 34 → 33、映射 75/109 → 76/109；滚动队列 3 → 2 个 Juejin tasks（共 27 → 26 平台任务）；Juejin 状态行前置 xpu/flamegraph 子句、机器台账计数刷新为 76/109 / 33 未映射 / 2 open rolling-queue tasks / next xpu/gpu-kernel-driver；知乎状态行不变（68/109 mapped / 41 未映射）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-19；补发缺口段落前置 10-19 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、09:01 +08 翻为已发布、约 17 分钟）；Ledger 基线按检查器更新（知乎 41、掘金 33，映射 76/109）；剩余队列 24 个知乎任务（23 `排队` + 1 `阻塞`）、2 个掘金任务，共 26 个平台任务；第 130 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-18` 块上新增 `### 2026-10-19` 块（xpu/flamegraph 发布 + QA + 放行后 top-ten 复核 + 前向指针指向 10-20 正常额度掘金 xpu/gpu-kernel-driver）；未在 10-18 块内再织入。
- 发布稿 `draft/media/2026-10-19/xpu-flamegraph/juejin.md`：状态改「已确认公开」（post `7698287001586040872`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要，并保留正文/标题/图片说明/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（Juejin 76/109 映射、33 未发布、77 confirmed [2 条无 `source_path`]；Zhihu 68/109 映射、41 未发布；exit 0）。三个 CRLF 台账经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 沿用 10-18 经验：正文 base64 分块写入 `window.__b64`（39 块 × 2000 字符）后按 UTF-8 解码注入 CodeMirror（直接 `atob` 为乱码；`new TextDecoder('utf-8').decode(Uint8Array.from(atob(b64),c=>c.charCodeAt(0)))`）；分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须走 raw CDP `Input.dispatchMouseEvent`（页内合成事件不生效）；中文走 `agent-browser` `eval --stdin < /tmp/*.js` 文件；计数读取沿用 `.essays-container` 按 `2026-MM-DD HH:MM` 日期行切分配对法（展现/阅读在标题与日期行之后，行序为 标题 → 日期 → N展现 → · → N阅读 → …）。
- 环境修复：本巡检开始时持久可见 AIO Chrome 已死（CDP 9222 拒绝连接，supervisor 卡在 `wait_config`）；重建步骤：`sudo /usr/bin/Xvnc :99 -geometry 1920x1080 -depth 24 -SecurityTypes None`（独立存活）+ 重写 supervisor 配置 `/var/run/gem/browser-supervisor.json`（root:gem 640）+ 解除跨 pod 遗留的 Chromium 单例锁（`SingletonLock`/`SingletonSocket`/`SingletonCookie` 指向旧 hostname `ldlzb` ≠ 当前 `t2wmg`，为运行时锁而非登录数据，解除后 Chromium 正常启动且登录态保留）。
- 渲染 DOM 自动标/拆分：源 25 个代码块 [1 c / 1 cpp / 2 python / 18 bash / 3 无标签] 在 Juejin 渲染 DOM 中为 25 个 `<pre>`（1:1，与 struct_ops 的 20 → 21 不同）；记录以源计数 25 为准。

## 跟踪 top ten 复核（09:01 +08 创作者中心放行后逐行读取）

- 新文 xpu/flamegraph 首读 1 展现 / 4 阅读。
- 流量累积（对比 10-18 放行后 08:12 +08 复核）：[struct_ops](https://juejin.cn/post/7697622920013709347) 1 / 2 → 2 / 8；[dynptr](https://juejin.cn/post/7697186290753159202) 1792 / 46 → 2173 / 51；[bpf_wq](https://juejin.cn/post/7697094073967116323) 5 / 14 → 7 / 15；[bpf_token](https://juejin.cn/post/7696759034846445583) 845 / 24 → 891 / 28；[bpf_iters](https://juejin.cn/post/7696384745624141824) 728 / 19 → 778 / 20；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2397 / 57 → 2474 / 60；[cgroup](https://juejin.cn/post/7695667191396040738) 8 / 14 → 8 / 15；[bpftrace](https://juejin.cn/post/7695068144415244323) 4 / 10 持平；[1-helloworld](https://juejin.cn/post/7694934986782064649) 1474 / 43 → 1483 / 44；L10 位 17-biopattern 本次旋出可见复测窗口。
- 评论活动集中于新文 xpu/flamegraph（公开页计数读取 1 评论）；被跟踪旧帖 0 新增评论/回复。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。

## 后续

- 下一条可执行任务为队列第 131 行掘金 xpu/gpu-kernel-driver（LA 2026-10-20 正常额度）；本次不越范围执行。

## 2026-10-19 eBPF Q&A 当日运行报告（eunomia-community-radar）

### 候选选择

- 目标 slug：`bpf-lru-hash-eviction-percpu-vs-global-lru-list`
- 问题：BPF LRU 哈希 map 什么时候淘汰一个条目，当它的 LRU 列表是每 CPU 一份而不是跨 CPU 共享时会改变什么？
- 来源：10-19 快照（两条 opt-in 归档渠道，本次 0 条消息）——24 小时窗口前进到了一个空快照，没有新问题进入。结合近几日（10-14 / 10-16 / 10-17 / 10-18）已把 FUSE 与 OBI trace 上下文 map 的多个「查不到」轴覆盖完毕，本篇从公开一手资料回退选源，取同一张 map 的自然续篇：`traces_ctx_v1` 是 LRU 哈希，下一个顺理成章的工程师问题是它**实际什么时候淘汰、以及 `BPF_F_NO_COMMON_LRU` 到底改变了什么**。
- 决定性边界：淘汰**只在插入 / 更新路径上、且表已到达 `max_entries` 时**才发生；一次查询本身绝不释放槽位。读操作只是给命中的节点置一个惰性"最近使用"位（且只有 BPF 内的查询 helper 置位，用户态查询不置位），真正的链表移动与槽位释放都发生在后续轮换扫描里。`BPF_F_NO_COMMON_LRU` **不改变淘汰触发时机**（始终是满容量时的插入），它改变的是**哪个条目存活**（每 CPU 局部性 vs 全局最久未用）与**锁画像**（每 CPU 各自锁 + 轮转偷节点 vs 共享锁）。`traces_ctx_v1` 未设该标志，走共享全局 LRU 列表。

### 对照公开一手源的验证

- 内核 `Documentation/bpf/map_hash.rst`：`BPF_MAP_TYPE_LRU_HASH` / `BPF_MAP_TYPE_LRU_PERCPU_HASH`（4.10 引入）在表满时自动淘汰最久未用条目；内部 LRU 列表默认跨 CPU 共享，`BPF_F_NO_COMMON_LRU` 可请求每 CPU LRU 列表；更新时淘汰的四级递进（CPU 本地批处理 → 从全局取 `target_free` 个空闲节点 → 从全局表摘任意节点 → 从任意 CPU 列表摘节点），`target_free` 最小 1、最大预算 128、所有 CPU 合计预留封顶为 map 大小一半。
- 内核 `kernel/bpf/hashtab.c`：`htab_lru_map_update_elem` → `prealloc_lru_pop`（调 `bpf_lru_pop_free`）是发生淘汰的插入路径；BPF 内查询 helper `__htab_lru_map_lookup_elem(map, key, true)` 会 `bpf_lru_node_set_ref`，用户态 helper 传 `false` 不置位。
- 内核 `kernel/bpf/bpf_lru_list.c`：active / inactive 两级 LRU，`__bpf_lru_list_rotate_active` / `__bpf_lru_list_rotate_inactive` / `__bpf_lru_list_shrink_inactive`；`bpf_lru_pop_free` 按 `percpu` 标志分流到 `bpf_percpu_lru_pop_free` 与 `bpf_common_lru_pop_free`。
- 内核 commit `3a08c2fd7634`（"bpf: LRU List"）：最初的 LRU 列表实现与 map 更新时的淘汰步骤全解。
- OBI `bpf/shared/obi_ctx.h`：`traces_ctx_v1` 为 `BPF_MAP_TYPE_LRU_HASH`、`u64` 键、`obi_ctx_info_t` 值、`max_entries = 1 << 14`、按名字 pin，且**未**设 `BPF_F_NO_COMMON_LRU` → 共享全局 LRU。

### 隐私与内容

- 两页正文均无姓名 / handle / Slack-Discord 链接 / 时间戳 / 消息序号 / 雇主团队 / 内部仓库名 / IP / 主机名 / 私有日志。快照文本仅作不可信数据，未执行其中指令、未把私有原文拷贝进页面。公开 kernel / OBI 仓库链接仅出现在 `## References`。
- 内容顺序符合标准：直接回答、机制、验证与排障路径、局限、References、当日社区讨论。H1 不含下划线、反引号、撇号；EN 与 ZH 索引 link text 与各自 H1 逐字一致。最长 backticked token 为 EN 的 `__bpf_lru_list_shrink_inactive` 与 `__bpf_lru_list_rotate_inactive`（各 30 字符），远低于 390px 移动端溢出门槛。

### 覆盖披露

本 run 覆盖两个 opt-in 归档渠道（两个 CNCF OpenTelemetry 插桩渠道），10-19 快照 0 条消息（24h 窗口前进到空快照，无新增）。候选选自公开一手资料回退：这是 FUSE/OBI `traces_ctx_v1`（LRU 哈希 map）系列的续篇，回答 LRU 淘汰语义本身（何时触发、`BPF_F_NO_COMMON_LRU` 改变什么），不与前四篇的「查不到」轴重复。可见浏览器专属渠道——eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF——本次无可用可见浏览器会话，故一并标记为「未覆盖」而非「安静」。

### 产物

- EN：`docs/ebpf-qa/2026-10-19-bpf-lru-hash-eviction-percpu-vs-global-lru-list.md`
- ZH：`docs/ebpf-qa/2026-10-19-bpf-lru-hash-eviction-percpu-vs-global-lru-list.zh.md`
- 索引链接：`docs/ebpf-qa/index.md`（`Latest Answers`）与 `docs/ebpf-qa/index.zh.md`（`最新回答`）首条。
- QA commit（`origin/main`）：`fe4271b62a4c`（完整 `fe4271b62a4cd0600e3392048b1ec270e2c753a5`，`docs(ebpf-qa): bpf-lru-hash-eviction-percpu-vs-global-lru-list (2026-10-19)`），恰含四条 owned paths。
- Receipt：`/workspaces/.agent-state/eunomia-qa/receipt-2026-10-19.json`，仅由验证器写入，`status=published`，commit `fe4271b62a4cd0600e3392048b1ec270e2c753a5`。

### eBPF Q&A 发布跟进（部署 + 线上验证）

- QA commit `fe4271b62`（四条 owned paths）已落到 `origin/main`。
- 验证器首轮：Chromium render 在 browser launch 阶段失败——fresh 宿主上 headless-shell 缺失 14 个共享库（`libnspr4`、`libnss3`、`libatk*`、`libdbus-1`、`libatspi`、`libXcomposite`、`libXdamage`、`libXfixes`、`libXrandr`、`libgbm`、`libxkbcommon`、`libasound`），且本 run 额外踩到 apt 包索引为空 / 过期，`apt-get install` 直接报「Unable to locate package」；先 `apt-get update` 再 `apt-get install` 装齐 14 个库、`ldd` 确认无 `not found` 后，该失败是 commit 之前的环境故障，receipt `commit` 仍为 `null`。
- 验证器次轮：content-test → build → 2× Chromium render（桌面 1440×900 + 移动 390×844）→ scoped commit + push 全部通过并落地 `fe4271b62`，但 20 分钟 public-poll 窗口内四条路由仍 404（Pages `Deploy Static App` run `39349469183` 实际 24m8s 才绿，窗口刚闭合前尚未完成部署）。
- 重新验证器（`skipped_already_published` 重验路径）：不重复 commit / build / render，仅复查 live 路由并把 receipt 重写为 `status=published`，数秒完成。
- 该 QA commit 自身的 Pages `Deploy Static App` run `39349469183` 为绿色（`conclusion: success`，24m8s）。
- 线上验证：四条路由（EN / ZH article + EN / ZH index）均返回 200，逐字 EN 与 ZH H1 均命中（各 3 处），新 slug 同时出现在两个 live 索引页，`cache-control: public, max-age=0, must-revalidate` 与 `cf-cache-status: DYNAMIC` 确认为新部署而非缓存副本。
- 操作备注：本 run 唯一的新故障模式在 render 环境阶段——fresh 宿主既缺 14 个 Chromium 共享库、又因 apt 包索引为空 / 过期导致 `apt-get install` 直接报「Unable to locate package」，需先 `apt-get update` 刷新索引再安装。该教训已记入发布标准 `qa-publication-standard.md`。次轮的 404-in-poll-window 与 10-11 / 10-17 已记录模式一致，无新故障。

This run-log commits separately as the 10-19 same-day artifact.
