# 2026-10-21 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-21。唯一提交时刻为公开页 JSON-LD `datePublished` 2026-10-21T16:13:55-07:00 = 16:13 PDT（= 2026-10-22 07:13:55 +08；创作者中心文章行显示 2026-10-21 16:13），落在 LA 2026-10-21。巡检开始时补发缺口为 0 条（2026-09-03 与 2026-09-16 两条缺口已于 2026-10-02 巡检核销；10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 → 10-18 → 10-19 → 10-20 → 10-21 连续巡检，无新增补发缺口），xpu/npu-kernel-driver 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 xpu/npu-kernel-driver（队列第 132 行）。

## 队列任务：掘金 xpu/npu-kernel-driver（LA 10-21 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 132 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 131 行 xpu/gpu-kernel-driver 已于 10-20 巡检完成；本行翻转后队列不再剩掘金行（剩余队列共 24 个平台任务，全部为知乎侧）。
- 源文：`docs/tutorials/xpu/npu-kernel-driver/README.zh.md`；发布稿 `draft/media/2026-10-21/xpu-npu-kernel-driver/juejin-body.md`（移除源 H1，其余正文逐字保留；9180 字符、15795 字节、280 行、9 个 H2 / 2 个 H3 / 0 个 H4、9 个代码块 [9 bash，全部带语言标签]、0 张图片、0 个表格、5 条唯一外链目标 [github.com/eunomia-bpf/bpf-developer-tutorial、eunomia.dev/tutorials/、github.com/intel/linux-npu-driver、github.com/iovisor/bpftrace/blob/master/docs/reference_guide.md、github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/xpu/npu-kernel-driver]，含源尾注 blockquote [eunomia.dev/tutorials + 教程仓库链接]）。
- 标题逐字取源 H1「eBPF 实例教程：跟踪 Intel NPU 内核驱动操作」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；分类 `后端`；标签 `Linux`、`后端`、`性能优化`（沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu-flamegraph / xpu-gpu-kernel-driver 教程标签组）。

## 提交

- 提交时刻以公开页 JSON-LD `datePublished` 2026-10-21T16:13:55-07:00 = 16:13 PDT 为准（= 2026-10-22 07:13:55 +08；创作者中心文章行显示 2026-10-21 16:13，两处一致）。提交前基线承 10-20 xpu/gpu-kernel-driver 巡检放行后读数 全部 90 / 已发布 90 / 审核中 0 / 未通过 0。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 post id `7698947426490318888`，URL `https://juejin.cn/post/7698947426490318888`。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7698947426490318888`；本 run 未记录窗口内中间读数，按结果判定——审核间隔结束后创作者中心读为 全部 91 / 已发布 91 / 审核中 0 / 未通过 0，随后暂存 URL 重定向至规范地址（登录态浏览器打开 `/spost/7698947426490318888` 落到 `https://juejin.cn/post/7698947426490318888`）。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu/flamegraph / xpu/gpu-kernel-driver；区别于 bpftrace / 26-sudo / 20-tc / 31-goroutine 直接公开条目无 `staged_url`）。NPU 是本系列首个观测到暂存 URL 后续重定向至 `/post/` 的放行先例。

## 公开页 QA（登录浏览器）

- xpu/npu-kernel-driver `/post/7698947426490318888`：标题逐字「eBPF 实例教程：跟踪 Intel NPU 内核驱动操作」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（9180 字符 artifact；渲染 DOM innerText 读为 8687，代码块空白归一化）；9 个 H2 / 2 个 H3 / 0 个 H4；9 个代码块 [9 bash，全部带语言标签，渲染 DOM 9 个 `<pre>`，与源 9 个围栏代码块 1:1]；0 张图片；0 个表格；5 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；公开页 `暂无评论`（0 评论、0 点赞、0 收藏）。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7698947426490318888` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 78 → 79，`last_checked` → 2026-10-21。
- `sources.json`：`last_checked` → 2026-10-21（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-21；xpu/gpu-kernel-driver 叙述段后补记 2026-10-21 xpu/npu-kernel-driver 叙述行（JSON-LD 16:13 PDT、无补发缺口核销、基线 90/90/0 → 91/91/0、/spost/ 现重定向 /post/）；Juejin confirmed 表格首行置顶 xpu/npu-kernel-driver（newest-first，含 `/post/` 正式地址与 QA 摘要）。
- `not-published.md`：`Last checked:` → 2026-10-21；掘金未映射 32 → 31、映射 77/109 → 78/109；滚动队列 1 → 0 个 Juejin tasks（共 25 → 24 平台任务）；Juejin 状态行前置 xpu/npu-kernel-driver 子句、机器台账计数刷新为 78/109 / 31 未映射 / 0 open rolling-queue tasks；知乎状态行不变（68/109 mapped / 41 未映射）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-21；补发缺口段落前置 10-21 巡检句（正常额度 1 条、JSON-LD 16:13 PDT、/spost/ 暂存、90/90/0 基线 → 91/91/0 放行、/spost/ 现重定向 /post/）；Ledger 基线按检查器更新（知乎 41、掘金 31，映射 78/109）；剩余队列 24 个知乎任务（23 `排队` + 1 `阻塞`）、0 个掘金任务，共 24 个平台任务；第 132 行翻 `[x]` 并附 JSON-LD 提交时刻、放行结果、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-20` 块上新增 `### 2026-10-21` 块（xpu/npu-kernel-driver 发布 + QA + 放行后 top-ten 复核 + 前向指针：掘金队列 0 剩余任务，下一巡检点无新发布、仅复核跟踪集）；未在 10-20 块内再织入。
- 发布稿 `draft/media/2026-10-21/xpu-npu-kernel-driver/juejin.md`：状态改「已确认公开」（post `7698947426490318888`，`confirmed`），记录正式地址、JSON-LD 提交时刻、/spost/ → /post/ 重定向与 QA 摘要，并保留正文/标题/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（Juejin 78/109 映射、31 未发布、79 confirmed；Zhihu 68/109 映射、41 未发布；exit 0）。CRLF 台账经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 标签/分类提交沿用 10-20 双通道经验（先试 CDP 指针点击 `.byte-select-option`，未命中即在选项中心改走页内合成点击）；本 run 成功提交 `后端` 分类与 `Linux`、`后端`、`性能优化` 三枚 chip，无新编辑器故障。
- 新审核间隔观察：与 10-20 xpu/gpu-kernel-driver run（审核窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200、可用性须以登录创作者中心判定）不同，本 run 观测到审核间隔结束、`/spost/` 后续重定向至 `/post/`（打开暂存地址即落到规范地址）；窗口内中间读数未记录，判定以上述结果读数为准。计数读取沿用 `.essays-container` 按 `2026-MM-DD HH:MM` 日期行切分配对法（展现/阅读在标题与日期行之后，行序为 标题 → 日期 → N展现 → · → N阅读 → …）。

## 跟踪 top ten 复核（创作者中心放行后逐行读取）

- 新文 xpu/npu-kernel-driver 首读 302 展现 / 20 阅读 / 0 点赞 / 0 评论 / 0 收藏（新文首读；窗口内三次读取计数 263 / 19 → 284 / 19 → 302 / 20 递增）。
- 流量累积（对比 10-20 放行后读取）：[xpu/gpu-kernel-driver](https://juejin.cn/post/7698552541928554511) 230 / 7 / 1 点赞 / 1 收藏 → 4504 / 42 / 1 点赞 / 1 收藏（发布以来累计）；[xpu/flamegraph](https://juejin.cn/post/7698287001586040872) 393 / 19 → 527 / 20；[struct_ops](https://juejin.cn/post/7697622920013709347) 4 / 10 → 8 / 10；[dynptr](https://juejin.cn/post/7697186290753159202) 2260 / 51 / 0 点赞 → 2312 / 51 / 1 点赞；[bpf_wq](https://juejin.cn/post/7697094073967116323) 9 / 17 → 14 / 18；[bpf_token](https://juejin.cn/post/7696759034846445583) 932 / 31 → 971 / 31；[bpf_iters](https://juejin.cn/post/7696384745624141824) 810 / 21 → 824 / 28；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2495 / 62 → 2508 / 64；[cgroup](https://juejin.cn/post/7695667191396040738) 8 / 19 → 12 / 23，入 #10 位；bpftrace-tutorial（10-20 第 10 位 4 / 10）旋出本窗口跟踪 top ten。
- 评论活动：新文 xpu/npu-kernel-driver 公开页 `暂无评论`；被跟踪 top ten 0 新增评论/回复（创作者中心评论卡片为懒渲染，评论以公开页为准）。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。

## 后续

- 掘金队列已 0 剩余任务（剩余 24 个平台任务全部为知乎侧，仍 `阻塞` 于登录状态恢复条件）。下一 LA 2026-10-22 正常额度无新掘金任务可执行；下一有用检查点为创作者中心跟踪 top ten 复核（无新发布）。

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
- 操作备注：本 run 遇到的两个故障模式均已见于发布标准（远程推进导致 push 被拒 → `reset --soft` 前移后重提，标准第 132–136 行；Pages 部署超出 20 分钟 public-poll 窗口导致 404 → 重跑验证器走 `skipped_already_published` 重验，10-11 / 10-17 已记录模式），无新故障模式；fresh 宿主的 Chromium 共享库教训未复现（本 run 运行前 `ldd` 检查干净，二进制可直接启动）。故发布标准无需新增条目，本次 lessons pass 为 no-op。另记一条 git 共享文件工作流教训（非验证器故障模式）：`draft/media/2026-10-21/run-log.md` 为 QA 与 media content-patrol 当日共享文件（单一 H1 下各挂一份 `##` 报告节、单条收尾行，10-19 / 10-20 既定合并范式）；本 run 的 QA scoped commit `fb33e886e` 误将 media 先提交的 52 行 run-log（`fe6f2b14e`）整体覆盖为 46 行 QA-only 版本（1 file changed, 30 insertions(+), 36 deletions(-)），丢失 media 内容巡检节；已按 10-19 / 10-20 范式将 media 节合回本文件并单独提交修复。该教训记于本操作备注，发布标准仍无需新增条目。

This run-log commits separately as the 10-21 same-day artifact.
