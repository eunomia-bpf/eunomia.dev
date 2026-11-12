# 2026-10-20 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-20。唯一提交时刻为 2026-10-21 07:29 +08（= 2026-10-20 16:29 PDT），落在 LA 2026-10-20。巡检开始时补发缺口为 0 条（2026-09-03 与 2026-09-16 两条缺口已于 2026-10-02 巡检核销；10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 → 10-16 → 10-17 → 10-18 → 10-19 → 10-20 连续巡检，无新增补发缺口），xpu/gpu-kernel-driver 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 xpu/gpu-kernel-driver（队列第 131 行）。

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

## 队列任务：掘金 xpu/gpu-kernel-driver（LA 10-20 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 131 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 130 行 xpu/flamegraph 已于 10-19 巡检完成；下一可执行掘金行第 132 行为 xpu/npu-kernel-driver（LA 2026-10-21 正常额度）。
- 源文：`docs/tutorials/xpu/gpu-kernel-driver/README.zh.md`；发布稿 `draft/media/2026-10-20/xpu-gpu-kernel-driver/juejin-body.md`（移除源 H1，其余正文逐字保留；11994 字符、8 个代码块 [1 c、5 bash、2 无语言标签]、0 张图片、0 个表格、6 条唯一外链目标、0 条 eunomia.dev 链接）。
- 标题逐字取源 H1「eBPF 实例教程：使用内核跟踪点监控 GPU 驱动活动」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（沿用 17-biopattern / 1-helloworld / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu/flamegraph 教程标签组；`Linux` chip 经真实 CDP 指针点击可见 `.byte-select-option` 提交，`后端` / `性能优化` 因 CDP 指针在选项中心反复未命中，改经页内合成点击提交）。

## 提交

- 提交时刻 2026-10-21 07:29 +08 = 2026-10-20 16:29 PDT；提交前创作者中心基线 全部 89 / 已发布 89 / 审核中 0 / 未通过 0（xpu/flamegraph 放行后基线）。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 post id `7698552541928554511`，URL `https://juejin.cn/post/7698552541928554511`（取自创作者中心审核中列表行）。提交后创作者中心 全部 90 / 已发布 89 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7698552541928554511`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；~07:32 +08 轮询翻为 全部 90 / 已发布 90 / 审核中 0（审核间隔约 3 分钟，快档，与 26-sudo / 20-tc / bpftrace / bpf_arena / bpf_iters / bpf_token 同级，快于 xpu/flamegraph 的 17 分钟），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops / xpu/flamegraph；区别于 bpftrace / 26-sudo / 20-tc / 31-goroutine 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- xpu/gpu-kernel-driver `/post/7698552541928554511`：标题逐字「eBPF 实例教程：使用内核跟踪点监控 GPU 驱动活动」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（11994 字符 artifact；渲染 DOM innerText 读为 11482，代码块空白归一化）；10 个 H2 / 3 个 H3 / 0 个 H4；8 个代码块 [1 c、5 bash、2 无语言标签，渲染 DOM 8 个 `<pre>`，与源 8 个围栏代码块 1:1]；0 张图片；0 个表格；6 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；公开页 5 评论。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7698552541928554511` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 77 → 78，`last_checked` → 2026-10-20。
- `sources.json`：`last_checked` → 2026-10-20（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-20；xpu/flamegraph 叙述段后补记 2026-10-20 xpu/gpu-kernel-driver 叙述行（16:29 PDT 提交、~07:32 +08 翻为已发布、约 3 分钟审核间隔）；Juejin confirmed 表格首行置顶 xpu/gpu-kernel-driver（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：`Last checked:` → 2026-10-20；掘金未映射 33 → 32、映射 76/109 → 77/109；滚动队列 2 → 1 个 Juejin tasks（共 26 → 25 平台任务）；Juejin 状态行前置 xpu/gpu-kernel-driver 子句、机器台账计数刷新为 77/109 / 32 未映射 / 1 open rolling-queue task / next xpu/npu-kernel-driver；知乎状态行不变（68/109 mapped / 41 未映射）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-20；补发缺口段落前置 10-20 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、~07:32 +08 翻为已发布、约 3 分钟）；Ledger 基线按检查器更新（知乎 41、掘金 32，映射 77/109）；剩余队列 24 个知乎任务（23 `排队` + 1 `阻塞`）、1 个掘金任务，共 25 个平台任务；第 131 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：在已有 `### 2026-10-19` 块上新增 `### 2026-10-20` 块（xpu/gpu-kernel-driver 发布 + QA + 放行后 top-ten 复核 + 前向指针指向 10-21 正常额度掘金 xpu/npu-kernel-driver）；未在 10-19 块内再织入。
- 发布稿 `draft/media/2026-10-20/xpu-gpu-kernel-driver/juejin.md`：状态改「已确认公开」（post `7698552541928554511`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要，并保留正文/标题/原文/分类/标签各行。
- 验证器：`check_media_ledger.py` 通过（Juejin 77/109 映射、32 未发布、78 confirmed；Zhihu 68/109 映射、41 未发布；exit 0）。CRLF 台账经字节级编辑，编辑后复查 0 条裸 LF。

## 编辑器经验

- 标签提交双通道：`Linux` chip 经真实 CDP 指针点击可见 `.byte-select-option` 正常提交；`后端` / `性能优化` 选项在可见视口内但 CDP 指针（`Input.dispatchMouseEvent`）在选项中心坐标反复点击不命中，改经页内合成点击（`elementFromPoint` 定位 + `dispatchEvent` 合成 `MouseEvent` 序列）提交成功。后续巡检沿用：先试 CDP 指针，未命中即在选项中心改走页内合成点击，不反复重试 CDP。
- 计数读取沿用 `.essays-container` 按 `2026-MM-DD HH:MM` 日期行切分配对法（展现/阅读在标题与日期行之后，行序为 标题 → 日期 → N展现 → · → N阅读 → …）；bpftrace 行以列表标题「bpftrace一行教程」标识。

## 跟踪 top ten 复核（~07:32 +08 创作者中心放行后逐行读取）

- 新文 xpu/gpu-kernel-driver 首读 230 展现 / 7 阅读 / 1 点赞 / 1 收藏（放行窗口首读 32 展现 / 2 阅读，~3 分钟审核间隔内已累积至 230 / 7）。
- 流量累积（对比 10-19 放行后 09:01 +08 复核）：[xpu/flamegraph](https://juejin.cn/post/7698287001586040872) 1 / 4 → 393 / 19；[struct_ops](https://juejin.cn/post/7697622920013709347) 2 / 8 → 4 / 10；[dynptr](https://juejin.cn/post/7697186290753159202) 2173 / 51 → 2260 / 51；[bpf_wq](https://juejin.cn/post/7697094073967116323) 7 / 15 → 9 / 17；[bpf_token](https://juejin.cn/post/7696759034846445583) 891 / 28 → 932 / 31；[bpf_iters](https://juejin.cn/post/7696384745624141824) 778 / 20 → 810 / 21；[bpf_arena](https://juejin.cn/post/7695977967902933026) 2474 / 60 → 2495 / 62；[cgroup](https://juejin.cn/post/7695667191396040738) 8 / 15 → 8 / 19；[bpftrace](https://juejin.cn/post/7695068144415244323) 4 / 10 持平；L10 位 1-helloworld 本次旋出可见复测窗口。
- 评论活动集中于新文 xpu/gpu-kernel-driver（公开页 5 评论）；被跟踪旧帖 0 新增评论/回复。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对 `.essays-container` 的 innerText 按 `2026-MM-DD HH:MM` 日期行切分逐行配对展现/阅读（行精确，避免跨行错位配对）。


## 后续

- 下一条可执行任务为队列第 132 行掘金 xpu/npu-kernel-driver（LA 2026-10-21 正常额度）；本次不越范围执行。

This run-log commits separately as the 10-20 same-day artifact.
