# 2026-10-15 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-15。唯一提交时刻为 2026-10-16 07:36:54 +08（= 2026-10-15 16:36:54 PDT），落在 LA 2026-10-15。巡检开始时补发缺口为 0 条（10-09 → 10-10 → 10-11 → 10-12 → 10-13 → 10-14 → 10-15 连续巡检，无新增补发缺口），bpf_token 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列第 13 行 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 bpf_token（队列第 124 行；第 121 / 123 行知乎 bpf_iters / bpf_token 行均 `阻塞`，跳过）；其后下一可执行任务为第 126 行掘金 bpf_wq（第 125 行为知乎 bpf_wq 行，`阻塞`）。本次未启动任何知乎任务。

## 队列任务：掘金 bpf_token（LA 10-15 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 124 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 122 行 bpf_iters 已于 10-14 巡检完成；第 121 / 123 行为知乎 bpf_iters / bpf_token 行（`阻塞`），跳过；下一可执行掘金行第 126 行为 bpf_wq（LA 2026-10-16 正常额度，第 125 行为知乎 bpf_wq 行，`阻塞`）。
- 源文：`docs/tutorials/features/bpf_token/README.zh.md`（289 行）；发布稿 `draft/media/2026-10-15/bpf_token/juejin-body.md`（移除源 H1，其余正文逐字保留；10102 字符、15930 字节、287 个换行、9 个 H2 / 7 个 H3 / 0 个 H4、14 个代码块 [5 bash：构建 / 运行 / 预期输出 / sudo 验证 / 复现示例；4 c：bpf_token_create.c 用户态程序；3 无语言标签：map 创建参数、预期 BPF token 输出、程序加载预期输出；2 text：预期内核日志 / 报错文本]、0 张图片、1 个表格 [6 行：BPF token 委托选项表]、10 条唯一外链目标 [docs.ebpf.io BPF token 概念文档、docs.ebpf.io bpf_token_create() libbpf API、docs.kernel.org/bpf/、documentation.ubuntu.com LXD BPF 说明、eunomia.dev/tutorials/、github.com/eunomia-bpf/bpf-developer-tutorial ×2（repo 根 + src/features/bpf_token 子树）、lore.kernel.org bpf token 补丁系列（Andrii Nakryiko）、lwn.net/Articles/947173/、lwn.net/Articles/959350/]、0 条相对链接）。
- 标题逐字取源 H1「eBPF 入门实践教程：BPF Token，安全的委托式权限与程序加载」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；标题经 `input[placeholder*="标题"]` 原生 setter 写入并回读，正文以 base64 分块写入 `window.__b64` 后按 UTF-8 解码注入 CodeMirror，写后回读 10102 字符与本地一致。
- 分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（`后端` chip 无下拉选项、经 Enter 落定，`Linux` 与 `性能优化` 经 CDP 指针点击可见 `.byte-select-option`；最终 chip DOM 顺序 `Linux`、`后端`、`性能优化`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters 教程标签组）。

## 提交

- 提交时刻 2026-10-16 07:36:54 +08 = 2026-10-15 16:36:54 PDT；提交前创作者中心基线 全部 84 / 已发布 84 / 审核中 0 / 未通过 0（bpf_iters 提交后基线）。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7696759034846445583`，URL `https://juejin.cn/post/7696759034846445583`。提交后创作者中心 全部 85 / 已发布 84 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7696759034846445583`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；07:41 +08 轮询翻为 全部 85 / 已发布 85 / 审核中 0（审核间隔约 5 分钟，与 26-sudo / 20-tc / bpftrace / bpf_arena / bpf_iters 快档先例一致；对比最长 1-helloworld 约 173 分钟、40 分钟档 38-btf-uprobe），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial / bpf_arena / bpf_iters；区别于 bpftrace / 26-sudo / 20-tc 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- bpf_token `/post/7696759034846445583`：标题逐字「eBPF 入门实践教程：BPF Token，安全的委托式权限与程序加载」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（10102 字符 artifact；渲染 DOM 对代码块空白做归一化，`.article-viewer.markdown-body` innerText 读为 9291）；9 个 H2 / 7 个 H3 / 0 个 H4；14 个代码块 [5 bash / 4 c / 3 无语言标签 / 2 text]；0 张图片（正文区域无内容图）；1 个表格 [3 th / 6 行]；10 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` / `发布失败` 标记；评论 0（`暂无评论` 空态）。早期计数（07:48 +08 创作者中心 top-ten 复核检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7696759034846445583` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 72 → 73，`last_checked` → 2026-10-15。
- `sources.json`：`last_checked` → 2026-10-15（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-15；bpf_iters 叙述段后补记 2026-10-15 bpf_token 叙述行（16:36 PDT 提交、07:41 +08 翻为已发布、约 5 分钟审核间隔、快档先例）；Juejin confirmed 表格首行置顶 bpf_token（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：`Last checked:` → 2026-10-15；掘金未映射 37 → 36（71/108 → 72/108）、滚动队列 7 → 6 Juejin tasks（24 Zhihu and 6 Juejin tasks，共 31 → 30）；掘金状态行前置 bpf_token 子句。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-15；补发缺口段落前置 10-15 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、07:41 +08 翻为已发布、约 5 分钟、与 26-sudo / 20-tc / bpftrace / bpf_arena / bpf_iters 快档先例一致）；Ledger 基线按检查器更新（掘金 36，映射 72/108）；剩余队列掘金 7 → 6、总计 31 → 30；第 124 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-15` 检查点新增于 `### 2026-10-14` 之上，含发布 + 审核间隔、公开页 QA + 早期计数、跟踪 top ten 移动、前向指针四条（07:48 +08 复核检查点 0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏），前向指针改指 10-16 正常额度掘金 bpf_wq。
- 发布稿 `draft/media/2026-10-15/bpf_token/juejin.md`：状态改「已确认公开」（post `7696759034846445583`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要；早期计数引用 07:48 +08 复核检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 72/108 映射、36 未发布、73 confirmed [2 条无 `source_path`]；exit 0）。

## 编辑器经验

- 正文注入：body 文件 base64 化后分块写入 `window.__b64`，再 `document.querySelector('.CodeMirror').CodeMirror.setValue(new TextDecoder('utf-8').decode(Uint8Array.from(atob(window.__b64), c => c.charCodeAt(0))))`（直接 `atob` 结果为乱码，必须按 UTF-8 解码）。
- 分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须用 raw CDP `Input.dispatchMouseEvent`——页内合成 `dispatchEvent` 不会真正选中/发布；提交成功判据为弹窗「发布成功」并落地 `https://juejin.cn/published`。
- 驱动走 `agent-browser-wrapper.sh`（daemon 已挂 CDP 9222）：中文经 `eval --stdin < /tmp/*.js` 走文件（inline shell 嵌中文字符串会撑破解析器）；计数读取沿用 10-13 对 `.essays-container a.link` 逐锚点 `innerText` 正则行配对法。

## 跟踪 top ten 复核（07:48 +08 创作者中心逐行读取）

- 流量累积：[bpf_arena](https://juejin.cn/post/7695977967902933026)（10-13 巡检 07:30 +08 放行）由 10-14 检查点 1764 展现 / 39 阅读 累积至 07:48 +08 轮询 2148 展现 / 49 阅读（+384 展现 / +10 阅读）。
- 持平：[bpf_iters](https://juejin.cn/post/7696384745624141824)（10-14 巡检 08:56 +08 放行）读 311 展现 / 13 阅读（与 10-14 检查点一致）。
- 未重测：[27-replace](https://juejin.cn/post/7691510191482519603)、[28-detach](https://juejin.cn/post/7691345821565141019)、[29-sockops](https://juejin.cn/post/7691585964054003758)、[31-goroutine](https://juejin.cn/post/7691326130511167526)、[32-wallclock-profiler](https://juejin.cn/post/7691151105851031590)、[33-funclatency](https://juejin.cn/post/7690830871915216922)、[34-syscall](https://juejin.cn/post/7690415131084324902)、[35-user-ringbuf](https://juejin.cn/post/7689742195037110323)、[37-uprobe-rust](https://juejin.cn/post/7689408456146599963) 九条已随较新帖（bpf_iters / bpf_token）挤入而下移出创作者中心「最新十条」可视窗口（全部 tab / 已发布 tab 均只渲染最新 10 行），故本次未逐行重测其计数；全部被跟踪行仍读 0 评论（`暂无评论数据` 空态持续），无需回复或更正。
- 新文 bpf_token 在 07:48 +08 复核检查点读 0 展现 / 1 阅读。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对每行 `.essays-container a.link` 锚点取 `innerText` 并按 `(\d+)\s*展现\s*·\s*(\d+)\s*阅读\s*·…` 逐行配对（行精确，避免跨行错位配对）。


This run-log commits separately as the 10-15 same-day artifact.
