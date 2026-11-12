# 2026-09-27 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-27。提交时刻 2026-09-28 07:08 +08 = 16:08 PDT，落在 LA 2026-09-27；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 35-user-ringbuf（LA 09-27 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 93 行（`排队`，现翻为 `[x]`）。第 92 行知乎同源文仍 `阻塞`（无 `z_c0` 会话，`/creator` 重定向到 `/signin`），本次未涉及。
- 源文：`docs/tutorials/35-user-ringbuf/README.zh.md`；发布稿 `draft/media/2026-09-27/35-user-ringbuf/juejin-body.md`（移除源 H1，其余正文逐字保留；7230 字符、11514 字节、4 个 H2、4 个 H3、0 个 H4、6 个代码块（4 c、1 sh、1 console）、0 张图片、0 个表格、8 条唯一外链；无相对链接残留）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`，草稿箱保持 4 条（ACRFence ×2、eBPF-runtime 09-09、exec-image 07-22），无自动暂存草稿被消费；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块经 `agent-browser eval --stdin` 注入 CodeMirror，写后回读 7230/7230 字符。
- 分类 `后端`（CDP 鼠标点击选中）；提交标签 `Linux`、`后端`、`性能优化`（原生 setter + CDP 选项点击提交，实际落定 chip 顺序已按 DOM 读回确认）。

## 提交

- 提交时刻 2026-09-28 07:08 +08 = 16:08 PDT；弹窗「确定并发布」用页内真实指针事件序列（pointerdown/mousedown/focus/pointerup/mouseup/click），一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7689742195037110323`：暂存 <https://juejin.cn/spost/7689742195037110323>，正式 <https://juejin.cn/post/7689742195037110323>。提交后创作者中心 全部 (65) / 已发布 (64) / 审核中 (1) / 未通过 (0)。

## 提交结果与判定

- 提交后 curl 探测不可作为发布信号（掘金 SPA 壳对任意 `/post/<id>`、`/spost/<id>` 路径在审核期即返回 200，沿用既有结论）；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 浏览器轮询：08:13 +08 创作者中心翻为 已发布 (65) / 审核中 (0)（约 65 分钟审核间隔，长于 37/38 的 36–40 分钟），`/post/7689742195037110323` 渲染公开正文，`/spost/` 跳转至 `/post/`。记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7689742195037110323，登录浏览器）

- 标题逐字「eBPF开发实践：使用 user ring buffer 向内核异步发送信息」（h1 单例，全文出现 1 次）；正文单份。
- 4 个 H2 / 4 个 H3 / 0 个 H4；6 个代码块（`pre code` 语言类统计：4 `language-c`、1 `language-sh`、1 `language-console`）；0 张正文图片；0 个表格。
- 8 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（eunomia.dev/tutorials/ ×1、eunomia.dev/tutorials/11-bootstrap/ ×1、eunomia.dev/zh/tutorials/ ×1、eunomia.dev/zh/tutorials/35-user-ringbuf/ ×1、github bpf-developer-tutorial 仓库根 ×2、github src/35-user-ringbuf ×1、bpftime ×1、lwn.net/Articles/907056/ ×1，共 10 个链接），正文内 0 相对链接（QA 脚本命中的裸斜杠链接均位于站点导航栏，非正文）。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态；首次简单探测在页面早期加载时误报 `COMMENTS_PRESENT`，后续独立探测确认 `has_no_comments_placeholder: true`）。早期计数（08:09–08:13 +08 检查点）：0 展现 / 2 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7689742195037110323`（`confirmed`，置顶至第 53 条），`last_checked` → 2026-09-27；notes 记录审核期、curl 不可信结论、草稿箱 4 条未变与 QA 摘要。
- `sources.json`：`last_checked` → 2026-09-27（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-09-27；`## Juejin` 表格首行新增该条目；补记 2026-09-27 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-27；掘金未映射 57 → 56（52/108）、滚动队列 27 → 26；掘金状态行置顶 35-user-ringbuf（下一位 34-syscall）；知乎计数不变（40 未映射、68/108）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-27；补发缺口段落补 09-27 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（掘金 56，映射 52/108）；剩余队列掘金 27 → 26、总计 51 → 50；第 93 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（52/108）；第 92 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-27` 新检查点（置于 `### 2026-09-26` 之前）：发布记录、审核间隔 07:08→08:13 +08（约 65 分钟）、早期计数 0 展现 / 2 阅读、37-uprobe-rust 10 / 11、38-btf-uprobe 6 / 11、41-xdp-tcpdump 1493 / 44 / 1 点赞 / 2 收藏、全部可见行仍 0 评论、下一检查点在 34-syscall 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式地址、审核时间线与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 52/108 映射、56 未发布、53 confirmed；exit 0）。

## 编辑器经验

- 本次编辑器操作全部复用 2026-09-24/25/26 已回写的技能条目（真实 CDP 指针事件选分类、原生 setter + CDP 点击选标签、页内指针序列点「确定并发布」、base64 分块经 `eval --stdin` 注入、curl 不可信结论），无新平台问题，技能文件未改动。
- 审核清除继续落在分钟级而非秒级：本次约 65 分钟（37/38 为 36–40 分钟）；继续按「浏览器轮询至 审核中 归零」判定，SPA 壳 200 不作为发布信号。
