# 2026-09-26 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-26。提交时刻 2026-09-27 07:21 +08 = 16:21 PDT，落在 LA 2026-09-26；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 37-uprobe-rust（LA 09-26 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 91 行（`排队`，现翻为 `[x]`）。第 90 行知乎同源文仍 `阻塞`（无 `z_c0` 会话，`/creator` 重定向到 `/signin`），本次未涉及。
- 源文：`docs/tutorials/37-uprobe-rust/README.zh.md`；发布稿 `draft/media/2026-09-26/37-uprobe-rust/juejin-body.md`（移除源 H1，其余正文逐字保留；5004 字符、7246 字节、5 个 H2、0 个 H3、0 个 H4、9 个代码块（2 rust、7 console）、0 张图片、0 个表格、8 条唯一外链；2 条相对交叉教程链接已改为规范 zh 地址 `https://eunomia.dev/zh/tutorials/30-sslsniff/` 与 `https://eunomia.dev/zh/tutorials/31-goroutine/`，两目标均 200；无相对链接残留）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`，编辑器自动暂存草稿 `7689077368330289203` 由本次提交消费；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块经 `agent-browser eval --stdin` 注入 CodeMirror，写后回读 5004/5004 字符。
- 分类 `后端`；提交标签 `性能优化`、`Linux`、`后端`（实际落定 chip 顺序，已按实际记录）。

## 提交

- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`，编辑器自动暂存草稿 `7689077368330289203` 由本次提交消费；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块（3 块 × ~3222 b64 字符）经 `agent-browser eval --stdin` 注入 CodeMirror，写后回读 5004/5004 字符。
- 提交时刻 2026-09-27 07:21 +08 = 16:21 PDT；弹窗「确定并发布」用页内真实指针事件序列（pointerdown/mousedown/focus/pointerup/mouseup/click），一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7689408456146599963`：暂存 <https://juejin.cn/spost/7689408456146599963>，正式 <https://juejin.cn/post/7689408456146599963>。提交后创作者中心 全部 (64) / 已发布 (63) / 审核中 (1) / 未通过 (0)。

## 提交结果与判定

- 提交后 curl 探测不可作为发布信号（掘金 SPA 壳对任意 `/post/<id>`、`/spost/<id>` 路径在审核期即返回 200，2026-09-25 已确认的结论沿用）；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 浏览器轮询：07:57 +08 创作者中心翻为 已发布 (64) / 审核中 (0)（约 36 分钟审核间隔，与 38-btf-uprobe 的约 40 分钟同量级），`/post/7689408456146599963` 渲染公开正文，`/spost/` 跳转至 `/post/`。记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7689408456146599963，登录浏览器）

- 标题逐字「eBPF 实践：使用 Uprobe 追踪用户态 Rust 应用」（h1 单例）；正文单份（首段「eBPF，即扩展的Berkeley包过滤器（Extended Berkeley Packet Filter）…」出现 1 次）。
- 5 个 H2 / 0 个 H3 / 0 个 H4；9 个代码块（`pre code` 语言类统计：2 `language-rust`、7 `language-console`）；0 张正文图片；0 个表格。
- 8 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（eunomia.dev/tutorials/、bpf-developer-tutorial 仓库根、/zh/tutorials/30-sslsniff/、/zh/tutorials/31-goroutine/、bpftime 仓库、rust-lang.org、Name_mangling、rustc symbol-mangling），`?target=` 内无裸斜杠，正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数（07:57 +08 检查点）：2 展现 / 6 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7689408456146599963`（`confirmed`，置顶至第 52 条），`last_checked` → 2026-09-26；notes 记录审核期、curl 不可信结论与 QA 摘要。
- `sources.json`：`last_checked` → 2026-09-26。
- `published.md`：`Last checked:` → 2026-09-26；`## Juejin` 表格首行新增该条目；补记 2026-09-26 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-26；掘金未映射 57（51/108）、滚动队列 28 → 27；掘金状态行置顶 37-uprobe-rust（下一位 35-user-ringbuf）；知乎计数按检查器更新为 40 个未映射（68/108）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-26；补发缺口段落补 09-26 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（知乎 40、掘金 57，映射 51/108）；剩余队列掘金 28 → 27、总计 52 → 51；第 91 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（51/108）；第 90 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-26` 新检查点（置于 `### 2026-09-25` 之前）：发布记录、审核间隔 07:21→07:57 +08、早期计数 2 展现 / 6 阅读、38-btf-uprobe 2 / 7、41-xdp-tcpdump 4123 / 43、39-nginx 1334 / 37 / 1 点赞 / 2 收藏、40-mysql 938 / 28、全部可见的 10 行跟踪公开文仍 0 评论、精确标题搜索仅见 eunomia.dev 规范页与 GitHub 仓库（无独立转载/引用）、下一检查点在 35-user-ringbuf 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式地址、审核时间线与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 51/108 映射、57 未发布、52 confirmed；exit 0）。

## 编辑器经验

- 本次编辑器操作全部复用 2026-09-24/25 已回写的技能条目（真实 CDP 指针事件选分类、原生 setter + CDP 点击选标签、页内指针序列点「确定并发布」、base64 分块经 `eval --stdin` 注入、curl 不可信结论），无新平台问题，技能文件未改动。
- 审核清除再次落在分钟级而非秒级：本次约 36 分钟；继续按「浏览器轮询至 审核中 归零」判定。
