# 2026-09-29 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-29。提交时刻 2026-09-30 08:51 +08 = 17:51 PDT，落在 LA 2026-09-29；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 33-funclatency（LA 09-29 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 96 行（`排队`，现翻为 `[x]`）。第 90/92/94 行知乎同源文仍 `阻塞`（无 `z_c0` 会话，`/creator` 重定向到 `/signin`），本次未涉及；第 97 行（32-wallclock-profiler）未启动。
- 源文：`docs/tutorials/33-funclatency/README.zh.md`（无「原文地址」页脚，仅 1/23/29/35 有）；发布稿 `draft/media/2026-09-29/33-funclatency/juejin-body.md`（移除源 H1，其余正文逐字保留；6136 字符、9254 字节、5 个 H2、3 个 H3、0 个 H4、6 个围栏 [3 个代码块：1 c、2 console]、0 张图片、0 个表格、5 条唯一外链 [github bpf-developer-tutorial 仓库根 ×1、github src/33-funclatency ×1、github bpftime 仓库 ×1、eunomia.dev/tutorials/ ×1、github iovisor/bcc funclatency.c ×1]；无相对链接）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿）；编辑器自动暂存草稿 7690839722769514505 由本次提交消费、草稿箱保持 4 条；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 6136/6136 字符。
- 分类 `后端`（CDP 鼠标点击选中）；提交标签 后端、Linux、性能优化（原生 setter + 选项点击提交，`byte-select__input` 提交顺序 后端 → Linux → 性能优化；提交新标签时先前已提交标签可能掉落，Vue 提交竞态后回读 chip 确认，缺失补回；最终 DOM 顺序 后端、Linux、性能优化）。

## 提交

- 提交时刻 2026-09-30 08:51 +08 = 17:51 PDT；弹窗「确定并发布」用页内合成指针事件序列（pointerover/enter/mouseover/enter → pointerdown/mousedown(+focus) → pointerup/mouseup → click）一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7690830871915216922`：暂存 <https://juejin.cn/spost/7690830871915216922>，正式 <https://juejin.cn/post/7690830871915216922>。提交前创作者中心 全部 (66) / 已发布 (66) / 审核中 (0) / 未通过 (0) [INFERENCE：由 34-syscall 于 09-28 09:52 +08 翻为 已发布 (66) 后至今无同日发布推得]；提交后 全部 (67) / 已发布 (67) / 审核中 (0) / 未通过 (0)。

## 提交结果与判定

- 直接公开，无 审核中 窗口：提交后创作者中心立即为 已发布 (67) / 审核中 (0)，`/spost/` 与 `/post/` 同时可公开访问，无需任何轮询。
- 提交后 curl 探测不可作为发布信号（掘金 SPA 壳对任意 `/post/<id>`、`/spost/<id>` 路径在审核期即返回 200，沿用既有结论）；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7690830871915216922，登录浏览器）

- 标题逐字「使用 eBPF 测量函数延迟」（title_occurrences==1）；正文单份。
- 5 个 H2 / 3 个 H3 / 0 个 H4（H2：什么是 eBPF？/为什么函数延迟很重要？/用于函数延迟的 eBPF 内核代码/运行函数延迟工具/结论；H3：代码解释/用户空间函数延迟/内核空间函数延迟）；3 个代码块（`pre code` 语言类统计：1 c、2 console）；0 张正文图片；0 个表格。
- 5 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（github bpf-developer-tutorial 仓库根 ×1、github src/33-funclatency ×1、github bpftime 仓库 ×1、eunomia.dev/tutorials/ ×1、github iovisor/bcc funclatency.c ×1），正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数（08:51 +08 检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7690830871915216922`（`confirmed`，置顶），`last_checked` → 2026-09-29；notes 记录直接公开无审核间隔、08:51 +08 检查点 已发布 (67) / 审核中 (0)、curl 不可信结论、自动暂存草稿消费与 QA 摘要（55 条 confirmed）。
- `sources.json`：`last_checked` → 2026-09-29（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-09-29；`## Juejin` 表格首行新增该条目；补记 2026-09-29 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-29；掘金未映射 55 → 54（54/108）、滚动队列 25 → 24（24 Zhihu and 24 Juejin tasks）；掘金状态行置顶 33-funclatency（沿列 34-syscall、35-user-ringbuf、37-uprobe-rust、38-btf-uprobe、41-xdp-tcpdump 不变）；知乎计数不变。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-29；补发缺口段落补 09-29 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（掘金 54，映射 54/108）；剩余队列掘金 25 → 24、总计 49 → 48；第 96 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（54/108）；第 90/92/94 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-29` 新检查点（置于 `### 2026-09-28` 之前）：发布记录、直接公开无审核间隔、早期计数 0 展现 / 1 阅读、34-syscall 4 / 6、35-user-ringbuf 6 / 10、37-uprobe-rust 23 / 17、38-btf-uprobe 11 / 15、41-xdp-tcpdump 4463 / 49 / 1 收藏、全部可见行仍 0 评论、下一检查点在 32-wallclock-profiler 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式/暂存地址、提交时刻与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 54/108 映射、54 未发布、55 confirmed；exit 0）。

## 编辑器经验

- 发布弹窗「确定并发布」面板在普通 CDP 鼠标点击下可能保持 `display:none`（面板零矩形、按钮 0×0）：首次点击未开面板，需对工具栏发布按钮（`button` class `xitu-btn`，右上 ≈ x=896 y=32）重新派发含 hover 的完整指针序列（pointerover/enter/mouseover/enter → pointerdown/mousedown(+focus) → pointerup/mouseup → click）→ 面板展开至 560×801、`确定并发布` 落至 90×30 @(897, 834)；随后对 `确定并发布` 以实时坐标重放页内合成指针序列即得 `发布成功`。已按坐标实时读取重开面板，未硬编码坐标。
- chip 激活态首次读回与 Vue 提交存在竞态（首次读回需再读一次确认）；提交新标签时先前已提交标签可能掉落（观察到提交新标签后 `Linux` 一度消失，补回）；已回读确认 后端、Linux、性能优化 落定，最终 DOM 顺序 后端、Linux、性能优化。
- 本次无审核间隔：判定直接取创作者中心 已发布 (67) / 审核中 (0)。SPA 壳 200 结论沿用：任意 `/post/<id>`、`/spost/<id>` 在审核期即返回 200，curl 状态码不作为发布信号，判定以登录浏览器 + 创作者中心 审核中 计数为准。
