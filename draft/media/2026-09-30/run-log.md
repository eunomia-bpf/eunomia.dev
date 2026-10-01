# 2026-09-30 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-30。提交时刻 2026-10-01 08:39 +08 = 17:39 PDT，落在 LA 2026-09-30；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 32-wallclock-profiler（LA 09-30 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 97 行（`排队`，现翻为 `[x]`）。队列自顶向下扫描：第 57 行起的知乎任务（agentcgroup-characterization 等 12 条）仍 `阻塞`——巡检开始时在可见浏览器实测 `https://www.zhihu.com/creator` 重定向到 `https://www.zhihu.com/signin?next=%2Fcreator`（无 `z_c0` 会话），阻塞条件未变化；故首个可执行的 `排队` 任务是第 97 行掘金 32-wallclock-profiler（第 97 行之前所有掘金条目均已 `[x]`）。本次未启动任何知乎任务。
- 源文：`docs/tutorials/32-wallclock-profiler/README.zh.md`（末尾有「本文原文链接」页脚）；发布稿 `draft/media/2026-09-30/32-wallclock-profiler/juejin-body.md`（移除源 H1，其余正文逐字保留；13029 字符、20963 字节、9 个 H2、3 个 H3、0 个 H4、10 个围栏 [6 个代码块：4 bash、2 c，其余 4 个围栏无语言标注]、1 张图片 [pinned-GitHub raw SVG，已核查 HTTP 200 image/svg+xml]、0 个表格、17 条唯一外链目标 [16 条尖括号引用 + 1 条图片 URL]，0 条相对链接）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿；草稿箱保持 4 条，无自动暂存草稿被消费）；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 13029/13029 字符。
- 分类 `后端`（CDP 鼠标点击选中）；提交标签 Linux、后端、性能优化（原生 setter 写入 + 选项上合成指针序列提交；CDP 鼠标直接点击选项未落定 chip，改对 `.byte-select-option` 元素本身派发含 hover 的完整指针序列后落定；提交新标签时先前已提交标签可能掉落，Vue 提交竞态后回读 chip 确认，缺失补回；最终 DOM 顺序 Linux、后端、性能优化）。

## 提交

- 提交时刻 2026-10-01 08:39 +08 = 17:39 PDT；弹窗「确定并发布」用页内合成指针事件序列一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7691151105851031590`：暂存 <https://juejin.cn/spost/7691151105851031590>，正式 <https://juejin.cn/post/7691151105851031590>。提交后创作者中心 全部 (68) / 已发布 (67) / 审核中 (1) / 未通过 (0)：进入审核期，/spost/ 暂存（期间 SPA 壳在 /post/ 上返回 200，curl 不可作为发布信号，沿用既有结论）。

## 提交结果与判定

- 登录浏览器轮询：约 09:06 +08 创作者中心翻为 已发布 (68) / 审核中 (0)，正式地址 <https://juejin.cn/post/7691151105851031590> 生效（审核间隔约 27 分钟，与 45-scx-nest 等历史案例一致，无界）。
- 记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7691151105851031590，登录浏览器）

- 标题逐字「eBPF 开发实践教程：示例 32 - 结合 On-CPU 和 Off-CPU 分析的挂钟时间分析」（页内出现次数 1）；正文单份（开头句与尾句各 1 次）。
- 9 个 H2 / 3 个 H3 / 0 个 H4；6 个代码块（`pre code` 语言类统计：2 c、4 bash，与源文一致）；1 张图片（pinned-GitHub SVG 正常渲染、非零尺寸）；0 个表格。
- 17 条外链目标全部被掘金改写为 `link.juejin.cn?target=…`（16 条唯一目标，与源文一致），正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数（09:06 +08 检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7691151105851031590`（`confirmed`，置顶），`last_checked` → 2026-09-30；notes 记录 /spost/ 暂存、09:06 +08 检查点 已发布 (68) / 审核中 (0)、curl 不可信结论、QA 摘要（56 条 confirmed）。
- `sources.json`：`last_checked` → 2026-09-30（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-09-30；`## Juejin` 表格首行新增该条目；补记 2026-09-30 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-30；掘金未映射 54 → 53（55/108）、滚动队列 24 → 23（24 Zhihu and 23 Juejin tasks）；掘金状态行置顶 32-wallclock-profiler（沿列 33-funclatency、34-syscall、35-user-ringbuf、37-uprobe-rust、38-btf-uprobe、41-xdp-tcpdump 不变）；知乎计数不变。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-30；补发缺口段落补 09-30 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（掘金 55，映射 55/108）；剩余队列掘金 24 → 23、总计 48 → 47；第 97 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（55/108）；知乎各行（第 57/61/65/69/75/78/85/88/90/92/94 等）保持 `阻塞`（实测 /creator 仍重定向 /signin，无 z_c0）。
- `community-feedback.md`：`### 2026-09-30` 新检查点（置于 `### 2026-09-29` 之前）：发布记录、/spost/ 暂存与 09:06 +08 清除、早期计数 0 展现 / 1 阅读、33-funclatency 3 / 5、34-syscall 8 / 8、35-user-ringbuf 8 / 11、37-uprobe-rust 28 / 17、38-btf-uprobe 13 / 15、41-xdp-tcpdump 4537 / 51 / 1 收藏、全部可见行仍 0 评论、下一检查点在 31-goroutine 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式/暂存地址、提交时刻与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 55/108 映射、53 未发布、56 confirmed；exit 0）。

## 编辑器经验

- 标签 chip 落定：CDP 鼠标在选项中心 move/down/up 不触发 Vue 提交（选项 hover 态可见但 chip 不落定、输入不清空）；改为对可见 `.byte-select-option` 元素本身派发含 hover 的完整合成指针序列（pointerover/enter/mouseover/enter → pointerdown/mousedown(+focus) → pointerup/mouseup → click）后一次落定。选项列表不在 `.publish-popup` 内（弹窗作用域查不到），需全页查 `.byte-select-option` 并按宽度 > 0 过滤。
- 提交「确定并发布」：对按钮直接派发页内合成指针序列即得 `发布成功`（本次面板已展开，无需重开；若面板零矩形则对工具栏 `button.xitu-btn` 发布按钮重放含 hover 指针序列展开面板）。
- 本次审核间隔约 27 分钟（08:39 +08 提交 → 09:06 +08 清除）：SPA 壳 200 结论沿用，curl 状态码不作为发布信号，判定以登录浏览器创作者中心 审核中 计数为准。
