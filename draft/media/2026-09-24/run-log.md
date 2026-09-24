# 2026-09-24 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-24。09-23 的掘金额度已被 42-xdp-loadbalancer 占用，同日不得再发同平台一篇，因此 41-xdp-tcpdump 顺延至 LA 09-24 的 00:00 PDT 之后提交（实际提交时刻 2026-09-24 07:10 UTC / 00:10 PDT，信息流显示北京时间 2026-09-24 15:10）。

## 队列任务：掘金 41-xdp-tcpdump

- 任务行：`draft/plan/publishing-queue.zh.md` 第 84 行，原状态 `排队`。
- 源文：`docs/tutorials/41-xdp-tcpdump/README.zh.md`；发布稿 `draft/media/2026-09-24/41-xdp-tcpdump/juejin-body.md`（移除源 H1，其余正文逐字保留；11793 字符）。
- 标题：eBPF 示例教程：使用 XDP 捕获 TCP 信息（源 H1 逐字，保留全角冒号）。
- 分类 `后端`；标签 `Linux`、`后端`、`性能优化`。
- 提交结果：可见编辑器提交页返回“发布成功”。文章先落在暂存地址 <https://juejin.cn/spost/7688905828694294566>（创作者中心显示为唯一 `审核中` 条目），同会话内审核通过，正式地址 <https://juejin.cn/post/7688905828694294566> 生效，`/spost/` 随后 404。
- 公开页 QA：标题逐字、正文单份、5 个 H2、9 个 H3、9 个 H4、17 个代码块（12 个 `c`、3 个 `bash`、2 个源文裸围栏示例输出块）、0 张正文图片、0 个表格、3 条唯一外链目标、无 `审核中`/`文章有更新`/`已被删除` 标记、评论区为空。作者主页将其列为最新一篇。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7688905828694294566`（`confirmed`），`last_checked` → 2026-09-24，已发布条目 47 → 48。
- `published.md`：`Last checked:` → 2026-09-24；`## Juejin` 表格首行新增该条目；补记 2026-09-24 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-24；掘金未映射 61 → 60、已映射 46/107 → 47/107；滚动队列 32 → 31；掘金状态行置顶该发布。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-24；Ledger 基线掘金 63 → 62、映射 46/107 → 47/107；剩余队列掘金 32 → 31、总计 56 → 55；第 84 行翻为 `[x]` 并附正式地址与 QA 摘要；补发缺口保持 4 条（2026-08-29、2026-09-01、2026-09-03、2026-09-16）。

## 编辑器经验（已回写技能）

- 分类 chip：合成 hover/pointer 事件序列本次仍未生效（`.category-list .item.active` 返回空），必须使用真实 CDP `page.mouse` 移动/按下/抬起；一次即成。
- 标签输入：`.publish-popup .byte-select__input` index 0 需用原生 value setter + 冒泡 `input` 事件写入，再点击文档级可见 `.byte-select-option`；CLI `keyboard type` 会把文本泄漏到标题/正文。
- 提交按钮：`确定并发布` 只需真实 CDP 指针序列（move → move → down → up），本次一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 审核不是稳定态：本次 `/spost/` 暂存仅维持数分钟，同会话内即换为 `/post/`。`/post/` 返回 200 且无 `审核中` 标记即可直接记 `confirmed`，不必按 09-22 的 `/spost/` 惯例先记 `review_pending`。
