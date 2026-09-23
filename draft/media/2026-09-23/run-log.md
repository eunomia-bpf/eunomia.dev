# 2026-09-23 内容巡检发布记录（America/Los_Angeles 自然日）

- 运行模式：eunomia-content-patrol 定时巡检。任务日 2026-09-23，LA 时区 03:28 PDT 起执行（10:28 UTC）。
- 前置核对：持久可见 Chrome CDP 9222 存活（`Chrome/146.0.7680.31`）；仓库 `main` 干净于上一轮提交，`git fetch` 后无新远端提交。

## 43-kfuncs 审核复查（不消耗新额度）

- 09-22 以 09-22 正常额度提交的 `docs/tutorials/43-kfuncs/README.zh.md` 审核通过：`/post/7688489073967251494` 由“找不到页面”变为公开页。
- 公开页 QA：原标题逐字一致；正文单份；9 个 H2、11 个 H3、3 个 H4、20 个 `<pre>`；8 条唯一外链（10 个锚点，2 个目标重复：bpf-developer-tutorial、eunomia.dev/tutorials 各出现两次，与源文一致）；1 张正文图片 `raw.githubusercontent.com/eunomia-bpf/code-survey/main/imgs/cumulative_helper_kfunc_timeline.png` 完整加载（`naturalWidth×Height = 800×500`，`complete=true`，无转存失败标记，首轮 `0x0` 为懒加载未触发）。无 `审核中`/`文章有更新`/`已被删除` 标记，评论空态。ledger 由 `review_pending` 改为 `confirmed`。
- 队列行由 `[ ] 阻塞` 改为 `[x] 排队`，附公开 URL 与 QA 摘要。

## 42-xdp-loadbalancer 发布（09-23 正常额度）

- 源文：`docs/tutorials/42-xdp-loadbalancer/README.zh.md`（23326 字节，H1 `# eBPF 开发者教程： 简单的 XDP 负载均衡器`）。发布稿只移除源 H1，正文其余保持不变。
- 本地 artifact：`draft/media/2026-09-23/42-xdp-loadbalancer/juejin-body.md`（16487 字符、559 行、5 个 H2、19 个 H3、2 个 H4、25 个代码块 [12 c + 8 sh + 4 console + 1 txt]、0 图片、0 表格、4 条唯一外链、无相对链接）与 `juejin.md`。
- 提交路径：可见编辑器 `https://juejin.cn/editor/drafts/new`，原生 setter 写入标题与正文（正文经 base64 stdin 注入，回读 16487 字符一致），分类 `后端`，标签 `Linux`、`后端`、`性能优化`（逐枚回读 `.byte-select__tag` 确认）。
- `确定并发布` 按钮沿用已记录的真实指针序列（合成 `click()` 仅触发自动保存）；提交页返回“发布成功”（`https://juejin.cn/published`，`document.title = 发布成功`）。
- 结果：**直接公开**，`https://juejin.cn/post/7688532185121947658`（无暂存审核间隔）。公开页 QA：原标题逐字一致、正文单份、5 个 H2、19 个 H3、2 个 H4、25 个 `<pre>`、4 条外链、0 正文图片、分类后端、三标签正常、无审核/更新/删除标记、评论空态（0 条）。ledger 记为 `confirmed`。

## 雷达巡检（10 篇公开掘金文章）

- 42-xdp-loadbalancer 1 阅读（5 H2）；43-kfuncs 16（9 H2）；44-scx-simple 49；45-scx-nest 33；46-xdp-test 26；ACRFence 53；Agent Sandbox 266；47-cuda-events 56；48-energy 34；Runtime Security 36。
- 全部 10 篇均无 `审核中`/`文章有更新`/`已被删除` 标记，评论均为 `暂无评论数据` 空态（0 条），无需回复或更正。
- 已把 2026-09-23 checkpoint 追加到 `.github/publisher/media/community-feedback.md`（append-only，CRLF 保持）。

## 其他

- 知乎 `/creator` 仍重定向 `https://www.zhihu.com/signin?next=%2Fcreator`、cookie 无 `z_c0`，24 条知乎任务保持 `阻塞`。
- 补发缺口保持 4 条（2026-08-29、2026-09-01、2026-09-03、2026-09-16），本日未核销（知乎仍阻塞）。
- 队列更新：头部日期、补发缺口、Ledger 基线（掘金 46/107）、剩余队列（知乎 24、掘金 32、共 56）、43-kfuncs 与 42-xdp-loadbalancer 两行。
- ledger checker `.github/publisher/media/check_media_ledger.py` 退出码 0；掘金 47 条 `confirmed`、0 条 `review_pending`。
