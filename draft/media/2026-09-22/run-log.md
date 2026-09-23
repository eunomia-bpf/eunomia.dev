# 2026-09-22 内容巡检

- 运行模式：定时巡检（eunomia-content-patrol），LA 自然日 2026-09-22。
- 调用子技能：eunomia-social-radar、juejin-publisher。
- 雷达（22:15-22:35 PDT，可见 Chrome，CDP 9222，持久 Yunwei profile）：8 篇受跟踪掘金文章均公开，无 `审核中` / `文章有更新` / `已被删除` 标记，评论区均为 `暂无评论数据`（0 评论）。阅读数：44-scx-simple 45、45-scx-nest 33、46-xdp-test 25、ACRFence 52、Agent Sandbox 266、47-cuda-events 54、48-energy 33、Runtime Security 35；读数包含巡检自身访问，不与上一基线严格可比。
- 发布：掘金 43-kfuncs 教程（`docs/tutorials/43-kfuncs/README.zh.md`），2026-09-22 使用 09-22 正常额度，通过可见编辑器导入 `draft/media/2026-09-22/43-kfuncs/juejin-body.md`（11047 字符）并提交；分类 `后端`，标签 `Linux`、`后端`、`性能优化`。提交页返回“发布成功”（post `7688489073967251494`，JSON-LD datePublished 2026-09-23T13:29:32+08:00 = 22:29 PDT），个人主页文章列表顶部可见新文（`11分钟前`，0 赞 / 0 评），但文章仍在审核中，公开 `/post/` URL 返回“找不到页面”。暂存页 <https://juejin.cn/spost/7688489073967251494> 结构 QA 通过：原标题、正文单份 11047 字符、9 个 H2、11 个 H3、3 个 H4、20 个渲染代码块（与源文 20 个围栏一一对应：2 c、2 makefile、10 bash、2 sh、4 txt，含有序列表内 7 个缩进围栏，无渲染拆分、无重复）、9 条外链、1 张外链图片（raw.githubusercontent 直连 800×500，无 `转存失败` 标记）。
- 记录：ledger 记为 `review_pending`（`.github/publisher/media/platforms/juejin.json` 新增 published + browser_observations 各 1 条，`last_checked` → 2026-09-22）；`draft/plan/publishing-queue.zh.md` 43-kfuncs 行由 `排队` 改 `阻塞` 并附恢复条件（审核通过后 `/spost/` 跳转公开 `/post/7688489073967251494`，届时公开页 QA 通过改 `[x]` 且 ledger 改 `confirmed`）；发布稿记录见 `draft/media/2026-09-22/43-kfuncs/juejin.md`。
- 额度：掘金每日上限为 1 篇 / LA 自然日，09-18…09-21 无发布日的 4 天空缺不产生补发额度；4 条补发缺口（2026-08-29、2026-09-01、2026-09-03、2026-09-16）保留到后续可用日。
- 阻塞：知乎 `/creator` 仍跳转 `/signin`，可见会话无 `z_c0`，24 条知乎任务继续阻塞，补发额度未核销。
- 经验教训：写入 `juejin-publisher` 技能——`确定并发布` 按钮不响应合成 `element.click()` 与 CLI `agent-browser click`（仅触发自动保存 toast），需经 `eval` 派发完整真实指针事件序列；标签搜索框 `.byte-select__input` 无法经 CLI `fill` / `keyboard type` 到达（输入泄漏进标题/正文），需原生 value setter + `input` 事件再点选 `.byte-select-option`；提交前回读并复位标题输入框（早期标签输入可能污染标题）。
- 下一步：下一次巡检复查 `/post/7688489073967251494`，审核通过并完成公开页 QA 后把队列行改 `[x]`、ledger 改 `confirmed`；未确认前不得重复提交同一源文。
