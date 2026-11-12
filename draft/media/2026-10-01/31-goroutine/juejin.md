# 掘金发布稿：31-goroutine 教程

- 状态：已发布
- 正文：`juejin-body.md`（源文 `docs/tutorials/31-goroutine/README.zh.md` 移除源 H1，其余正文逐字保留；3823 字符、6797 字节、119 行、3 个 H2、3 个 H3、0 个 H4、10 个围栏 [5 个代码块：1 c、3 bash、1 console，其余 4 个围栏无语言标注]、0 张图片、0 个表格；4 条唯一外链目标 [eunomia.dev/tutorials/、github bpf-developer-tutorial 仓库根、tree main/src/31-goroutine、github bpftime]，0 条相对链接；源文末尾无「本文原文链接」页脚，正文止于结论段）
- 标题：eBPF 实践教程：使用 eBPF 跟踪 Go 协程状态（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/31-goroutine/
- 分类：后端；标签：Linux、后端、性能优化（按 48-energy / 38-btf-uprobe 等近期教程先例：48-energy 仅 Linux/后端/性能优化 可选，eBPF 与 开源 不在候选列表）
- 代码块说明：源文代码围栏语言为 `c`（×1）、`bash`（×3）与 `console`（×1），其余 4 个围栏为缩进输出/说明围栏（保持源文原样）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-01）

- 正式地址：<https://juejin.cn/post/7691326130511167526>；审核期暂存地址：<https://juejin.cn/spost/7691326130511167526>
- 提交时刻：2026-10-02 07:25 +08 = 2026-10-01 16:25 PDT；新编辑器会话（`/editor/drafts/new`），无预置草稿；编辑器草稿箱为 5 条（上次中断会话留有 1 条自动暂存草稿，未被本次提交消费）
- 流程：标题经原生 setter 写入，正文经 base64 分块注入 CodeMirror（3823 字符与本地一致），单次弹窗提交「确定并发布」返回「发布成功」；进入审核期（`/spost/` 暂存、创作者中心 审核中 (1)），2026-10-02 08:18 +08 创作者中心翻为 已发布 (69) / 审核中 (0)，审核间隔约 53 分钟；审核期间 SPA 壳对 `/post/` 一律回 200，故以登录态创作者中心与公开页渲染为准
- 公开页 QA：精确标题、正文单份、3 H2 / 3 H3 / 0 H4、5 个代码块（1 c、3 bash、1 console）、0 张内容图、0 表格、4 条唯一外链全部改写为 `link.juejin.cn?target=`、无「审核中 / 文章有更新 / 已被删除」标记、0 评论；发布后早期计数 1 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏
- Ledger：`platforms/juejin.json` 新增 `confirmed` 条目 `juejin-7691326130511167526`（置顶），`sources.json`/`juejin.json` `last_checked` 与 `published.md`「Last checked」更新为 2026-10-01；`not-published.md` 更新为 52 on Juejin（56/108 已映射）；queue 第 98 行转 `[x]`、掘金 22 条 / 共 46 个平台任务；`community-feedback.md` 新增 `### 2026-10-01` 检查点
