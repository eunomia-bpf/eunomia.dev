# 掘金发布稿：22-android 教程

- 状态：已发布（LA 2026-10-05 正常额度；提交于 2026-10-06 07:45 +08 = 16:45 PDT，审核期 /spost/ 暂存、09:09 +08 清除，公开页 QA 通过）
- 正文：`juejin-body.md`（源文 `docs/tutorials/22-android/README.zh.md` 移除源 H1，其余正文逐字保留；9228 字符、11684 字节、7 个 H2、2 个 H3、3 个 H4、4 个代码块 [4 console：bootstrap 运行输出 ×1、tcpstates 运行输出 ×2、opensnoop 报错输出 ×1]、0 张图片、0 个表格；14 条唯一外链目标 [seeflower 博客 ×2、kanxue、github eunomia-bpf 仓库 ×4、tiann/eadb、eunomia.dev/zh/tutorials、eunomia.dev 构建文档、source.android.google.cn BPF 文档、mp.weixin.qq.com 微信文章]，0 条相对链接；`## 参考` 段含 3 条 `[^X]:<url>` footnote 定义，逐字保留）
- 标题：在 Android 上使用 eBPF 程序（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/22-android/
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 24-hide / 26-sudo / 27-replace 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言全部为 `console`（bootstrap / tcpstates 的 eunomia-bpf 运行输出与 opensnoop 的 libbpf 报错输出），掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-05）

- 正式地址：<https://juejin.cn/post/7692860578532933642>；审核期暂存地址：<https://juejin.cn/spost/7692860578532933642>
- 提交时刻：2026-10-06 07:45 +08 = 2026-10-05 16:45 PDT；新编辑器会话（`/editor/drafts/new`），无预置草稿；编辑器草稿箱为 5 条（未被本次提交消费）
- 流程：标题经原生 setter 写入，正文经 base64 分块注入 CodeMirror（9228 字符与本地一致），单次弹窗提交「确定并发布」返回「发布成功」；进入审核期（`/spost/` 暂存、创作者中心 审核中 (1)），2026-10-06 09:09 +08 创作者中心翻为 已发布 (75) / 审核中 (0)，审核间隔约 84 分钟；审核期间 SPA 壳对 `/post/` 一律回 200，故以登录态创作者中心与公开页渲染为准
- 公开页 QA：精确标题、正文单份、7 H2 / 2 H3 / 3 H4、4 个代码块 [4 console]、0 张内容图、0 表格、14 条唯一外链全部改写为 `link.juejin.cn?target=`、`## 参考` 3 条 [^X] footnote 定义逐字保留、无「审核中 / 文章有更新 / 已被删除」标记、0 评论；发布后早期计数（09:10 +08 检查点）0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏
- Ledger：`platforms/juejin.json` 新增 `confirmed` 条目 `juejin-7692860578532933642`（置顶，带 `staged_url`），`sources.json`/`juejin.json` `last_checked` 与 `published.md`「Last checked」更新为 2026-10-05；`not-published.md` 更新为 46 on Juejin（62/108 已映射）；queue 第 108 行转 `[x]`、掘金 16 条 / 共 40 个平台任务；`community-feedback.md` `### 2026-10-05` 检查点新增发布与 QA 两条
