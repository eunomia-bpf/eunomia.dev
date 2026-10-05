# 掘金发布稿：24-hide 教程

- 状态：已发布（LA 2026-10-04 正常额度；提交于 2026-10-05 07:41 +08 = 16:41 PDT，审核期 /spost/ 暂存、08:35 +08 清除，公开页 QA 通过）
- 正文：`juejin-body.md`（源文 `docs/tutorials/24-hide/README.zh.md` 移除源 H1，其余正文逐字保留；15109 字符、21711 字节、4 个 H2、1 个 H3、0 个 H4、14 个代码块 [9 c、1 bash、1 sh、3 console]、0 张图片、0 个表格；3 条唯一外链目标 [github eunomia-bpf/bpf-developer-tutorial 仓库根、tree main/src/24-hide、eunomia.dev/zh/tutorials]，0 条相对链接）
- 标题：eBPF 开发实践：使用 eBPF 隐藏进程或文件信息（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/24-hide/
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 27-replace / 28-detach / 26-sudo 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言为 `c`（×9，含 getdents64 内核态 enter/exit/patch 与用户态 skeleton/open/load/poll/handle_event 完整实现）、`bash`（×1，make）、`sh`（×1，sudo ./pidhide 运行命令）、`console`（×3，ps 与 pidhide 输出）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-04）

- 正式地址：<https://juejin.cn/post/7692741078310912027>；审核期暂存地址：<https://juejin.cn/spost/7692741078310912027>
- 提交时刻：2026-10-05 07:41 +08 = 2026-10-04 16:41 PDT；新编辑器会话（`/editor/drafts/new`），无预置草稿；编辑器草稿箱为 5 条（未被本次提交消费）
- 流程：标题经原生 setter 写入，正文经 base64 分块注入 CodeMirror（15109 字符与本地一致），单次弹窗提交「确定并发布」返回「发布成功」；进入审核期（`/spost/` 暂存、创作者中心 审核中 (1)），2026-10-05 08:35 +08 创作者中心翻为 已发布 (74) / 审核中 (0)，审核间隔约 54 分钟；审核期间 SPA 壳对 `/post/` 一律回 200，故以登录态创作者中心与公开页渲染为准
- 公开页 QA：精确标题、正文单份、4 H2 / 1 H3 / 0 H4、14 个代码块（9 c、1 bash、1 sh、3 console）、0 张内容图、0 表格、3 条唯一外链全部改写为 `link.juejin.cn?target=`、无「审核中 / 文章有更新 / 已被删除」标记、0 评论；发布后早期计数（08:40 +08 检查点）0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏
- Ledger：`platforms/juejin.json` 新增 `confirmed` 条目 `juejin-7692741078310912027`（置顶，带 `staged_url`），`sources.json`/`juejin.json` `last_checked` 与 `published.md`「Last checked」更新为 2026-10-04；`not-published.md` 更新为 47 on Juejin（61/108 已映射）；queue 第 107 行转 `[x]`、掘金 17 条 / 共 41 个平台任务；`community-feedback.md` `### 2026-10-04` 检查点新增发布与 QA 两条
