# 掘金发布稿：21-xdp 教程

- 状态：已发布（LA 2026-10-06 正常额度；提交于 2026-10-07 07:43 +08 = 16:43 PDT，审核期 /spost/ 暂存、08:07 +08 清除，公开页 QA 通过）
- 正文：`juejin-body.md`（源文 `docs/tutorials/21-xdp/README.zh.md` 移除源 H1，其余正文逐字保留；5224 字符、9726 字节、7 个 H2、6 个 H3、0 个 H4、6 个代码块 [2 C：xdp_pass 完整程序 ×1、挂载注释 ×1；4 console：docker run ×1、ecc 编译输出 ×1、ecli 运行命令 ×1、trace_pipe 输出 ×1]、0 张图片、0 个表格；10 条唯一外链目标 [github eunomia-bpf/bpf-developer-tutorial、eunomia.dev/zh/tutorials、cilium.io、github facebookincubator/katran、blog.cloudflare.com l4drop、github xdp-project/xdp-tutorial basic01-xdp-pass、patchwork.kernel.org 20220120061422 patch、prototype-kernel.readthedocs.io xdp_actions、arthurchiao.art xdp-paper-acm-2018-zh、arthurchiao.art linux-net-stack-implementation-rx-zh]，0 条相对链接）
- 标题：eBPF 入门实践教程二十一： 使用 XDP 进行可编程数据包处理（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/21-xdp/
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 22-android / 24-hide / 26-sudo 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言为 `C`（×2，xdp_pass 完整内核态程序与挂载注释）与 `console`（×4，docker run、ecc 编译、ecli 运行、trace_pipe 输出）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-06）

- 正式地址：<https://juejin.cn/post/7693018382872330278>；审核期暂存地址：<https://juejin.cn/spost/7693018382872330278>
- 提交时刻：2026-10-07 07:43 +08 = 2026-10-06 16:43 PDT；新编辑器会话（`/editor/drafts/new`），无预置草稿
- 流程：标题经原生 setter 写入，正文经 base64 分块注入 CodeMirror（5224 字符与本地一致），单次弹窗提交「确定并发布」返回「发布成功」；进入审核期（`/spost/` 暂存、创作者中心 审核中 (1)），2026-10-07 08:07 +08 创作者中心翻为 已发布 (76) / 审核中 (0)，审核间隔约 24 分钟；审核期间 SPA 壳对 `/post/` 一律回 200，故以登录态创作者中心与公开页渲染为准
- 公开页 QA：精确标题、正文单份、7 H2 / 6 H3 / 0 H4、6 个代码块 [2 C、4 console]、0 张内容图、0 表格、10 条唯一外链全部改写为 `link.juejin.cn?target=`、无「审核中 / 文章有更新 / 已被删除」标记、0 评论；发布后早期计数（08:51 +08 检查点）0 展现 / 2 阅读 / 0 点赞 / 0 评论 / 0 收藏
- Ledger：`platforms/juejin.json` 新增 `confirmed` 条目 `juejin-7693018382872330278`（置顶，带 `staged_url`），`sources.json`/`juejin.json` `last_checked` 与 `published.md`「Last checked」更新为 2026-10-06；`not-published.md` 更新为 45 on Juejin（63/108 已映射）；queue 第 109 行转 `[x]`、掘金 15 条 / 共 39 个平台任务；`community-feedback.md` `### 2026-10-06` 检查点新增发布、QA 与跟踪 top ten 三条

