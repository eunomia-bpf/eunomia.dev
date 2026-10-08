# 掘金发布稿：20-tc 教程

- 状态：已发布（LA 2026-10-07 正常额度；提交于 2026-10-08 07:22 +08 = 16:22 PDT，直接公开无审核间隔，公开页 QA 通过）
- 正文：`juejin-body.md`（源文 `docs/tutorials/20-tc/README.zh.md` 移除源 H1，其余正文逐字保留；3621 字符、5531 字节、6 个 H2、0 个 H3、0 个 H4、6 个代码块 [2 C：tc_ingress 完整程序 ×1、挂载注释 ×1；3 console：docker run ×1、ecc 编译输出 ×1、trace_pipe 输出 ×1；1 shell：ecli 运行命令 ×1]、0 张图片、0 个表格；5 条唯一外链目标 [just4coding.com tc 博客、arthurchiao.art tc-da-mode-zh 博客、patchwork.kernel.org 20210512103451 patch、github eunomia-bpf/bpf-developer-tutorial、eunomia.dev/zh/tutorials]，0 条相对链接）
- 标题：eBPF 入门实践教程二十：使用 eBPF 进行 tc 流量控制（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/20-tc/
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 21-xdp / 22-android / 24-hide 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言为 `c`（×2，tc_ingress 完整内核态程序与挂载注释）、`console`（×3，docker run、ecc 编译输出、trace_pipe 输出）与 `shell`（×1，ecli 运行命令）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-07）

- 正式地址：<https://juejin.cn/post/7693579496968831022>（直接公开，无审核期暂存地址）
- 提交时刻：2026-10-08 07:22 +08 = 2026-10-07 16:22 PDT；新编辑器会话（`/editor/drafts/new`），无预置草稿；提交前创作者中心 全部 76 / 已发布 76 / 审核中 0 / 未通过 0
- 流程：标题经原生 setter + CodeMirror 写入，正文经 base64 分块注入 CodeMirror（3621 字符与本地一致，`cm.getValue().length == 3621` 验证），分类 `后端`（真实 CDP 点击），标签 Linux、后端、性能优化（原生 setter + 选项点击，DOM 顺序 Linux→后端→性能优化），提交前逐字复核标题；单次弹窗「确定并发布」返回「发布成功」
- 清除判定：直接公开、无审核间隔——提交后创作者中心 07:34 +08 检查点已读 全部 77 / 已发布 77 / 审核中 0 / 未通过 0（区别于 21-xdp 等审核期巡检，与 26-sudo 先例一致）；期间 SPA 壳对 `/post/` 一律回 200，判定以登录态创作者中心为准
- 公开页 QA：精确标题、正文单份（3621 字符，开头与摘要各出现 1 次）、6 个 H2 / 0 个 H3 / 0 个 H4、6 个代码块 [2 c、3 console、1 shell]、0 张内容图、0 表格、5 条唯一外链全部改写为 `link.juejin.cn?target=`、无「审核中 / 文章有更新 / 已被删除」标记、0 评论（`暂无评论数据` 空态）；发布后早期计数（07:34 +08 检查点）0 展现 / 2 阅读 / 0 点赞 / 0 评论 / 0 收藏
- Ledger：`platforms/juejin.json` 置顶新增 `confirmed` 条目 `juejin-7693579496968831022`（直接公开，无 `staged_url`，沿 26-sudo 先例），`published[]` 64 → 65；`sources.json` / `juejin.json` `last_checked` 与 `published.md`「Last checked」更新为 2026-10-07；`not-published.md` 更新为 44 on Juejin（64/108 已映射）、滚动队列 14 Juejin / 共 38 个平台任务；queue 第 110 行翻 `[x]`；`community-feedback.md` 新增 `### 2026-10-07` 检查点
