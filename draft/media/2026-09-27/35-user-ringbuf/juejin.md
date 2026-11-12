# 掘金发布稿：35-user-ringbuf 教程

- 状态：已发布。LA 2026-09-27 正常额度；提交 2026-09-28 07:08 +08（16:08 PDT），经可见编辑器单次弹窗提交（新编辑器、无预置草稿，草稿箱保持 4 条、无自动暂存草稿被消费），返回「发布成功」。审核期暂存 https://juejin.cn/spost/7689742195037110323（创作者中心 审核中 (1)，SPA 壳在 /post/ 上返回 200 不作为发布信号），约 08:13 +08 创作者中心转为 已发布 (65) / 审核中 (0) 后正式地址生效。
- 正式地址：https://juejin.cn/post/7689742195037110323（台账 `platforms/juejin.json` 条目 `juejin-7689742195037110323`，状态 confirmed，52/108）
- 正文：`juejin-body.md`（源文 `docs/tutorials/35-user-ringbuf/README.zh.md` 移除源 H1，其余正文逐字保留；7230 字符、11514 字节、4 个 H2、4 个 H3、0 个 H4、12 个围栏 [6 个代码块：4 c、1 sh、1 console]、0 张图片、0 个表格；8 条唯一外链 [eunomia.dev/tutorials/ ×2、eunomia.dev/tutorials/11-bootstrap/ ×1、eunomia.dev/zh/tutorials/ ×2、eunomia.dev/zh/tutorials/35-user-ringbuf/ ×1、github bpf-developer-tutorial ×3 [含仓库根 ×2、src/35-user-ringbuf ×1]、bpftime ×1、lwn.net ×2]，无相对链接）
- 标题：eBPF开发实践：使用 user ring buffer 向内核异步发送信息（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/35-user-ringbuf/
- 代码：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/35-user-ringbuf
- 分类：后端
- 标签：提交时实际落定 chip 为 `Linux`、`后端`、`性能优化`（DOM 顺序；沿用 2026-09-12 以来 eBPF 教程标签组）
- 公开页 QA：通过（标题逐字单份、4 H2 / 4 H3 / 0 H4、6 个代码块 [4 c、1 sh、1 console]、0 图片、0 表格、8 条唯一外链 [10 个链接] 全部改写为 `link.juejin.cn?target=`、无审核/更新/删除标记、0 评论；早期计数 0 展现 / 2 阅读）
- 已核查：`platforms/juejin.json` 无 35-user-ringbuf 条目（52 条 confirmed），源文未被掘金发布过，可首发布
- 代码块说明：源文代码围栏语言为 `c`（×4）、`sh`（×1）、`console`（×1）（保持源文标签）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签
