# 掘金发布稿：19-lsm-connect 教程

- 状态：已发布（LA 2026-10-08 正常额度；审核间隔约 56 分钟，08:03 +08 创作者中心轮询翻为已发布）
- 正文：`juejin-body.md`（源文 `docs/tutorials/19-lsm-connect/README.zh.md` 移除源 H1，其余正文逐字保留；4812 字符、7042 字节、7 个 H2、0 个 H3、0 个 H4、9 个代码块 [6 console：`/boot/config` 内核配置 ×1、`/sys/kernel/security/lsm` ×1、docker run ×1、ecc 编译输出 ×1、ping/curl/wget 与 trace_pipe 输出 ×2；1 conf：GRUB_CMDLINE_LINUX lsm= 行 ×1；1 C：lsm-connect.bpf.c 完整内核态程序 ×1；1 shell：sudo ecli run 命令 ×1]、0 张图片、0 个表格；7 条唯一外链目标 [aya-rs.dev LSM 文档、eunomia.dev/zh/tutorials/、git.kernel.org bpf_tracing.h、github eunomia-bpf/bpf-developer-tutorial（×2，含 src/19-lsm-connect 子目录）、github leodido/demo-cloud-native-ebpf-day、github torvalds/linux lsm_hooks.h]，0 条相对链接）
- 标题：eBPF 入门实践教程：使用 LSM 进行安全检测防御（源 H1 逐字，无系列编号）
- 原文：https://eunomia.dev/zh/tutorials/19-lsm-connect/
- 代码仓库：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/19-lsm-connect
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 21-xdp / 22-android / 24-hide / 26-sudo / 20-tc 教程标签组；以实际落定 chip 为准）
- 代码块说明：源文代码围栏语言为 `console`（×6，内核配置检查、`/sys/kernel/security/lsm`、docker run、ecc 编译、ping/curl/wget、trace_pipe 输出）、`conf`（×1，GRUB 命令行 lsm= 行）、`C`（×1，lsm-connect.bpf.c 完整程序）与 `shell`（×1，sudo ecli run 命令）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签

## 发布结果（LA 2026-10-08）

- 提交：2026-10-09 07:07 +08（16:07 PDT），新编辑器会话、无预置草稿，单次弹窗提交，返回"发布成功"，落地 `https://juejin.cn/published`；提交前创作者中心 全部 77 / 已发布 77 / 审核中 0 / 未通过 0。
- 审核间隔：进入审核期，`/spost/7694131662616428571` 暂存（创作者中心 审核中 (1)）；08:03 +08 轮询翻为 已发布 (78) / 审核中 (0)，审核间隔约 56 分钟。
- 正式地址：<https://juejin.cn/post/7694131662616428571>（暂存 `<https://juejin.cn/spost/7694131662616428571>`）
- 公开页 QA：精确标题、正文单份（4812 字符）、7 H2 / 0 H3 / 0 H4、9 个代码块 [6 console、1 conf、1 C、1 shell]、0 张图片、0 个表格、7 条外链目标全部改写为 `link.juejin.cn?target=`、0 审核中/更新/删除 标记、0 评论（`暂无评论数据`）。
- 早期计数器（08:03 +08 创作者中心放行检查点）：0 展现 · 0 阅读 · 0 点赞 · 0 评论 · 0 收藏；数分钟后逐行复核时新文已读 1 阅读。
- ledger：`platforms/juejin.json` 已 `confirmed`（带 `staged_url`）；`last_checked` → 2026-10-08；无补发缺口，无未偿缺口。
