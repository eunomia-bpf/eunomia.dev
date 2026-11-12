# 掘金发布稿：32-wallclock-profiler 教程

- 状态：已发布（LA 2026-09-30 正常额度；正式地址 <https://juejin.cn/post/7691151105851031590>，暂存地址 <https://juejin.cn/spost/7691151105851031590>；提交 2026-10-01 08:39 +08 = 17:39 PDT，经 /spost/ 暂存进入审核期后于 09:06 +08 翻为 已发布 (68) / 审核中 (0)，无需 /spost/ 轮询；公开页 QA 通过：标题逐字、正文单份、9 H2 / 3 H3 / 0 H4、6 个代码块 [4 bash、2 c]、1 图片（pinned-GitHub SVG 正常渲染）、0 表格、17 条唯一外链目标全部改写为 link.juejin.cn?target=、0 评论；标签 `Linux`、`后端`、`性能优化`；编辑器草稿箱保持 4 条、无自动暂存草稿消费）
- 正文：`juejin-body.md`（源文 `docs/tutorials/32-wallclock-profiler/README.zh.md` 移除源 H1，其余正文逐字保留；13029 字符、20963 字节、9 个 H2、3 个 H3、0 个 H4、10 个围栏 [6 个代码块：4 bash、2 c]、1 张图片（pinned-GitHub SVG，已核查 200 image/svg+xml）、0 个表格；17 条唯一外链目标 [16 条尖括号引用 [github bpf-developer-tutorial 仓库根 ×1、tree main/src/32-wallclock-profiler ×1、iovisor/bcc libbpf-tools ×2、libbpf/blazesym ×1、brendangregg FlameGraph ×1、github eBPF 教程外链、brendangregg offcpuanalysis ×1、brendangregg SIGCOMM'24 slides ×1、usenet/usenix ×4、acm ×2、sigops HotOS'21 ×1、spec ICPE'19 ×1] + 1 条图片 URL]，0 条相对链接）
- 标题：eBPF 开发实践教程：示例 32 - 结合 On-CPU 和 Off-CPU 分析的挂钟时间分析（源 H1 逐字）
- 原文：https://eunomia.dev/zh/tutorials/32-wallclock-profiler/（源文末尾「本文原文链接」页脚）
- 代码：https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/32-wallclock-profiler
- 分类：后端
- 标签：`Linux`、`后端`、`性能优化`（沿用 2026-09-12 以来 eBPF 教程标签组；以实际落定 chip 为准）
- 图片：1 张 pinned-commit GitHub raw SVG（https://raw.githubusercontent.com/eunomia-bpf/bpf-developer-tutorial/c9d3d65c15fb6528ee378657a05ec0b062eff5b7/src/32-wallclock-profiler/tests/example.svg），源文原样保留，已核查 HTTP 200 image/svg+xml；发布后于公开页复核渲染（若出现「转存失败」按 skill 处理）
- 已核查：`platforms/juejin.json` 无 32-wallclock-profiler 条目（55 条 confirmed），源文未被掘金发布过，可首发布
- 代码块说明：源文代码围栏语言为 `bash`（×4）与 `c`（×2），其余 4 个围栏无语言标注（保持源文原样）；掘金编辑器按语言类渲染，属平台能力差异，不改源文标签
