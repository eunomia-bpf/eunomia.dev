# 掘金发布稿：xpu/flamegraph

- 状态：已确认公开（LA 2026-10-19 正常额度，队列 L130；post `7698287001586040872`）
- 提交：2026-10-20 08:44 +08（17:44 PDT），全新 `/editor/drafts/new` 会话、无预置草稿，单次弹窗提交返回「发布成功」，落地 https://juejin.cn/published
- 提交前创作者中心：全部 88 / 已发布 88 / 审核中 0 / 未通过 0；提交后：全部 89 / 已发布 88 / 审核中 1 / 未通过 0
- 审核间隔：约 17 分钟（快于 dynptr 的 23 分钟与 struct_ops 的 49 分钟先例，慢于 bpf_wq / bpf_arena / bpf_iters / bpf_token 的 1–5 分钟快档）——`/spost/7698287001586040872` 暂存、SPA 在 `/post/` 上答 HTTP 200，09:01 +08 前后创作者中心轮询翻为 全部 89 / 已发布 89 / 审核中 0，规范页 `/post/7698287001586040872` 上线
- 分类：后端；标签 chips `Linux`、`后端`、`性能优化`（DOM 序；每个 chip 经 native-setter 搜索 + 真实 CDP 指针点击 `.byte-select-option`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup / bpf_arena / bpf_iters / bpf_token / bpf_wq / dynptr / struct_ops 教程标签组）
- 公开页 QA（规范 `/post/`，登录浏览器）：标题 H1 ×1（「eBPF 示例：使用 CUPTI 构建 GPU 火焰图分析器」，正文无重复 H1）、正文单份（源 31718 字符 → 渲染 31260）、11 H2 / 17 H3 / 5 H4、25 代码块 [1 c、1 cpp、2 python、18 bash、3 无语言标签 [渲染 25 <pre>，与源 25 个围栏代码块 1:1]]、0 图片、0 表格、11 条唯一外链目标全部改写为 `link.juejin.cn?target=`、0 条裸外链、无 审核中 / 文章有更新 / 已被删除 / 找不到页面 / 发布失败 标记、0 评论 [创作者中心列表读取时] / 1 评论 [公开页计数读取]
- 规范 URL：https://juejin.cn/post/7698287001586040872（暂存 https://juejin.cn/spost/7698287001586040872）
- 正文：`juejin-body.md`（源文 `docs/tutorials/xpu/flamegraph/README.zh.md` 移除源 H1，其余正文逐字保留；31718 字符、57736 字节、723 行、11 个 H2 / 17 个 H3 / 5 个 H4、25 个代码块 [1 c、1 cpp、2 python、18 bash、3 无语言标签]、0 张图片、0 个表格、11 条唯一外链目标 [docs.nvidia.com Nsight 用户指南、github.com/eunomia-bpf/bpftime/tree/master/example/gpu、github.com/eunomia-bpf/xpu-perf、github.com/intel/iaprof、perfetto.dev、pytorch.org 博客与文档、rocm.docs.amd.com、alcf.anl.gov HPCToolkit 2024 幻灯片、brendangregg.com AI 火焰图博文、intel.com Level Zero tracer]）
- 图片说明：源文第 11 行引用 `qwen3_flamegraph.svg`（教程 `.gitignore` 的 `*.svg` 规则忽略的构建产物，仓库内无此文件、上游 bpf-developer-tutorial 无、eunomia.dev 线上 404）；发布稿省略该图片行（无法转存为已检公共 URL 或上传，避免猜测 404 地址），正文其余逐字保留
- 标题：eBPF 示例：使用 CUPTI 构建 GPU 火焰图分析器（源 H1 逐字，无系列编号）
- 原文：https://eunomia.dev/zh/tutorials/xpu/flamegraph/
