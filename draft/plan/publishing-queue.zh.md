# 内容发布滚动队列

> 状态：内部执行 checklist，不发布。更新时间：2026-08-11。
> 节奏：从上到下处理，每个 checkbox 是一个平台发布任务，每个自然日最多完成一条。
> 授权：`排队` 表示定时巡检可以完成该任务的准备、浏览器预览、真实发布和公开页检查；`待确认` 与 `阻塞` 不授权发布。

## 当前状态

- 已核定：BPFix 已通过 API 发布到 Medium 和 DEV，公开页完整滚动检查通过；Medium 的可见标题已通过网页编辑器修复。
- 已核定：SchedCP 掘金文章与 tutorial 54 的掘金更新均已确认公开。
- 已核定：AgentNebula 已发布到 [LinkedIn](https://www.linkedin.com/posts/yunwei37_aiagents-observability-developertools-activity-7485819115337637888-iFAK) 和 [X](https://x.com/yunwei37/status/2080063281637830665)；Akeep 已发布到 [LinkedIn](https://www.linkedin.com/posts/yunwei37_github-eunomia-bpfakeep-privacy-first-activity-7489412627769548800-SV__) 和 [X](https://x.com/yunwei37/status/2083647587132035205)，不再重复排期。
- 暂停：AgentPProf 暂不进入发布队列；Hacker News 暂不发；Lobsters 没有账号。
- 阻塞：小红书尚未确认账号 URL、登录状态和图片卡片工作流。
- Ledger 基线：知乎有 40 个未映射中文源，掘金有 77 个；其中 AgentPProf、目录/兼容性/进一步阅读页和两个重复教程表示不进入发布队列。
- 历史回补：2026 年以前的非 tutorial Blog 不进入常规平台同步队列；只有出现新的证据、结论或明确发布需求时才单独重新评估。
- LinkedIn 只安排新的 tutorial 50–54；SchedCP、AgentCgroup、CPU noise 和已有发布记录的内容不重复安排。
- Medium 与 DEV 的 BPFix 和 tutorial 50 已完成；tutorial 51–54 以及 AgentCgroup、CPU noise、Agent Sandbox、ACRFence、Runtime Security 的英文长文交叉排入现有队列，保留英文原文标题和正文，只做必要的平台格式适配。
- Weekly Analysis 和研究报告由独立研究流程负责，不进入本发布队列。
- 剩余队列：知乎 26 条、掘金 49 条、Medium 9 条、DEV 9 条、LinkedIn 3 条，共 96 个平台任务；后续新增内容直接插入合适位置。

## 发布队列

- [x] `排队` 掘金：`docs/blog/posts/schedcp-agentic-os.zh.md`。2026-07-25 已发布：<https://juejin.cn/post/7666245761520615439>；2026-07-26 公开页正文和图片 QA 通过，ledger 已记录为 `confirmed`。
- [x] `排队` 知乎：`docs/blog/posts/bpfix.zh.md`。2026-07-26 已发布：<https://zhuanlan.zhihu.com/p/2064970067830480995>；公开页标题、正文、图片、表格、代码块和链接 QA 已通过。
- [x] `排队` 掘金：`docs/blog/posts/bpfix.zh.md`。2026-07-29 恢复任务时在可见个人文章列表发现已公开：<https://juejin.cn/post/7667474278616547380>；未重复提交，公开页标题、正文、图片、表格、代码块和链接 QA 已通过，ledger 已记录为 `confirmed`。
- [x] `排队` Medium：`docs/blog/posts/bpfix.md`。2026-08-02 已发布：<https://medium.com/@yunwei356/why-ebpf-verifier-errors-are-hard-to-fix-the-diagnostic-gap-d7904b9432f0>；公开页标题、正文结构、代码块、表格降级、链接和两张图片 QA 通过。
- [x] `排队` DEV：`docs/blog/posts/bpfix.md`。2026-08-02 已发布：<https://dev.to/yunwei37/why-ebpf-verifier-errors-are-hard-to-fix-the-diagnostic-gap-443o>；公开页标题、canonical、正文结构、代码块、表格、链接和两张图片 QA 通过。
- [x] `排队` 知乎：`docs/tutorials/50-tcx/README.zh.md`。2026-08-01 已发布：<https://zhuanlan.zhihu.com/p/2067144957194924576>；公开页标题、21 个正文标题、8 个代码块、2 个表格（12 行）和 10 个正文外链 QA 已通过，ledger 已记录为 `confirmed`。
- [x] `排队` 掘金：`docs/tutorials/50-tcx/README.zh.md`。2026-08-03 已提交，2026-08-04 确认公开：<https://juejin.cn/post/7669635386159824936>；公开页标题、21 个正文标题、8 个代码块、2 张表格（12 行）和 10 个正文链接完整，ledger 已记录为 `confirmed`。
- [x] `排队` Medium：`docs/tutorials/50-tcx/README.md`。已核定公开：<https://medium.com/@yunwei356/ebpf-tutorial-by-example-50-composable-traffic-control-with-tcx-links-0b64b72f7a59>。
- [x] `排队` DEV：`docs/tutorials/50-tcx/README.md`。已核定公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-50-composable-traffic-control-with-tcx-links-5hmo>。
- [x] `排队` LinkedIn：`docs/tutorials/50-tcx/README.md`。2026-08-04 已发布：<https://www.linkedin.com/feed/update/urn:li:share:7490597138595229696>；公开页两句正文、公开可见范围、eunomia.dev 链接以及带标题和图片的教程预览卡片 QA 已通过，ledger 已记录为 `confirmed`。
- [x] `排队` 知乎：`docs/tutorials/51-tcp-quarantine/README.zh.md`。2026-08-05 已发布：<https://zhuanlan.zhihu.com/p/2068597605345531007>；公开页标题、15 个正文标题、9 个代码块、1 个表格（7 行）和 6 个正文外链 QA 已通过，ledger 已记录为 `confirmed`。
- [x] `排队` 掘金：`docs/tutorials/51-tcp-quarantine/README.zh.md`。2026-08-07 已发布并核定公开：<https://juejin.cn/post/7671185597861675008>；公开页精确标题、15 个正文标题、9 个代码块、1 个表格（7 行）、6 个源文外链、无源文图片占位、`Linux`、`开源`、`安全` 标签和 0 条评论 QA 已通过。
- [x] `排队` Medium：`docs/tutorials/51-tcp-quarantine/README.md`。2026-08-09 已发布：<https://medium.com/@yunwei356/ebpf-tutorial-precisely-isolating-established-tcp-connections-9d08644b44b4>；公开页完整滚动 QA 确认原标题、完整正文、9 个代码块、GitHub/eunomia.dev 链接和标签正常，失真的 Markdown 表格已在网页编辑器中改为可读说明。
- [x] `排队` DEV：`docs/tutorials/51-tcp-quarantine/README.md`。2026-08-10 已发布：<https://dev.to/yunwei37/ebpf-tutorial-precisely-isolating-established-tcp-connections-56ld>；公开页完整滚动 QA 确认原标题、4 个标签、完整正文、9 个代码块、1 个表格、Requirements、Summary、References 和源码链接正常。
- [x] `排队` LinkedIn：`docs/tutorials/51-tcp-quarantine/README.md`。2026-08-11 已发布：<https://www.linkedin.com/feed/update/urn:li:share:7493164412321812480/>；公开页两句正文、站内外公开可见范围、LinkedIn 短链落地与 eunomia.dev 教程预览卡片（标题、图片、域名）QA 已通过，ledger 已记录为 `confirmed`。
- [x] `排队` 知乎：`docs/tutorials/52-fsession-latency/README.zh.md`。2026-08-14 已发布：<https://zhuanlan.zhihu.com/p/2071960422765310563>；选择问题“在 Linux 环境下，如何有效诊断和优化高并发应用的 I/O 性能瓶颈？”以及话题 `ebpf`、`Linux 内核`，公开文章页与问题页 QA 确认原标题、完整正文、18 个正文标题、20 个代码块、1 个表格、源码链接和参考资料正常。
- [ ] `排队` 掘金：`docs/tutorials/52-fsession-latency/README.zh.md`。2026-08-15 已提交：<https://juejin.cn/spost/7673940698534232064>；当前 `审核中`，原标题、18 个正文标题、20 个代码块、1 个表格、源码与参考链接 QA 通过，待公开后再确认完成。
- [ ] `排队` Medium：`docs/tutorials/52-fsession-latency/README.md`。
- [ ] `排队` DEV：`docs/tutorials/52-fsession-latency/README.md`。
- [ ] `排队` LinkedIn：`docs/tutorials/52-fsession-latency/README.md`。
- [ ] `排队` 知乎：`docs/tutorials/53-egress-pacer/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/53-egress-pacer/README.zh.md`。
- [ ] `排队` Medium：`docs/tutorials/53-egress-pacer/README.md`。
- [ ] `排队` DEV：`docs/tutorials/53-egress-pacer/README.md`。
- [ ] `排队` LinkedIn：`docs/tutorials/53-egress-pacer/README.md`。
- [ ] `排队` Medium：`docs/tutorials/54-exec-image-inspector/README.md`。
- [ ] `排队` DEV：`docs/tutorials/54-exec-image-inspector/README.md`。
- [ ] `排队` LinkedIn：`docs/tutorials/54-exec-image-inspector/README.md`。
- [ ] `排队` 知乎：`docs/blog/posts/agent-work-unit.zh.md`。
- [ ] `排队` Medium：`docs/blog/posts/agentcgroup-characterization.md`。
- [ ] `排队` 掘金：`docs/blog/posts/agent-work-unit.zh.md`。
- [ ] `排队` DEV：`docs/blog/posts/cpu-noise-gpu-inference.md`。
- [ ] `排队` 知乎：`docs/blog/posts/agentcgroup-characterization.zh.md`。
- [ ] `排队` Medium：`docs/blog/posts/agent_sandbox.md`。
- [ ] `排队` 掘金：`docs/blog/posts/agentcgroup-characterization.zh.md`。
- [ ] `排队` DEV：`docs/blog/posts/agent-check-restore-safety.md`。
- [ ] `排队` 知乎：`docs/blog/posts/cpu-noise-gpu-inference.zh.md`。
- [ ] `排队` Medium：`docs/blog/posts/runtime-security-for-opaque-ai-agents.md`。
- [ ] `排队` 掘金：`docs/blog/posts/cpu-noise-gpu-inference.zh.md`。
- [ ] `排队` DEV：`docs/blog/posts/agentcgroup-characterization.md`。
- [ ] `排队` 知乎：`docs/tutorials/49-hid/README.zh.md`。
- [ ] `排队` Medium：`docs/blog/posts/cpu-noise-gpu-inference.md`。
- [ ] `排队` 掘金：`docs/tutorials/49-hid/README.zh.md`。
- [ ] `排队` DEV：`docs/blog/posts/agent_sandbox.md`。
- [ ] `排队` 知乎：`docs/blog/posts/agent_sandbox.zh.md`。
- [ ] `排队` Medium：`docs/blog/posts/agent-check-restore-safety.md`。
- [ ] `排队` 掘金：`docs/blog/posts/agent_sandbox.zh.md`。
- [ ] `排队` DEV：`docs/blog/posts/runtime-security-for-opaque-ai-agents.md`。
- [ ] `排队` 掘金：`docs/blog/posts/agent-check-restore-safety.zh.md`。
- [ ] `排队` 掘金：`docs/blog/posts/runtime-security-for-opaque-ai-agents.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/48-energy/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/48-energy/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/47-cuda-events/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/46-xdp-test/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/46-xdp-test/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/45-scx-nest/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/44-scx-simple/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/43-kfuncs/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/42-xdp-loadbalancer/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/41-xdp-tcpdump/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/40-mysql/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/40-mysql/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/39-nginx/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/38-btf-uprobe/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/38-btf-uprobe/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/37-uprobe-rust/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/37-uprobe-rust/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/35-user-ringbuf/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/35-user-ringbuf/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/34-syscall/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/34-syscall/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/33-funclatency/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/32-wallclock-profiler/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/31-goroutine/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/29-sockops/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/29-sockops/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/28-detach/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/28-detach/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/27-replace/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/27-replace/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/26-sudo/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/26-sudo/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/24-hide/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/22-android/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/21-xdp/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/20-tc/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/19-lsm-connect/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/17-biopattern/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/1-helloworld/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/6-sigsnoop/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/bpftrace-tutorial/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/bpftrace-tutorial/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/cgroup/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/cgroup/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/features/bpf_arena/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/bpf_arena/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/features/bpf_iters/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/bpf_iters/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/features/bpf_token/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/bpf_token/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/features/bpf_wq/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/bpf_wq/README.zh.md`。
- [ ] `排队` 知乎：`docs/tutorials/features/dynptr/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/dynptr/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/features/struct_ops/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/xpu/flamegraph/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/xpu/gpu-kernel-driver/README.zh.md`。
- [ ] `排队` 掘金：`docs/tutorials/xpu/npu-kernel-driver/README.zh.md`。

## 执行规则

- 每次巡检从顶部找到第一条仍为 `排队` 的任务，当天只处理这一条；同一源同步到两个平台是两个独立任务。
- 发布前立即确认 ledger 仍缺失、源文是可独立阅读的完整文章。发现重复、过时或不适合平台时，把该行改为 `跳过` 并写一句原因，然后继续寻找当天唯一可发布项。
- 长文保持源标题和正文，只处理本地上传 artifact、图片、代码、链接、标签和平台渲染。
- 历史回补默认只处理 tutorial；2026 年以前的非 tutorial Blog 不因 ledger 缺失自动进入队列。
- 发布后改为 `[x]`，紧跟公开 URL、审核状态和 ledger 结果；真实异常或待跟进事项才写 `draft/media/YYYY-MM-DD/run-log.md`。
