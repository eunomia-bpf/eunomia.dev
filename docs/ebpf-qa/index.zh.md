# eBPF 问答

这里每天收录一个关于 eBPF、Linux 可观测性、性能分析、运行时扩展或安全问题的回答。每篇回答都以公开的一手资料为依据，并在末尾附上匿名化的当日社区讨论摘要。

问题可能来自公开 issue、邮件列表、论坛或技术社区。发布前，我们会重新表述问题，删除身份、私有部署细节和可能定位原讨论的措辞，再用公开资料核对答案。每次成功完成当日巡检后发布一篇；如果访问或证据不足，则明确视为失败，不用虚构内容填补日期。

## 最新回答

- [为什么 syscall rewriting trampoline 会让 `clone` 或 `clone3` 创建的线程崩溃？](/zh/ebpf-qa/2026-08-24-clone-syscall-trampoline-child-stack/)
- [OpenTelemetry GenAI 评估结果是否应该携带可验证证据的引用？](/zh/ebpf-qa/2026-08-23-opentelemetry-genai-evaluation-evidence-reference/)
- [为什么无人抓取的 OBI Prometheus endpoint 仍会让内存持续增长？](/zh/ebpf-qa/2026-08-22-obi-unused-prometheus-exporter-memory-growth/)
- [OpenTelemetry 指标生产者是否应该把服务身份复制到每个数据点？](/zh/ebpf-qa/2026-08-21-opentelemetry-resource-attributes-prometheus-labels/)
- [如何判断 OpenTelemetry GenAI 属性是否已经稳定到可以依赖？](/zh/ebpf-qa/2026-08-20-opentelemetry-genai-attribute-stability/)
- [为什么安装了许多内核模块的主机上，libbpf 加载 BPF 对象会变慢？](/zh/ebpf-qa/2026-08-19-libbpf-selective-kmod-btf-loading/)
- [BPF 可扩展调度生效时，cgroup v2 的 `cpu.max` 仍会限制 CPU 时间吗？](/zh/ebpf-qa/2026-08-18-sched-ext-cgroup-cpu-max/)
- [为什么使用 BPF 私有栈的可抢占内核可能被 classic uprobe 程序触发崩溃？](/zh/ebpf-qa/2026-08-17-uprobe-private-stack-preemption-crash/)
- [eBPF 程序应如何在多个网络 hook 之间携带每包元数据？](/zh/ebpf-qa/2026-08-16-cross-hook-packet-metadata/)
- [Linux VM eBPF 后端应如何支持 macOS 和 Windows，同时不误报宿主机覆盖范围？](/zh/ebpf-qa/2026-08-15-cross-platform-ebpf-linux-vm-backend/)
- [OpenInference 应如何与 OpenTelemetry 的 GenAI 语义约定共存？](/zh/ebpf-qa/2026-08-13-openinference-opentelemetry-genai/)
- [为什么 `scxctl` 接受了调度器切换命令，服务却仍未按预期启动？](/zh/ebpf-qa/2026-08-11-scxctl-scheduler-arguments/)
- [为什么在 TC egress 中把 socket 放入 `SOCKHASH` 会导致内核 soft lock？](/zh/ebpf-qa/2026-08-10-tc-egress-sockhash-soft-lock/)
- [为什么 `sched_ext` 调度器要升级 `pahole` 后才能加载？](/zh/ebpf-qa/2026-08-09-sched-ext-pahole-version/)
- [eBPF 能否识别网络流量中的密钥，同时不采集密钥本身？](/zh/ebpf-qa/2026-08-08-detect-secrets-with-ebpf/)
