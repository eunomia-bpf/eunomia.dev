# Linux VM eBPF 后端应如何支持 macOS 和 Windows，同时不误报宿主机覆盖范围？

**简短回答：**在 Linux guest 中运行现有 Linux eBPF collector，在 guest 外保留宿主机原生 session 发现，并通过工具既有的 event model 合并两路数据。每条事件都必须标记采集后端与执行边界。Lima 或 WSL 2 集成提供的是**在 macOS 或 Windows 机器上的 Linux guest 采集**，并不是对宿主机内核、宿主机进程或宿主机用户态 TLS 的原生 eBPF 可见性。

这不是措辞问题，而是架构边界。Lima 启动 Linux 虚拟机；WSL 2 则在托管的轻量虚拟机中运行 Linux 内核。因此 eBPF 程序看到的是 guest 内核，以及真正运行在 guest 中或经过 guest 的 workload。仅仅因为 guest 与宿主机位于同一台电脑，就声称观察到了宿主机进程，是不成立的。

## 分开定义三类“支持”

跨平台可观测工具应公布 capability matrix，而不是只给出一个笼统的“已支持”标记：

| 后端 | 可以观察什么 | 不得声称什么 |
| --- | --- | --- |
| Linux eBPF | 在已启用 hook 与权限范围内观察 Linux guest 或 host 内核事件 | 来自另一套宿主机内核或进程命名空间的事件 |
| 宿主机原生 session source | 主机集成明确暴露的 agent session 文件或 API | 从未采集到的内核来源、syscall 覆盖或解密流量 |
| 平台原生遥测 | 该操作系统后端实际实现的 hook | 未经验证的 Linux hook 对等性或源码兼容性 |

这一区分在 Windows 上尤其重要。eBPF for Windows 是真实的原生实现，但它的官方文档说明，其运行环境是 Windows 专用的，并且源码兼容目标只覆盖可跨操作系统成立的 hook 和 helper。这与在 WSL 2 中运行 Linux eBPF 是两个不同后端，不能静默互相替代。

在 macOS 上，Lima 后端同样只是为项目提供受控的 Linux 执行环境，不会把宿主机进程变成 Linux task。宿主机专属上下文必须来自文档化的原生 source；如果没有，就应报告不可用。

## 把 VM 边界放在现有 collector 契约之后

AgentSight 当前仓库已经有合适的接入边界。可复用 capture crate 将 `sources/`、`runners/`、sinks 和公共 event model 分开。第一版 VM 集成应保留这一契约：

```text
macOS 或 Windows 宿主机
├── 原生 session source ────────────────┐
├── VM 生命周期与传输 adapter           │
└── Linux VM                            │
    └── 现有 Linux eBPF runner ─────────┤
                                        ↓
                              公共 collector/event model
                                        ↓
                                  view、report 或 sink
```

宿主机 adapter 应检测显式配置的 Lima instance 或 WSL 2 distribution，在其中启动现有 Linux capture 命令，再通过范围很窄的本地传输送回规范化事件。不要为每个操作系统复制一套 event schema。传输 framing、重连和生命周期状态属于 adapter；eBPF 加载与 Linux 专属 feature check 仍留在 guest runner。

每条事件都需要不可变的 provenance，以免不同证据被意外混在一起。至少记录：

- `capture_backend`，例如 `linux-ebpf`、`native-session` 或未来的平台原生后端；
- 被观察系统的 OS 与 kernel，而不只是 UI 所在宿主机；
- 能区分 host 与 guest 的执行环境标识；
- source clock domain 与采集时间；
- 含义明确的 correlation key。

不要把 Linux guest PID 当作宿主机进程标识。PID、namespace、path、network interface 与 clock 穿过 VM 边界后含义都不同。如果宿主机 agent session 会在 guest 内启动任务，应在启动或传输时生成 correlation token。没有可信 token 时，应展示两条相邻 timeline，而不是断言存在因果关联。

## 明确展示部分覆盖

最危险的失败模式是“看起来完整”的残缺 trace。因此 CLI 与 report 必须说明实际运行了哪个后端、能够看到什么。合理状态包括：

- `linux-vm capture active`，并显示 guest 身份与已启用 probe；
- `host session source active`，明确没有内核采集；
- 当 guest 可达但缺少 hook 或权限时显示 `degraded`；
- 当配置的 guest 无法启动或传输认证失败时显示 `unavailable`。

不要在 guest eBPF 失败后静默回退到宿主机 session 文件，却仍然显示同一个“capturing”状态。仅由 session 文件形成的 report 仍有价值，但证据类型不同。

网络与 TLS 观察也适用同一规则。guest probe 可以观察 guest 自身产生的流量，以及被显式路由经过 guest 的流量；它不会自动看到 host-local socket，也无法恢复在宿主机进程中加密后才进入 guest 的明文。文件共享与端口转发让集成更方便，却不会消除采集边界。

## 先测试边界，再增加功能

小而可信的第一版优于宽泛的兼容层。应在三种环境运行同一个聚焦 capture 命令：原生 Linux、macOS 上的 Linux VM、以及 WSL 2。验收测试至少验证：

1. guest 运行未修改的 Linux capture 路径；
2. 事件保持相同公共 schema，同时暴露不同 provenance；
3. 宿主机原生 session 上下文只能通过显式 token 关联；
4. guest 停止或重启会产生可见的生命周期转换；
5. 只运行在宿主机的进程绝不会被报告成 guest 已观察；
6. 权限缺失、hook 不支持、时钟漂移与传输中断会 fail closed，而不会生成看似完整的 trace。

还要测试版本偏差。记录 host integration 版本、guest collector 版本、guest kernel 与 event-schema 版本；streaming 前协商 schema，不兼容的 major version 应直接拒绝，不能猜测。这样 VM image 更新就不会悄悄变成数据模型变更。

## 把 guest 视为安全边界

可观测 guest 通常会得到共享文件、转发端口、较高的 BPF 权限和敏感 agent 事件。应只挂载必需路径、只授予必需 capability、认证本地传输、避免把 collector 暴露到宿主机之外，并显式设置留存策略。不要为了改善 correlation 而复制 prompt、response、credential 或完整流量正文。

因此合理路线是增量式的：先实现具有明确 provenance 与负向测试的诚实 Linux-VM 后端；再完善 host-to-guest correlation；最后才在独立 capability 声明下加入真正的平台原生遥测。“可以在 macOS 或 Windows 上运行”是打包结论；“可以观察 macOS 或 Windows 宿主机”是证据结论，需要不同后端支持。

## 参考资料

- [AgentSight issue：跨平台支持与 Linux-VM 采集边界](https://github.com/eunomia-bpf/agentsight/issues/17)
- [AgentSight capture crate 源码结构](https://github.com/eunomia-bpf/agentsight/tree/master/agentsight-capture/src)
- [AgentSight README：Linux eBPF 要求与原生 session 回退行为](https://github.com/eunomia-bpf/agentsight)
- [Lima 文档：带文件共享和端口转发的 Linux 虚拟机](https://lima-vm.io/docs/)
- [Microsoft 文档：WSL 2 在轻量 utility VM 中运行 Linux 内核](https://learn.microsoft.com/en-us/windows/wsl/about#what-is-wsl-2)
- [eBPF for Windows：架构、hook、helper 与源码兼容边界](https://github.com/microsoft/ebpf-for-windows)
- [sched-ext 公共源码中的 `scx_lib_init_probe` fentry probe](https://github.com/sched-ext/scx/blob/558aa09863e7bddb09101e4b242cc6efaee3dd5f/scheds/include/scx/common.bpf.h#L522-L540)
- [BPF 邮件列表：verifier 诊断重构](https://lore.kernel.org/bpf/178682023625.53386.10978136746024990805.git-patchwork-notify@kernel.org/T/#t)
- [OpenTelemetry Python contrib：弃用 GenAI 插件的 security-only patch 策略](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4955)
- [用于测量 eBPF cache 的内核内滚动窗口设计](https://naveensrinivasan.com/posts/2026-08-02-measuring-an-ebpf-cache-without-leaving-the-kernel/)

## 今日社区讨论

今天通过普通可见浏览器检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面。所有目标均可访问；其中一个公开论坛的当前界面出现人机验证，因此改用其普通可见的旧版界面。入选问题来自过去 24 小时，没有使用 7 天回退。以下内容已删除姓名、账号、雇主、频道身份、消息链接、精确时间、私有拓扑、原始日志与可回搜措辞；没有保留原始 transcript。

### 跨平台支持首先需要证据边界

当日最强问题是：贡献者应把基于 VM 的 macOS 或 Windows 原型接到现有 eBPF 可观测工具的哪一层。真正重要的答案不是某段平台检测代码，而是 portable source、Linux capture runner 与共享 event model 之间的分工。这个边界允许只增加一个小 adapter，同时保留现有事件的含义。

更广泛的关注是 Linux VM 能否被宣传为宿主机支持。它可以提供有用的产品体验：从宿主机安装，在 guest 内运行 probe，并把 guest event 与宿主机原生 session context 关联起来。但 report 必须保留 provenance。这正是“工具在这里运行”与“工具观察了这个内核”的区别。实践者还应要求负向测试，证明 host-only workload 不会被归到 guest 名下。

### attach 失败仍要先隔离失败层

一段 scheduler 支持讨论描述了在更新内核上 attach 非 `struct_ops` BPF 程序失败的现象。直接启动 scheduler 仍可复现，因而可以排除 service loader。剩余公共代码路径包含一个附加到 scheduler 注册函数的弱 fentry probe。下一步应建立最小版本矩阵，并在公共 issue 中提供第一条 libbpf attach error、kernel 版本、scheduler 版本、architecture，以及禁用该 optional probe 后同一 object 是否可以加载。

巡检结束时，这一问题仍未解决。现有证据只足以把故障缩小到 program attachment，不能断言 kernel 根因。可复用的排障方法是逐层去掉 orchestration，识别具体 BPF program 与 attach type，并保留最早出现的 verifier 或 attach diagnostic，而不是只记录最终 aggregate error。

### release policy 与 runtime compatibility 仍是两份契约

一段 GenAI instrumentation 讨论关注已弃用 package：它们仍可以接受 security patch，却不再出现在 major/minor release workflow 中。公共 release-policy 变更确认这种不对称是有意设计，但它不保证未来上游 SDK major 的兼容性。Security-only maintenance、dependency constraint 与 semantic-convention migration 是三份不同契约；release automation 与文档必须分别写清楚。

其他项目专用区域较安静，或只有 build notification 而没有新的实践者问题。通用 eBPF 讨论区也没有新的当日排障线程，其最近的 socket-map 主题已经由更早的 Q&A 回答。

### 上游讨论集中在可诊断性与低开销测量

公共 BPF 归档非常活跃。最相关的系列提出结构化 verifier diagnostic category、源码与指令上下文，并在 verifier 放弃路径时清理无关诊断。这与 scheduler attach 故障体现同一运维原则：保留最早、最具体的 failure 及其 execution context，不要让用户只面对一连串寄存器状态后果。

公开论坛最新技术帖询问：如何为 eBPF cache 保存滚动 usage metric，同时避免持续向 user space 发送事件、增加共享锁或使用通用 LRU。其设计采用 per-CPU state 与有界内核内 bucket，以精确全局顺序换取可预测 overhead。采用前仍应测量 update cost、aggregation error、bucket rollover、每 CPU 内存，以及 CPU hotplug 行为。贯穿今天各话题的共同要求是如实声明范围：VM 后端要说明在哪里观察，loader 要说明哪一次 attach 失败，metric 要说明引入了什么近似。
