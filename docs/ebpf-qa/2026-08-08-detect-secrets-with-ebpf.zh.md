# eBPF 能否识别网络流量中的密钥，同时不采集密钥本身？

**简短回答：**可以，但探针必须位于能够看到明文的位置，而且原始值不能离开这个位置。

挂载在 TC、XDP 或数据包套接字上的 eBPF 程序看到的是线上传输的字节。对于 HTTPS，这些字节通常已经加密，因此程序可以判断连接类型、统计流量，却无法从密文中识别 API key。OpenTelemetry eBPF Instrumentation（OBI）也区分网络可观测性和应用可观测性；它对 TLS 请求的支持依赖用户态 uprobe，并可能需要额外权限，而不是只靠普通的数据包检查。

如果探针位于加密前或解密后的明文边界，例如受支持的 TLS 库函数，它就可以读取足够的信息来判断一段内容是否可能是密钥。不过，这也意味着探针本身能够接触敏感字节。隐私保护取决于后续处理：在本地完成匹配，只输出类别或计数，然后丢弃原始值，不让它进入 ring buffer、日志、trace 或采集器。

## 怎样验证这套设计

验证对象应该是整条数据路径，而不只是 eBPF 程序：

1. 在测试环境中通过加密请求发送一个专用的模拟密钥。
2. 确认数据包层只能看到密文，明文边界的探针只输出预期的分类结果。
3. 检查 BPF map、ring buffer 事件、调试日志、trace 和导出的遥测数据，确认其中没有模拟密钥。
4. 再测试请求分片、重试、不受支持的 TLS 库和非 TLS 流量，避免隐私结论只在一条理想路径上成立。

下游脱敏仍然可以作为第二层保护，但它解决的是另一个问题。例如，Grafana Alloy 的 `loki.secretfilter` 会在日志发送到 Loki 之前识别并遮盖密钥，却无法让上游已经采集的原始载荷自动变得安全。

实际部署时，可以把原则概括为一句话：尽量靠近明文边界完成分类，只导出能够回答运维问题的最少量非敏感结果，再检查整条链路中是否残留原始值。如果当前运行时不支持所需的 TLS 边界，应退回到基于元数据的判断或应用层埋点，而不是扩大载荷采集范围。

## 参考资料

- [OpenTelemetry eBPF Instrumentation：安全与运行模式](https://opentelemetry.io/docs/zero-code/obi/security/)
- [OpenTelemetry eBPF Instrumentation：TLS 可见性故障排查](https://opentelemetry.io/docs/zero-code/obi/troubleshooting/)
- [Grafana Alloy `loki.secretfilter`](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.secretfilter/)
- [Linux 内核文档：libbpf 应用生命周期与 CO-RE](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html)
- [Linux 内核文档：`BPF_MAP_TYPE_SOCKMAP` 与 `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux 内核文档：`sched_ext`](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux 内核文档：BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [OpenTelemetry：代码埋点与零代码埋点](https://opentelemetry.io/docs/concepts/instrumentation/)
- [OpenTelemetry：Go 编译期埋点](https://opentelemetry.io/docs/zero-code/go/compile-time/)

## 今日社区讨论

今天通过普通可见浏览器检查了 4 个技术社区中的 7 个已批准公开频道。下面只保留聚合后的技术主题，参与者身份、频道、消息链接、精确时间和具体部署信息都已删除。

### 识别密钥，又不把密钥带出探针

当天最集中的问题是怎样识别服务之间传递的凭据，同时避免让可观测系统变成另一份敏感数据存储。关键在于观测位置：数据包钩子可以描述加密连接，却无法检查 HTTPS 请求中的密钥；受支持的用户态 TLS 钩子能够看到明文，因此本身就必须按照敏感代码处理。可行的做法是在这个边界完成分类，只导出类别或计数，不导出匹配到的字节，然后用专门的模拟密钥检查 map、ring buffer、日志、trace 和采集器输出。OBI 的安全与 TLS 排障文档说明了可见性和权限边界，`loki.secretfilter` 的文档则解释了为什么下游脱敏只能作为第二层保护。

### 加载器退出后怎样恢复数据平面

另一组讨论关心的是：用户态加载器重启时，内核中的 eBPF 对象可能仍然存活，控制器应该怎样恢复。恢复流程可以把 `bpffs` 中的内容当作“观测到的现状”，但不能直接认定它就是健康的目标状态。控制器需要重新打开固定的 map、程序和 link，比较对象标识、map 结构、挂载点和预期版本，复用兼容对象，再按明确顺序替换陈旧或不完整的状态。libbpf 文档中的打开、加载、挂载和清理四个阶段，正好可以作为这套核对流程的基础。

控制循环还需要隔离可能阻塞的工作。BPF 系统调用、link 挂载和清理都可能独立阻塞或失败，适合放进有并发上限、超时、取消和幂等重试的工作线程，而不是直接占住异步事件循环。健康状态也应来自一次成功的状态核对和可用的数据路径，不能只看 `bpffs` 下是否存在文件。仍需由具体项目决定的是，升级时哪些 map 保存了值得延续的在线状态；这需要逐个 map 定义兼容规则，不能统一按“全部复用”或“全部重载”处理。

### 从 TC 路径加入 `SOCKHASH` 时出现卡死

一个网络问题描述了把已经建立的套接字从 TC 程序加入 `SOCKHASH` 时出现的内核卡死。排查时应先把场景缩减到内核提供的 sockmap 自测规模，并逐项确认内核版本、程序类型、attach type、套接字状态，以及该套接字是否已经继承了其他 parser 或 verdict 程序。套接字加入 map 时会替换回调并挂上 `sk_psock`；内核文档也明确说明，parser 或 verdict 程序发生冲突时，更新可能失败。因此，TC 数据包路径观察到一个套接字，并不能直接证明此时适合把它纳入 sockmap。

实现路径应跟随内核公开支持的接口：用户态可以通过文件描述符加入套接字，`BPF_PROG_TYPE_SOCK_OPS` 程序则可以在拿到套接字上下文时调用 `bpf_sock_hash_update()`；后续流处理放在文档列出的 `SK_MSG` 或 `SK_SKB` verdict 钩子中。如果缩减后的配置与上游自测一致，内核依然卡死而不是返回错误，就应保留最小复现，采集内核栈、verifier 日志或相关 trace，并按内核回归报告。公开的 sockmap 文档及其链接的自测代码给出了参照行为，无需暴露社区中原始部署的细节。

### 让 `sched_ext` 对准正在运行的内核

调度器讨论始于一个 `sched_ext` 对象加载失败：它所期待的调度器 kfunc 类型与运行中内核的 BTF 对不上。继续用相同环境重编译，或者只调整运行参数，都无法解释这种差异。更直接的核对方式是先确认运行内核启用了 `CONFIG_SCHED_CLASS_EXT` 和 `CONFIG_DEBUG_INFO_BTF`，再导出 `/sys/kernel/btf/vmlinux`，检查调度器对象及其 `vmlinux.h` 是否来自同一套内核和 `sched_ext` 接口版本。Linux 内核的 libbpf 与 BTF 文档把 `/sys/kernel/btf/vmlinux` 定义为 CO-RE 在运行时使用的权威类型来源。

如果使用发行版或带补丁的内核，还应记录生成该内核的确切源码版本、补丁、编译器和 pahole 版本。这样才能区分三类外观相近的问题：内核根本没有该能力、kfunc 存在但签名不同，或者 BPF 对象针对另一版接口构建。`sched_ext` 文档同时列出了必要的内核配置和运行状态文件，因此下一步应先用同一源码树构建一个最小的已知可用调度器，确认基础链路后再回到更大的程序。

### 选择 eBPF 还是编译期埋点

最后一组讨论比较了 eBPF 观测和编译期语言埋点。选择取决于问题需要什么上下文，以及部署流程允许改动哪一层。运维方无法重编应用、而进程、网络和库边界已经包含足够信息时，eBPF 可以较快获得广泛覆盖；如果可以修改构建流程、不适合部署高权限运行时探针，或者需要覆盖 eBPF 方案尚未支持的库调用，编译期埋点更合适。需要应用业务意图和自定义 span 时，则仍要使用 OpenTelemetry 的代码埋点。

OpenTelemetry 的总览把代码埋点和零代码方案明确视为互补手段，Go 编译期埋点文档也列出了构建流程与运行时权限之间的取舍。实际比较时，可以在同一个服务上验证四件事：trace 上下文能否保留、目标库是否覆盖、升级后是否仍能工作，以及权限成本是否符合部署要求。项目自有频道在检查窗口内主要出现自动化开发活动，没有新的实质性用户提问；这里把它与上述技术讨论分开记录，不把自动消息计入社区参与度。
