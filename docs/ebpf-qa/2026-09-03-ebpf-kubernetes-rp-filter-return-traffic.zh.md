# 为什么反向路径过滤会丢弃 eBPF Kubernetes 数据路径中的返回流量？

**简短回答：** Linux 的严格反向路径过滤会用内核转发信息库（FIB）验证每个进入节点的 IPv4 包。如果返回到该包源地址的最佳路由不经过实际收包接口，内核就会丢包。eBPF 数据路径可能根据 BPF map、endpoint 元数据或更早发生的 redirect 作出转发决定，但这些信息不会自动进入普通 FIB。因此，同一个包可能在 CNI 数据路径看来完全合法，却无法通过 `rp_filter`。

常见表象是连接只有单向：请求已经到达 Pod，Pod 也发出了 SYN-ACK；回包能在节点某个入口看到，却在预期的转发点或出口前消失。长期正确的修复是让 Linux 反向查路结果与真实路径一致；如果非对称路由本来就是设计目标，而且已有另一道反欺骗边界，再只对受影响的接口放宽源地址校验。

## 严格 `rp_filter` 究竟证明什么

设一个源地址为 `S` 的 IPv4 包从接口 `I` 进入。严格模式（`rp_filter=1`）会检查 FIB 返回 `S` 的最佳路径是否使用 `I`；不一致就判定源地址验证失败。宽松模式（`rp_filter=2`）只检查能否通过任一接口到达 `S`。关闭模式（`rp_filter=0`）不执行这两种检查。

[内核 IP sysctl 文档](https://docs.kernel.org/networking/ip-sysctl.html#rp-filter-integer)定义了这三种模式，并建议非对称或复杂路由使用宽松模式。文档还有一条很容易漏掉的规则：Linux 取 `conf/all/rp_filter` 与 `conf/<interface>/rp_filter` 的最大值。即使把接口值设为零，只要 `conf/all/rp_filter` 仍为一，严格过滤就没有关闭。部分发行版也会在内核默认值之外主动启用它。

这是源地址可达性检查，不是 TCP 状态检查。存在合法 conntrack 记录、SYN 与 SYN-ACK 完整匹配、eBPF policy 已返回成功，都不能替代它。当前内核实现会在严格验证无法把反向路由匹配到收包设备时返回明确的 [`SKB_DROP_REASON_IP_RPFILTER`](https://github.com/torvalds/linux/blob/master/net/ipv4/fib_frontend.c)。

`net.ipv4.conf.*.rp_filter` 只适用于 IPv4。不能依据它推断 IPv6 的诊断结论或安全策略；双栈路径必须分别验证。

## 为什么 eBPF CNI 会与 FIB 得出不同结论

Kubernetes Service 虚拟 IP 通常不对应一台真正持有该地址的主机。Service proxy 会截获发往 `clusterIP` 与端口的流量，再把它重定向到选中的 endpoint；[Kubernetes Service proxy 参考文档](https://kubernetes.io/docs/reference/networking/virtual-ips/)描述了这一过程。之后，CNI 数据路径还要把转换后的包送往 endpoint，并把返回流量送回客户端。

eBPF 实现可以在不向普通 Linux FIB 安装等价最佳路由的情况下完成这些工作。例如，它可能：

- 在 BPF map 中把 Pod 地址映射到 endpoint 或 peer 设备；
- 在普通 IPv4 转发决策之前通过 XDP 或 TC redirect；
- 使用虚拟设备作为稳定的程序挂载点；
- 依靠 packet mark 选择策略路由表；或者
- 解封装或重新注入流量，使 IPv4 input 看到的接口不同于物理收包接口。

现在看一个源地址为 Pod IP 的回包。eBPF 程序可能已经掌握足够元数据，知道应把它 redirect 到哪里；但反向 FIB 查找却认为该 Pod IP 应经另一个设备、路由表或 VRF 到达。严格 `rp_filter` 只相信后者，于是拒绝这个包。使用虚拟 host-routing 设备、有意让 BPF 挂载拓扑与路由拓扑分离时，尤其容易出现这种不一致。

hook 顺序也很重要。XDP 和 TC ingress 早于普通 IPv4 input 路由查找。如果早期程序 redirect 了包，使它从另一设备重新进入协议栈，源地址校验会使用后一个 IPv4 接收点所看到的设备。因此，“网卡收到了包”不等于“`rp_filter` 在这张网卡上验证了包”。

## 一套只读诊断顺序

先证明包究竟消失在哪里，不要只因为症状是非对称就修改 `rp_filter`。

1. **记录每个边界上的五元组。** 只抓取诊断所需的包头，分别查看 Pod 一侧、host peer 或 CNI 设备，以及预期的节点出口。确认回包源地址是否仍为 Pod IP，并确认两个方向的 DNAT/SNAT 变化。若第一个 host 侧观察点就没有回包，这还不是 `rp_filter` 诊断。
2. **确认真正的接收上下文。** redirect、VRF 和 network namespace 都可能改变相关设备与 sysctl 所在的命名空间。应在包进入 IPv4 路由的位置检查，而不是只看物理网卡。
3. **读取两处有效配置。** 在相关 network namespace 内，同时检查 `net.ipv4.conf.all.rp_filter` 与 `net.ipv4.conf.<ingress>.rp_filter`；实际模式取两者最大值。如果策略路由使用 firewall mark，也要记录 `src_valid_mark`。
4. **复现反向查路。** 使用 `ip route get <包源地址> from <包目的地址> iif <入接口>`，必要时加入相关 VRF 和 mark。[`ip route get` 手册](https://man7.org/linux/man-pages/man8/ip-route.8.html)说明，`iif` 会让内核按“包从该接口到达”来解析它实际看到的路由。对严格校验而言，应比较解析出的反向路径设备与真实入接口。
5. **在支持时观察内核丢包原因。** 新内核在 [`skb_drop_reason` 枚举](https://github.com/torvalds/linux/blob/master/include/net/dropreason-core.h)中定义了 `IP_RPFILTER`。能读取 `skb:kfree_skb` tracepoint 并解码原因的工具，可以把它与 netfilter、checksum、policy 或 BPF 程序丢包区分开。先查看正在运行的内核所暴露的 tracepoint 格式，不要假设任意版本工具都能解码该字段。
6. **做一次有界对照。** 在隔离测试节点或 namespace 内，只对相关入接口临时比较严格与宽松模式，并预先写好回滚。如果包在宽松模式下通过，而且反向 FIB 不一致已经得到证明，因果链就比较完整。单独做一次“全局关闭后恢复”的实验证据很弱，还会制造不必要的反欺骗缺口。

如果 mark 用于选择路由表，应分别在存在和不存在该 mark 时重复查路与抓包。默认情况下，反向路径查找不使用 mark。若策略路由确实要求两个方向一致使用该 mark，可以按内核 [`src_valid_mark` 文档](https://docs.kernel.org/networking/ip-sysctl.html#src-valid-mark-boolean)把它纳入校验。

## 在正确层次修复不一致

建议按下面的优先级处理：

1. **让 FIB 表达真实源路径。** 修正或安装 per-Pod、prefix、VRF 或 policy route，使源地址的最佳反向查找经过回包实际进入的设备。这样既保留严格源地址校验，也让其他路由工具与数据路径得出一致结论。
2. **让策略路由输入保持对称。** 如果 mark 本来就用于选择路由表，应在相关路径保留它；只有在证明正反向查找都应使用该 mark 后，才启用 `src_valid_mark`。只在一个方向出现的 mark 可能让校验更不准确。
3. **只在特定入口使用宽松模式。** 如果非对称是有意设计，源地址可能合法地拥有不同的最佳返回接口，那么宽松模式既允许非对称，又保留“源地址通过某条路径可达”的检查。由于存在取最大值规则，需要同时设置并复查 `all` 和接口值。
4. **只有存在替代信任边界时才关闭检查。** CNI 可能已经在入口验证 endpoint 身份、源前缀和 policy。如果这些保护确实完整，在专用虚拟入口关闭 `rp_filter` 可以是合理选择；不要全局关闭，也不要影响无关 uplink。
5. **重新审视 redirect 与挂载拓扑。** 如果重新注入让包出现在既非预期信任边界、也非 FIB 反向路径的设备上，调整挂载点或 redirect 设计，通常比不断增加 sysctl 例外更清楚。

不要把 SNAT 当成首选补丁，仅仅因为改写源地址会让反向查路通过。它会改变 policy、可观测性和应用看到的来源身份，还可能掩盖不一致的路由设计。Kubernetes 的 [Service 源 IP 行为指南](https://kubernetes.io/docs/tutorials/services/source-ip/)说明，在 iptables 模式下，集群内部 ClusterIP 流量通常不会做源 NAT。

## 怎样测试才能确认修复完成

一次同节点 TCP 握手远远不够。至少应覆盖：

- 直接 Pod IP 与 Service ClusterIP；
- 同节点与跨节点 endpoint；
- 本地与远端 Service backend；
- 每个 CNI-facing 设备或 VRF 的入口；
- 带预期 mark 的策略路由，以及故意缺少 mark 的情况；
- 严格模式与实际计划上线的源地址验证配置；
- 双栈集群中的 IPv4 与 IPv6 独立路径；以及
- endpoint 替换、节点重启和路由重新收敛。

每个用例都应断言验证点看到的包 tuple 与入接口、反向 FIB 结果、实际丢包原因和最终应用结果。这样才能区分四种在客户端看来完全一样的故障：Service 转换、eBPF redirect、反向路径校验，以及更晚发生的 policy 或 forwarding 问题。

## 参考资料

- [Linux 内核文档：IPv4 `rp_filter` 模式与有效值规则](https://docs.kernel.org/networking/ip-sysctl.html#rp-filter-integer)
- [Linux 内核源码：FIB 源地址校验与 `IP_RPFILTER` 返回](https://github.com/torvalds/linux/blob/master/net/ipv4/fib_frontend.c)
- [Linux 内核源码：反向路径过滤的 `skb_drop_reason` 定义](https://github.com/torvalds/linux/blob/master/include/net/dropreason-core.h)
- [iproute2 手册：使用 `ip route get` 解析 FIB 查找](https://man7.org/linux/man-pages/man8/ip-route.8.html)
- [Kubernetes 文档：虚拟 IP 与 Service proxy](https://kubernetes.io/docs/reference/networking/virtual-ips/)
- [Kubernetes 文档：Service 的源 IP 行为](https://kubernetes.io/docs/tutorials/services/source-ip/)
- [OpenTelemetry 规范：插桩库、版本与稳定性](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/overview.md)

## 当日社区讨论

本次通过普通可见浏览器检查了全部 6 个批准社区和 15 个白名单频道或公开页面，所有目标均可访问。选中的网络故障出现在严格的过去 24 小时窗口内，因此没有使用七天回退。以下综合分析已删除参与者、项目、雇主和频道身份、消息链接、精确时间、私有拓扑、原始日志以及可回搜的原文措辞；没有保存原始 transcript，也没有执行回复、表情互动、私信、关注、邀请或管理操作。

### 返回路径校验是最明确的实践故障

窗口内最强的排障报告沿着 Service 请求检查了地址转换与 eBPF endpoint routing，随后发现 Pod 的 TCP 回包在 host return path 消失。关键线索是 BPF 路径与普通反向 FIB 查找对入接口的判断不一致。前文诊断顺序把这条线索变成可证伪测试：先定位准确丢失边界，再用真实入接口复现反向 tuple 查路，最后确认内核 drop reason，然后才修改策略。原始报告并未证明所有拓扑都有同一原因，因此这里的结论只适用于这三项观察一致的路径。

另一个网络项目描述了 eBPF load balancer：用户态健康检查会把紧凑的 backend 选择写入 map。这也暴露了一个相关可靠性问题：health bit 只能证明探测层观察到的状态，不能证明 NAT、返回路由、邻居解析和源地址校验共同组成了一条可用连接。因此，端到端测试应断言真实流量的两个方向，而不只是 map 内容或 backend health endpoint。

### 内核审查集中在边界与生命周期故障

公开内核开发归档当天的活跃主题包括限制 resizable map 迭代、处理 batch 与 queue 操作的整数溢出、socket 引用生命周期、callback 与 tail call 限制、JIT 内存检查，以及选择性加载模块 BTF。这些主题体现了同一种审查模式：只有显式处理异常边界和所有权转换，快路径才是安全的。对应到网络诊断，把“早期 eBPF hook 返回成功”当成后续 IPv4 校验与路由必然成功，同样是错误推理。

### 插桩维护者讨论 beta API 覆盖范围

一个可观测性工作组讨论了自动插桩是否应覆盖 beta API，同时排除 legacy surface。这里需要区分被插桩 SDK 的成熟度与 telemetry schema 的成熟度：支持 beta SDK 调用，并不要求把相应 span 结构承诺成稳定契约。实用实现可以隔离 beta patch、按上游版本启用、在符号不存在时 fail open，并对明确版本区间运行兼容性测试。尚未解决的部分取决于具体上游：没有公开兼容政策时，应测量维护成本与破坏频率，而不能靠猜测。

同一讨论还提醒维护者注意尚未经过人工审查的批量生成工作项。生成报告可以帮助收集线索，但是否可进入实现阶段，应由已复现行为、有边界的兼容性声明和能够独立审查的改动决定，而不是由生成文本的数量或语气决定。

### 安静目标也完成了检查

项目帮助与功能区在窗口内为空、安静，或只有自动构建通知。调度器支持区没有新的 24 小时技术问题；一个 eBPF 聊天区也没有比已经回答的 tail-call 问题更新的讨论。这些是已访问但安静的结果，不是把覆盖缺口报告成零活动。
