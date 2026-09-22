# 为什么 TLS 加密的服务间 trace 无法拼接，而零代码 eBPF 追踪中真正跨越加密边界的又是哪个组件？

**简短回答：** 因为 trace 上下文放在 HTTP header 里，而 TLS 恰好加密的就是这部分 payload。零代码的 eBPF tracer 无法在内核里解密，因此既读不到 `traceparent`，也写不进密文。桥梁不是 payload，而是传输层：OpenTelemetry eBPF Instrumentation (OBI) 把上下文放在 **TLS 之前附着在连接上的自定义 TCP option（kind 25）** 上——内核无论 payload 是否加密都能看到它；此外对 Go 应用还有一条 userspace uprobe 路径，在 TLS 加密之前把 header 写进应用 buffer。代价是：这条通道是 OBI 私有的（只有其他 OBI 插桩端点能读懂），且是连接级的，因此经过会丢弃并重放报文的 L7 代理就会断掉，也无法在一条 HTTP/2 或 gRPC 连接上表示多个并发 stream。

## 为什么加密边会打断基于 header 的 propagation

零代码分布式追踪的原理是把 W3C 的 `traceparent` 值序列化到请求跨越的边界上：eBPF 程序读入传入的上下文，把它带过进程，再写到发出的请求上。在明文 HTTP/1 中，请求字节（包括 header）在内核 socket buffer 里是明文的：OBI 对 `tcp_sendmsg`/`tcp_recvmsg` 的 `kprobe` 直接可见，`sk_msg` 程序（tpinjector）还能扩展报文、补上应用没写的 header。任何 W3C SDK 对端都能读懂它，因为 `traceparent` 是标准 header。

TLS 把情况反过来。应用的 TLS 库在内核 socket buffer 之前就加密了请求；对内核而言这些字节是密文：

- 对 `tcp_sendmsg`/`tcp_recvmsg` 的 `kprobe` 解析不出 HTTP header；
- `sk_msg` 注入器无法把 `traceparent` 塞进密文；
- 接收端同样读不到对端应用加密过的上下文。

于是 header 注入——最主要的 W3C 兼容通道——在加密边上彻底不可用。这是 TLS 终结点位置（userspace，在 eBPF 探针之上）决定的性质，不是配置缺口：再多挂探针，内核也解不了密。

## 加密边上的上下文实际靠什么传递

OBI 的 [context propagation 架构](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/context-propagation.md) 文档列了三种注入方式，每种 TLS 情形落点不同：

1. **HTTP/1 header（L7）**——明文 HTTP 里的 `traceparent`。W3C 兼容，任何 SDK 插桩端点都能读。TLS 上不可用。
2. **TCP option（L4）**——自定义 TCP option，kind 25，写进连接的 segment。它位于 TLS 之前（IP 包层面），因此不受 payload 加密影响。发送端 tpinjector 调度该 option，接收端的 `BPF_SOCK_OPS` 程序在 ingress 时把它解析进 `incoming_trace_map`。两个硬限制：option 是连接级的；只有 OBI 插桩端点认识 kind 25，所以上下文到不了纯 SDK 端点。OBI 官方文档的直接表述：TLS 场景下 OBI 把信息"injects the information at TCP/IP packet level"（在 TCP/IP 包层面注入），且"is only able to send the trace information to other OBI instrumented services"（只能把 trace 信息发给其他 OBI 插桩服务）。
3. **按 stream 的 HPACK 注入（L7，多路复用协议）**——HTTP/2 与 gRPC 一条连接承载 N 个并发 stream，单个连接级 TCP option 无法表示 N 个互不相同的 trace 上下文（"a connection-scoped option cannot represent N concurrent stream contexts"）。OBI 改为通过 `bpf_msg_push_data` 把按 stream 的 `traceparent` HPACK 字段写进出站 HEADERS frame，这是多路复用 HTTP/2 唯一的网络机制。

按 stream 的 HPACK 路径再次撞上 TLS：generic（非 Go）的 TLS HTTP/2 里，HPACK 拼接会落进密文，行不通——文档给出的边界是 "Generic non-gRPC HTTP/2 context propagation remains limited to Go library instrumentation"（generic 非 gRPC 的 HTTP/2 上下文传播仅限于 Go 库插桩）。Go 之所以例外，是因为一条 userspace uprobe 进入 Go 的 HTTP/TLS 层，在 TLS 加密之前把上下文写进 Go 的明文请求 buffer（Go 的 `persistConnRoundTrip` 路径用 `bpf_probe_write_user` 写应用 buffer），因此加密线上传输的是接收端 TLS 层最终会暴露出来的 header。

Ingress 侧采用"last one wins"：`BPF_SOCK_OPS` 先解析 TCP option，`kprobe`/`protocol_http` 后解析 HTTP header 并覆盖，于是最可靠的方式（标准 header）在两者并存时自然胜出。值得注意的是，从 OpenTelemetry SDK 插桩服务读入传 `traceparent` "still work"（仍然有效）——只要对方在明文中发标准 header，即使出站 TCP option 通道被干扰也能拼上。

middlebox 的告诫是最容易踩的实操细节。已建立连接上的自定义 TCP option 在很多网络路径上不会被端到端保留：L7 代理与负载均衡器会丢弃原始报文并在新连接上重放，部分 middlebox 或托管端点会丢弃或重置携带未知 option 的 segment——客户端看到 `connection reset by peer`，通常发生在连接后第一个请求上，且表现为间歇性。文档给出的对策：插桩服务经过这类中间层时，用 `headers` 且不开 `tcp`；TCP option 只在两端都是 OBI 且路径保留 option（通常是没有 option-stripping middlebox 的直连 L2/L3 网络）时才安全。

## 如何判断自己落在哪种情形

1. **看配置状态。** OBI 的网络级上下文传播默认关闭。通过 `OTEL_EBPF_BPF_CONTEXT_PROPAGATION=all`（或子集）或 OBI 配置的 `context_propagation` 键开启（`all`、`headers`、`tcp`、`headers,tcp`；旧的 `http` 别名已移除，弃用的 `ip` 值无效）。
2. **读线。** 在两个 TLS 插桩端点之间抓包，发送端 segment 上应能看到 kind 25 的 TCP option。若 option 在经过 L7 代理或负载均衡器后消失，那个代理就是边界：它之后的边无法承载 L4 上下文，trace 就在那断掉。
3. **盯住 reset 特征。** TLS 路径上开启 `tcp` 后、间歇性出现连接后首请求的 `connection reset by peer`，就是文档描述的 option-stripping middlebox 症状；从传播模式里去掉 `tcp` 即可消除。
4. **确认接收端。** OBI 在 header 与 TCP option 两条通道上自动解析传入的 `traceparent`，因此即使 L4 通道已断，发标准 header 的 SDK 上游仍能正确拼接——检查 server span 是否带上了上游 trace ID。
5. **核对协议与运行时。** TLS 上的 HTTP/2 或 gRPC，只有在对端是 Go 插桩（uprobe 先写后加密）时 trace 才拼得上；其他运行时的 generic TLS HTTP/2 客户端处于文档所述限制内，其按 stream 的上下文无法在零代码层跨加密边注入。

## 答案的边界

- kind 25 的 TCP option 是 OBI 私有机制，不是 W3C 机制：纯 SDK 端点读不懂，上下文在那条边界上被静默丢弃。
- 终结并重建 TCP 连接的 L7 代理/负载均衡会打断 L4 通道，trace 恰好断在代理处。
- 会 strip option 的 middlebox 与托管端点可能把该通道表现为连接 reset 而非干净的"不传播"。
- 多路复用的 TLS HTTP/2 与 gRPC 上下文只能经 Go 的 userspace uprobe 路径注入；其他运行时在零代码层没有 TLS 可用的按 stream 机制。
- 以上都不影响接收端读标准 header：任何 W3C 合规发送方（SDK 或 OBI）的传入 `traceparent`，只要字节对探针可见，就能在明文位置被解析。

## 参考资料

- [OBI context propagation 架构（devdoc）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/context-propagation.md)
- [OBI gRPC/HTTP2 context propagation（devdoc）](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/grpc-context-propagation.md)
- [OBI distributed traces 文档（TLS 限制与 `context_propagation` 配置）](https://opentelemetry.io/docs/zero-code/obi/distributed-traces/)
- [W3C Trace Context 规范](https://www.w3.org/TR/trace-context/)

## 当日社区讨论

如实的覆盖说明：两个 watchlist 选中的 Slack 存档在滚动 7 天窗内今天**零消息**，allowlist 里的 Discord 频道仅限 visible-browser，且本次运行没有可用的浏览器会话——因此 2026-09-22 没有任何私有社区材料可用。上文问题是回退选择：这是被监控的 OpenTelemetry eBPF 社区中一个真实、反复出现且仍未解决清楚的边界，完全以上述公开一手资料为依据，而不是任何具体 thread。

该社区反复出现的从业者症状——"服务间 trace 在 TLS 边就断了"——正是文档所述边界的落点：header 传播在 TLS 终结处失效，L4 TCP option 是 TLS 唯一的零代码桥梁，而这座桥有明确的兼容性面（OBI-only 端点、保留 option 的路径、多路复用协议上的按 stream 限制）。公开文档中反复出现的两个主题：(1) option-stripping middlebox 造成的间歇性"连接后首请求 reset"，在 L7 代理后开启 `tcp` 时极易被误读为应用不稳定；(2) 任何非 OBI 端点处的静默断开——TCP option 在那里被直接忽略而非拒绝，于是 trace 在一侧看起来完整、另一侧成为孤儿。文档本身留下的未解问题是：L4 通道在有状态负载均衡与改写 segment header 的 NAT 设备上如何表现；文档给出的应对是退回 `headers`，并接受非 OBI 或经代理端点会断链。
