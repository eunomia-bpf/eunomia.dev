# 为什么在 TLS uprobe 捕获下，TLS 连接上第一个之后的每个 HTTP/2 请求都会 HPACK 解码失败，解码器应如何保持每连接状态？

这不是解码器的 bug。HTTP/2 的字段压缩（HPACK）是有状态的，而状态属于整条连接，不属于捕获管线发出的某一条 TLS 记录或某个事件。RFC 9113 的字段块规则写明：字段压缩是 "stateful. Each endpoint has an HPACK encoder context and an HPACK decoder context ... that is used for all field blocks on a connection"，并且 "a decoding error in a field block MUST be treated as a connection error"——具体到 HPACK 就是 `COMPRESSION_ERROR`（RFC 9113 §4.3 / §7.2，解码器无法维持字段节压缩上下文时）。RFC 7541 把归属关系说得更具体：动态表 "specific to an encoding or decoding context"，全新上下文的静态表和动态表都是空的，被索引的字段名可以指向静态表加动态表的合并空间。如果捕获管线每次 uprobe 事件（或每个接到事件的 worker）都重建一个解码器，它的动态表每次都是空的：连接上的第一个头块如果只引用静态表字段就能解出来，而第一个引用动态表索引的头块（对端在连接建立后已经写进了动态表）就是解码错误。所以要求是每条连接一个解码器实例，并按字节顺序喂入该连接的事件——任何丢失的事件都要把这条连接的解码器标记为不可信，而不是静默重放或跳过。

## 机制

捕获侧解码器需要镜像的 HPACK 状态机（来自 RFC 7541）：

- §2.2：编码端和解码端的动态表 "completely independent"——编码端（对端 TLS 栈）根据它实际发出的内容建立动态表，解码端必须按顺序消费头块才能重建同样的表。
- §2.3.2：解码端的动态表属于它自己的解码上下文，初始为空，按 FIFO 维护："as each header block is decompressed"，被索引的条目加入表头，超出表大小时从表尾逐出最久未用条目。
- §2.3.3 与 §4.1：索引空间是静态表条目接动态表条目；每个动态条目占条目大小 32 字节加字段名和值，受动态表大小约束（HTTP/2 `SETTINGS_HEADER_TABLE_SIZE` 默认 4096）。
- §3.2：任何畸形头块都是解码错误，而 HTTP/2 把它升级为连接错误。§6.1 补充：索引 0 保留不用，超过静态表加动态表大小的索引是错误。

所以故障是结构性的。当捕获管线对每条 `gnutls_record_send` 发出的 TLS 记录（或每个被 worker 拿到的事件）都新建解码器时，每个记录开始时解码器的动态表都是空的。连接上的第 1 个请求通常只引用静态表字段（method、path、host），能解出来；第 2 个请求的头块引用了编码端在更早头块里创建的一个动态表索引——新建的解码器从未见过——解码器就报出非法的索引表示。同一条连接上后续每个请求都这样失败，对端自己看到的 HTTP/2 错误就是 `COMPRESSION_ERROR` 连接错误。这正是 eCapture 长期 issue 744 里看到的公开症状：TLS 上的 HTTP/2 追踪在每条 TCP 流上丢失第一个请求之后的所有请求包。那里的诊断是同一机制："the deep reason is hpack dynamic table cannot be shared"——动态表无法在事件处理器之间共享；尝试过的修法（用 4-tuple 或 `struct sock` 指针给每连接解码器做 key、每连接一个共享动态表的事件 worker）就是下文的设计空间。

该 issue 还记录了 keying 的陷阱：`struct sock` 指针不是稳定身份，内核套接字分配器（`net/core/sock.c` 的 `sk_prot_alloc`）没有文档化保证一个被释放的地址不会被新套接字复用。同样的推理适用于 GnuTLS 会话指针：`gnutls_session_t` 是 `gnutls_record_send` 的第一个参数，在 uprobe 事件里随手可得，但没有任何文档化保证保证会话指针在释放后不会被复用。GnuTLS 自己的清理路径 `gnutls_deinit` 被文档描述为 "clear all buffers ... remove session data from the session database"——每连接解码器应该在解绑时丢弃，而不是靠某种 LRU 逐出。健壮的身份是传输层绑定：`gnutls_transport_set_ptr(session, ptr)` 和 `gnutls_transport_set_int(session, fd)` 是传输绑定入口，探测后者加 `gnutls_deinit`，就能得到从 `(pid, transport fd)` 到解码器状态的映射，这个映射在会话指针失效后仍然成立。有一个绕开 GnuTLS 探测时要知道的细节：选择 GnuTLS 后端的 curl 这类 TLS 客户端通过 `gnutls_transport_set_ptr(session, cf)` 绑定自己的内部上下文结构，而不是 `int` fd，所以这类进程的记录函数参数里没有可用的文件描述符——映射探测必须以实际发生的传输绑定为准，而这一点同样可以通过这两个传输函数观察到。

顺序是第二个结构性问题，且与正确性正交：即使 key 对了，解码器的好坏也只取决于它收到的头块顺序和完整性。多个 worker 读取的 per-CPU perf 事件数组会乱序交付，溢出会永久丢记录。内核 ring buffer 文档正是为此动机引入它的："more efficient memory utilization by sharing ring buffer across CPUs" 加上 "preserving ordering of events ... even across multiple CPUs"，而 perf buffer "fails to satisfy both"（ring buffer 是 MPSC 环形缓冲）。与 HPACK 要求匹配的捕获侧设计因此是：

1. 每条连接一个解码器状态，归该连接的 worker 所有。
2. 每条连接一个读取者（一个 worker 按字节顺序排空该连接的事件），而不是可能把一个连接的事件拆到多个线程上的池子。
3. 任何丢失、截断或乱序的事件，都把该连接的解码器标记为不可信并丢弃其后续头块数据，而不是猜测。缺口之后"没报错"的解码并不可信——缺口已经永久改变了对端正在索引的表，丢失的 TLS 记录无法重放。

## 验证与调试路径

首先确认错误是结构性的而非偶发的。把管线改成每连接保留一个解码器，观察故障是否消失：第 2 个之后的请求在每连接状态下能解、在每事件状态下仍然失败，即坐实了动态表机制。具体步骤：

1. 解码器状态以传输绑定为 key，不用会话指针。探测 `gnutls_transport_set_int`（对 `ptr` 绑定的栈再加 `gnutls_transport_set_ptr`）和 `gnutls_deinit`，维护 `(pid, transport fd) → 解码器` 映射，deinit 时丢弃解码器。这与 eCapture issue 744 的每连接 worker 提案同形，只是身份固定到内核给出的东西上。
2. 把多读者 per-CPU perf 数组换成每连接一个读取者——每条连接一个专用 worker，或每 CPU 一个消费者、前面挂 per-connection 队列。内核 ring buffer 文档陈述的设计需要的性质就是：单消费者下跨 CPU 保序。
3. 每条连接加一个存活信号：ring buffer 溢出或序列缺口时，把该连接解码器置为不可信，在它重连（这会重置 HPACK 上下文）前不再输出解码后的头部。这是诚实的做法：缺口之后能"无错解码"的解码器不可信。
4. 用已知公开症状验证：GnuTLS 后端的 HTTP/2 over TLS 客户端，在每事件解码器下应恰好丢失每条连接第一个请求之后的所有请求；在每连接解码器下应全部解出。

如果想干脆绕开 HPACK，另一条路是在编码之前捕获头部：对 HTTP/2 库的请求提交 API 打 uprobe，读压缩前的明文头部数组。对 Go gRPC 栈，Pixie 的 eBPF 方案探测 `loopyWriter.writeHeader`，读 `[]hpack.HeaderField` 切片——文档化的小节是这个方法特定于单个 HTTP/2 库的内部实现，且会在 Go 调用约定变化时失效。对 libnghttp2 栈，公开类比是 `nghttp2_submit_request` / `nghttp2_submit_request2`，签名取 `nghttp2_nv` 头部数组（`session, pri_spec, nva, nvlen, data_prd, stream_user_data`）：uprobe 在库 HPACK 压缩前读这个数组，同类边界同样适用——捕获的是"已提交"的头部而不是内核实际发出的字节，符号集随发行版包和静态链接变化，且每个受支持的 HTTP/2 库都是一块独立维护面。

实现对照用的 Go 参考解码器是 `golang.org/x/net/http2/hpack`：`NewDecoder` 接收 reader，动态表默认 4096 字节，超过静态表加动态表大小的索引表现为 `InvalidIndexError`——这是用户态可见的、与捕获侧解码器撞到的同一个解码错误形状。

## 局限

硬边界是事件丢失，且不可恢复。丢一条 TLS 记录，意味着解码器永久错过了扩展对端动态表的头块；之后任何"干净"的解码都是巧合而非正确。设计的后果是不对称的：正确性是可以工程化的每连接不变量（每连接一个状态、有序喂入、可靠的丢弃信号），但跨有损捕获路径的完整性不是——诚实的行为是标记连接不可信并停止输出解码数据，而不是发出部分或猜测的头部。这个不对称也是 eCapture 文本模式实现被关闭而不是修好的原因："The text mode has many flaws ... temporarily closed"——缺陷正是上面 keying 与顺序两节讨论的东西。

## 参考

- [RFC 7541 — HPACK: Hyperframe Compression](https://www.rfc-editor.org/rfc/rfc7541) — §2.2（编码/解码动态表独立）、§2.3.2（每上下文动态表，FIFO 维护）、§2.3.3（索引空间）、§3.2（解码错误）、§4.1（条目大小）、§6.1（索引表示）。
- [RFC 9113 — HTTP/2](https://www.rfc-editor.org/rfc/rfc9113) — §4.3 / §4.3.1（每连接有状态的字段压缩；解码错误即连接错误）、§6.5.2（`SETTINGS_HEADER_TABLE_SIZE`）、§7.2（`COMPRESSION_ERROR`）。
- [GnuTLS 手册](https://www.gnutls.org/manual/gnutls.html) — §6.5 传输函数（`gnutls_transport_set_ptr` / `gnutls_transport_set_int`）、§6.7 记录与清理函数（`gnutls_record_send`、`gnutls_bye`、`gnutls_deinit`）。
- [eCapture issue 744 — 每条连接第一个之后的 HTTP/2 请求丢失](https://github.com/gojue/ecapture/issues/744) — 公开先例：`COMPRESSION_ERROR` 症状、动态表共享诊断、4-tuple/`sock` keying 尝试、每连接 worker 提案。
- [Pixie — eBPF 的 HTTP/2 追踪](https://blog.px.dev/ebpf-http2-tracing/) — 经 `loopyWriter.writeHeader` uprobe 的明文头部捕获；库特定的维护边界。
- [nghttp2 API 参考](https://nghttp2.org/documentation/apiref.html) — `nghttp2_submit_request` / `nghttp2_submit_request2` 与 `nghttp2_nv` 头部数组。
- [golang.org/x/net/http2/hpack](https://pkg.go.dev/golang.org/x/net/http2/hpack) — `NewDecoder`、`InvalidIndexError`、动态表默认大小。
- [Linux 内核 — BPF ring buffer](https://www.kernel.org/doc/html/latest/bpf/ringbuf.html) — MPSC 环形缓冲、跨 CPU 保序与共享内存动机。

## 当日社区讨论

当日有实质 eBPF 信号的两个讨论来自 opt-in 存档。

第一个，也是本篇选取的问题：一个基于 GnuTLS uprobe 的 HTTPS 传感器——`gnutls_record_send` 捕获加用户态 perf reader 和 HTTP/2 帧解析器，TLS 连接上的第一个请求能解，之后每个请求都失败 `COMPRESSION_ERROR`——即本文描述的每连接 HPACK 解码器状态问题。它的公开先例是 eCapture issue 744，带同样的症状和诊断（hpack 动态表无法跨事件处理器共享，RFC 7541 §2.2），外加 keying 尝试和已关闭的文本模式告诫。

第二个是 LoadBalancer 行为讨论：共享 VIP 下，第二个服务认领已被占用的 frontend 时被拒，报错 "frontend already owned by another service"（公开错误是 `pkg/loadbalancer/errors.go` 的 `ErrFrontendConflict`，在 `pkg/loadbalancer/writer/writer.go` 里被包装），社区陈述的理解是：删除属主服务并不会把认领者重新排队，而对认领者做一次仅改 annotation 的更新大约在七秒内恢复状态、无 BPF flush——怀疑机制在 Kubernetes informer 侧的事件合并。讨论点名的公开文件是 `pkg/loadbalancer/writer/writer.go`、`pkg/loadbalancer/reflectors/k8s.go` 和 `pkg/container/insert_ordered_map.go`。以上按社区陈述的理解呈现，不是已验证的诊断：合并假设尚未在公开源码里钉到具体代码路径。

第三个讨论——ring-0 socket drop 与 context switch 的基准对比——信号太薄未选入，仅在此记为低信号。

本次运行的渠道覆盖：visible-browser-only 来源（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能审阅——没有可用的 visible-browser 会话——因此标记为未覆盖，而非平静。仅两个 opt-in 存档来源可用并已在上面覆盖。
