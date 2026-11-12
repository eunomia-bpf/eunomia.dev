# 为什么 UMEM 的 chunk 尺寸会封顶 AF-XDP 零拷贝模式下最大的包，FILL 环又该怎样一直喂饱，内核才不会静默丢入向流量？

零拷贝模式下内核什么也不拷贝：它只是把收到的每个包交给应用——通过让描述符指向 UMEM 里一个空闲的 chunk。所以 chunk 尺寸（只有 2K 或 4K 两档）就是单个 buffer 能装下最大包的上限，而 FILL 环的深度则决定了入向何时丢包。一个装不下单个 chunk 的包，要么开多 buffer 模式（`XDP_USE_SG` 加 `xdp.frags` 程序段），要么整个被丢掉；而丢包在应用侧是静默的，除非你读 `XDP_STATISTICS` 计数器。真正要盯的那个数字是 FILL 环：它必须一直被喂满，因为零拷贝下内核只能把数据放进应用已经提供出来的 chunk。

## 机制

UMEM 是一段虚拟连续内存，被切成等大的 chunk（文档里也叫 frame）；每个环里的描述符用一个整块区域内的字节偏移来引用某个 chunk。一个 socket 绑定到一个 UMEM 上，且绑定到某一个 netdev 的某一条 queue；两条单生产者/单消费者环在 kernel 与 user space 之间传递 chunk 的所有权。FILL 环是从 user space 往 kernel 送的方向：应用提交 chunk 地址，kernel 把收到的数据填进去。COMPLETION 环是反方向：kernel 用完 chunk 后把它们还回来，包括那些引用了非法 TX 描述符、被 kernel 拒绝并回收的 chunk。

零拷贝模式在 bind 时选定。bind 时 kernel 会先尝试零拷贝；如果设备不支持，就回退到 copy 模式（把每个包都拷到 user space）。`XDP_COPY` 强制 copy 模式，copy 模式不可用时 bind 失败；`XDP_ZEROCOPY` 强制零拷贝，不可用时 bind 失败。这决定了丢包问题的核心：零拷贝下 kernel 并不持有包的拷贝，它只能把一个收到的包放进应用已经放进 FILL 环的那个 chunk。所以 FILL 环就是吸收突发流量的缓冲：应用回填得慢了，kernel 下一个包就无处安放，这个包被丢掉。这个丢包应用侧看不到，它只是被记在 socket 上。

chunk 尺寸不是随便选的：只能是 2K 或 4K。128K 的 UMEM 配 2K chunk，能装 128K/2K = 64 个包，单个 buffer 最大的包就是 2K。所以 chunk 尺寸就是单 buffer 模式下包大小的天花板。要收巨型帧，得用 `XDP_USE_SG` bind 标志开多 buffer 模式，并把 XDP 程序放进 `xdp.frags` 段：一个包变成一串 2K 或 4K 的 frame（一个 9K 巨型帧就是三个 4K chunk），最后一个 frame 用 `XDP_PKT_CONTD` 为 false 标记结束；不开这个，kernel 照旧把多 buffer 的包整个丢掉。

## 验证与调试路径

1. 确认你真的在零拷贝模式。`XDP_OPTIONS` getsockopt 会报告 `XDP_OPTIONS_ZEROCOPY`。如果设备不支持零拷贝，kernel 已经回退到 copy 模式，上面这套 chunk/所有权模型跑的不是你以为的那个；要零拷贝就用 `XDP_ZEROCOPY` 强制。
2. 读 `XDP_STATISTICS`。它暴露 `rx_dropped`（非非法描述符导致的丢包）、`rx_invalid_descs`、`tx_invalid_descs`。负载下 FILL 环干涸时 `rx_dropped` 爬升，是 UMEM 深度不足的典型信号；`rx_invalid_descs` 爬升则指向应用提交的描述符在 chunk 尺寸、对齐或 headroom 上不匹配，而不是容量问题。
3. 盯应用侧的 FILL 环深度。零拷贝下应用必须持续回补 FILL 环。FILL 环趋近于空，是 `rx_dropped` 即将爬升的前兆：kernel 没有空闲 chunk 可以交给下一个包。
4. 把丢包和发送失败分开。COMPLETION 只表示 kernel 用完这个 chunk，不代表包发出去了。completion 把 chunk 所有权还给 user space，并不保证发送成功，所以别把 COMPLETION 环当成交付证明。

## 局限

- chunk 尺寸固定为 2K 或 4K，单个 chunk 永远装不下巨型帧。巨型流量必须 `XDP_USE_SG` 多 buffer 模式，程序放在 `xdp.frags` 段；多 buffer 下只有当一个包的所有 frame 都放得下时整个包才会被交付，RX 环没空间时这个包的每个 frame 都会被丢掉。
- 零拷贝是设备能力。驱动没有它时，socket 会静默回退到 copy 模式，除非你强制 `XDP_ZEROCOPY`；在确认 `XDP_OPTIONS_ZEROCOPY` 已置位之前，就当作 copy 模式来算。
- FILL 和 COMPLETION 环是单生产者/单消费者的。跨进程共享一个 UMEM 时，只有一个进程拥有这两条环，其它进程不能并发使用；libbpf 对此没有提供同步原语。
- completion 不等于交付。COMPLETION 环在 kernel 用完后归还 chunk 所有权，对 TX 而言这不代表包已发出，所以交付核算得靠驱动，不能靠环。

## 参考

- [Linux 内核文档：AF_XDP sockets](https://docs.kernel.org/networking/af_xdp.html) — UMEM 的 2K/4K 等长 chunk、FILL 与 COMPLETION 环的所有权模型、bind 时零拷贝的选定（`XDP_COPY` / `XDP_ZEROCOPY`）与对 copy 模式的回退、`XDP_STATISTICS` 丢包计数、`XDP_USE_SG` 多 buffer 路径与 `xdp.frags` 程序段，以及 completion 不等于交付的注意事项。
- [xdp-project AF_XDP 示例](https://github.com/xdp-project/bpf-examples/tree/main/AF_XDP-example) — 内核文档指向的 user space 与 XDP 程序对，给出 AF_XDP 完整搭建与使用示例。

## 当日社区讨论

本次选中的问题不在今日快照里。两个 opt-in archive 渠道产出 7 条消息，全部属于两个早已发布的线程：OBI 的 Kubernetes 缓存地址环境变量在 Config v2 文档下被忽略（现已带有一个 helm-charts 拉取请求和一个已确认的手动 workaround），以及那个记录智能体调用开始时可用技能、让 trace 能区分"压根没提供"和"提供了但更弱的候选赢了"的拉取请求。archive 窗口没有新问题，所以本页回退到一个反复出现的 AF_XDP 零拷贝 sizing 实践问题，完全以公开的内核文档与上游示例为依据：chunk 尺寸封顶单 buffer 最大包，FILL 环深度是决定入向何时丢包的突发上限，而这个丢包不计数就不可见。

本次渠道覆盖：两个 opt-in archive 提供 7 条消息，全部是上面点名的两个已发布线程。可见浏览器专属渠道（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能复查，因为没有可见浏览器会话，所以标记为"未覆盖"，而不是"安静"。
