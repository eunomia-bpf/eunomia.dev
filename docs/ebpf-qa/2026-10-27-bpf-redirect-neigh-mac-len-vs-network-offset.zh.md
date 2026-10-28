# BPF 重定向到邻居的辅助函数为什么会在以太网头很短的包上触发内核 BUG，修复又为什么在接收路径和发送路径上读取真实头部长度要使用不同的字段？

因为 `skb->mac_len` 只在接收路径上被填充。一个用 `mac_len` 是否小于 `ETH_HLEN` 为 `bpf_redirect_neigh()` 的 pull 做保护的修复，在 ingress 上是安全的；但在发送路径上——本地 socket 生成、经 tc egress 出去的包，或经 netkit 转发的包——每个 skb 的 `mac_len` 都是 0，所以同一个保护会把这些路径上所有合法的 redirect-neighbor 流量静默丢掉。在两条路径上都有效的头部长度字段是 `skb_network_offset()`，即从 `skb->data` 到 `network_header` 的距离；修正后的修复改为从这个字段读取长度。

## 机制

1. Bug 本体。`bpf_redirect_neigh()` 经由 `skb_do_redirect()` 进入 `net/core/filter.c` 的 `__bpf_redirect_neigh()`。原代码是 `ethh = eth_hdr(skb)`，只检查了 `mac_header` 低于 `network_header`，然后调用 `skb_pull(skb, sizeof(*ethh))`——即一次 14 字节 `ETH_HLEN` 的 pull。`__skb_pull()` 执行 `skb->len -= len` 后，若在线性部分下溢就 `BUG()`。VLAN pop-eth 之后，一个特权 tc 程序可以把 skb 的有效 MAC 头缩到 1 字节、线性 headroom 只剩 1 字节（复现器是 `bpf_skb_change_head(skb, 1, 0)`），再调用 `bpf_redirect_neigh()`；旧检查通过，14 字节 pull 触发 BUG。

2. v1 修复与它的回归。修复在 pull 前加保护、其后重读 `ethh`：当 `mac_len` 小于 `ETH_HLEN`、或 `pskb_may_pull()` 无法把头部线性化时丢弃，然后 `ethh = eth_hdr(skb);`。`pskb_may_pull()` 不 panic 地线性化头部，`mac_len` 检查丢掉真正过短的包。但 `skb->mac_len` 只在接收路径被写入，由 `__netif_receive_skb_core()` 调用 `skb_reset_mac_len()` 把它设为 `network_header` 减 `mac_header`。发送路径没有任何地方写 `mac_len`；`__dev_queue_xmit()` 和 `dev_loopback_xmit()` 只调用 `skb_reset_mac_header()` 而不调用 `skb_reset_mac_len()`，所以本地生成的包携带 `mac_len == 0`。在 tc egress 和 netkit 上，`0 < ETH_HLEN` 恒为真，每个合法的 redirect-neighbor skb 都会落到丢弃路径（`kfree_skb`、`-ENOTSUPP`）——一个也会进入 stable 的回归。

3. 与路径无关的修复。`__bpf_redirect_neigh()` 运行时，两条路径上 `skb->data` 都停在 L2 头起始处：ingress 上 `sch_handle_ingress()` 在 `skb_do_redirect()` 前 push 了 `mac_len` 字节；egress 上 `dev_hard_header()` 已 push 好 L2 头，`__dev_queue_xmit()` 把 `mac_header` 设到那个偏移。所以真实的 L2 头长度就是 `network_header` 减 `data`，恰好是 `skb_network_offset()`。修正后的保护在 `skb_network_offset()` 小于 `ETH_HLEN`、或 `pskb_may_pull()` 无法线性化时丢弃；它仍拒绝来自 `bpf_skb_change_head` 的 1 字节头，在 `mac_len` 为 0 的 egress 与 netkit 上同样正确。

4. 内核本来就清楚这点。`act_mirred.c` 在 ingress 取 `mac_len`、其余路径取 `skb_network_offset()`——它早就把 ingress 之外的 `mac_len` 视为不可靠。而 `bpf_redirect_neigh` 的 selftest 全部挂在 tc ingress，那里 `mac_len` 有效，因此抓不到 egress 或 netkit 的回归。

## 验证与排障路径

1. 看哪个字段是 0。在 egress 或 netkit 路径上，观察包上 `mac_len`（0）与 `skb_network_offset()`（真实 L2 头长度），看哪个匹配真实头部。egress 路径从不调用 `skb_reset_mac_len`，所以 `mac_len` 按构造就是 0。

2. 确定性复现原 BUG。在 tc ingress 挂一个特权 SCHED_CLS 程序，做 VLAN pop-eth，调用 `bpf_skb_change_head(skb, 1, 0)` 再调用 `bpf_redirect_neigh()`；在未修复的内核上，`__skb_pull()` 内部的 `skb->len` 下溢检查会触发。

3. 证明回归。把同一个 redirect-neighbor 程序挂在 tc egress 上处理本地 socket 生成的包，或经 netkit 转发，观察 v1 的 `mac_len` 保护以 `-ENOTSUPP` 丢掉该包；`skb_network_offset()` 版本则保留它。

## 局限

- 该保护仍会丢掉任何 L2 头短于 `ETH_HLEN` 的 skb；这是有意为之，不是缺陷。
- `pskb_may_pull()` 可能重新分配 `skb->head`，所以修复在其后重读 `eth_hdr(skb)`，而不是复用保护前取的指针。
- `mac_header` 高于 `network_header` 是另一种失败模式，与头部过短不同；`skb_network_offset()` 覆盖的是过短的情况，两者不是同一个检查。

## References

- [Kernel `net/core/filter.c`](https://github.com/torvalds/linux/blob/master/net/core/filter.c) — `__bpf_redirect_neigh` 与原始的 `skb_pull(ETH_HLEN)` 路径。
- [Kernel `include/linux/skbuff.h`](https://github.com/torvalds/linux/blob/master/include/linux/skbuff.h) — `skb_reset_mac_len`、`skb_reset_mac_header`、`skb_network_offset`、`__skb_pull`。
- [Kernel `net/sched/act_mirred.c`](https://github.com/torvalds/linux/blob/master/net/sched/act_mirred.c) — ingress 取 `mac_len`、其余路径取 `skb_network_offset()` 的既有写法。
- [Kernel `include/uapi/linux/if_ether.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/if_ether.h) — `ETH_HLEN`（14）与 `ETH_ALEN`（6）。
- [Commit `b4ab31414970` ("bpf: Add redirect_neigh helper as redirect drop-in")](https://github.com/torvalds/linux/commit/b4ab31414970) — redirect-neighbor 辅助函数的 `Fixes:` 目标。
- [Public bpf mailing-list patch thread: fix `skb_pull()` BUG in `__bpf_redirect_neigh()`](https://lore.kernel.org/bpf/bpf_redirect_neigh_short_linear.v1.r0.20261028033717-1@mojatatu.com/) — 公开的补丁与 egress/netkit 回归评审。

## 当日社区讨论

一封公开的 bpf 列表补丁修复了 `__bpf_redirect_neigh()` 里的 `skb_pull()` BUG：特权 tc 程序在 VLAN pop-eth 之后把 MAC 头缩短即可触发。第一版用 `mac_len` 是否小于 `ETH_HLEN` 来保护 pull，但自动评审发现 `mac_len` 只在接收路径上被设置，在 tc egress 与 netkit 上恒为 0，这个保护会丢掉那些路径上所有合法的 redirect-neighbor 流量；线程达成共识，后续版本改为从 `skb_network_offset()` 读取长度，该字段在两条路径上都有效。目前的讨论范围很窄：边界在哪一个字段承载每条路径的 L2 头长度，而现有的 redirect-neighbor selftest 全部挂在 ingress，因此抓不到 egress 的回归。

本次运行的存档来源在 7 天窗口内产出 0 条消息（两个 opt-in 存档来源在快照时为空），因此上面的问题取自被允许的公开 bpf 邮件列表，经其公开的 atom feed 审阅，补丁线程作为一手参考。eunomia-bpf 与 sched-ext 的 Discord 频道以及 r/eBPF 在本次运行未覆盖（没有可用的可见浏览器会话），故标记为未覆盖而非安静。
