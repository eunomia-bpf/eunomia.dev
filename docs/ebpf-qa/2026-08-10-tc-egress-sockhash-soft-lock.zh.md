# 为什么在 TC egress 中把 socket 放入 `SOCKHASH` 会导致内核 soft lock？

**简短回答：** 不要把 TC 数据包 hook 当成把活动 TCP socket 加入 `BPF_MAP_TYPE_SOCKHASH` 的控制路径。sockhash 不是一个保存借用 `struct sock *` 指针的普通哈希表。插入 socket 时，内核会持有引用、创建或复用 `sk_psock`、替换 socket callback，并让 socket 继承 map 上挂载的程序。在同一个 socket 已经处于发送路径时执行这次状态转换，可能进入 TC 分类器本不应触发的锁与 callback 交互。程序通过 verifier，并不能证明每个内核版本都能从这个 hook 安全完成 socket 生命周期转换。

连接进入目标 TCP 状态时，应使用 `BPF_PROG_TYPE_SOCK_OPS` 与 `bpf_sock_hash_update()`；也可以由用户态通过文件描述符插入 socket。让 TC 负责观察数据包或实施数据包策略，让面向 socket 的 hook 负责 sockhash 成员关系。

已经观察到的 soft lock 足以说明这里存在内核缺陷或不受支持的交互，但还不足以断言某一条具体锁循环。要定位根因，仍需最小程序、完整内核配置与 commit、watchdog 堆栈，以及与受支持 sockops 插入路径的对照。

## 为什么 sockhash 更新是生命周期操作

内核文档说明，sockmap 和 sockhash 的 value 是 socket 引用。插入时，内核会附加 `struct sk_psock`、替换 socket callback，并继承 map 上的 parser 或 verdict 程序。一个 socket 可以存在于多个 map 中，但只能继承一套 parser 或 verdict 程序；发生冲突时更新会返回错误。

这与向普通哈希表写入标量有本质区别。更新会改变后续数据如何进出 socket，因此下面三类对象的生命周期必须一致：

1. TCP socket 与它的引用计数；
2. sockhash 条目与对应的 `sk_psock`；
3. map 上挂载的 `SK_SKB` 或 `SK_MSG` 程序。

在 TC egress 中，当前 `skb` 已经位于发送路径。`skb->sk` 是与该数据包关联的上下文状态，并不表示可以就地重配 socket callback。改用 `bpf_skc_lookup_tcp()` 再查一次 socket 也不会改变这个边界：返回的引用必须释放，而取得一个带引用的 socket 并不会让 TC hook 变成受支持的 sockhash 注册 hook。

专用 helper 清楚地表达了预期控制路径：

```c
long bpf_sock_hash_update(struct bpf_sock_ops *skops,
                          void *map, void *key, __u64 flags);
```

它的第一个参数是 `struct bpf_sock_ops *`，文档列出的程序类型是 `BPF_PROG_TYPE_SOCK_OPS`。helper 使用该上下文表示的 socket 作为新 value。这给出了安全架构边界：在内核提供 socket 生命周期上下文的地方建立成员关系，而不是在 TC 中通过观察 SYN/ACK 的 ACK 来反推连接建立。

## 更安全的设计

连接建立后，由 sockops 程序填充 sockhash。最小结构可以是：

```c
struct flow_key {
    __u32 local_ip4;
    __u32 remote_ip4;
    __u32 local_port;
    __u32 remote_port;
};

SEC("sockops")
int enroll_socket(struct bpf_sock_ops *skops)
{
    struct flow_key key = {};

    if (skops->family != AF_INET)
        return 1;

    if (skops->op != BPF_SOCK_OPS_ACTIVE_ESTABLISHED_CB &&
        skops->op != BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB)
        return 1;

    key.local_ip4 = skops->local_ip4;
    key.remote_ip4 = skops->remote_ip4;
    key.local_port = skops->local_port;
    key.remote_port = bpf_ntohl(skops->remote_port);

    return bpf_sock_hash_update(skops, &sockets, &key, BPF_ANY);
}
```

把 sockops 程序挂到拥有目标 socket 的 cgroup。需要根据配套的 lookup 或 redirect 程序核对字节序与 key 布局；不同 BPF 上下文中的 `local_port` 与 `remote_port` 并不总是使用相同表示。helper 失败也应作为诊断数据：按操作和 key 冲突模式计数，而不是回到 TC 中静默重试。

如果用户态已经持有或能够发现 socket 文件描述符，也可以从控制面更新 map。这样同样能把 callback 安装移出活动数据包路径。无论由哪一条路径负责插入，都应由同一控制面负责删除与关闭语义。

TC 仍然可以参与，但不负责成员关系。它可以提取 flow key、记录计数器、标记流量或查询普通 map。如果目标是按策略分配 socket，应使用文档明确支持 socket assignment 的 helper 与程序类型，而不是把 sockhash 当成通用指针容器。

## 如何在不丢失证据的情况下排查 soft lock

先删除 TC 中的 sockhash 更新，确认卡死消失；再把同一套 key 与 map 更新迁移到 sockops。这组 A/B 测试能够把数据包解析和 key 构造，与 socket 注册状态转换分离开。

准备可复现的内核报告时，保留：

```console
$ uname -a
$ bpftool prog show
$ bpftool map show
$ bpftool net
$ zcat /proc/config.gz > kernel.config
```

还要捕获完整 soft-lockup watchdog 报告或串口堆栈，包括所有 CPU；如果 lockdep 内核能够复现，也要保存 lockdep 输出。记录 map 是否挂载了 `SK_SKB` 或 `SK_MSG` 程序、socket 是否已经存在于其他 sockmap 或 sockhash，以及把 `skb->sk` 替换为 `bpf_skc_lookup_tcp()` 后堆栈是否变化。不要公开生产地址或流量 payload；用只包含一个 client 和一个 server 的 network namespace 制作 reproducer 更合适。

依次缩减四种场景：

- 没有 parser 或 verdict 程序的空 sockhash；
- 向这个空 map 执行 sockops 插入；
- TC 只观察、完全不更新 sockhash；
- 仍能触发卡死的最小 TC 更新。

如果只有最后一种失败，应把问题报告为 TC 与 sockmap 的交互。如果空 sockhash 在受支持的 sockops 路径也失败，范围就更广。如果必须挂载 verdict 程序才失败，应附上该程序与 attach type，因为 callback 继承本身就是这次转换的一部分。

不要把“另一个内核版本能工作”直接当成结论。只有在两边使用相同配置、程序、attach 顺序和流量序列时，bisect 才有意义。

## 参考资料

- [Linux 内核文档：`BPF_MAP_TYPE_SOCKMAP` 与 `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux UAPI 中的 `bpf_sock_hash_update` 定义](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux 内核 sockmap 实现](https://github.com/torvalds/linux/blob/master/net/core/sock_map.c)
- [Linux selftest：从 sockops 更新 sockmap](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/progs/test_sockmap_update.c)
- [Linux sockmap 与 sockhash selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf/prog_tests)
- [Linux 内核文档：BPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [BPF 邮件列表：防止 `BTF.ext` 边界检查发生溢出](https://lore.kernel.org/bpf/CAOZ_KyvV-Ha7XDZHCUcyYKPMSUvMSrZ0NhmtMVy_KqWD0Ar4kg@mail.gmail.com/T/#u)
- [BPF 邮件列表：多种 JIT 上的 arena atomic fault 处理](https://lore.kernel.org/bpf/84bd0041b0a35b87b15d9b00696b5abe1f16c340d2a02255e65e6758a03d4dc1@mail.kernel.org/T/#t)

## 今日社区讨论

今天通过普通可见界面检查了 6 个获准社区中的 15 个 allowlist 频道或公开页面，均可访问。24 小时窗口整体稀疏，因此按规范使用最近 7 天回退：选中的真实 socket-map 问题仍未解决，并在今天收到新的技术回复。以下内容已删除姓名、账号、雇主、频道身份、消息链接、精确时间、私有拓扑、日志与可回搜措辞；没有保留原始 transcript。

### 缺失的边界是 socket 所有权

最强的未解决实践问题是：近期内核在把一个出站数据包关联的 socket 插入 sockhash 时发生 soft lock。显式重新查找 socket，而不是直接读取数据包上下文，也没有避免故障。最新回复认为内核卡死显然不正常，但讨论中还没有最小 reproducer 或 watchdog 堆栈。

因此，可执行答案必须比“已经找到根因”更克制：把插入迁移到 sockops 或用户态，把 map 缩减到一个 socket 且不挂 verdict 程序，并在尝试修内核前保存锁堆栈。核心风险不在如何取得指针，而在发送路径内部修改 socket callback 与 `sk_psock` 状态。

### 可观测性讨论聚焦边界与最小化

一个可观测性社区继续讨论如何在不保留 credential 值的情况下识别网络中的 credential 类别。合理边界是只输出有界类别与策略事件，不输出 payload 字节。另一个相关社区在当日窗口内没有实质活动。一个项目讨论还关注如何分别比较运行时二进制大小、strip 后磁盘大小、常驻内存和注入延迟；把它们合成一个“大小”数字会掩盖真正的优化目标。

多个项目专用频道较安静，一个开发通知频道主要是自动检查与 review 状态。调度器支持区在前一天的 BTF 工具链问题后没有出现新的独立问题；其论坛当前可见问题也早于回退窗口。这些页面都已检查并计为安静，没有被用来虚构需求。

### 上游工作强调失败路径正确性

公开 BPF 开发归档非常活跃。当天线程涉及 `BTF.ext` 边界检查的整数溢出防护、分配失败后的 arena 状态一致性、多种 JIT 的 arena atomic fault、queue/stack map 下标溢出、bpftool batch 文件处理、ARM64 分支记录对 BPF branch snapshot 的支持，以及 16 字节聚合返回值。共同关注点不是 happy path，而是算术、分配、架构特定 fault 或部分解析失败时仍然保持不变量。

公开论坛的新内容是一篇 verifier 约束说明，讨论重点是为什么必须保持有界状态，而不是新的排障报告。通用项目聊天主要是新成员活动，其余专用频道较安静。结合 sockhash 卡死，今天的共同主题很明确：verifier 接受建立的是静态安全属性；生命周期转换与架构特定失败路径仍要求正确 hook、明确所有权和运行时测试。
