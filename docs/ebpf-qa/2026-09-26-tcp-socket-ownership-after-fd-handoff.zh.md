# 为什么 connect 或 accept 探针会漏掉通过文件描述符交接而换了主人的 socket？

简短回答：因为 `connect` 和 `accept` 是进程侧的动作，而 socket 的归属权属于一个生命周期比任何单个进程都长的内核对象。一个 `struct sock` 可以经由 `systemd` socket 激活的描述符、`SCM_RIGHTS` 的 `sendmsg`，或者继承而来的文件描述符被交给另一个进程，而接收方进程里并不会因此多出一条新的 `connect`/`accept` 系统调用。内核追踪的是这个 socket 的*状态*，而不是*创建它的那次系统调用*。`sock:inet_sock_set_state` tracepoint 在每次 TCP 状态迁移时触发，并交出存活的 `struct sock` 指针，于是可以靠把该指针 join 到 socket 上稳定的归属字段来归属 socket，而不是去盯那个根本没有在新主人进程里发生过的系统调用。

## 机制

关键的分界，在于基于系统调用做追踪的模型所依赖的两层东西：

- **进程侧** — `connect`、`accept4`、`bind`、`close`。它们运行在具体的 `task_struct` 里，携带 `struct file *`。挂在 connect/accept 路径上的按进程探针，只能看到 socket 在*当前*进程主动创建或回收它的瞬间。
- **内核对象侧** — `struct sock`。这个对象的生命周期独立于任何单个文件描述符。它的归属由稳定字段承载，比如 `sk_uid`（socket 所有者的 user id），在 `struct file` 创建时写入、文件被重新归属时刷新（`net/socket.c`），而不会因为描述符被 `dup` 或发给别的进程就改变 —— 底层的 `struct sock` 始终是同一个对象。

所以当 `systemd` 服务启动时，监听描述符被交给服务，服务的 `accept` 返回一个*新的* `struct sock`（连接 socket），它是由*服务端*线程创建的，而不是服务里某次客户端 `connect`。或者 `SCM_RIGHTS` 把一个已打开的数据 socket 传给另一个进程，而那个 socket 最初建连发生在发送方。两种情况下，按进程的 connect/accept 探针都是盲的：归属已经移动了，但新主人进程里没有 `connect`/`accept` 发生。

内核保证有的是状态迁移的追踪。对 TCP 而言，每次状态变化都经过 `tcp_set_state`（`net/ipv4/tcp.c`），它跑完 MIB 统计与解哈希逻辑后，最后调用 `inet_sk_state_store(sk, state)`（`net/ipv4/af_inet.c`）：

```c
void inet_sk_state_store(struct sock *sk, int newstate)
{
	trace_inet_sock_set_state(sk, sk->sk_state, newstate);
	smp_store_release(&sk->sk_state, newstate);
}
```

`inet_sk_state_store` 在状态落库*之前*触发 `sock:inet_sock_set_state` tracepoint，因此事件里带 `oldstate`、`newstate`、`sport`/`dport`、`family`、`protocol`，以及 IPv4/IPv6 双方地址。关键在于它还暴露了 `skaddr`，即原始的 `struct sock *`。状态整数与 BPF uapi 枚举 `bpf_tcp_state`（`include/uapi/linux/bpf.h`）一一对应，其开头是：

```c
enum bpf_tcp_state {
	BPF_TCP_ESTABLISHED = 1,
	BPF_TCP_SYN_SENT,
	BPF_TCP_SYN_RECV,
	BPF_TCP_FIN_WAIT1,
	BPF_TCP_FIN_WAIT2,
	BPF_TCP_TIME_WAIT,
	/* ... */
	BPF_TCP_MAX_STATES
};
```

`tcp_set_state` 用一连串 `BUILD_BUG_ON` 把 `BPF_TCP_*` 值钉死到内部的 `TCP_*` 常量上，所以事件里的 `int oldstate` / `newstate` 可以直接当作 BPF 枚举值使用。于是 tracepoint 处理器可以以四元组 `(saddr, sport, daddr, dport)` 为 key 建表，当看到 `LISTEN -> ESTABLISHED` 或 `ESTABLISHED -> CLOSE_WAIT -> TIME_WAIT` 时，挂上*当前*持有该 socket 的对象，并从中读出属主 UID。

accept 路径正是归属问题藏身的地方。`__inet_accept`（`net/ipv4/af_inet.c`）调用 `sock_graft(newsk, newsock)`（定义在 `net/core/stream.c`）把新连接 `sock` 接到 accept 方的 `struct file` 上，并置 `newsock->state = SS_CONNECTED`。那个 socket 的属主 UID，是共享内核对象上 `sk_uid` 早就编码的值 —— 它在交接过程中被*继承*，而不是被 accept 方的系统调用重建。accept 方进程"拥有"的是这个 fd；而 socket 的 `sk_uid` 说的是这个 socket *来自*谁。

## 验证与调试路径

用 `bpftrace` 处理器可以把状态机变成一张按 socket 的归属表。在四元组迁到 `ESTABLISHED` 时，透过 BTF 解出 `sk` 指针并读取属主 UID：

```
tracepoint:sock:inet_sock_set_state /args->newstate == 1/ {
	printf("%d:%d owner_uid=%d\n", args->sport, args->dport,
		read((uint32 *)(((struct sock *)args->skaddr)->sk_uid)));
}
```

在 BPF-C（BCC/libbpf）里，同样的查询用显式内核读取（`sk` 是任意内核指针，verifier 要求显式读而不能直接解引用）：

```c
SEC("tracepoint/sock/inet_sock_set_state")
int established(struct trace_event_raw_inet_sock_set_state *ctx)
{
	if (ctx->newstate != BPF_TCP_ESTABLISHED)
		return 0;
	struct sock *sk = (struct sock *)ctx->skaddr;
	__u32 uid = 0;
	/* sk 是任意内核指针：verifier 要求显式读而非直接解引用 */
	bpf_probe_read_kernel(&uid, sizeof(uid), &sk->sk_uid);
	/* 以 (family, sport, dport, daddr) 为 key；value = { uid, sk, ts } */
	return 0;
}
```

调试的要点是**做关联，而不是假设**：

1. 记下迁移事件里的 `skaddr`。
2. 在同一探针里透过 BTF 解出 `skaddr->sk_uid`（对 Unix socket 还有 `sk_peer_pid`）。
3. 把该 UID 与你实际怀疑持有该 socket 那个进程的文件描述符表对账。当两者不一致时，说明 socket 经历了 fd 交接，属主是内核对象上的 `sk_uid` 值，而不是最初调用 `connect` 的那个 pid。

再用 `perf tracepoint -p <pid> -e sock:inet_sock_set_state` 交叉核对同一次迁移，确认 `oldstate`/`newstate` 对得上。`TIME_WAIT`/`CLOSE_WAIT` 的"复活"场景是两种视图分歧最大的地方：一个 `TIME_WAIT` socket 是 `sk_user` 在 close 时已放下的孤儿 `struct sock`，但随后对同一四元组的 `SO_REUSEADDR`/`SO_REUSEPORT` bind 会让这条流在一个*不同*的属主下复活。tracepoint 能捕获这次再进入；系统调用视图捕获不到，因为没有新的 `connect` 发生。

## 局限

`inet_sock_set_state` 是 `inet` 协议族的 tracepoint：它读的是 `inet_sk(sk)`，只建模 TCP 状态表。后果有：

- **UDP 对它不可见。** 对 UDP socket 调用 `connect()` 只是设定默认目的地址；UDP 没有连接状态机，不存在迁到"已建立"的过程。不能用状态 tracepoint 来归属"UDP 连接"。对 UDP，应以 socket inode 加 `sk_uid` 为 key，配 `bind`/`sendto` 探针，并彻底放弃"建连"这一语义。
- **tracepoint 打在 `sk` 上，不是打在进程上。** 一次 `SCM_RIGHTS` 交接后，共享同一个 socket 的两个进程共享同一个 `struct sock`，所以单靠 tracepoint 说不清*当前持有* fd 的是哪个进程。要解析当前持有者仍需要按进程的文件描述符表 join；tracepoint 只给你那个能扛过交接的稳定 `sk_uid`。
- **它看到的是迁移，不是驻留。** 一个创建后停在 `LISTEN` 的 socket，在状态变化前不产生任何迁移；一条稳定 `ESTABLISHED` 的流在两次迁移之间静默。稳态 socket 只能在*下一次*迁移时被捕获，或者靠按 `sk` 指针做周期性重扫。

## 参考

- `inet_sock_set_state` tracepoint 定义：https://elixir.bootlin.com/linux/latest/source/include/trace/events/sock.h
- `inet_sk_state_store` 触发点：https://elixir.bootlin.com/linux/latest/source/net/ipv4/af_inet.c
- `tcp_set_state` 与 `BUILD_BUG_ON` 状态钉死：https://elixir.bootlin.com/linux/latest/source/net/ipv4/tcp.c
- 调用 `sock_graft` 的 `__inet_accept`：https://elixir.bootlin.com/linux/latest/source/net/ipv4/af_inet.c
- `sock_graft` 定义：https://elixir.bootlin.com/linux/latest/source/net/core/stream.c
- uapi `bpf_tcp_state` 枚举：https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/bpf.h
- 含 `sk_uid` 等归属字段的 `struct sock`：https://elixir.bootlin.com/linux/latest/source/include/net/sock.h
- `SCM_RIGHTS` socket 交接语义：https://man7.org/linux/man-pages/man7/socket.7.html
- `SO_REUSEADDR` / `SO_REUSEPORT` 复活行为：https://man7.org/linux/man-pages/man7/socket.7.html

## 当日社区讨论

覆盖情况：通过只读快照可触达**2**个已加入 watchlist 的 Slack 归档，滚动 7 天窗口内共**3**条消息。允许名单里的浏览器专属社区（两个 Discord 服务器、公开的 kernel 邮件列表，以及 eBPF subreddit）在本次运行中无可见浏览器会话可用，标记为**不可触达，而非无讨论**。下文不复现任何 Slack/Discord 工作区、频道、参与者或消息 URL。

可触达窗口里的两个主题：

- **把 socket 归属到进程，是反复出现的诉求。** 有厂商发布了一篇关于仅用 tracepoint 构建端点监控 agent 的文章，论证周期性的 `/proc` 轮询正在被 eBPF 取代，且 eBPF 与 ETW 把 socket 归属到属主进程的方式不同、互为补充。另一条性能向的讨论从规模侧讲了同一点：在现代化的多 socket 服务器上，完整遍历 `/proc` 要数秒，而 eBPF 迭代器能在远小于一秒内完成，并明确建议挂 `sock:inet_sock_set_state` tracepoint 而非 `connect`/`accept`，因为 socket 激活的交接与复活会绕过系统调用视图。其中未解决的分界，正是本页回答的内容：状态 tracepoint 给出稳定的内核对象属主，而要说清当前*持有* socket 的是哪个*进程*，仍需文件描述符表 join，且整套方法仅限 TCP。
- **策略查找中用户态与 BPF 的优先级错位。** 有贡献者阅读某策略引擎的用户态 map 解析器时发现，Go 解析器与 BPF 数据面在"同一优先级下，聚合身份条目与具体身份条目谁赢"上不一致。这是与归属问题不同的一题：同一张决策表的两个实现发生了漂移，实际的下一步是一个差分测试 —— 把同一组 key 同时喂给用户态解析器和 BPF map，断言两者选出相同的胜者。它在可触达窗口内没有进一步展开，此处按悬而未决的问题如实记录。

可触达的 3 条消息里没有其他实质性讨论；上述两主题即完整可达集，浏览器专属社区仍未审阅，未擅自当作无讨论。
