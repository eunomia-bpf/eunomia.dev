# Why does a connect or accept probe miss a socket that changed owner through a file descriptor handoff?

Short answer: because `connect` and `accept` are process actions, whereas socket ownership is a kernel object that outlives any one process. A `struct sock` can be handed to another process through a `systemd` socket-activation descriptor, an `SCM_RIGHTS` `sendmsg`, or an inherited file descriptor with no new `connect`/`accept` syscall in that process. The kernel still traces the *state* of that socket, not the *syscall that created it*. The `sock:inet_sock_set_state` tracepoint fires on every TCP state transition and hands you the live `struct sock` pointer, so you attribute the socket by joining that pointer to the socket's stable ownership fields instead of by watching a syscall that did not run in the owning process.

## The mechanism

The decisive boundary is between the two things a syscall-based tracing model is built on:

- **The process side** — `connect`, `accept4`, `bind`, `close`. These run in a specific `task_struct` and carry a `struct file *`. A per-process probe on the connect/accept path only ever sees a socket while the *current* process is actively creating or reaping it.
- **The kernel-object side** — `struct sock`. This object has a lifetime independent of any single file descriptor. Its ownership is carried in stable fields such as `sk_uid`, the socket owner's user ID, which is set on `struct file` creation and refreshed when the file is re-owned (`net/socket.c`), and does not change just because a descriptor naming the socket was `dup`'d or sent to another process — the `struct sock` is the same object underneath.

So when a `systemd` service starts, the listening descriptor is handed to the service and the service's `accept` returns a *new* `struct sock` (the connection socket) that was created by the *server* thread, not by a client `connect` in the service. Or `SCM_RIGHTS` passes an open data socket into another process, and that socket's original connection already happened in the sender. In both cases a per-process connect/accept probe is blind: ownership moved, but no `connect`/`accept` ran in the new owner.

What the kernel does guarantee is a state-transition trace. For TCP, every state change goes through `tcp_set_state` (`net/ipv4/tcp.c`), which runs the MIB accounting and unhash logic and finally calls `inet_sk_state_store(sk, state)` (`net/ipv4/af_inet.c`):

```c
void inet_sk_state_store(struct sock *sk, int newstate)
{
	trace_inet_sock_set_state(sk, sk->sk_state, newstate);
	smp_store_release(&sk->sk_state, newstate);
}
```

`inet_sk_state_store` fires the `sock:inet_sock_set_state` tracepoint *before* the state is stored, so the trace event carries `oldstate`, `newstate`, `sport`/`dport`, `family`, `protocol`, and both IPv4/IPv6 addresses. Critically it also exposes `skaddr`, the raw `struct sock *`. The state integers map 1:1 onto the BPF uapi enum `bpf_tcp_state` (`include/uapi/linux/bpf.h`), whose start is:

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

`tcp_set_state` pins the `BPF_TCP_*` values to the internal `TCP_*` constants with a run of `BUILD_BUG_ON`, so the `int oldstate` / `newstate` in the trace event can be treated as the BPF enum values directly. That means a tracepoint handler can key its table on the four-tuple `(saddr, sport, daddr, dport)` and, when it sees `LISTEN -> ESTABLISHED` or `ESTABLISHED -> CLOSE_WAIT -> TIME_WAIT`, attach the *current* owning socket and read the owner UID off the live `sk`.

The accept path is where the ownership question actually hides. `__inet_accept` (`net/ipv4/af_inet.c`) calls `sock_graft(newsk, newsock)` (defined in `net/core/stream.c`) to attach the fresh connection `sock` to the acceptor's `struct file`, and sets `newsock->state = SS_CONNECTED`. The owning UID of that socket is whatever `sk_uid` already encoded on the shared kernel object — it is *inherited* across the handoff, not re-created by the acceptor's syscall. The acceptor process "owns" the fd; the socket's `sk_uid` says who the socket is *from*.

## Verification and debugging path

A `bpftrace` handler turns the state machine into a per-socket ownership table. On a 4-tuple transition to `ESTABLISHED`, dereference the `sk` pointer through BTF and read the owner UID:

```
tracepoint:sock:inet_sock_set_state /args->newstate == 1/ {
	printf("%d:%d owner_uid=%d\n", args->sport, args->dport,
		read((uint32 *)(((struct sock *)args->skaddr)->sk_uid)));
}
```

In BPF-C (BCC/libbpf) the same lookup is an explicit kernel read of the field, since `sk` is an arbitrary pointer the verifier will not let you dereference directly:

```c
SEC("tracepoint/sock/inet_sock_set_state")
int established(struct trace_event_raw_inet_sock_set_state *ctx)
{
	if (ctx->newstate != BPF_TCP_ESTABLISHED)
		return 0;
	struct sock *sk = (struct sock *)ctx->skaddr;
	__u32 uid = 0;
	/* sk is an arbitrary kernel pointer: the verifier requires
	 * an explicit read instead of a direct dereference */
	bpf_probe_read_kernel(&uid, sizeof(uid), &sk->sk_uid);
	/* key on (family, sport, dport, daddr); value = { uid, sk, ts } */
	return 0;
}
```

The debugging step is to **correlate, not assume**:

1. Record `skaddr` from the transition event.
2. Dereference `skaddr->sk_uid` (and, for a Unix socket, `sk_peer_pid`) through BTF in the same probe.
3. Reconcile that UID against the file-descriptor table of the process you actually suspect of owning the socket. When they disagree, the socket moved through an fd handoff and the owner is the `sk_uid` value on the kernel object, not the pid that first called `connect`.

Cross-check the same transition with `perf tracepoint -p <pid> -e sock:inet_sock_set_state` and confirm `oldstate`/`newstate` line up. The `TIME_WAIT`/`CLOSE_WAIT` "revivify" case is where the two views diverge most: a `TIME_WAIT` socket is an orphaned `struct sock` whose `sk_user` was dropped at close, yet a later `SO_REUSEADDR`/`SO_REUSEPORT` bind on the same four-tuple revives the flow under a *different* owner. The tracepoint captures that re-entry; the syscall view does not, because no new `connect` ran.

## The limitation

`inet_sock_set_state` is an `inet`-protocol tracepoint: it reads `inet_sk(sk)` and only models the TCP state table. Consequences:

- **UDP is invisible to it.** `connect()` on a UDP socket only sets the default destination; there is no transition into an established state because UDP has no connection state machine. You cannot attribute a "UDP connection" with the state tracepoint. For UDP, key on the socket inode plus `sk_uid` and pair with `bind`/`sendto` probes, and drop the connection-establishment semantic entirely.
- **The tracepoint is on the `sk`, not on a process.** Two processes that share one socket after an `SCM_RIGHTS` handoff share one `struct sock`, so the tracepoint alone cannot say which process *currently holds* the fd. Resolving the current holder still needs a per-process file-descriptor-table join; the tracepoint only gives the stable `sk_uid` that survives the handoff.
- **It sees transitions, not dwell.** A socket that is created and then sits in `LISTEN` produces no transition until it moves; a steady `ESTABLISHED` stream emits nothing between two transitions. A steady-state socket is only captured at its *next* transition, or by a periodic rescan keyed on the `sk` pointer.

## References

- `inet_sock_set_state` tracepoint definition: https://elixir.bootlin.com/linux/latest/source/include/trace/events/sock.h
- `inet_sk_state_store` fire site: https://elixir.bootlin.com/linux/latest/source/net/ipv4/af_inet.c
- `tcp_set_state` and the `BUILD_BUG_ON` state-pinning: https://elixir.bootlin.com/linux/latest/source/net/ipv4/tcp.c
- `__inet_accept` calling `sock_graft`: https://elixir.bootlin.com/linux/latest/source/net/ipv4/af_inet.c
- `sock_graft` definition: https://elixir.bootlin.com/linux/latest/source/net/core/stream.c
- UAPI `bpf_tcp_state` enum: https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/bpf.h
- `struct sock` ownership fields including `sk_uid`: https://elixir.bootlin.com/linux/latest/source/include/net/sock.h
- `SCM_RIGHTS` socket handoff semantics: https://man7.org/linux/man-pages/man7/socket.7.html
- `SO_REUSEADDR` / `SO_REUSEPORT` revivify behaviour: https://man7.org/linux/man-pages/man7/socket.7.html

## Community discussion today

Coverage: **2** watchlist-opted-in Slack archives were reachable via the read-only snapshot, **3 messages** in the rolling 7-day window. The browser-only communities in the allowlist (the two Discord servers, the public kernel mailing list, and the eBPF subreddit) had no visible-browser session available this run and are marked **unavailable, not quiet**. No Slack/Discord workspace, channel, participant, or message URL is reproduced below.

Two themes from the reachable window:

- **Socket-to-process attribution is the recurring ask.** A vendor posted write-ups for an endpoint-monitoring agent built on tracepoints only, arguing that interval `/proc` polling is being retired in favour of eBPF and that eBPF and ETW attribute sockets to owning processes in different, complementary ways. An adjacent performance thread made the same point from the scale side: on a modern multi-socket server, a full `/proc` trawl takes multiple seconds while an eBPF iterator closes it in well under a second, and it specifically recommended hooking the `sock:inet_sock_set_state` tracepoint rather than `connect`/`accept`, because handoffs and `systemd` socket-activation revive sockets that the syscall view misses. The unresolved boundary there is exactly the one this page answers: the state tracepoint gives the stable kernel-object owner, while the file-descriptor-table join is still needed to say which *process* currently holds the socket, and the whole approach is TCP-only.
- **A user-space/BPF precedence mismatch in policy lookups.** A contributor reading a policy engine's user-space map resolver flagged that the Go resolver and the BPF datapath disagree on how an aggregate-identity entry and a specific-identity entry at the same precedence should resolve. This is a distinct theme from attribution: two implementations of the same decision table have drifted, and the practical next step is a differential test that feeds the same key set to both the user-space resolver and the BPF map and asserts identical winner selection. It was not developed further in the reachable window and is recorded here as the open question it remains.

No other substantive discussion was present in the 3-message reachable window; the two themes above are the full reachable set, and the browser-only communities remain unreviewed rather than assumed silent.
