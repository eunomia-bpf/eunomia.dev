# Why can inserting a socket into `SOCKHASH` from TC egress soft-lock the kernel?

**Short answer:** do not use a TC packet hook as the control path that enrolls a live TCP socket into a `BPF_MAP_TYPE_SOCKHASH`. A sockhash is not an ordinary hash map of borrowed `struct sock *` pointers. Inserting a socket takes a reference, creates or reuses a `sk_psock`, replaces socket callbacks, and makes the socket inherit programs attached to the map. Performing that transition while the same socket is already on its transmit path can enter lock and callback interactions that a packet classifier was not designed to initiate. A verifier-accepted program is not proof that every kernel version safely supports that lifecycle transition from that hook.

Use `BPF_PROG_TYPE_SOCK_OPS` and `bpf_sock_hash_update()` when the connection reaches the intended TCP state, or insert the socket from userspace by file descriptor. Let TC observe packets or apply packet policy; let the socket-oriented hook own sockhash membership.

The observed soft lock is evidence of a kernel defect or unsupported interaction, but it is not enough to name one exact lock cycle. A useful report still needs the smallest program, full kernel configuration and commit, watchdog trace, and a comparison with the supported sockops insertion path.

## Why a sockhash update is a lifecycle operation

The kernel documentation says that sockmap and sockhash values are references to sockets. On insertion, the kernel attaches a `struct sk_psock`, replaces socket callbacks, and inherits parser or verdict programs from the map. A socket may appear in several maps, but it may inherit only one parser or verdict program; conflicting attachment returns an error.

That is materially different from storing a scalar in a regular hash map. The update changes how future data enters and leaves the socket. Consequently, three objects must have compatible lifetimes:

1. the TCP socket and its reference count;
2. the sockhash entry and its `sk_psock`; and
3. any `SK_SKB` or `SK_MSG` programs attached to the map.

At TC egress, the current `skb` is already being transmitted. `skb->sk` is contextual state associated with that packet, not an invitation to reconfigure the socket's callbacks in place. Looking up the socket again with `bpf_skc_lookup_tcp()` does not change this boundary: the returned reference must be released, and obtaining a referenced socket does not turn the TC hook into a supported sockhash enrollment hook.

The dedicated helper makes the intended control plane explicit:

```c
long bpf_sock_hash_update(struct bpf_sock_ops *skops,
                          void *map, void *key, __u64 flags);
```

Its first argument is a `struct bpf_sock_ops *`, and the documented program type is `BPF_PROG_TYPE_SOCK_OPS`. The helper uses the socket represented by that context as the new map value. This is the safe architectural clue: establish membership where the kernel presents a socket lifecycle context, not by reverse-engineering connection establishment from a transmitted ACK.

## A safer design

Populate the sockhash from a sockops program after the connection is established. A minimal shape is:

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

Attach the sockops program to the cgroup that owns the target sockets. Confirm byte order and key layout against the matching lookup or redirect program; `local_port` and `remote_port` do not use identical representations in every BPF context. Treat helper failures as data: count errors by operation and key collision mode rather than silently retrying from TC.

If userspace already owns or can discover the socket file descriptor, it may update the map from the control plane. This also keeps callback installation outside the active packet path. Whichever path owns insertion should also own deletion and shutdown behavior.

TC can still participate without owning membership. It may extract a flow key, collect counters, mark traffic, or consult ordinary maps. If the goal is to assign a socket selected by policy, use the helpers and program types documented for socket assignment rather than using sockhash insertion as a general pointer store.

## How to diagnose the soft lock without losing the evidence

First remove the sockhash update from TC and verify that the lockup disappears. Then move the same key and map update to sockops. This A/B test separates packet parsing and key construction from the socket enrollment transition.

For a reproducible kernel report, preserve:

```console
$ uname -a
$ bpftool prog show
$ bpftool map show
$ bpftool net
$ zcat /proc/config.gz > kernel.config
```

Also capture the complete soft-lockup watchdog report or serial-console trace, including all CPUs and lockdep output if a lockdep kernel can reproduce it. Record whether the map has `SK_SKB` or `SK_MSG` programs attached, whether the socket was already present in another sockmap or sockhash, and whether replacing `skb->sk` with `bpf_skc_lookup_tcp()` changes the trace. Do not publish production addresses or traffic payloads; a network namespace with one client and one server is a better reproducer.

Run four reductions:

- an empty sockhash with no attached parser or verdict program;
- sockops insertion into that empty map;
- TC observation with no sockhash update; and
- the smallest TC update that still locks the kernel.

If only the last case fails, report it as a TC-to-sockmap interaction. If an empty sockhash also fails from the supported sockops path, the fault is broader. If the failure requires an attached verdict program, include that program and its attach type because callback inheritance is part of the transition.

Do not treat a different kernel version merely “working” as closure. Bisecting is valuable only after the reproducer records the same configuration, program, attach order, and traffic sequence on both kernels.

## References

- [Linux kernel documentation: `BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux UAPI definition of `bpf_sock_hash_update`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux kernel sockmap implementation](https://github.com/torvalds/linux/blob/master/net/core/sock_map.c)
- [Linux selftest: updating a sockmap from sockops](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/progs/test_sockmap_update.c)
- [Linux selftests for sockmap and sockhash](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf/prog_tests)
- [Linux kernel documentation: BPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [BPF mailing list: protect `BTF.ext` bounds checks from overflow](https://lore.kernel.org/bpf/CAOZ_KyvV-Ha7XDZHCUcyYKPMSUvMSrZ0NhmtMVy_KqWD0Ar4kg@mail.gmail.com/T/#u)
- [BPF mailing list: arena atomic fault handling across JITs](https://lore.kernel.org/bpf/84bd0041b0a35b87b15d9b00696b5abe1f16c340d2a02255e65e6758a03d4dc1@mail.kernel.org/T/#t)

## Community discussion today

Today's visible review covered 6 approved communities and 15 allowlisted channels or public pages. All were accessible. The 24-hour window was sparse, so the selected question uses the permitted seven-day fallback: a real unresolved socket-map problem received a new technical reply today. Names, accounts, employers, channel identities, message links, exact times, private topology, logs, and searchable wording have been removed. No raw transcript was retained.

### Socket ownership is the missing boundary

The strongest unresolved practitioner report involved a recent kernel soft-locking when a socket associated with an outgoing packet was inserted into a sockhash. Looking the socket up explicitly instead of reading it from the packet context did not avoid the failure. The latest response agreed that a kernel lockup is not expected, but the thread did not yet contain a reduced reproducer or watchdog trace.

The actionable answer is therefore narrower than a claimed root cause: move insertion to sockops or userspace, reduce the map to one socket with no attached verdict program, and capture the lock trace before proposing a kernel fix. The key concern is not pointer acquisition; it is changing socket callbacks and `sk_psock` state from inside the transmit path.

### Observability discussions focused on boundaries and minimization

One observability group continued considering how to recognize credential classes in network traffic without retaining credential values. The useful design boundary is to emit a bounded class and policy event, not payload bytes. Another group had no substantive activity in the daily window. A related project discussion focused on comparing runtime binary size, stripped on-disk size, resident memory, and injection latency as separate measurements; collapsing them into one “size” number would hide the actual optimization target.

Several project-specific channels were quiet, while a development-notification channel mainly reported automated checks and review state. The scheduler support area had no new independent question after the prior BTF-toolchain incident; its forum's visible question list was older than the fallback window. These surfaces were checked and counted as quiet rather than used to manufacture demand.

### Upstream work emphasized failure-path correctness

The public BPF development archive was active. Current threads covered integer-overflow hardening in `BTF.ext` bounds checks, consistent arena bookkeeping after allocation failure, faulting arena atomics across several JITs, queue and stack map index overflow, safer batch-file handling in bpftool, ARM64 branch-record support for BPF branch snapshots, and continued work on 16-byte aggregate returns. Across these topics, the recurring concern was not the happy path but preserving invariants when arithmetic, allocation, architecture-specific faults, or partial parsing fails.

The public forum's new material was an explainer about verifier constraints, with discussion centered on why bounded state is necessary rather than on a new troubleshooting report. The general project chat mostly contained new-member activity, and the remaining specialist channels were quiet. Together with the sockhash lockup, the day's discussions point to one practical theme: verifier acceptance establishes static safety properties, but lifecycle transitions and architecture-specific failure paths still require the correct hook, explicit ownership, and runtime tests.
