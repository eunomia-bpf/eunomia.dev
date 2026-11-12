# BPF LRU 哈希 map 什么时候淘汰一个条目，当它的 LRU 列表是每 CPU 一份而不是跨 CPU 共享时会改变什么？

淘汰只发生在插入或更新路径上，而且只在表已经到达 `max_entries` 之后。一次查询本身绝不会释放槽位：读操作不会释放节点。读操作做的，只是给命中的节点置一个"最近使用"位；真正的链表移动和真正的槽位释放都发生在稍后——当一次更新被迫腾出节点、LRU 列表的轮换扫描跑起来的时候。LRU 是一个 active 加 inactive 的两级列表，不是一条单一的最久未用队列，所以"最近最少使用"是内核增量维护的一种启发式，而不是某一时刻能直接读出来的值。

`BPF_F_NO_COMMON_LRU` 标志并不改变淘汰发生的时机。它改变的是哪个条目存活，以及锁的画像。没有这个标志时，所有 CPU 共享一条全局 LRU 列表，所有轮换与空闲节点记账都走共享状态；有了它，每个 CPU 各维护一条自己的 LRU 列表，先从自己列表里淘汰，再去别的 CPU 那里偷节点。两种情况下触发点完全一致：都是到达容量时的插入。这个标志是用"全局最久未用"的性质去换每 CPU 的局部性和更低的锁开销，而不是去减少淘汰次数。

## 机制

1. **淘汰在插入路径上。** `htab_lru_map_update_elem` 先调用 `prealloc_lru_pop`，向 LRU 列表要一个空闲节点（`bpf_lru_pop_free`）。只要表还没到 `max_entries`，预分配池里就有空闲节点，不会淘汰任何东西。当表到了容量，`bpf_lru_pop_free` 拿不到空闲节点，就强制腾一个出来：它扫描 inactive 列表（`__bpf_lru_list_shrink_inactive`），对"最近使用"位未置的节点调用 map 删除钩子，把节点挪到空闲列表上。一个真实的键此刻才是从哈希表里被真正丢掉的时刻。

2. **最近性是一个惰性位，不是一个位置。** BPF 内的查询 helper 会置节点的"最近使用"位（`bpf_lru_node_set_ref`）；用户态的查询 helper 不会。读的时候什么都不移动。链表只有在轮换扫描（`__bpf_lru_list_rotate_active` 与 `__bpf_lru_list_rotate_inactive`）时才被改动：被扫到的、位已置的节点回到 active 列表头再争取一次存活，位未置的节点从 active 落到 inactive。节点只从 inactive 列表的尾部被释放。

3. **`BPF_F_NO_COMMON_LRU` 开关。** `bpf_lru_pop_free` 按每 CPU 标志分流。没有它时，`bpf_common_lru_pop_free` 维护一条共享 LRU 列表加每 CPU 的 pending 列表，并用一把共享锁做记账。有了它，`bpf_percpu_lru_pop_free` 给每个 CPU 一条自己的、由自己的锁保护的列表，一个 CPU 只有在自己的列表耗尽时才会按轮转去偷别的 CPU 的节点。每次更新要拉取的空闲节点批量 `target_free` 在 map 初始化时算好：最小 1 个、预算 128 个，所有 CPU 的合计预留上限为 map 大小的一半。

4. **OBI trace 上下文 map 作为具体例子。** OBI 固定的 `traces_ctx_v1` 是一个 `BPF_MAP_TYPE_LRU_HASH`，`max_entries` 为 `1 << 14`，且没有 `BPF_F_NO_COMMON_LRU`，因此用的是共享全局 LRU 列表。OBI 不为任何线程预留槽位，所以活跃的被 stamp 线程数很高时，最久未用的条目会在下一次插入时被淘汰，一个当前活跃的线程也可能查不到自己的上下文。

## 验证与排障路径

1. **看活的元素数。** 对固定的 map 用 `bpf_map_get_info` 或 `bpftool map show` 读元素数，和 `max_entries` 比。只有当计数达到 `max_entries` 时淘汰才可能发生；不到这个数，插入绝不会丢键。

2. **盯更新的返回值，不要盯查询。** 满了的普通非 LRU 哈希插入新键会返回 `-E2BIG`。满了的 LRU 哈希会淘汰 LRU 受害者并返回 `0`，或者在锁竞争下拿不到空闲节点时偶尔返回 `-ENOMEM`。如果你声明为 LRU 的 map 在满的时候插新键报 `-E2BIG`，那它其实是普通的 `BPF_MAP_TYPE_HASH`，不是 LRU 变体。

3. **用删除钩子追受害者。** 探测 map 更新路径，或 attach 到 map 删除，看把元素推到容量之上时是哪个键消失了。消失的那个键就是列表选出的 LRU 受害者。

4. **对标志问题，测受害者局部性。** 用共享 LRU 时，被淘汰的是全局最久未用的键。用 `BPF_F_NO_COMMON_LRU` 时，是发起插入那个 CPU 上最久未用的键，在多核热点负载下，这可能是一个仍然活跃、但和全局答案不同的键。

## 局限

- LRU 性质是启发式。内核维护两条列表、做有界的轮换扫描，不是一条严格的最久未用队列，而且内核把这些内部实现标记为将来可能变更。
- 最近性不对称。BPF 内的读会置位，用户态的读不会。两个只差"谁来读"的相同负载，会淘汰不同的键。
- 标志改变的是受害者和锁开销，不是触发点。`BPF_F_NO_COMMON_LRU` 不会停止淘汰，也不会保住全局最久未用的条目，它只是让淘汰变成每 CPU 的。
- 满了的 LRU map 上查不到，不等于这个键从未写过。它只说明这个键在最后一次插入时是 LRU 受害者，值可能曾经存在、随后被淘汰。

## References

- [内核 Documentation/bpf/map_hash.rst](https://github.com/torvalds/linux/blob/master/Documentation/bpf/map_hash.rst) - `BPF_MAP_TYPE_LRU_HASH` 在表满时自动淘汰最久未用条目；`BPF_F_NO_COMMON_LRU` 标志请求一条每 CPU 的 LRU 列表而非共享的那条。
- [内核 kernel/bpf/hashtab.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/hashtab.c) - `htab_lru_map_update_elem` 与 `prealloc_lru_pop` 是发生淘汰的插入路径；查询 helper 在 BPF 内置"最近使用"位，从用户态则跳过。
- [内核 kernel/bpf/bpf_lru_list.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/bpf_lru_list.c) - active 与 inactive 两级 LRU：`__bpf_lru_list_rotate_active`、`__bpf_lru_list_shrink_inactive`，以及 `bpf_lru_pop_free` 里每 CPU 与共享的分流。
- [内核 commit 3a08c2fd7634 ("bpf: LRU List")](https://github.com/torvalds/linux/commit/3a08c2fd7634) - 最初的 LRU 列表实现，以及 map 更新时各淘汰步骤。
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - `traces_ctx_v1`，一个固定的 `BPF_MAP_TYPE_LRU_HASH`，`max_entries` 为 `1 << 14`，且没有 `BPF_F_NO_COMMON_LRU`。

## 当日社区讨论

两个 opt-in 归档频道（两个 CNCF OpenTelemetry 插桩频道）本次运行产出 0 条消息；24 小时窗口前进到了一个空快照。归档里没有新问题，而近几日已经把 FUSE 与 OBI trace 上下文 map 的查不到轴都覆盖了，本篇因此取同一张 map 的自然续篇：`traces_ctx_v1` 是 LRU 哈希，下一个顺理成章的问题是它实际淘汰什么、以及每 CPU 标志会改变什么。答案基于内核 LRU map 源码与 OBI map 规格，而不是某条新的归档提问。

本次运行的频道覆盖：两个 opt-in 归档提供 0 条消息。仅可见浏览器可访问的来源 - eunomia-bpf 与 sched-ext Discord 服务器、bpf 邮件列表、r/eBPF - 本次运行没有可用的可见浏览器会话，无法审阅，因此标记为未覆盖而非安静。
