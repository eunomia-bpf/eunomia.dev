# 删除 BPF hash 表的一个 key 时，值内存会被清零吗？被复用的槽位是否可能残留旧值的字节？

**简短回答：** 不会。`bpf_map_delete_elem` 只是把元素从 hash 桶里摘链，并把它的内存交还给 per-CPU freelist（预分配 hash）或 RCU 延迟回收的 BPF 分配器（`BPF_F_NO_PREALLOC` 的 hash）；它**不会**清除 value 的字节。你无法通过 map API 把已删除的 value 读回来——`bpf_map_lookup_elem` 返回 `NULL`，`bpf_map_get_next_key` 也会跳过它——但这些物理字节会留在被复用的槽位里，直到下次更新把它覆盖；而一次常规的整值写入*确实*会在每个 CPU 上重写整个 value 区域——只有一个例外：在 per-CPU hash map 的 `BPF_F_CPU` 创建路径上，只写入指定 CPU 的槽位，其他 CPU 保留上一个使用者的字节，因此查找新 key 时（`BPF_F_CPU` 指向某个非指定 CPU）仍可能在这些 CPU 上读到已删除 key 的 per-CPU 值（针对该 bug 的清零补丁于 2026-09-23 提交到 bpf 邮件列表，截至 2026-09-24 的主线抓取尚未合入）。在补丁合入前，这是经由 map API 唯一能触发的跨 key 值泄漏：普通 HASH 的整值更新会在 key 可见前在所有 CPU 上重写整个值区域，BPF 程序创建 per-CPU value 走 `pcpu_init_value`，会把非当前 CPU 的槽位清零。规则是：不要指望 `delete`（或 LRU 驱逐）帮你抹掉 value 内容；自己把 value 清零，并且不要假设内嵌的内核对象字段（定时器、工作队列、任务工作队列）还活着，因为元素释放时内核会取消这些字段。

## delete 实际做了什么

`htab_map_delete_elem` 算出 key 所在的桶，取桶锁，若元素存在就调用 `hlist_nulls_del_rcu(&l->hash_node)` 摘链，然后 `free_htab_elem`。整条路径上，value 区域没有任何 `memset` 之类的清零。接下来内存如何处置，取决于 map 的创建方式：

- **预分配 hash（默认）。** 建 map 时 `prealloc_init` 把所有 `max_entries` 个元素一次性放进一个连续的 `htab->elems` 池，并填好一条 per-CPU freelist。delete 时元素回到这条 per-CPU freelist（`pcpu_freelist_push`）。它立即可复用——下一个同 CPU 的更新会把它弹回来（`__pcpu_freelist_pop`，LIFO）——但值字节在删除时从未被触碰。池本身在 map 创建时经 `__GFP_ZERO`（`__bpf_map_area_alloc`，`kernel/bpf/syscall.c`）一次性清零；per-CPU 变体则由 `prealloc_init` 内的 `bpf_map_alloc_percpu` 分配单独的 per-CPU 值区域，在创建时经内核 `alloc_percpu` 的零填充保证清零（BPF 代码里没有传 `__GFP_ZERO` 标志——这是 `alloc_percpu` 本身的性质）。两个区域在删除时都不再清零；元素被复用时，唯一的创建后清零是 BPF 程序创建 per-CPU value 时 `pcpu_init_value` 对非当前 CPU 槽位的清零。
- **惰性 hash（`BPF_F_NO_PREALLOC`）。** 元素从 per-CPU 的 BPF 内存缓存按需分配；delete 时 `htab_elem_free` 把它交还给该分配器。物理回收是**RCU 延迟**的（分配器的 `free_by_rcu` / `call_rcu_tasks_trace` 路径），所以页面要等一个 grace period 结束才会被回收；value 字节停留得更久。

两种路径下元素*计数*都是同步减一（`bpf_map_dec_elem_count` / `dec_elem_count`），所以 delete 会立刻腾出"再放一个新元素"的名额，即使底层*内存*的回收要再晚一点。

## 为什么一次整值写入不会把一个 key 的 value 漏进另一个

当新 key 被写进一个被复用的元素时，`alloc_htab_elem` 先复制完整的 `key_size` 字节，再复制**整个** value 区域（`copy_map_value`，或按取整后的 `value_size` 做 `memcpy`）。一个格式良好、写满的 value 因此会完全替换掉上一个使用者的字节。不存在"我 `bpf_map_lookup_elem` 取 key B，却通过 API 读到 key A 的 value"的路径——有一个已记录的例外：在 per-CPU map 的 `BPF_F_CPU` 创建路径上，`pcpu_copy_value` 只写指定 CPU 的槽位并提前返回，其余 CPU 保留被回收元素的原始字节，因此查找新 key 会把这些 CPU 上已删除 key 的 per-CPU 值读回来。针对这条路径的清零补丁（`Fixes: c6936161fd55`）于 2026-09-23 提交到 bpf 邮件列表，截至 2026-09-24 的主线抓取尚未合入。普通整值更新或 BPF 程序写入留下的残留，是*物理内存*层面的事实（旧字节一直待在槽位里，直到被覆盖或清刷），而不是*逻辑*泄漏：普通 HASH map 的值区域在 key 可见前会在所有 CPU 上被完整重写，BPF 程序创建 per-CPU value 时 `pcpu_init_value` 会把非当前 CPU 的槽位清零。上面那个 `BPF_F_CPU` 创建路径的例外则是*逻辑*的、API 可见的泄漏：在补丁合入前，用 `BPF_F_CPU|cpu<<32` 指向非指定 CPU 的 `bpf_map_lookup_elem_flags` 会读回上一个使用者的字节。

## 真正不安全的角落

- **value 字节没有被擦除。** delete 没有清刷步骤，惰性分配器回收对象时也没有一个 BPF 程序能依赖的清刷。如果某个 value 曾经放过敏感数据，`delete` 并不会抹掉它——这些字节会一直留在预分配池或分配器 freelist 里，直到被复用。对于显式 `delete`，安全做法——也是唯一可靠的擦除方式——是释放该条目前显式把 value 清零；自动 LRU 驱逐在元素离开之前没有用户态挂钩点，被驱逐 value 的字节可能残留在池里，直到后续分配把它们覆盖。（创建时清零——非 per-CPU 池的 `__GFP_ZERO`、per-CPU 值区域的 `alloc_percpu` 保证——是一次性事件，删除或复用时不适用。）
- **内嵌的内核对象字段是被取消的，不是被保留的。** 释放时，`check_and_cancel_fields` / `bpf_obj_cancel_fields` 会取消值里的 BPF 对象字段（定时器、工作队列、任务工作队列）。对 per-CPU hash map，这些字段类型在 map 创建时就被 `map_check_btf()` 拒绝，因此取消在这类 map 上是空操作；对非 per-CPU map 则是真实保证。程序在元素被释放或复用之后去读该 value，不能假设内嵌对象仍然有效。
- **LRU map 会自行驱逐。** `BPF_MAP_TYPE_LRU_HASH`（及 `LRU_PERCPU_HASH`）在 map 满时会驱逐最久未使用的元素，走的是同一条摘链并释放的路径（`htab_lru_map_delete_node`：`hlist_nulls_del_rcu` + 计数递减 + 取消字段，不清零）。你从未调用过 `delete` 的 key，也可能在驱逐之后读回 `NULL`。带 `BPF_F_NO_COMMON_LRU` 时每个 CPU 维护自己的 LRU 列表（`LRU_PERCPU_HASH` 是 per-CPU LRU + per-CPU map；`LRU_HASH` 是 per-CPU LRU + 全局 map），因此驱逐哪个元素是每 CPU 的，而不是全局的。

## 如何验证

1. **确认摘链。** `bpf_map_delete_elem(key)` 之后立刻 `bpf_map_lookup_elem(key)` 返回 `NULL`，且 `bpf_map_get_next_key` / `bpf_map_get_next_key_batch` 永远不会返回这个已删除的 key。
2. **注意迭代会重启。** 文档指出，若你传入的 `cur_key` 恰好刚被删除，`bpf_map_get_next_key` 会从表里*第一个* key 重新开始；要做稳定的全表扫描请用 batched lookup。
3. **和 array 对比。** `BPF_MAP_TYPE_ARRAY` 的 value 在*创建时*就被零初始化（文档明确写了）。非 per-CPU 预分配 hash 的 value 同样在 map 创建时清零（`__bpf_map_area_alloc` 的 `__GFP_ZERO`）；per-CPU 值区域在创建时经 `alloc_percpu` 保证清零。区别在于：array 不会逐元素回收，而 hash 元素会——且删除时没有任何路径重新清零 hash value（唯一的创建后清零是 BPF 程序创建 per-CPU value 时 `pcpu_init_value` 对非当前 CPU 槽位的清零）。回收而不重新清零，才是整条边界。
4. **保证 value 干净。** 对于显式 `delete`，唯一可靠的擦除方式是在 `delete` 前自己写入全零；自动 LRU 驱逐在元素离开之前没有可保证的用户态时间点，因此无法保证驱逐时清刷；`bpf_map_delete_elem` 并没有任何"顺带清零"的 flag。

## 答案的边界

- 这里谈的是 value *内存*，而非 map API 的契约：被删除或被驱逐的 key 依旧读作 `NULL`、也不会被枚举到，所以*逻辑*视图是干净的，哪怕*物理*字节还留着——上面那个例外除外：在未修复的 `BPF_F_CPU` 创建路径上，清零补丁合入前，对新 key 在某个非指定 CPU 上的查找仍会读回上一个使用者的字节，是经由 map API 可见的跨 key*逻辑*泄漏。
- slab 之后是否会把回收的页面清刷，并不是一个 BPF 程序可见的保证；程序不能依赖它。
- 每-CPU LRU flag（`BPF_F_NO_COMMON_LRU`）让每个 CPU 维护自己的 LRU 列表（`LRU_PERCPU_HASH` 是 per-CPU LRU + per-CPU map；`LRU_HASH` 是 per-CPU LRU + 全局 map），改变的是每 CPU 驱逐*哪个*元素，而不是"摘链并释放且不清零"这一机制本身。

## 参考资料

- [BPF_MAP_TYPE_HASH，含 PERCPU 与 LRU 变体（内核文档）](https://docs.kernel.org/bpf/map_hash.html)——默认预分配、`BPF_F_NO_PREALLOC`、删除语义、`bpf_map_get_next_key` 的"已删 key 会重启"注意点、per-CPU slot。
- [BPF_MAP_TYPE_ARRAY（内核文档）](https://docs.kernel.org/bpf/map_array.html)——"所有 array 元素在创建时都被预分配并零初始化"。
- [kernel/bpf/hashtab.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/hashtab.c)——`htab_map_delete_elem` → `free_htab_elem` → `pcpu_freelist_push` / `htab_elem_free`；`prealloc_init`；`alloc_htab_elem` 的整值覆盖；`pcpu_init_value`（per-CPU 创建路径清零）；`check_and_cancel_fields`。
- [kernel/bpf/memalloc.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/memalloc.c)——惰性 BPF 内存缓存及其 RCU 延迟回收路径。
- [kernel/bpf/syscall.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/syscall.c)——`__bpf_map_area_alloc` 里的 `__GFP_ZERO`（预分配元素池，第 384 行）；`bpf_map_alloc_percpu` → `__alloc_percpu_gfp`（per-CPU 值区域，第 583 行）；`__bpf_alloc_page`（第 606 行）；`bpf_map_value_size` 在 `BPF_F_CPU`/`BPF_F_ALL_CPUS` 下返回单一 value 大小（第 137 行）。
- [bpf 邮件列表，2026-09-23：`BPF_F_CPU` per-CPU hash 创建路径的清零补丁](https://lists.openwall.net/linux-kernel/2026/09/23/16)——`pcpu_init_value` 的 diff（`Fixes: c6936161fd55`）；元素复用 selftest 在 [第 18 条消息](https://lists.openwall.net/linux-kernel/2026/09/23/18)。截至 2026-09-24 的主线抓取尚未合入。仅以 URL 和主题引用。

## 当日社区讨论

如实的覆盖说明：两个 watchlist 选中的 Slack 存档在滚动窗口内今天**零消息**，allowlist 里的 Discord 频道与公开的 bpf 邮件列表、r/eBPF 仅限 visible-browser，且本次运行没有可用的浏览器会话——因此 2026-09-23 没有任何私有社区材料可用。上文问题是回退选择：这是被监控的 eBPF 开发社区中一个真实、反复出现的从业者边界，完全以上述公开一手资料（内核 BPF 文档与上游源码）为依据，而不是任何具体 thread。

该领域反复出现的从业者主题——"我的 hash 表 value 变得陈旧，已删除的 key 看起来还残留旧数据"——正是落在这条边界上：删除是一次*摘链*，而非一次*清零*，元素的内存要么经 per-CPU freelist（预分配）要么经 RCU 延迟分配器（惰性）被回收。公开文档锚定的两个后续点在此反复出现。其一是 **LRU 的意外**：一个从未被删除的 key，在 map 满、最久未用元素被驱逐后，会读回 `NULL`；人们容易把它误读成 delete 的 bug，实际是容量驱逐。其二是**密钥处理**的推论：既然 delete 不清刷 value 字节，一个曾经放过敏感数据的 map 在物理上仍保留这些数据直到被复用，因此安全做法是在释放该条目前把 value 清零（并显式丢弃内嵌内核对象）。文档留下的未解边界是：并不存在"删除时顺带清零"的 API；保持 value 内存干净是程序的责任，而非内核的责任。同一天公开的一手信源印证这条边界在上游开发中是活着的：上面提到的 bpf 邮件列表 `BPF_F_CPU` per-CPU 创建路径清零补丁（[该补丁](https://lists.openwall.net/linux-kernel/2026/09/23/16)，元素复用 selftest 在 [第 18 条消息](https://lists.openwall.net/linux-kernel/2026/09/23/18)）——这是同日的公开信源，而非私有社区消息；上文的如实覆盖说明不变。
