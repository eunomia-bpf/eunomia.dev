# 为什么 BPF hash 表满了以后，插入新 key 会报 E2BIG，而更新已存在的 key 仍然成功？

**简短回答：** 对于 `BPF_MAP_TYPE_HASH`，`max_entries` 是对**不同 key 数量**的硬上限，而它是在*元素分配*时执行的：*新 key* 需要一个全新元素，当没有可用元素时，update 返回 `E2BIG`（文档写明：map 的元素数量达到了 *max_entries* 上限）。*更新已存在的 key* 不需要新元素：预分配路径用 per-CPU 的 `extra_elems` 槽位做交换，惰性路径（`BPF_F_NO_PREALLOC`）里源码注释明确写着——"当 map 满了而 update() 是替换旧元素时，允许分配，因为旧元素会立即被释放"。所以两种创建方式下，满容量时更新旧 key 都照常成功。`BPF_MAP_TYPE_LRU_HASH` 在满时则不这样失败：它驱逐最久未用条目来腾位置，而不是返回 `E2BIG`——这就是为什么一个你从未 delete 过的 key 会在容量压力后突然读回 `NULL`。

## E2BIG 从哪来

uapi 契约是明确的。`bpf_map_update_elem(2)` 的 man 页（由 `include/uapi/linux/bpf.h` 生成）把 `E2BIG` 的 errno 描述写成了 "map 的元素数量达到了 map 创建时指定的 *max_entries* 上限"，与 `EEXIST`（对已存在 key 传了 `BPF_NOEXIST`）和 `ENOENT`（对不存在的 key 传了 `BPF_EXIST`）并列。

两个 map 家族在各自不同的代码里碰到这个上限：

- **预分配 hash（默认）。** `prealloc_init` 一次性把全部 `max_entries` 个元素放进一个连续池子，并配一条 per-CPU freelist。freelist 空了以后，`alloc_htab_elem` 里的 `__pcpu_freelist_pop` 失败并返回 `ERR_PTR(-E2BIG)`。
- **惰性 hash（`BPF_F_NO_PREALLOC`）。** `alloc_htab_elem` 先调 `is_map_full` 检查容量：它把元素计数（大 map 用 batch 为 32 的 `percpu_counter`，否则普通 `atomic_t`）与 `max_entries` 比较。满了且没有旧元素可替换 → `ERR_PTR(-E2BIG)`。惰性路径上真正缺内存是另一个错：`ENOMEM`，而不是 `E2BIG`。

map *创建* 时也会出现 `E2BIG`——`key_size + value_size` 大到一个元素无法 kmalloc、per-CPU 值超过 `PCPU_MIN_UNIT_SIZE`、或 `max_entries > 1UL << 31`——所以光看 errno 不能区分你碰到的是哪一种。

## 为什么更新已存在的 key 不碰上限

`htab_map_update_elem` 先做 key 查找。找到 `l_old` 之后，满容量只在*分配*这一步才有意义：

- **预分配：** `alloc_htab_elem` 取该 CPU 的 `extra_elems` 槽位，把 `l_old` 存进去（`*pl_new = old_elem`），新元素被换到桶头。不弹 freelist、不做容量检查、不会 `E2BIG`——元素总数没有增长。
- **惰性：** `is_map_full` 返回 true，但 `alloc_htab_elem` 里的注释写了规则：map 满了且这次 update 是替换旧元素，允许分配，因为旧元素会立即被释放（bucket 链表交换之后 `free_htab_elem`，之前 `check_and_cancel_fields` 会取消内嵌 BPF 对象字段）。计数保持持平：新元素 `inc_elem_count`，旧元素 `dec_elem_count`。

容量检查 `is_map_full` 只在"满 map + 新 key"时被参考。这就是整个不对称的全部：到达 `max_entries` 后，"新 key" → `E2BIG`，"旧 key" → 成功。

## spin lock 值的例外

如果 map 的 value 内嵌 `bpf_spin_lock`（update 时传 `BPF_F_LOCK` 标志——map 的 BTF record 必须带 `BPF_SPIN_LOCK`，否则 update 以 `-EINVAL` 被拒），`htab_map_update_elem` 走一条捷径：key 已存在时，它取元素上的自旋锁并调用 `copy_map_value_locked` *就地*更新值，在任何分配之前返回。更新已存在的 spin-locked 值因此从不分配、不可能被容量挡住——`bpf_spin_lock`（kernel 5.1+）就是文档化的、让不同 CPU 上的程序并发修改同一值时保持值一致的方式。这类 map 上的*新 key* 仍走正常分配路径，满时同样 `E2BIG`。

## LRU 变体为什么不一样

`BPF_MAP_TYPE_LRU_HASH` / `LRU_PERCPU_HASH` 把满时的 `E2BIG` 换成了驱逐。内核文档直说："LRU hash 在 hash 表达到容量时会*自动驱逐*最久未用的条目。" update 路径（`htab_lru_map_update_elem` / `__htab_lru_percpu_map_update_elem`）在取桶锁*之前*就从 LRU 拿元素，因为驱逐可能需要的摘链操作本身要拿桶锁：`prealloc_lru_pop` → `bpf_lru_pop_free`。

`bpf_lru_pop_free`（在 `kernel/bpf/bpf_lru_list.c`）按级升级：

1. 从本 CPU 的本地 free 列表（`free_llist`）取一个节点；pending 节点不在这一步直接取，它们在下面的 fetch 里刷进全局列表。
2. 取不到就从全局列表取至多 `target_free` 个 free 节点进本地 free 列表——先把本 CPU 的 pending 节点按 ref bit 刷进全局 free/active/inactive 列表，再抽干全局 free 列表；然后重取本地 free 列表。
3. 全局 free 列表不够就 shrink：把 inactive 列表里 ref bit 已清除的节点经 `lru->del_from_htab` 移入 free 列表（从 hash 表摘链——与 delete 同一条摘链并释放路径，不清 value 字节）。
4. inactive 列表也榨不出时做 *force shrink*：无视 ref bit 摘下一个节点，优先 inactive 列表而非 active 列表，保证即使 map 很"热"也能逼出一个元素。
5. 仍没有就从某个 CPU 的本地列表偷取（free 再 pending）——按内核注释，轮转范围是"current CPU 和 remote CPU"（即全部 CPU），从 `next_steal` 开始——拿不到锁的 CPU 跳过。
6. 全失败则 `prealloc_lru_pop` 返回空，update 以 `-ENOMEM` 失败——LRU 变体满容量时的失败形态是分配失败，不是普通 hash 那个显式的 `E2BIG` full 检查。

文档把这组升级概括为：用 CPU 本地状态、从全局列表取空闲节点、从全局列表拉任意节点并从 hash 表摘除、再从任意 CPU 的列表拉任意节点。一个值得记住的推论：LRU map 的"满"意味着*现在驱逐*，所以容量压力之后，一个你从未 delete 过的 key 可能查回 `NULL`——是容量驱逐，不是 delete。用户态用 `bpf_map_info` 的 map `type` 字段确认自己在跟哪一类打交道。

## 如何验证

1. **盯 errno，不要盯消息。** 把普通 `BPF_MAP_TYPE_HASH` 填到 `max_entries`，然后发两个 update：新 key 必须以 `E2BIG` 失败（`bpf_map_update_elem` / libbpf 的 `bpf_map__update_elem` 返回的 `-ret` 经 `strerror`），对已存在 key 的 update 必须返回 0。之后 `bpf_map_delete_elem` 一次再看：新 key 又能成功。
2. **确认配置的上限。** `bpf_map_info` 会报 `max_entries` 和 map 类型，你以为在 1000 个 key 时失败，可能只是这个 map 创建时 `max_entries` 更小——下结论前先查 fd 的 info。
3. **把 LRU 驱逐和 bug 区分开。** 容量满载的 `BPF_MAP_TYPE_LRU_HASH` 上，你从未 delete 的 key 可能在别的 update 驱逐之后开始查失败；用 `bpf_map_lookup_elem` 返回 `NULL` 加上一次在普通 hash 上会 `E2BIG` 的新 key 插入来交叉验证。
4. **不依赖计数器数元素。** 普通 hash map 的 `bpf_map_info` 没有稳定的"当前元素数"字段；用 `bpf_map_get_next_key` 迭代 key 估算（稳定全表扫描用 batch 版），并注意若传入的 `cur_key` 恰好刚被删除，迭代会从表里*第一个* key 重新开始。
5. **确认是哪一类分配。** 若 map 用了 `BPF_F_NO_PREALLOC` 创建，满容量时的 update 失败可能是 `ENOMEM`（分配器耗尽）而非 `E2BIG`——惰性路径上的 `E2BIG` 是那个显式的 full 检查，而 `ENOMEM` 来自 `bpf_mem_cache_alloc` 真的缺内存。

## 答案的边界

- map *创建* 时的 `E2BIG`（`key_size + value_size` 过大、per-CPU 值超 `PCPU_MIN_UNIT_SIZE`、`max_entries > 1UL << 31`）与满 map 时的 update 共用 errno；单看返回码分不清是哪一种。
- `BPF_F_NO_COMMON_LRU` 的 per-CPU LRU map 让每个 CPU 维护自己的 LRU 列表，改变的是每 CPU 驱逐*哪个*元素，而不是"驱逐代替 E2BIG"这一行为本身。
- `is_map_full` 用的元素计数在大 map 上是一个 batch 为 32 的 `percpu_counter`，"满"的比较按设计是近似的；LRU map 上驱逐的保证是 LRU 性质，不是精确大小的承诺。
- 被复用元素里的 value 字节是否残留是另一条边界（2026-09-23 那篇有文档）：摘链与驱逐都不清 value 区域；`E2BIG` 与驱逐是容量决策，不是内存清刷保证。

## 参考资料

- [BPF_MAP_TYPE_HASH，含 PERCPU 与 LRU 变体（内核文档）](https://docs.kernel.org/bpf/map_hash.html)——默认预分配、`BPF_F_NO_PREALLOC`、"hash 表达到容量时自动驱逐最久未用条目"、LRU update 的升级步骤、`bpf_spin_lock` 同步。
- [kernel/bpf/hashtab.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/hashtab.c)——`alloc_htab_elem`（freelist pop → `E2BIG`；"当 map 满了而 update() 是替换旧元素"注释；`extra_elems` 交换）；`htab_map_update_elem`（先查找、`BPF_F_LOCK` 就地捷径）；`htab_lru_map_update_elem` / `__htab_lru_percpu_map_update_elem`（先分配后锁桶）；`is_map_full`（`percpu_counter` batch 32 对 `atomic_t`）；创建时 `E2BIG` 检查。
- [kernel/bpf/bpf_lru_list.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/bpf_lru_list.c)——`bpf_lru_pop_free`（本地 free 列表、跨 CPU 偷取、inactive shrink 经 `del_from_htab`、force shrink 无视 ref bit）。
- [include/uapi/linux/bpf.h](https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/bpf.h)——`bpf_map_update_elem(2)` 的 `E2BIG` 契约文字、`BPF_F_LOCK`/`BPF_F_NO_PREALLOC`/`BPF_F_NO_COMMON_LRU` 标志、`struct bpf_map_info`（`max_entries`、map 的 `type` 字段）。

## 当日社区讨论

如实的覆盖说明：本次运行两个 watchlist 选中的 Slack 存档**可达但滚动窗口内零消息**（快照 ok，0 字节，0 条消息），allowlist 里的 Discord 频道、公开 bpf 邮件列表与 r/eBPF 仍仅限 visible-browser，且本次运行没有可用的浏览器会话——因此 2026-09-25 没有任何私有社区材料可用。上文问题是回退选择：这是被监控的 eBPF 开发社区中一个真实、反复出现的从业者边界（"我的 map 满了以后拒收新 key，但更新同一个 key 却没事" / "我 LRU map 里一个从没 delete 过的 key 不见了"），完全以上述公开一手资料（内核 BPF 文档与上游 `kernel/bpf/hashtab.c`、`bpf_lru_list.c`、uapi 头文件）为依据，而不是任何具体 thread。

该领域反复出现的从业者主题——"我的 hash map value 变旧了，新会话没被记录，但老会话还在计数"——正落在这条边界上：一旦 key 空间超出 `max_entries`，每个新 key 都在 update 处以 `E2BIG` 被丢弃，而热的旧 key 照常工作；症状读起来像"新会话丢了、老的还在计数"，修法是给 `max_entries` 留出余量（或换 `BPF_F_NO_PREALLOC`、或换 LRU 变体并接受驱逐），而不是加重试循环。LRU 一侧则反复以另一种惊讶出现：从未被 delete 的 key 在被驱逐后读回 `NULL`，常被误诊为 delete bug，实际是容量驱逐。公开文档锚定的收尾规则是：普通 hash map 的 `max_entries` 是分配时执行的硬上限（新 key `E2BIG`，更新旧 key 豁免，spin-locked 值就地更新）；LRU hash map 把同样的容量变成一次驱逐决策——"满"意味着"更旧的东西现在离开"，而不是"下一次插入失败"。本页不复现任何私有文字、身份、频道或链接。
