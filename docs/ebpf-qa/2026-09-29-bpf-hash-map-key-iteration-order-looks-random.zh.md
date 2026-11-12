# 为什么 BPF 哈希 map 的键迭代顺序是随机的，而不是插入顺序？

简短回答：BPF 哈希 map 按"键字节哈希值"把键放进开放寻址的桶里，而不是按任何调用方可见的顺序存储。内核在 map 创建时取一个随机的每 map 种子，用 jhash 对键做哈希；新键被插入到对应桶的冲突链表头部。`bpf_map_get_next_key` 按桶下标升序遍历，逐个交出每个非空链表的头元素，所以顺序由哈希散射加链头插入决定，而那个随机种子又让它在不同 map、甚至不同启动之间都不相同。内核只保证覆盖每一个键，不保证任何特定顺序；这个 API 没有游标，一次遍历总是从 `NULL` 键开始。

## 机制

**桶由哈希选定，带一个随机的每 map 种子。** map 分配时用 jhash 对键做哈希，喂入的种子是随机的，除非该 map 显式以测试专用方式创建：

```c
htab->n_buckets = roundup_pow_of_two(htab->map.max_entries);
...
if (htab->map.map_flags & BPF_F_ZERO_SEED)
	htab->hashrnd = 0;
else
	htab->hashrnd = get_random_u32();
```

查找路径再用掩码把它降成桶下标：

```c
static inline u32 htab_map_hash(const void *key, u32 key_len, u32 hashrnd)
{
	if (likely(key_len % 4 == 0))
		return jhash2(key, key_len / 4, hashrnd);
	return jhash(key, key_len, hashrnd);
}
```

`jhash` 与 `jhash2` 把种子混入其初始状态，因此两个装相同键的 map，桶布局不同——除非两者都用零种子标志创建。

**插入把新元素放到链头。** 更新时分配元素，并把它推到同一桶的 null 终止 RCU 链表旧元素之前，注释写明了原因：

```c
/* add new element to the head of the list, so that
 * concurrent search will find it before old elem
 */
hlist_nulls_add_head_rcu(&l_new->hash_node, head);
```

于是在同一个桶内，最近插入的键排在前面。但桶与桶之间完全不是插入顺序。

**迭代器走的是桶下标，不是历史链表。** `htab_map_get_next_key` 先定位给定键的桶，若该桶里还有下一个元素就返回它；否则向上扫描桶下标，直到找到下一个非空链表；`NULL` 键则直接跳到第一个非空桶：

```c
static int htab_map_get_next_key(struct bpf_map *map, void *key, void *next_key)
{
	...
	if (!key)
		goto find_first_elem;

	hash = htab_map_hash(key, key_size, htab->hashrnd);
	head = select_bucket(htab, hash);

	l = lookup_nulls_elem_raw(head, hash, key, key_size, htab->n_buckets);
	if (!l)
		goto find_first_elem;

	/* key was found, get next key in the same bucket */
	next_l = hlist_nulls_entry_safe(rcu_dereference_raw(hlist_nulls_next_rcu(&l->hash_node)),
				  struct htab_elem, hash_node);
	if (next_l) {
		memcpy(next_key, next_l->key, key_size);
		return 0;
	}

	/* no more elements in this hash list, go to the next bucket */
	i = hash & (htab->n_buckets - 1);
	i++;

find_first_elem:
	for (; i < htab->n_buckets; i++) {
		head = select_bucket(htab, i);
		next_l = hlist_nulls_entry_safe(rcu_dereference_raw(hlist_nulls_first_rcu(head)),
					  struct htab_elem, hash_node);
		if (next_l) {
			memcpy(next_key, next_l->key, key_size);
			return 0;
		}
	}

	/* iterated over all buckets and all elements */
	return -ENOENT;
}
```

三个性质叠加：哈希种子在每 map 实例中是随机的，所以键落到桶的顺序无法从键值本身预测；插入总是落在链表头部，所以桶内的新旧可见、桶间不可见；而遍历本身按桶下标推进，桶下标是哈希的函数，与任何东西的插入时间无关。同一个函数被注册在普通、per-CPU、LRU 与 hash-of-maps 各变体的 `bpf_map_ops` 块里，所以这些 map 都用同样的方式迭代。

## 验证与调试路径

1. 插入一个已知的键序列并记录迭代顺序。例如按升序插入 16 个整型键，然后用 `bpf_map__get_next_key` 从 `NULL` 迭代到 `-ENOENT`，逐个记下键。这个顺序几乎肯定不是插入顺序。
2. 证明种子是差异的来源。再建一个同尺寸的 map，插入完全相同的键，再迭代一次。两个序列应当不同。在测试专用环境里，用 `BPF_F_ZERO_SEED` 创建两个 map，两者的序列就会一致，从而把随机种子隔离为唯一的区分因素。
3. 演练契约。确认从 `NULL` 开始能取到某个键，从某个返回键逐步前进，最终在 `-ENOENT` 之前恰好覆盖每一个已插入的键；批量 `BPF_MAP_GET_NEXT_KEY` syscall 路径也表现出同样的无序行为。
4. 交叉核对各变体。`BPF_MAP_TYPE_LRU_HASH`、per-CPU 哈希各变体与 `BPF_MAP_TYPE_HASH_OF_MAPS` 都注册到同一个 `htab_map_get_next_key` 遍历，因此迭代时同样是无序的；LRU 淘汰只是叠加在元素上，不引入任何顺序。哈希家族里唯一走不同遍历的是 `BPF_MAP_TYPE_RHASH`，它由 rhashtable 支撑、经 `rhashtable_next_key` 迭代，同样无序。

一个最小的 libbpf 遍历。高层 helper 接收 `bpf_map` 对象；底层 syscall 封装接收文件描述符：

```c
#include <bpf/bpf.h>
#include <linux/bpf.h>

__u32 cur = 0, next;
int map_fd;
struct bpf_map *map; /* 同一个 map 的 libbpf 对象 */

while (1) {
	int err = bpf_map__get_next_key(map, cur ? &cur : NULL, &next, sizeof(next));
	/* 底层形式是 bpf_map_get_next_key(map_fd, &cur, &next)；
	 * 遍历穷尽时返回 -ENOENT */
	if (err)
		break; /* 迭代到末尾的 -ENOENT */
	cur = next;
	/* 记录该键；不要假设它按插入顺序到达 */
}
```

## 局限

- 遍历是无状态的。每次 `bpf_map_get_next_key` 调用都是对上一个键做一次全新查找加一次桶扫描；没有一个能跨越 map 变更而存活的游标。如果你在遍历中删除或新增条目，看到的序列就是哈希表在那一刻的样子。
- `BPF_F_ZERO_SEED` 文档写明只用于测试。它不是让应用层迭代可复现的支持手段；它把 jhash 种子清零，使给定相同键时桶布局确定，内核 uapi 注释也说明只应在测试中用。
- 无序契约贯穿整个 htab 家族。`BPF_MAP_TYPE_HASH`、`PERCPU_HASH`、`LRU_HASH`、`LRU_PERCPU_HASH` 与 `HASH_OF_MAPS` 都绑定到同一个 `htab_map_get_next_key`，所以没有任何一个能承诺顺序；唯一不同的哈希家族遍历是 `BPF_MAP_TYPE_RHASH`（rhashtable 支撑），它同样无序。

## 参考

- `htab_map_get_next_key`、桶分配、`get_random_u32` 种子、链头插入：https://elixir.bootlin.com/linux/latest/source/kernel/bpf/hashtab.c
- `jhash` / `jhash2` 种子混合：https://elixir.bootlin.com/linux/latest/source/include/linux/jhash.h
- `BPF_F_ZERO_SEED` uapi 标志与 `BPF_MAP_GET_NEXT_KEY` 命令文档：https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/bpf.h
- `bpf_map__get_next_key` 用户态 helper（NULL 键 = 第一个键，穷尽时 `-ENOENT`）：https://github.com/libbpf/libbpf/blob/master/src/libbpf.h
- `bpf` syscall 手册页：https://man7.org/linux/man-pages/man2/bpf.2.html

## 当日社区讨论

覆盖情况：本次运行滚动窗口内可触达的 watchlist 已选社区归档为 **0** 个——运行开始时不存在 2026-09-29 的快照文件，且允许名单里的浏览器专属社区在本次运行中无可见浏览器会话可用，标记为**不可触达，而非无讨论**。下文不复现任何工作区、频道、参与者或消息 URL。

本条目的问题改自 Q&A 系列自身：2026-09-25 的条目已经讲过为什么满的哈希 map 会拒绝新键报 `E2BIG` 而更新仍会成功，那个答案让哈希表内部实现成为固定背景。读完那一页的读者自然会追问下一个问题：遍历这个 map 时，键到底按什么顺序回来。上面的答案依据的是 References 里的公开一手资料，而不是某个受监控的社区线程；当日没有可汇总的讨论窗口，因此没有为填补空白而虚构任何内容。
