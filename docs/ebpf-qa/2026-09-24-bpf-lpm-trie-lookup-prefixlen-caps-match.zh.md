# 为什么 BPF LPM-trie 查找会返回更不具体的已存储前缀，key 的 prefixlen 字段又是怎么决定哪条记录命中的？

**简短回答：** 把查找 key 的 `prefixlen` 设为该 map 的最大前缀长度（4 字节 IPv4 数据域是 32，16 字节 IPv6 数据域是 128）。内核会把匹配截断在 `min(node->prefixlen, key->prefixlen)`（`kernel/bpf/lpm_trie.c`）上，因此一个带较小 `prefixlen` 的查找 key 相当于告诉遍历"最多只匹配这么多 bit"，即使存在更具体的已存储前缀，查找也可能停在（或回落到）一条更不具体的记录上。内核文档把这条规则讲得很直白：做最长前缀查找时，`key->prefixlen` 应当等于 `max_prefixlen`。如果查找 key 的 `prefixlen` 超过 map 上限，查找返回 `NULL`（而 `update`/`delete` 返回 `-EINVAL`），并且中间节点从不携带 value。

## 查找如何遍历 trie

`trie_lookup_elem` 从根节点开始遍历。每个节点保存自己的 `prefixlen` 和 `data`（按大端存储，`data[0]` 是最前一个字节），并有两条子指针指向更具体的匹配。在每个节点上：

- `__longest_prefix_match` 统计节点与 key 相同的连续前导 bit 数，但这个计数被截断在 `limit = min(node->prefixlen, key->prefixlen)`（第 172 行）。比较按 64/32/16/8-bit 大端分块进行（`be64_to_cpu`、`be32_to_cpu`、…），所以 key 的 `data` 第 0 字节是最高有效字节。
- 如果匹配达到了 map 的最大前缀（`matchlen == trie->max_prefixlen`），该节点就是一个全长的精确匹配，被直接返回（第 258–262 行）；只有当一条最大长度的已存储前缀（例如 /32）与 key 匹配时才会触发这条分支。
- 如果匹配比节点自身的 `prefixlen` 更短（`matchlen < node->prefixlen`，第 268–269 行），遍历中止，返回路径上最后一个*非中间*节点——即到目前为止完全匹配的最长已存储前缀。
- 带 `LPM_TREE_NODE_FLAG_IM` 标志的节点是人为插入的中间节点，不携带 value，也永远不会被返回；它只为了让两个更具体的子节点共享同一个父节点而存在（第 271–275 行）。
- 否则，最近访问过的携带 value 的节点成为候选，key 的下一个 bit（`extract_bit(key->data, node->prefixlen)`）决定沿哪条子指针继续向下（第 281–283 行）。

`max_prefixlen` 是 `data_size * 8` bit（第 596 行）——即 key 数据长度的 8 倍，所以 4 字节 key 是 32，16 字节 key 是 128。

## 为什么查找 key 的 prefixlen 会限制匹配范围

上限就是 `limit = min(node->prefixlen, key->prefixlen)`。`__longest_prefix_match` 的累计计数一旦达到 `limit` 就立即返回（每个分块结尾都有 `if (prefixlen >= limit) return limit;`）。因此遍历被告知：这个 *key* 最多只匹配 `key->prefixlen` 个 bit，不管更具体的节点里实际还有多少 bit 是相同的。

用文档里自己的数字做一个例子：存储 `192.168.0.0/16` value 1 和 `192.168.0.0/24` value 2。/24 是同一条路径上更具体的前缀，所以它直接作为 /16 的 `child[0]` 挂上（只有当同一 bit 区间里出现两个更具体的子节点、必须共享父节点时——例如文档里 `192.168.1.0/24` 与 `192.168.128.0/24` 通过 /23 分裂——才需要中间节点）。

- 查找 `192.168.0.5`，`prefixlen = 32`：遍历下探到 /24 节点，该节点自身 `prefixlen` 是 24；匹配达到该节点的 24 bit，它是路径上最后一个携带 value 的节点，且没有更深的子节点可下探——于是返回 /24，value 2，即最长的已存储匹配。（精确匹配分支只在一条最大长度 /32 的已存储前缀与 key 匹配时才会触发。）
- 查找 `192.168.0.5`，`prefixlen = 16`：遍历到达 /24 节点，但匹配被截断在 16；此时 `matchlen (16) < node->prefixlen (24)`，遍历中止，返回路径上最后找到的节点——/16，value 1。更具体的 /24 记录被静默跳过。
- 查找时 `prefixlen = 33`（超过 32 上限）：`trie_lookup_elem` 在第 244 行返回 `NULL`；`update`/`delete` 返回 `-EINVAL`（第 338、469 行）。

文档的一句规则："做最长前缀查找时，`key` 的 `prefixlen` 应当设为 `max_prefixlen`。"

## 真正不安全的角落

- **查找 key 的 `prefixlen` 低于 map 上限，就会截断结果。** 把 key 截断到某条已存储前缀的长度，trie 就会停在*未被截断的*最长匹配上，更具体的已存储前缀会被绕过。始终传最大值（32/128）。
- **`prefixlen` 超过 map 上限时，查找是"无命中"而不是报错。** 查找返回 `NULL`（第 244 行）；只有 `update`/`delete` 才把它报成 `-EINVAL`。
- **key 的 data 是大端。** 比较是对存储与查找两侧数组做 `be32_to_cpu`，`data[0]` 是最高有效字节；一个把地址存成 native 整型的小端机会路由错误。规范 key 结构是 `struct bpf_lpm_trie_key_u8`（`prefixlen` + 柔性数组 `data[]`）；旧的 `struct bpf_lpm_trie_key` 已弃用。
- **中间节点不携带 value。** 它们是内部结构节点（`LPM_TREE_NODE_FLAG_IM` 位），永远不会被返回；分裂很多的 trie 只会返回最近的非中间祖先节点的 value，即使只有叶子才带 value。
- **map 创建时必须带 `BPF_F_NO_PREALLOC`。** 内核文档要求这个标志；最大前缀长度必须是 8 的倍数，范围 8 到 2048 bit（`LPM_DATA_SIZE_MAX` 为 256 字节，第 557 行）。

## 如何验证

1. **精确匹配优先。** 存储 `192.168.0.0/16`（value 1）与 `192.168.0.0/24`（value 2）。key 为 `192.168.0.5`、`prefixlen = 32` 的 `bpf_map_lookup_elem` 返回 value 2。
2. **截断效果。** 同样两条记录；key 为 `192.168.0.5`、`prefixlen = 16` 时返回 value 1——匹配被截断在 16，/24 被跳过。
3. **超过上限。** key 的 `prefixlen = 33` 使查找返回 `NULL`；`bpf_map_update_elem`/`bpf_map_delete_elem` 返回 `-EINVAL`。
4. **迭代顺序。** `bpf_map_get_next_key` 从最左叶子开始遍历，因此更具体的 key 先于更不具体的出现；第一次调用传 `NULL`。上游 selftest `tools/testing/selftests/bpf/test_lpm_map.c` 覆盖了这些路径。

## 答案的边界

- 这说的是*查找*语义：存储不受影响——一条已存储的 /24 仍然在，被正确截断的查找也能找到它。截断只是限制了一次*查找*被允许匹配多少 bit。
- trie 是不平衡的；一个装满的 IPv4 trie 高度是 32（每个 bit 一层）。这个上限是某次查找能表达的匹配边界，不是存储正确性的 bug。
- 其他 map 类型没有"前缀"概念；这条边界只适用于 `BPF_MAP_TYPE_LPM_TRIE`。

## 参考资料

- [BPF_MAP_TYPE_LPM_TRIE（内核文档）](https://docs.kernel.org/bpf/map_lpm_trie.html)——`prefixlen`/`max_prefixlen` 规则、大端 `data`、`BPF_F_NO_PREALLOC` 要求、`bpf_map_lookup_elem`/`update`/`delete`/`get_next_key` 契约、IPv4/IPv6 数据长度、`192.168.0.0` 示例，以及 selftest 指引。
- [kernel/bpf/lpm_trie.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/lpm_trie.c)——`trie_lookup_elem`（遍历、`matchlen` 截断、中间节点标志）；`__longest_prefix_match`（第 172 行的 `limit = min(node->prefixlen, key->prefixlen)`，大端分块比较）；`trie->max_prefixlen = trie->data_size * 8`（第 596 行）；`LPM_DATA_SIZE_MAX 256`（第 557 行）；`key->prefixlen > max_prefixlen` 防护（第 244、338、469 行）。
- [include/uapi/linux/bpf.h](https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/bpf.h)——`struct bpf_lpm_trie_key`（已弃用）、`struct bpf_lpm_trie_key_hdr`、`struct bpf_lpm_trie_key_u8`；`enum bpf_map_type` 中的 `BPF_MAP_TYPE_LPM_TRIE`。

## 当日社区讨论

如实的覆盖说明：两个 watchlist 选中的 Slack 存档本次运行**不可访问**——只读的快照读取器拒绝覆盖一个已存在的 0 字节快照文件，Step 0 快照因此返回 `output_exists`，没有读到任何存档内容。allowlist 里的 Discord 频道与公开邮件列表仅限 visible-browser，且本次运行没有可用的浏览器会话——因此 2026-09-24 没有任何私有社区材料可用。上文问题是回退选择：这是被监控的 eBPF 开发社区中一个真实、反复出现的从业者边界，完全以上述公开一手资料（内核 BPF 文档与上游源码）为依据，而不是任何具体 thread。

该领域反复出现的从业者主题——在 XDP 或 TC 数据路径里建一张 CIDR/路由表，然后问"为什么命中了更不具体的路由"——正是落在这条边界上：查找 key 的 `prefixlen` 会截断最长前缀匹配，文档的规则是把它设为 `max_prefixlen`（IPv4 是 32，IPv6 是 128），更具体的已存储前缀才会真正胜出。公开文档锚定的两个后续点在此反复出现。其一是**大端**意外：`data` 数组按网络字节序解释，把地址存成 native 小端整型的主机就会静默路由错误，而人们容易把它误读成"前缀没匹配上"，其实是字节序 bug。其二是**超过上限**的角落：传入超过 map 上限的 `prefixlen` 会让查找返回 `NULL`（而不是报错），看起来像"无命中"，其实是 key 形状的 bug；而同一个越界值在 `update`/`delete` 上返回 `-EINVAL`，两条路径对同一个 key 的表现不一致。
