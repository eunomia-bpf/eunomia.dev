---
date: 2026-09-17
slug: cxl-memory-tier-isolation
title: "容器能保证自己的内存不被放进 CXL 吗？"
description: "Linux 可以把本地 DRAM 和 CXL 暴露成不同 NUMA tier，但 cpuset、reclaim demotion、迁移与共享页仍不能保证容器整个生命周期的内存驻留隔离。"
tags:
  - 每日报告
  - Linux
  - CXL
  - Memory Tiering
  - 容器
  - cgroups
research_question: "当 allocation policy、reclaim demotion、共享页和多租户 tiering 可能做出不同 placement decision 时，Linux 应该怎样表达并验证每个容器的 memory-tier isolation？"
source_cutoff: 2026-09-17
status: daily-report
---

# 容器能保证自己的内存不被放进 CXL 吗？

CXL 内存接进 Linux 后，常见做法是把本地 DRAM 和较慢的 CXL memory 暴露成不同 NUMA node。这样很容易形成一个直觉：把容器允许使用的 node 写进 `cpuset.mems`，不把 CXL node 放进去，就等于给这个容器画出了一条内存隔离边界。

这个配置很有用，但它还不是完整的隔离契约。

Linux 里能改变物理页面位置的路径不只有第一次 allocation。内存压力出现后，reclaim 可以把页面 demote 到更慢的 memory tier；修改 cpuset 后触发的 migration 明确允许不完全成功；file-backed page、shared library 和其他共享映射可能同时被多个 cgroup 使用，却只有一个物理页面；tiering controller 也可能优先优化全局 hotness 或公平性，而不是保持某个 tenant 最初的 placement intent。

Linux 当前的 CXL 文档直接暴露了这个边界。reclaim 文档说明，较早的 demotion 路径不会遵守 `cpuset.mems_allowed`；后续实现会尽量尊重它，但由别的 cgroup 创建的共享内存仍可能被 demote 到当前 consumer 不希望使用的节点。因此，文档明确提醒：`mems_allowed` 依然不能提供对 remote node 的完美隔离。

所以真正的问题不是 Linux 能不能限制内存放在哪里。它当然能。更难的问题是：**当一个容器说“这类内存绝不能驻留在 CXL tier”时，这句话到底约束哪些页面、哪些迁移路径，以及内存压力下系统应该怎样失败？**

<!-- more -->

## CXL allocation policy 不等于内存驻留隔离 policy

Linux 的内存 placement 首先经过 NUMA policy 和 page allocator。把本地 DRAM 与 CXL memory 暴露成不同 NUMA node 后，`cpuset.mems` 可以限制一个 cgroup 里的 task 被允许在哪些 memory node 上分配；`cpuset.mems.effective` 则显示经过 parent 约束和在线节点状态之后真正生效的集合。

光这一层就已经不是一个永远不变的 node list。cgroup v2 文档指出，对已经有 task 的 cgroup 修改 `cpuset.mems` 时，内核会尝试把现有页面迁移到新的 node 集合，但 migration 可能无法全部完成，一些页面可以留下。因此文档建议，能在 workload 启动前设好 `cpuset.mems` 就尽量不要运行中频繁修改。

Tiered memory 又加入了另一条路径。开启 NUMA demotion 后，reclaim 在本地内存吃紧时，可以优先把页面迁到更低一层 memory tier，而不是立刻 swap 或直接回收。CXL reclaim 文档描述的正是这条 vmscan 路径，tier 之间的关系来自 kernel memory-tier topology，以及 HMAT、CDAT 之类的平台性能信息。

于是，一个页面在生命周期里至少会遇到两次独立的 placement decision：

```text
第一次分配
    -> page allocator / NUMA policy / cpuset

之后出现内存压力
    -> reclaim / demotion policy / memory tier topology
```

如果这两条路径执行的不是同一个 tenant contract，那么“允许从哪里分配”就不能自动推出“之后永远驻留在哪里”。一个页面可能因为 local DRAM 不够而第一次就落在 CXL；也可能一开始在本地，冷下来以后被主动 demote；还可能是在 cpuset 更新后，因为 migration 没有完全成功而留在旧节点。

## 共享页会把“这是谁的内存”变成策略问题

Private anonymous memory 是最容易处理的情况。页面通常有比较清楚的 memcg charge，也容易判断哪个进程族会受到 placement 的性能影响。

共享页就不一样。file-backed page、shared library、tmpfs 对象或其他共享映射，可以同时被多个 cgroup 的 task 访问，但物理上仍然只有一份 page。Linux 必须给这份物理页选择一个 residency，而“谁最先创建了它”或者“账记在哪个 memcg 上”并不等于“只有这个 cgroup 在乎它的延迟和隔离策略”。

假设容器 A 允许使用 CXL，容器 B 不允许，两边又都在执行同一个共享文件里的代码页。如果这个 page 被 charge 给 A，随后 reclaim 认为它可以 demote 到 CXL，那么 B 之后访问同一个物理页时，就会读到一个它自己的 placement policy 永远不会主动选择的 tier。

一种最直接的解法是按 placement domain 把共享页复制成多份，但这会牺牲内存容量和 page cache sharing。继续共享一份，则必须显式定义 policy conflict 怎么解决。

Linux 当前 CXL reclaim 文档点名了这一类情况：即使 demotion 开始尊重 `mems_allowed`，由其他 cgroup 实例化的 shared memory 仍可能被 demote，因此 `mems_allowed` 不能保证完全远离 remote node。

所以，一个完整的 tier-isolation 设计至少要回答三个问题：

1. placement authority 属于 allocating cgroup、当前 charger、所有 active consumer，还是物理 page 本身？
2. 多个 consumer 的 tier policy 互相冲突时，谁优先？
3. local memory 无法满足 hard policy 时，系统应该 reclaim、swap、复制、throttle、allocation failure，还是 OOM，而不是偷偷 fallback 到 CXL？

如果这些都没有定义，“这个容器的 cpuset 里没有 CXL”只能说明配置意图，不能证明 workload 所访问的所有字节在整个生命周期里都不会落到 CXL。

## 多租户公平性和硬隔离是两种不同契约

Tiering 做得公平，不代表它提供了 hard isolation。

TPP 说明了为什么 CXL 需要透明 page placement：把 hot page 尽量留在较快的本地内存，提前把 colder page demote 下去，并为新的 hot allocation 保留 local headroom。这个目标主要是效率与性能。

2026 年的 Equilibria 则把 multi-tenancy 往前推进了一大步。它的生产部署研究显示，普通的全局 tiering 在多租户环境里会产生公平性问题：更热的 workload 可以吃掉大部分 local memory；晚启动的 workload 可能长期处于劣势；一个不断在 local 与 CXL 之间 thrash 的 tenant 还会消耗 migration 工作，拖累邻居。Equilibria 增加了 per-container tier observability、local-memory lower protection 与 upper bound、受控的 promotion/demotion，以及 thrashing mitigation。论文报告，相对 Linux/TPP baseline，生产 workload 最多提升 52%，benchmark 最多达到 1.7x。

这些机制解决的是一个真实问题：**每个 tenant 应该分到多少 fast memory，才能让 colocated workload 达到自己的性能目标？**

Hard tier isolation 问的却是另一件事：**这个 workload 的数据到底允不允许出现在某个 tier？**

这两个 policy 可以同时存在，但不能互相替代。fairness controller 完全可能故意把每个 tenant 的一部分数据放进 CXL；而真正 local-only 的 workload 即使因此遭遇 reclaim pressure 或 OOM，也可能仍然要求拒绝 CXL。生产接口不应该用同一个 knob 同时表达“我要多少 fast memory”和“我绝不允许去哪一层”。

这和之前的 [GPU 内存 placement 报告](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/) 有相似之处：真正可运营的 placement mechanism 不只要做 decision，还要有证据说明为什么这样放，以及现在的 placement 是否仍然满足最初的 contract。CXL 的特殊之处在于 Linux 的 reclaim、cpuset、cgroup 和 shared-page ownership 会一起参与这个生命周期。

## 位置计数还不能证明策略真的被遵守

Linux 已经提供不少有用证据。`memory.numa_stat` 可以按 NUMA node 和 memory type 展开一个 memory cgroup 的 footprint。NUMA topology 与 memory-tier sysfs 可以告诉用户哪些 node 属于哪一层。VM 统计可以看到 migration 和 demotion 活动。Equilibria 进一步证明，per-container 的 local/CXL 使用量以及 promotion、demotion counter 对实际运营很有价值。

但“事后看见页面在哪里”不等于“有一个 policy witness 证明它为什么在那里”。

如果 operator 发现一个号称 local-only 的 workload 有 2 GiB 页面出现在 CXL node，上游原因可能完全不同：

- 页面在修改 cpuset 之前就已经分配；
- cpuset 更新后的 migration 没有全部完成；
- reclaim demotion 走了一套不同的 eligibility rule；
- shared page 的 charge 或 owner 属于其他 cgroup；
- tiering controller 有意把它迁走；
- memory node hotplug 或 topology 变化改变了 effective policy；
- workload 正在读一个共享物理页，但 accounting identity 和实际 consumer 不一致。

一个 node counter 可以告诉你症状，却不一定说明哪条 policy path 造成了它。对于 hard isolation，系统还需要知道这是合法 exception，还是 policy escape。

## 现有研究还缺什么

第一个缺口是 **contract semantics**。Linux 分别提供 cpuset eligibility、cgroup memory control、NUMA policy、tier topology 和 reclaim behavior，却没有一个统一声明，例如：“这个 cgroup 的 private anonymous page 只能在 local tier 0；shared executable page 只有在所有 active consumer 都允许时才能进入 remote tier；否则 fail closed。”

第二个缺口是 **shared-page policy composition**。一个物理 page 可能同时服务 placement requirement 不同的 consumer。谁承担 charge、谁完成第一次 allocation，并不自动等于谁拥有 placement authority。

第三个缺口是 **failure behavior**。所谓 hard local-only 只有在 local memory 不够时仍然有明确语义才是真的。fallback 到 CXL 会违反规则；拒绝 CXL 可能导致更多 reclaim、swap、throttling、allocation failure 或 OOM。这些生产结果差别很大，应该由 policy 主动选择。

第四个缺口是 **跨 migration path 的可解释性**。现有 counter 能显示最终 residency，但强隔离声明还需要回答：页面为什么跨 tier、使用的是哪个 policy generation、这次 transition 当时到底允许不允许。

Equilibria 已经把多租户 CXL 的 fairness 与 observability 向前推进很多。这里剩下的问题更窄：如何把 best-effort placement preference 变成一个可以验证的 per-tenant residency contract，而且覆盖共享页与内存压力下的行为。

## 兼具学术价值与生产价值的方向

### 1. 做一个带明确失败语义的 tier-residency hardwall

可以在 cgroup 层把 **allocation preference** 与 **residency permission** 正式拆开。

例如：

```text
tier_policy:
  allowed_tiers: [local]
  private_anon: hard
  file_private: hard
  shared_file: composed
  on_pressure: reclaim_then_swap
  on_conflict: deny_demotion
  generation: 184
```

语法本身不重要。真正重要的是：任何可能改变 physical residency 的路径都检查同一个 policy generation，包括第一次 allocation、reclaim demotion、NUMA migration、显式迁移、cpuset 更新，以及 tiering controller 的动作。

Hard policy 还必须定义“满足不了时怎么办”。local memory 已满之后，可以先 reclaim 其他 page、走 swap、throttle 这个 cgroup、在 API 允许时返回 allocation failure，或者进入 cgroup-scoped OOM。悄悄落到 CXL 应该是一个明确的 `best_effort` 模式，而不是意外 fallback。

研究原型可以在 Linux 里给 memcg 加 tier mask 与 pressure behavior，并让 allocation 与 migration eligibility 都走同一套检查。生产集成点可以是 container runtime 或 Kubernetes node agent，把 workload class 翻译成 tier policy。

评测要让两个以上 tenant 在越来越大的 local-memory pressure 下运行 private anonymous、private file-backed 与 shared mapping。最主要的指标应该是 **forbidden-residency byte-seconds**：多少不该进入某 tier 的字节，在那里停留了多久。其次再测 reclaim amplification、swap traffic、OOM 次数、tail latency 与闲置 CXL 容量。

如果在同一套 migration 与 pressure matrix 里，只靠现有 cpuset/mempolicy、并在 task 启动前正确配置，就已经能做到零 forbidden residency，那么这个方向没有必要。另一种失败情况是 strict enforcement 放大的 OOM、swap 或 tail latency 成本远高于显式 best-effort 策略。

### 2. 把共享页当成真正的 multi-owner placement object

第二个方向是不再假设“一份 memcg charge 就等于这个 shared physical page 的 placement authority”。

原型可以给 shared folio 保存一个紧凑的 **placement-interest set**。不一定永久枚举所有 mapper，只追踪带 hard tier restriction 的 cgroup，或者近期确实活跃访问的 consumer；表示方式可以是小型 inline set，再配一个 bounded overflow summary。

大家 policy 一致时很好处理。冲突时，kernel 需要显式 composition rule。例如 hard deny 优先，共享页只能留在所有 consumer allowed tier 的交集里；对只读 file page 按不兼容的 placement domain 选择性复制；或者在允许 remote residency 时显式记录 policy-violation budget。

学术问题在于：能不能用足够便宜的 consumer 信息提高 placement correctness，而不让每个 page 都背一个巨大的多租户 metadata object。生产目标包括 shared library、page cache、model weight 和合并服务器里常见的其他 read-mostly object。

评测可以人为构造 tier policy 冲突、access rate 已知的共享映射，对比当前 charge-based behavior、hard intersection、selective duplication 与 demand-weighted policy。指标包括 forbidden residency、复制内存开销、remote-access latency、migration volume 与 metadata cost。

如果真实生产里 shared-page conflict 极少，或者 selective duplication 消耗的容量和 CPU 比接受一个明确记录的 remote exception 更高，这个方向就不值得做。

### 3. 建一个专门攻击 tier-isolation 声明的 conformance benchmark

Hard contract 最难的不是写出来，而是证明压力上来时不会从别的路径漏掉。

Benchmark 应该先定义一小组明确 policy，再主动制造会破坏它的场景：在修改 `cpuset.mems` 前后分别 allocation；用 local-memory pressure 强制 reclaim 与 NUMA demotion；让 policy 冲突的 cgroup mmap 同一个文件；测 tmpfs 和 shared anonymous mapping；改变 tenant launch order；主动制造 tier thrashing；组合 zswap 与 demotion；在平台支持时 hot-add 或 offline memory node；让 policy update 和 migration 并发。

测试 harness 观察到每一次 physical-page transition 时，都尽量记录 source tier、destination tier、page class、owner/consumer cgroup、policy generation 与 migration reason。Oracle 再把 transition 分类成 permitted、explicitly degraded 或 forbidden。

这里真正有用的指标不只是吞吐。应该分别报告 policy escape 次数、最长 escape duration、无法解释的 migration、false violation report、migration overhead，以及 policy change 后多久恢复一致。一个 controller 即使快 5%，只要偶尔突破 hard residency rule，就不能和从不突破的实现说成“等价”。

学术价值是给 tier isolation 建一个统一 failure taxonomy；生产价值是，在把 CXL 开给有数据位置、低延迟或可预测性要求的 workload 之前，有一套 qualification test。

如果现有 kernel selftest 和 memory-tier test suite 已经覆盖同样的 cross-cgroup、shared-page、pressure 与 policy-generation failure class，并且有同等强度的 oracle，那么这个 benchmark 就是重复工作。第一轮实验应该先尝试证明“新的 counterexample 一个都找不到”。

## 现在部署时可以怎么做

目前更稳妥的做法是：把 `cpuset.mems` 当成重要的 placement constraint，但不要把它直接解释成“这个 workload 整个生命周期里访问的每个字节都永远只会在这些 NUMA node”。

能在启动 task 之前设置 memory node 就提前设置；检查 `cpuset.mems.effective`，不要只看请求值；如果开启 NUMA demotion，要搞清楚当前 kernel 的 reclaim path 怎样处理 cpuset eligibility；用 `memory.numa_stat` 和 tier migration counter 验证实际 residency；shared mapping 与 private anonymous memory 分开测试；同时提前决定 local memory 不够时到底允许 CXL fallback、swap、throttling、allocation failure 还是 OOM。

如果真正要的是性能 SLO，那么 Equilibria 这类 lower protection / upper bound 的 fair-share 模型，比硬说每个 tenant 都需要 hardwall 更合适。如果真正要的是隔离，就应该把 forbidden state 写清楚，然后专门在内存压力下攻击它。

之前的 [逐页 eBPF 内存归因报告](https://eunomia.dev/zh/research/page-level-ebpf-memory-attribution/) 讨论过一个相关问题：allocation owner、RSS、page hotness 与 physical-page activity 回答的并不是同一个问题。CXL tier isolation 从另一个角度暴露了同一种 category error：谁分配了页面、谁被记账，并不一定覆盖所有在乎这个页面驻留位置的 workload。

## 哪些结果会改变这个判断？

如果 Linux 未来提供并正式文档化一个端到端的 cgroup memory-tier contract，并要求 allocation、reclaim、migration、shared-page 与 hotplug 路径全部遵守，而且 selftest 能证明 hard allowed-tier mask 不会被绕过，那么本文的结论会明显变弱。那时 `cpuset.mems` 或它的后继接口就真的可以被当作 residency boundary，而不是 placement 的一个输入。

如果生产测量发现，对真正需要 tier isolation 的 workload 来说，共享页和 migration exception 几乎可以忽略，也会削弱新增机制的价值。假如 task 启动前设好 cpuset，再配合当前 demotion behavior，就能在真实 pressure test 里稳定做到零有意义的 violation，那就没有必要再加一层 hardwall。

最后，很多应用根本不需要 isolation。如果真正需求只是 SLO，或者保证一定比例的 local DRAM，那么 fair-share tiering 才是正确抽象。把它强行写成“绝不能用 CXL”反而会浪费容量，并提高 OOM 风险。

因此，一个更精确的心智模型是：**NUMA 与 cpuset 控制首先约束内存可以从哪里分配，而页面在整个生命周期里的 tier residency 还取决于 reclaim、migration、sharing 与 pressure policy。要声称存在 hard CXL isolation，就必须覆盖这些路径，而不能只检查第一次 allocation。**

## 参考资料

- Linux kernel documentation, [CXL memory allocation: Reclaim](https://docs.kernel.org/driver-api/cxl/allocation/reclaim.html), accessed 2026-09-17.
- Linux kernel documentation, [Linux CXL early boot and memory tiers](https://docs.kernel.org/driver-api/cxl/linux/early-boot.html), accessed 2026-09-17.
- Linux kernel documentation, [Control Group v2](https://docs.kernel.org/admin-guide/cgroup-v2.html), accessed 2026-09-17.
- Linux kernel documentation, [Page migration](https://docs.kernel.org/mm/page_migration.html), accessed 2026-09-17.
- Kaiyang Zhao et al., [Equilibria: Fair Multi-Tenant CXL Memory Tiering At Scale](https://arxiv.org/abs/2602.08800), 2026.
- Hasan Al Maruf et al., [TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory](https://arxiv.org/abs/2206.02878), 2022/2023.
- Linux kernel source, [torvalds/linux](https://github.com/torvalds/linux), 作为文中相关内核路径的实现参考。
