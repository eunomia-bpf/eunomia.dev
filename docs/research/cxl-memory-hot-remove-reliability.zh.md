---
date: 2026-09-13
slug: cxl-memory-hot-remove-reliability
title: "Linux 能保证 CXL 内存以后还能热移除吗？"
description: "CXL 内存通过 dax_kmem 转成 System RAM 后，只有相关内存块都能安全迁出并 offline，才可能继续热移除。ZONE_MOVABLE 能提高成功率，但长期页固定、页表和巨页仍会让未来可移除性取决于运行时 ownership。"
tags:
  - Daily Report
  - Linux
  - CXL
  - Memory Hotplug
  - Memory Management
research_question: "要怎样的证据和运行时策略，才能判断今天加入 Linux 的 CXL 内存在未来仍可安全迁出、offline 并热移除，而不是最终只能重启？"
source_cutoff: 2026-09-13
status: daily-report
---

# Linux 能保证 CXL 内存以后还能热移除吗？

一块 CXL 内存设备可以在 Linux 运行期间接入，通过 DAX 暴露出来，再转换成普通 System RAM。表面上看，这很像真正的弹性内存：现在把容量加进来，让应用照常使用，以后需要重新平衡内存池、维护硬件或者更换设备时，再把它拆掉。

真正困难的是最后一步。

Linux 要物理移除一段内存，必须先让对应的 memory block 全部进入 offline 状态。这个过程不是简单改一个状态位。内核要把所有可迁移页面搬走，确认没有分配仍依赖这段物理地址，再拆除表示 CXL region 和设备的软件对象。一块刚接入时很容易使用的 CXL 内存，运行几个小时或几天后，可能已经变成实际上拆不掉的内存。

`ZONE_MOVABLE` 很大程度上就是为这个问题服务的。它把大多数不可迁移的内核分配挡在选定内存之外，让未来 memory offlining 更可能成功。当前 Linux 的 CXL 文档说得更直接：放进 `ZONE_NORMAL` 的 CXL 容量应当视为已经长期附着在 page allocator 上，而 `ZONE_MOVABLE` 的目标之一就是保留以后 hot-unplug 整个区域的可能性。

但“成功概率更高”不等于“保证以后能拆”。长期 page pin、巨页、页表和 memory map 的资源要求、混合 zone 布局、并发分配以及设备相关 mapping，都可能让迁出变慢甚至失败。旧的 sysfs `removable` 属性也不能证明当前真的能移除。Linux 文档已经明确指出，它现在只表示内核是否支持 memory offlining，而不是这个 memory block 此刻大概率能否成功 offline。

所以真正值得问的不是 Linux 支不支持 CXL hotplug，而是系统能否把**未来可移除性作为一个需要长期维持、最后还要证明的生命周期属性**。

<!-- more -->

## CXL 内存容易加进来，不代表以后容易拿出去

Linux 通过 DAX 子系统承接 CXL 内存。一个 CXL DAX region 可以继续作为 `/dev/daxN.Y` 这样的文件式设备，由用户态直接 `mmap()`；也可以通过 `dax_kmem` 转换成 memory-hotplug 管理的内存块，加入普通 page allocator。后一种模式很有吸引力，因为应用几乎不需要修改 allocator，就能直接把 CXL 容量当作 System RAM 使用。

但一旦转换成 System RAM，ownership 问题就变了。内核和应用会按照所选 memory zone 与普通分配策略，把页面放到这段容量里。以后想拆设备时，不能直接把这些页面撤销。所有仍然指向这段物理地址的依赖，都必须先消失。

Linux 通用的 memory-hotplug 路径明确体现了这一点。hot-unplug 之前先要 offline memory block；offlining 期间，内核迁走 movable page，并把空闲页从 allocator 中移除。只有 offlining 成功以后，物理内存才能继续进入 remove 流程。内核文档同时强调，如果没有 `ZONE_MOVABLE`，memory block 能否成功 offline 根本没有保证，因为普通 kernel zone 可能包含页表、`kmalloc()` 对象和其他不可迁移内核状态。

`ZONE_MOVABLE` 的作用，是缩小这段内存允许承载的 allocation class。大多数用户态匿名页和 page cache 可以迁移，因此可以从 `ZONE_MOVABLE` 分配；大多数内核对象则留在 kernel zone。这个结构很有用，但它马上带来第二个约束：机器必须保留足够的 `ZONE_NORMAL`，给内核 metadata 和不可迁移工作使用。CXL 文档专门提醒，超大的 CXL `ZONE_MOVABLE` pool 在没有使用 `memmap_on_memory` 时，还需要足够的本地 `ZONE_NORMAL` 来承载这些热插拔内存对应的 memory map。

所以一个看似简单的 zone 选择，其实是在做系统级权衡。把 CXL 内存放进 `ZONE_NORMAL`，它可以承载更广泛的分配，但以后热移除会非常不可靠。放进 `ZONE_MOVABLE`，未来拆除的可能性更高，但主机必须留足 kernel-zone 容量，同时还要避免与 movable memory 相冲突的 workload。

## `ZONE_MOVABLE` 是策略边界，不是可移除证书

Linux 的 memory-hotplug 文档对剩余风险写得很直接。即使使用了 `ZONE_MOVABLE`，memory hole、混合 zone 或 NUMA node、特殊 memory block、huge page 等情况仍可能让 offlining 失败。长期 page pin 尤其麻烦，因为只要一页无法迁移，就可能让整个 memory block 一直留在线上。

同一份文档还强调，movable 与 kernel zone 的比例取决于 workload。`ZONE_MOVABLE` 太多时，即使系统看起来还有大量空闲 RAM，也可能因为 kernel zone 不够而无法满足不可迁移分配，严重时甚至影响系统稳定性。极端的长期 page pin workload 可能根本不适合大量 `ZONE_MOVABLE`。HugeTLB 的 huge page 与 gigantic page 又有自己的迁移条件，在部分配置下，hot-remove 甚至可能长时间等待可用的迁移目标。

这就是为什么一个静态 capability bit 不够。系统运行以后，这个属性会继续变化。

可以想象两台完全相同的主机，它们都把同型号 CXL region 以 `ZONE_MOVABLE` 方式 online。主机 A 主要运行普通匿名内存和 page cache，没有设备栈长期 pin 这段内存。主机 B 则在运行过程中出现了长期 pin，又积累了一些当前无法迁移的大页状态。两台主机最开始的硬件能力和 kernel capability 完全相同，但几个小时以后，它们能不能热移除 CXL 的答案已经不同。

如果维护控制器只知道“支持 CXL hotplug”和“region 在 `ZONE_MOVABLE`”，它分辨不出这两种状态。

## CXL 还多了一层设备 teardown 边界

成功 offline memory block 只是安全移除 CXL 设备的必要条件，还不是全部条件。当前 CXL device-hotplug 文档要求在物理移除之前，仔细拆除管理该设备的 memory region、driver 等软件结构。如果 CXL.mem 仍然被这些结构使用就直接硬拔，系统很可能触发 machine check；如果访问被限制在用户态，也至少可能得到 `SIGBUS`。

CXL region topology 还会限制动态变化方式。固件需要在启动时准备足够的 CXL Fixed Memory Window；由 Host-managed Device Memory decoder 构建的 region，在设备成员变化时可能需要整体 teardown 后重新创建。因此，真正的物理移除位于一条更长的链条末端：

```text
阻止新的 owner 进入目标容量
        |
        v
迁走或释放现有 page / mapping
        |
        v
把 region 对应的所有 memory block offline
        |
        v
拆除 DAX / CXL region 与 driver 状态
        |
        v
物理移除或重新配置设备
```

任何前置阶段失败，后续阶段都应该停止。这里的失败不是简单的“这块容量还在线”。如果在仍有 stale reference 时继续物理移除，结果可能直接升级成主机故障。

用户态工具其实已经暴露出这种现实。`daxctl reconfigure-device` 在把 `system-ram` 模式转换回 `devdax` 之前，预期相关 memory section 已经 offline。`--force` 可以尝试替用户完成 offlining，但工具文档也明确警告：如果强行绕过 auto-online policy，虽然一次 reconfiguration 可能成功，之后却可能再也无法在不重启的情况下把这些内存 offline。ndctl 的一个真实 issue 还展示过 CXL 设备卡在 `system-ram` 模式的情况，因为其中一个 memory section 在转换回 `devdax` 时返回 `Device or resource busy`。

这类生产证据说明的是同一个边界：**内存能顺利加入系统，不代表以后一定能顺利撤销。**

## 为什么现有内存观测还不够

近年的 CXL 系统研究投入了很多工作去回答“内存应该放在哪里”以及“应该怎样测量内存行为”。例如 OSDI 2026 的 NEMO 在 FPGA CXL memory expander 上实现了新的硬件 telemetry engine，用更高质量的访问证据帮助内存分层和干扰判断。这类观测对于 placement 很有价值，但 hot-remove 需要的是另一类证据。

placement controller 关心哪些 page 热、哪些 page 冷，应该留在本地 DRAM 还是远端 CXL。removal controller 则需要知道：当前仍占有某段物理地址的每一个 owner，能不能在可接受时间内撤销或迁移。访问频率只是其中一个输入。一个完全不热的页面也可能不可迁移；最近没有访问的页面仍可能被 DMA 长期 pin；利用率很低的 CXL 设备也可能只因为一个 allocation，就让最后一个 memory block 无法 offline。

此前的 [GPU 内存放置报告](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/) 区分了真正的 placement evidence 与单纯 page-fault 信号。这里缺失的证据不同，它关心的是**能不能迁出，以及谁还拥有这段内存**，而不是内存热不热。

## 现有研究还缺什么

第一个缺口是**在准入时约束未来可移除性**。Linux 可以让 operator 选择 `ZONE_MOVABLE`，现有 auto-online policy 也会尝试平衡 movable 与 kernel zone，但没有一个面向应用或运维的 contract 能说：“只要这个 region 按这些 workload 限制投入使用，它就能在给定时间内重新迁空。”真正影响这个承诺的长期 pin、巨页行为、memory-map placement、本地 kernel-zone reserve 等约束分散在不同子系统里。

第二个缺口是**在移除失败时解释 blocker**。memory offlining 可能直接失败，也可能长时间不结束。运维系统需要知道阻塞原因究竟是 page pin、huge page、zone layout、DAX mapping、kernel metadata、并发 allocation，还是 CXL region topology。只有一个失败返回值，无法决定下一步应该重试、迁走 workload、重配设备，还是安排 reboot。

第三个缺口是**跨子系统的完整 teardown 证据**。memory block 变成 offline，并不能单独证明 DAX mapping、CXL region、decoder state 和 driver ownership 都已经适合物理拔除。这个边界同时跨越 memory management 和 device topology。现有机制提供了多个局部阶段，但生产自动化仍然要自己正确组合这些阶段。

第四个缺口是**按真实 blocker 类型做评测**。一个刚 online 完就立刻 offline 的 clean-region 测试，很难代表运行几天后的主机。qualification 应该主动制造 page-table pressure、长期 pin、huge page、DAX mapping、内存压力以及 offlining 期间的并发分配。否则一个平台可以在实验室里宣称支持 hot-remove，却在真实环境中经常退化成“需要 reboot 才能拆”。

## 兼具学术价值与生产价值的方向

### 1. 在 CXL region 准入时保留一份“可移除预算”

第一个方向，是把未来 hot-remove 从 `ZONE_MOVABLE` 的附带收益，提升成明确的 admission goal。

当一个 CXL region online 时，可以由一个小型 controller 给它绑定 removability policy。这个 policy 记录目标 zone、memory-map placement、本地 kernel-zone 最小 reserve、是否允许长期 page pin、large-page 约束、期望的最大 evacuation time，以及承诺失效后的 fallback。没必要让所有 allocation path 都去调用一个复杂策略引擎。更实际的做法，是只在少数高价值边界做硬约束，例如拒绝或重定向落到“必须可移除 region”里的长期 pin，并在本地 `ZONE_NORMAL` reserve 低于预算时立即把承诺标记为失效。

可实现的 artifact 可以是一套用户态 policy daemon，加上 kernel tracepoint、现有 sysfs 状态，以及只有在无法从用户态做硬 admission 时才加入的小型 kernel change。真正有价值的输出不是“movable memory enabled”，而是一份带版本的 region contract，能解释 operator 在承诺什么，以及哪些运行时事件会让这个承诺失效。

评测可以覆盖匿名内存、page cache、THP/hugetlb、长期 GUP、类似 RDMA/VFIO 的 pinning 和内存压力，并在不同 CXL pool size 与 zone ratio 下比较普通 `ZONE_MOVABLE`、现有 auto-online policy 和新的 admission controller。核心指标包括 false-safe decision、hot-remove 成功率、迁出耗时、无法收回的容量、应用 tail latency，以及 kernel-zone pressure。还应该分别移除 pin admission 和 reserve accounting 做 ablation，确认到底是哪一个机制改变了结果。

它的学术问题是：removability 能不能被建模成一种会随着 allocation decision 被消耗或保存的资源属性。生产上，它直接服务于内存池调度和硬件维护控制器，让“在线维护”成为可以约束的目标，而不是一句 reboot 兜底。

如果普通 `ZONE_MOVABLE` 加现有 allocation policy 已经能达到相同的移除成功率和 evacuation bound，而且额外的 admission rule 从未改变任何部署决定，那么这个方向没有必要。

### 2. 物理拆除前生成一份 evacuation witness

第二个方向，是为一个具体 region generation 建立分阶段的 quiescence proof。

开始移除时，controller 先阻止新的 allocation 或 mapping 进入目标 region，然后枚举并分类剩余 ownership：普通 movable page、长期 pin、huge page、DAX mapping、memory-block zone state，以及 CXL region/device dependency。迁移与 teardown 全部完成后，生成一份绑定到当前 region generation 的紧凑 witness：

```text
region_generation = G42
new_allocations = blocked
memory_blocks = all offline
remaining_pins = 0
remaining_dax_mappings = 0
cxl_region_users = 0
driver_teardown = complete
physical_remove = permitted
```

它不需要被包装成复杂的密码学机制。价值在于，自动化流程中的每一步都有明确前置条件，最终 detach 命令也不能拿上一代 region configuration 的旧证据继续执行。

原型可以直接集成 `daxctl`、`cxl` 工具与现有 memory-hotplug 接口。评测时主动制造 race，例如 evacuation 过程中启动新的 mapping、pin 和 memory pressure，在检查与 detach 之间重新配置 region，再与一个只等待 memory block offline 的简单脚本比较。最重要的 correctness metric 是错误或 stale detach approval；运维指标则包括 evacuation latency、false refusal、诊断时间，以及 witness 能否把 opaque `busy` 进一步归因到具体 blocker。

学术上，这是一个跨 subsystem 的 quiescence 问题：memory ownership 与 device topology 必须对同一个 removal cut 达成一致。生产上，它给硬件维护自动化一个可以执行的停止条件，不再把 hot-remove 变成“先试试，看看主机会不会出事”。

如果实际系统中 memory offlining 成功加现有 driver remove 已经天然构成完整、无 race 的证明，而且 witness 从未发现 stale state、隐藏 owner 或 sequencing error，那么额外机制就没有价值。

### 3. 测量“可移除性债务”，而不是只测 hotplug 成功

第三个方向，是让 benchmark 先把一台主机真正“用旧”，再尝试移除内存。

从干净的 CXL region 开始，按明确 policy online，随后运行一个专门积累某类 blocker 的 workload phase，最后再触发 evacuation。测试应该分别覆盖：

- 普通可迁移匿名页和 page cache；
- movable zone 之外的页表与其他不可迁移 kernel pressure；
- 长期 GUP 或 DMA 风格的 pin；
- transparent huge page、huge page 与 gigantic page 配置；
- direct DAX mapping 与 `system-ram` 转换；
- offlining 期间持续发生的新 allocation；
- 需要 region teardown 或 decoder topology 变化的 CXL 设备组合。

结果不要压缩成一个 pass/fail。更有用的是画出一条随运行时间变化的**removability-debt curve**：还有多少容量能迁走，剩余部分被什么阻塞，完整 evacuation 需要多久，以及什么时候唯一现实的恢复动作已经变成 reboot。大范围 fault schedule 可以在 QEMU 或 emulated CXL 上跑，时延和真实 device behavior 则放到物理 CXL 平台验证。

最有区分度的比较不是另一个 bandwidth benchmark，而是让几种在 admission 时看起来同样健康的 online policy，在经历几个小时代表性 workload 后暴露出不同 removal outcome。benchmark 还应该验证 blocker diagnosis 是否真的能预测正确的 corrective action，例如迁走 workload、释放 pin、调整 huge-page reservation、重建 region 或 reboot。

学术价值在于给 hotpluggable memory 一个生命周期 metric，而不是一次性的 capability test。生产价值是 qualification：服务器厂商与 operator 可以判断某种 CXL 配置在自己的 workload 下到底是不是“可在线维护”的。

如果现有 Linux memory-hotplug 和 ndctl test 已经覆盖相同的长期 blocker matrix，并且能以接近成本准确预测线上 removal outcome，那么新的 benchmark 不应该重复造轮子。

## 一个更稳妥的部署规则

如果以后真的需要 hot-remove，就不要把 CXL 自动 online 当作无害默认值。

先决定这段容量应该保留为 `devdax`，还是转换成 System RAM。如果要进入 System RAM，而且未来 detach 是硬要求，就明确使用与 `ZONE_MOVABLE` 一致的 policy，同时给 kernel allocation 和 memory metadata 留够 `ZONE_NORMAL`，并限制会建立长期 pin 或其他不可迁移 ownership 的 workload。qualification 也不要只在开机后测试一次，而应该让代表性 workload 跑过一段时间，再真正执行 offlining。

真正移除时，应要求 region 中的每个 memory block 都进入 offline 状态，然后完成 DAX/CXL region 与 driver teardown，最后才允许物理 detach。不要把旧的 sysfs `removable` bit 当成“当前内容一定能迁走”的证明。如果 evacuation 失败，应把 blocker 当作需要诊断的证据，而不是继续强行执行硬件操作的理由。

此前的 [GPU Checkpoint 恢复报告](https://eunomia.dev/zh/research/gpu-checkpoint-recovery-consistency/) 讨论过多个组件是否真的落在同一个可恢复 cut 上。CXL hot-remove 也有类似的生命周期结构：page ownership、memory-block state、DAX mapping、region topology 与物理设备都必须对“这一代资源已经结束”达成一致，设备才能消失。

## 哪些结果会改变这个判断？

如果未来 Linux 提供一种硬约束的 hot-removable allocation domain，只允许进入其中的状态都能保证迁移，并给出有界 evacuation time 和完整 blocker attribution，同时 CXL stack 又提供从 page allocator 到物理设备的一体化 teardown 边界，那么上面的判断会明显减弱。operator 可以直接依赖内核 contract，而不再需要单独维护 admission 与 evacuation protocol。

另一种情况是部署本身从来不要求 live removal。如果机房接受每次重配或维护 CXL 容量都 reboot，那么为了保留 hot-remove 而牺牲 kernel-zone headroom、限制 pinning 并引入额外 policy，可能得不偿失。

最后，实验也可能证明这些新机制没有必要。如果长期测试覆盖 pinning、huge page、DAX、内存压力和 region topology 后，发现现有 `ZONE_MOVABLE`、memory offlining 与 CXL teardown 已经能稳定预测 removal 结果，而且诊断信息足够明确，那么新的 removability contract 或 witness 只会增加流程复杂度。

因此更准确的心智模型是：**CXL hotplug support 定义内存怎样进入和离开机器，但未来能不能真的移除，取决于这段内存在服役期间积累了什么 ownership。这个属性需要在运行过程中持续保存，并在最终拆除前重新验证。**

## 参考资料

- Linux kernel documentation, [CXL Memory Hotplug](https://docs.kernel.org/next/driver-api/cxl/linux/memory-hotplug.html)，访问于 2026-09-13。
- Linux kernel documentation, [Memory Hot(Un)Plug](https://docs.kernel.org/admin-guide/mm/memory-hotplug.html)，访问于 2026-09-13。
- Linux kernel documentation, [CXL Device Hotplug](https://docs.kernel.org/driver-api/cxl/platform/device-hotplug.html)，访问于 2026-09-13。
- Linux kernel documentation, [CXL DAX Driver Operation](https://docs.kernel.org/next/driver-api/cxl/linux/dax-driver.html)，访问于 2026-09-13。
- pmem/ndctl, [`daxctl reconfigure-device`](https://github.com/pmem/ndctl/blob/main/Documentation/daxctl/daxctl-reconfigure-device.txt)，访问于 2026-09-13。
- pmem/ndctl issue #256, [CXL device cannot be changed from system-ram mode to devdax mode](https://github.com/pmem/ndctl/issues/256)，创建于 2023-10-09，访问于 2026-09-13。
- Shihang Li 等，[Finding NEMO: Nimble and Expressive Memory Observability](https://www.usenix.org/conference/osdi26/presentation/li-shihang)，OSDI 2026。
