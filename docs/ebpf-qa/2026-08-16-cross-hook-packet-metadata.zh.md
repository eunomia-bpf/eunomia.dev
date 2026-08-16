# eBPF 程序应如何在多个网络 hook 之间携带每包元数据？

**简短回答：**目前上游内核没有一个能让 `skb` 穿过所有网络 hook 时始终携带数据的通用 BPF 暂存区。生产者是 XDP、消费者是 TC ingress 时，应使用 `data_meta`；数据实际属于 flow 或 socket 时，应使用稳定标识作为 key 的 map；`skb->cb` 只能视为由当前网络层短期拥有的暂存区，不能当作跨协议栈 ABI。正在评审的 BPF `skb` extension 提案可以补上真正与数据包生命周期绑定的存储，但它尚未成为可依赖的生产接口。

因此，第一个设计问题不是调用哪个 helper，而是元数据需要存活多久、由哪个对象拥有。

## 按生命周期选择存储，而不是按便利性选择

| 需要覆盖的范围 | 当前可用的上游方案 | 关键边界 |
| --- | --- | --- |
| 单次程序调用 | 寄存器与 BPF stack | 调用返回后即消失 |
| 同一路径上的 XDP 到 TC ingress | 用 `bpf_xdp_adjust_meta()` 预留字节，再通过 `data_meta` 读取 | 不是供后续 socket、LSM、tracing 或 netfilter hook 使用的通用设施 |
| 同一 flow 或 socket 的多个数据包 | Hash、LRU、socket storage 或其他以稳定标识为 key 的 map | 状态不再自动绑定到某一个数据包的分配和释放 |
| 受控且相邻的 `skb` 处理阶段 | 明确划分所有权的 control buffer 区域 | `skb->cb` 会被内核不同层复用，不是持久的跨栈契约 |
| 任意基于 `skb` 的 hook | 当前没有通用上游 BPF 接口 | 需要显式关联设计，或依赖尚未上游化的新内核能力 |

XDP 到 TC 的交接路径范围很窄，但已被正式文档化。XDP 程序通过 `bpf_xdp_adjust_meta()` 向前扩展 `data_meta` 区域，在包数据之前写入固定的应用结构；TC 程序先检查 `data_meta` 与 `data` 的边界，再读取内容。内核文档明确说明 `XDP_PASS` 后 TC-BPF 可以访问这一区域，同时也列出了 redirect、AF_XDP descriptor 和驱动元数据的限制。不要把这份契约推广成所有后续 `skb` hook 都能访问的能力。

Map 解决的是另一类问题。Map 可以在 BPF 程序与用户态之间共享状态，但必须定义标识与清理策略。数据属于 flow 时，可以考虑 socket cookie、带 generation 的连接 tuple，或应用生成的 correlation ID。这样的 key 并不会让 value 自动变成“每包状态”：重传、分片、NAT、隧道转换和并发数据包都会改变“同一条 flow”的实际含义。

用 `skb` 地址作为 map key 尤其脆弱。Clone、copy、分段以及对象地址复用都可能破坏预期标识；地址重新使用前，还必须确保每一条释放路径都完成清理。正因如此，依赖 `consume_skb` 或 `kfree_skb` tracing 清理的设计需要单独分析丢事件和竞态。它适合验证生命周期假设的受控实验，不应被包装成可移植 ABI。

内核自己的 `sk_buff` control buffer 也不是通用答案。源码明确说明，各网络层可以在这里放置私有数据，并由当前持有队列的层拥有。一个受控阶段写入、下一个相邻阶段读取是可行的；让数据穿过无关网络层则可能被覆盖或发生所有权冲突。BPF 可见的 `cb` 字段必须有局部、显式且足够短的所有权契约。

## 新的 BPF `skb` extension 提案改变了什么

当前提案为 BPF 元数据增加专用 `skb` extension。第一份实现补丁描述了一个编译期确定大小的 buffer，并通过新的 `bpf_dynptr_from_skb_ext()` kfunc 暴露。请求创建的调用者可以获得存储和可写 dynptr；如果 extension 已被多个 clone 共享，创建操作会执行 copy-on-write。只读调用在 extension 存在时返回只读 dynptr，否则返回 `-ENOENT`。

这个模型修复了地址 map 的三类弱点：

- 存储直接附着在数据包对象上，而不是依靠外部 key 重建关系；
- 内核会在释放 `skb` 时一并释放它；
- clone 可以先共享数据，写入者需要修改时再获得独立副本。

该系列还计划让多种基于 `skb` 的程序类型访问这块区域，并为 clone、虚拟以太网、隧道、cgroup、socket、LWT、netfilter、tracing、LSM 和 stream verdict 路径增加测试。后续版本还明确选择让 BPF extension 穿过 packet scrubbing，而不是在隧道或虚拟设备转换时静默删除。

这些仍是提案语义，不是上游保证。巡检时，该系列仍在邮件列表接受反馈，当前上游 UAPI header 也没有暴露 `bpf_dynptr_from_skb_ext()` 或对应的创建 flag。生产软件必须检测具体 kfunc，并验证最小 BPF object 能否加载；只判断 kernel version 不够，因为发行版回移植与实验内核都会破坏版本号推断。

## 让元数据记录本身可移植

无论承载方式是 `data_meta`、map 还是未来的 `skb` extension，记录都应自描述且有界：

```c
struct packet_meta_v1 {
    __u16 version;
    __u16 length;
    __u32 flags;
    __u64 correlation_id;
};
```

生产者应初始化每一个字节，把 `length` 设置为实际写入长度；如果存在发布顺序要求，应最后写入 `version`。消费者必须拒绝未知版本、短于必要前缀的长度，以及支持范围外的 flag。不要在记录中放指针、凭据、完整包载荷副本或可变的用户态地址。

字段含义应尽量独立于 hook。Correlation ID 或分类结果可以在更换 attachment point 后继续成立；裸指针和 parser offset 通常不行。如果多个程序都能写，应划分字段所有权，或者只允许一个 producer、其余全部只读。即使底层存储没有数据竞争，隐藏的 last-writer-wins 也很难排查。

## 验证真实的数据包路径

测试矩阵既要证明元数据保留，也要证明丢失边界：

1. 在 XDP 写入，令数据包通过，并在 TC ingress 读取。
2. 穿过生产环境实际使用的设备和 redirect map，确认哪些自定义元数据保留、哪些硬件元数据不可用。
3. 分别触发 clone 与 copy，再修改一个副本，确认是否具备 copy-on-write 隔离。
4. 单独测试封装、解封装、namespace 穿越、GRO/GSO、分段、重传与错误丢包，不能用一条路径推断另一条。
5. 确认没有元数据的数据包产生显式 miss，而不是看似有效的全零记录。
6. 反复执行 attach、detach、queue reset 与故障注入，并核对分配和释放是否平衡。
7. 按真实的元数据携带比例做基准测试。稀疏分配可能很便宜，几乎每个包都分配则可能完全不同。

在实验性 `skb` extension 内核上，应增加加载期 capability probe，并把结果记录到遥测中。如果 kfunc 不存在，就 fail closed，或切换到文档中明确标注生命周期更弱的 map 模式。不能静默替换实现，却继续报告相同保证。

长期有效的规则很简单：使用生命周期刚好匹配数据的最窄存储。`data_meta` 是 XDP 交接机制，map 是关联状态，`skb->cb` 是局部暂存区。BPF `skb` extension 有机会成为缺失的跨 hook 每包原语，但要等 ABI 与生命周期语义真正进入上游后才能据此承诺兼容性。

## 参考资料

- [Linux 内核文档：XDP RX 元数据与 XDP 到 TC 的 `data_meta` 路径](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- [Linux 内核源码：`sk_buff` 布局、control buffer 所有权、clone 与 extension](https://github.com/torvalds/linux/blob/master/include/linux/skbuff.h)
- [Linux 内核 UAPI header：当前 BPF context 与接口](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux 内核文档：BPF map](https://docs.kernel.org/bpf/maps.html)
- [Linux 内核文档：BPF kfunc 与 dynptr annotation](https://docs.kernel.org/bpf/kfuncs.html)
- [BPF 邮件列表提案：用于每包 BPF 元数据的 `skb` extension](https://lore.kernel.org/bpf/20260814-bpf-meta-inside-skb-ext-v1-0-767edd862656%40cloudflare.com/T/#t)
- [提案实现：`bpf_dynptr_from_skb_ext()` 与生命周期语义](https://lore.kernel.org/bpf/20260814-bpf-meta-inside-skb-ext-v1-1-767edd862656%40cloudflare.com/T/#t)
- [BPF 邮件列表：结构化 verifier 诊断的后续修正](https://lore.kernel.org/bpf/20260816015746.2632990-1-memxor%40gmail.com/T/#t)
- [BPF 邮件列表：拒绝由不兼容指针类型到达的同一条原子指令](https://lore.kernel.org/bpf/20260816-bpf-next-038-mixed-atomic-v1-v2-0-4644c1886dbc%40mails.tsinghua.edu.cn/T/#t)
- [BPF 邮件列表：网络驱动中的 XDP 与 AF_XDP 生命周期修复](https://lore.kernel.org/bpf/CALuQH%2BWN%2BxftDONAk%3DT8zeB1qF5y%3DyaJ4F23DAg4W6puP%2BjToA%40mail.gmail.com/T/#t)

## 今日社区讨论

今天通过普通可见浏览器检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面，所有目标均可访问。公开论坛使用其普通可见旧版界面完成检查。入选问题来自过去 24 小时，没有使用七天回退。以下内容已删除姓名、账号、雇主、频道身份、消息链接、精确时间、私有拓扑、原始日志与可回搜措辞；没有保留原始 transcript。

### 每包元数据需要明确的所有者与生命周期

当天最强讨论关注：位于不同网络 hook 的 BPF 程序，如何共享同一个数据包的信息，又不维护外部的地址 key cache。提案答案是通过 dynptr 访问 BPF 专用 `skb` extension。它的价值不只是多出一些字节，而是让分配、copy-on-write、数据包释放和只读查询都跟随拥有数据的内核对象。

尚未解决的是兼容性边界。提案仍在评审，自测试也仍在接收正确性反馈，当前上游内核没有新 kfunc。今天就需要这一能力的实践者，应先判断数据究竟属于 packet、flow 还是 socket。XDP 到 TC 的元数据和稳定 key map 能覆盖很多部署，却都不能描述成通用跨 hook 每包存储。实验内核还必须覆盖 clone、scrub、tunnel、namespace、分段与释放路径，不能只凭一次 loopback 成功就下结论。

### Verifier 诊断必须保留拒绝原因，而不只是拒绝结果

另一个活跃系列修正了结构化 verifier 输出把失败归因到错误机制的情况。读取 verifier 管理的 stack state 可能被误报成未初始化内存；variable-offset atomic stack access 也可能被描述成 helper 调用。这样会把排障带向错误方向：初始化字节或调整 capability，都无法修复直接读取 opaque dynptr 或 iterator 表示的问题。

修正方案先按 stack slot 类型分类，只对真正无效的字节保留未初始化内存建议，并把 atomic access 明确标成原子操作。相关安全修复还为每一条 atomic read-modify-write 路径记录 destination pointer type。此前 arena pointer 与普通 stack pointer 可以在同一条指令汇合，而验证后的 fixup 只根据其中一条路径选择 encoding。诊断路径与安全路径因此有同一项要求：在控制流状态合并的准确指令处保留类型和来源。

排障时应保留第一条结构化拒绝、instruction index、program type 和目标内核 BTF，并在实际运行内核上复现，因为 verifier 接受行为与诊断可能一起变化。尚不确定的是这些增强多久会进入 stable 与发行版内核；工具仍需兼容传统 verifier log，但不得据此虚构根因。

### XDP 正确性还包括 budget、所有权与 teardown 顺序

当天的驱动修复说明，dataplane 通过单包功能测试后，仍可能在持续负载或重配置时失败。如果被 XDP 消耗的数据包不计入 NAPI budget，poll loop 会运行过久、占住 CPU，并延迟 AF_XDP transmit。如果 error descriptor 或 ring shutdown 没有释放 XSK buffer，反复切换 queue 会泄漏内存。停止 queue 或 DMA unmap 的顺序错误，则可能造成 deadlock 或暴露失效的 pool pointer。

因此验证不能只看吞吐：应使用以非 `XDP_PASS` action 为主的负载观察 NAPI reschedule，反复 bind/unbind XSK pool，在流量中 reset queue，注入 RX error，并确认 DMA unmap 只发生在 queue 完全停止后。每一条失败路径前后都要核对 buffer accounting。这些修复属于具体驱动，但工程规则可以复用：每个 descriptor 只能有一次 budget 决策、一个 owner 与一条最终释放路径。

其余聊天和项目专用页面在当日窗口内较安静，或只有自动化构建通知而没有新的实践者问题。公开论坛最新的未解决技术帖子落在 24 小时窗口之外，因此没有用它替换更强的当日上游讨论。
