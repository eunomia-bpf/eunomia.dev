# 为什么一个 profile 样本能链接到从未被采样的 trace span，而这个链接本身又需要先打开哪些开关？

因为链接是挂在样本上的一对标识符，而不是"trace 被导出了"的结果。eBPF profiler 在中断线程的那一刻，
读取这个线程当时公布的 trace id 与 span id，把它们写进该样本的链接条目，全程不问这个 span 是否被采样。
它也没法问：它读的两个来源都没有可用的 trace-flags 字节。所以链接的含义只是"采样这一刻，这个线程
正处在这个 span 里"，一份 profile 完全可能链接到一个后端从未收到的 span。

而且默认情况下链接根本不存在。两个互相独立的开关必须同时打开，一个在插桩侧，一个在 profiler 侧；
此外还有第三条路径，只对用 SDK 公布线程本地记录的进程才有效。只跑 profiler，每个样本看上去都像是
在 span 之外执行的；只跑插桩，则连可链接的 profile 都没有。

## 机制

1. **线格式里留了位置，也用零表示"没有"。** Profiles 的 `Sample` 带一个 `link_index`，指向共享字典
   的链接表，而索引 0 是字典里必须存在的零值条目，所以这个字段的默认值就等于"不存在链接"。链接本身
   恰好是两个标识符：16 字节 trace id 和 8 字节 span id。里面没有 flags 字段，没有父 span，也没有
   任何采样决定。

2. **第一个来源：插桩侧写入的一张 pinned map。** OpenTelemetry eBPF Instrumentation 把
   `traces_ctx_v1` 钉在 bpffs 上，这是一张以打包的进程号与线程号为键、以 trace id 和 span id 两个
   数组为值的 LRU 哈希表。它是个通用接口：同一张表也供插桩自己的日志富化器使用。它的头文件明确
   声明布局对外部读者稳定。

3. **只有在有人读的时候，插桩才往里写。** 让这张表与被插桩运行时当前活跃的请求保持一致，代价是
   该运行时每次异步上下文切换都要刷新一次，因此写入方被一个 BPF 常量统一控制。启用日志富化器，
   或者为外部读者打开 `populate_trace_context`，表才会被填。没有读者时常量保持关闭，那些写操作被
   整段消掉，钉住的表是空的。不填充的那次运行还会在启动时把这张表清空一遍，所以上一次运行留下的
   条目不会存活。

4. **profiler 自己也得点头。** `obi_process_ctx` 关着时，profiler 把这张表的大小设成一个条目并且
   不去钉它，正是为了不为一张共享表白付代价。打开之后，profiler 先尝试打开已有的 pin，只有在插桩
   侧没建的时候才自己创建并钉住。两侧文档写的是同一条规则：谁先启动，谁建 pin。

5. **第二个来源：SDK 公布的线程本地记录。** profiler 还会读一个普通的 ELF 线程本地指针，它由 SDK
   导出在动态符号表里，指向一条记录，记录中放着 trace id、span id、有效标志字节、trace-flags 字节
   和一小段属性载荷。请求上下文挂到线程上时记录被写入，解绑时被清空或置为无效。这条路径既不需要
   共享 map，也不需要插桩进程。

6. **线程本地记录优先。** 在栈回溯结束时，profiler 先把 OBI map 里的 id 填上，再读线程上下文；
   一旦发现有效记录且至少有一个 id 非零，就用记录里的两个 id 替换掉前面的。记录里两个 id 都是零，
   表示这个线程没有活跃 span，此时保留原先的 id，而不是清掉。

7. **进入样本的东西。** 只有两个 id 都非零，profiler 才写一条链接条目，并把相同的 id 对去重成同一个
   字典项，然后让样本指向它。样本的身份由栈、属性集合和链接一起决定，所以同一段栈在不同 span 下
   是两个样本，而不是被聚合成一行。

## 链接能证明与不能证明什么

- **span id 与采样决定无关。** trace SDK 在采样决定之前、并且独立于它生成 span id，正是为了让其他
  组件在 span 不被记录的情况下也能依赖一个唯一 id。线程本地记录在上下文挂载时写入，这和"span 被
  采样"不是同一个事件。所以链接可以指向一个被丢弃的 span，而这份 profile 并没有错。

- **trace-flags 字节是有的，却被丢掉了。** 线程本地记录的布局在有效标志字节和载荷长度之间定义了一个
  trace-flags 字节，并且刻意对齐，好让长度字段落在两字节边界上。profiler 为同一条记录写的 C 结构
  把同一个偏移命名为保留字节，整个 profiler 里没有任何代码读 trace flags。唯一能承载采样位的位置
  结构上存在，功能上闲置。

- **链接是某一刻的观测，不是请求归属。** 两个来源都以被中断的那个线程为键。服务端 span 结束时 map
  条目被删除，客户端 span 结束时回退到父 span，线程本地指针在解绑时清空。在 span 结束之后落到样本
  上，或者落在一个已经服务过别的请求的热线程上，链接到的是线程当时持有的东西。

- **span 基数上升会削弱聚合。** 因为链接是样本身份的一部分，一个热点栈在每个请求不同的 span 下就
  变成每请求一个样本，而不是聚合成一个。这正是"哪个代码在这条 trace 里烧了 CPU"要付的价钱，也是
  带链接的 profile 更大、对全局 top-N 分析更粗的原因。

## 验证方法

1. **先查插桩侧。** 确认 `populate_trace_context` 已设（或日志富化器非空），然后再看 pin：表是空的
   而填充开关是开的，说明问题在插桩侧的协议覆盖或运行时支持，不在 profiler。

   ```console
   # bpftool map show pinned /sys/fs/bpf/otel/traces_ctx_v1
   ```

   pin 不存在而填充开关是开的，意味着插桩退回了内部 map，这发生在 bpffs 目录无法创建或访问时；
   它的日志会在启动时说一次，并关掉所有 map 的 pinning。

2. **再查 profiler 侧。** 确认 `obi_process_ctx` 是开的，否则 profiler 根本不读那张共享表，即使 pin
   里塞满了条目，每个样本也都是无链接的。

3. **然后查进程自己的公布。** 线程上下文加载器要在进程或其库中解析出那个 TLS 符号，符号和进程公布的
   schema 都齐了才会生效。三个计数器把几种情况分开：TLS 指针读不到的样本、记录读不到的样本、读成功
   的样本。一个什么都不公布的进程，这三个都不会增加。

   ```console
   # profiler 的计数器，在开启指标端点时可见
   # bpf.threadctx.read_successes
   # bpf.threadctx.errors.read_tls_ptr
   # bpf.threadctx.errors.read_thread_ctx_buf
   ```

4. **把属性载荷与 id 分开看。** 载荷超过 640 字节上限的记录会被截断并计数，而样本仍然带着 id。载荷被
   截断意味着标签变短了，不是链接失败。

5. **区分两种失败形态。** 完全没有链接，是某个开关没开，或者进程什么都不公布。链接到了错误的 span，
   是线程在你以为的请求和采样时刻之间切换过；拿链接到的 span 去对照采样那一刻线程正在服务的请求，
   而不是你正在排查的那个请求。

## 边界

- **profiles 信号仍是 alpha。** 格式、字典布局和线程上下文 schema 版本都标记为开发中，profiler 遇到
  不认识的 schema 版本会计数并拒绝，而不是猜着解析。自己写发布方时要把记录布局当成会变的东西。

- **pinned map 是快路径，且有硬上限。** 它是加载时预分配的 LRU 哈希，写不写都占满整块内存，繁忙线程
  的条目在压力下会被淘汰。要及时读取，或者接受被淘汰。

- **只有一份有限的交接清单能进这张表。** 插桩通过若干固定运行时的机制、加上与语言无关的线程与套接字
  信号，把出站请求与入站请求关联起来。搬到它看不见的队列上的工作没有父 span，链接于是落在错误的
  span 上，或者根本没有。

- **这里没有任何东西能证明采样位。** 用链接去找某个请求期间跑过的代码；不要把它读成"对应的 span 在
  trace 后端里存在"的证据。

## 参考

- [Profiles 数据格式规范](https://opentelemetry.io/docs/specs/otel/profiles/data-format/) — 字典、按引用组织的布局，以及样本与链接的关系。
- [Profiles 的 protobuf 定义](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/profiles/v1development/profiles.proto) — `Sample.link_index` 用零表示无链接、只有两个字段的 `Link`，以及把链接算进去的样本身份。
- [OTEP 4947：线程级上下文共享](https://github.com/open-telemetry/opentelemetry-specification/blob/main/oteps/profiles/4947-thread-ctx.md) — 包含 trace-flags 字节的线程本地记录布局、公布与读取协议，以及 640 字节的建议上限。
- [OTEP 4719：进程级上下文共享](https://github.com/open-telemetry/opentelemetry-specification/blob/main/oteps/profiles/4719-process-ctx.md) — 线程本地 schema 键所在的位置，以及进程级机制自己声明的 trace 关联边界。
- [插桩文档：profiles 与 traces 的关联](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/trace-profile-correlation.md) — 共享 map 的规格、键与值，以及 `populate_trace_context` 开关。
- [插桩文档：trace 上下文与填充时机](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/trace-log-correlation.md) — 代价模型、按需读取的门控，以及关闭填充时只清空一次的行为。
- [插桩 `pkg/config/ebpf_tracer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/config/ebpf_tracer.go) — `populate_trace_context` 写明的契约。
- [插桩 `pkg/ebpf/tracer_linux.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/ebpf/tracer_linux.go) — bpffs 下的 otel 目录、退回内部 map 的路径，以及"外部 profile 关联会失效"的告警。
- [profiler `tracer/tracer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/tracer/tracer.go) — 共享 map 的查找、开关关闭时退化成一个条目，以及自己创建并钉住的路径。
- [profiler interpreter dispatcher](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/support/ebpf/interpreter_dispatcher.ebpf.c) — 该文件里的两个 id 来源、先后顺序，以及线程上下文的覆盖规则。
- [profiler `support/ebpf/types.h`](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/support/ebpf/types.h) — 带保留字节的线上记录结构，以及 640 字节的载荷上限。
- [profiler 样本上报器](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/reporter/internal/pdata/generate.go) — pdata 生成器里的非零 id 判断，以及样本指向之前对链接的去重。
- [profiler 线程上下文加载器](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/interpreter/threadcontext/threadcontext.go) — 导出的 TLS 符号查找与等待 DTV 的路径。
- [trace SDK 规范](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/sdk.md) — span id 独立于采样决定生成，以及记录与采样两个标志的对应表。

## 今天的社区讨论

本次运行查看了两个选择加入私有只读归档的聊天工作区，共三个允许名单通道，并在可见浏览器里读了公开的
bpf 邮件列表。归档在严格意义上也是空的：24 小时窗口里没有消息，7 天窗口里共 8 条，集中在其中一个
工作区的两个通道，另一个工作区为零。另外四个允许名单内的聊天工作区仍然只能在可见浏览器里看，这次
读不到：两个 Discord 邀请和两个 Slack 邀请打开后都是加入或接受页面，没有通道列表，所以记为无法
访问，而不是"没有讨论"。一个实践者论坛对同一次抓取返回了网络策略拦截页。邮件列表可读，当日窗口里
有 32 个新线程的根消息，技术内容如下；上面这个问题来自归档 7 天窗口的末端——两个项目在比较网络
trace 与持续 profile——而不是来自某条直接提出它的消息。

### 那 8 条消息到底是什么

八条里有七条是同一个主题：一个项目自己通报，某会议上有若干演讲谈到了它的 eBPF 网络可观测性 agent，
其中包括一个第三方用法，把网络流与持续 profile 结合起来做威胁检测，并有两位维护者讲了如何把该 agent
嵌入其他产品。这个组合才是重点，而且恰好就是上面回答的那条边界：网络流数据说明哪个连接做了什么，
profile 说明哪段代码烧了 CPU，把它们接起来靠的是每个样本上的 span id，而不是每台主机一个聚合值。
第八条是一位 GenAI 插桩贡献者自述工作：让一份 JavaScript 插桩对齐 GenAI 语义约定，并为他每个 agent 的
token 用量采用新的 agent 接口。两条都不是提问，所以上面的回答是从两个项目写在文档里的契约重建的，
而不是从消息里摘的。

### 不需要每个驱动再养一个分配器的零拷贝 AF_XDP

当日窗口里最完整的一个系列，把 UMEM 变成 page-pool 的内存提供者，而不是要求每个驱动再长出一套接收
分配器。机制值得跟，因为它是转移责任而不是增加路径：核心在 bind 时通过队列接口装上提供者，驱动只
声明自己支持，并校验 chunk 大小、headroom 与多缓冲模式；一个 refill 契约取代了每个驱动的 NAPI 唤醒，
驱动报告每次 refill 的结束，核心公布该队列的 NAPI 供提供者与套接字使用。协议栈会被告知它是否可以把
提供者的缓冲区留下；不能留时，在 pass 路径、帧转换和非 XSKMAP 重定向上都复制，而因为帧复制是线性的，
发往 devmap 与 cpumap 的多缓冲帧被丢弃而不是处理。单队列上的数字是：接收 2120 万包每秒，发送 2890 万
包每秒，30 条指令的 `XDP_DROP` 程序带来 1.9% 的丢包代价。信里也写明两个未尽之处：只支持对齐的
4 KiB chunk，以及通用 XDP 加 cpumap 仍无法喂给提供者套接字。

### 把宽限期数出来，而不是假设它

一个针对多程序挂载路径的两补丁改动，产生了当日窗口里最好的一段评审算术。提案把 RCU 同步推迟到 peer
被复用时；评审回复接受了想法，接着纠正了位置，因为等待被挂在每次改动之后都会跑的提交上，而不是挂在
真正需要它的那一次交换上。回复自己的表格就是论证：对于在前一次的一个宽限期之内到来的第二次改动，
先 attach 再 attach 省掉一次等待，而 detach 接 detach、replace 接 detach、attach 接 detach 各多一次。
修正是只在 attach 的交换处启动宽限期，detach 与 replace 则不动旧状态，因为 peer 查找已经为旧状态
等过了。评审那句概括才是耐用的部分：等待保护的是条目的上一代，所以对任何这类系列该问的是"它等的是
哪一代"。没有做启动测试，只有 x86-64 上的干净构建。

### sockmap 迭代器规则，已经合入

本周早些时候在这里出现过的限制——只有 sockmap 迭代器才能从迭代器上下文更新 sockmap——在第三版之后
已经进入 BPF 树。最终版本在三处超出了标题，而这恰好暴露了底层缺陷：交给迭代器程序的套接字指针现在
被标记为可信，这是让更新能被检查的前提；销毁辅助函数的调用方被收窄到 TCP 与 UDP 迭代器；而那个有意
从 UNIX 套接字迭代器里更新 sockmap 以触发死锁的自测，其更新被删掉了。对任何写会改 map 的迭代器的人，
教训是：这项许可是迭代器程序的性质，不是程序目标 map 的性质。

### 过去被静默忽略的错误

一个十三补丁的文件系统系列教 dentry 解析辅助函数的调用方处理失败，其中有两处 BPF 触点：真实 inode
解析不出来时，uprobe 在创建时就失败；一个 BPF kfunc 返回 NULL，而不是返回一个它从未拿到的 inode。
错误的原因是 overlayfs 下层上的惰性查找，过去只被丢掉并打一行告警。这件事属于 BPF 讨论，是因为
uprobe 这条路径：挂到解析不出真实 inode 的探针上等于什么都没挂，永远不会触发，而这与"函数只是没被
调用"无法区分。系列删掉了那个能忽略错误的布尔辅助函数，于是编译器强制新契约。同一窗口里另一个两
补丁的 libbpf 修复形状相同、规模更小：一个未对齐的枚举值经零宽度的位域辅助函数读取，把 64 位值
左移了 64 位，于是转储出来的值带着错误的符号。

### 调度器能把安全遥测饿死

内核列表之外的一份 RFC 提了一个对 BPF 用户有实在答案的问题：带 `CAP_SYS_ADMIN` 加载的调度策略，
可以把安全守护进程路由到低优先级调度队列上，用微秒级时间片加概率性跳过，让它们始终不触发可运行任务
看门狗，同时其环形缓冲消费者落在内核事件产生之后。报告的测量声称，每种传感器 150 次试验里，静默告警
丢弃最高可达全部丢光，而无需摘掉任何探针。提议的缓解是在 map 创建处挂一个 BPF LSM 钩子，用允许名单
校验 struct-ops 注册，外加一对用于调度器加载与分发的 tracepoint。采用之前值得核对两个前提：时间片是
策略自己的选择，所以强制面是策略允许名单而不是调度器核心；作者同时给出的用户态办法——把环形缓冲消费者
线程放进实时调度类——取决于守护进程的部署方式，而不是内核。

### 两个更小的形状

一个修复重定向之后数据丢失的 sockmap 补丁，收到了一份把"新引入"与"既有"缺陷分开的自动评审：一处由本次
改动新增的高危泄漏、一处中危的陈旧 cork 大小，以及一处高危的既有问题，补丁只是让它更容易被触发。回答
这类评审时，这个区分很关键，因为只有新引入的缺陷才是作者要在同一系列里修的。最后，一个 kallsyms 系列
给内核符号表加了按名字与按地址查询的 ioctl，这类接口让跟踪工具能在不解析文本文件、也不持有可能卸载的
模块引用的情况下解析一个内核地址。