# 构建进程行为重构工具时，架构上需要把哪些 eBPF/内核限制考虑进去？

**简短回答：** 有两个，而且都是架构层面的，不是调参层面的：

1. **事件投递是有序的，但不保证完整。** 内核的 BPF 输出通道——每 CPU 的 perf event 数组和共享 BPF ring buffer——在缓冲区写满时会*静默丢弃*事件。perf buffer 至少会尝试写一条 `PERF_RECORD_LOST` 记录，而且只在"尽可能时"（when possible）才写；ring buffer 则根本不写任何丢失记录。因此，重构出的时间线只是进程*实际行为*的下界（lower bound），不是完整记录——除非工具自己测量丢失。
2. **`pid` 不是稳定的身份标识。** PID 按命名空间分配（新的 PID 命名空间从 1 开始），只在所属命名空间内唯一，而且是循环分配的——任务退出后 PID 会被复用。只以 pid 为键的关联引擎会悄悄把两个无关进程合并，或把事件归到错误的进程名下。

其他限制（挂载可行性、tracepoint 覆盖、跨内核版本的 CO-RE 漂移）决定你*能捕获哪些*事件；上面这两条决定你捕获到的东西有多完整、多正确。

## 为什么内核无法承诺完整的事件流

进程行为重构工具本质上是一个事件消费者：把 BPF 程序挂载到关心的 hook（clone/fork/exit 家族、`execve`、`openat`、`socket`/`connect` 等）上，每个 hook 把记录写入两条内核输出通道之一：

- **每 CPU 的 perf event 数组**（`bpf_perf_event_output()`）：每个 CPU 一个通过 `perf_event_open(2)` 建立的 mmap 环形缓冲，用户态各自消费。
- **共享 BPF ring buffer**（`bpf_ringbuf_output()` / `bpf_ringbuf_reserve()`，map 类型 `BPF_MAP_TYPE_RINGBUF`）：所有 CPU 共享一条 ring。内核的 ring buffer 设计文档正是因为 perf buffer 在两点上"不满足需求"而写的：跨 CPU 的内存利用效率，以及"保持按时间顺序发生的事件的顺序，即使它们发生在多个 CPU 上（例如一个任务的 fork/exec/exit 事件）"——这个括号里的例子恰好就是进程行为重构工具的工作负载。

两条通道都不给你完整性。perf buffer 一侧，`perf_event_open(2)` 说得很明确：消费者跟不上时，内核直接丢弃样本，这些样本"被视为丢失（considered lost），并尽可能（when possible）生成一条 PERF_RECORD_LOST 样本"。注意这个限定词：连丢失通知本身都只是尽力而为。丢失数量在受支持时可以读出来（`PERF_FORMAT_LOST`，Linux 6.0 起），libbpf 的 `perf_buffer__new()` 接受一个"在发生记录丢失时被调用"的 `lost_cb`——这个回调是你唯一的信号，而它只在内核成功写入该记录时才会触发。

ring buffer 更简洁，但给的信息更少：它的设计文档用一行写清了溢出规则——"如果 ring buffer 没有剩余空间，reservation 失败，不阻塞（no blocking）"。ring 满了意味着 `bpf_ringbuf_reserve()` 返回 `NULL`、`bpf_ringbuf_output()` 返回 0，事件就此消失：没有丢失记录，没有通知，流里也没有任何标记空白的东西。`bpf_ringbuf_query()` 系列 helper（`BPF_RB_AVAIL_DATA` 等）在文档里被定义为"瞬时快照（momentarily snapshots）"，用途是"调试/报告（debugging/reporting）"——是启发式，不是丢失计数器。（近期的内核工作增加了可选项的覆写模式：新事件替换最旧的事件；这是把静默丢弃换成了静默*覆写*——单看流本身仍然无法察觉。）

架构上的结论：**完整性是你必须自己制造出来的性质。** 按峰值（而非平均）事件速率给缓冲定容；在 BPF 程序里对每个尝试发出的事件累加一个计数器；在退出或定周期时把生产端计数与消费端计数对比；两者不一致时，把该时间窗标记为"可能不完整"，而不是当作事实呈现。没有这一步，工具会把一个缺口报告成一次"没有发生"。

## 用启动时间而不是 pid 来标识任务

关联引擎是第二个静默失效点，同样有一手资料佐证：

- `pid_namespaces(7)`："新 PID 命名空间中的 PID 从 1 开始……fork(2)、vfork(2)、clone(2) 产生的进程，其 PID 在命名空间内唯一（unique within the namespace）。"唯一性是命名空间内的性质。
- 内核的 PID 分配器（`kernel/pid.c`）从命名空间的区间内循环分配：到达区间顶端后回绕，所以十分钟前还在使用的 PID 可以发给一个新任务。

对重构工具的后果：

- 你在 `t` 时刻观察到的 PID，到 `t + 1s` 可能属于*另一个*任务——原来持有它的任务退出了，分配器把这个号回收了。
- 在容器化部署里，同一个任务在每个命名空间层级的 PID 都不同；挂在宿主命名空间的探测器和容器内的探测器看到的是同一个任务的不同 id。两者都没错，但单靠 pid 无法把它们关联起来。

把关联对象以 **(pid, 任务启动时间)** 为键，或在任务首次出现时分配一个工具生成的 id（从你看到的第一条事件捕获启动时间，或从 `/proc` 读取）。这样 PID 复用与命名空间差异就只是展示细节，而不是正确性 bug。（这与多路复用网络流量的连接归属问题是不同的边界——参见 [2026-08-31 回答](/zh/ebpf-qa/2026-08-31-tls-http2-sse-connection-correlation/)——那里即便 pid 稳定，也无法把流量归属到共享 socket 上的某条流；而这里的问题是身份本身不稳定。）

## 如何验证

1. **ring buffer 丢失演示。** 把一个小程序挂到高频 tracepoint 上；每次命中时累加一个 map 计数器，并尝试向一个故意很小的 ring buffer（4 KiB 就够）发记录。慢慢消费。预期：生产端计数大于已消费记录数；缓冲写满时 `bpf_ringbuf_reserve()` 返回 `NULL`、`bpf_ringbuf_output()` 返回 0——这就是文档记载的基线行为。如果目标内核有覆写模式，记录会被*替换*：流看起来仍然完整——覆写标志改变的是失效模式，没有治愈"无法察觉"这件事。
2. **perf buffer 丢失演示。** 同样的形态，改用 `bpf_perf_event_output()` 加 libbpf 的 `perf_buffer`（带 `lost_cb`）。突发流量下 lost 回调会带着数量触发；当消费者明显滞后而回调没触发时，"when possible" 的条款已经生效——把该时间窗视为丢失。
3. **身份演示。** 把探测器跑在 PID 命名空间里（`unshare --pid --fork`，需要查看 `/proc` 时再挂一个新挂载），看同一组任务的 PID 从 1 重新开始；然后在安静的机器上制造一波短命进程，观察同一个捕获窗口内被回收的 PID 带着不同的 comm 和启动时间出现。
4. **"哪些事件是必需的"：** 从 fork/exec 家族（`clone`、`fork`、`exit`、`execve`）、`openat`、以及网络动作的 `socket`/`connect` 起步——但把这份清单当作设计输入，而不是内核保证。工具只能关联目标内核 hook 暴露出来的东西，所以在敲定事件 schema 前先探测目标：可用的 tracepoint、CO-RE 重定位是否成功、kprobe 能否挂载，这些决定了你的事件表能装下什么。

## 回答的适用边界

- 这里覆盖的是*输出*路径——捕获的事件如何到达用户态。它不覆盖挂载可行性（内核版本、配置、架构、你要的 hook 是否存在且能重定位），后者决定哪些事件根本存在。
- 引用的丢失语义针对标准的 `BPF_MAP_TYPE_RINGBUF` 与 perf event 数组行为。用户态 ring buffer 变体和提议中的覆写模式会改变溢出行为，引用任何保证前先确认目标内核。
- 重构出的时间线是*可观测*行为的一个下界。要做取证级结论，请用独立证据（审计日志、抓包、进程核算）交叉核对疑似丢失窗口——工具应当明确说出来，而不是把缺口抹平。
- PID 复用边界假设标准分配器；`pid_max` 设得很小时回收更快，这使得以启动时间为键更加重要，而不是更不重要。

## 参考

- Linux 内核文档：[BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)——共享 MPSC 设计；"reservation 失败，不阻塞"；`BPF_RB_*` 仅作调试快照
- [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)——mmap 环形布局；`PERF_RECORD_LOST`；被丢弃样本"被视为丢失……尽可能时"生成记录；`PERF_FORMAT_LOST`
- 内核树中的 libbpf API（`tools/lib/bpf/libbpf.h`）——带 `lost_cb`（"在发生记录丢失时被调用"）的 `perf_buffer__new()`，以及没有任何丢失回调的 `ring_buffer__new()`——因为根本没有可投递的丢失记录
- [pid_namespaces(7)](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)——新命名空间中 PID 从 1 开始；唯一性是命名空间内的性质
- 内核 PID 分配器 `kernel/pid.c`——到达命名空间区间顶端后回绕的循环分配
- LWN：[Make BPF ring buffer over writable](https://lwn.net/Articles/904407/) 与 [Add overwrite mode for bpf ring buffer](https://lwn.net/Articles/1032293/)——基线行为（"ring buffer 满时……在 eBPF 代码中调用 `bpf_ringbuf_reserve()` 返回 NULL"）与提议的覆写语义

## 当日社区讨论

本次覆盖情况：两个归档授权（opt-in）的 Slack workspace 中有一个可读——严格 24 小时窗口安静，7 天回退窗口里只有一条消息，即 [2026-09-08 回答](/zh/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/)已回答过的那个研究定范围问题。另一个授权 workspace 的白名单频道在其归档中不存在，记录为不可访问而非安静。两个白名单 Discord workspace 只能经可见浏览器访问，本次没有可用的可见浏览器会话，作为覆盖缺口如实上报而非视为沉默。公开 bpf@vger.kernel.org 归档与公开技术论坛经其常规公开页面审阅。

### 成为今日回答的那个问题

本周公开论坛上最强、且尚未得到实质回答的实践者问题，来自一位用 eBPF/CO-RE 构建进程行为重构工具的工程师：把原始 syscall 事件（`execve`、`openat`、`socket`/`connect`、`clone`）关联成行为时间线，带版本化的捕获/回放格式；他问的包括：架构上应计入哪些 eBPF/内核限制、哪些事件是必需的、调查时哪些关联才有用。该线程还没有实质性的技术回答（仅一条玩笑回复），因此上面的答案在此发布：投递有序但不完整，pid 不是稳定的键，这两个性质都必须由工具自己制造。

### 内核邮件列表：策略对象、验证器扩展性、内存回收

今日公开 bpf 归档以补丁级工作为主，有三条实质性线程：

- **把安全策略变成 BPF 对象。** 一个 15 补丁的 bpf-next 系列（v3，约 30 条消息）通过 LSM "policy object" 和一族 `bpf_lsm_policy_*` kfunc，让 Landlock ruleset 可以从 BPF 施加。内核正在把 confinement 策略变成已加载程序可以管理的对象，这抬高了 [2026-09-08 回答](/zh/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/)中讨论的可见性边界 stakes。
- **验证器扩展性。** 一个 bpf 系列（6 补丁，11+ 条消息）修复 `bpf_compute_scc` 的二次复杂度后继重扫，限制每程序的间接跳转边数量，并为子程序缓存跳转表。验证器运行时间是运维愿意加载多大的实际天花板，跑大型策略程序的人值得跟踪。
- **从 BPF 做内存管理。** 一个 bpf-next 系列（v9）加入 `bpf_proactive_reclaim` kfunc，支持 BPF 驱动的主动 memcg 回收；另有 2 补丁系列保留跨子程序返回的逃逸 dynptr slice 谱系。BTF 内联函数位置信息系列（v2，扩展 libbpf 的 `LOC` 种类）延续着同样的内核 ABI 演进。

队列中还有：hash map 批量删除路径的软锁死（soft lockup）修复、sockmap 自重定向 `copied_seq` 重复计数的修复、以及 `bpf_fib_lookup()` 邻接表读取修复。总体看：能力面持续向安全策略与内存管理扩张，同时验证器与 map 机制在做后台扩展性工作。

### 技术论坛

两部分的公开重构系列继续更新，第二部分详述了某大型运营方用 eBPF 填补的十一个 Linux 生态缺口；截至写作时尚无讨论。本周早些时候的论文定范围线程（即 2026-09-08 回答的来源）仍是讨论最多的安全线程；上面的行为重构问题是窗口内唯一一个悬而未决的架构问题。
