# 为什么 iocost 的 cgroup 代价模型文件只能在内建模型的 auto 与 user 系数集之间切换，而 BPF struct ops 代价模型接管设备之后，这个文件为什么不再选择模型？

因为 cgroup 文件是内建模型的接口，而 BPF 程序是带外挂载的。blk-iocost 的 cgroup 代价模型文件的写路径解析 `ctrl=` 令牌（`auto` 或 `user`）、`model=` 令牌（恒为 `linear`）和七个线性系数令牌，并翻转 `user_cost_model` 位；它没有任何可运行的程序。公开的补丁系列把一个 BPF struct ops 实现绑定到某个设备的 iocost 实例上，经由 struct ops link 完成；link 附着期间，设备用这个程序给 IO 定价，cgroup 文件便不再选择模型。它仍接受 `ctrl=auto` 与 `ctrl=user`，但内建系数只在 link 摘下之后才重新生效。

## 机制

1. 内建侧（主线）。blk-iocost 的 cgroup 文件是 `cost.model`，注册为 `CFTYPE_ONLY_ON_ROOT`。其写处理 `ioc_cost_model_write()` 只解析 `ctrl=` 令牌（`auto` 或 `user`）、`model=` 令牌（只接受 `linear`）以及七个线性系数令牌（`rbps`、`rseqiops`、`rrandiops`、`wbps`、`wseqiops`、`wrandiops`、`flushiops`）。写入任何系数或 `ctrl=user` 会置上 `user_cost_model` 位，`ioc_refresh_params` 套用用户系数集；`ctrl=auto` 清除该位，由设备类推导出的自动 profile 再次生效。显示路径打印 `ctrl=user` 或 `ctrl=auto`，连同 `model=linear` 与当前系数。这里没有任何能指名程序的取值。
2. 接管（公开补丁）。该系列加了一个 struct ops 代价模型接口：BPF 程序实现模型 ops，挂载路径在一个 BPF 代价模型配置项的门槛下把它绑定到某个设备的 iocost 实例上。attach 与 detach 路径在队列锁下通过把程序 ops 与 link 同 ioc 已挂模型（`ioc_attached_model_locked()`）比对来确认挂载关系。模型对已在线的 cgroup 投递 `iocg_init()` 与 `iocg_free()`；两个回调都在 ioc 锁持有下运行且必须不可睡眠。设备移除会弹出已挂模型，而 struct ops link 保持有效直到用户态销毁它。同一系列还加了 `iocost_ioc_tick` tracepoint，报告每个周期结束时的设备摘要。
3. 附着期间 cgroup 文件的语义。该系列补充的 cgroup-v2 文档措辞说明：BPF 模型附着期间，文件中的模型取值为 `bpf`；内建系数令牌仍可读写，但只在切回 `linear` 之后才生效，且自动 profile 步进在 BPF 模型使用期间不会切换 profile——与用户设置参数一样，设备类选定的 profile 仍然适用。只有摘下 link（或移除设备）才恢复内建定价。

## 验证与排障路径

1. **读文件。** 读该设备 cgroup 的 iocost 代价模型文件。BPF 模型附着时，模型取值显示 `bpf`，而内建系数行仍列着线性系数；这些数值在 link 摘下之前是惰性的。
2. **带外摘除。** 在用户态销毁 struct ops link。设备重新用内建模型定价，cgroup 文件的 `ctrl` 与系数写入重新生效。设备移除则弹出已挂模型，而 link 保持有效直到用户态销毁。
3. **观察每周期定价。** 启用同系列新增的 `iocost_ioc_tick` tracepoint。它报告每个周期的最终摘要——busy level、cur period、nr active 与 usage 时长——因此 BPF 模型的效果按周期可见。注意对 `io.cost.qos` 的周期中途写入只改变周期时间（period time）与 vrate，在周期已经跑完之后才反映，所以摘要显示的是该周期运行时持有的值。

## 局限

- 主线没有 attach 路径：master 的 `block/blk-iocost.c` 只知道内建线性模型与 `user_cost_model` 位；struct ops 绑定来自公开补丁系列，且受 BPF 代价模型配置项门槛约束。
- 绑定是每设备粒度：每个设备的 iocost 实例挂一个模型，不是全局覆盖。
- BPF 模型附着期间的系数写入会被接受但不生效；只有切回内建模型之后才生效。
- 摘除不对称：设备移除弹出已挂模型，但 struct ops link 保持有效直到用户态销毁它。

## References

- [kernel `block/blk-iocost.c`](https://github.com/torvalds/linux/blob/master/block/blk-iocost.c) — 注册为 `CFTYPE_ONLY_ON_ROOT` 的 `cost.model` cgroup 文件、只接受 `ctrl` auto/user 与 `model=linear` 的写处理，以及 `user_cost_model` 位。
- [kernel Documentation/block/cgroup-v2.rst](https://github.com/torvalds/linux/blob/master/Documentation/block/cgroup-v2.rst) — iocost 小节与文档补丁所扩展的自动 profile 步进。
- [Public bpf mailing-list patch series: blk-iocost BPF struct ops cost model support](https://lore.kernel.org/bpf/895cb52c-1542-4ff7-abaf-844a73126fa8@linux.dev/) — attach/detach 路径、`iocg_init()` / `iocg_free()` 回调、`iocost_ioc_tick` tracepoint 与 cgroup-v2 文档措辞。

## 当日社区讨论

一封公开的 bpf 列表补丁系列为 blk-iocost 加入 BPF struct ops 代价模型支持：绑定到某个设备 iocost 实例的 BPF 程序接管该设备的 IO 定价，cgroup 代价模型文件随之停止选择模型。该系列为五部分：共享定义迁移（随后折回主文件）、struct ops attach/detach 核心、selftest、每周期 `iocost_ioc_tick` tracepoint，以及 cgroup-v2 文档措辞。评审线程集中在生命周期边界：挂载确认必须同时比对程序 ops 与 link，iocg init/free 回调在 ioc 锁持有下运行且不可睡眠，设备移除弹出已挂模型而 link 保持有效，且文档必须说明 BPF 模型附着期间内建系数仍可写、但只在切回之后才生效。

本次运行的存档来源在 7 天窗口内产出 0 条消息（两个 opt-in 存档来源在快照时为空），因此上面的问题取自被允许的公开 bpf 邮件列表，经其公开的 atom feed 审阅，补丁线程作为一手参考。eunomia-bpf 与 sched-ext 的 Discord 频道以及 r/eBPF 在本次运行未覆盖（没有可用的可见浏览器会话），故标记为未覆盖而非安静。
